from __future__ import annotations

import numpy as np

from tb_baselines.teacher import (
    DELTAS,
    HOLD,
    CompoundTeacher,
    Teacher,
    arbitrate_orders,
    known,
)


def make_scenario_board(
    rows: int = 24,
    cols: int = 24,
    ants: tuple[tuple[int, int], ...] = ((12, 12),),
    hills: tuple[tuple[int, int, int], ...] = (),
    food: tuple[tuple[int, int], ...] = (),
    water_coords: tuple[tuple[int, int], ...] = (),
    visible_coords: tuple[tuple[int, int], ...] | None = None,
    ids: list[int] | None = None,
) -> dict:
    water_mask = np.zeros((rows, cols), dtype=bool)
    for r, c in water_coords:
        if 0 <= r < rows and 0 <= c < cols:
            water_mask[r, c] = True

    seen_mask = np.ones((rows, cols), dtype=bool)
    if visible_coords is not None:
        seen_mask.fill(False)
        for r, c in visible_coords:
            if 0 <= r < rows and 0 <= c < cols:
                seen_mask[r, c] = True

    def to_rle(mask: np.ndarray) -> dict:
        flat = mask.flatten()
        diffs = np.diff(flat.astype(np.int8))
        change_indices = np.where(diffs != 0)[0] + 1
        starts = np.insert(change_indices, 0, 0)
        ends = np.append(change_indices, len(flat))
        runs = []
        for s, e in zip(starts, ends):
            runs.extend([int(flat[s]), int(e - s)])
        return {"rle": runs}

    if ids is None:
        ids = list(range(len(ants)))

    return {
        "size": [rows, cols],
        "mine": [list(a) for a in ants],
        "ids": list(ids),
        "foes": [],
        "food": [list(f) for f in food],
        "hills": [list(h) for h in hills],
        "water": to_rle(water_mask),
        "vis": to_rle(seen_mask),
    }


def compute_destinations(
    mine: list[list[int]] | list[tuple[int, int]] | np.ndarray,
    orders: np.ndarray,
    rows: int,
    cols: int,
) -> list[tuple[int, int]]:
    destinations: list[tuple[int, int]] = []
    for (r, c), act in zip(mine, orders):
        if act == HOLD:
            destinations.append((int(r), int(c)))
        else:
            dr, dc = DELTAS[int(act)]
            destinations.append(((int(r) + dr) % rows, (int(c) + dc) % cols))
    return destinations


def test_stress_ouroboros_directed_cycles_various_sizes():
    # 验证闭合环形队列在不同尺寸下可在单轮内无碰撞平移
    cycle_sizes = [2, 3, 4, 8, 16, 32, 64, 128, 256]
    for k in cycle_sizes:
        rows = k
        cols = 10
        mine = [(i, 5) for i in range(k)]
        # 每只蚂蚁声明向南移动（索引 2: dr=1, dc=0），第 k-1 只蚂蚁环绕至 (0, 5)
        candidates = [[2, HOLD] for _ in range(k)]
        potentials = [1 for _ in range(k)]
        stalls = {i: 0 for i in range(k)}

        orders, _, iteration_count = arbitrate_orders(
            mine_coords=mine,
            candidate_sequences=candidates,
            potentials=potentials,
            stall_counters=stalls,
            fair_mode=True,
            rows=rows,
            cols=cols,
            return_iteration_count=True,
        )

        assert iteration_count == 1
        assert len(orders) == k
        assert all(act == 2 for act in orders)
        dests = compute_destinations(mine, orders, rows, cols)
        assert len(set(dests)) == k
        expected_dests = [((i + 1) % rows, 5) for i in range(k)]
        assert dests == expected_dests


def test_stress_broken_cycle_cascade_wave():
    # 验证闭合环在单一节点驻留时产生反向排斥波，且收敛轮次严格等于 K+1 <= 4K
    cycle_sizes = [5, 15, 30, 60, 100]
    for k in cycle_sizes:
        rows = k
        cols = 10
        mine = [(i, 5) for i in range(k)]
        # 蚂蚁 0 意图原地驻留 HOLD，其余蚂蚁意图向南移入下一只蚂蚁的原位
        candidates: list[list[int]] = []
        for i in range(k):
            if i == 0:
                candidates.append([HOLD])
            else:
                candidates.append([2, HOLD])

        potentials = [10 for _ in range(k)]
        stalls = {i: 0 for i in range(k)}

        orders, _, iteration_count = arbitrate_orders(
            mine_coords=mine,
            candidate_sequences=candidates,
            potentials=potentials,
            stall_counters=stalls,
            fair_mode=True,
            rows=rows,
            cols=cols,
            return_iteration_count=True,
        )

        assert iteration_count == k
        assert iteration_count <= 4 * k
        assert all(act == HOLD for act in orders)
        dests = compute_destinations(mine, orders, rows, cols)
        assert len(set(dests)) == k
        assert dests == mine


def test_stress_interlocking_figure_eight_mesh():
    # 验证在双环相交的 8 字形网格中，共享交叉点的仲裁与波前收敛
    # 环 A：(0, 5) -> (1, 5) -> (2, 5) -> (1, 4) -> (0, 5)
    # 环 B：(2, 5) -> (3, 5) -> (4, 5) -> (3, 6) -> (2, 5)
    # 交叉点位于 (2, 5)
    mine = [
        (0, 5), (1, 5), (2, 5), (1, 4),
        (3, 5), (4, 5), (3, 6),
    ]
    # 蚂蚁 0..3 构成左环，蚂蚁 2 共享，蚂蚁 4..6 构成右环
    # 设定蚂蚁 1 意图移向 (2, 5)，蚂蚁 6 意图移向 (2, 5)，发生交叉点竞争
    candidates = [
        [2, HOLD],  # 0 -> (1, 5)
        [2, HOLD],  # 1 -> (2, 5)
        [3, HOLD],  # 2 -> (2, 4)
        [0, HOLD],  # 3 -> (0, 4)
        [2, HOLD],  # 4 -> (4, 5)
        [1, HOLD],  # 5 -> (4, 6)
        [3, HOLD],  # 6 -> (3, 5)
    ]
    potentials = [5, 2, 8, 5, 6, 7, 3]
    stalls = {i: 0 for i in range(len(mine))}

    orders, _, iteration_count = arbitrate_orders(
        mine_coords=mine,
        candidate_sequences=candidates,
        potentials=potentials,
        stall_counters=stalls,
        fair_mode=True,
        rows=10,
        cols=10,
        return_iteration_count=True,
    )

    assert iteration_count <= 4 * len(mine)
    dests = compute_destinations(mine, orders, 10, 10)
    assert len(set(dests)) == len(mine)


def test_stress_adversarial_worst_case_cascade_iteration_bound():
    # 构造级联冲突序列，检验多重候选回退路径下的收敛轮次严格受限于 4N
    num_ants = 30
    rows = 40
    cols = 40
    mine = [(i, 10) for i in range(num_ants)]
    # 每只蚂蚁具备 4 个正交移动候选与保底 HOLD
    candidates = []
    for i in range(num_ants):
        # 尝试北、东、南、西四个方向，最后回退至 HOLD
        candidates.append([0, 1, 2, 3, HOLD])

    potentials = [i for i in range(num_ants)]
    stalls = {i: 0 for i in range(num_ants)}

    orders, _, iteration_count = arbitrate_orders(
        mine_coords=mine,
        candidate_sequences=candidates,
        potentials=potentials,
        stall_counters=stalls,
        fair_mode=True,
        rows=rows,
        cols=cols,
        return_iteration_count=True,
    )

    assert iteration_count <= 4 * num_ants
    dests = compute_destinations(mine, orders, rows, cols)
    assert len(set(dests)) == num_ants


def test_stress_dense_grid_fuzzing_500_ants():
    # 在 24x24 棋盘布设 500 只蚂蚁（86.8% 极高密度），随机生成候选序列进行压力检验
    rng = np.random.RandomState(42)
    rows, cols = 24, 24
    num_ants = 500

    all_cells = [(r, c) for r in range(rows) for c in range(cols)]
    chosen_indices = rng.choice(len(all_cells), size=num_ants, replace=False)
    mine = [all_cells[i] for i in chosen_indices]

    for trial in range(10):
        candidates = []
        for _ in range(num_ants):
            seq_len = int(rng.randint(1, 5))
            acts = list(rng.choice([0, 1, 2, 3], size=seq_len, replace=False))
            acts.append(HOLD)
            candidates.append(acts)

        potentials = [int(rng.randint(0, 100)) for _ in range(num_ants)]
        stalls = {i: int(rng.randint(0, 10)) for i in range(num_ants)}

        orders, updated_stalls, iteration_count = arbitrate_orders(
            mine_coords=mine,
            candidate_sequences=candidates,
            potentials=potentials,
            stall_counters=stalls,
            fair_mode=True,
            rows=rows,
            cols=cols,
            return_iteration_count=True,
        )

        assert iteration_count <= 4 * num_ants
        dests = compute_destinations(mine, orders, rows, cols)
        assert len(set(dests)) == num_ants


def test_stress_100_percent_packed_grid():
    # 在 8x8 满载棋盘（64格容纳64只蚂蚁，0空格）验证无碰撞性
    rows, cols = 8, 8
    num_ants = rows * cols
    mine = [(r, c) for r in range(rows) for c in range(cols)]

    rng = np.random.RandomState(2026)
    for trial in range(15):
        candidates = []
        for _ in range(num_ants):
            seq_len = int(rng.randint(1, 4))
            acts = list(rng.choice([0, 1, 2, 3], size=seq_len, replace=False))
            acts.append(HOLD)
            candidates.append(acts)

        potentials = [int(rng.randint(0, 50)) for _ in range(num_ants)]
        stalls = {i: int(rng.randint(0, 5)) for i in range(num_ants)}

        orders, _, iteration_count = arbitrate_orders(
            mine_coords=mine,
            candidate_sequences=candidates,
            potentials=potentials,
            stall_counters=stalls,
            fair_mode=True,
            rows=rows,
            cols=cols,
            return_iteration_count=True,
        )

        assert iteration_count <= 4 * num_ants
        dests = compute_destinations(mine, orders, rows, cols)
        assert len(set(dests)) == num_ants


def test_stress_massive_single_cell_convergence():
    # 100 只蚂蚁同时将首选目标申报为同一中心格子 (10, 10)
    rows, cols = 25, 25
    target_pos = (10, 10)
    rng = np.random.RandomState(777)

    all_cells = [(r, c) for r in range(rows) for c in range(cols) if (r, c) != target_pos]
    chosen = rng.choice(len(all_cells), size=100, replace=False)
    mine = [all_cells[i] for i in chosen]

    candidates = []
    for (r, c) in mine:
        # 首选动作对准 target_pos 的方向，若不正交则选取靠近的方向
        dr = (target_pos[0] - r) % rows
        dc = (target_pos[1] - c) % cols
        first_act = 2 if dr <= rows // 2 else 0
        second_act = 1 if dc <= cols // 2 else 3
        candidates.append([first_act, second_act, HOLD])

    potentials = [int(rng.randint(1, 100)) for _ in range(100)]
    stalls = {i: int(rng.randint(0, 5)) for i in range(100)}

    orders, _, iteration_count = arbitrate_orders(
        mine_coords=mine,
        candidate_sequences=candidates,
        potentials=potentials,
        stall_counters=stalls,
        fair_mode=True,
        rows=rows,
        cols=cols,
        return_iteration_count=True,
    )

    assert iteration_count <= 4 * 100
    dests = compute_destinations(mine, orders, rows, cols)
    assert len(set(dests)) == 100


def test_stress_sovereign_hold_immune_to_all_priority_extremes():
    # 驻留主权测试：驻留单位具备最低优先级参数，四周入侵单位具备极高优先级参数
    mine = [
        (10, 10),  # 驻留防守蚂蚁（索引 0）
        (9, 10),   # 北方入侵蚂蚁（索引 1，向南进入 (10, 10)）
        (11, 10),  # 南方入侵蚂蚁（索引 2，向北进入 (10, 10)）
        (10, 9),   # 西方入侵蚂蚁（索引 3，向东进入 (10, 10)）
        (10, 11),  # 东方入侵蚂蚁（索引 4，向西进入 (10, 10)）
    ]
    candidates = [
        [HOLD],
        [2, HOLD],
        [0, HOLD],
        [1, HOLD],
        [3, HOLD],
    ]
    # 驻留蚂蚁设定最差势能、零受阻计数、最差编号
    potentials = [999999, 0, 0, 0, 0]
    stalls = {0: 0, 1: 999, 2: 999, 3: 999, 4: 999}
    squad_tiers = [99, 0, 0, 0, 0]
    ant_ids = [9999, 1, 2, 3, 4]

    # 测试公平模式
    orders_fair, stalls_fair = arbitrate_orders(
        mine_coords=mine,
        candidate_sequences=candidates,
        potentials=potentials,
        stall_counters=stalls,
        fair_mode=True,
        rows=20,
        cols=20,
        ids=ant_ids,
        squad_tiers=squad_tiers,
    )
    assert orders_fair[0] == HOLD
    dests_fair = compute_destinations(mine, orders_fair, 20, 20)
    assert dests_fair[0] == (10, 10)
    assert all(dests_fair[i] != (10, 10) for i in range(1, 5))
    assert len(set(dests_fair)) == 5

    # 测试非公平模式
    orders_unfair, _ = arbitrate_orders(
        mine_coords=mine,
        candidate_sequences=candidates,
        potentials=potentials,
        stall_counters=stalls,
        fair_mode=False,
        rows=20,
        cols=20,
        ids=ant_ids,
        squad_tiers=squad_tiers,
    )
    assert orders_unfair[0] == HOLD
    dests_unfair = compute_destinations(mine, orders_unfair, 20, 20)
    assert dests_unfair[0] == (10, 10)
    assert all(dests_unfair[i] != (10, 10) for i in range(1, 5))
    assert len(set(dests_unfair)) == 5


def test_stress_acquired_sovereignty_on_fallback():
    # 获得性主权验证：意图移动的蚂蚁因外部阻挡回退至 HOLD 时，立即享有初始方格的排他保留权
    mine = [
        (5, 5),  # 蚂蚁 0：首选移向 (5, 6)，次选 HOLD
        (5, 6),  # 蚂蚁 1：原地驻留防守 (5, 6)
        (5, 4),  # 蚂蚁 2：首选移向 (5, 5)，次选 HOLD
    ]
    candidates = [
        [1, HOLD],  # 0 尝试移向 (5, 6)
        [HOLD],     # 1 驻留在 (5, 6)
        [1, HOLD],  # 2 尝试移向 (5, 5)
    ]
    potentials = [10, 10, 1]  # 蚂蚁 2 势能优于蚂蚁 0
    stalls = {0: 0, 1: 0, 2: 5}  # 蚂蚁 2 受阻计数更高

    orders, updated_stalls, iters = arbitrate_orders(
        mine_coords=mine,
        candidate_sequences=candidates,
        potentials=potentials,
        stall_counters=stalls,
        fair_mode=True,
        rows=20,
        cols=20,
        return_iteration_count=True,
    )

    # 蚂蚁 1 驻留 (5, 6)，蚂蚁 0 无法进入 (5, 6) 从而回退至 HOLD 于 (5, 5)；
    # 蚂蚁 0 回退至 HOLD 后排他拥有 (5, 5)，蚂蚁 2 必须被排斥并回退至 HOLD 于 (5, 4)
    assert orders[1] == HOLD
    assert orders[0] == HOLD
    assert orders[2] == HOLD

    dests = compute_destinations(mine, orders, 20, 20)
    assert dests == [(5, 5), (5, 6), (5, 4)]
    assert len(set(dests)) == 3
    assert iters <= 4 * 3


def test_stress_multi_ant_concentric_perimeter_siege():
    # 验证 2x2 实心驻留蚂蚁方阵抵抗外围 8 只入侵蚂蚁的全向冲击
    core_mine = [(10, 10), (10, 11), (11, 10), (11, 11)]
    perimeter_mine = [
        (9, 10), (9, 11), (12, 10), (12, 11),
        (10, 9), (11, 9), (10, 12), (11, 12),
    ]
    mine = core_mine + perimeter_mine
    candidates = []
    # 核心 4 只蚂蚁保持 HOLD
    for _ in range(4):
        candidates.append([HOLD])

    # 外围 8 只蚂蚁全部意图突入相邻的核心格子
    candidates.append([2, HOLD])  # (9, 10) -> (10, 10)
    candidates.append([2, HOLD])  # (9, 11) -> (10, 11)
    candidates.append([0, HOLD])  # (12, 10) -> (11, 10)
    candidates.append([0, HOLD])  # (12, 11) -> (11, 11)
    candidates.append([1, HOLD])  # (10, 9) -> (10, 10)
    candidates.append([1, HOLD])  # (11, 9) -> (11, 10)
    candidates.append([3, HOLD])  # (10, 12) -> (10, 11)
    candidates.append([3, HOLD])  # (11, 12) -> (11, 11)

    potentials = [99] * 4 + [1] * 8
    stalls = {i: 0 for i in range(12)}

    orders, _, iteration_count = arbitrate_orders(
        mine_coords=mine,
        candidate_sequences=candidates,
        potentials=potentials,
        stall_counters=stalls,
        fair_mode=True,
        rows=20,
        cols=20,
        return_iteration_count=True,
    )

    assert iteration_count <= 4 * 12
    # 核心 4 只蚂蚁全量驻留
    for i in range(4):
        assert orders[i] == HOLD
    dests = compute_destinations(mine, orders, 20, 20)
    assert len(set(dests)) == 12
    # 验证核心 4 格未被任何入侵蚂蚁侵占
    for core_pos in core_mine:
        claimants = [idx for idx, d in enumerate(dests) if d == core_pos]
        assert len(claimants) == 1
        assert claimants[0] < 4


def test_stress_corridor_snake_linear_advance():
    # 检验不同长度的直线走廊蚂蚁队列能够在单一回合无延迟完成腾空推移
    lengths = [5, 10, 25, 50, 100]
    for length in lengths:
        rows = 20
        cols = length + 10
        # 蚂蚁排布于 (5, 0) 至 (5, length-1)，头部位于 (5, length-1)
        mine = [(5, c) for c in range(length)]
        # 全部蚂蚁声明向东移动（索引 1: dr=0, dc=1）
        candidates = [[1, HOLD] for _ in range(length)]
        potentials = [length - c for c in range(length)]
        stalls = {i: 0 for i in range(length)}

        orders, _, iteration_count = arbitrate_orders(
            mine_coords=mine,
            candidate_sequences=candidates,
            potentials=potentials,
            stall_counters=stalls,
            fair_mode=True,
            rows=rows,
            cols=cols,
            return_iteration_count=True,
        )

        assert iteration_count == 1
        assert all(act == 1 for act in orders)
        dests = compute_destinations(mine, orders, rows, cols)
        assert len(set(dests)) == length
        expected_dests = [(5, c + 1) for c in range(length)]
        assert dests == expected_dests


def test_stress_corridor_snake_head_blocked_cascade():
    # 检验走廊队列头部被阻挡时，全队平稳依次回退至 HOLD，无追尾碰撞
    length = 50
    rows = 20
    cols = length + 5
    mine = [(5, c) for c in range(length)]
    # 头部 (5, length-1) 前方为障碍，只能 HOLD；其余蚂蚁均希望向东进入前车格子
    candidates = []
    for c in range(length):
        if c == length - 1:
            candidates.append([HOLD])
        else:
            candidates.append([1, HOLD])

    potentials = [length - c for c in range(length)]
    stalls = {i: 0 for i in range(length)}

    orders, _, iteration_count = arbitrate_orders(
        mine_coords=mine,
        candidate_sequences=candidates,
        potentials=potentials,
        stall_counters=stalls,
        fair_mode=True,
        rows=rows,
        cols=cols,
        return_iteration_count=True,
    )

    assert iteration_count == length
    assert iteration_count <= 4 * length
    assert all(act == HOLD for act in orders)
    dests = compute_destinations(mine, orders, rows, cols)
    assert len(set(dests)) == length
    assert dests == mine


def test_stress_converging_snakes_alternating_bottleneck():
    # 两支队列汇入单通道瓶颈，验证受阻计数器消除饥饿，实现交替放行
    # 通道入口为 (5, 5)
    # 队列 A：(4, 5) 尝试向南进入 (5, 5)
    # 队列 B：(5, 4) 尝试向东进入 (5, 5)
    mine = [(4, 5), (5, 4)]
    candidates = [[2, HOLD], [1, HOLD]]
    potentials = [5, 5]
    stalls = {0: 0, 1: 0}

    winner_history = []
    for turn in range(10):
        orders, stalls = arbitrate_orders(
            mine_coords=mine,
            candidate_sequences=candidates,
            potentials=potentials,
            stall_counters=stalls,
            fair_mode=True,
            rows=20,
            cols=20,
        )
        if orders[0] == 2:
            winner_history.append(0)
        elif orders[1] == 1:
            winner_history.append(1)
        else:
            raise AssertionError("瓶颈格子无蚂蚁成功进入")

    # 验证双方交替放行，任何一方不会连续受阻超过 1 回合
    for turn in range(len(winner_history) - 1):
        assert winner_history[turn] != winner_history[turn + 1]


def test_stress_teacher_orders_dense_multi_turn_simulation():
    # 端到端 Teacher 仿真：在包含障碍物与食物的密集场景下连续推演 30 回合，校验零碰撞
    rows, cols = 20, 20
    water_coords = []
    for r in range(rows):
        if r % 4 == 0:
            for c in range(cols):
                if c != 5 and c != 15:
                    water_coords.append((r, c))

    rng = np.random.RandomState(888)
    all_land = [
        (r, c) for r in range(rows) for c in range(cols)
        if (r, c) not in set(water_coords)
    ]
    chosen = rng.choice(len(all_land), size=40, replace=False)
    ant_coords = [all_land[i] for i in chosen[:25]]
    food_coords = [all_land[i] for i in chosen[25:]]

    current_ants = list(ant_coords)
    teacher = Teacher()

    for step in range(30):
        obs = make_scenario_board(
            rows=rows,
            cols=cols,
            ants=tuple(current_ants),
            food=tuple(food_coords),
            water_coords=tuple(water_coords),
        )
        water, seen = known(obs)
        orders, telemetry = teacher.orders_with_telemetry(obs, water, seen)

        assert len(orders) == len(current_ants)
        dests = compute_destinations(current_ants, orders, rows, cols)
        # 100% 杜绝终点碰撞
        assert len(set(dests)) == len(current_ants)

        # 验证没有蚂蚁迈入水域或食物方格
        water_set = set(water_coords)
        food_set = set(food_coords)
        for d in dests:
            assert d not in water_set
            assert d not in food_set

        # 演进蚂蚁坐标至下一回合
        current_ants = list(dests)


def test_stress_compound_teacher_phase_b_dense_multi_turn():
    # 端到端 CompoundTeacher 仿真：在包含分队与掠夺梯度的密集场景下连续推演 30 回合，校验零碰撞
    rows, cols = 24, 24
    rng = np.random.RandomState(999)
    hills = ((12, 12, 1),)
    water_coords = [(r, 6) for r in range(rows) if r != 10 and r != 14]

    all_land = [
        (r, c) for r in range(rows) for c in range(cols)
        if (r, c) not in set(water_coords) and (r, c) != (12, 12)
    ]
    chosen = rng.choice(len(all_land), size=50, replace=False)
    ant_coords = [all_land[i] for i in chosen[:30]]
    food_coords = [all_land[i] for i in chosen[30:]]

    current_ants = list(ant_coords)
    ct = CompoundTeacher(min_raid_ants=6, raid_ratio=0.5)

    for step in range(30):
        obs = make_scenario_board(
            rows=rows,
            cols=cols,
            ants=tuple(current_ants),
            food=tuple(food_coords),
            hills=hills,
            water_coords=tuple(water_coords),
        )
        water, seen = known(obs)
        orders, telemetry = ct.orders_with_telemetry(obs, water, seen)

        assert len(orders) == len(current_ants)
        dests = compute_destinations(current_ants, orders, rows, cols)
        assert len(set(dests)) == len(current_ants)

        water_set = set(water_coords)
        food_set = set(food_coords)
        for d in dests:
            assert d not in water_set
            assert d not in food_set

        current_ants = list(dests)


def test_stress_arbitrate_orders_empty_and_single_ant_boundaries():
    # 边界测试：0 蚂蚁与单只蚂蚁各种序列的稳健性
    # 0 蚂蚁
    empty_orders, empty_stalls, iters = arbitrate_orders(
        mine_coords=[],
        candidate_sequences=[],
        potentials=[],
        return_iteration_count=True,
    )
    assert len(empty_orders) == 0
    assert iters == 0

    # 1 蚂蚁，仅 HOLD
    orders_1, stalls_1, iters_1 = arbitrate_orders(
        mine_coords=[(5, 5)],
        candidate_sequences=[[HOLD]],
        potentials=[10],
        return_iteration_count=True,
    )
    assert len(orders_1) == 1
    assert orders_1[0] == HOLD
    assert iters_1 == 1

    # 1 蚂蚁，有效移动
    orders_2, stalls_2, iters_2 = arbitrate_orders(
        mine_coords=[(5, 5)],
        candidate_sequences=[[1, HOLD]],
        potentials=[10],
        return_iteration_count=True,
    )
    assert len(orders_2) == 1
    assert orders_2[0] == 1
    assert iters_2 == 1
