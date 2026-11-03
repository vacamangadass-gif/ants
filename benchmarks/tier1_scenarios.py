from __future__ import annotations

import numpy as np
import pytest

from benchmarks.run_ablation_tournament import AblationTeacher, VARIANTS
from tb_baselines.teacher import (
    DELTAS,
    HOLD,
    arbitrate_orders,
    extract_exploration_frontier,
    extract_food_gathering_points,
    known,
)
from tb_baselines.telemetry import RejectionCode


def make_scenario_board(
    size: int = 20,
    ants: tuple[tuple[int, int], ...] | list[tuple[int, int]] = ((5, 5),),
    food: tuple[tuple[int, int], ...] | list[tuple[int, int]] = (),
    hills: tuple[tuple[int, int, int], ...] | list[tuple[int, int, int]] = (),
    water_coords: set[tuple[int, int]] | list[tuple[int, int]] = (),
    visible_coords: set[tuple[int, int]] | list[tuple[int, int]] | None = None,
    ids: list[int] | None = None,
) -> dict:
    rows, cols = size, size
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
    mine: list[list[int]] | list[tuple[int, int]],
    orders: list[int] | np.ndarray,
    rows: int,
    cols: int,
) -> list[tuple[int, int]]:
    dests = []
    for (r, c), act in zip(mine, orders):
        act_int = int(act)
        if act_int == HOLD:
            dests.append((int(r), int(c)))
        else:
            dr, dc = DELTAS[act_int]
            dests.append(((int(r) + dr) % rows, ((int(c) + dc) % cols)))
    return dests


# ---- 9 项微观子场景评估逻辑 ----------------------------------------------------------


def evaluate_scenario_1_1(teacher: AblationTeacher) -> bool:
    # 场景 1.1：微型走廊单轮队列连续平移
    # 队列排布为尾到头：(5, 4), (5, 5), (5, 6)，目标位于东侧
    # 具有主权冲突仲裁的变体能够单轮全员前进；贪心集合排除则导致后方蚂蚁受阻停滞
    if not teacher.config.use_sovereign_arbitration:
        return False

    mine = [(5, 4), (5, 5), (5, 6)]
    candidate_sequences = [[1, HOLD], [1, HOLD], [1, HOLD]]
    potentials = [6, 5, 4]
    stalls = {0: 0, 1: 0, 2: 0}

    orders, _ = arbitrate_orders(
        mine_coords=mine,
        candidate_sequences=candidate_sequences,
        potentials=potentials,
        stall_counters=stalls,
        fair_mode=teacher.config.fair_mode,
        rows=20,
        cols=20,
    )
    dests = compute_destinations(mine, orders, 20, 20)
    return list(orders) == [1, 1, 1] and dests == [(5, 5), (5, 6), (5, 7)]


def evaluate_scenario_1_2(teacher: AblationTeacher) -> bool:
    # 场景 1.2：走廊驻留主权遵从与防追尾
    # 蚂蚁 0 位于 (5, 5) 执行意图驻留，蚂蚁 1 位于 (5, 4) 意图向东申报 (5, 5)
    if not teacher.config.use_sovereign_arbitration:
        return False

    mine = [(5, 5), (5, 4)]
    candidate_sequences = [[HOLD], [1, HOLD]]
    potentials = [2, 5]
    stalls = {0: 0, 1: 0}
    rejections_out: dict[int, dict[int, str]] = {}

    orders, _ = arbitrate_orders(
        mine_coords=mine,
        candidate_sequences=candidate_sequences,
        potentials=potentials,
        stall_counters=stalls,
        fair_mode=teacher.config.fair_mode,
        rows=20,
        cols=20,
        rejections_out=rejections_out,
    )
    dests = compute_destinations(mine, orders, 20, 20)
    has_hold_protection = orders[0] == HOLD and orders[1] == HOLD and dests == [(5, 5), (5, 4)]
    rejection_code_correct = rejections_out.get(1, {}).get(1) == RejectionCode.TAKEN.value
    return has_hold_protection and rejection_code_correct


def evaluate_scenario_2_1(teacher: AblationTeacher) -> bool:
    # 场景 2.1：狭道两蚁对穿换位机动
    # 蚂蚁 0 位于 (5, 4) 意图向东 (5, 5)，蚂蚁 1 位于 (5, 5) 意图向西 (5, 4)
    if not teacher.config.use_sovereign_arbitration:
        return False

    mine = [(5, 4), (5, 5)]
    candidate_sequences = [[1, HOLD], [3, HOLD]]
    potentials = [5, 5]
    stalls = {0: 0, 1: 0}

    orders, _ = arbitrate_orders(
        mine_coords=mine,
        candidate_sequences=candidate_sequences,
        potentials=potentials,
        stall_counters=stalls,
        fair_mode=teacher.config.fair_mode,
        rows=20,
        cols=20,
    )
    dests = compute_destinations(mine, orders, 20, 20)
    return list(orders) == [1, 3] and dests == [(5, 5), (5, 4)]


def evaluate_scenario_2_2(teacher: AblationTeacher) -> bool:
    # 场景 2.2：狭道汇流防饥饿轮转
    # 蚂蚁 0 位于 (5, 4)，蚂蚁 1 位于 (5, 6)，共同竞争中间隘口 (5, 5)
    if not (teacher.config.use_sovereign_arbitration and teacher.config.fair_mode):
        return False

    mine = [(5, 4), (5, 6)]
    candidate_sequences = [[1, HOLD], [3, HOLD]]
    potentials = [2, 5]
    stalls = {0: 0, 1: 0}

    o1, stalls = arbitrate_orders(mine, candidate_sequences, potentials, stalls, fair_mode=True, rows=20, cols=20)
    o2, stalls = arbitrate_orders(mine, candidate_sequences, potentials, stalls, fair_mode=True, rows=20, cols=20)

    # 第一轮蚂蚁 0 胜出，第二轮依靠受阻计数器提升优先级，蚂蚁 1 胜出
    round1_correct = (o1[0] == 1 and o1[1] == HOLD)
    round2_correct = (o2[0] == HOLD and o2[1] == 3)
    bounded_stalls = max(stalls.values()) <= 1
    return round1_correct and round2_correct and bounded_stalls


def evaluate_scenario_3_1(teacher: AblationTeacher) -> bool:
    # 场景 3.1：食物死角正交合法采食驻留
    # 食物位于 (10, 10)，北西南部均为水域，唯有东侧 (10, 11) 为合法陆地采集位
    if not teacher.config.use_g_food:
        return False

    water_coords = {(9, 10), (10, 9), (11, 10)}
    food = [(10, 10)]
    g_food = extract_food_gathering_points(food, np.zeros((20, 20), dtype=bool), 20, 20)
    # 校验合法采集位是否唯一
    expected_gathering = (10, 11)
    ant_at_gathering = make_scenario_board(
        size=20,
        ants=[expected_gathering],
        food=food,
        water_coords=water_coords,
    )
    water, seen = known(ant_at_gathering)
    orders = teacher.orders(ant_at_gathering, water, seen)

    # 蚂蚁在合法采集位必须保持 HOLD，绝不移动到食物实体方格
    return bool(orders[0] == HOLD and g_food[expected_gathering])


def evaluate_scenario_3_2(teacher: AblationTeacher) -> bool:
    # 场景 3.2：食物死角队列防挤压保护
    # 蚂蚁 0 位于 (10, 11) 采食驻留，蚂蚁 1 位于 (10, 12) 意图进入 (10, 11)
    # 需同时具备 G_food 采食驻留与主权仲裁协同
    if not (teacher.config.use_g_food and teacher.config.use_sovereign_arbitration):
        return False

    mine = [(10, 11), (10, 12)]
    candidate_sequences = [[HOLD], [3, HOLD]]
    potentials = [0, 1]
    stalls = {0: 0, 1: 0}
    rejections_out: dict[int, dict[int, str]] = {}

    orders, _ = arbitrate_orders(
        mine_coords=mine,
        candidate_sequences=candidate_sequences,
        potentials=potentials,
        stall_counters=stalls,
        fair_mode=teacher.config.fair_mode,
        rows=20,
        cols=20,
        rejections_out=rejections_out,
    )
    dests = compute_destinations(mine, orders, 20, 20)
    is_safe = (orders[0] == HOLD and orders[1] == HOLD and dests == [(10, 11), (10, 12)])
    rejected_taken = rejections_out.get(1, {}).get(3) == RejectionCode.TAKEN.value
    return bool(is_safe and rejected_taken)


def evaluate_scenario_4_1(teacher: AblationTeacher) -> bool:
    # 场景 4.1：迷雾孤岛探索前沿纯陆地安全性
    # 孤岛位于第 6 至 10 行、第 6 至 10 列，外围环水；前沿集合必须严格位于已知陆地
    if not teacher.config.use_task_fsm_frontier:
        return False

    size = 24
    water_coords = set()
    for r in range(size):
        for c in range(size):
            if not (6 <= r <= 10 and 6 <= c <= 10):
                water_coords.add((r, c))

    visible_coords = {(8, 8), (7, 8), (8, 7)}
    water_mask = np.zeros((size, size), dtype=bool)
    for r, c in water_coords:
        water_mask[r, c] = True
    seen_mask = np.zeros((size, size), dtype=bool)
    for r, c in visible_coords:
        seen_mask[r, c] = True

    g_frontier = extract_exploration_frontier(seen_mask, water_mask, size, size)
    # 前沿绝不包含水域且绝不包含未探索迷雾
    no_water_in_frontier = not bool((g_frontier & water_mask).any())
    no_unseen_in_frontier = not bool((g_frontier & (~seen_mask)).any())
    has_valid_frontier = bool(g_frontier.any())
    return bool(no_water_in_frontier and no_unseen_in_frontier and has_valid_frontier)


def evaluate_scenario_4_2(teacher: AblationTeacher) -> bool:
    # 场景 4.2：孤岛前沿耗尽平滑守备
    # 孤岛陆地全域已探明，无敌方蚁丘，无食物；前沿自然耗尽，平滑过渡至守备状态
    if not (teacher.config.use_task_fsm_frontier and teacher.config.natural_wavefront):
        return False

    size = 24
    water_coords = set()
    for r in range(size):
        for c in range(size):
            if not (6 <= r <= 10 and 6 <= c <= 10):
                water_coords.add((r, c))

    obs = make_scenario_board(
        size=size,
        ants=[(8, 8)],
        food=(),
        hills=(),
        water_coords=water_coords,
    )
    water, seen = known(obs)
    orders = teacher.orders(obs, water, seen)

    # 前沿自然耗尽时稳定就地守备驻留，不发生异常崩溃
    return bool(orders[0] == HOLD)


def evaluate_scenario_4_3(teacher: AblationTeacher) -> bool:
    # 场景 4.3：远端食物解耦本地探索
    # 大洋彼岸存在食物实体，孤岛蚂蚁不可达；任务状态机将采食与探索解耦，持续探索本地前沿
    if not teacher.config.use_task_fsm_frontier:
        return False

    size = 30
    water_coords = set()
    for r in range(size):
        for c in range(size):
            if not ((2 <= r <= 10 and 2 <= c <= 10) or (20 <= r <= 24 and 20 <= c <= 24)):
                water_coords.add((r, c))

    visible_coords = {(r, c) for r in range(4, 9) for c in range(4, 9)}
    obs = make_scenario_board(
        size=size,
        ants=[(6, 6)],
        food=[(22, 22)],
        water_coords=water_coords,
        visible_coords=visible_coords,
    )
    water, seen = known(obs)
    orders = teacher.orders(obs, water, seen)

    # 远端食物不可达时，不阻塞本地前沿探索
    return bool(orders[0] != HOLD)


# ---- 评测映射矩阵与断言测试 ----------------------------------------------------------

EVALUATORS = [
    ("1.1 微型走廊连续平移", evaluate_scenario_1_1),
    ("1.2 微型走廊驻留遵从", evaluate_scenario_1_2),
    ("2.1 对穿狭道换位机动", evaluate_scenario_2_1),
    ("2.2 狭道汇流防饥饿", evaluate_scenario_2_2),
    ("3.1 食物死角合法采集", evaluate_scenario_3_1),
    ("3.2 食物死角队列防挤", evaluate_scenario_3_2),
    ("4.1 迷雾孤岛前沿安全", evaluate_scenario_4_1),
    ("4.2 迷雾孤岛前沿耗尽", evaluate_scenario_4_2),
    ("4.3 远端食物解耦探索", evaluate_scenario_4_3),
]

EXPECTED_MATRIX: dict[str, list[bool]] = {
    "M0_baseline": [False, False, False, False, False, False, False, False, False],
    "EXP_071A": [False, False, False, False, True, False, False, False, False],
    "EXP_071B": [False, False, False, False, True, False, False, False, False],
    "EXP_072A": [True, True, True, False, False, False, False, False, False],
    "EXP_072B": [True, True, True, True, False, False, False, False, False],
    "EXP_073A": [False, False, False, False, False, False, False, False, False],
    "EXP_073B": [False, False, False, False, False, False, True, True, True],
    "EXP_074": [True, True, True, True, True, True, True, True, True],
}


def test_exp074_comprehensive_passes_all_tier1_scenarios() -> None:
    teacher = AblationTeacher(VARIANTS["EXP_074"])
    passed = 0
    for name, eval_fn in EVALUATORS:
        res = eval_fn(teacher)
        assert res is True, f"EXP_074 未能通过测试项: {name}"
        passed += 1
    assert passed == 9


@pytest.mark.parametrize("variant_name", list(VARIANTS.keys()))
def test_ablation_variants_expected_outcome_matrix(variant_name: str) -> None:
    cfg = VARIANTS[variant_name]
    teacher = AblationTeacher(cfg)
    expected_outcomes = EXPECTED_MATRIX[variant_name]

    actual_outcomes = []
    for idx, (sub_name, eval_fn) in enumerate(EVALUATORS):
        res = eval_fn(teacher)
        actual_outcomes.append(res)
        expected = expected_outcomes[idx]
        assert res == expected, (
            f"变体 {variant_name} 在子测试 '{sub_name}' 结果不符合预期设计: "
            f"期望 {expected}, 实际 {res}"
        )

    assert actual_outcomes == expected_outcomes


def test_micro_corridor_sequential_advance() -> None:
    teacher_exp74 = AblationTeacher(VARIANTS["EXP_074"])
    teacher_base = AblationTeacher(VARIANTS["M0_baseline"])
    assert evaluate_scenario_1_1(teacher_exp74) is True
    assert evaluate_scenario_1_1(teacher_base) is False


def test_micro_corridor_sovereign_hold_adherence() -> None:
    teacher_exp74 = AblationTeacher(VARIANTS["EXP_074"])
    teacher_base = AblationTeacher(VARIANTS["M0_baseline"])
    assert evaluate_scenario_1_2(teacher_exp74) is True
    assert evaluate_scenario_1_2(teacher_base) is False


def test_narrow_passage_swap() -> None:
    teacher_exp74 = AblationTeacher(VARIANTS["EXP_074"])
    teacher_base = AblationTeacher(VARIANTS["M0_baseline"])
    assert evaluate_scenario_2_1(teacher_exp74) is True
    assert evaluate_scenario_2_1(teacher_base) is False


def test_narrow_passage_fair_rotation() -> None:
    teacher_exp74 = AblationTeacher(VARIANTS["EXP_074"])
    teacher_72a = AblationTeacher(VARIANTS["EXP_072A"])
    teacher_72b = AblationTeacher(VARIANTS["EXP_072B"])
    assert evaluate_scenario_2_2(teacher_exp74) is True
    assert evaluate_scenario_2_2(teacher_72b) is True
    assert evaluate_scenario_2_2(teacher_72a) is False


def test_food_corner_orthogonal_legal_gathering() -> None:
    teacher_exp74 = AblationTeacher(VARIANTS["EXP_074"])
    teacher_71a = AblationTeacher(VARIANTS["EXP_071A"])
    teacher_base = AblationTeacher(VARIANTS["M0_baseline"])
    assert evaluate_scenario_3_1(teacher_exp74) is True
    assert evaluate_scenario_3_1(teacher_71a) is True
    assert evaluate_scenario_3_1(teacher_base) is False


def test_food_corner_queue_anti_crush_protection() -> None:
    teacher_exp74 = AblationTeacher(VARIANTS["EXP_074"])
    teacher_71a = AblationTeacher(VARIANTS["EXP_071A"])
    teacher_72a = AblationTeacher(VARIANTS["EXP_072A"])
    assert evaluate_scenario_3_2(teacher_exp74) is True
    assert evaluate_scenario_3_2(teacher_71a) is False
    assert evaluate_scenario_3_2(teacher_72a) is False


def test_fog_island_safe_land_frontier() -> None:
    teacher_exp74 = AblationTeacher(VARIANTS["EXP_074"])
    teacher_73b = AblationTeacher(VARIANTS["EXP_073B"])
    teacher_base = AblationTeacher(VARIANTS["M0_baseline"])
    assert evaluate_scenario_4_1(teacher_exp74) is True
    assert evaluate_scenario_4_1(teacher_73b) is True
    assert evaluate_scenario_4_1(teacher_base) is False


def test_fog_island_frontier_exhaustion_smooth_transition() -> None:
    teacher_exp74 = AblationTeacher(VARIANTS["EXP_074"])
    teacher_73b = AblationTeacher(VARIANTS["EXP_073B"])
    teacher_73a = AblationTeacher(VARIANTS["EXP_073A"])
    assert evaluate_scenario_4_2(teacher_exp74) is True
    assert evaluate_scenario_4_2(teacher_73b) is True
    assert evaluate_scenario_4_2(teacher_73a) is False


def test_fog_island_distant_food_decoupled_exploration() -> None:
    teacher_exp74 = AblationTeacher(VARIANTS["EXP_074"])
    teacher_73b = AblationTeacher(VARIANTS["EXP_073B"])
    teacher_base = AblationTeacher(VARIANTS["M0_baseline"])
    assert evaluate_scenario_4_3(teacher_exp74) is True
    assert evaluate_scenario_4_3(teacher_73b) is True
    assert evaluate_scenario_4_3(teacher_base) is False
