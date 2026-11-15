from __future__ import annotations

import time
import numpy as np
import pytest

from tb_baselines.coordination import (
    DELTAS,
    FAR,
    HOLD,
    build_capacity_constrained_bipartite_graph,
    clear_corridor_width_cache,
    compute_batch_torus_geodesics,
    compute_batch_torus_geodesics_multibucket,
    compute_corridor_width,
    compute_corridor_width_grid,
    compute_cost_matrix_bitmask,
    compute_cost_matrix_bitmask_multibucket,
    compute_dynamic_congestion_penalty,
    compute_local_ant_density,
    compute_local_density_field,
    compute_path_congestion_penalties,
    solve_mcmf_ssp,
    trace_geodesic_allocation_path,
)
from tb_baselines.mcmf_teacher import MCMFTeacher


# ---------------------------------------------------------------------------
# 挑战项 1: 非对称双通道分流临界检验
# ---------------------------------------------------------------------------

def build_asymmetric_dual_corridor(num_ants_in_narrow: int = 0):
    rows, cols = 40, 50
    water = np.ones((rows, cols), dtype=bool)
    entrance = (15, 10)
    exit_tile = (15, 35)

    water[entrance[0], entrance[1]] = False
    water[exit_tile[0], exit_tile[1]] = False

    # 短窄走廊: 行 10, 宽度为 1, 长度为 5 + 24 + 5 = 34 步
    narrow_row = 10
    for c in range(entrance[1], exit_tile[1] + 1):
        water[narrow_row, c] = False
    for r in range(narrow_row, entrance[0] + 1):
        water[r, entrance[1]] = False
        water[r, exit_tile[1]] = False

    # 长宽走廊: 行 24..27, 宽度为 4, 长度为 9 + 24 + 9 = 42 步
    wide_row = 24
    for c in range(entrance[1], exit_tile[1] + 1):
        for w in range(4):
            water[wide_row + w, c] = False
    for r in range(entrance[0], wide_row + 4):
        water[r, entrance[1]] = False
        water[r, exit_tile[1]] = False

    # 出口后方设置 10 颗食物与对应采集位，确保所有蚂蚁均能进入采食获配流
    foods = []
    gather_positions = []
    for offset in range(-5, 5):
        fr = exit_tile[0] + offset
        fc = exit_tile[1] + 4
        water[fr, exit_tile[1] : fc + 1] = False
        foods.append((fr, fc))
        gather_positions.append((fr, fc - 1))

    # 窄走廊中放置指定数量的蚂蚁
    narrow_ants = []
    for i in range(num_ants_in_narrow):
        narrow_c = entrance[1] + 3 + i * 3
        if narrow_c < exit_tile[1]:
            narrow_ants.append((narrow_row, narrow_c))

    deciding_ant = entrance
    all_ants = [deciding_ant] + narrow_ants

    return {
        "rows": rows,
        "cols": cols,
        "water": water,
        "passable": ~water,
        "entrance": entrance,
        "exit_tile": exit_tile,
        "foods": foods,
        "gather_positions": gather_positions,
        "deciding_ant": deciding_ant,
        "narrow_ants": narrow_ants,
        "all_ants": all_ants,
    }


def test_asymmetric_corridors_single_ant_no_false_diversion():
    # 验证单只蚂蚁前行时不触发分流，保持测地最短路（短窄走廊，North 动作 0）
    env = build_asymmetric_dual_corridor(num_ants_in_narrow=0)
    teacher = MCMFTeacher(
        max_depth=60,
        dynamic_congestion=True,
        congestion_alpha=10.0,
        congestion_beta=1.5,
    )
    obs = {
        "size": (env["rows"], env["cols"]),
        "mine": env["all_ants"],
        "food": env["foods"],
        "ids": list(range(len(env["all_ants"]))),
    }
    seen = np.ones((env["rows"], env["cols"]), dtype=bool)

    orders = teacher.step(obs, env["water"], seen)
    assert len(orders) == 1
    # 动作 0 为 North (短窄走廊方向)，动作 2 为 South (长宽走廊方向)
    assert orders[0] == 0, f"单只蚂蚁应选择测地最短路 North(0)，实际输出了动作 {orders[0]}"


def test_asymmetric_corridors_critical_diversion_threshold():
    # 扫描窄走廊内蚂蚁数量 k=0,1,2,3,4,5,6，验证临界分流转换点
    # 当 k=0, 1 时保持短窄走廊 (North 0)
    # 当 k>=3 时二次拥塞成本引发自动分流至长宽走廊 (South 2)
    actions = {}

    for k in range(7):
        env = build_asymmetric_dual_corridor(num_ants_in_narrow=k)
        teacher = MCMFTeacher(
            max_depth=60,
            dynamic_congestion=True,
            congestion_alpha=10.0,
            congestion_beta=1.5,
        )
        obs = {
            "size": (env["rows"], env["cols"]),
            "mine": env["all_ants"],
            "food": env["foods"],
            "ids": list(range(len(env["all_ants"]))),
        }
        seen = np.ones((env["rows"], env["cols"]), dtype=bool)

        orders = teacher.step(obs, env["water"], seen)
        deciding_action = int(orders[0])
        actions[k] = deciding_action

    # 验证单只或少量蚂蚁无假分流 (短窄走廊 North 动作 0)
    assert actions[0] == 0, f"k=0 时决策动作应为 0，实际为 {actions[0]}"
    assert actions[1] == 0, f"k=1 时决策动作应为 0，实际为 {actions[1]}"

    # 验证当 k>=3 时必须触发分流至长宽走廊 (South 动作 2)
    assert actions[3] == 2, f"k=3 时应触发分流输出 South(2)，实际为 {actions[3]}"
    assert actions[4] == 2, f"k=4 时应触发分流输出 South(2)，实际为 {actions[4]}"
    assert actions[5] == 2, f"k=5 时应触发分流输出 South(2)，实际为 {actions[5]}"
    assert actions[6] == 2, f"k=6 时应触发分流输出 South(2)，实际为 {actions[6]}"


# ---------------------------------------------------------------------------
# 挑战项 2: 环面拓扑连续性与无碰撞性质
# ---------------------------------------------------------------------------

def test_torus_boundary_corridor_width_continuity():
    # 构造跨越环面上下边界的宽度为 1 与宽度为 3 的走廊
    rows, cols = 30, 30
    water = np.ones((rows, cols), dtype=bool)

    # 走廊 A: 跨越上下边界 (行 28, 29, 0, 1)，列 10，宽度严格为 1
    for r in [28, 29, 0, 1]:
        water[r, 10] = False

    # 走廊 B: 跨越左右边界 (列 27, 28, 29, 0, 1, 2)，行 20..22，宽度严格为 3
    for c in [27, 28, 29, 0, 1, 2]:
        for r in range(20, 23):
            water[r, c] = False

    clear_corridor_width_cache()
    width_map = compute_corridor_width(water, rows, cols)
    width_grid = compute_corridor_width_grid(water, use_cache=False)

    # 验证走廊 A 跨界处的宽度计算
    for r in [28, 29, 0, 1]:
        assert width_map[r, 10] == 1, f"走廊 A 行 {r} 宽度应为 1，实际为 {width_map[r, 10]}"
        assert width_grid[r, 10] == 1, f"走廊 A 逐行网格行 {r} 宽度应为 1，实际为 {width_grid[r, 10]}"

    # 验证走廊 B 跨界处的宽度计算
    for c in [27, 28, 29, 0, 1, 2]:
        for r in range(20, 23):
            assert width_map[r, c] == 3, f"走廊 B 单元格 ({r}, {c}) 宽度应为 3，实际为 {width_map[r, c]}"
            assert width_grid[r, c] == 3, f"走廊 B 逐行网格 ({r}, {c}) 宽度应为 3，实际为 {width_grid[r, c]}"


def test_torus_boundary_congestion_and_geodesic_continuity():
    # 验证跨越环面边界时的局部密度卷积与多桶测地距离场连续无跳变
    rows, cols = 32, 32
    passable = np.ones((rows, cols), dtype=bool)

    # 边界蚂蚁对: (0, 5) 与 (rows-1, 5) 环面欧氏相邻
    boundary_ants = [(0, 5), (rows - 1, 5)]
    dens = compute_local_ant_density(boundary_ants, passable, rows, cols, radius=2)

    # 验证边界处的局部密度对称且无坐标越界或符号反转
    assert dens[0, 5] == dens[rows - 1, 5]
    assert dens[0, 5] > 0.0
    assert not np.isnan(dens).any()
    assert not np.isinf(dens).any()

    # 构造带延迟的跨界测地场
    cell_delay = np.zeros((rows, cols), dtype=np.int32)
    cell_delay[0, 5] = 2
    cell_delay[rows - 1, 5] = 2

    target = [(rows - 1, 5)]
    fields = compute_batch_torus_geodesics_multibucket(
        target, passable, cell_delay, rows, cols, max_depth=10, max_delay=4
    )
    dist_field = fields[0]

    cost_matrix = compute_cost_matrix_bitmask_multibucket(
        boundary_ants, target, passable, cell_delay, rows, cols, max_depth=10, max_delay=4
    )

    # 目标自身距离为 0
    assert dist_field[rows - 1, 5] == 0
    assert cost_matrix[1, 0] == 0

    # 跨界单元格 (0, 5) 测地场与延迟矩阵数值有效性
    assert dist_field[0, 5] > 0
    assert cost_matrix[0, 0] > 0
    assert not np.isinf(dist_field[0, 5])

    # 逆向路径回溯验证: 从 (0, 5) 沿环面边界回到 (rows - 1, 5)
    path = trace_geodesic_allocation_path((0, 5), target[0], dist_field, passable, rows, cols)
    assert path[0] == (0, 5)
    assert path[-1] == (rows - 1, 5)
    assert len(path) == 2


def test_multi_ant_simulation_no_collision_and_no_deadlock():
    # 构造环面密集通道多蚂蚁推演环境，执行 30 回合真实推演
    # 检验铁律:
    # 1. 任意时刻友方蚂蚁不发生坐标重叠 (互斥)
    # 2. 蚂蚁严禁踏入水域
    # 3. 蚂蚁持续推进采食与探索，不发生全军冻结死锁
    rows, cols = 32, 32
    water = np.zeros((rows, cols), dtype=bool)

    # 构造十字交叉隔断水墙
    water[8:12, 4:28] = True
    water[20:24, 4:28] = True
    water[8:12, 10] = False
    water[8:12, 22] = False
    water[20:24, 10] = False
    water[20:24, 22] = False

    passable = ~water
    rng = np.random.default_rng(2026)

    # 20 只蚂蚁
    ant_positions = []
    while len(ant_positions) < 20:
        r = int(rng.integers(0, rows))
        c = int(rng.integers(0, cols))
        if passable[r, c] and (r, c) not in ant_positions:
            ant_positions.append((r, c))

    # 8 颗初始食物
    foods = []
    while len(foods) < 8:
        r = int(rng.integers(0, rows))
        c = int(rng.integers(0, cols))
        if passable[r, c] and (r, c) not in ant_positions and (r, c) not in foods:
            foods.append((r, c))

    teacher = MCMFTeacher(
        max_depth=30,
        dynamic_congestion=True,
        congestion_alpha=10.0,
        congestion_beta=1.5,
    )

    current_ants = list(ant_positions)
    current_foods = set(foods)
    total_gathered = 0
    seen = np.zeros((rows, cols), dtype=bool)

    for turn in range(30):
        # 迷雾更新: 蚂蚁周围半径 5 可见
        for ar, ac in current_ants:
            for dr in range(-5, 6):
                for dc in range(-5, 6):
                    if dr * dr + dc * dc <= 25:
                        seen[(ar + dr) % rows, (ac + dc) % cols] = True

        visible_foods = [f for f in current_foods if seen[f[0], f[1]]]

        obs = {
            "size": (rows, cols),
            "mine": current_ants,
            "food": visible_foods,
            "ids": list(range(len(current_ants))),
        }
        orders = teacher.orders(obs, water, seen)
        assert len(orders) == len(current_ants)

        next_positions = []
        for i, action in enumerate(orders):
            ar, ac = current_ants[i]
            if action == HOLD:
                next_positions.append((ar, ac))
            else:
                dr, dc = DELTAS[int(action)]
                nr = (ar + dr) % rows
                nc = (ac + dc) % cols
                next_positions.append((nr, nc))

        # 检验 1: 友方无碰撞互斥
        assert len(next_positions) == len(set(next_positions)), (
            f"第 {turn} 回合发生友方坐标重叠碰撞: {next_positions}"
        )

        # 检验 2: 严禁踏入水域
        for pos in next_positions:
            assert not water[pos[0], pos[1]], f"第 {turn} 回合蚂蚁踏入水域方格: {pos}"

        # 采食并补给食物
        gathered_this_turn = set()
        for f in current_foods:
            for ar, ac in next_positions:
                diff_r = abs(ar - f[0])
                dr = min(diff_r, rows - diff_r)
                diff_c = abs(ac - f[1])
                dc = min(diff_c, cols - diff_c)
                if dr + dc <= 1:
                    gathered_this_turn.add(f)
                    break

        if gathered_this_turn:
            current_foods -= gathered_this_turn
            total_gathered += len(gathered_this_turn)
            while len(current_foods) < 8:
                r = int(rng.integers(0, rows))
                c = int(rng.integers(0, cols))
                if passable[r, c] and (r, c) not in next_positions and (r, c) not in current_foods:
                    current_foods.add((r, c))

        current_ants = next_positions

    # 检验 3: 采食活动持续推进，未死锁
    assert total_gathered > 20, f"30 回合内采食总数过低 ({total_gathered})，存在死锁停滞风险"


# ---------------------------------------------------------------------------
# 挑战项 3: 100 蚂蚁 64x64 规模单回合决策延迟基准检验
# ---------------------------------------------------------------------------

def test_challenger_latency_benchmark_100_ants_64x64():
    # 测量 100 蚂蚁、64x64 棋盘下单回合决策耗时
    # 门禁要求: 平均单回合耗时必须严格小于 10.0 ms
    rows, cols = 64, 64
    water = np.zeros((rows, cols), dtype=bool)
    water[15:25, 20:25] = True
    water[15:25, 35:40] = True
    water[40:50, 15:20] = True
    water[40:50, 45:50] = True
    for c in range(10, 50):
        water[30, c] = True
        water[32, c] = True
    water[31, 10:50] = False

    passable = ~water
    rng = np.random.default_rng(2026)

    ant_set = set()
    for c in range(20, 35):
        ant_set.add((31, c))
    while len(ant_set) < 100:
        r = int(rng.integers(0, rows))
        c = int(rng.integers(0, cols))
        if passable[r, c]:
            ant_set.add((r, c))
    ants = list(ant_set)

    food_set = set()
    while len(food_set) < 30:
        r = int(rng.integers(0, rows))
        c = int(rng.integers(0, cols))
        if passable[r, c] and (r, c) not in ant_set:
            food_set.add((r, c))
    foods = list(food_set)

    teacher = MCMFTeacher(
        max_depth=16,
        dynamic_congestion=True,
        congestion_alpha=10.0,
        congestion_beta=1.5,
    )
    obs = {
        "size": (rows, cols),
        "mine": ants,
        "food": foods,
        "ids": list(range(len(ants))),
    }
    seen = np.ones((rows, cols), dtype=bool)

    # 预热一次
    teacher.orders(obs, water, seen)

    # 执行 20 回合压测采样
    latencies = []
    for _ in range(20):
        t0 = time.perf_counter()
        orders = teacher.orders(obs, water, seen)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        latencies.append(elapsed_ms)
        assert len(orders) == 100

    mean_ms = float(np.mean(latencies))
    median_ms = float(np.median(latencies))
    p95_ms = float(np.percentile(latencies, 95))
    min_ms = float(np.min(latencies))
    max_ms = float(np.max(latencies))

    print(
        f"\n[LATENCY REPORT] 100 蚂蚁 64x64 单回合耗时:"
        f" Mean={mean_ms:.3f} ms, Median={median_ms:.3f} ms, P95={p95_ms:.3f} ms, "
        f" Min={min_ms:.3f} ms, Max={max_ms:.3f} ms"
    )

    # 门禁判定断言
    assert mean_ms < 10.0, f"单回合平均耗时 ({mean_ms:.3f} ms) 超过 10.0 ms 门禁上限"
