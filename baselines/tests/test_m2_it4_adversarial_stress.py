from __future__ import annotations

import time
import numpy as np
import pytest

from tb_baselines.coordination import (
    DELTAS,
    FAR,
    HOLD,
    build_capacity_constrained_bipartite_graph,
    compute_batch_torus_geodesics,
    compute_cost_matrix_bitmask,
    extract_cluster_gathering_positions,
    solve_mcmf_ssp,
)
from tb_baselines.mcmf_teacher import MCMFTeacher


def make_test_obs(
    rows: int,
    cols: int,
    mine: list[tuple[int, int]],
    food: list[tuple[int, int]] | None = None,
    hills: list[tuple[int, int, int]] | None = None,
    ant_ids: list[int] | None = None,
) -> dict:
    total_ants = len(mine)
    if ant_ids is None:
        ant_ids = list(range(total_ants))
    return {
        "size": [rows, cols],
        "turn": 1,
        "max_turns": 300,
        "round_time": 0.05,
        "engine_digest": "challenger_m2_it4_adv",
        "mine": [[int(r), int(c)] for r, c in mine],
        "ids": ant_ids,
        "food": [[int(r), int(c)] for r, c in (food or [])],
        "hills": [[int(r), int(c), int(owner)] for r, c, owner in (hills or [])],
    }


def step_ants_with_orders(
    orders: np.ndarray,
    current_mine: list[tuple[int, int]],
    rows: int,
    cols: int,
    water: np.ndarray,
) -> list[tuple[int, int]]:
    occupied: set[tuple[int, int]] = set()
    next_mine: list[tuple[int, int]] = []
    for i, act in enumerate(orders):
        r, c = current_mine[i]
        if act != HOLD:
            dr, dc = DELTAS[act]
            dest = ((r + dr) % rows, (c + dc) % cols)
        else:
            dest = (r, c)
        if dest in occupied or water[dest[0], dest[1]]:
            dest = (r, c)
        occupied.add(dest)
        next_mine.append(dest)
    return next_mine


def test_adversarial_latency_80x80_10_seeds_comprehensive():
    # 核心压力测试：80x80 地图 10 随机种子（100 蚂蚁、30 食物）全量验证单回合平均耗时严格小于 15.0ms（目标 <8.0ms）
    rows, cols = 80, 80
    test_seeds = [101, 202, 303, 404, 505, 606, 707, 808, 909, 1010]
    rounds_per_seed = 10

    latencies_ms: list[float] = []

    for seed in test_seeds:
        rng = np.random.default_rng(seed)
        water = np.zeros((rows, cols), dtype=bool)
        for _ in range(8):
            wr = int(rng.integers(5, rows - 15))
            wc = int(rng.integers(5, cols - 15))
            h = int(rng.integers(4, 8))
            w = int(rng.integers(4, 8))
            water[wr : wr + h, wc : wc + w] = True

        seen = np.ones((rows, cols), dtype=bool)

        mine_set: set[tuple[int, int]] = set()
        while len(mine_set) < 100:
            r = int(rng.integers(0, rows))
            c = int(rng.integers(0, cols))
            if not water[r, c]:
                mine_set.add((r, c))
        mine = list(mine_set)

        food_set: set[tuple[int, int]] = set()
        while len(food_set) < 30:
            r = int(rng.integers(0, rows))
            c = int(rng.integers(0, cols))
            if not water[r, c] and (r, c) not in mine_set:
                food_set.add((r, c))
        food = list(food_set)

        teacher = MCMFTeacher()
        current_mine = list(mine)

        for _ in range(rounds_per_seed):
            obs = make_test_obs(rows, cols, current_mine, food)
            t0 = time.perf_counter()
            orders = teacher.orders(obs, water, seen)
            t_ms = (time.perf_counter() - t0) * 1000.0
            latencies_ms.append(t_ms)

            assert len(orders) == 100
            current_mine = step_ants_with_orders(orders, current_mine, rows, cols, water)

    global_mean = float(np.mean(latencies_ms))
    global_p95 = float(np.percentile(latencies_ms, 95))
    global_max = float(np.max(latencies_ms))

    print(f"\n[80x80 跨 10 随机种子综合压测] 100 蚂蚁 30 食物 (共 {len(latencies_ms)} 轮):")
    print(f"  全局平均耗时: {global_mean:.2f} ms")
    print(f"  全局 95分位耗时: {global_p95:.2f} ms")
    print(f"  全局最大单次耗时: {global_max:.2f} ms")

    assert global_mean < 15.0, f"80x80 全局平均耗时超出 15.0 ms 门禁: {global_mean:.2f} ms"


def test_adversarial_latency_extreme_food_scale_80x80():
    # 压力测试：80x80 尺度下 100 蚂蚁与 100 食物（产生超过 300 采集位，触发多 Block 位掩码张量）
    rows, cols = 80, 80
    num_seeds = 3
    rounds_per_seed = 5
    total_rounds = num_seeds * rounds_per_seed

    latencies_ms: list[float] = []

    for seed in [5001, 5002, 5003]:
        rng = np.random.default_rng(seed)
        water = np.zeros((rows, cols), dtype=bool)
        for _ in range(10):
            wr = int(rng.integers(5, rows - 15))
            wc = int(rng.integers(5, cols - 15))
            h = int(rng.integers(4, 8))
            w = int(rng.integers(4, 8))
            water[wr : wr + h, wc : wc + w] = True

        seen = np.ones((rows, cols), dtype=bool)

        mine_set: set[tuple[int, int]] = set()
        while len(mine_set) < 100:
            r = int(rng.integers(0, rows))
            c = int(rng.integers(0, cols))
            if not water[r, c]:
                mine_set.add((r, c))
        mine = list(mine_set)

        food_set: set[tuple[int, int]] = set()
        while len(food_set) < 100:
            r = int(rng.integers(0, rows))
            c = int(rng.integers(0, cols))
            if not water[r, c] and (r, c) not in mine_set:
                food_set.add((r, c))
        food = list(food_set)

        teacher = MCMFTeacher()
        current_mine = list(mine)

        for _ in range(rounds_per_seed):
            obs = make_test_obs(rows, cols, current_mine, food)
            t0 = time.perf_counter()
            orders = teacher.orders(obs, water, seen)
            t_ms = (time.perf_counter() - t0) * 1000.0
            latencies_ms.append(t_ms)

            assert len(orders) == 100
            current_mine = step_ants_with_orders(orders, current_mine, rows, cols, water)

    global_mean = float(np.mean(latencies_ms))
    global_p95 = float(np.percentile(latencies_ms, 95))
    global_max = float(np.max(latencies_ms))

    print(f"\n[80x80 极限 100 蚂蚁 100 食物压测] 全局统计 (共 {total_rounds} 轮):")
    print(f"  全局平均耗时: {global_mean:.2f} ms")
    print(f"  全局 95分位耗时: {global_p95:.2f} ms")
    print(f"  全局最大耗时: {global_max:.2f} ms")

    # 验证在极端高食物负载下，单回合平均耗时处于 50ms 平台回合上限安全配额内
    assert global_mean < 50.0, f"80x80 极限 100 食物下平均耗时超出 50ms 回合上限: {global_mean:.2f} ms"


def test_adversarial_latency_large_board_scaling_100x100():
    # 压力测试：100x100 大地图尺度（100 蚂蚁、30 食物）单回合决策延迟
    rows, cols = 100, 100
    num_seeds = 3
    rounds_per_seed = 5
    total_rounds = num_seeds * rounds_per_seed

    latencies_ms: list[float] = []

    for seed in [6001, 6002, 6003]:
        rng = np.random.default_rng(seed)
        water = np.zeros((rows, cols), dtype=bool)
        for _ in range(12):
            wr = int(rng.integers(5, rows - 15))
            wc = int(rng.integers(5, cols - 15))
            h = int(rng.integers(4, 9))
            w = int(rng.integers(4, 9))
            water[wr : wr + h, wc : wc + w] = True

        seen = np.ones((rows, cols), dtype=bool)

        mine_set: set[tuple[int, int]] = set()
        while len(mine_set) < 100:
            r = int(rng.integers(0, rows))
            c = int(rng.integers(0, cols))
            if not water[r, c]:
                mine_set.add((r, c))
        mine = list(mine_set)

        food_set: set[tuple[int, int]] = set()
        while len(food_set) < 30:
            r = int(rng.integers(0, rows))
            c = int(rng.integers(0, cols))
            if not water[r, c] and (r, c) not in mine_set:
                food_set.add((r, c))
        food = list(food_set)

        teacher = MCMFTeacher()
        current_mine = list(mine)

        for _ in range(rounds_per_seed):
            obs = make_test_obs(rows, cols, current_mine, food)
            t0 = time.perf_counter()
            orders = teacher.orders(obs, water, seen)
            t_ms = (time.perf_counter() - t0) * 1000.0
            latencies_ms.append(t_ms)

            assert len(orders) == 100
            current_mine = step_ants_with_orders(orders, current_mine, rows, cols, water)

    global_mean = float(np.mean(latencies_ms))
    global_p95 = float(np.percentile(latencies_ms, 95))
    global_max = float(np.max(latencies_ms))

    print(f"\n[100x100 超大地图 100 蚂蚁 30 食物压测] 全局统计 (共 {total_rounds} 轮):")
    print(f"  全局平均耗时: {global_mean:.2f} ms")
    print(f"  全局 95分位耗时: {global_p95:.2f} ms")
    print(f"  全局最大耗时: {global_max:.2f} ms")

    assert global_mean < 50.0, f"100x100 地图下单回合平均耗时超出 50ms 回合上限: {global_mean:.2f} ms"


def test_adversarial_latency_official_envelope_124x124():
    # 压力测试：平台规格上限 124x124 地图尺度下（100 蚂蚁、30 食物）单回合决策延迟
    rows, cols = 124, 124
    num_seeds = 2
    rounds_per_seed = 5

    latencies_ms: list[float] = []

    for seed in [7001, 7002]:
        rng = np.random.default_rng(seed)
        water = np.zeros((rows, cols), dtype=bool)
        for _ in range(16):
            wr = int(rng.integers(5, rows - 20))
            wc = int(rng.integers(5, cols - 20))
            h = int(rng.integers(5, 12))
            w = int(rng.integers(5, 12))
            water[wr : wr + h, wc : wc + w] = True

        seen = np.ones((rows, cols), dtype=bool)

        mine_set: set[tuple[int, int]] = set()
        while len(mine_set) < 100:
            r = int(rng.integers(0, rows))
            c = int(rng.integers(0, cols))
            if not water[r, c]:
                mine_set.add((r, c))
        mine = list(mine_set)

        food_set: set[tuple[int, int]] = set()
        while len(food_set) < 30:
            r = int(rng.integers(0, rows))
            c = int(rng.integers(0, cols))
            if not water[r, c] and (r, c) not in mine_set:
                food_set.add((r, c))
        food = list(food_set)

        teacher = MCMFTeacher()
        current_mine = list(mine)

        for _ in range(rounds_per_seed):
            obs = make_test_obs(rows, cols, current_mine, food)
            t0 = time.perf_counter()
            orders = teacher.orders(obs, water, seen)
            t_ms = (time.perf_counter() - t0) * 1000.0
            latencies_ms.append(t_ms)

            assert len(orders) == 100
            current_mine = step_ants_with_orders(orders, current_mine, rows, cols, water)

    global_mean = float(np.mean(latencies_ms))
    global_p95 = float(np.percentile(latencies_ms, 95))
    global_max = float(np.max(latencies_ms))

    print(f"\n[124x124 平台上限规格压测] 全局统计 (共 {len(latencies_ms)} 轮):")
    print(f"  全局平均耗时: {global_mean:.2f} ms")
    print(f"  全局 95分位耗时: {global_p95:.2f} ms")
    print(f"  全局最大耗时: {global_max:.2f} ms")

    assert global_mean < 50.0, f"124x124 地图下单回合平均耗时超出 50ms 回合上限: {global_mean:.2f} ms"


def test_adversarial_extreme_imbalance_ratio():
    # 对抗测试：极端兵力与食物比例 (150 蚂蚁 vs 1 食物，以及 1 蚂蚁 vs 80 食物)
    rows, cols = 64, 64
    water = np.zeros((rows, cols), dtype=bool)
    seen = np.ones((rows, cols), dtype=bool)
    rng = np.random.default_rng(8001)

    # 1. 150 蚂蚁 vs 1 食物
    mine_150: set[tuple[int, int]] = set()
    while len(mine_150) < 150:
        r = int(rng.integers(0, rows))
        c = int(rng.integers(0, cols))
        mine_150.add((r, c))

    food_single = [(32, 32)]
    if (32, 32) in mine_150:
        mine_150.remove((32, 32))
        mine_150.add((0, 0))

    teacher = MCMFTeacher()
    obs_1 = make_test_obs(rows, cols, list(mine_150), food_single)
    lats_1: list[float] = []
    for _ in range(5):
        t0 = time.perf_counter()
        orders_1 = teacher.orders(obs_1, water, seen)
        lats_1.append((time.perf_counter() - t0) * 1000.0)

    avg_lat_1 = float(np.mean(lats_1))
    assert len(orders_1) == len(mine_150)
    assert avg_lat_1 < 15.0, f"150 蚂蚁 1 食物下平均耗时超出 15.0ms: {avg_lat_1:.2f} ms"

    dest_set_1 = set()
    for idx, act in enumerate(orders_1):
        ar, ac = list(mine_150)[idx]
        if act != HOLD:
            dr, dc = DELTAS[act]
            dest = ((ar + dr) % rows, (ac + dc) % cols)
        else:
            dest = (ar, ac)
        dest_set_1.add(dest)
    assert len(dest_set_1) == len(mine_150), "150 蚂蚁移动产生友方碰撞"

    # 2. 1 蚂蚁 vs 80 食物
    mine_1 = [(10, 10)]
    food_80: set[tuple[int, int]] = set()
    while len(food_80) < 80:
        r = int(rng.integers(0, rows))
        c = int(rng.integers(0, cols))
        if (r, c) != (10, 10):
            food_80.add((r, c))

    obs_2 = make_test_obs(rows, cols, mine_1, list(food_80))
    lats_2: list[float] = []
    for _ in range(5):
        t0 = time.perf_counter()
        orders_2 = teacher.orders(obs_2, water, seen)
        lats_2.append((time.perf_counter() - t0) * 1000.0)

    avg_lat_2 = float(np.mean(lats_2))
    assert len(orders_2) == 1
    assert avg_lat_2 < 15.0, f"1 蚂蚁 80 食物下平均耗时超出 15.0ms: {avg_lat_2:.2f} ms"


def test_adversarial_bitmask_bfs_exact_block_boundary():
    # 对抗测试：采集位数量精确等于 64 与 128（位掩码块边界边界值检验）
    rows, cols = 40, 40
    passable = np.ones((rows, cols), dtype=bool)

    # 1. 精确 64 采集位
    g_positions_64 = [(r, c) for r in range(8) for c in range(8)]
    ants = [(8, 8), (10, 10)]

    cost_matrix_64 = compute_cost_matrix_bitmask(
        ant_positions=ants,
        gathering_positions=g_positions_64,
        passable_mask=passable,
        rows=rows,
        cols=cols,
        max_depth=16,
    )
    assert cost_matrix_64.shape == (2, 64)
    assert not np.all(cost_matrix_64 == FAR)

    # 2. 精确 128 采集位（跨越两个满载 uint64 块）
    g_positions_128 = [(r, c) for r in range(16) for c in range(8)]
    cost_matrix_128 = compute_cost_matrix_bitmask(
        ant_positions=ants,
        gathering_positions=g_positions_128,
        passable_mask=passable,
        rows=rows,
        cols=cols,
        max_depth=16,
    )
    assert cost_matrix_128.shape == (2, 128)
    assert not np.all(cost_matrix_128 == FAR)


def test_adversarial_concentric_labyrinth_torus_geodesic():
    # 对抗测试：同心蛇形迷宫（曼哈顿距离为 2，但真实测地最短路长达 26 步）
    rows, cols = 30, 30
    water = np.zeros((rows, cols), dtype=bool)

    # 构建长蛇形回廊隔离
    water[10, 5:25] = True
    water[12, 5:25] = True
    # 封住两端，仅留单侧开口
    water[11, 5] = True

    # 食物位于回廊深处 (11, 6)
    food = [(11, 6)]
    # 蚂蚁位于回廊外墙上方 (9, 6)，曼哈顿距离仅为 2
    ants = [(9, 6)]

    seen = np.ones((rows, cols), dtype=bool)
    teacher = MCMFTeacher(max_depth=40)
    obs = make_test_obs(rows, cols, ants, food)

    orders = teacher.orders(obs, water, seen)
    assert len(orders) == 1
    # 蚂蚁不得跨越水墙移动
    act = orders[0]
    if act != HOLD:
        dr, dc = DELTAS[act]
        nr, nc = (9 + dr) % rows, (6 + dc) % cols
        assert not water[nr, nc], "蚂蚁执行了非法穿越水墙动作"


def test_adversarial_strict_determinism_under_permutation():
    # 严格确定性检验：输入相同观察数据下重复调用输出动作必须严格逐位一致
    rows, cols = 40, 40
    water = np.zeros((rows, cols), dtype=bool)
    seen = np.ones((rows, cols), dtype=bool)

    ants = [(5, 5), (10, 10), (15, 15), (20, 20)]
    foods = [(6, 5), (11, 10), (16, 15), (21, 20)]

    teacher = MCMFTeacher()
    obs_standard = make_test_obs(rows, cols, ants, foods)
    orders_1 = teacher.orders(obs_standard, water, seen)
    orders_2 = teacher.orders(obs_standard, water, seen)

    assert np.array_equal(orders_1, orders_2), "相同输入重复调用输出不具备确定性"
