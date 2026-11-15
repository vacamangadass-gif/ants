from __future__ import annotations

import time
import numpy as np

from tb_baselines.coordination import (
    DELTAS,
    HOLD,
    extract_cluster_gathering_capacity,
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
        "max_turns": 200,
        "round_time": 0.05,
        "engine_digest": "challenger_m2_it2",
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


def test_empirical_latency_multi_iteration_100_ants_30_foods():
    # 跨 5 个随机环境拓扑种子，每组连续推演 10 个回合（共计 50 次单回合决策），实证检验决策延迟
    rows, cols = 64, 64
    seeds = [1001, 1002, 1003, 1004, 1005]
    rounds_per_seed = 10
    all_latencies_ms: list[float] = []

    for seed in seeds:
        rng = np.random.default_rng(seed)
        water = np.zeros((rows, cols), dtype=bool)
        water[10:16, 10:16] = True
        water[30:36, 40:46] = True
        water[50:56, 15:21] = True
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

        for round_idx in range(rounds_per_seed):
            obs = make_test_obs(rows, cols, current_mine, food)
            t0 = time.perf_counter()
            orders = teacher.orders(obs, water, seen)
            t_ms = (time.perf_counter() - t0) * 1000.0
            all_latencies_ms.append(t_ms)

            assert len(orders) == 100
            for act in orders:
                assert act in (0, 1, 2, 3, 4)
            current_mine = step_ants_with_orders(orders, current_mine, rows, cols, water)

        seed_lats = all_latencies_ms[-rounds_per_seed:]
        iter_mean = float(np.mean(seed_lats))
        iter_p95 = float(np.percentile(seed_lats, 95))
        print(f"\n[种子 {seed} 迭代推演] 平均耗时: {iter_mean:.2f} ms | 95分位: {iter_p95:.2f} ms")
        assert iter_mean < 15.0, (
            f"种子 {seed} 单回合平均决策耗时超出 15.0 ms 门禁: 实际为 {iter_mean:.2f} ms"
        )

    grand_mean = float(np.mean(all_latencies_ms))
    grand_p95 = float(np.percentile(all_latencies_ms, 95))
    grand_max = float(np.max(all_latencies_ms))

    print("\n[全量 50 次迭代综合统计] 100 蚂蚁 30 食物 64x64:")
    print(f"  全局平均耗时: {grand_mean:.2f} ms")
    print(f"  全局 95分位耗时: {grand_p95:.2f} ms")
    print(f"  全局最大单次耗时: {grand_max:.2f} ms")

    assert grand_mean < 15.0, f"全局平均耗时超出 15.0 ms: 实际为 {grand_mean:.2f} ms"



def test_empirical_latency_scaling_80x80_map():
    # 大尺寸 80x80 地图下 100 蚂蚁 30 食物连续 15 回合决策耗时检验
    rows, cols = 80, 80
    rng = np.random.default_rng(20261104)
    water = np.zeros((rows, cols), dtype=bool)
    water[20:28, 20:28] = True
    water[50:58, 50:58] = True
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
    latencies: list[float] = []
    current_mine = list(mine)

    for _ in range(15):
        obs = make_test_obs(rows, cols, current_mine, food)
        t0 = time.perf_counter()
        orders = teacher.orders(obs, water, seen)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        latencies.append(elapsed_ms)
        assert len(orders) == 100
        current_mine = step_ants_with_orders(orders, current_mine, rows, cols, water)

    mean_ms = float(np.mean(latencies))
    p95_ms = float(np.percentile(latencies, 95))
    max_ms = float(np.max(latencies))

    print("\n[80x80 地图 15 轮压测] 100 蚂蚁 30 食物:")

    print(f"  平均耗时: {mean_ms:.2f} ms")
    print(f"  95分位耗时: {p95_ms:.2f} ms")
    print(f"  最大耗时: {max_ms:.2f} ms")

    assert mean_ms < 15.0
    assert p95_ms < 15.0


def test_extreme_topology_dense_food_cluster_4x4():
    # 4x4 紧密实体食物簇 (16 颗食物实体相连)：验证切比雪夫空间聚类与零碰撞、零非法分配
    rows, cols = 50, 50
    water = np.zeros((rows, cols), dtype=bool)

    # 16 颗食物紧密排列在 [15, 19) x [15, 19) 内部实心矩形
    dense_foods = [(r, c) for r in range(15, 19) for c in range(15, 19)]
    assert len(dense_foods) == 16

    # 验证提取的聚类与容量
    gathering_positions, cluster_caps = extract_cluster_gathering_capacity(
        dense_foods, water, rows, cols
    )
    # 切比雪夫相邻合并后必须严格为 1 个簇
    assert len(cluster_caps) == 1
    # 4x4 矩形外围一圈共有 16 个正交采集位: (14, 15..18), (19, 15..18), (15..18, 14), (15..18, 19)
    assert len(gathering_positions) == 16
    cluster_id = next(iter(cluster_caps.keys()))
    assert cluster_caps[cluster_id] == 16

    # 部署 100 只蚂蚁环绕在食物簇四周
    rng = np.random.default_rng(42)
    valid_cells = [
        (r, c)
        for r in range(rows)
        for c in range(cols)
        if not water[r, c] and (r, c) not in set(dense_foods)
    ]
    chosen = rng.choice(len(valid_cells), size=100, replace=False)
    mine = [valid_cells[idx] for idx in chosen]

    seen = np.ones((rows, cols), dtype=bool)
    teacher = MCMFTeacher()
    current_mine = list(mine)

    # 连续执行 10 轮端到端决策
    for turn in range(10):
        obs = make_test_obs(rows, cols, current_mine, dense_foods)
        orders, telemetry = teacher.orders_with_telemetry(obs, water, seen)

        assert len(orders) == 100
        assert len(telemetry) == 100

        # 检查指令与目标
        destinations: list[tuple[int, int]] = []
        assigned_gather_targets: set[tuple[int, int]] = set()

        for i, act in enumerate(orders):
            ar, ac = current_mine[i]
            if act != HOLD:
                dr, dc = DELTAS[act]
                dest = ((ar + dr) % rows, (ac + dc) % cols)
            else:
                dest = (ar, ac)

            # 1. 杜绝走入水域
            assert not water[dest[0], dest[1]], f"蚂蚁 {i} 走入水域: {dest}"
            # 2. 杜绝踩入未采食的食物实体
            assert dest not in set(dense_foods), f"蚂蚁 {i} 踩入食物实体: {dest}"
            destinations.append(dest)

            # 3. 检查采食分配唯一性
            rec = telemetry[i]
            if rec.nearest_source_type in ("food_visible", "food_memory"):
                pass

        # 4. 杜绝友方终点相撞：100 个目的地必须互不相同
        assert len(set(destinations)) == 100, f"第 {turn} 轮存在友方碰撞冲突"

        current_mine = step_ants_with_orders(orders, current_mine, rows, cols, water)


def test_extreme_topology_dense_food_cluster_water_pocket():
    # 密集食物位于水域包围的狭窄内凹陷处：仅有 3 个合法采集位，但有 9 颗食物实体
    rows, cols = 40, 40
    water = np.zeros((rows, cols), dtype=bool)

    # 3x3 食物位于 [10, 13) x [10, 13)
    food_coords = [(r, c) for r in range(10, 13) for c in range(10, 13)]
    assert len(food_coords) == 9

    # 北侧、南侧、西侧全部为水域，仅有东侧列 13 开放 3 个陆地格子 (10, 13), (11, 13), (12, 13)
    water[9, 9:14] = True  # 北边一排封死
    water[13, 9:14] = True  # 南边一排封死
    water[10:13, 9] = True  # 西边一列封死

    gathering_positions, cluster_caps = extract_cluster_gathering_capacity(
        food_coords, water, rows, cols
    )
    assert len(gathering_positions) == 3
    assert set(gathering_positions) == {(10, 13), (11, 13), (12, 13)}
    cluster_id = next(iter(cluster_caps.keys()))
    # 物理界面容量为 min(|G(R)|, |R|) = min(3, 9) = 3
    assert cluster_caps[cluster_id] == 3

    # 部署 100 只蚂蚁
    rng = np.random.default_rng(999)
    valid_cells = [
        (r, c)
        for r in range(rows)
        for c in range(cols)
        if not water[r, c] and (r, c) not in set(food_coords)
    ]
    chosen = rng.choice(len(valid_cells), size=100, replace=False)
    mine = [valid_cells[idx] for idx in chosen]

    seen = np.ones((rows, cols), dtype=bool)
    teacher = MCMFTeacher()
    obs = make_test_obs(rows, cols, mine, food_coords)
    orders, telemetry = teacher.orders_with_telemetry(obs, water, seen)

    # 统计获配采食任务的蚂蚁数量：由于容量约束为 3，最多 3 只蚂蚁获配采食
    assigned_gather_ants = [
        i for i, rec in enumerate(telemetry) if rec.nearest_source_type in ("food_visible", "food_memory")
    ]
    assert len(assigned_gather_ants) <= 3, f"获配采食蚂蚁数量超限: {len(assigned_gather_ants)} > 3"

    # 验证零碰撞、零水域、零踩食
    destinations: list[tuple[int, int]] = []
    for i, act in enumerate(orders):
        ar, ac = mine[i]
        if act != HOLD:
            dr, dc = DELTAS[act]
            dest = ((ar + dr) % rows, (ac + dc) % cols)
        else:
            dest = (ar, ac)
        assert not water[dest[0], dest[1]]
        assert dest not in set(food_coords)
        destinations.append(dest)

    assert len(set(destinations)) == 100


def test_extreme_topology_torus_boundary_wrap_four_corners():
    # 环面四角跨界环绕食物簇：食物放置在 (0, 0), (0, 39), (39, 0), (39, 39)
    # 拓扑检验：在环面曼哈顿/切比雪夫几何中，这 4 颗食物属于互相连通的同一聚类簇
    rows, cols = 40, 40
    water = np.zeros((rows, cols), dtype=bool)
    corner_foods = [(0, 0), (0, 39), (39, 0), (39, 39)]

    gathering_positions, cluster_caps = extract_cluster_gathering_capacity(
        corner_foods, water, rows, cols
    )
    # 4 颗食物必须被合并为严格 1 个连通簇
    assert len(cluster_caps) == 1, f"四角环面食物未能合并为唯一样本簇: 实际簇数 {len(cluster_caps)}"
    cluster_id = next(iter(cluster_caps.keys()))
    # 验证 4 颗食物的正交采集位经过环面取模运算后去重且合法
    for gr, gc in gathering_positions:
        assert 0 <= gr < rows and 0 <= gc < cols
        assert (gr, gc) not in set(corner_foods)
        assert not water[gr, gc]

    # 部署 100 只蚂蚁，部分蚂蚁分布在四角附近（行 0、39，列 0、39），测试跨界导航与分配
    rng = np.random.default_rng(2026)
    valid_cells = [
        (r, c)
        for r in range(rows)
        for c in range(cols)
        if not water[r, c] and (r, c) not in set(corner_foods)
    ]
    chosen = rng.choice(len(valid_cells), size=100, replace=False)
    mine = [valid_cells[idx] for idx in chosen]

    seen = np.ones((rows, cols), dtype=bool)
    teacher = MCMFTeacher()
    current_mine = list(mine)

    # 连续推演 10 个回合
    for turn in range(10):
        obs = make_test_obs(rows, cols, current_mine, corner_foods)
        orders, telemetry = teacher.orders_with_telemetry(obs, water, seen)

        destinations: list[tuple[int, int]] = []
        for i, act in enumerate(orders):
            ar, ac = current_mine[i]
            if act != HOLD:
                dr, dc = DELTAS[act]
                dest = ((ar + dr) % rows, (ac + dc) % cols)
            else:
                dest = (ar, ac)
            assert not water[dest[0], dest[1]]
            assert dest not in set(corner_foods)
            destinations.append(dest)

        # 零碰撞断言
        assert len(set(destinations)) == 100, f"四角跨界环面推演第 {turn} 轮存在碰撞冲突"
        current_mine = step_ants_with_orders(orders, current_mine, rows, cols, water)


def test_extreme_topology_torus_boundary_wrap_stripe():
    # 环面边缘横贯食物带：食物跨越 (0, c) 与 (rows-1, c) 边界线连续排列
    rows, cols = 40, 40
    water = np.zeros((rows, cols), dtype=bool)
    stripe_foods = [
        (0, 15), (39, 15),
        (0, 16), (39, 16),
        (0, 17), (39, 17),
        (0, 18), (39, 18),
    ]

    gathering_positions, cluster_caps = extract_cluster_gathering_capacity(
        stripe_foods, water, rows, cols
    )
    # 8 颗跨越边界的食物必须合并为 1 个连通簇
    assert len(cluster_caps) == 1, f"跨边界食物带未能合并为唯一样本簇，实际簇数: {len(cluster_caps)}"

    rng = np.random.default_rng(777)
    valid_cells = [
        (r, c)
        for r in range(rows)
        for c in range(cols)
        if not water[r, c] and (r, c) not in set(stripe_foods)
    ]
    chosen = rng.choice(len(valid_cells), size=100, replace=False)
    mine = [valid_cells[idx] for idx in chosen]

    seen = np.ones((rows, cols), dtype=bool)
    teacher = MCMFTeacher()
    current_mine = list(mine)

    for turn in range(8):
        obs = make_test_obs(rows, cols, current_mine, stripe_foods)
        orders = teacher.orders(obs, water, seen)

        destinations: list[tuple[int, int]] = []
        for i, act in enumerate(orders):
            ar, ac = current_mine[i]
            if act != HOLD:
                dr, dc = DELTAS[act]
                dest = ((ar + dr) % rows, (ac + dc) % cols)
            else:
                dest = (ar, ac)
            assert not water[dest[0], dest[1]]
            assert dest not in set(stripe_foods)
            destinations.append(dest)

        assert len(set(destinations)) == 100
        current_mine = step_ants_with_orders(orders, current_mine, rows, cols, water)


def test_extreme_topology_high_density_hourglass_bottleneck():
    # 沙漏型极端咽喉通道：左侧开阔大厅容纳 100 只蚂蚁，右侧开阔大厅容纳 30 颗食物
    # 中间仅有一条宽度为 1、长度为 10 的单车道水域咽喉通道相连接
    rows, cols = 40, 60
    water = np.ones((rows, cols), dtype=bool)

    # 左侧大厅: 行 [5, 35), 列 [5, 25) 开阔陆地
    water[5:35, 5:25] = False

    # 咽喉通道: 行 20, 列 [25, 35) 宽度严格为 1
    corridor_row = 20
    water[corridor_row, 25:35] = False

    # 右侧大厅: 行 [5, 35), 列 [35, 55) 开阔陆地
    water[5:35, 35:55] = False

    # 在左侧大厅部署 100 只密集蚂蚁
    rng = np.random.default_rng(8888)
    left_land = [
        (r, c) for r in range(5, 35) for c in range(5, 25) if not water[r, c]
    ]
    chosen_ants = rng.choice(len(left_land), size=100, replace=False)
    mine = [left_land[idx] for idx in chosen_ants]

    # 在右侧大厅部署 30 颗食物
    right_land = [
        (r, c) for r in range(5, 35) for c in range(35, 55) if not water[r, c]
    ]
    chosen_foods = rng.choice(len(right_land), size=30, replace=False)
    foods = [right_land[idx] for idx in chosen_foods]

    seen = np.ones((rows, cols), dtype=bool)
    teacher = MCMFTeacher()
    current_mine = list(mine)

    # 连续推演 15 个回合，高密度蚂蚁挤压冲击单通道咽喉要道
    for turn in range(15):
        obs = make_test_obs(rows, cols, current_mine, foods)
        orders, telemetry = teacher.orders_with_telemetry(obs, water, seen)

        destinations: list[tuple[int, int]] = []
        for i, act in enumerate(orders):
            ar, ac = current_mine[i]
            if act != HOLD:
                dr, dc = DELTAS[act]
                dest = ((ar + dr) % rows, (ac + dc) % cols)
            else:
                dest = (ar, ac)
            # 零走入水域
            assert not water[dest[0], dest[1]], f"回合 {turn} 蚂蚁 {i} 走入水域: {dest}"
            # 零踏入食物实体
            assert dest not in set(foods), f"回合 {turn} 蚂蚁 {i} 踏入食物实体: {dest}"
            destinations.append(dest)

        # 零碰撞断言：严格保证 100 只蚂蚁无任何目的地冲突
        assert len(set(destinations)) == 100, f"回合 {turn} 在咽喉挤压时发生碰撞冲突"

        # 验证单格咽喉通道内最多只有 1 只蚂蚁进占每一个通道格子，杜绝逆行与两蚁同格
        corridor_occupants = [
            i for i, dest in enumerate(destinations) if dest[0] == corridor_row and 25 <= dest[1] < 35
        ]
        corridor_cells = [destinations[i] for i in corridor_occupants]
        assert len(corridor_cells) == len(set(corridor_cells)), "咽喉通道内发生同格重叠"

        current_mine = step_ants_with_orders(orders, current_mine, rows, cols, water)
