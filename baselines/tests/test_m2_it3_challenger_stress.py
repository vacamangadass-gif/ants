from __future__ import annotations

import time
import numpy as np
import pytest

from tb_baselines.coordination import (
    DELTAS,
    FAR,
    HOLD,
    MCMFGraph,
    assign_ants_to_exclusive_gathering,
    build_capacity_constrained_bipartite_graph,
    compute_batch_torus_geodesics,
    compute_single_source_torus_geodesic,
    extract_cluster_gathering_capacity,
    extract_cluster_gathering_positions,
    solve_mcmf_ssp,
    solve_mcmf_successive_shortest_path,
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
        "engine_digest": "challenger_m2_it3",
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


def test_adversarial_torus_wrap_narrow_corridor_horizontal():
    # 场景 1：宽度为 1 的水平长廊跨越环面列两端 (col cols-1 <-> col 0)
    rows, cols = 20, 60
    water = np.ones((rows, cols), dtype=bool)

    corridor_row = 10
    water[corridor_row, :] = False
    # 在列 30 设置水墙，封堵直接通行，迫使波前与蚂蚁必须跨越环面两端 (0 与 59)
    water[corridor_row, 30] = True

    # 食物位于 (10, 31)
    food_coords = [(corridor_row, 31)]
    # 合法采集位在东侧 (10, 32)
    gathering_positions, food_to_gather, gather_to_food = extract_cluster_gathering_positions(
        food_coords, water, rows, cols
    )
    assert len(gathering_positions) == 1
    assert gathering_positions[0] == (corridor_row, 32)

    passable = (~water).copy()
    passable[corridor_row, 31] = False

    geodesic_fields = compute_batch_torus_geodesics(
        gathering_positions, passable, rows, cols, max_depth=60
    )
    gf = geodesic_fields[0]

    # 验证采集位自身距离为 0
    assert gf[corridor_row, 32] == 0
    # 验证沿水平向东传播至边界 (10, 59)
    assert gf[corridor_row, 59] == 59 - 32  # 27
    # 验证跨越环面边界至 (10, 0)
    assert gf[corridor_row, 0] == 28
    # 验证继续向东传播至 (10, 21) 与 (10, 25)
    assert gf[corridor_row, 21] == 28 + 21  # 49
    assert gf[corridor_row, 25] == 28 + 25  # 53
    # 验证水墙西侧 (10, 29) 距离为 57
    assert gf[corridor_row, 29] == 57
    # 验证水墙本身不可通行
    assert gf[corridor_row, 30] == FAR

    # 部署 5 只蚂蚁排列在 (10, 21) 至 (10, 25)
    # 蚂蚁 0 位于 (10, 21)，测地距离为 49；蚂蚁 4 位于 (10, 25)，测地距离为 53
    ant_positions = [(corridor_row, 21 + i) for i in range(5)]
    cost_matrix = np.zeros((5, 1), dtype=np.int32)
    for i, (ar, ac) in enumerate(ant_positions):
        cost_matrix[i, 0] = gf[ar, ac]

    graph, offsets = build_capacity_constrained_bipartite_graph(
        ant_positions=ant_positions,
        food_positions=food_coords,
        gathering_positions=gathering_positions,
        geodesic_cost_matrix=cost_matrix,
        gather_to_food=gather_to_food,
        rows=rows,
        cols=cols,
        cluster_limit=4,
        g_per_cluster=1,
    )

    total_flow, total_cost, flow_dict = solve_mcmf_ssp(graph, offsets["source"], offsets["sink"])
    # 5 只蚂蚁争夺 1 个食物，流量严格为 1
    assert total_flow == 1
    # 距离最近的蚂蚁是第 0 号蚂蚁 (10, 21)，距离为 49
    assert total_cost == 49

    cand_seqs, assignment = assign_ants_to_exclusive_gathering(
        flow_dict,
        ant_positions,
        gathering_positions,
        geodesic_fields=geodesic_fields,
        passable_mask=passable,
        rows=rows,
        cols=cols,
        return_candidates=True,
    )
    assert assignment == {0: 0}
    # 验证动作下降方向：从 (10, 21) 沿长廊向西减小测地距离（走向 20, 19 ... 0 -> 59 -> 32）
    # DELTAS 中 (0, -1) 对应西向动作（索引 3）
    assert cand_seqs[0][0] == 3


def test_adversarial_torus_wrap_narrow_corridor_vertical():
    # 场景 2：宽度为 1 的垂直长廊跨越环面行两端 (row rows-1 <-> row 0)
    rows, cols = 30, 20
    water = np.ones((rows, cols), dtype=bool)

    corridor_col = 8
    water[:, corridor_col] = False
    # 在行 15 设置水墙，封堵直接通行
    water[15, corridor_col] = True

    # 食物位于 (16, 8)
    food_coords = [(16, corridor_col)]
    # 合法采集位在 (17, 8)
    gathering_positions, food_to_gather, gather_to_food = extract_cluster_gathering_positions(
        food_coords, water, rows, cols
    )
    assert len(gathering_positions) == 1
    assert gathering_positions[0] == (17, corridor_col)

    passable = (~water).copy()
    passable[16, corridor_col] = False

    geodesic_fields = compute_batch_torus_geodesics(
        gathering_positions, passable, rows, cols, max_depth=35
    )
    gf = geodesic_fields[0]

    # 验证沿长廊向下穿过边界 (29, 8) -> (0, 8)
    assert gf[17, corridor_col] == 0
    assert gf[29, corridor_col] == 12
    assert gf[0, corridor_col] == 13
    assert gf[14, corridor_col] == 27
    assert gf[15, corridor_col] == FAR

    # 蚂蚁部署在 (10, 8)
    ant_positions = [(10, corridor_col)]
    cost_matrix = np.array([[gf[10, corridor_col]]], dtype=np.int32)
    assert cost_matrix[0, 0] == 23

    graph, offsets = build_capacity_constrained_bipartite_graph(
        ant_positions=ant_positions,
        food_positions=food_coords,
        gathering_positions=gathering_positions,
        geodesic_cost_matrix=cost_matrix,
        gather_to_food=gather_to_food,
        rows=rows,
        cols=cols,
        cluster_limit=4,
        g_per_cluster=1,
    )
    flow, cost, flow_dict = solve_mcmf_ssp(graph, offsets["source"], offsets["sink"])
    assert flow == 1
    assert cost == 23

    cand_seqs, assignment = assign_ants_to_exclusive_gathering(
        flow_dict,
        ant_positions,
        gathering_positions,
        geodesic_fields=geodesic_fields,
        passable_mask=passable,
        rows=rows,
        cols=cols,
        return_candidates=True,
    )
    # 从 (10, 8) 下降走向 (9, 8)，对应北向移动 (-1, 0) 即动作 0
    assert cand_seqs[0][0] == 0


def test_adversarial_negative_cost_and_cycle_rejection():
    # 场景 3：验证负费用注入、零费用环路与异常维度的就地崩溃行为

    # 1. 直接初始正向负费用弧：构建图时就地拦截
    graph_neg = MCMFGraph(3, 10)
    graph_neg.add_edge(0, 1, cap=2, cost=-5)
    graph_neg.add_edge(1, 2, cap=2, cost=1)
    with pytest.raises(ValueError, match="初始有向弧存在负费用边"):
        solve_mcmf_ssp(graph_neg, 0, 2)

    # 2. 负容量输入：add_edge 就地抛出 ValueError
    graph_cap = MCMFGraph(2, 4)
    with pytest.raises(ValueError, match="弧容量不得为负数"):
        graph_cap.add_edge(0, 1, cap=-1, cost=5)

    # 3. 越界节点索引：add_edge 就地抛出 ValueError
    graph_bounds = MCMFGraph(3, 4)
    with pytest.raises(ValueError, match="节点索引超出范围"):
        graph_bounds.add_edge(0, 3, cap=1, cost=1)

    # 4. 零费用环路：验证算法正常收敛，不发生死循环
    graph_zero_cycle = MCMFGraph(5, 20)
    graph_zero_cycle.add_edge(0, 1, cap=1, cost=2)
    # 零费用环路 1 -> 2 -> 3 -> 1
    graph_zero_cycle.add_edge(1, 2, cap=1, cost=0)
    graph_zero_cycle.add_edge(2, 3, cap=1, cost=0)
    graph_zero_cycle.add_edge(3, 1, cap=1, cost=0)
    # 汇点连接
    graph_zero_cycle.add_edge(3, 4, cap=1, cost=3)

    flow, cost, flow_dict = solve_mcmf_ssp(graph_zero_cycle, 0, 4)
    assert flow == 1
    assert cost == 5
    assert (0, 1) in flow_dict
    assert (3, 4) in flow_dict

    # 5. 矩阵 API 维数不匹配与负数校验
    cap_mat = np.array([[0, 1], [0, 0]], dtype=np.int32)
    cost_mat = np.array([[0, -2], [0, 0]], dtype=np.int32)
    with pytest.raises(ValueError, match="初始有向弧存在负费用边"):
        solve_mcmf_successive_shortest_path(2, 0, 1, cap_mat, cost_mat)

    # 6. 残量网络规约费用异常抛出验证
    graph_corrupt = MCMFGraph(4, 20)
    graph_corrupt.add_edge(0, 1, cap=1, cost=1)
    graph_corrupt.add_edge(1, 2, cap=1, cost=1)
    graph_corrupt.add_edge(2, 3, cap=1, cost=1)
    graph_corrupt.cost[0] = -10
    with pytest.raises(ValueError, match="初始有向弧存在负费用边"):
        solve_mcmf_ssp(graph_corrupt, 0, 3)


def test_adversarial_extreme_food_sparsity_single_food():
    # 场景 4：极端稀疏食物（64x64 地图上仅有 1 颗食物，100 只蚂蚁竞争）
    rows, cols = 64, 64
    water = np.zeros((rows, cols), dtype=bool)
    food_coords = [(32, 32)]

    rng = np.random.default_rng(20261104)
    valid_coords = [(r, c) for r in range(rows) for c in range(cols) if (r, c) != (32, 32)]
    chosen = rng.choice(len(valid_coords), size=100, replace=False)
    mine = [valid_coords[idx] for idx in chosen]

    # 部分视野（带有迷雾），符合实际对抗场景
    seen = np.zeros((rows, cols), dtype=bool)
    for r, c in mine:
        for dr in range(-4, 5):
            for dc in range(-4, 5):
                seen[(r + dr) % rows, (c + dc) % cols] = True
    seen[32, 32] = True

    teacher = MCMFTeacher()
    obs = make_test_obs(rows, cols, mine, food_coords)

    orders, telemetry = teacher.orders_with_telemetry(obs, water, seen)
    assert len(orders) == 100
    assert len(telemetry) == 100

    gather_assignments = [
        i for i, rec in enumerate(telemetry) if rec.nearest_source_type in ("food_visible", "food_memory")
    ]
    # 单颗食物最多仅有 1 只蚂蚁获配采食
    assert len(gather_assignments) == 1

    # 其余 99 只蚂蚁分流转入探索前沿
    frontier_assignments = [
        i for i, rec in enumerate(telemetry) if rec.nearest_source_type == "frontier_unseen"
    ]
    assert len(frontier_assignments) == 99

    # 验证无碰撞、无踩食、无水域
    destinations = []
    for i, act in enumerate(orders):
        ar, ac = mine[i]
        if act != HOLD:
            dr, dc = DELTAS[act]
            dest = ((ar + dr) % rows, (ac + dc) % cols)
        else:
            dest = (ar, ac)
        assert not water[dest[0], dest[1]]
        assert dest != (32, 32)
        destinations.append(dest)
    assert len(set(destinations)) == 100


def test_adversarial_extreme_food_density_5x5_block():
    # 场景 5：极端密集食物（5x5 实心食物块，共计 25 颗相连食物实体）
    rows, cols = 50, 50
    water = np.zeros((rows, cols), dtype=bool)

    # 5x5 食物块位于 [20, 25) x [20, 25)
    dense_foods = [(r, c) for r in range(20, 25) for c in range(20, 25)]
    assert len(dense_foods) == 25

    gathering_positions, cluster_caps = extract_cluster_gathering_capacity(
        dense_foods, water, rows, cols
    )
    # 切比雪夫邻域合并后严格为 1 个连通簇
    assert len(cluster_caps) == 1
    # 5x5 食物外围正交采集位为 4 * 5 = 20 个合法陆地格子
    assert len(gathering_positions) == 20
    cluster_id = next(iter(cluster_caps.keys()))
    # 容量为 min(|G(R)|, |R|) = min(20, 25) = 20
    assert cluster_caps[cluster_id] == 20

    # 部署 100 只蚂蚁环绕在外部
    rng = np.random.default_rng(5555)
    food_set = set(dense_foods)
    valid_coords = [(r, c) for r in range(rows) for c in range(cols) if (r, c) not in food_set]
    chosen = rng.choice(len(valid_coords), size=100, replace=False)
    mine = [valid_coords[idx] for idx in chosen]

    seen = np.ones((rows, cols), dtype=bool)
    teacher = MCMFTeacher()

    # 连续执行 5 轮推演，验证高密度采食稳定性
    current_mine = list(mine)
    for turn in range(5):
        obs = make_test_obs(rows, cols, current_mine, dense_foods)
        orders, telemetry = teacher.orders_with_telemetry(obs, water, seen)

        assert len(orders) == 100
        gather_ants = [
            i for i, rec in enumerate(telemetry) if rec.nearest_source_type in ("food_visible", "food_memory")
        ]
        # 获配采食的蚂蚁数量严格受容量上限 20 约束
        assert len(gather_ants) <= 20

        destinations = []
        for i, act in enumerate(orders):
            ar, ac = current_mine[i]
            if act != HOLD:
                dr, dc = DELTAS[act]
                dest = ((ar + dr) % rows, (ac + dc) % cols)
            else:
                dest = (ar, ac)
            assert not water[dest[0], dest[1]]
            assert dest not in food_set
            destinations.append(dest)
        assert len(set(destinations)) == 100

        current_mine = step_ants_with_orders(orders, current_mine, rows, cols, water)


def test_adversarial_pruning_robustness_all_far_and_mixed():
    # 场景 6：候选边多样性剪枝健壮性极限检验

    # 1. 全部采集位不可达（全 FAR）：不抛异常，流量为 0
    ant_positions = [(0, 0), (1, 1)]
    food_positions = [(10, 10)]
    gathering_positions = [(9, 10), (11, 10)]
    cost_all_far = np.full((2, 2), FAR, dtype=np.int32)
    gather_to_food = {0: [0], 1: [0]}

    graph_far, offsets_far = build_capacity_constrained_bipartite_graph(
        ant_positions=ant_positions,
        food_positions=food_positions,
        gathering_positions=gathering_positions,
        geodesic_cost_matrix=cost_all_far,
        gather_to_food=gather_to_food,
        rows=20,
        cols=20,
        cluster_limit=4,
        g_per_cluster=1,
    )
    flow_far, cost_far, dict_far = solve_mcmf_ssp(graph_far, offsets_far["source"], offsets_far["sink"])
    assert flow_far == 0
    assert len(dict_far) == 0

    # 2. 混合连通：蚂蚁 0 可达，蚂蚁 1 全 FAR
    cost_mixed = np.full((2, 2), FAR, dtype=np.int32)
    cost_mixed[0, 0] = 5
    graph_mix, offsets_mix = build_capacity_constrained_bipartite_graph(
        ant_positions=ant_positions,
        food_positions=food_positions,
        gathering_positions=gathering_positions,
        geodesic_cost_matrix=cost_mixed,
        gather_to_food=gather_to_food,
        rows=20,
        cols=20,
        cluster_limit=4,
        g_per_cluster=1,
    )
    flow_mix, cost_mix, dict_mix = solve_mcmf_ssp(graph_mix, offsets_mix["source"], offsets_mix["sink"])
    assert flow_mix == 1
    assert cost_mix == 5

    # 3. 剪枝极端参数 cluster_limit=1, g_per_cluster=1
    ant_many = [(r, 0) for r in range(10)]
    food_many = [(r, 10) for r in range(15)]
    g_many = [(r, 9) for r in range(15)]
    cost_many = np.random.default_rng(42).integers(1, 20, size=(10, 15), dtype=np.int32)
    g_to_f = {i: [i] for i in range(15)}

    graph_p1, offsets_p1 = build_capacity_constrained_bipartite_graph(
        ant_positions=ant_many,
        food_positions=food_many,
        gathering_positions=g_many,
        geodesic_cost_matrix=cost_many,
        gather_to_food=g_to_f,
        rows=30,
        cols=30,
        cluster_limit=1,
        g_per_cluster=1,
    )
    flow_p1, _, _ = solve_mcmf_ssp(graph_p1, offsets_p1["source"], offsets_p1["sink"])
    assert flow_p1 > 0
    assert flow_p1 <= 10


def test_adversarial_latency_stress_80x80_map_comprehensive():
    # 场景 7：超大地图（80x80、100 蚂蚁、30 食物）单回合决策耗时严苛检验
    rows, cols = 80, 80
    num_seeds = 5
    rounds_per_seed = 10
    total_rounds = num_seeds * rounds_per_seed

    latencies_ms: list[float] = []
    seed_means: list[float] = []

    for seed in [4001, 4002, 4003, 4004, 4005]:
        rng = np.random.default_rng(seed)
        # 生成不规则水域块
        water = np.zeros((rows, cols), dtype=bool)
        for _ in range(8):
            wr = int(rng.integers(5, rows - 15))
            wc = int(rng.integers(5, cols - 15))
            h = int(rng.integers(4, 9))
            w = int(rng.integers(4, 9))
            water[wr : wr + h, wc : wc + w] = True

        seen = np.ones((rows, cols), dtype=bool)

        mine_set = set()
        while len(mine_set) < 100:
            r = int(rng.integers(0, rows))
            c = int(rng.integers(0, cols))
            if not water[r, c]:
                mine_set.add((r, c))
        mine = list(mine_set)

        food_set = set()
        while len(food_set) < 30:
            r = int(rng.integers(0, rows))
            c = int(rng.integers(0, cols))
            if not water[r, c] and (r, c) not in mine_set:
                food_set.add((r, c))
        food = list(food_set)

        teacher = MCMFTeacher()
        current_mine = list(mine)

        seed_lats: list[float] = []
        for _ in range(rounds_per_seed):
            obs = make_test_obs(rows, cols, current_mine, food)
            t0 = time.perf_counter()
            orders = teacher.orders(obs, water, seen)
            t_ms = (time.perf_counter() - t0) * 1000.0
            seed_lats.append(t_ms)
            latencies_ms.append(t_ms)

            assert len(orders) == 100
            current_mine = step_ants_with_orders(orders, current_mine, rows, cols, water)

        seed_mean = float(np.mean(seed_lats))
        seed_p95 = float(np.percentile(seed_lats, 95))
        seed_means.append(seed_mean)
        print(f"\n[80x80 种子 {seed}] 平均耗时: {seed_mean:.2f} ms | 95分位: {seed_p95:.2f} ms")

    global_mean = float(np.mean(latencies_ms))
    global_p95 = float(np.percentile(latencies_ms, 95))
    global_max = float(np.max(latencies_ms))

    print(f"\n[80x80 全量 {total_rounds} 轮综合统计] 100 蚂蚁 30 食物:")
    print(f"  全局平均耗时: {global_mean:.2f} ms")
    print(f"  全局 95分位耗时: {global_p95:.2f} ms")
    print(f"  全局最大单次耗时: {global_max:.2f} ms")

    # 门禁要求：超大地图（80x80、100 蚂蚁、30 食物）单回合决策全局平均耗时严格小于 15.0 毫秒
    assert global_mean < 15.0, f"80x80 全局平均耗时超出 15.0 ms: 实际为 {global_mean:.2f} ms"
