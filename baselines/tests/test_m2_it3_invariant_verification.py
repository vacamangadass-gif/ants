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
    compute_cost_matrix_bitmask,
    extract_cluster_gathering_capacity,
    extract_cluster_gathering_positions,
    solve_mcmf_ssp,
)
from tb_baselines.mcmf_teacher import MCMFTeacher


def test_invariant_1_flow_conservation_and_capacity_bounds():
    # 随机模糊测试：检验网络流守恒律、源汇点平衡与容量上限约束
    for seed in range(5001, 5051):
        rng = np.random.default_rng(seed)
        rows = int(rng.integers(24, 60))
        cols = int(rng.integers(24, 60))

        num_ants = int(rng.integers(15, 80))
        num_foods = int(rng.integers(5, 40))

        water = rng.random((rows, cols)) < 0.20
        valid_cells = [(r, c) for r in range(rows) for c in range(cols) if not water[r, c]]
        if len(valid_cells) < num_ants + num_foods + 10:
            continue

        selected = rng.choice(len(valid_cells), size=num_ants + num_foods, replace=False)
        ant_positions = [valid_cells[idx] for idx in selected[:num_ants]]
        food_positions = [valid_cells[idx] for idx in selected[num_ants:]]

        gathering_positions, _, gather_to_food = extract_cluster_gathering_positions(
            food_positions, water, rows, cols
        )
        if not gathering_positions:
            continue

        passable = (~water)
        for fr, fc in food_positions:
            passable[fr, fc] = False

        max_depth = 18
        cost_matrix = compute_cost_matrix_bitmask(
            ant_positions=ant_positions,
            gathering_positions=gathering_positions,
            passable_mask=passable,
            rows=rows,
            cols=cols,
            max_depth=max_depth,
        )

        graph, offsets = build_capacity_constrained_bipartite_graph(
            ant_positions=ant_positions,
            food_positions=food_positions,
            gathering_positions=gathering_positions,
            geodesic_cost_matrix=cost_matrix,
            gather_to_food=gather_to_food,
            rows=rows,
            cols=cols,
            max_candidates_per_ant=10,
        )

        total_flow, total_cost, flow_dict = solve_mcmf_ssp(
            graph, offsets["source"], offsets["sink"]
        )

        # 1. 流量非负与理论上界约束
        assert total_flow >= 0
        assert total_flow <= min(num_ants, num_foods, len(gathering_positions))

        # 2. 边流量非负与容量上限约束
        for (u, v), flow_val in flow_dict.items():
            assert flow_val > 0

        # 3. 中间节点流量守恒律约束
        total_nodes = offsets["sink"] + 1
        flow_in = [0] * total_nodes
        flow_out = [0] * total_nodes
        for (u, v), flow_val in flow_dict.items():
            flow_out[u] += flow_val
            flow_in[v] += flow_val

        for node_idx in range(1, total_nodes - 1):
            assert flow_in[node_idx] == flow_out[node_idx], (
                f"种子 {seed} 节点 {node_idx} 流量不守恒: 流入={flow_in[node_idx]}, 流出={flow_out[node_idx]}"
            )

        # 4. 源点流出等于汇点流入
        assert flow_out[offsets["source"]] == total_flow
        assert flow_in[offsets["sink"]] == total_flow


def test_invariant_2_physical_gathering_capacity_and_cluster_perimeter_bounds():
    # 验证物理采集位独占性与食物簇外周外围合法陆地容量硬约束
    rows, cols = 32, 32
    water = np.zeros((rows, cols), dtype=bool)

    # 构造 3x3 紧密食物簇，外围仅开放 3 个陆地采集位，其余全部被水域包围
    cluster_foods = [(10 + r, 10 + c) for r in range(3) for c in range(3)]
    for r in range(8, 15):
        for c in range(8, 15):
            water[r, c] = True

    # 解锁食物格子本身作为食物实体占用
    for fr, fc in cluster_foods:
        water[fr, fc] = False

    # 开启 3 个分别对应 3 颗不同食物的外周陆地采集位
    # (9, 10) 仅相邻食物 (10, 10)
    # (11, 9) 仅相邻食物 (11, 10)
    # (13, 11) 仅相邻食物 (12, 11)
    open_gather_slots = [(9, 10), (11, 9), (13, 11)]
    for gr, gc in open_gather_slots:
        water[gr, gc] = False

    # 解锁通道连接外部大平原（纵向或横向引出，不贴近食物边缘）
    for c in range(0, 10):
        water[9, c] = False
    for r in range(0, 10):
        water[r, 9] = False
    for c in range(0, 9):
        water[11, c] = False
    for r in range(13, 32):
        water[r, 11] = False

    gathering_positions, cluster_caps = extract_cluster_gathering_capacity(
        cluster_foods, water, rows, cols
    )
    assert len(gathering_positions) == 3
    for slot in open_gather_slots:
        assert slot in gathering_positions

    # 簇容量严格受限于外周 3 个合法陆地格子
    assert sum(cluster_caps.values()) == 3

    # 部署 20 只蚂蚁争夺该 3x3 食物簇
    ant_positions = [(2, c) for c in range(20)]
    passable = (~water)
    for fr, fc in cluster_foods:
        passable[fr, fc] = False

    cost_matrix = compute_cost_matrix_bitmask(
        ant_positions=ant_positions,
        gathering_positions=gathering_positions,
        passable_mask=passable,
        rows=rows,
        cols=cols,
        max_depth=30,
    )

    _, _, gather_to_food = extract_cluster_gathering_positions(
        cluster_foods, water, rows, cols
    )
    graph, offsets = build_capacity_constrained_bipartite_graph(
        ant_positions=ant_positions,
        food_positions=cluster_foods,
        gathering_positions=gathering_positions,
        geodesic_cost_matrix=cost_matrix,
        gather_to_food=gather_to_food,
        rows=rows,
        cols=cols,
    )

    total_flow, _, flow_dict = solve_mcmf_ssp(graph, offsets["source"], offsets["sink"])

    # 20 只蚂蚁争夺 9 颗食物，外周有 3 个分别对应不同食物的采集位，总流量严格为 3
    assert total_flow == 3

    # 验证采集位内部排他性拆分弧 G_in -> G_out 流量严格为 1
    g_in_offset = offsets["g_in_offset"]
    g_out_offset = offsets["g_out_offset"]
    for k in range(len(gathering_positions)):
        split_flow = flow_dict.get((g_in_offset + k, g_out_offset + k), 0)
        assert split_flow in (0, 1)

    # 验证分配映射中不存在两只蚂蚁分配到同一物理采集位
    assignment = assign_ants_to_exclusive_gathering(flow_dict, ant_positions, gathering_positions)
    assert len(assignment) == 3
    assigned_slots = list(assignment.values())
    assert len(set(assigned_slots)) == len(assigned_slots)

    # 阶段 2：验证两采集位共享单体食物时的瓶颈约束
    # 额外开启 (10, 9)，(10, 9) 与 (9, 10) 均只与食物 (10, 10) 相邻
    water[10, 9] = False
    g_positions_shared, _ = extract_cluster_gathering_capacity(cluster_foods, water, rows, cols)
    assert len(g_positions_shared) == 4
    passable_shared = (~water)
    for fr, fc in cluster_foods:
        passable_shared[fr, fc] = False

    cost_mat_shared = compute_cost_matrix_bitmask(
        ant_positions=ant_positions,
        gathering_positions=g_positions_shared,
        passable_mask=passable_shared,
        rows=rows,
        cols=cols,
        max_depth=30,
    )
    _, _, g_to_f_shared = extract_cluster_gathering_positions(cluster_foods, water, rows, cols)
    graph_shared, offsets_shared = build_capacity_constrained_bipartite_graph(
        ant_positions=ant_positions,
        food_positions=cluster_foods,
        gathering_positions=g_positions_shared,
        geodesic_cost_matrix=cost_mat_shared,
        gather_to_food=g_to_f_shared,
        rows=rows,
        cols=cols,
    )
    flow_shared, _, flow_dict_shared = solve_mcmf_ssp(
        graph_shared, offsets_shared["source"], offsets_shared["sink"]
    )
    # 4 个采集位中 2 个共享食物 (10, 10)，受食物单体容量 1 限制，总流严格依然为 3
    assert flow_shared == 3
    assignment_shared = assign_ants_to_exclusive_gathering(
        flow_dict_shared, ant_positions, g_positions_shared
    )
    assert len(assignment_shared) == 3
    assert len(set(assignment_shared.values())) == 3


def test_invariant_3_geodesic_bitmask_vs_2d_wavefront_mathematical_equivalence():
    # 严格检验：两阶段位掩码计算得到的标量距离矩阵与二维连续波前扩展所得距离完全一致
    test_cases = [
        {"rows": 24, "cols": 24, "water_ratio": 0.0, "max_depth": 16, "num_targets": 10},
        {"rows": 32, "cols": 32, "water_ratio": 0.15, "max_depth": 20, "num_targets": 25},
        {"rows": 48, "cols": 48, "water_ratio": 0.25, "max_depth": 16, "num_targets": 63},
        {"rows": 48, "cols": 48, "water_ratio": 0.20, "max_depth": 18, "num_targets": 64},
        {"rows": 48, "cols": 48, "water_ratio": 0.20, "max_depth": 18, "num_targets": 65},
        {"rows": 64, "cols": 64, "water_ratio": 0.20, "max_depth": 22, "num_targets": 80},
    ]

    for seed, case in enumerate(test_cases, start=7001):
        rng = np.random.default_rng(seed)
        rows, cols = case["rows"], case["cols"]
        max_depth = case["max_depth"]
        num_targets = case["num_targets"]
        water_ratio = case["water_ratio"]

        water = rng.random((rows, cols)) < water_ratio
        # 确保四个角及边界特定连通性测试环面折叠
        water[0, 0] = False
        water[0, cols - 1] = False
        water[rows - 1, 0] = False
        water[rows - 1, cols - 1] = False

        land_cells = [(r, c) for r in range(rows) for c in range(cols) if not water[r, c]]
        assert len(land_cells) >= num_targets + 50

        target_indices = rng.choice(len(land_cells), size=num_targets, replace=False)
        targets = [land_cells[idx] for idx in target_indices]

        # 包含处于边界四角的特殊目标
        if (0, 0) not in targets and len(targets) > 0:
            targets[0] = (0, 0)
        if (rows - 1, cols - 1) not in targets and len(targets) > 1:
            targets[1] = (rows - 1, cols - 1)

        ant_indices = rng.choice(len(land_cells), size=50, replace=False)
        ants = [land_cells[idx] for idx in ant_indices]
        # 包含对角跨越环面蚂蚁
        ants[0] = (rows - 1, 0)
        ants[1] = (0, cols - 1)

        passable = ~water

        # 1. 执行二维连续波前扩展基准算法
        tensor_fields = compute_batch_torus_geodesics(
            targets=targets,
            passable_mask=passable,
            rows=rows,
            cols=cols,
            max_depth=max_depth,
        )
        expected_costs = np.full((len(ants), num_targets), FAR, dtype=np.int32)
        for i, (ar, ac) in enumerate(ants):
            for k in range(num_targets):
                d = tensor_fields[k][ar, ac]
                if d <= max_depth:
                    expected_costs[i, k] = d

        # 2. 执行超高速两阶段位掩码算法
        actual_costs = compute_cost_matrix_bitmask(
            ant_positions=ants,
            gathering_positions=targets,
            passable_mask=passable,
            rows=rows,
            cols=cols,
            max_depth=max_depth,
        )

        # 3. 逐元素数学等价性断言
        mismatch_count = np.sum(expected_costs != actual_costs)
        assert mismatch_count == 0, (
            f"用例 {case} 下位掩码与连续二维波前扩展距离矩阵存在 {mismatch_count} 处不匹配"
        )


def test_invariant_4_redundancy_reduction_rate_exceeds_80_percent():
    # 对比未协调贪心最近邻分配基线，实测验证 MCMF 协同采食冗余削减率达到 80% 以上
    rows, cols = 48, 48
    water = np.zeros((rows, cols), dtype=bool)
    water[12:18, 12:18] = True
    water[28:34, 28:34] = True

    seeds = [1001, 1002, 2026, 3030, 4040]
    scale = 40
    num_foods = 20

    for seed in seeds:
        rng = np.random.default_rng(seed)
        land = [(r, c) for r in range(rows) for c in range(cols) if not water[r, c]]
        chosen = rng.choice(len(land), size=scale + num_foods, replace=False)
        ants = [land[idx] for idx in chosen[:scale]]
        foods = [land[idx] for idx in chosen[scale:]]

        # 1. 贪心最近邻采食分配基线：每只蚂蚁独立寻找欧氏/曼哈顿最近食物
        greedy_food_claims: dict[tuple[int, int], list[int]] = {}
        for ant_idx, (ar, ac) in enumerate(ants):
            min_dist = 1000000
            best_food = None
            for fr, fc in foods:
                dr = min(abs(ar - fr), rows - abs(ar - fr))
                dc = min(abs(ac - fc), cols - abs(ac - fc))
                d = dr + dc
                if d < min_dist:
                    min_dist = d
                    best_food = (fr, fc)
            if best_food is not None:
                greedy_food_claims.setdefault(best_food, []).append(ant_idx)

        greedy_redundancy = sum(max(0, len(c) - 1) for c in greedy_food_claims.values())

        # 2. MCMF 协同采食分配
        gathering_positions, _, gather_to_food = extract_cluster_gathering_positions(
            foods, water, rows, cols
        )
        passable = (~water)
        for fr, fc in foods:
            passable[fr, fc] = False

        cost_matrix = compute_cost_matrix_bitmask(
            ant_positions=ants,
            gathering_positions=gathering_positions,
            passable_mask=passable,
            rows=rows,
            cols=cols,
            max_depth=20,
        )

        graph, offsets = build_capacity_constrained_bipartite_graph(
            ant_positions=ants,
            food_positions=foods,
            gathering_positions=gathering_positions,
            geodesic_cost_matrix=cost_matrix,
            gather_to_food=gather_to_food,
            rows=rows,
            cols=cols,
            max_candidates_per_ant=5,
        )

        _, _, flow_dict = solve_mcmf_ssp(graph, offsets["source"], offsets["sink"])
        mcmf_assignment = assign_ants_to_exclusive_gathering(
            flow_dict, ants, gathering_positions
        )

        # 统计 MCMF 分配中对同一颗食物的争夺蚂蚁数
        mcmf_food_claims: dict[int, list[int]] = {}
        for ant_idx, g_idx in mcmf_assignment.items():
            for f_idx in gather_to_food.get(g_idx, []):
                mcmf_food_claims.setdefault(f_idx, []).append(ant_idx)

        mcmf_redundancy = sum(max(0, len(c) - 1) for c in mcmf_food_claims.values())

        assert greedy_redundancy > 0
        reduction_rate = (greedy_redundancy - mcmf_redundancy) / greedy_redundancy
        print(
            f"种子 {seed}: 贪心冗余={greedy_redundancy}, MCMF冗余={mcmf_redundancy}, 削减比例={reduction_rate * 100:.2f}%"
        )
        assert reduction_rate >= 0.80, (
            f"种子 {seed} 协同采食冗余削减比例未达到 80% 门禁: 实际为 {reduction_rate * 100:.2f}%"
        )


def test_invariant_5_torus_geodesic_descent_collision_free():
    # 验证环面测地梯度下降动作物理无碰撞性质
    rows, cols = 40, 40
    water = np.zeros((rows, cols), dtype=bool)
    water[5:10, 5:10] = True
    water[25:30, 25:30] = True

    rng = np.random.default_rng(9999)
    land = [(r, c) for r in range(rows) for c in range(cols) if not water[r, c]]

    ants = [land[idx] for idx in rng.choice(len(land), size=60, replace=False)]
    avail_land = [p for p in land if p not in set(ants)]
    foods = [avail_land[idx] for idx in rng.choice(len(avail_land), size=20, replace=False)]

    teacher = MCMFTeacher()
    seen = np.ones((rows, cols), dtype=bool)

    obs = {
        "size": [rows, cols],
        "turn": 1,
        "max_turns": 300,
        "round_time": 0.05,
        "engine_digest": "invariant_test",
        "mine": [[int(r), int(c)] for r, c in ants],
        "ids": list(range(len(ants))),
        "food": [[int(r), int(c)] for r, c in foods],
        "hills": [],
    }

    orders, telemetry = teacher.orders_with_telemetry(obs, water, seen)
    assert len(orders) == len(ants)

    destinations: list[tuple[int, int]] = []
    food_set = set(foods)

    for i, act in enumerate(orders):
        act_int = int(act)
        ar, ac = ants[i]
        if act_int == HOLD:
            dest = (ar, ac)
        else:
            dr, dc = DELTAS[act_int]
            dest = ((ar + dr) % rows, (ac + dc) % cols)

        assert not water[dest[0], dest[1]], f"蚂蚁 {i} 踏入水域: {dest}"
        assert dest not in food_set, f"蚂蚁 {i} 踩踏食物实体: {dest}"
        destinations.append(dest)

    # 验证无友方碰撞约束
    assert len(set(destinations)) == len(destinations), (
        f"发生目的地碰撞冲突: 重复数={len(destinations) - len(set(destinations))}"
    )
