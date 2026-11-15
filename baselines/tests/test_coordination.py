from __future__ import annotations

import numpy as np
import pytest

from tb_baselines.coordination import (
    FAR,
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


def test_gather_capacity_single_isolated_food():
    rows, cols = 10, 10
    water = np.zeros((rows, cols), dtype=bool)
    food_coords = [(5, 5)]

    gathering_positions, cluster_caps = extract_cluster_gathering_capacity(food_coords, water, rows, cols)
    assert len(gathering_positions) == 4
    assert (4, 5) in gathering_positions
    assert (6, 5) in gathering_positions
    assert (5, 4) in gathering_positions
    assert (5, 6) in gathering_positions

    # 单体食物容量严格为 1
    total_cap = sum(cluster_caps.values())
    assert total_cap == 1

    # 部署 4 只蚂蚁围绕该食物
    ants = [(2, 5), (8, 5), (5, 2), (5, 8)]
    passable = ~water
    passable[5, 5] = False
    geodesic_fields = compute_batch_torus_geodesics(gathering_positions, passable, rows, cols)

    cost_matrix = np.zeros((len(ants), len(gathering_positions)), dtype=np.int32)
    for i, (ar, ac) in enumerate(ants):
        for k in range(len(gathering_positions)):
            cost_matrix[i, k] = geodesic_fields[k][ar, ac]

    graph, offsets = build_capacity_constrained_bipartite_graph(
        ant_positions=ants,
        food_positions=food_coords,
        gathering_positions=gathering_positions,
        geodesic_cost_matrix=cost_matrix,
        rows=rows,
        cols=cols,
    )

    total_flow, _, flow_dict = solve_mcmf_ssp(graph, offsets["source"], offsets["sink"])
    # 尽管有 4 个可通行采集位和 4 只蚂蚁，最大流受食物实体数量约束严格为 1
    assert total_flow == 1

    assignment = assign_ants_to_exclusive_gathering(flow_dict, ants, gathering_positions)
    assert len(assignment) == 1


def test_gather_capacity_corridor_chokepoint():
    rows, cols = 10, 10
    water = np.zeros((rows, cols), dtype=bool)
    # 食物位于通道死胡同 (5, 5)，三面为水域
    water[4, 5] = True
    water[6, 5] = True
    water[5, 4] = True
    # 仅 (5, 6) 为可通行陆地
    food_coords = [(5, 5)]

    gathering_positions, cluster_caps = extract_cluster_gathering_capacity(food_coords, water, rows, cols)
    assert len(gathering_positions) == 1
    assert gathering_positions[0] == (5, 6)
    assert sum(cluster_caps.values()) == 1

    # 食物完全被水域封闭场景
    water[5, 6] = True
    gathering_positions_blocked, cluster_caps_blocked = extract_cluster_gathering_capacity(food_coords, water, rows, cols)
    assert len(gathering_positions_blocked) == 0
    assert sum(cluster_caps_blocked.values()) == 0


def test_gather_capacity_dense_food_cluster():
    rows, cols = 12, 12
    water = np.zeros((rows, cols), dtype=bool)
    # 2x2 密集食物簇 (4 颗食物)
    food_coords = [(4, 4), (4, 5), (5, 4), (5, 5)]

    gathering_positions, cluster_caps = extract_cluster_gathering_capacity(food_coords, water, rows, cols)
    # 开阔地形下，2x2 食物四周环绕 8 个外围陆地单元格
    assert len(gathering_positions) == 8
    # 容量 C_j = min(|G(R_j)|, |R_j|) = min(8, 4) = 4
    assert len(cluster_caps) == 1
    assert list(cluster_caps.values())[0] == 4
    assert sum(cluster_caps.values()) == 4

    # 将外围 6 个单元格设为水域，仅留 2 个陆地采集位
    water[3, 4] = True
    water[3, 5] = True
    water[4, 3] = True
    water[4, 6] = True
    water[5, 3] = True
    water[5, 6] = True
    # 仅留 (6, 4) 与 (6, 5)
    gathering_positions_constrained, cluster_caps_constrained = extract_cluster_gathering_capacity(food_coords, water, rows, cols)
    assert len(gathering_positions_constrained) == 2
    # 物理界面容量受采集位数量瓶颈制约: min(2, 4) = 2
    assert len(cluster_caps_constrained) == 1
    assert list(cluster_caps_constrained.values())[0] == 2
    assert sum(cluster_caps_constrained.values()) == 2

    # 部署 5 只蚂蚁测试网络流
    ants = [(7, 4), (7, 5), (8, 4), (8, 5), (9, 4)]
    passable = ~water
    for fr, fc in food_coords:
        passable[fr, fc] = False
    geodesic_fields = compute_batch_torus_geodesics(gathering_positions_constrained, passable, rows, cols)

    cost_matrix = np.zeros((len(ants), len(gathering_positions_constrained)), dtype=np.int32)
    for i, (ar, ac) in enumerate(ants):
        for k in range(len(gathering_positions_constrained)):
            cost_matrix[i, k] = geodesic_fields[k][ar, ac]

    graph, offsets = build_capacity_constrained_bipartite_graph(
        ant_positions=ants,
        food_positions=food_coords,
        gathering_positions=gathering_positions_constrained,
        geodesic_cost_matrix=cost_matrix,
        rows=rows,
        cols=cols,
    )
    total_flow, _, _ = solve_mcmf_ssp(graph, offsets["source"], offsets["sink"])
    assert total_flow == 2


def test_network_construction_node_splitting():
    rows, cols = 10, 10
    # 两颗食物 (5, 4) 与 (5, 6) 共享唯一的中间陆地采集位 (5, 5)
    water = np.zeros((rows, cols), dtype=bool)
    # 阻断两颗食物的其他正交相邻格
    water[4, 4] = True
    water[6, 4] = True
    water[5, 3] = True

    water[4, 6] = True
    water[6, 6] = True
    water[5, 7] = True

    food_coords = [(5, 4), (5, 6)]
    gathering_positions, _, gather_to_food = extract_cluster_gathering_positions(food_coords, water, rows, cols)
    assert len(gathering_positions) == 1
    assert gathering_positions[0] == (5, 5)
    # 该采集位同时连接两颗食物
    assert len(gather_to_food[0]) == 2

    ants = [(0, 5), (1, 5)]
    cost_matrix = np.array([[5], [4]], dtype=np.int32)

    graph, offsets = build_capacity_constrained_bipartite_graph(
        ant_positions=ants,
        food_positions=food_coords,
        gathering_positions=gathering_positions,
        geodesic_cost_matrix=cost_matrix,
        gather_to_food=gather_to_food,
        rows=rows,
        cols=cols,
    )

    # 验证节点拆分弧 G_in -> G_out 的容量严格为 1
    g_in = offsets["g_in_offset"]
    g_out = offsets["g_out_offset"]
    found_split_arc = False
    e = graph.head[g_in]
    while e != -1:
        if graph.to[e] == g_out:
            assert graph.cap[e] == 1
            assert graph.cost[e] == 0
            found_split_arc = True
            break
        e = graph.next_edge[e]
    assert found_split_arc

    total_flow, _, flow_dict = solve_mcmf_ssp(graph, offsets["source"], offsets["sink"])
    # 即使存在两颗食物和两只蚂蚁，由于采集位节点拆分容量限制为 1，最大流为 1
    assert total_flow == 1

    assignment = assign_ants_to_exclusive_gathering(flow_dict, ants, gathering_positions)
    assert len(assignment) == 1
    # 费用更低的 Ant 1 (cost 4) 获配
    assert 1 in assignment


def test_mcmf_flow_conservation_and_bounds():
    rng = np.random.default_rng(2026)
    num_ants = 40
    num_foods = 15
    num_gather = 30

    total_nodes = 2 + num_ants + 2 * num_gather + num_foods
    source = 0
    sink = total_nodes - 1

    ants = [(int(rng.integers(0, 50)), int(rng.integers(0, 50))) for _ in range(num_ants)]
    foods = [(int(rng.integers(0, 50)), int(rng.integers(0, 50))) for _ in range(num_foods)]
    gather = [(int(rng.integers(0, 50)), int(rng.integers(0, 50))) for _ in range(num_gather)]

    cost_matrix = rng.integers(1, 40, size=(num_ants, num_gather), dtype=np.int32)
    gather_to_food: dict[int, list[int]] = {}
    for k in range(num_gather):
        gather_to_food[k] = list(rng.choice(num_foods, size=int(rng.integers(1, 3)), replace=False))

    graph, offsets = build_capacity_constrained_bipartite_graph(
        ant_positions=ants,
        food_positions=foods,
        gathering_positions=gather,
        geodesic_cost_matrix=cost_matrix,
        gather_to_food=gather_to_food,
    )

    total_flow, total_cost, flow_dict = solve_mcmf_ssp(graph, source, sink)
    assert total_flow > 0
    assert total_cost > 0

    # 1. 容量上下界断言
    for (u, v), f in flow_dict.items():
        assert f > 0

    # 2. 节点流量守恒定律断言: 对除源汇点外的所有内部节点，入流总和等于出流总和
    in_flow = [0] * total_nodes
    out_flow = [0] * total_nodes
    for (u, v), f in flow_dict.items():
        out_flow[u] += f
        in_flow[v] += f

    for node in range(1, total_nodes - 1):
        assert in_flow[node] == out_flow[node], f"节点 {node} 违背流量守恒定律"

    assert out_flow[source] == total_flow
    assert in_flow[sink] == total_flow


def test_mcmf_assignment_cost_optimality():
    # 4 只蚂蚁与 2 颗独立食物，设定非对称成本
    # Ant 0: 到 Food 0 成本 2，到 Food 1 成本 10
    # Ant 1: 到 Food 0 成本 3，到 Food 1 成本 5
    # Ant 2: 到 Food 0 成本 8，到 Food 1 成本 8
    # Ant 3: 到 Food 0 成本 15，到 Food 1 成本 12
    # 全局最优分配: Ant 0 -> Food 0 (2), Ant 1 -> Food 1 (5)，理论最小总成本为 7
    cost_matrix = np.array([
        [2, 10],
        [3, 5],
        [8, 8],
        [15, 12],
    ], dtype=np.int32)

    ants = [(0, 0), (1, 1), (2, 2), (3, 3)]
    foods = [(5, 5), (6, 6)]
    gather = [(5, 4), (6, 5)]
    gather_to_food = {0: [0], 1: [1]}

    graph, offsets = build_capacity_constrained_bipartite_graph(
        ant_positions=ants,
        food_positions=foods,
        gathering_positions=gather,
        geodesic_cost_matrix=cost_matrix,
        gather_to_food=gather_to_food,
    )

    total_flow, total_cost, flow_dict = solve_mcmf_ssp(graph, offsets["source"], offsets["sink"])
    assert total_flow == 2
    assert total_cost == 7

    assignment = assign_ants_to_exclusive_gathering(flow_dict, ants, gather)
    assert assignment[0] == 0
    assert assignment[1] == 1


def test_mcmf_disconnected_unreachable_graceful():
    rows, cols = 20, 20
    water = np.zeros((rows, cols), dtype=bool)
    # 食物位于由完整水环包围的孤立小岛上
    for r in range(4, 9):
        water[r, 4] = True
        water[r, 8] = True
    for c in range(4, 9):
        water[4, c] = True
        water[8, c] = True

    food_coords = [(6, 6)]
    passable = ~water
    passable[6, 6] = False

    gathering_positions, _, gather_to_food = extract_cluster_gathering_positions(food_coords, water, rows, cols)
    assert len(gathering_positions) == 4

    # 蚂蚁位于水环外部的主大陆 (0, 0)
    ants = [(0, 0)]
    geodesic_fields = compute_batch_torus_geodesics(gathering_positions, passable, rows, cols)
    cost_matrix = np.full((1, len(gathering_positions)), FAR, dtype=np.int32)
    for k in range(len(gathering_positions)):
        cost_matrix[0, k] = geodesic_fields[k][0, 0]
        # 验证距离场严格为 FAR 哨兵值
        assert cost_matrix[0, k] == FAR

    graph, offsets = build_capacity_constrained_bipartite_graph(
        ant_positions=ants,
        food_positions=food_coords,
        gathering_positions=gathering_positions,
        geodesic_cost_matrix=cost_matrix,
        gather_to_food=gather_to_food,
        rows=rows,
        cols=cols,
    )

    # 验证算法在不可达图中安全终止，返回 0 流量与 0 成本
    total_flow, total_cost, flow_dict = solve_mcmf_ssp(graph, offsets["source"], offsets["sink"])
    assert total_flow == 0
    assert total_cost == 0
    assert len(flow_dict) == 0


def test_mcmf_negative_cycle_fast_fail():
    # 1. 负容量输入断言
    cap_mat = np.array([[0, -1], [0, 0]], dtype=np.int32)
    cost_mat = np.array([[0, 1], [0, 0]], dtype=np.int32)
    with pytest.raises(ValueError, match="容量矩阵中存在负数容量"):
        solve_mcmf_successive_shortest_path(2, 0, 1, cap_mat, cost_mat)

    # 2. 矩阵维度不匹配断言
    cap_mat_valid = np.zeros((3, 3), dtype=np.int32)
    cost_mat_invalid = np.zeros((3, 2), dtype=np.int32)
    with pytest.raises(ValueError, match="费用矩阵维度不匹配"):
        solve_mcmf_successive_shortest_path(3, 0, 2, cap_mat_valid, cost_mat_invalid)

    # 3. 初始正容量弧负费用快速失败断言
    graph = MCMFGraph(3, 10)
    graph.add_edge(0, 1, 1, -5)
    graph.add_edge(1, 2, 1, 2)
    with pytest.raises(ValueError, match="初始有向弧存在负费用边"):
        solve_mcmf_ssp(graph, 0, 2)

    # 4. 节点添加负容量快速失败断言
    graph2 = MCMFGraph(2, 10)
    with pytest.raises(ValueError, match="弧容量不得为负数"):
        graph2.add_edge(0, 1, -1, 5)


def test_compute_single_source_torus_geodesic_boundary():
    rows, cols = 8, 8
    passable = np.ones((rows, cols), dtype=bool)
    # 单源位于 (0, 0)，测试环面环绕距离
    dist = compute_single_source_torus_geodesic((0, 0), passable, rows, cols)
    assert dist[0, 0] == 0
    # 环面正交相邻格距离为 1
    assert dist[7, 0] == 1
    assert dist[1, 0] == 1
    assert dist[0, 7] == 1
    assert dist[0, 1] == 1
    # 最远角点 (4, 4) 曼哈顿距离为 4 + 4 = 8
    assert dist[4, 4] == 8


def test_gather_capacity_chebyshev_adjacent_pairs():
    rows, cols = 10, 10
    water = np.zeros((rows, cols), dtype=bool)

    # 1. 正交相邻食物实体
    food_orthogonal = [(5, 5), (5, 6)]
    pos_ortho, caps_ortho = extract_cluster_gathering_capacity(food_orthogonal, water, rows, cols)
    assert len(pos_ortho) == 6
    assert len(caps_ortho) == 1
    assert list(caps_ortho.values()) == [2]

    # 2. 对角相邻且中间阻隔水域的食物实体
    water[5, 6] = True
    water[6, 5] = True
    food_diagonal = [(5, 5), (6, 6)]
    pos_diag, caps_diag = extract_cluster_gathering_capacity(food_diagonal, water, rows, cols)
    assert len(pos_diag) == 4
    assert len(caps_diag) == 1
    assert list(caps_diag.values()) == [2]


def test_gather_capacity_torus_wrap_clustering():
    rows, cols = 10, 10
    water = np.zeros((rows, cols), dtype=bool)

    # 1. 环面对角四角跨界合并 (0, 0) 与 (9, 9)
    food_corner = [(0, 0), (rows - 1, cols - 1)]
    _, caps_corner = extract_cluster_gathering_capacity(food_corner, water, rows, cols)
    assert len(caps_corner) == 1
    assert list(caps_corner.values()) == [2]

    # 2. 环面垂直边缘跨界合并 (0, 5) 与 (9, 5)
    food_edge = [(0, 5), (rows - 1, 5)]
    _, caps_edge = extract_cluster_gathering_capacity(food_edge, water, rows, cols)
    assert len(caps_edge) == 1
    assert list(caps_edge.values()) == [2]


def test_gather_capacity_multiple_disjoint_clusters():
    rows, cols = 12, 12
    water = np.zeros((rows, cols), dtype=bool)
    # 两组空间相互隔离的独立食物簇
    food_coords = [(2, 2), (2, 3), (8, 8), (8, 9)]
    positions, caps = extract_cluster_gathering_capacity(food_coords, water, rows, cols)
    assert len(positions) == 12
    assert len(caps) == 2
    assert sorted(caps.values()) == [2, 2]
    assert sum(caps.values()) == 4


def test_gather_capacity_linear_chain_clustering():
    rows, cols = 12, 12
    water = np.zeros((rows, cols), dtype=bool)
    # 1x4 连续直线食物列
    food_coords = [(5, 3), (5, 4), (5, 5), (5, 6)]
    positions, caps = extract_cluster_gathering_capacity(food_coords, water, rows, cols)
    assert len(positions) == 10
    assert len(caps) == 1
    assert list(caps.values())[0] == 4


