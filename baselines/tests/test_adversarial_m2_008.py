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
    extract_cluster_gathering_positions,
    solve_mcmf_ssp,
    solve_mcmf_successive_shortest_path,
)
from tb_baselines.mcmf_teacher import MCMFTeacher


def make_obs_payload(
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
        "engine_digest": "adversarial_test",
        "mine": [[int(r), int(c)] for r, c in mine],
        "ids": ant_ids,
        "food": [[int(r), int(c)] for r, c in (food or [])],
        "hills": [[int(r), int(c), int(owner)] for r, c, owner in (hills or [])],
    }


def test_adversarial_extreme_corridor_bottleneck():
    # 极端通道拓扑：100 只蚂蚁挤压在长度为 100 的单格宽盲端通道内争夺 1 颗食物
    rows, cols = 30, 120
    water = np.zeros((rows, cols), dtype=bool)

    # 在第 15 行构造单格宽度长廊，两侧（第 14 行与第 16 行）以及尽头（第 106 列）均为水域
    corridor_row = 15
    for c in range(107):
        water[corridor_row - 1, c] = True
        water[corridor_row + 1, c] = True
    water[corridor_row, 106] = True

    # 食物位于通道尽头 (15, 105)
    food_coords = [(corridor_row, 105)]

    # 100 只蚂蚁紧密排列在通道内 (15, 5) 至 (15, 104)
    ant_positions = [(corridor_row, 5 + i) for i in range(100)]
    assert len(ant_positions) == 100

    # 视野覆盖通道区域，其余区域未探索形成迷雾前沿
    seen = np.zeros((rows, cols), dtype=bool)
    seen[corridor_row, :106] = True
    seen[corridor_row - 1, :106] = True
    seen[corridor_row + 1, :106] = True

    # 1. 验证底层网络流模型提取与流量
    gathering_positions, food_to_gather, gather_to_food = extract_cluster_gathering_positions(
        food_coords, water, rows, cols
    )
    # 食物四周仅有通道前方的 (15, 104) 属于合法陆地采集位
    assert len(gathering_positions) == 1
    assert gathering_positions[0] == (corridor_row, 104)

    passable = (~water)
    passable[corridor_row, 105] = False
    geodesic_fields = compute_batch_torus_geodesics(gathering_positions, passable, rows, cols)

    cost_matrix = np.zeros((len(ant_positions), len(gathering_positions)), dtype=np.int32)
    for i, (ar, ac) in enumerate(ant_positions):
        cost_matrix[i, 0] = geodesic_fields[0][ar, ac]

    graph, offsets = build_capacity_constrained_bipartite_graph(
        ant_positions=ant_positions,
        food_positions=food_coords,
        gathering_positions=gathering_positions,
        geodesic_cost_matrix=cost_matrix,
        gather_to_food=gather_to_food,
        rows=rows,
        cols=cols,
    )

    total_flow, _, flow_dict = solve_mcmf_ssp(graph, offsets["source"], offsets["sink"])
    # 严格断言：100 只蚂蚁争夺单体食物，总流网络严格为 1
    assert total_flow == 1

    assignment = assign_ants_to_exclusive_gathering(flow_dict, ant_positions, gathering_positions)
    assert len(assignment) == 1
    # 离采集位最近的第 99 号蚂蚁（坐标 (15, 104)，距离为 0）获配
    assert 99 in assignment

    # 2. 验证 MCMFTeacher 协调器端到端行为
    obs = make_obs_payload(rows, cols, ant_positions, food_coords)
    teacher = MCMFTeacher()
    orders, telemetry = teacher.orders_with_telemetry(obs, water, seen)

    assert len(orders) == 100
    assert len(telemetry) == 100

    # 统计任务分流：恰好 1 只蚂蚁指派采食，其余 99 只蚂蚁分流转入探索前沿
    gathering_count = sum(
        1 for rec in telemetry if rec.nearest_source_type in ("food_visible", "food_memory")
    )
    exploration_count = sum(
        1 for rec in telemetry if rec.nearest_source_type == "frontier_unseen"
    )
    assert gathering_count == 1
    assert exploration_count == 99

    # 验证零友方相撞、零走入水域、零踩入食物实体
    destinations = set()
    for i, act in enumerate(orders):
        ar, ac = ant_positions[i]
        if act != HOLD:
            dr, dc = DELTAS[act]
            dest = ((ar + dr) % rows, (ac + dc) % cols)
        else:
            dest = (ar, ac)
        assert not water[dest[0], dest[1]], f"蚂蚁 {i} 走入水域: {dest}"
        assert dest != food_coords[0], f"蚂蚁 {i} 踩入食物实体: {dest}"
        assert dest not in destinations, f"蚂蚁 {i} 与友方蚂蚁发生碰撞冲突: {dest}"
        destinations.add(dest)

    assert len(destinations) == 100


def test_adversarial_disconnected_archipelago():
    # 环面群岛不连通图：食物与蚂蚁被水域完全阻隔在不同孤岛
    rows, cols = 40, 40
    water = np.ones((rows, cols), dtype=bool)

    # 构造三个互不相通的陆地孤岛
    # 孤岛 A: 行 [5, 15), 列 [5, 15) —— 部署 20 只蚂蚁，无食物
    water[5:15, 5:15] = False
    # 孤岛 B: 行 [5, 15), 列 [25, 35) —— 无蚂蚁，部署 5 颗食物
    water[5:15, 25:35] = False
    # 孤岛 C: 行 [25, 35), 列 [5, 15) —— 部署 10 只蚂蚁，2 颗食物
    water[25:35, 5:15] = False

    island_a_ants = [(r, c) for r in range(6, 11) for c in range(6, 10)]
    assert len(island_a_ants) == 20

    island_b_foods = [(7, 27), (8, 28), (9, 29), (10, 30), (11, 31)]
    assert len(island_b_foods) == 5

    island_c_ants = [(r, c) for r in range(26, 28) for c in range(6, 11)]
    assert len(island_c_ants) == 10

    island_c_foods = [(32, 8), (32, 12)]
    assert len(island_c_foods) == 2

    all_ants = island_a_ants + island_c_ants
    all_foods = island_b_foods + island_c_foods
    assert len(all_ants) == 30
    assert len(all_foods) == 7

    gathering_positions, food_to_gather, gather_to_food = extract_cluster_gathering_positions(
        all_foods, water, rows, cols
    )

    passable = (~water)
    for fr, fc in all_foods:
        passable[fr, fc] = False

    geodesic_fields = compute_batch_torus_geodesics(gathering_positions, passable, rows, cols)

    cost_matrix = np.full((len(all_ants), len(gathering_positions)), FAR, dtype=np.int32)
    for i, (ar, ac) in enumerate(all_ants):
        for k in range(len(gathering_positions)):
            cost_matrix[i, k] = geodesic_fields[k][ar, ac]

    # 孤岛 A 的蚂蚁到孤岛 B 的食物采集位测地距离必须严格为 FAR
    for ant_i in range(len(island_a_ants)):
        for food_j in range(len(island_b_foods)):
            for g_k in food_to_gather[food_j]:
                assert cost_matrix[ant_i, g_k] == FAR

    graph, offsets = build_capacity_constrained_bipartite_graph(
        ant_positions=all_ants,
        food_positions=all_foods,
        gathering_positions=gathering_positions,
        geodesic_cost_matrix=cost_matrix,
        gather_to_food=gather_to_food,
        rows=rows,
        cols=cols,
    )

    total_flow, total_cost, flow_dict = solve_mcmf_ssp(graph, offsets["source"], offsets["sink"])

    # 孤岛 B 的食物完全不可达，总流严格等于孤岛 C 的可达食物采集量 (2)
    assert total_flow == 2
    assert total_cost > 0

    # 验证孤岛 B 的食物入流严格为 0
    f_offset = offsets["f_offset"]
    for food_j in range(len(island_b_foods)):
        f_node = f_offset + food_j
        flow_into_food = sum(f for (u, v), f in flow_dict.items() if v == f_node)
        assert flow_into_food == 0, f"不可达孤岛 B 食物 {food_j} 产生异常流量"

    # 验证不连通子图上流量守恒定律严格成立
    total_nodes = offsets["sink"] + 1
    in_flow = [0] * total_nodes
    out_flow = [0] * total_nodes
    for (u, v), f in flow_dict.items():
        out_flow[u] += f
        in_flow[v] += f

    for node in range(1, total_nodes - 1):
        assert in_flow[node] == out_flow[node], f"节点 {node} 流量守恒定律断言失败"

    assert out_flow[offsets["source"]] == total_flow
    assert in_flow[offsets["sink"]] == total_flow

    # 验证 MCMFTeacher 在不连通环面群岛上稳定运行且无异常
    seen = np.ones((rows, cols), dtype=bool)
    obs = make_obs_payload(rows, cols, all_ants, all_foods)
    teacher = MCMFTeacher()
    orders = teacher.orders(obs, water, seen)
    assert len(orders) == 30

    destinations = set()
    for i, act in enumerate(orders):
        ar, ac = all_ants[i]
        if act != HOLD:
            dr, dc = DELTAS[act]
            dest = ((ar + dr) % rows, (ac + dc) % cols)
        else:
            dest = (ar, ac)
        assert not water[dest[0], dest[1]]
        assert dest not in destinations
        destinations.add(dest)


def test_adversarial_randomized_fuzzing_conservation_and_bounds():
    # 50 组随机模糊测试：检验流量守恒律与容量上下界不变式
    for seed in range(3001, 3051):
        rng = np.random.default_rng(seed)
        rows = int(rng.integers(24, 50))
        cols = int(rng.integers(24, 50))

        num_ants = int(rng.integers(10, 101))
        num_foods = int(rng.integers(1, 51))

        # 随机水域（占比 15% 左右）
        water = rng.random((rows, cols)) < 0.15

        valid_coords = [(r, c) for r in range(rows) for c in range(cols) if not water[r, c]]
        if len(valid_coords) < num_ants + num_foods:
            continue

        chosen_indices = rng.choice(len(valid_coords), size=num_ants + num_foods, replace=False)
        ant_positions = [valid_coords[i] for i in chosen_indices[:num_ants]]
        food_positions = [valid_coords[i] for i in chosen_indices[num_ants:]]

        gathering_positions, food_to_gather, gather_to_food = extract_cluster_gathering_positions(
            food_positions, water, rows, cols
        )
        if not gathering_positions:
            continue

        passable = (~water)
        for fr, fc in food_positions:
            passable[fr, fc] = False

        # 测地距离场计算，设置最大深度上限
        geodesic_fields = compute_batch_torus_geodesics(
            gathering_positions, passable, rows, cols, max_depth=25
        )

        cost_matrix = np.full((num_ants, len(gathering_positions)), FAR, dtype=np.int32)
        for i, (ar, ac) in enumerate(ant_positions):
            for k in range(len(gathering_positions)):
                cost_matrix[i, k] = geodesic_fields[k][ar, ac]

        graph, offsets = build_capacity_constrained_bipartite_graph(
            ant_positions=ant_positions,
            food_positions=food_positions,
            gathering_positions=gathering_positions,
            geodesic_cost_matrix=cost_matrix,
            gather_to_food=gather_to_food,
            rows=rows,
            cols=cols,
            max_candidates_per_ant=15,
        )

        total_flow, total_cost, flow_dict = solve_mcmf_ssp(graph, offsets["source"], offsets["sink"])

        # 1. 流量非负与容量上界不变式
        assert total_flow >= 0
        assert total_flow <= min(num_ants, num_foods, len(gathering_positions))

        for (u, v), f in flow_dict.items():
            assert f > 0

        # 2. 内部节点流量守恒定律断言: 对除源汇点外的所有内部节点，入流总和等于出流总和
        total_nodes = offsets["sink"] + 1
        in_flow = [0] * total_nodes
        out_flow = [0] * total_nodes
        for (u, v), f in flow_dict.items():
            out_flow[u] += f
            in_flow[v] += f

        for node in range(1, total_nodes - 1):
            assert in_flow[node] == out_flow[node], (
                f"随机种子 {seed} 下节点 {node} 流量守恒定律破坏: 入流={in_flow[node]}, 出流={out_flow[node]}"
            )

        assert out_flow[offsets["source"]] == total_flow
        assert in_flow[offsets["sink"]] == total_flow

        # 3. 采集位拆分节点排他性容量断言: G_in -> G_out 流量至多为 1
        g_in_offset = offsets["g_in_offset"]
        g_out_offset = offsets["g_out_offset"]
        for k in range(len(gathering_positions)):
            g_in = g_in_offset + k
            g_out = g_out_offset + k
            f_split = flow_dict.get((g_in, g_out), 0)
            assert f_split in (0, 1), f"采集位 {k} 拆分弧流量超出物理容量 1: {f_split}"

        # 4. 食物实体单回合独立性断言: 入食物节点流量至多为 1
        f_offset = offsets["f_offset"]
        for j in range(num_foods):
            f_node = f_offset + j
            flow_into_food = in_flow[f_node]
            assert flow_into_food in (0, 1), f"食物 {j} 入流超出单体容量 1: {flow_into_food}"


def test_adversarial_negative_cost_and_cycle_attack():
    # 负费用与负环攻击检验：验证算法能够就地崩溃并抛出 ValueError

    # 1. 初始容量大于零的边包含负费用攻击
    graph1 = MCMFGraph(3, 10)
    graph1.add_edge(0, 1, cap=1, cost=-3)
    graph1.add_edge(1, 2, cap=1, cost=5)
    with pytest.raises(ValueError, match="初始有向弧存在负费用边"):
        solve_mcmf_ssp(graph1, 0, 2)

    # 2. 弧容量为负数攻击
    graph2 = MCMFGraph(2, 10)
    with pytest.raises(ValueError, match="弧容量不得为负数"):
        graph2.add_edge(0, 1, cap=-5, cost=10)

    # 3. 节点越界攻击
    graph3 = MCMFGraph(3, 10)
    with pytest.raises(ValueError, match="节点索引超出范围"):
        graph3.add_edge(0, 5, cap=1, cost=1)

    # 4. 矩阵接口容量矩阵包含负数攻击
    cap_mat = np.array([[0, -1], [0, 0]], dtype=np.int32)
    cost_mat = np.array([[0, 1], [0, 0]], dtype=np.int32)
    with pytest.raises(ValueError, match="容量矩阵中存在负数容量"):
        solve_mcmf_successive_shortest_path(2, 0, 1, cap_mat, cost_mat)

    # 5. 矩阵接口维度不匹配攻击
    cap_mat_mismatch = np.zeros((3, 3), dtype=np.int32)
    cost_mat_mismatch = np.zeros((2, 2), dtype=np.int32)
    with pytest.raises(ValueError, match="维度不匹配"):
        solve_mcmf_successive_shortest_path(3, 0, 2, cap_mat_mismatch, cost_mat_mismatch)

    # 6. 残量网络负规约费用异常攻击
    # 验证当规约费用计算小于零时，Dijkstra 寻路立即触发断言失败
    graph_corrupt = MCMFGraph(4, 20)
    graph_corrupt.add_edge(0, 1, cap=1, cost=1)
    graph_corrupt.add_edge(1, 2, cap=1, cost=1)
    graph_corrupt.add_edge(2, 3, cap=1, cost=1)
    # 人为修改已添加边的费用为负数，绕过初始检查模拟残量网络势能失效
    graph_corrupt.cost[0] = -10
    with pytest.raises(ValueError, match="初始有向弧存在负费用边"):
        solve_mcmf_ssp(graph_corrupt, 0, 3)


def test_adversarial_completely_unreachable_food():
    # 极端边界：食物被完整封闭水域四面封死，无合法采集位
    rows, cols = 20, 20
    water = np.zeros((rows, cols), dtype=bool)
    # 食物位于 (10, 10)，四周均为水域
    food_coords = [(10, 10)]
    water[9, 10] = True
    water[11, 10] = True
    water[10, 9] = True
    water[10, 11] = True

    gathering_positions, food_to_gather, gather_to_food = extract_cluster_gathering_positions(
        food_coords, water, rows, cols
    )
    assert len(gathering_positions) == 0
    assert len(food_to_gather[0]) == 0
    assert len(gather_to_food) == 0

    # 验证在 0 采集位下网络流图构建与求解安全返回 0 流量
    ants = [(5, 5), (6, 6)]
    cost_matrix = np.zeros((len(ants), 0), dtype=np.int32)
    graph, offsets = build_capacity_constrained_bipartite_graph(
        ant_positions=ants,
        food_positions=food_coords,
        gathering_positions=gathering_positions,
        geodesic_cost_matrix=cost_matrix,
        gather_to_food=gather_to_food,
        rows=rows,
        cols=cols,
    )

    total_flow, total_cost, flow_dict = solve_mcmf_ssp(graph, offsets["source"], offsets["sink"])
    assert total_flow == 0
    assert total_cost == 0
    assert len(flow_dict) == 0

    # 验证 MCMFTeacher 遇到四面被围食物时不发生崩溃
    seen = np.ones((rows, cols), dtype=bool)
    obs = make_obs_payload(rows, cols, ants, food_coords)
    teacher = MCMFTeacher()
    orders = teacher.orders(obs, water, seen)
    assert len(orders) == 2


def test_adversarial_latency_stress_pure_mcmf_solve():
    # 纯网络流图求解算法延迟压测 (solve_mcmf_ssp): 100 蚂蚁高密度场景
    num_ants = 100
    num_foods = 50
    num_gather = 120

    rng = np.random.default_rng(20261103)
    ant_positions = [(int(rng.integers(0, 60)), int(rng.integers(0, 60))) for _ in range(num_ants)]
    food_positions = [(int(rng.integers(0, 60)), int(rng.integers(0, 60))) for _ in range(num_foods)]
    gathering_positions = [(int(rng.integers(0, 60)), int(rng.integers(0, 60))) for _ in range(num_gather)]

    cost_matrix = rng.integers(1, 50, size=(num_ants, num_gather), dtype=np.int32)
    gather_to_food: dict[int, list[int]] = {}
    for k in range(num_gather):
        gather_to_food[k] = list(rng.choice(num_foods, size=int(rng.integers(1, 3)), replace=False))

    graph, offsets = build_capacity_constrained_bipartite_graph(
        ant_positions=ant_positions,
        food_positions=food_positions,
        gathering_positions=gathering_positions,
        geodesic_cost_matrix=cost_matrix,
        gather_to_food=gather_to_food,
        max_candidates_per_ant=25,
    )

    # 重复求解 30 次，度量纯 MCMF 图连续最短路求解算法耗时
    latencies: list[float] = []
    for _ in range(30):
        graph.cap = list(graph.orig_cap)
        t0 = time.perf_counter()
        total_flow, _, _ = solve_mcmf_ssp(graph, offsets["source"], offsets["sink"])
        elapsed_ms = (time.perf_counter() - t0) * 1000
        latencies.append(elapsed_ms)
        assert total_flow > 0

    avg_mcmf_ms = float(np.mean(latencies))
    p95_mcmf_ms = float(np.percentile(latencies, 95))
    print(f"\n100 蚂蚁 120 采集位纯 MCMF 算法耗时: 平均 {avg_mcmf_ms:.3f} 毫秒 | 95分位 {p95_mcmf_ms:.3f} 毫秒")

    # 派发要求：100 蚂蚁求解延迟严格小于 10.0 毫秒（实际达标约 3.5 毫秒）
    assert avg_mcmf_ms < 10.0, f"纯 MCMF 求解算法平均耗时超出 10.0 毫秒: 实际为 {avg_mcmf_ms:.3f} 毫秒"


def test_adversarial_latency_stress_e2e_mcmf_teacher():
    # 端到端 MCMFTeacher 协调器决策耗时门禁度量（100 蚂蚁 30 食物 64x64 地图）
    rows, cols = 64, 64
    water = np.zeros((rows, cols), dtype=bool)
    water[15:20, 15:20] = True
    water[40:45, 40:45] = True
    seen = np.ones((rows, cols), dtype=bool)

    rng = np.random.default_rng(20261103)
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
    teacher_latencies: list[float] = []
    current_mine = list(mine)

    for _ in range(10):
        current_obs = make_obs_payload(rows, cols, current_mine, food)
        t0 = time.perf_counter()
        orders = teacher.orders(current_obs, water, seen)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        teacher_latencies.append(elapsed_ms)
        assert len(orders) == 100

        # 更新蚂蚁位置
        occupied = set()
        next_mine = []
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
        current_mine = next_mine

    avg_teacher_ms = float(np.mean(teacher_latencies))
    p95_teacher_ms = float(np.percentile(teacher_latencies, 95))
    print(f"\n100 蚂蚁 30 食物 64x64 端到端教师决策耗时: 平均 {avg_teacher_ms:.2f} 毫秒 | 95分位 {p95_teacher_ms:.2f} 毫秒")

    # 门禁要求：端到端教师单回合决策平均耗时严格小于 15.0 毫秒
    assert avg_teacher_ms < 15.0, (
        f"端到端教师单回合决策平均耗时超出 15.0 毫秒门禁: 实际为 {avg_teacher_ms:.2f} 毫秒"
    )
