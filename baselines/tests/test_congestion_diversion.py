from __future__ import annotations

import time
import numpy as np
import pytest

from tb_baselines.coordination import (
    FAR,
    build_capacity_constrained_bipartite_graph,
    clear_corridor_width_cache,
    compute_batch_torus_geodesics,
    compute_corridor_width_grid,
    compute_local_density_field,
    compute_path_congestion_penalties,
    solve_mcmf_ssp,
    trace_geodesic_allocation_path,
)
from tb_baselines.mcmf_teacher import MCMFTeacher


def test_corridor_width_grid_precision():
    rows, cols = 16, 22
    water = np.zeros((rows, cols), dtype=bool)

    # 通道 A (狭窄走廊，第 3 行，宽度为 1)
    water[2, 3:17] = True
    water[4, 3:17] = True

    # 通道 B (宽走廊，第 9 到 12 行，宽度为 4)
    water[8, 3:17] = True
    water[13, 3:17] = True

    clear_corridor_width_cache()
    width_grid = compute_corridor_width_grid(water, use_cache=True)
    assert width_grid.shape == (rows, cols)

    # 验证通道 A 内部陆地宽度精确为 1
    assert (width_grid[3, 4:16] == 1).all()
    # 验证通道 B 内部陆地宽度精确为 4
    assert (width_grid[9:13, 4:16] == 4).all()
    # 验证水域方格宽度度量精确为 0
    assert (width_grid[water] == 0).all()

    # 验证缓存命中一致性
    cached_grid = compute_corridor_width_grid(water, use_cache=True)
    np.testing.assert_array_equal(width_grid, cached_grid)

    # 异常输入快速就地崩溃
    with pytest.raises(TypeError):
        compute_corridor_width_grid(water.astype(np.float32))
    with pytest.raises(ValueError):
        compute_corridor_width_grid(np.zeros((10, 10, 2), dtype=bool))


def test_local_density_field_torus_convolution():
    rows, cols = 20, 20
    passable = np.ones((rows, cols), dtype=bool)

    # 空载状态全图密度严格为 0
    density_empty = compute_local_density_field([], rows, cols)
    assert (density_empty == 0.0).all()

    # 单只蚂蚁位于 (0, 0)
    density_single = compute_local_density_field([(0, 0)], rows, cols)
    # 中心自身及正交 4 邻域（含环面回绕点）密度为 1.0
    assert density_single[0, 0] == 1.0
    assert density_single[0, 1] == 1.0
    assert density_single[0, cols - 1] == 1.0
    assert density_single[1, 0] == 1.0
    assert density_single[rows - 1, 0] == 1.0
    # 距离大于 1 的单元格密度为 0.0
    assert density_single[5, 5] == 0.0

    # 障碍物掩码置零测试
    passable[0, 1] = False
    density_masked = compute_local_density_field([(0, 0)], rows, cols, passable_mask=passable)
    assert density_masked[0, 1] == 0.0

    # 异常坐标快速就地崩溃
    with pytest.raises(ValueError):
        compute_local_density_field([(25, 0)], rows, cols)
    with pytest.raises(ValueError):
        compute_local_density_field([(0, 0)], 0, 10)


def test_path_congestion_penalties_scaling():
    rows, cols = 16, 22
    water = np.zeros((rows, cols), dtype=bool)
    water[2, 3:17] = True
    water[4, 3:17] = True
    water[:, 21] = True
    passable = ~water

    width_grid = compute_corridor_width_grid(water)
    target = (3, 18)
    ant = (3, 1)

    geodesic_fields = compute_batch_torus_geodesics([target], passable, rows, cols)

    # 场景 1: 无蚂蚁拥塞，惩罚为 0
    zero_density = np.zeros((rows, cols), dtype=np.float32)
    pen_zero = compute_path_congestion_penalties(
        [ant], [target], geodesic_fields, width_grid, zero_density, passable, rows, cols
    )
    assert pen_zero[0, 0] == 0

    # 场景 2: 通道内聚集 7 只蚂蚁产生阶跃拥塞
    congested_ants = [(3, c) for c in range(5, 12)]
    density_congested = compute_local_density_field(congested_ants, rows, cols, passable_mask=passable)
    pen_congested = compute_path_congestion_penalties(
        [ant], [target], geodesic_fields, width_grid, density_congested, passable, rows, cols, alpha=2.0, beta=1.5
    )
    assert pen_congested[0, 0] > 15
    assert isinstance(int(pen_congested[0, 0]), int)


def test_alternate_corridor_diversion_via_mcmf():
    rows, cols = 16, 22
    water = np.zeros((rows, cols), dtype=bool)

    # 通道 A (狭窄走廊，第 3 行，宽度 1)
    water[2, 3:17] = True
    water[4, 3:17] = True

    # 通道 B (宽走廊，第 9 至 12 行，宽度 4)
    water[8, 3:17] = True
    water[13, 3:17] = True

    # 中部垂直分隔水域墙 (列 10)，迫使流量只能经由通道 A 或通道 B
    water[:2, 10] = True
    water[5:8, 10] = True
    water[14:, 10] = True

    # 右侧出口分隔水域墙
    water[5:8, 17:] = True
    # 阻断外侧环面回绕
    water[:, 21] = True

    passable = ~water
    width_grid = compute_corridor_width_grid(water)

    gather_positions = [(3, 18), (10, 18)]
    food_positions = [(3, 19), (10, 19)]
    for fr, fc in food_positions:
        passable[fr, fc] = False

    deciding_ant = [(3, 1)]
    geodesic_fields = compute_batch_torus_geodesics(gather_positions, passable, rows, cols)
    d_A = int(geodesic_fields[0][3, 1])
    d_B = int(geodesic_fields[1][3, 1])
    assert d_A < d_B  # 通道 A 几何测地距离更短

    cost_matrix = np.array([[d_A, d_B]], dtype=np.int32)

    # 阶段 1: 无拥塞基线分配，选择测地距离更短的通道 A (采集位 0)
    graph_base, offsets_base = build_capacity_constrained_bipartite_graph(
        ant_positions=deciding_ant,
        food_positions=food_positions,
        gathering_positions=gather_positions,
        geodesic_cost_matrix=cost_matrix,
        congestion_penalty_matrix=None,
        rows=rows,
        cols=cols,
    )
    _, _, flow_base = solve_mcmf_ssp(
        graph_base, offsets_base["source"], offsets_base["sink"], max_flow_limit=1
    )
    assigned_g_base = [
        v - offsets_base["g_in_offset"]
        for (u, v), f in flow_base.items()
        if u == offsets_base["u_offset"] and f > 0
    ][0]
    assert assigned_g_base == 0

    # 阶段 2: 通道 A 发生蚂蚁密集拥塞，计算动态拥塞惩罚
    congested_ants = [(3, 5), (3, 6), (3, 7), (3, 8), (3, 9), (3, 11), (3, 12)]
    all_ants = deciding_ant + congested_ants
    density_grid = compute_local_density_field(all_ants, rows, cols, passable_mask=passable)

    penalty_matrix = compute_path_congestion_penalties(
        ant_positions=deciding_ant,
        gathering_positions=gather_positions,
        geodesic_fields=geodesic_fields,
        width_grid=width_grid,
        density_grid=density_grid,
        passable_mask=passable,
        rows=rows,
        cols=cols,
        alpha=2.0,
        beta=1.5,
    )

    total_cost_A = cost_matrix[0, 0] + penalty_matrix[0, 0]
    total_cost_B = cost_matrix[0, 1] + penalty_matrix[0, 1]
    assert total_cost_A > total_cost_B

    # 阶段 3: 注入拥塞惩罚矩阵，验证网络流求解器自动分流至通道 B (采集位 1)
    graph_dyn, offsets_dyn = build_capacity_constrained_bipartite_graph(
        ant_positions=deciding_ant,
        food_positions=food_positions,
        gathering_positions=gather_positions,
        geodesic_cost_matrix=cost_matrix,
        congestion_penalty_matrix=penalty_matrix,
        rows=rows,
        cols=cols,
    )
    _, _, flow_dyn = solve_mcmf_ssp(
        graph_dyn, offsets_dyn["source"], offsets_dyn["sink"], max_flow_limit=1
    )
    assigned_g_dyn = [
        v - offsets_dyn["g_in_offset"]
        for (u, v), f in flow_dyn.items()
        if u == offsets_dyn["u_offset"] and f > 0
    ][0]
    assert assigned_g_dyn == 1


def test_mcmf_teacher_step_congestion_diversion():
    rows, cols = 40, 50
    water = np.ones((rows, cols), dtype=bool)
    entrance = (15, 10)
    exit_tile = (15, 35)

    water[entrance[0], entrance[1]] = False
    water[exit_tile[0], exit_tile[1]] = False

    # 短窄走廊: 行 10 (长度 35 步, 宽度 1)
    for c in range(entrance[1] + 1, exit_tile[1]):
        water[10, c] = False
    for r in range(10, entrance[0] + 1):
        water[r, entrance[1]] = False
        water[r, exit_tile[1]] = False

    # 长宽走廊: 行 24..26 (宽度 3，长度 43 步)
    for c in range(entrance[1] + 1, exit_tile[1]):
        for w in range(3):
            water[24 + w, c] = False
    for r in range(entrance[0], 27):
        water[r, entrance[1]] = False
        water[r, exit_tile[1]] = False

    # 食物与采集位
    water[13:18, 35:39] = False
    foods = [(13 + i, 38) for i in range(5)]

    # 4 只蚂蚁堵塞在短窄走廊
    narrow_ants = [(10, entrance[1] + 3 + i * 3) for i in range(4)]
    all_ants = [entrance] + narrow_ants

    teacher = MCMFTeacher(max_depth=60, dynamic_congestion=True, congestion_alpha=10.0, congestion_beta=1.5)
    obs = {
        "size": (rows, cols),
        "mine": all_ants,
        "food": foods,
        "ids": list(range(len(all_ants))),
    }
    seen = np.ones((rows, cols), dtype=bool)

    orders = teacher.step(obs, water, seen)
    assert len(orders) == len(all_ants)
    # 决策蚂蚁绕行走向长宽走廊 (South 动作 2)
    assert orders[0] == 2
    assert teacher.last_congestion_penalty_matrix is not None


def test_dynamic_congestion_latency_benchmark():
    rows, cols = 64, 64
    water = np.zeros((rows, cols), dtype=bool)
    water[15:20, 15:20] = True
    water[40:45, 40:45] = True

    rng = np.random.default_rng(2026)
    ants = []
    while len(ants) < 100:
        r = int(rng.integers(0, rows))
        c = int(rng.integers(0, cols))
        if not water[r, c] and (r, c) not in ants:
            ants.append((r, c))

    gathers = []
    while len(gathers) < 16:
        r = int(rng.integers(0, rows))
        c = int(rng.integers(0, cols))
        if not water[r, c] and (r, c) not in gathers:
            gathers.append((r, c))

    passable = ~water
    t0 = time.perf_counter()
    width_grid = compute_corridor_width_grid(water, use_cache=True)
    density_grid = compute_local_density_field(ants, rows, cols, passable_mask=passable)
    geodesic_fields = compute_batch_torus_geodesics(gathers, passable, rows, cols, max_depth=16)
    penalty_matrix = compute_path_congestion_penalties(
        ant_positions=ants,
        gathering_positions=gathers,
        geodesic_fields=geodesic_fields,
        width_grid=width_grid,
        density_grid=density_grid,
        passable_mask=passable,
        rows=rows,
        cols=cols,
        alpha=2.0,
        beta=1.5,
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000

    assert penalty_matrix.shape == (100, 16)
    assert elapsed_ms < 15.0
