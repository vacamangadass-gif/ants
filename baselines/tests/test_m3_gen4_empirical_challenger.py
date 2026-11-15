from __future__ import annotations

import time
import numpy as np
import pytest

from tb_baselines.coordination import (
    FAR,
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
    build_capacity_constrained_bipartite_graph,
    solve_mcmf_ssp,
)
from tb_baselines.mcmf_teacher import MCMFTeacher


def build_dual_corridor_fixture(width_narrow: int = 1, width_wide: int = 3, num_ants_in_narrow: int = 0):
    rows, cols = 40, 50
    water = np.ones((rows, cols), dtype=bool)
    entrance = (15, 10)
    exit_tile = (15, 35)

    water[entrance[0], entrance[1]] = False
    water[exit_tile[0], exit_tile[1]] = False

    # 狭窄走廊: 位于北部行 10
    narrow_start_r = 10
    for c in range(entrance[1] + 1, exit_tile[1]):
        for w in range(width_narrow):
            water[narrow_start_r + w, c] = False
    # 连接 entrance 与 exit_tile 至狭窄走廊
    for r in range(narrow_start_r, entrance[0] + 1):
        water[r, entrance[1]] = False
        water[r, exit_tile[1]] = False

    # 较宽走廊: 位于南部行 24
    wide_start_r = 24
    for c in range(entrance[1] + 1, exit_tile[1]):
        for w in range(width_wide):
            water[wide_start_r + w, c] = False
    # 连接 entrance 与 exit_tile 至较宽走廊
    for r in range(entrance[0], wide_start_r + width_wide):
        water[r, entrance[1]] = False
        water[r, exit_tile[1]] = False

    # 终点区域配置足够食物，确保每只蚂蚁均有对应获配采集位
    total_ant_count = 1 + num_ants_in_narrow
    num_foods = max(8, total_ant_count + 3)
    foods = []
    for i in range(num_foods):
        fr = (12 + i) % rows
        fc = 38
        water[fr, 35:40] = False
        foods.append((fr, fc))

    narrow_ants = []
    for i in range(num_ants_in_narrow):
        col_pos = entrance[1] + 3 + i * 2
        if col_pos < exit_tile[1]:
            narrow_ants.append((narrow_start_r, col_pos))

    deciding_ant = entrance
    all_ants = [deciding_ant] + narrow_ants

    return {
        "rows": rows,
        "cols": cols,
        "water": water,
        "passable": ~water,
        "foods": foods,
        "deciding_ant": deciding_ant,
        "narrow_ants": narrow_ants,
        "all_ants": all_ants,
        "entrance": entrance,
        "exit_tile": exit_tile,
    }


def test_adversarial_corridor_width_1_diversion_transition():
    # 验证宽度为 1 的狭窄走廊在密度增加下的分流翻转行为
    # 动作定义: 0 为 North (短窄走廊), 2 为 South (长宽走廊)
    fixture_0 = build_dual_corridor_fixture(width_narrow=1, width_wide=3, num_ants_in_narrow=0)
    seen_0 = np.ones((fixture_0["rows"], fixture_0["cols"]), dtype=bool)
    obs_0 = {
        "size": (fixture_0["rows"], fixture_0["cols"]),
        "mine": fixture_0["all_ants"],
        "food": fixture_0["foods"],
        "ids": list(range(len(fixture_0["all_ants"]))),
    }

    # 空载状态：决策蚂蚁坚定选择短窄走廊 (North)
    teacher_dyn = MCMFTeacher(max_depth=60, dynamic_congestion=True, congestion_alpha=10.0, congestion_beta=1.5)
    orders_0 = teacher_dyn.orders(obs_0, fixture_0["water"], seen_0)
    assert orders_0[0] == 0

    # 静态模式 (无拥塞感知)：即使窄走廊塞入 4 只蚂蚁，仍然选择短窄走廊 (North)
    fixture_4 = build_dual_corridor_fixture(width_narrow=1, width_wide=3, num_ants_in_narrow=4)
    seen_4 = np.ones((fixture_4["rows"], fixture_4["cols"]), dtype=bool)
    obs_4 = {
        "size": (fixture_4["rows"], fixture_4["cols"]),
        "mine": fixture_4["all_ants"],
        "food": fixture_4["foods"],
        "ids": list(range(len(fixture_4["all_ants"]))),
    }
    teacher_stat = MCMFTeacher(max_depth=60, dynamic_congestion=False)
    orders_stat = teacher_stat.orders(obs_4, fixture_4["water"], seen_4)
    assert orders_stat[0] == 0

    # 动态拥塞模式：窄走廊塞入 4 只蚂蚁后，决策蚂蚁主动分流选择长宽走廊 (South)
    orders_dyn = teacher_dyn.orders(obs_4, fixture_4["water"], seen_4)
    assert orders_dyn[0] == 2
    assert teacher_dyn.last_congestion_penalty_matrix is not None


def test_adversarial_corridor_width_2_vs_width_1_scaling():
    # 检验走廊宽度对拥塞惩罚的调节效应：
    # 宽度 2 的走廊承载容量更高，相同蚂蚁聚集量下的流阻应显著低于宽度 1
    fixture_w1 = build_dual_corridor_fixture(width_narrow=1, width_wide=4, num_ants_in_narrow=2)
    fixture_w2 = build_dual_corridor_fixture(width_narrow=2, width_wide=4, num_ants_in_narrow=2)

    width_map_1 = compute_corridor_width(fixture_w1["water"], fixture_w1["rows"], fixture_w1["cols"])
    width_map_2 = compute_corridor_width(fixture_w2["water"], fixture_w2["rows"], fixture_w2["cols"])

    # 狭窄走廊内部宽度校验
    assert (width_map_1[10, 15:30] == 1).all()
    assert (width_map_2[10:12, 15:30] == 2).all()

    # 计算相同聚集蚂蚁下的局部密度
    density_1 = compute_local_ant_density(fixture_w1["narrow_ants"], fixture_w1["passable"], fixture_w1["rows"], fixture_w1["cols"], radius=2)
    density_2 = compute_local_ant_density(fixture_w2["narrow_ants"], fixture_w2["passable"], fixture_w2["rows"], fixture_w2["cols"], radius=2)

    raw_pen_1, delay_1 = compute_dynamic_congestion_penalty(width_map_1, density_1, fixture_w1["passable"], alpha=10.0, beta=1.5)
    raw_pen_2, delay_2 = compute_dynamic_congestion_penalty(width_map_2, density_2, fixture_w2["passable"], alpha=10.0, beta=1.5)

    # 宽度为 2 的走廊惩罚必须显著低于宽度为 1 的走廊
    pen_max_1 = float(np.max(raw_pen_1[10, 12:25]))
    pen_max_2 = float(np.max(raw_pen_2[10, 12:25]))
    assert pen_max_1 > pen_max_2 * 2.0


def test_adversarial_torus_wrapping_bottleneck_diversion():
    # 环面跨界走廊：走廊穿越行 0 与行 rows-1 的环面边界
    rows, cols = 40, 50
    water = np.ones((rows, cols), dtype=bool)

    # 入口与出口
    entrance = (0, 10)
    exit_tile = (0, 35)
    water[entrance[0], entrance[1]] = False
    water[exit_tile[0], exit_tile[1]] = False

    # 跨越环面上边缘的短窄走廊 (位于行 38..39 与行 0..1 环面回绕区)
    # 北部环面走廊: 行 36 (宽度 1, 长度短)
    for c in range(entrance[1] + 1, exit_tile[1]):
        water[36, c] = False
    # 连接 entrance 与 exit_tile (跨界 0 -> 39 -> 38 -> 37 -> 36)
    for r in [0, 39, 38, 37, 36]:
        water[r, entrance[1]] = False
        water[r, exit_tile[1]] = False

    # 南部平原长宽走廊: 行 10..13 (宽度 4, 几何距离长)
    for c in range(entrance[1] + 1, exit_tile[1]):
        for w in range(4):
            water[10 + w, c] = False
    for r in range(0, 14):
        water[r, entrance[1]] = False
        water[r, exit_tile[1]] = False

    # 食物区域
    foods = []
    for i in range(8):
        fr = (33 + i) % rows
        water[fr, 35:40] = False
        foods.append((fr, 38))

    seen = np.ones((rows, cols), dtype=bool)

    # 场景 1: 空载状态，决策蚂蚁经由环面边界向北走入短窄走廊 (North 动作 0，跨越 0 -> 39)
    teacher_dyn = MCMFTeacher(max_depth=60, dynamic_congestion=True, congestion_alpha=10.0, congestion_beta=1.5)
    obs_empty = {
        "size": (rows, cols),
        "mine": [entrance],
        "food": foods,
        "ids": [0],
    }
    orders_empty = teacher_dyn.orders(obs_empty, water, seen)
    assert orders_empty[0] == 0

    # 场景 2: 环面窄走廊聚集 4 只蚂蚁形成严重拥堵
    narrow_ants = [(36, entrance[1] + 3 + i * 3) for i in range(4)]
    all_ants = [entrance] + narrow_ants
    obs_congested = {
        "size": (rows, cols),
        "mine": all_ants,
        "food": foods,
        "ids": list(range(len(all_ants))),
    }
    orders_congested = teacher_dyn.orders(obs_congested, water, seen)
    # 决策蚂蚁成功感应环面跨界走廊拥堵，翻转分流向南走入长宽走廊 (South 动作 2)
    assert orders_congested[0] == 2


def test_adversarial_multi_agent_stream_diversion_flow():
    # 连续多回合流推演：模拟蚂蚁逐批注入瓶颈地图，验证群体自动分流
    rows, cols = 40, 50
    water = np.ones((rows, cols), dtype=bool)

    # 走廊 1 (短窄走廊，行 10，宽度 1)
    water[10, 5:45] = False
    # 走廊 2 (长宽走廊，行 25..27，宽度 3)
    water[25:28, 5:45] = False

    # 竖直连通纵向通道
    water[10:28, 5] = False
    water[10:28, 44] = False

    # 食物资源
    foods = [(10, 43), (25, 43), (26, 43), (27, 43), (18, 43)]
    for fr, fc in foods:
        water[fr, fc] = False

    # 初始 8 只蚂蚁集中在左侧纵向通道 (行 10 至 17, 列 5)
    ants = [(r, 5) for r in range(10, 18)]
    teacher = MCMFTeacher(max_depth=60, dynamic_congestion=True, congestion_alpha=10.0, congestion_beta=1.5)

    seen = np.ones((rows, cols), dtype=bool)
    deltas = [(-1, 0), (0, 1), (1, 0), (0, -1), (0, 0)]

    corridor_narrow_entered = 0
    corridor_wide_entered = 0

    current_ants = list(ants)
    for _ in range(20):
        obs = {
            "size": (rows, cols),
            "mine": current_ants,
            "food": foods,
            "ids": list(range(len(current_ants))),
        }
        orders = teacher.orders(obs, water, seen)
        assert len(orders) == len(current_ants)

        next_ants = []
        for i, (ar, ac) in enumerate(current_ants):
            action = int(orders[i])
            dr, dc = deltas[action]
            nr = (ar + dr) % rows
            nc = (ac + dc) % cols
            assert not water[nr, nc]
            assert (nr, nc) not in foods
            next_ants.append((nr, nc))

            # 统计通过列 6 进入两条走廊的蚂蚁数量
            if ac == 5 and nc == 6:
                if nr == 10:
                    corridor_narrow_entered += 1
                elif 25 <= nr <= 27:
                    corridor_wide_entered += 1

        assert len(set(next_ants)) == len(next_ants)
        current_ants = next_ants

    # 验证两条走廊均有蚂蚁进入，表明分流机制成功起效
    assert corridor_narrow_entered >= 1
    assert corridor_wide_entered >= 1


def test_adversarial_latency_scaling_under_load():
    # 压力与延迟基准：在 64x64 环面网格上对 50、80 与 100 只蚂蚁进行决策耗时测试
    rows, cols = 64, 64
    water = np.zeros((rows, cols), dtype=bool)

    # 复杂水域与人工瓶颈走廊
    water[15:25, 15:25] = True
    water[40:50, 40:50] = True
    water[20, 5:55] = True
    water[22, 5:55] = True
    water[21, 5:55] = False

    passable = ~water
    rng = np.random.default_rng(2026)

    teacher = MCMFTeacher(
        max_depth=16,
        dynamic_congestion=True,
        congestion_alpha=10.0,
        congestion_beta=1.5,
    )
    seen = np.ones((rows, cols), dtype=bool)

    for ant_count in [50, 80, 100]:
        ant_set = set()
        for c in range(15, min(15 + ant_count // 3, 50)):
            ant_set.add((21, c))
        while len(ant_set) < ant_count:
            r = int(rng.integers(0, rows))
            c = int(rng.integers(0, cols))
            if passable[r, c]:
                ant_set.add((r, c))
        ants = list(ant_set)

        food_set = set()
        while len(food_set) < 25:
            r = int(rng.integers(0, rows))
            c = int(rng.integers(0, cols))
            if passable[r, c] and (r, c) not in ant_set:
                food_set.add((r, c))
        foods = list(food_set)

        obs = {
            "size": (rows, cols),
            "mine": ants,
            "food": foods,
            "ids": list(range(len(ants))),
        }

        import gc
        # 预热并隔离垃圾回收扰动
        for _ in range(2):
            teacher.orders(obs, water, seen)
        gc.collect()

        latencies = []
        gc.disable()
        try:
            for _ in range(5):
                t0 = time.perf_counter()
                orders = teacher.orders(obs, water, seen)
                lat = (time.perf_counter() - t0) * 1000.0
                latencies.append(lat)
                assert len(orders) == ant_count
        finally:
            gc.enable()

        avg_lat = float(np.mean(latencies))
        max_lat = float(np.max(latencies))
        print(f"\n[LATENCY BENCHMARK] 蚂蚁数: {ant_count}, 均值: {avg_lat:.2f} ms, 最大值: {max_lat:.2f} ms")

        # 严格门禁：单回合决策平均耗时必须严格小于 15.0 ms
        assert avg_lat < 15.0

