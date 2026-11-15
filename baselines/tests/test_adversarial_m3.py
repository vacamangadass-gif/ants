from __future__ import annotations

import time
import numpy as np
import pytest

from tb_baselines.coordination import (
    FAR,
    compute_batch_torus_geodesics_multibucket,
    compute_corridor_width,
    compute_corridor_width_grid,
    compute_cost_matrix_bitmask_multibucket,
    compute_dynamic_congestion_penalty,
    compute_local_ant_density,
    compute_local_density_field,
)
from tb_baselines.mcmf_teacher import MCMFTeacher


def build_asymmetric_dual_corridor_fixture(num_ants_in_narrow: int = 0):
    rows, cols = 40, 50
    water = np.ones((rows, cols), dtype=bool)
    entrance = (15, 10)
    exit_tile = (15, 35)

    water[entrance[0], entrance[1]] = False
    water[exit_tile[0], exit_tile[1]] = False

    # 短窄走廊: 行 10 (长度 5 + 25 + 5 = 35 步, 宽度 1)
    narrow_row = 10
    for c in range(entrance[1] + 1, exit_tile[1]):
        water[narrow_row, c] = False
    for r in range(narrow_row, entrance[0] + 1):
        water[r, entrance[1]] = False
        water[r, exit_tile[1]] = False

    # 长宽走廊: 行 24..26 (宽度 3，长度 9 + 25 + 9 = 43 步)
    wide_row = 24
    for c in range(entrance[1] + 1, exit_tile[1]):
        for w in range(3):
            water[wide_row + w, c] = False
    for r in range(entrance[0], wide_row + 3):
        water[r, entrance[1]] = False
        water[r, exit_tile[1]] = False

    # 导出垂直连通的 5 颗食物与采集位
    water[13:18, 35] = False
    foods = []
    gather_positions = []
    for i in range(5):
        fr = 13 + i
        water[fr, 35:39] = False
        foods.append((fr, 38))
        gather_positions.append((fr, 37))

    gather_pos = (exit_tile[0], exit_tile[1] + 2)
    food_pos = (exit_tile[0], exit_tile[1] + 3)

    narrow_ants = [(narrow_row, entrance[1] + 3 + i * 3) for i in range(num_ants_in_narrow)]
    deciding_ant = entrance
    all_ants = [deciding_ant] + narrow_ants

    return {
        "rows": rows,
        "cols": cols,
        "water": water,
        "passable": ~water,
        "gather_pos": gather_pos,
        "food_pos": food_pos,
        "foods": foods,
        "gather_positions": gather_positions,
        "deciding_ant": deciding_ant,
        "narrow_ants": narrow_ants,
        "all_ants": all_ants,
    }


def test_adversarial_three_state_critical_transition():
    # 检验 0 只、1 只与 4 只前导蚂蚁下的三态临界翻转行为
    # 动作说明: 0 代表 North (短窄走廊)，2 代表 South (长宽走廊)
    for ants_cnt, expected_action, desc in [
        (0, 0, "空载状态必须选择短窄走廊"),
        (1, 0, "1 只前导蚂蚁必须坚守短窄走廊杜绝伪分流"),
        (4, 2, "4 只前导蚂蚁拥堵必须自动翻转选择长宽走廊"),
    ]:
        env = build_asymmetric_dual_corridor_fixture(num_ants_in_narrow=ants_cnt)
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

        orders = teacher.orders(obs, env["water"], seen)
        deciding_action = orders[0]
        assert deciding_action == expected_action, (
            f"前导蚂蚁数量为 {ants_cnt} ({desc}) 时，决策动作应为 {expected_action}，"
            f"实际输出为 {deciding_action}"
        )


def test_adversarial_static_vs_dynamic_ablation():
    # 对比静态无拥塞与动态二次拥塞下的分流差异
    env_4 = build_asymmetric_dual_corridor_fixture(num_ants_in_narrow=4)
    obs = {
        "size": (env_4["rows"], env_4["cols"]),
        "mine": env_4["all_ants"],
        "food": env_4["foods"],
        "ids": list(range(len(env_4["all_ants"]))),
    }
    seen = np.ones((env_4["rows"], env_4["cols"]), dtype=bool)

    # 静态模式 (dynamic_congestion=False): 无论拥堵多严重，均固执选择短窄走廊
    static_teacher = MCMFTeacher(max_depth=60, dynamic_congestion=False)
    static_orders = static_teacher.orders(obs, env_4["water"], seen)
    assert static_orders[0] == 0, "静态模式下决策蚂蚁必须选择短窄走廊 (North 动作 0)"

    # 动态模式 (dynamic_congestion=True): 4 只蚂蚁拥挤时自动分流至长宽走廊
    dynamic_teacher = MCMFTeacher(
        max_depth=60,
        dynamic_congestion=True,
        congestion_alpha=10.0,
        congestion_beta=1.5,
    )
    dynamic_orders = dynamic_teacher.orders(obs, env_4["water"], seen)
    assert dynamic_orders[0] == 2, "动态模式下决策蚂蚁必须分流至长宽走廊 (South 动作 2)"


def test_adversarial_multi_agent_long_horizon_simulation():
    # 构造复杂多走廊瓶颈地图，进行 50 回合长程连续动态推演
    rows, cols = 40, 50
    water = np.ones((rows, cols), dtype=bool)

    # 建立 3 条横向连通走廊
    # 走廊 1: 行 8, 宽度 1
    water[8, 5:45] = False
    # 走廊 2: 行 20..21, 宽度 2
    water[20:22, 5:45] = False
    # 走廊 3: 行 32..34, 宽度 3
    water[32:35, 5:45] = False

    # 建立左右两端连接主走廊的纵向通道
    water[8:35, 5] = False
    water[8:35, 44] = False

    # 放置 10 颗食物分布在右侧出口
    foods = [(8, 43), (9, 44), (20, 43), (21, 43), (32, 43), (33, 43), (34, 43)]
    for fr, fc in foods:
        water[fr, fc] = False

    # 在左侧集结 15 只蚂蚁
    ant_positions = []
    for r in range(8, 23):
        ant_positions.append((r, 5))

    teacher = MCMFTeacher(
        max_depth=60,
        dynamic_congestion=True,
        congestion_alpha=10.0,
        congestion_beta=1.5,
    )

    deltas = [(-1, 0), (0, 1), (1, 0), (0, -1), (0, 0)]
    seen = np.ones((rows, cols), dtype=bool)

    current_ants = list(ant_positions)
    current_ids = list(range(len(current_ants)))

    for step_idx in range(50):
        obs = {
            "size": (rows, cols),
            "mine": current_ants,
            "food": foods,
            "ids": current_ids,
        }
        orders = teacher.orders(obs, water, seen)
        assert len(orders) == len(current_ants)

        next_positions = []
        for i, (ar, ac) in enumerate(current_ants):
            action = int(orders[i])
            assert 0 <= action <= 4, f"动作代码必须处于 0 至 4 范围内，实际为 {action}"
            dr, dc = deltas[action]
            nr = (ar + dr) % rows
            nc = (ac + dc) % cols

            # 约束 1: 绝对禁止走入水域
            assert not water[nr, nc], f"第 {step_idx} 回合蚂蚁 {i} 发生进入水域违规: ({nr}, {nc})"
            # 约束 2: 绝对禁止直接走入食物实体
            assert (nr, nc) not in foods, f"第 {step_idx} 回合蚂蚁 {i} 发生踩踏食物违规: ({nr}, {nc})"

            next_positions.append((nr, nc))

        # 约束 3: 严格保证无任何友方相撞 (无单元格顶点重叠冲突)
        unique_next_positions = set(next_positions)
        assert len(unique_next_positions) == len(next_positions), (
            f"第 {step_idx} 回合发生友方相撞冲突，活跃个体数 {len(next_positions)}，"
            f"唯一位置数 {len(unique_next_positions)}"
        )

        current_ants = next_positions


def test_adversarial_latency_stress_64x64_and_80x80():
    # 极端网格压力测试：分别测量 64x64 与 80x80 网格下单回合决策耗时
    scales = [(64, 64, 100, 30), (80, 80, 100, 40)]
    results = {}

    for rows, cols, num_ants, num_foods in scales:
        water = np.zeros((rows, cols), dtype=bool)
        # 设置多处大面积水域阻挡
        water[10:20, 10:20] = True
        water[30:40, 30:40] = True
        water[50:60, 20:35] = True
        if rows >= 80:
            water[65:75, 50:65] = True

        # 设置狭窄走廊
        for c in range(15, cols - 15):
            water[25, c] = True
            water[27, c] = True
        water[26, 15 : cols - 15] = False

        passable = ~water
        rng = np.random.default_rng(2026)

        ant_set = set()
        # 走廊内密集布置 20 只蚂蚁
        for c in range(20, min(40, cols - 16)):
            ant_set.add((26, c))

        while len(ant_set) < num_ants:
            r = int(rng.integers(0, rows))
            c = int(rng.integers(0, cols))
            if passable[r, c]:
                ant_set.add((r, c))
        ants = list(ant_set)

        food_set = set()
        while len(food_set) < num_foods:
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

        import gc
        # 预热一次并隔离垃圾回收扰动
        teacher.orders(obs, water, seen)
        gc.collect()

        times = []
        for _ in range(5):
            t0 = time.perf_counter()
            orders = teacher.orders(obs, water, seen)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            times.append(elapsed_ms)
            assert len(orders) == num_ants

        avg_ms = float(np.mean(times))
        max_ms = float(np.max(times))
        results[(rows, cols)] = (avg_ms, max_ms)
        print(f"\n网格 {rows}x{cols} (蚂蚁={num_ants}, 食物={num_foods}) 单回合平均耗时: {avg_ms:.3f} ms, 最大耗时: {max_ms:.3f} ms")

        # 硬性门禁：单回合平均耗时必须坚决低于 15.0 ms
        assert avg_ms < 15.0, f"{rows}x{cols} 平均耗时 {avg_ms:.3f} ms 超出 15.0 ms 硬性门禁"
        # 优化目标：64x64 网格必须低于 10.0 ms
        if rows <= 64:
            assert avg_ms < 10.0, f"{rows}x{cols} 平均耗时 {avg_ms:.3f} ms 超出 10.0 ms 优化门禁"
