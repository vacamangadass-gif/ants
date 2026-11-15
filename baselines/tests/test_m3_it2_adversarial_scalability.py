from __future__ import annotations

import gc
import time
import numpy as np
import pytest

from tb_baselines.coordination import (
    DELTAS,
    FAR,
    HOLD,
    compute_corridor_width,
    compute_dynamic_congestion_penalty,
    compute_local_ant_density,
)
from tb_baselines.mcmf_teacher import MCMFTeacher
from tb_baselines.teacher import TaskState


def make_torus_test_grid(
    rows: int,
    cols: int,
    num_ants: int,
    num_foods: int,
    seed: int = 2026,
) -> tuple[np.ndarray, list[tuple[int, int]], list[tuple[int, int]], np.ndarray]:
    rng = np.random.default_rng(seed)
    water = np.zeros((rows, cols), dtype=bool)

    # 构造局部不规则水域障碍物
    block_count = max(2, (rows * cols) // 400)
    for _ in range(block_count):
        br = int(rng.integers(2, rows - 6))
        bc = int(rng.integers(2, cols - 6))
        bh = int(rng.integers(3, 7))
        bw = int(rng.integers(3, 7))
        water[br : br + bh, bc : bc + bw] = True

    passable = ~water
    passable_coords = [
        (int(r), int(c)) for r in range(rows) for c in range(cols) if passable[r, c]
    ]

    chosen_indices = rng.choice(len(passable_coords), size=num_ants + num_foods, replace=False)
    ant_positions = [passable_coords[i] for i in chosen_indices[:num_ants]]
    food_positions = [passable_coords[i] for i in chosen_indices[num_ants:]]

    seen = np.ones((rows, cols), dtype=bool)
    return water, ant_positions, food_positions, seen


@pytest.mark.parametrize(
    ("rows", "cols", "num_ants", "num_foods"),
    [
        (30, 30, 5, 5),
        (40, 40, 20, 10),
        (64, 64, 50, 25),
        (80, 80, 100, 50),
        (30, 30, 10, 20),
        (64, 64, 50, 50),
    ],
)
def test_mcmf_allocation_ratios_and_no_hold_stall(
    rows: int,
    cols: int,
    num_ants: int,
    num_foods: int,
):
    water, ant_positions, food_positions, seen = make_torus_test_grid(
        rows, cols, num_ants, num_foods, seed=42 + num_ants
    )

    teacher = MCMFTeacher(
        max_depth=20,
        dynamic_congestion=True,
        congestion_alpha=10.0,
        congestion_beta=1.5,
    )
    obs = {
        "size": (rows, cols),
        "mine": ant_positions,
        "food": food_positions,
        "ids": list(range(num_ants)),
    }

    orders = teacher.orders(obs, water, seen)
    assert len(orders) == num_ants

    last_tasks = teacher.get_last_tasks()
    gather_count = sum(1 for task in last_tasks.values() if task == TaskState.GATHER)

    expected_gather = min(num_ants, num_foods)
    # 验证获配 GATHER 任务的蚂蚁数量严格达到理论最优最大流
    assert gather_count == expected_gather, (
        f"规模 ({num_ants} 蚂蚁, {num_foods} 食物) 下，期望采集匹配数为 {expected_gather}，实际为 {gather_count}"
    )

    # 验证所有获配 GATHER 任务的蚂蚁在未到达目标点时绝不发出停滞 HOLD 动作
    for idx_ant, task in last_tasks.items():
        if task == TaskState.GATHER:
            action = int(orders[idx_ant])
            ant_pos = ant_positions[idx_ant]
            adjacent_to_food = any(
                (abs(ant_pos[0] - fr) % rows + abs(ant_pos[1] - fc) % cols) == 1
                for fr, fc in food_positions
            )
            if not adjacent_to_food:
                assert action != HOLD, (
                    f"蚂蚁 {idx_ant} 位于 {ant_pos} 未就位却输出了停滞 HOLD 动作"
                )


def test_mcmf_adversarial_torus_boundary_wrapping_allocation():
    rows, cols = 40, 40
    water = np.zeros((rows, cols), dtype=bool)
    seen = np.ones((rows, cols), dtype=bool)

    # 在环面边界两端极端分布蚂蚁与食物
    ants = [(0, 0), (0, 2), (2, 0)]
    foods = [(rows - 1, cols - 1), (rows - 1, cols - 3), (rows - 3, cols - 1)]

    teacher = MCMFTeacher(max_depth=16, dynamic_congestion=True)
    obs = {
        "size": (rows, cols),
        "mine": ants,
        "food": foods,
        "ids": list(range(len(ants))),
    }

    orders = teacher.orders(obs, water, seen)
    assert len(orders) == 3

    # 验证环面测地负梯度直接穿越边界跨越，而非反向绕行
    ant0_action = int(orders[0])
    assert ant0_action in (0, 3), (
        f"环面跨界最优动作应为 North (0) 或 West (3)，实际为 {ant0_action}"
    )


def test_ant_stalls_in_hold_due_to_candidate_truncation():
    rows, cols = 30, 30
    rng = np.random.default_rng(47)
    water = np.zeros((rows, cols), dtype=bool)

    block_count = max(2, (rows * cols) // 400)
    for _ in range(block_count):
        br = int(rng.integers(2, rows - 6))
        bc = int(rng.integers(2, cols - 6))
        bh = int(rng.integers(3, 7))
        bw = int(rng.integers(3, 7))
        water[br : br + bh, bc : bc + bw] = True

    passable = ~water
    passable_coords = [
        (int(r), int(c)) for r in range(rows) for c in range(cols) if passable[r, c]
    ]

    chosen_indices = rng.choice(len(passable_coords), size=5 + 5, replace=False)
    ant_positions = [passable_coords[i] for i in chosen_indices[:5]]
    food_positions = [passable_coords[i] for i in chosen_indices[5:]]

    # 全图已知，无未探索前沿
    seen = np.ones((rows, cols), dtype=bool)

    teacher = MCMFTeacher(
        max_depth=20,
        dynamic_congestion=True,
        congestion_alpha=10.0,
        congestion_beta=1.5,
    )
    obs = {
        "size": (rows, cols),
        "mine": ant_positions,
        "food": food_positions,
        "ids": list(range(5)),
    }

    orders = teacher.orders(obs, water, seen)
    tasks = teacher.get_last_tasks()

    # 验证修复后候选边不被过早截断，蚂蚁 3 成功获配 GATHER 任务且不发生停滞 HOLD 动作
    assert tasks[3] == TaskState.GATHER
    assert orders[3] != HOLD


def test_mcmf_long_horizon_multi_ratio_conflict_free():
    rows, cols = 64, 64
    num_ants = 40
    num_foods = 20

    water, ant_positions, food_positions, seen = make_torus_test_grid(
        rows, cols, num_ants, num_foods, seed=1024
    )

    teacher = MCMFTeacher(
        max_depth=16,
        dynamic_congestion=True,
        congestion_alpha=10.0,
        congestion_beta=1.5,
    )

    current_ants = list(ant_positions)
    current_ids = list(range(num_ants))
    deltas = [(-1, 0), (0, 1), (1, 0), (0, -1), (0, 0)]

    for step in range(30):
        obs = {
            "size": (rows, cols),
            "mine": current_ants,
            "food": food_positions,
            "ids": current_ids,
        }

        orders = teacher.orders(obs, water, seen)
        assert len(orders) == num_ants

        next_ants: list[tuple[int, int]] = []
        for i, (ar, ac) in enumerate(current_ants):
            action = int(orders[i])
            dr, dc = deltas[action]
            nr = (ar + dr) % rows
            nc = (ac + dc) % cols

            assert not water[nr, nc], f"第 {step} 回合蚂蚁 {i} 踩入水域 ({nr}, {nc})"
            assert (nr, nc) not in food_positions, f"第 {step} 回合蚂蚁 {i} 踩踏食物 ({nr}, {nc})"
            next_ants.append((nr, nc))

        # 验证全局零碰撞
        assert len(set(next_ants)) == len(next_ants), (
            f"第 {step} 回合发生友方相撞冲突，活跃个体数 {len(next_ants)}，唯一位置数 {len(set(next_ants))}"
        )
        current_ants = next_ants


def test_mcmf_dynamic_congestion_chokepoint_diversion():
    rows, cols = 40, 50
    water = np.ones((rows, cols), dtype=bool)
    entrance = (15, 10)
    exit_tile = (15, 35)

    water[entrance[0], entrance[1]] = False
    water[exit_tile[0], exit_tile[1]] = False

    # 短窄走廊: 行 10 (长度 35 步, 宽度 1)
    narrow_row = 10
    for c in range(entrance[1] + 1, exit_tile[1]):
        water[narrow_row, c] = False
    for r in range(narrow_row, entrance[0] + 1):
        water[r, entrance[1]] = False
        water[r, exit_tile[1]] = False

    # 长宽走廊: 行 24..26 (宽度 3, 长度 43 步)
    wide_row = 24
    for c in range(entrance[1] + 1, exit_tile[1]):
        for w in range(3):
            water[wide_row + w, c] = False
    for r in range(entrance[0], wide_row + 3):
        water[r, entrance[1]] = False
        water[r, exit_tile[1]] = False

    # 终点放置 5 颗食物
    foods = []
    for i in range(5):
        fr = 13 + i
        water[fr, 35:39] = False
        foods.append((fr, 38))

    seen = np.ones((rows, cols), dtype=bool)

    # 场景 1: 空载状态，决策蚂蚁选择短窄走廊 (North, 0)
    teacher_empty = MCMFTeacher(max_depth=60, dynamic_congestion=True)
    obs_empty = {
        "size": (rows, cols),
        "mine": [entrance],
        "food": foods,
        "ids": [0],
    }
    orders_empty = teacher_empty.orders(obs_empty, water, seen)
    assert orders_empty[0] == 0

    # 场景 2: 窄走廊中布置 4 只前导蚂蚁造成严重拥堵，决策蚂蚁翻转选择长宽走廊 (South, 2)
    narrow_ants = [(narrow_row, entrance[1] + 3 + i * 3) for i in range(4)]
    all_ants = [entrance] + narrow_ants
    teacher_congested = MCMFTeacher(max_depth=60, dynamic_congestion=True)
    obs_congested = {
        "size": (rows, cols),
        "mine": all_ants,
        "food": foods,
        "ids": list(range(len(all_ants))),
    }
    orders_congested = teacher_congested.orders(obs_congested, water, seen)
    assert orders_congested[0] == 2
    assert teacher_congested.last_congestion_penalty_matrix is not None


def test_mcmf_latency_scaling_and_predictability():
    benchmark_scales = [
        (30, 30, 5, 5, 6.0),
        (40, 40, 20, 10, 10.0),
        (64, 64, 50, 25, 15.0),
        (80, 80, 100, 50, 20.0),
    ]

    measured_latencies: dict[tuple[int, int], float] = {}

    for rows, cols, num_ants, num_foods, max_allowed_ms in benchmark_scales:
        water, ant_positions, food_positions, seen = make_torus_test_grid(
            rows, cols, num_ants, num_foods, seed=888
        )

        teacher = MCMFTeacher(
            max_depth=16,
            dynamic_congestion=True,
            congestion_alpha=10.0,
            congestion_beta=1.5,
        )
        obs = {
            "size": (rows, cols),
            "mine": ant_positions,
            "food": food_positions,
            "ids": list(range(num_ants)),
        }

        teacher.orders(obs, water, seen)
        gc.collect()

        latencies: list[float] = []
        gc.disable()
        try:
            for _ in range(5):
                t0 = time.perf_counter()
                orders = teacher.orders(obs, water, seen)
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                latencies.append(elapsed_ms)
                assert len(orders) == num_ants
        finally:
            gc.enable()

        avg_lat = float(np.mean(latencies))
        measured_latencies[(num_ants, num_foods)] = avg_lat

        assert avg_lat < max_allowed_ms, (
            f"规模 ({num_ants} 蚂蚁, {num_foods} 食物) 单回合平均耗时 {avg_lat:.2f} ms 超出限制 {max_allowed_ms:.1f} ms"
        )

    lat_50 = measured_latencies[(50, 25)]
    lat_100 = measured_latencies[(100, 50)]

    assert lat_50 < 15.0
    assert lat_100 < 20.0
