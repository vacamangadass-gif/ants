from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tb_baselines import planes
from tb_baselines.teacher import DELTAS, HOLD, CompoundTeacher, Teacher


def rle_from_mask(mask: np.ndarray) -> list[int]:
    flat = mask.ravel().astype(np.int8)
    if len(flat) == 0:
        return []
    diffs = np.diff(flat)
    change_indices = np.where(diffs != 0)[0] + 1
    split_points = [0] + change_indices.tolist() + [len(flat)]
    rle: list[int] = []
    for start, end in zip(split_points[:-1], split_points[1:]):
        rle.extend([int(flat[start]), int(end - start)])
    return rle


def create_stress_board(
    rows: int,
    cols: int,
    n_ants: int,
    water_density: float,
    n_food: int,
    n_enemy_hills: int,
    seed: int,
    include_memory: bool = True,
) -> tuple[dict, np.ndarray, np.ndarray, dict | None]:
    rng = np.random.default_rng(seed)
    b = planes.Board(rows, cols)

    if water_density <= 0.0:
        water = np.zeros((rows, cols), dtype=bool)
    else:
        raw = rng.random((rows, cols))
        if min(rows, cols) >= 10:
            smoothed = (
                raw
                + np.roll(raw, 1, 0)
                + np.roll(raw, -1, 0)
                + np.roll(raw, 1, 1)
                + np.roll(raw, -1, 1)
            ) / 5.0
            water = smoothed < np.percentile(smoothed, water_density * 100)
        else:
            water = raw < water_density

    land_cells = [(r, c) for r in range(rows) for c in range(cols) if not water[r, c]]
    if len(land_cells) < n_ants + n_food + n_enemy_hills + 5:
        for p in land_cells[: n_ants + n_food + n_enemy_hills + 10]:
            water[p[0], p[1]] = False
        land_cells = [
            (r, c) for r in range(rows) for c in range(cols) if not water[r, c]
        ]

    rng.shuffle(land_cells)
    ant_positions = land_cells[:n_ants]
    mine = [list(p) for p in ant_positions]
    remaining_land = land_cells[n_ants:]

    n_food_actual = min(n_food, len(remaining_land) // 2)
    food_positions = remaining_land[:n_food_actual]
    food = [list(p) for p in food_positions]
    remaining_land = remaining_land[n_food_actual:]

    n_hills_actual = min(n_enemy_hills, len(remaining_land))
    hills_positions = remaining_land[:n_hills_actual]
    hills = [[r, c, 1] for r, c in hills_positions]

    mine_plane = b.scatter(mine) if n_ants > 0 else b.zeros()
    vis_plane = b.dilate(mine_plane, planes.VIEW_RADIUS2) if n_ants > 0 else b.zeros()
    seen = water | vis_plane.astype(bool)

    obs = {
        "size": [rows, cols],
        "mine": mine,
        "ids": list(range(n_ants)),
        "foes": [],
        "food": food,
        "hills": hills,
        "water": {"rle": rle_from_mask(water)},
        "vis": {"rle": rle_from_mask(vis_plane)},
    }

    memory = None
    if include_memory:
        memory = {}
        mem_hills = [[r, c] for r, c, _ in hills]
        if remaining_land:
            fog_p = remaining_land[rng.integers(len(remaining_land))]
            mem_hills.append(list(fog_p))
        memory["hill_foe_seen"] = mem_hills
        mem_food = [list(p) for p in food_positions[: len(food_positions) // 2]]
        memory["food_seen"] = mem_food

    return obs, water, seen, memory


def create_corridor_bottleneck_board() -> tuple[dict, np.ndarray, np.ndarray, dict | None]:
    rows, cols = 32, 32
    water = np.zeros((rows, cols), dtype=bool)
    water[15, 4:28] = True
    water[17, 4:28] = True
    mine = [[16, c] for c in range(6, 18)]
    food = [[16, 26]]
    hills = [[16, 27, 1]]
    b = planes.Board(rows, cols)
    mine_plane = b.scatter(mine)
    vis_plane = b.dilate(mine_plane, planes.VIEW_RADIUS2)
    seen = water | vis_plane.astype(bool)
    obs = {
        "size": [rows, cols],
        "mine": mine,
        "ids": list(range(len(mine))),
        "foes": [],
        "food": food,
        "hills": hills,
        "water": {"rle": rle_from_mask(water)},
        "vis": {"rle": rle_from_mask(vis_plane)},
    }
    return obs, water, seen, None


def test_single_instance_compound_teacher_100_repeated_calls():
    ct = CompoundTeacher(min_raid_ants=4, raid_ratio=0.5, max_depth=32, hill_lead=6)

    test_boards = [
        create_stress_board(24, 24, 6, 0.10, 4, 1, seed=1001, include_memory=True),
        create_stress_board(36, 40, 16, 0.20, 8, 2, seed=1002, include_memory=True),
        create_stress_board(48, 48, 24, 0.25, 12, 2, seed=1003, include_memory=False),
        create_corridor_bottleneck_board(),
    ]

    for b_idx, (obs, water, seen, memory) in enumerate(test_boards):
        first_orders = ct.orders(obs, water, seen, memory)
        first_act = np.asarray(ct.act(obs, memory), dtype=np.int64)
        assert np.array_equal(first_orders, first_act)

        for step in range(100):
            current_orders = ct.orders(obs, water, seen, memory)
            assert np.array_equal(first_orders, current_orders), (
                f"Board {b_idx} iteration {step}: CompoundTeacher.orders 输出不一致"
            )

            current_act = np.asarray(ct.act(obs, memory), dtype=np.int64)
            assert np.array_equal(first_orders, current_act), (
                f"Board {b_idx} iteration {step}: CompoundTeacher.act 输出不一致"
            )

            telem_orders, telem_records = ct.orders_with_telemetry(obs, water, seen, memory)
            assert np.array_equal(first_orders, telem_orders), (
                f"Board {b_idx} iteration {step}: orders_with_telemetry 输出不一致"
            )
            assert len(telem_records) == len(obs["mine"])


def test_single_instance_teacher_100_repeated_calls():
    teacher = Teacher(hill_lead=6, max_depth=32)

    test_boards = [
        create_stress_board(24, 24, 8, 0.12, 5, 1, seed=2001, include_memory=True),
        create_stress_board(40, 40, 18, 0.18, 10, 2, seed=2002, include_memory=True),
        create_corridor_bottleneck_board(),
    ]

    for b_idx, (obs, water, seen, memory) in enumerate(test_boards):
        first_orders = teacher.orders(obs, water, seen, memory)

        for step in range(100):
            current_orders = teacher.orders(obs, water, seen, memory)
            assert np.array_equal(first_orders, current_orders), (
                f"Board {b_idx} iteration {step}: Teacher.orders 输出不一致"
            )

            telem_orders, telem_records = teacher.orders_with_telemetry(obs, water, seen, memory)
            assert np.array_equal(first_orders, telem_orders), (
                f"Board {b_idx} iteration {step}: Teacher orders_with_telemetry 输出不一致"
            )
            assert len(telem_records) == len(obs["mine"])


def test_interleaved_calls_state_isolation_compound_teacher():
    boards = [
        create_stress_board(26, 26, 6, 0.08, 4, 1, seed=3001, include_memory=True),
        create_stress_board(32, 32, 12, 0.15, 6, 2, seed=3002, include_memory=False),
        create_stress_board(44, 44, 20, 0.22, 10, 2, seed=3003, include_memory=True),
        create_corridor_bottleneck_board(),
        create_stress_board(28, 36, 14, 0.18, 8, 1, seed=3004, include_memory=True),
    ]

    references: list[np.ndarray] = []
    for obs, water, seen, memory in boards:
        fresh_ct = CompoundTeacher(min_raid_ants=4, raid_ratio=0.5, max_depth=32, hill_lead=6)
        ref_orders = fresh_ct.orders(obs, water, seen, memory)
        references.append(ref_orders)

    single_persistent_ct = CompoundTeacher(
        min_raid_ants=4, raid_ratio=0.5, max_depth=32, hill_lead=6
    )

    rng = np.random.default_rng(7771)
    call_sequence = rng.integers(0, len(boards), size=500)

    for step_idx, board_idx in enumerate(call_sequence):
        obs, water, seen, memory = boards[board_idx]
        actual_orders = single_persistent_ct.orders(obs, water, seen, memory)
        expected_orders = references[board_idx]
        assert np.array_equal(actual_orders, expected_orders), (
            f"Step {step_idx} (Board {board_idx}): 状态残留导致动作输出偏离独立基准"
        )


def test_interleaved_calls_state_isolation_teacher():
    boards = [
        create_stress_board(24, 28, 5, 0.10, 3, 1, seed=4001, include_memory=True),
        create_stress_board(36, 36, 15, 0.16, 7, 2, seed=4002, include_memory=False),
        create_stress_board(42, 42, 22, 0.24, 9, 2, seed=4003, include_memory=True),
        create_corridor_bottleneck_board(),
        create_stress_board(30, 30, 10, 0.14, 5, 1, seed=4004, include_memory=True),
    ]

    references: list[np.ndarray] = []
    for obs, water, seen, memory in boards:
        fresh_teacher = Teacher(hill_lead=6, max_depth=32)
        ref_orders = fresh_teacher.orders(obs, water, seen, memory)
        references.append(ref_orders)

    single_persistent_teacher = Teacher(hill_lead=6, max_depth=32)

    rng = np.random.default_rng(8881)
    call_sequence = rng.integers(0, len(boards), size=500)

    for step_idx, board_idx in enumerate(call_sequence):
        obs, water, seen, memory = boards[board_idx]
        actual_orders = single_persistent_teacher.orders(obs, water, seen, memory)
        expected_orders = references[board_idx]
        assert np.array_equal(actual_orders, expected_orders), (
            f"Step {step_idx} (Board {board_idx}): 状态残留导致 Teacher 动作输出偏离独立基准"
        )


def test_explicit_stall_counter_isolation():
    ct = CompoundTeacher(min_raid_ants=4, raid_ratio=0.5, max_depth=32, hill_lead=6)
    obs, water, seen, memory = create_corridor_bottleneck_board()

    fresh_ref = CompoundTeacher(min_raid_ants=4, raid_ratio=0.5, max_depth=32, hill_lead=6)
    ref_default_orders = fresh_ref.orders(obs, water, seen, memory)

    external_stalls: dict[int, int] = {}
    for _ in range(10):
        ct.orders(obs, water, seen, memory, stall_counters=external_stalls)

    assert len(external_stalls) > 0, "走廊竞争场景应产生外部受阻计数"

    subsequent_default_orders = ct.orders(obs, water, seen, memory)
    assert np.array_equal(subsequent_default_orders, ref_default_orders), (
        "显式传递外部受阻字典后，随后的默认调用受阻状态发生污染"
    )

    fresh_default_orders = fresh_ref.orders(obs, water, seen, memory)
    assert np.array_equal(subsequent_default_orders, fresh_default_orders)


def test_interleaved_telemetry_toggling_isolation():
    ct = CompoundTeacher(min_raid_ants=4, raid_ratio=0.5, max_depth=32, hill_lead=6)
    obs, water, seen, memory = create_stress_board(30, 30, 10, 0.15, 6, 1, seed=5001)

    ref_orders = ct.orders(obs, water, seen, memory)

    for iteration in range(50):
        telem_orders, records = ct.orders_with_telemetry(obs, water, seen, memory)
        assert np.array_equal(ref_orders, telem_orders)
        assert len(records) == len(obs["mine"])

        plain_orders = ct.orders(obs, water, seen, memory)
        assert np.array_equal(ref_orders, plain_orders)
