from __future__ import annotations

import numpy as np
import pytest

from tb_baselines.teacher import (
    DELTAS,
    FAR,
    HOLD,
    CompoundTeacher,
    Teacher,
    arbitrate_orders,
    known,
)
from tb_baselines.telemetry import (
    HoldReason,
    RejectionCode,
    TelemetryRecord,
)


def make_board(
    size: int = 24,
    ants: tuple[tuple[int, int], ...] = ((12, 12),),
    hills: tuple[tuple[int, int, int], ...] = (),
    food: tuple[tuple[int, int], ...] = (),
    water_coords: tuple[tuple[int, int], ...] = (),
    visible_coords: tuple[tuple[int, int], ...] | None = None,
    ids: list[int] | None = None,
) -> dict:
    rows, cols = size, size
    water_mask = np.zeros((rows, cols), dtype=bool)
    for r, c in water_coords:
        if 0 <= r < rows and 0 <= c < cols:
            water_mask[r, c] = True

    seen_mask = np.ones((rows, cols), dtype=bool)
    if visible_coords is not None:
        seen_mask.fill(False)
        for r, c in visible_coords:
            if 0 <= r < rows and 0 <= c < cols:
                seen_mask[r, c] = True

    def to_rle(mask: np.ndarray) -> dict:
        flat = mask.flatten()
        diffs = np.diff(flat.astype(np.int8))
        change_indices = np.where(diffs != 0)[0] + 1
        starts = np.insert(change_indices, 0, 0)
        ends = np.append(change_indices, len(flat))
        runs = []
        for s, e in zip(starts, ends):
            runs.extend([int(flat[s]), int(e - s)])
        return {"rle": runs}

    if ids is None:
        ids = list(range(len(ants)))

    return {
        "size": [rows, cols],
        "mine": [list(a) for a in ants],
        "ids": list(ids),
        "foes": [],
        "food": [list(f) for f in food],
        "hills": [list(h) for h in hills],
        "water": to_rle(water_mask),
        "vis": to_rle(seen_mask),
    }


def compute_destinations(mine: list[list[int]] | list[tuple[int, int]], orders: np.ndarray, rows: int, cols: int) -> list[tuple[int, int]]:
    destinations = []
    for (r, c), act in zip(mine, orders):
        if act == HOLD:
            destinations.append((int(r), int(c)))
        else:
            dr, dc = DELTAS[int(act)]
            destinations.append(((int(r) + dr) % rows, (int(c) + dc) % cols))
    return destinations


# ------------------------------------------------------------------------------
# 测试集 1：特化微观场景验证
# ------------------------------------------------------------------------------


def test_corridor_queue_advance_single_turn_no_lag():
    rows, cols = 20, 20
    water_coords = []
    for c in range(cols):
        water_coords.append((4, c))
        water_coords.append((6, c))

    ants = ((5, 6), (5, 5), (5, 4))
    food = ((5, 10),)
    obs = make_board(size=20, ants=ants, food=food, water_coords=tuple(water_coords))

    teacher = Teacher()
    water, seen = known(obs)
    orders = teacher.orders(obs, water, seen)

    assert len(orders) == 3
    assert orders[0] == 1
    assert orders[1] == 1
    assert orders[2] == 1

    dests = compute_destinations(obs["mine"], orders, rows, cols)
    assert len(set(dests)) == 3
    assert dests == [(5, 7), (5, 6), (5, 5)]


def test_corridor_sovereign_hold_prevents_intrusion():
    rows, cols = 20, 20
    water_coords = []
    for c in range(cols):
        water_coords.append((4, c))
        water_coords.append((6, c))

    ants = ((5, 5), (5, 4))
    food = ((5, 6),)
    obs = make_board(size=20, ants=ants, food=food, water_coords=tuple(water_coords))

    teacher = Teacher()
    water, seen = known(obs)
    orders, telemetry = teacher.orders_with_telemetry(obs, water, seen)

    assert orders[0] == HOLD
    assert orders[1] == HOLD

    dests = compute_destinations(obs["mine"], orders, rows, cols)
    assert dests == [(5, 5), (5, 4)]
    assert len(set(dests)) == 2

    assert len(telemetry) == 2
    assert isinstance(telemetry[0], TelemetryRecord)
    assert telemetry[0].potential_self < FAR
    assert telemetry[0].hold_reason == HoldReason.HARVEST_WAIT.value
    assert telemetry[0].is_intentional_hold is True
    assert telemetry[1].hold_reason == HoldReason.CONFLICT_WAIT.value
    assert telemetry[1].candidate_rejections.get(1) == RejectionCode.TAKEN.value


def test_vacated_cell_entry_seamless_handoff():
    rows, cols = 20, 20
    water_coords = []
    for r in range(rows):
        for c in range(cols):
            if not ((r == 5 and c in (3, 4, 5)) or (c == 5 and r in (3, 4, 5))):
                water_coords.append((r, c))

    ants = ((5, 5), (5, 4))
    food = ((2, 5),)
    obs = make_board(size=20, ants=ants, food=food, water_coords=tuple(water_coords))

    teacher = Teacher()
    water, seen = known(obs)
    orders = teacher.orders(obs, water, seen)

    assert orders[0] == 0
    assert orders[1] == 1

    dests = compute_destinations(obs["mine"], orders, rows, cols)
    assert dests == [(4, 5), (5, 5)]
    assert len(set(dests)) == 2


def test_torus_wrapping_queue_advance():
    rows, cols = 20, 20
    water_coords = []
    for r in range(rows):
        water_coords.append((r, 9))
        water_coords.append((r, 11))

    ants = ((0, 10), (19, 10), (18, 10))
    food = ((4, 10),)
    obs = make_board(size=20, ants=ants, food=food, water_coords=tuple(water_coords))

    teacher = Teacher()
    water, seen = known(obs)
    orders = teacher.orders(obs, water, seen)

    assert orders[0] == 2
    assert orders[1] == 2
    assert orders[2] == 2

    dests = compute_destinations(obs["mine"], orders, rows, cols)
    assert dests == [(1, 10), (0, 10), (19, 10)]
    assert len(set(dests)) == 3


def test_head_on_swap_and_bottleneck_alternation():
    mine = [(5, 4), (5, 5)]
    candidate_sequences = [[1, HOLD], [3, HOLD]]
    potentials = [5, 5]
    stall_counters = {0: 0, 1: 0}

    orders, updated_stalls = arbitrate_orders(
        mine_coords=mine,
        candidate_sequences=candidate_sequences,
        potentials=potentials,
        stall_counters=stall_counters,
        fair_mode=True,
        rows=20,
        cols=20,
    )

    assert orders[0] == 1
    assert orders[1] == 3
    dests = compute_destinations(mine, orders, 20, 20)
    assert dests == [(5, 5), (5, 4)]
    assert len(set(dests)) == 2


def test_deterministic_priority_resolution_empty_cell():
    mine = [(4, 5), (5, 4), (6, 5)]
    candidate_sequences = [[2, HOLD], [1, HOLD], [0, HOLD]]

    potentials_1 = [10, 4, 8]
    stalls_1 = {0: 0, 1: 0, 2: 0}
    orders_1, _ = arbitrate_orders(mine, candidate_sequences, potentials_1, stalls_1, fair_mode=True, rows=20, cols=20)
    assert orders_1[1] == 1
    assert orders_1[0] == HOLD
    assert orders_1[2] == HOLD

    potentials_2 = [5, 5, 5]
    stalls_2 = {0: 0, 1: 0, 2: 0}
    orders_2, _ = arbitrate_orders(mine, candidate_sequences, potentials_2, stalls_2, fair_mode=True, rows=20, cols=20)
    assert orders_2[0] == 2
    assert orders_2[1] == HOLD
    assert orders_2[2] == HOLD

    potentials_3 = [2, 2, 10]
    stalls_3 = {0: 0, 1: 0, 2: 3}
    orders_3, _ = arbitrate_orders(mine, candidate_sequences, potentials_3, stalls_3, fair_mode=True, rows=20, cols=20)
    assert orders_3[2] == 0
    assert orders_3[0] == HOLD
    assert orders_3[1] == HOLD


def test_starvation_elimination_dynamic_priority():
    mine = [(5, 4), (5, 6)]
    candidate_sequences = [[1, HOLD], [3, HOLD]]
    potentials = [2, 5]
    stalls = {0: 0, 1: 0}

    orders_t1, stalls_t1 = arbitrate_orders(mine, candidate_sequences, potentials, stalls, fair_mode=True, rows=20, cols=20)
    assert orders_t1[0] == 1
    assert orders_t1[1] == HOLD
    assert stalls_t1[0] == 0
    assert stalls_t1[1] == 1

    orders_t2, stalls_t2 = arbitrate_orders(mine, candidate_sequences, potentials, stalls_t1, fair_mode=True, rows=20, cols=20)
    assert orders_t2[0] == HOLD
    assert orders_t2[1] == 3
    assert stalls_t2[0] == 1
    assert stalls_t2[1] == 0


# ------------------------------------------------------------------------------
# 测试集 2：量化不变量验证
# ------------------------------------------------------------------------------


def test_invariant_1_destination_uniqueness_zero_collisions():
    rng = np.random.RandomState(42)
    rows, cols = 24, 24

    for _ in range(30):
        num_ants = int(rng.randint(10, 40))
        all_cells = [(r, c) for r in range(rows) for c in range(cols)]
        chosen_indices = rng.choice(len(all_cells), size=num_ants + 5, replace=False)
        ant_coords = [all_cells[i] for i in chosen_indices[:num_ants]]
        food_coords = [all_cells[i] for i in chosen_indices[num_ants:]]

        obs = make_board(size=24, ants=tuple(ant_coords), food=tuple(food_coords))
        teacher = Teacher()
        water, seen = known(obs)
        orders = teacher.orders(obs, water, seen)

        dests = compute_destinations(ant_coords, orders, rows, cols)
        assert len(dests) == num_ants
        assert len(set(dests)) == num_ants


def test_invariant_2_sovereign_hold_100_percent_preservation():
    rng = np.random.RandomState(101)
    rows, cols = 20, 20

    for _ in range(20):
        num_ants = int(rng.randint(6, 20))
        all_cells = [(r, c) for r in range(rows) for c in range(cols)]
        chosen = rng.choice(len(all_cells), size=num_ants, replace=False)
        mine = [all_cells[i] for i in chosen]

        candidates = []
        for i in range(num_ants):
            if i % 3 == 0:
                candidates.append([HOLD])
            else:
                candidates.append([int(rng.randint(0, 4)), HOLD])

        potentials = [int(rng.randint(1, 20)) for _ in range(num_ants)]
        stalls = {i: int(rng.randint(0, 3)) for i in range(num_ants)}

        orders, _ = arbitrate_orders(mine, candidates, potentials, stalls, fair_mode=True, rows=rows, cols=cols)
        dests = compute_destinations(mine, orders, rows, cols)

        for i in range(num_ants):
            if orders[i] == HOLD:
                assert dests[i] == mine[i]
                for j in range(num_ants):
                    if j != i:
                        assert dests[j] != mine[i]


def test_invariant_3_arbitration_convergence_under_4n_iterations():
    rows, cols = 30, 30
    num_ants = 25
    mine = [(r, 15) for r in range(num_ants)]
    candidate_sequences = [[2, 1, 3, HOLD] for _ in range(num_ants)]
    potentials = [i for i in range(num_ants)]
    stalls = {i: 0 for i in range(num_ants)}

    orders, _, iteration_count = arbitrate_orders(
        mine_coords=mine,
        candidate_sequences=candidate_sequences,
        potentials=potentials,
        stall_counters=stalls,
        fair_mode=True,
        rows=rows,
        cols=cols,
        return_iteration_count=True,
    )

    assert iteration_count <= 4 * num_ants
    dests = compute_destinations(mine, orders, rows, cols)
    assert len(set(dests)) == num_ants

    with pytest.raises(AssertionError):
        arbitrate_orders(
            mine_coords=[(5, 4), (5, 6)],
            candidate_sequences=[[1, HOLD], [3]],
            potentials=[2, 5],
        )


def test_invariant_4_stall_counter_bounded_no_permanent_stagnation():
    mine = [(5, 4), (5, 6)]
    candidate_sequences = [[1, HOLD], [3, HOLD]]
    potentials = [2, 5]
    stalls = {0: 0, 1: 0}

    max_stall_observed = 0
    for turn in range(10):
        orders, stalls = arbitrate_orders(
            mine_coords=mine,
            candidate_sequences=candidate_sequences,
            potentials=potentials,
            stall_counters=stalls,
            fair_mode=True,
            rows=20,
            cols=20,
        )
        for val in stalls.values():
            if val > max_stall_observed:
                max_stall_observed = val

    assert max_stall_observed <= 1


def test_compound_teacher_arbitration_phase_b_no_collisions():
    obs = make_board(
        size=24,
        ants=((10, 10), (10, 12), (10, 14), (10, 16)),
        hills=((10, 5, 1),),
        food=((10, 20),),
    )
    water, seen = known(obs)
    ct = CompoundTeacher(min_raid_ants=4, raid_ratio=0.5)
    orders = ct.orders(obs, water, seen)
    assert len(orders) == 4
    dests = compute_destinations(obs["mine"], orders, 24, 24)
    assert len(set(dests)) == 4
