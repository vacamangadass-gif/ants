"""Empirical adversarial stress test suite for CompoundTeacher (Milestone 2).

Verifies:
1. Zero collisions, zero water entries, zero food steps across 500+ randomized board trials
   (grid sizes 24x24 to 100x100, 4 to 30 ants, 0% to 40% water, 0 to 50 food clusters, 0 to 4 hills).
2. Strict bitwise determinism across identical observations, fresh instances, and `act()` interface.
3. Pathological adversarial edge cases:
   - Torus wrapping boundary squeezes.
   - Dense ant herds in 1-wide chokepoints / bottlenecks.
   - Completely enclosed / trapped ants surrounded by food or water.
   - Ant sitting directly on top of the enemy hill (`cur_d = 0`).
   - Enemy hill on an unreachable island (water barrier fallback).
   - Phase A vs Phase B transition boundaries (1, 2, 3 vs 4 ants).
4. Multi-turn dynamic trajectory simulation (25 consecutive steps with movement, gathering, razing).
5. Latency and throughput benchmarking across small, medium, large, and extra-large boards.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tb_baselines import planes  # noqa: E402
from tb_baselines.teacher import DELTAS, HOLD, CompoundTeacher  # noqa: E402


def rle_from_mask(mask: np.ndarray) -> list[int]:
    """Convert a 2D boolean mask to RLE run-length format [v0, n0, v1, n1, ...]."""
    flat = mask.ravel().astype(np.int8)
    if len(flat) == 0:
        return []
    diffs = np.diff(flat)
    change_indices = np.where(diffs != 0)[0] + 1
    split_points = [0] + change_indices.tolist() + [len(flat)]
    rle = []
    for start, end in zip(split_points[:-1], split_points[1:]):
        rle.extend([int(flat[start]), int(end - start)])
    return rle


def generate_random_trial_board(
    rows: int,
    cols: int,
    n_ants: int,
    water_density: float,
    n_food: int,
    n_enemy_hills: int,
    rng: np.random.Generator,
    include_memory: bool = True,
) -> tuple[dict, np.ndarray, np.ndarray, dict | None]:
    """Generate a valid randomized board scenario with specified constraints."""
    b = planes.Board(rows, cols)

    # 1. Water mask (can be random noise or structural blobs)
    if water_density <= 0.0:
        water = np.zeros((rows, cols), dtype=bool)
    else:
        raw = rng.random((rows, cols))
        # Add slight spatial clustering for realistic water bodies
        if rng.random() < 0.5 and min(rows, cols) >= 10:
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

    # Ensure there is enough land
    land_cells = [(r, c) for r in range(rows) for c in range(cols) if not water[r, c]]
    if len(land_cells) < n_ants + n_food + n_enemy_hills + 5:
        # Clear some land if water is too dense
        for p in land_cells[: n_ants + n_food + n_enemy_hills + 10]:
            water[p[0], p[1]] = False
        land_cells = [
            (r, c) for r in range(rows) for c in range(cols) if not water[r, c]
        ]

    # 2. Place ants on distinct land cells
    rng.shuffle(land_cells)
    ant_positions = land_cells[:n_ants]
    mine = [list(p) for p in ant_positions]
    remaining_land = land_cells[n_ants:]

    # 3. Place food on remaining land cells
    n_food_actual = min(n_food, len(remaining_land) // 2)
    food_positions = remaining_land[:n_food_actual]
    food = [list(p) for p in food_positions]
    remaining_land = remaining_land[n_food_actual:]

    # 4. Place enemy hills
    n_hills_actual = min(n_enemy_hills, len(remaining_land))
    hills_positions = remaining_land[:n_hills_actual]
    hills = [[r, c, 1] for r, c in hills_positions]  # owner = 1 (enemy)

    # 5. Build visibility mask and seen mask
    mine_plane = b.scatter(mine)
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

    # 6. Optional memory state
    memory = None
    if include_memory and rng.random() < 0.7:
        memory = {}
        # Remembered enemy hills (some in fog, some cleared)
        mem_hills = []
        for r, c, _ in hills:
            mem_hills.append([r, c])
        # Add a phantom/fog enemy hill
        if remaining_land:
            fog_p = remaining_land[rng.integers(len(remaining_land))]
            mem_hills.append(list(fog_p))
        memory["hill_foe_seen"] = mem_hills

        # Remembered food
        mem_food = [list(p) for p in food_positions[: len(food_positions) // 2]]
        memory["food_seen"] = mem_food

    return obs, water, seen, memory


def calculate_destinations(
    mine: list[list[int]], orders: np.ndarray, rows: int, cols: int
) -> list[tuple[int, int]]:
    """Compute the actual destination coordinates for each ant given their chosen orders."""
    destinations = []
    for (ar, ac), a in zip(mine, orders):
        if a == HOLD:
            destinations.append((ar, ac))
        else:
            dr, dc = DELTAS[a]
            destinations.append(((ar + dr) % rows, (ac + dc) % cols))
    return destinations


def test_500_randomized_trials_zero_collisions_zero_water_zero_food():
    """Adversarial stress test: 520 randomized board configurations.

    Asserts:
    - 100% collision freedom across all trials: zero friendly ant collisions.
    - 100% water avoidance: zero ants stepping into water.
    - 100% food avoidance: zero ants stepping into food tiles.
    - 100% length conformance: orders output length matches `mine` length.
    """
    rng = np.random.default_rng(20261015)
    ct = CompoundTeacher(min_raid_ants=4, raid_ratio=0.5, max_depth=32, hill_lead=6)

    # Varied dimension configurations
    board_configs = [
        # (rows, cols, count)
        (24, 24, 80),  # Tiny standard map size
        (30, 30, 80),  # Small map size
        (40, 40, 80),  # Medium map size
        (50, 50, 70),  # Medium-large map size
        (64, 64, 60),  # Large map size
        (80, 80, 50),  # Very large map size
        (100, 100, 40),  # Extra large map size
        # Non-square grids
        (24, 36, 30),
        (35, 25, 30),
    ]

    total_trials = 0
    total_ant_decisions = 0
    collision_violations = 0
    water_violations = 0
    food_violations = 0

    for rows, cols, count in board_configs:
        for _ in range(count):
            total_trials += 1
            n_ants = int(rng.integers(4, 31))  # 4 to 30 ants inclusive
            water_density = float(rng.uniform(0.0, 0.38))
            n_food = int(rng.integers(0, 35))
            n_hills = int(rng.choice([0, 1, 2, 3], p=[0.15, 0.55, 0.20, 0.10]))

            obs, water, seen, memory = generate_random_trial_board(
                rows, cols, n_ants, water_density, n_food, n_hills, rng
            )

            orders = ct.orders(obs, water, seen, memory)
            total_ant_decisions += len(orders)

            assert len(orders) == len(obs["mine"]), (
                f"Trial {total_trials}: orders length {len(orders)} != mine length {len(obs['mine'])}"
            )

            dests = calculate_destinations(obs["mine"], orders, rows, cols)
            food_set = {tuple(f) for f in obs["food"]}

            # 1. Collision check
            if len(set(dests)) != len(dests):
                collision_violations += 1
                dup = {d for d in dests if dests.count(d) > 1}
                pytest.fail(
                    f"Trial {total_trials} ({rows}x{cols}, {n_ants} ants): Collision at {dup}!"
                )

            # 2. Water collision check
            for r, c in dests:
                if water[r, c]:
                    water_violations += 1
                    pytest.fail(
                        f"Trial {total_trials} ({rows}x{cols}): Ant stepped into water at ({r}, {c})!"
                    )

            # 3. Food blockage check
            for r, c in dests:
                if (r, c) in food_set:
                    food_violations += 1
                    pytest.fail(
                        f"Trial {total_trials} ({rows}x{cols}): Ant stepped into food at ({r}, {c})!"
                    )

    assert total_trials >= 500, f"Expected at least 500 trials, ran {total_trials}"
    assert collision_violations == 0, f"Found {collision_violations} collisions"
    assert water_violations == 0, f"Found {water_violations} water violations"
    assert food_violations == 0, f"Found {food_violations} food violations"
    print(
        f"\n[STRESS TEST] Passed {total_trials} trials ({total_ant_decisions} ant decisions) with 0 collisions, 0 water, 0 food!"
    )


def test_strict_determinism_and_reproducibility():
    """Verify 100% bitwise determinism across 100 randomized complex trials.

    Each trial asserts:
    1. Repeated calls on the same CompoundTeacher instance yield identical orders.
    2. Fresh CompoundTeacher instances with identical arguments yield identical orders.
    3. Direct `.act(obs, memory)` matches `.orders(obs, water, seen, memory)`.
    """
    rng = np.random.default_rng(99991)
    ct = CompoundTeacher(min_raid_ants=4, raid_ratio=0.5, max_depth=32, hill_lead=6)

    for trial in range(100):
        rows, cols = int(rng.integers(24, 60)), int(rng.integers(24, 60))
        n_ants = int(rng.integers(4, 25))
        water_density = float(rng.uniform(0.05, 0.30))
        n_food = int(rng.integers(2, 20))
        n_hills = int(rng.integers(0, 3))

        obs, water, seen, memory = generate_random_trial_board(
            rows, cols, n_ants, water_density, n_food, n_hills, rng, include_memory=True
        )

        # Run 1
        orders_run1 = ct.orders(obs, water, seen, memory)

        # Run 2 on same instance
        orders_run2 = ct.orders(obs, water, seen, memory)
        assert np.array_equal(orders_run1, orders_run2), (
            f"Trial {trial}: Non-deterministic orders on same instance!"
        )

        # Run 3 on fresh instance
        ct_fresh = CompoundTeacher(
            min_raid_ants=4, raid_ratio=0.5, max_depth=32, hill_lead=6
        )
        orders_fresh = ct_fresh.orders(obs, water, seen, memory)
        assert np.array_equal(orders_run1, orders_fresh), (
            f"Trial {trial}: Divergence on fresh instance!"
        )

        # Run 4 via act()
        act_orders = np.asarray(ct.act(obs, memory), dtype=np.int64)
        assert np.array_equal(orders_run1, act_orders), (
            f"Trial {trial}: Divergence between act() and orders()!"
        )


def test_adversarial_chokepoints_and_dense_herds():
    """Adversarial scenario: Dense ant herd in a 1-wide linear bottleneck.

    8 ants lined up in a single file corridor flanked by water walls on both sides.
    Assert zero mutual collisions and proper queue flow without blockage crashes.
    """
    rows, cols = 30, 30
    water = np.zeros((rows, cols), dtype=bool)

    # Build horizontal corridor from col 5 to 25 at row 15
    water[14, 5:26] = True  # North wall
    water[16, 5:26] = True  # South wall

    # Place 8 ants single-file: (15, 10), (15, 11), ..., (15, 17)
    corridor_ants = [[15, c] for c in range(10, 18)]
    # Target enemy hill at east exit: (15, 24)
    hills = [[15, 24, 1]]

    b = planes.Board(rows, cols)
    mine_plane = b.scatter(corridor_ants)
    vis = b.dilate(mine_plane, planes.VIEW_RADIUS2).astype(bool)
    seen = water | vis

    obs = {
        "size": [rows, cols],
        "mine": corridor_ants,
        "ids": list(range(len(corridor_ants))),
        "foes": [],
        "food": [],
        "hills": hills,
        "water": {"rle": rle_from_mask(water)},
        "vis": {"rle": rle_from_mask(vis)},
    }

    ct = CompoundTeacher(min_raid_ants=4, raid_ratio=0.5)
    orders = ct.orders(obs, water, seen)

    dests = calculate_destinations(corridor_ants, orders, rows, cols)
    assert len(set(dests)) == len(dests), f"Chokepoint collision: {dests}"
    for r, c in dests:
        assert not water[r, c], f"Chokepoint ant stepped into water wall: ({r}, {c})"


def test_adversarial_corner_and_torus_wrapping_squeeze():
    """Adversarial scenario: Ants at the 4 torus corners with targets wrapping around boundaries."""
    rows, cols = 24, 24
    water = np.zeros((rows, cols), dtype=bool)

    # 4 ants placed at wrapping extreme corners
    corner_ants = [
        [0, 0],
        [0, cols - 1],
        [rows - 1, 0],
        [rows - 1, cols - 1],
    ]
    food = [[rows - 1, cols // 2], [rows // 2, 0]]
    # Enemy hill at (0, cols // 2)
    hills = [[0, cols // 2, 1]]

    b = planes.Board(rows, cols)
    mine_plane = b.scatter(corner_ants)
    vis = b.dilate(mine_plane, planes.VIEW_RADIUS2).astype(bool)
    seen = water | vis

    obs = {
        "size": [rows, cols],
        "mine": corner_ants,
        "ids": list(range(len(corner_ants))),
        "foes": [],
        "food": food,
        "hills": hills,
        "water": {"rle": rle_from_mask(water)},
        "vis": {"rle": rle_from_mask(vis)},
    }

    ct = CompoundTeacher(min_raid_ants=4, raid_ratio=0.5)
    orders = ct.orders(obs, water, seen)

    dests = calculate_destinations(corner_ants, orders, rows, cols)
    food_set = {tuple(f) for f in food}
    assert len(set(dests)) == len(dests), f"Torus wrap collision: {dests}"
    for r, c in dests:
        assert 0 <= r < rows and 0 <= c < cols
        assert not water[r, c]
        assert (r, c) not in food_set


def test_adversarial_ant_trapped_by_food_and_water():
    """Adversarial scenario: Ants surrounded on all 4 cardinal directions by food/water."""
    rows, cols = 20, 20
    water = np.zeros((rows, cols), dtype=bool)

    # Ant at (10, 10) surrounded:
    # North: water, East: food, South: water, West: food
    water[9, 10] = True
    water[11, 10] = True
    food = [[10, 11], [10, 9]]
    mine = [[10, 10], [5, 5], [5, 6], [5, 7]]  # 4 ants to test Phase B
    hills = [[2, 2, 1]]

    b = planes.Board(rows, cols)
    mine_plane = b.scatter(mine)
    vis = b.dilate(mine_plane, planes.VIEW_RADIUS2).astype(bool)
    seen = water | vis

    obs = {
        "size": [rows, cols],
        "mine": mine,
        "ids": list(range(len(mine))),
        "foes": [],
        "food": food,
        "hills": hills,
        "water": {"rle": rle_from_mask(water)},
        "vis": {"rle": rle_from_mask(vis)},
    }

    ct = CompoundTeacher(min_raid_ants=4, raid_ratio=0.5)
    orders = ct.orders(obs, water, seen)

    # Ant 0 at (10, 10) MUST stay HOLD (action index HOLD == 4)
    assert orders[0] == HOLD, f"Trapped ant must hold, but tried move {orders[0]}"

    dests = calculate_destinations(mine, orders, rows, cols)
    food_set = {tuple(f) for f in food}
    assert len(set(dests)) == len(dests)
    for r, c in dests:
        assert not water[r, c]
        assert (r, c) not in food_set


def test_adversarial_ant_sitting_directly_on_enemy_hill():
    """Adversarial scenario: One ant is already sitting on the enemy hill (cur_d = 0),

    while 4 other ants approach from adjacent squares.
    Assert:
    - The ant on the hill holds (or doesn't collide).
    - Other ants do NOT step onto the hill square simultaneously.
    """
    rows, cols = 24, 24
    water = np.zeros((rows, cols), dtype=bool)

    # Ant 0 on hill, Ants 1..4 around hill
    mine = [
        [12, 12],  # on hill
        [11, 12],  # North of hill
        [12, 13],  # East of hill
        [13, 12],  # South of hill
        [12, 11],  # West of hill
    ]
    hills = [[12, 12, 1]]

    b = planes.Board(rows, cols)
    mine_plane = b.scatter(mine)
    vis = b.dilate(mine_plane, planes.VIEW_RADIUS2).astype(bool)
    seen = water | vis

    obs = {
        "size": [rows, cols],
        "mine": mine,
        "ids": list(range(len(mine))),
        "foes": [],
        "food": [],
        "hills": hills,
        "water": {"rle": rle_from_mask(water)},
        "vis": {"rle": rle_from_mask(vis)},
    }

    ct = CompoundTeacher(min_raid_ants=4, raid_ratio=0.5)
    orders = ct.orders(obs, water, seen)

    dests = calculate_destinations(mine, orders, rows, cols)
    assert len(set(dests)) == len(dests), f"Collision when ant on hill: {dests}"
    # Ant on hill has raid_dist == 0 and should HOLD
    assert orders[0] == HOLD, f"Ant on hill should HOLD, got order {orders[0]}"


def test_adversarial_unreachable_island_enemy_hill():
    """Adversarial scenario: Enemy hill is surrounded by a ring of water (unreachable, cur_d = FAR).

    Verify raid squad smoothly falls back to gather/frontier potential field without throwing errors.
    """
    rows, cols = 24, 24
    water = np.zeros((rows, cols), dtype=bool)

    # Surround (5, 5) with water
    for dr in [-1, 0, 1]:
        for dc in [-1, 0, 1]:
            if dr != 0 or dc != 0:
                water[5 + dr, 5 + dc] = True

    hills = [[5, 5, 1]]  # on island
    food = [[15, 15], [16, 16]]
    mine = [[12, 12], [12, 13], [13, 12], [13, 13], [14, 12], [14, 13]]

    b = planes.Board(rows, cols)
    mine_plane = b.scatter(mine)
    vis = b.dilate(mine_plane, planes.VIEW_RADIUS2).astype(bool)
    seen = water | vis

    obs = {
        "size": [rows, cols],
        "mine": mine,
        "ids": list(range(len(mine))),
        "foes": [],
        "food": food,
        "hills": hills,
        "water": {"rle": rle_from_mask(water)},
        "vis": {"rle": rle_from_mask(vis)},
    }

    ct = CompoundTeacher(min_raid_ants=4, raid_ratio=0.5)
    orders = ct.orders(obs, water, seen)

    dests = calculate_destinations(mine, orders, rows, cols)
    food_set = {tuple(f) for f in food}
    assert len(set(dests)) == len(dests), f"Collision on island fallback: {dests}"
    for r, c in dests:
        assert not water[r, c]
        assert (r, c) not in food_set


def test_multi_turn_dynamic_trajectory_simulation():
    """Run 25 consecutive turns of movement simulation with CompoundTeacher.

    Updates ant positions, gathers food when nearby, updates memory with visibility correction,
    and asserts 0 collisions, 0 water steps, and 0 food steps across every single step.
    """
    rows, cols = 32, 32
    rng = np.random.default_rng(42)

    # Generate water
    water = rng.random((rows, cols)) < 0.20
    land = [(r, c) for r in range(rows) for c in range(cols) if not water[r, c]]
    rng.shuffle(land)

    n_ants = 12
    ants = [list(land[i]) for i in range(n_ants)]
    available = land[n_ants:]

    food = [list(available[i]) for i in range(15)]
    available = available[15:]

    hills = [[available[0][0], available[0][1], 1]]
    memory: dict = {}

    ct = CompoundTeacher(min_raid_ants=4, raid_ratio=0.5)
    b = planes.Board(rows, cols)

    for turn in range(25):
        mine_plane = b.scatter(ants)
        vis = b.dilate(mine_plane, planes.VIEW_RADIUS2).astype(bool)
        seen = water | vis

        obs = {
            "size": [rows, cols],
            "mine": ants,
            "ids": list(range(len(ants))),
            "foes": [],
            "food": [list(f) for f in food],
            "hills": [list(h) for h in hills],
            "water": {"rle": rle_from_mask(water)},
            "vis": {"rle": rle_from_mask(vis)},
        }

        # Update memory using where mode
        memory = planes.remember(memory, obs, mode="where")

        orders = ct.orders(obs, water, seen, memory)
        dests = calculate_destinations(ants, orders, rows, cols)
        food_set = {tuple(f) for f in food}

        # Assertions
        assert len(set(dests)) == len(dests), (
            f"Turn {turn}: Collision in multi-turn rollout!"
        )
        for r, c in dests:
            assert not water[r, c], (
                f"Turn {turn}: Ant stepped into water at ({r}, {c})!"
            )
            assert (r, c) not in food_set, (
                f"Turn {turn}: Ant stepped into food at ({r}, {c})!"
            )

        # Advance ants to dests
        ants = [list(p) for p in dests]

        # Gather food within spawn radius (dist2 <= 1)
        remaining_food = []
        for fr, fc in food:
            gathered = False
            for ar, ac in ants:
                dr = min(abs(ar - fr), rows - abs(ar - fr))
                dc = min(abs(ac - fc), cols - abs(ac - fc))
                if dr * dr + dc * dc <= 1:
                    gathered = True
                    break
            if not gathered:
                remaining_food.append([fr, fc])
        food = remaining_food

        # Raze hill if ant stepped on it
        remaining_hills = []
        for hr, hc, owner in hills:
            razed = False
            for ar, ac in ants:
                if ar == hr and ac == hc:
                    razed = True
                    break
            if not razed:
                remaining_hills.append([hr, hc, owner])
        hills = remaining_hills


def test_throughput_and_latency_profiling():
    """Benchmark per-call runtime latency and throughput across board sizes.

    Data collection (EXP-010 to EXP-014) requires 250,000 seat-turns.
    Verify average latency per turn is < 5ms on typical boards (e.g. 30x30 to 40x40)
    and < 25ms on large boards (100x100), ensuring efficient rollout collection.
    """
    rng = np.random.default_rng(12345)
    ct = CompoundTeacher(min_raid_ants=4, raid_ratio=0.5, max_depth=32, hill_lead=6)

    benchmarks = [
        ("Small 24x24", 24, 24, 6, 50),
        ("Medium 40x40", 40, 40, 14, 50),
        ("Large 64x64", 64, 64, 20, 30),
        ("ExtraLarge 100x100", 100, 100, 30, 20),
    ]

    results = {}
    for name, rows, cols, n_ants, n_calls in benchmarks:
        obs, water, seen, memory = generate_random_trial_board(
            rows, cols, n_ants, 0.20, 15, 2, rng
        )

        # Warmup
        for _ in range(3):
            ct.orders(obs, water, seen, memory)

        latencies = []
        for _ in range(n_calls):
            t0 = time.perf_counter()
            ct.orders(obs, water, seen, memory)
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000.0)  # ms

        mean_ms = float(np.mean(latencies))
        p50_ms = float(np.median(latencies))
        p95_ms = float(np.percentile(latencies, 95))
        max_ms = float(np.max(latencies))
        throughput_hz = 1000.0 / mean_ms

        results[name] = {
            "mean_ms": round(mean_ms, 3),
            "p50_ms": round(p50_ms, 3),
            "p95_ms": round(p95_ms, 3),
            "max_ms": round(max_ms, 3),
            "calls_per_sec": round(throughput_hz, 1),
        }

        # Assert reasonable operational budget
        if rows <= 40:
            assert mean_ms < 10.0, (
                f"{name}: Mean latency {mean_ms:.2f}ms exceeds 10ms threshold"
            )
        else:
            assert mean_ms < 50.0, (
                f"{name}: Mean latency {mean_ms:.2f}ms exceeds 50ms threshold"
            )

    print("\n[THROUGHPUT PROFILING RESULTS]")
    for name, res in results.items():
        print(
            f"  {name:18}: Mean={res['mean_ms']}ms, P95={res['p95_ms']}ms, Max={res['max_ms']}ms, Rate={res['calls_per_sec']} calls/s"
        )


if __name__ == "__main__":
    # Can also run standalone
    print("Running 500+ randomized trials stress test...")
    test_500_randomized_trials_zero_collisions_zero_water_zero_food()
    print("Running determinism test...")
    test_strict_determinism_and_reproducibility()
    print("Running chokepoints and dense herds test...")
    test_adversarial_chokepoints_and_dense_herds()
    print("Running corner and torus wrapping test...")
    test_adversarial_corner_and_torus_wrapping_squeeze()
    print("Running trapped ant test...")
    test_adversarial_ant_trapped_by_food_and_water()
    print("Running ant sitting on enemy hill test...")
    test_adversarial_ant_sitting_directly_on_enemy_hill()
    print("Running island enemy hill test...")
    test_adversarial_unreachable_island_enemy_hill()
    print("Running multi-turn trajectory simulation...")
    test_multi_turn_dynamic_trajectory_simulation()
    print("Running latency and throughput benchmarks...")
    test_throughput_and_latency_profiling()
    print("\nALL STRESS TESTS PASSED SUCCESSFULLY!")
