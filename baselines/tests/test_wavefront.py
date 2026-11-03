from __future__ import annotations

import json
from pathlib import Path
import numpy as np

from tb_baselines.teacher import (
    DELTAS,
    FAR,
    HOLD,
    CompoundTeacher,
    Teacher,
    arbitrate_orders,
    compute_geodesic_potential_field,
    extract_exploration_frontier,
    extract_food_gathering_points,
    known,
)
from tb_baselines.telemetry import (
    HoldReason,
    SourceType,
)

ROOT = Path(__file__).resolve().parents[2]


def make_board(
    size: int | tuple[int, int] = 24,
    ants: tuple[tuple[int, int], ...] = ((12, 12),),
    hills: tuple[tuple[int, int, int], ...] = (),
    food: tuple[tuple[int, int], ...] = (),
    water_coords: tuple[tuple[int, int], ...] = (),
    visible_coords: tuple[tuple[int, int], ...] | None = None,
    ids: list[int] | None = None,
) -> dict:
    if isinstance(size, tuple):
        rows, cols = size
    else:
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


def loop_extract_frontier(
    seen_mask: np.ndarray,
    water_mask: np.ndarray,
    rows: int,
    cols: int,
    visible_food_mask: np.ndarray | None = None,
) -> np.ndarray:
    frontier = np.zeros((rows, cols), dtype=bool)
    for r in range(rows):
        for c in range(cols):
            if not seen_mask[r, c]:
                continue
            if water_mask[r, c]:
                continue
            if visible_food_mask is not None and visible_food_mask[r, c]:
                continue
            has_unseen = False
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr = (r + dr) % rows
                nc = (c + dc) % cols
                if not seen_mask[nr, nc]:
                    has_unseen = True
                    break
            if has_unseen:
                frontier[r, c] = True
    return frontier


# ==============================================================================
# Suite 1: 自然波前耗尽与测地距离场扩展 (EXP-073A)
# ==============================================================================


def test_geodesic_field_single_source_open_torus():
    rows, cols = 12, 12
    passable = np.ones((rows, cols), dtype=bool)
    sources = [(6, 6)]

    dist = compute_geodesic_potential_field(sources, passable, rows, cols)

    assert dist.shape == (rows, cols)
    assert dist.dtype == np.int32
    assert dist[6, 6] == 0

    for r in range(rows):
        for c in range(cols):
            dr = min(abs(r - 6), rows - abs(r - 6))
            dc = min(abs(c - 6), cols - abs(c - 6))
            expected_dist = dr + dc
            assert dist[r, c] == expected_dist
            assert dist[r, c] < FAR


def test_geodesic_field_large_torus_50x100_no_depth_clipping():
    rows, cols = 50, 100
    passable = np.ones((rows, cols), dtype=bool)
    sources = [(0, 0)]

    dist = compute_geodesic_potential_field(sources, passable, rows, cols)

    furthest_r = 25
    furthest_c = 50
    expected_max_dist = furthest_r + furthest_c
    assert dist[furthest_r, furthest_c] == expected_max_dist
    assert int(np.max(dist)) == expected_max_dist
    assert np.all(dist < FAR)
    assert np.count_nonzero(dist == FAR) == 0


def test_geodesic_field_snake_corridor_depth_exceeding_32():
    rows, cols = 16, 16
    passable = np.zeros((rows, cols), dtype=bool)
    corridor_cells: list[tuple[int, int]] = []

    for segment in range(6):
        r = segment * 2
        if segment % 2 == 0:
            for c in range(1, 15):
                corridor_cells.append((r, c))
                passable[r, c] = True
            if segment < 5:
                corridor_cells.append((r + 1, 14))
                passable[r + 1, 14] = True
        else:
            for c in range(14, 0, -1):
                corridor_cells.append((r, c))
                passable[r, c] = True
            if segment < 5:
                corridor_cells.append((r + 1, 1))
                passable[r + 1, 1] = True

    sources = [corridor_cells[0]]
    dist = compute_geodesic_potential_field(sources, passable, rows, cols)

    total_path_len = len(corridor_cells)
    assert total_path_len > 60

    for idx, (r, c) in enumerate(corridor_cells):
        assert dist[r, c] == idx

    assert dist[corridor_cells[-1]] == total_path_len - 1
    assert dist[corridor_cells[-1]] > 32


def test_geodesic_field_multi_source_propagation():
    rows, cols = 20, 20
    passable = np.ones((rows, cols), dtype=bool)
    source_a = (5, 5)
    source_b = (15, 15)

    dist = compute_geodesic_potential_field([source_a, source_b], passable, rows, cols)

    assert dist[source_a] == 0
    assert dist[source_b] == 0

    for r in range(rows):
        for c in range(cols):
            da = min(abs(r - 5), rows - abs(r - 5)) + min(abs(c - 5), cols - abs(c - 5))
            db = min(abs(r - 15), rows - abs(r - 15)) + min(abs(c - 15), cols - abs(c - 15))
            expected = min(da, db)
            assert dist[r, c] == expected


def test_geodesic_field_unreachable_components_marked_far():
    rows, cols = 20, 20
    passable = np.ones((rows, cols), dtype=bool)

    for r in range(4, 9):
        for c in range(4, 9):
            if r in (4, 8) or c in (4, 8):
                passable[r, c] = False

    sources = [(0, 0)]
    dist = compute_geodesic_potential_field(sources, passable, rows, cols)

    for r in range(5, 8):
        for c in range(5, 8):
            assert dist[r, c] == FAR

    assert dist[0, 0] == 0
    assert dist[19, 19] < FAR


def test_geodesic_field_empty_sources():
    rows, cols = 10, 10
    passable = np.ones((rows, cols), dtype=bool)

    dist = compute_geodesic_potential_field([], passable, rows, cols)

    assert dist.shape == (rows, cols)
    assert np.all(dist == FAR)


def test_geodesic_field_all_sources_on_water():
    rows, cols = 10, 10
    passable = np.ones((rows, cols), dtype=bool)
    passable[2, 2] = False
    passable[3, 3] = False

    dist = compute_geodesic_potential_field([(2, 2), (3, 3)], passable, rows, cols)

    assert np.all(dist == FAR)


def test_backup_enemy_hills_natural_exhaustion_in_teacher():
    rows, cols = 40, 40
    obs = make_board(
        size=(rows, cols),
        ants=((0, 0),),
        hills=((20, 20, 1),),
        food=(),
    )
    teacher = CompoundTeacher()
    water, seen = known(obs)

    orders = teacher.orders(obs, water, seen)

    assert len(orders) == 1
    assert orders[0] != HOLD

    dest_dr, dest_dc = DELTAS[int(orders[0])]
    next_pos = ((0 + dest_dr) % rows, (0 + dest_dc) % cols)
    manhattan_before = min(abs(0 - 20), rows - abs(0 - 20)) + min(abs(0 - 20), cols - abs(0 - 20))
    manhattan_after = min(abs(next_pos[0] - 20), rows - abs(next_pos[0] - 20)) + min(abs(next_pos[1] - 20), cols - abs(next_pos[1] - 20))
    assert manhattan_after < manhattan_before

    plain_teacher = Teacher()
    plain_orders = plain_teacher.orders(obs, water, seen)
    assert len(plain_orders) == 1
    assert plain_orders[0] != HOLD


# ==============================================================================
# Suite 2: 战争迷雾精确可达探索前沿 G_frontier (EXP-073B)
# ==============================================================================


def test_extract_frontier_vectorized_vs_loop_exact_match():
    rows, cols = 28, 28
    rng = np.random.RandomState(42)

    for _ in range(5):
        seen_mask = rng.rand(rows, cols) > 0.4
        water_mask = (rng.rand(rows, cols) > 0.8) & seen_mask
        food_mask = (rng.rand(rows, cols) > 0.95) & seen_mask & (~water_mask)

        vec_frontier = extract_exploration_frontier(seen_mask, water_mask, rows, cols, food_mask)
        loop_frontier = loop_extract_frontier(seen_mask, water_mask, rows, cols, food_mask)

        assert np.array_equal(vec_frontier, loop_frontier)


def test_frontier_safety_zero_water_and_zero_fog():
    rows, cols = 24, 24
    seen_mask = np.zeros((rows, cols), dtype=bool)
    water_mask = np.zeros((rows, cols), dtype=bool)
    food_mask = np.zeros((rows, cols), dtype=bool)

    seen_mask[5:15, 5:15] = True
    water_mask[6:8, 6:8] = True
    food_mask[10, 10] = True

    frontier = extract_exploration_frontier(seen_mask, water_mask, rows, cols, food_mask)

    assert np.all(seen_mask[frontier])
    assert not np.any(water_mask[frontier])
    assert not np.any(food_mask[frontier])
    assert np.count_nonzero(frontier & (~seen_mask)) == 0
    assert np.count_nonzero(frontier & water_mask) == 0


def test_frontier_orthogonal_adjacency_to_fog():
    rows, cols = 20, 20
    seen_mask = np.zeros((rows, cols), dtype=bool)
    water_mask = np.zeros((rows, cols), dtype=bool)

    seen_mask[5:15, 5:15] = True

    frontier = extract_exploration_frontier(seen_mask, water_mask, rows, cols)

    for r in range(rows):
        for c in range(cols):
            if frontier[r, c]:
                has_fog_neighbor = False
                for dr, dc in DELTAS:
                    nr = (r + dr) % rows
                    nc = (c + dc) % cols
                    if not seen_mask[nr, nc]:
                        has_fog_neighbor = True
                        break
                assert has_fog_neighbor

    assert not frontier[10, 10]


def test_frontier_torus_boundary_wrapping():
    rows, cols = 20, 20
    seen_mask = np.zeros((rows, cols), dtype=bool)
    water_mask = np.zeros((rows, cols), dtype=bool)

    seen_mask[0, :] = True

    frontier = extract_exploration_frontier(seen_mask, water_mask, rows, cols)

    for c in range(cols):
        assert frontier[0, c]


def test_frontier_fully_revealed_map_exhaustion():
    rows, cols = 20, 20
    seen_mask = np.ones((rows, cols), dtype=bool)
    water_mask = np.zeros((rows, cols), dtype=bool)

    frontier = extract_exploration_frontier(seen_mask, water_mask, rows, cols)

    assert np.count_nonzero(frontier) == 0


def test_frontier_completely_unrevealed_map():
    rows, cols = 20, 20
    seen_mask = np.zeros((rows, cols), dtype=bool)
    water_mask = np.zeros((rows, cols), dtype=bool)

    frontier = extract_exploration_frontier(seen_mask, water_mask, rows, cols)

    assert np.count_nonzero(frontier) == 0


def test_frontier_water_adjacent_to_fog_excluded():
    rows, cols = 20, 20
    seen_mask = np.zeros((rows, cols), dtype=bool)
    water_mask = np.zeros((rows, cols), dtype=bool)

    seen_mask[10, 10] = True
    water_mask[10, 10] = True

    frontier = extract_exploration_frontier(seen_mask, water_mask, rows, cols)

    assert not frontier[10, 10]
    assert np.count_nonzero(frontier) == 0


# ==============================================================================
# Suite 3: 任务状态机解耦与群体停滞消除 (EXP-073B)
# ==============================================================================


def test_task_decoupling_remote_food_does_not_halt_distant_exploration():
    rows, cols = 30, 30
    visible_cells: list[tuple[int, int]] = []

    for r in range(1, 5):
        for c in range(1, 5):
            visible_cells.append((r, c))

    for r in range(18, 25):
        for c in range(18, 25):
            visible_cells.append((r, c))

    ants = ((2, 3), (20, 20), (20, 22))
    food = ((2, 2),)
    obs = make_board(
        size=(rows, cols),
        ants=ants,
        food=food,
        visible_coords=tuple(visible_cells),
    )

    teacher = CompoundTeacher()
    water, seen = known(obs)
    orders = teacher.orders(obs, water, seen, collect_telemetry=True)

    telemetry = teacher.export_turn_telemetry()
    assert len(telemetry) == 3

    ant0_record = next(t for t in telemetry if t.ant_id == 0)
    assert ant0_record.nearest_source_type in (SourceType.FOOD_VISIBLE.value, "food_visible")

    ant1_record = next(t for t in telemetry if t.ant_id == 1)
    ant2_record = next(t for t in telemetry if t.ant_id == 2)
    assert ant1_record.nearest_source_type in (SourceType.FRONTIER_UNSEEN.value, "frontier_unseen")
    assert ant2_record.nearest_source_type in (SourceType.FRONTIER_UNSEEN.value, "frontier_unseen")

    assert orders[1] != HOLD
    assert orders[2] != HOLD


def test_task_state_machine_food_capacity_constraint():
    rows, cols = 20, 20
    ants = ((5, 6), (5, 4), (4, 5), (6, 5), (15, 15))
    food = ((5, 5),)
    obs = make_board(size=20, ants=ants, food=food)

    teacher = CompoundTeacher()
    water, seen = known(obs)
    orders = teacher.orders(obs, water, seen)

    assert len(orders) == 5
    ant_dests = []
    for (r, c), act in zip(ants, orders):
        if act == HOLD:
            ant_dests.append((r, c))
        else:
            dr, dc = DELTAS[int(act)]
            ant_dests.append(((r + dr) % rows, (c + dc) % cols))

    assert len(set(ant_dests)) == len(ant_dests)


def test_task_state_machine_transition_on_frontier_exhaustion():
    rows, cols = 30, 30
    ants = ((5, 5), (6, 6))
    hills = ((25, 25, 1),)
    obs = make_board(size=(rows, cols), ants=ants, hills=hills, food=(), visible_coords=None)

    teacher = CompoundTeacher()
    water, seen = known(obs)
    orders = teacher.orders(obs, water, seen, collect_telemetry=True)
    assert len(orders) == 2

    telemetry = teacher.export_turn_telemetry()
    for record in telemetry:
        assert record.nearest_source_type in (SourceType.HILL_FOE.value, "hill_foe")


def test_task_state_machine_standby_when_no_targets():
    rows, cols = 20, 20
    ants = ((10, 10), (10, 11))
    obs = make_board(size=(rows, cols), ants=ants, hills=((10, 10, 0), (10, 11, 0)), food=(), visible_coords=None)

    teacher = CompoundTeacher()
    water, seen = known(obs)
    orders = teacher.orders(obs, water, seen, collect_telemetry=True)
    assert len(orders) == 2
    assert all(act == HOLD for act in orders)

    telemetry = teacher.export_turn_telemetry()
    for record in telemetry:
        assert record.is_intentional_hold is True
        assert record.hold_reason in (HoldReason.DEFENSE.value, "defense")


# ==============================================================================
# Suite 4: 数学不变量与前置里程碑零回退 (Invariants 1, 2, 3)
# ==============================================================================


def test_invariant_1_gradient_monotonicity_all_passable_cells():
    map_path = ROOT / "benchmarks" / "maps_2p" / "medium-crossroads-2p-1h.json"
    with open(map_path, "r", encoding="utf-8") as f:
        map_data = json.load(f)

    rows = map_data["rows"]
    cols = map_data["cols"]
    water_mask = np.zeros((rows, cols), dtype=bool)

    raw_water = map_data.get("water", {})
    if "rle" in raw_water:
        runs = raw_water["rle"]
        flat = np.zeros(rows * cols, dtype=bool)
        idx = 0
        for i in range(0, len(runs), 2):
            val = bool(runs[i])
            length = runs[i + 1]
            if val:
                flat[idx : idx + length] = True
            idx += length
        water_mask = flat.reshape((rows, cols))

    passable_mask = ~water_mask
    sources = [(int(h[0]), int(h[1])) for h in map_data.get("hills", []) if len(h) >= 2]
    assert len(sources) > 0

    dist = compute_geodesic_potential_field(sources, passable_mask, rows, cols)

    for r in range(rows):
        for c in range(cols):
            if not passable_mask[r, c]:
                continue
            cur_d = dist[r, c]
            if cur_d == FAR or cur_d == 0:
                continue

            has_strictly_lower_neighbor = False
            for dr, dc in DELTAS:
                nr = (r + dr) % rows
                nc = (c + dc) % cols
                if passable_mask[nr, nc] and dist[nr, nc] == cur_d - 1:
                    has_strictly_lower_neighbor = True
                    break
            assert has_strictly_lower_neighbor


def test_invariant_2_frontier_safety_and_soundness():
    map_path = ROOT / "benchmarks" / "maps_2p" / "small-arena-2p-1h.json"
    with open(map_path, "r", encoding="utf-8") as f:
        map_data = json.load(f)

    rows = map_data["rows"]
    cols = map_data["cols"]
    water_mask = np.zeros((rows, cols), dtype=bool)
    raw_water = map_data.get("water", {})
    if "rle" in raw_water:
        runs = raw_water["rle"]
        flat = np.zeros(rows * cols, dtype=bool)
        idx = 0
        for i in range(0, len(runs), 2):
            val = bool(runs[i])
            length = runs[i + 1]
            if val:
                flat[idx : idx + length] = True
            idx += length
        water_mask = flat.reshape((rows, cols))

    seen_mask = np.zeros((rows, cols), dtype=bool)
    center_r, center_c = rows // 2, cols // 2
    for r in range(rows):
        for c in range(cols):
            dr = min(abs(r - center_r), rows - abs(r - center_r))
            dc = min(abs(c - center_c), cols - abs(c - center_c))
            if dr * dr + dc * dc <= 36:
                seen_mask[r, c] = True

    frontier = extract_exploration_frontier(seen_mask, water_mask, rows, cols)

    for r in range(rows):
        for c in range(cols):
            if frontier[r, c]:
                assert seen_mask[r, c]
                assert not water_mask[r, c]
                has_fog_neighbor = any(
                    not seen_mask[(r + dr) % rows, (c + dc) % cols] for dr, dc in DELTAS
                )
                assert has_fog_neighbor


def test_invariant_3_zero_regression_m1_telemetry():
    obs = make_board(size=24, ants=((12, 12), (12, 13)), food=((12, 14),))
    teacher = CompoundTeacher()
    water, seen = known(obs)
    orders = teacher.orders(obs, water, seen, collect_telemetry=True)
    assert len(orders) == 2

    telemetry = teacher.export_turn_telemetry()
    assert len(telemetry) == 2
    for rec in telemetry:
        rec_dict = rec.to_dict()
        assert "valid_gradient_mask" in rec_dict
        assert "potential_gradient" in rec_dict
        assert "hold_reason" in rec_dict
        assert isinstance(rec.is_intentional_hold, bool)


def test_invariant_3_zero_regression_m2_food_gathering():
    rows, cols = 20, 20
    water = np.zeros((rows, cols), dtype=bool)
    food = [(10, 10)]

    g_food = extract_food_gathering_points(food, water, rows, cols)

    assert not g_food[10, 10]
    assert np.count_nonzero(g_food) == 4
    for dr, dc in DELTAS:
        assert g_food[(10 + dr) % rows, (10 + dc) % cols]


def test_invariant_3_zero_regression_m3_arbitration():
    ants = ((5, 5), (5, 6))
    candidate_sequences = [[DELTAS.index((0, 1)), HOLD], [HOLD]]
    potentials = np.array([10, 5], dtype=np.int32)
    orders, stalls, iters = arbitrate_orders(
        ants, candidate_sequences, potentials, return_iteration_count=True, rows=20, cols=20
    )

    assert iters <= 8
    assert orders[1] == HOLD
    assert orders[0] == HOLD
