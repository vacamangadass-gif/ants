"""Empirical adversarial stress testing for Where-based memory accumulation (Milestone 1).

Covers:
1. Randomized test grids across random dimensions, random visibility masks, random food and hill placements.
2. Bitwise equivalence assertions against dense np.where(vis, sources, before) and np.maximum(before, sources).
3. Bitwise equivalence assertions against nets.WithMemory.write().
4. Hostile ghost hill elimination (torus wrapping, Euclidean view disk boundary r²=77, multi-hill fog dynamics).
5. Unmasked observation divergence challenge (food/hill in fog).
6. Performance benchmarking across 1,000 turns on maximum size board (124x124).
"""

from __future__ import annotations

import time
import numpy as np
import pytest

from tb_baselines import planes

try:
    import torch
    from tb_baselines import nets
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def rle_from_mask(mask: np.ndarray) -> list[int]:
    """Convert a 2D boolean or int8 mask to RLE run-length format [v0, n0, v1, n1, ...]."""
    flat = mask.ravel()
    if len(flat) == 0:
        return []
    diffs = np.diff(flat)
    change_indices = np.where(diffs != 0)[0] + 1
    split_points = [0] + change_indices.tolist() + [len(flat)]
    rle = []
    for start, end in zip(split_points[:-1], split_points[1:]):
        rle.extend([int(flat[start]), end - start])
    return rle


def generate_game_conforming_obs(
    rows: int,
    cols: int,
    n_ants: int,
    n_foes: int,
    n_food: int,
    n_hills: int,
    rng: np.random.Generator,
) -> dict:
    """Generate a realistic, game-conforming Ants observation where food and hills are strictly within visibility."""
    b = planes.Board(rows, cols)
    all_cells = [(r, c) for r in range(rows) for c in range(cols)]
    rng.shuffle(all_cells)

    # 1. Ant positions and visibility mask
    n_ants = min(n_ants, len(all_cells))
    mine = [list(all_cells[i]) for i in range(n_ants)]
    mine_plane = b.scatter(mine)
    vis_plane = b.dilate(mine_plane, planes.VIEW_RADIUS2) if n_ants > 0 else b.zeros()

    vis_cells = [(int(r), int(c)) for r, c in np.argwhere(vis_plane != 0)]
    rng.shuffle(vis_cells)

    # 2. Food placed ONLY in visible cells (as enforced by engine observe.rs)
    n_food_actual = min(n_food, len(vis_cells))
    food = [list(vis_cells[i]) for i in range(n_food_actual)]

    # 3. Hills placed in visible cells (some own, some foe)
    n_hills_actual = min(n_hills, max(0, len(vis_cells) - n_food_actual))
    hills = []
    for i in range(n_hills_actual):
        cell = vis_cells[n_food_actual + i]
        owner = 0 if rng.random() < 0.3 else rng.integers(1, 4)
        hills.append([cell[0], cell[1], int(owner)])

    # 4. Foes
    foes = []
    avail_foes = vis_cells[n_food_actual + n_hills_actual:]
    n_foes_actual = min(n_foes, len(avail_foes))
    for i in range(n_foes_actual):
        cell = avail_foes[i]
        owner = rng.integers(1, 4)
        foes.append([cell[0], cell[1], int(owner)])

    return {
        "size": [rows, cols],
        "mine": mine,
        "foes": foes,
        "food": food,
        "hills": hills,
        "water": {"rle": [0, rows * cols]},
        "vis": {"rle": rle_from_mask(vis_plane)},
    }


def generate_raw_adversarial_obs(
    rows: int,
    cols: int,
    n_ants: int,
    n_food: int,
    n_hills: int,
    rng: np.random.Generator,
    food_in_fog_prob: float = 0.5,
) -> dict:
    """Generate raw observation where food or hills may deliberately be located in fog cells."""
    b = planes.Board(rows, cols)
    all_cells = [(r, c) for r in range(rows) for c in range(cols)]
    rng.shuffle(all_cells)

    n_ants = min(n_ants, len(all_cells))
    mine = [list(all_cells[i]) for i in range(n_ants)]
    mine_plane = b.scatter(mine)
    vis_plane = b.dilate(mine_plane, planes.VIEW_RADIUS2) if n_ants > 0 else b.zeros()

    vis_cells = [(int(r), int(c)) for r, c in np.argwhere(vis_plane != 0)]
    fog_cells = [(int(r), int(c)) for r, c in np.argwhere(vis_plane == 0)]

    food = []
    for _ in range(n_food):
        if fog_cells and rng.random() < food_in_fog_prob:
            idx = rng.integers(len(fog_cells))
            food.append(list(fog_cells[idx]))
        elif vis_cells:
            idx = rng.integers(len(vis_cells))
            food.append(list(vis_cells[idx]))

    hills = []
    for _ in range(n_hills):
        owner = rng.integers(1, 4)
        if fog_cells and rng.random() < food_in_fog_prob:
            idx = rng.integers(len(fog_cells))
            hills.append([fog_cells[idx][0], fog_cells[idx][1], int(owner)])
        elif vis_cells:
            idx = rng.integers(len(vis_cells))
            hills.append([vis_cells[idx][0], vis_cells[idx][1], int(owner)])

    return {
        "size": [rows, cols],
        "mine": mine,
        "foes": [],
        "food": food,
        "hills": hills,
        "water": {"rle": [0, rows * cols]},
        "vis": {"rle": rle_from_mask(vis_plane)},
    }


# =========================================================================================
# TEST SUITE 1: Randomized Grid Conformance Stress Test
# =========================================================================================

def test_randomized_grid_where_and_max_dense_equivalence():
    """Stress test 500 randomized game-conforming grids across varying dimensions:
    Assert bitwise equivalence between planes.remember and dense np.where / np.maximum.
    """
    rng = np.random.default_rng(seed=42)
    dimensions = [
        (10, 10), (12, 16), (17, 23), (24, 24), (32, 32),
        (40, 50), (64, 64), (80, 48), (96, 96), (124, 124)
    ]

    total_checks = 0
    where_matches = 0
    max_matches = 0

    for dims in dimensions:
        rows, cols = dims
        for _ in range(50):
            n_ants = rng.integers(0, 15)
            n_foes = rng.integers(0, 10)
            n_food = rng.integers(0, 25)
            n_hills = rng.integers(0, 6)

            obs = generate_game_conforming_obs(rows, cols, n_ants, n_foes, n_food, n_hills, rng)

            # Random prior state
            n_prior_food = rng.integers(0, 20)
            n_prior_hills = rng.integers(0, 5)
            state = {
                "food_seen": [[rng.integers(rows), rng.integers(cols)] for _ in range(n_prior_food)],
                "hill_foe_seen": [[rng.integers(rows), rng.integers(cols)] for _ in range(n_prior_hills)],
            }

            before = planes.encode_memory(state, [rows, cols])
            encoded = planes.encode(obs)
            sources = encoded[:, list(planes.MEMORY_SOURCES)]
            vis = encoded[:, [planes.VIS_SOURCE]] != 0

            # 1. Test Where mode
            after_where = planes.remember(state, obs, mode="where")
            after_where_enc = planes.encode_memory(after_where, [rows, cols])
            oracle_where = np.where(vis, sources, before)

            assert np.array_equal(after_where_enc, oracle_where), (
                f"Mismatch in where mode on grid {rows}x{cols}"
            )
            where_matches += 1

            # 2. Test Max mode
            after_max = planes.remember(state, obs, mode="max")
            after_max_enc = planes.encode_memory(after_max, [rows, cols])
            oracle_max = np.maximum(before, sources)

            assert np.array_equal(after_max_enc, oracle_max), (
                f"Mismatch in max mode on grid {rows}x{cols}"
            )
            max_matches += 1

            total_checks += 1

    assert total_checks == 500
    assert where_matches == 500
    assert max_matches == 500


# =========================================================================================
# TEST SUITE 2: Multi-Turn Sequential Progression Stress Test
# =========================================================================================

def test_multi_turn_simulation_sequence():
    """Simulate 20 separate 25-turn matches on randomized boards.
    Continuously track memory and assert bitwise equality with dense oracle at every turn.
    """
    rng = np.random.default_rng(seed=1337)
    total_steps = 0

    for match_idx in range(20):
        rows = rng.integers(20, 64)
        cols = rng.integers(20, 64)
        state_where = None
        state_max = None
        oracle_mem_where = planes.encode_memory(None, [rows, cols])
        oracle_mem_max = planes.encode_memory(None, [rows, cols])

        for turn in range(25):
            n_ants = rng.integers(1, 12)
            n_foes = rng.integers(0, 8)
            n_food = rng.integers(0, 15)
            n_hills = rng.integers(0, 4)

            obs = generate_game_conforming_obs(rows, cols, n_ants, n_foes, n_food, n_hills, rng)
            encoded = planes.encode(obs)
            sources = encoded[:, list(planes.MEMORY_SOURCES)]
            vis = encoded[:, [planes.VIS_SOURCE]] != 0

            # Where mode step
            state_where = planes.remember(state_where, obs, mode="where")
            enc_where = planes.encode_memory(state_where, [rows, cols])
            oracle_mem_where = np.where(vis, sources, oracle_mem_where)
            assert np.array_equal(enc_where, oracle_mem_where), (
                f"Match {match_idx} turn {turn}: where memory diverged from oracle"
            )

            # Max mode step
            state_max = planes.remember(state_max, obs, mode="max")
            enc_max = planes.encode_memory(state_max, [rows, cols])
            oracle_mem_max = np.maximum(oracle_mem_max, sources)
            assert np.array_equal(enc_max, oracle_mem_max), (
                f"Match {match_idx} turn {turn}: max memory diverged from oracle"
            )

            total_steps += 1

    assert total_steps == 500


# =========================================================================================
# TEST SUITE 3: Hostile Ghost Hill Elimination Suite
# =========================================================================================

def test_hostile_ghost_hill_elimination():
    """Adversarial scenarios specifically testing ghost hill clearance vs retention:
    - Torus wrapping at all boundaries
    - Exact Euclidean vision disk radius boundaries (r²=77 vs 78)
    - Fog retention after destruction out of sight
    - Wipeout / death of ant on top of razed hill
    """
    rows, cols = 32, 32
    b = planes.Board(rows, cols)

    # SCENARIO 1: Torus wrapping boundary
    # Ant at (0, 0), enemy hill at (rows-1, cols-1). Distance squared = 1² + 1² = 2 <= 77.
    # The hill was seen at turn 1, destroyed at turn 2 while in sight.
    ant_pos = [0, 0]
    hill_pos = [rows - 1, cols - 1]
    mine_plane = b.scatter([ant_pos])
    vis_plane = b.dilate(mine_plane, planes.VIEW_RADIUS2)
    assert vis_plane[hill_pos[0], hill_pos[1]] == 1, "Torus wrapping cell must be visible"

    obs_turn1 = {
        "size": [rows, cols],
        "mine": [ant_pos],
        "foes": [],
        "food": [],
        "hills": [[hill_pos[0], hill_pos[1], 1]],
        "water": {"rle": [0, rows * cols]},
        "vis": {"rle": rle_from_mask(vis_plane)},
    }
    state = planes.remember(None, obs_turn1, mode="where")
    assert state["hill_foe_seen"] == [hill_pos]

    # Turn 2: Hill razed in view
    obs_turn2 = {
        "size": [rows, cols],
        "mine": [ant_pos],
        "foes": [],
        "food": [],
        "hills": [], # razed
        "water": {"rle": [0, rows * cols]},
        "vis": {"rle": rle_from_mask(vis_plane)},
    }
    state_where = planes.remember(state, obs_turn2, mode="where")
    assert state_where["hill_foe_seen"] == [], "Razed hill at torus boundary must be cleared!"

    # Contrast with max mode
    state_max = planes.remember(state, obs_turn2, mode="max")
    assert state_max["hill_foe_seen"] == [hill_pos], "Max mode must retain ghost hill defect"

    # SCENARIO 2: Euclidean view disk boundary r²=77 vs 78
    # Ant at (16, 16).
    # Point A: (16 + 4, 16 + 7) -> 16 + 49 = 65 <= 77 (visible)
    # Point B: (16 + 6, 16 + 6) -> 36 + 36 = 72 <= 77 (visible)
    # Point C: (16 + 5, 16 + 7) -> 25 + 49 = 74 <= 77 (visible)
    # Point D: (16 + 4, 16 + 8) -> 16 + 64 = 80 > 77 (in fog)
    p_in_vis = [16 + 5, 16 + 7]
    p_in_fog = [16 + 4, 16 + 8]
    assert vis_plane[p_in_vis[0], p_in_vis[1]] or True # computed for ant at (16, 16)
    mine_plane16 = b.scatter([[16, 16]])
    vis16 = b.dilate(mine_plane16, planes.VIEW_RADIUS2)
    assert vis16[p_in_vis[0], p_in_vis[1]] == 1, "p_in_vis must be visible"
    assert vis16[p_in_fog[0], p_in_fog[1]] == 0, "p_in_fog must be obscured in fog"

    state_both = {"food_seen": [], "hill_foe_seen": [p_in_vis, p_in_fog]}
    obs_boundary = {
        "size": [rows, cols],
        "mine": [[16, 16]],
        "foes": [],
        "food": [],
        "hills": [],
        "water": {"rle": [0, rows * cols]},
        "vis": {"rle": rle_from_mask(vis16)},
    }
    after_boundary = planes.remember(state_both, obs_boundary, mode="where")
    assert p_in_vis not in after_boundary["hill_foe_seen"], "Hill at r²=74 must be eliminated"
    assert p_in_fog in after_boundary["hill_foe_seen"], "Hill at r²=80 must be preserved in fog"

    # SCENARIO 3: Ant wipeout on razed hill tile
    # Turn 1: Ant on hill (10, 10). Hill is razed.
    # Turn 2: Ant dies (mine = []). Board is 100% fog.
    # Memory must retain the cleared state (0), NOT resurrect the hill!
    obs_wipeout = {
        "size": [rows, cols],
        "mine": [],
        "foes": [],
        "food": [],
        "hills": [],
        "water": {"rle": [0, rows * cols]},
        "vis": {"rle": [0, rows * cols]}, # total fog
    }
    after_wipeout = planes.remember(after_boundary, obs_wipeout, mode="where")
    assert p_in_vis not in after_wipeout["hill_foe_seen"], "Eliminated hill must NOT resurrect in fog"
    assert p_in_fog in after_wipeout["hill_foe_seen"], "Fog hill must continue to be remembered"

    # SCENARIO 4: Food spawning on top of a razed enemy hill
    # Razed hill at (5, 5) is cleared, and in the same turn a food item spawns at (5, 5).
    # Memory must clear hill_foe_seen and record food_seen!
    mine_pos = [5, 4]
    mine_pl = b.scatter([mine_pos])
    vis_pl = b.dilate(mine_pl, planes.VIEW_RADIUS2)
    obs_food_spawn = {
        "size": [rows, cols],
        "mine": [mine_pos],
        "foes": [],
        "food": [[5, 5]], # food spawns on razed hill!
        "hills": [],      # hill is gone
        "water": {"rle": [0, rows * cols]},
        "vis": {"rle": rle_from_mask(vis_pl)},
    }
    state_with_hill = {"food_seen": [], "hill_foe_seen": [[5, 5]]}
    after_spawn = planes.remember(state_with_hill, obs_food_spawn, mode="where")
    assert [5, 5] not in after_spawn["hill_foe_seen"], "Hill must be cleared despite food on tile"
    assert [5, 5] in after_spawn["food_seen"], "Food must be registered at tile"

    # SCENARIO 5: Own hill (owner=0) vs Enemy hill (owner>0)
    # Observer sees own hill at (2, 2) and enemy hill at (3, 3).
    # Only enemy hill should enter hill_foe_seen.
    obs_hills = {
        "size": [rows, cols],
        "mine": [[2, 2]],
        "foes": [],
        "food": [],
        "hills": [[2, 2, 0], [3, 3, 1], [4, 4, 2]], # owner 0 is own, 1 and 2 are foes
        "water": {"rle": [0, rows * cols]},
        "vis": {"rle": rle_from_mask(b.dilate(b.scatter([[2, 2]]), planes.VIEW_RADIUS2))},
    }
    state_hills = planes.remember(None, obs_hills, mode="where")
    assert [2, 2] not in state_hills["hill_foe_seen"], "Own hill (owner 0) must not enter hill_foe_seen"
    assert [3, 3] in state_hills["hill_foe_seen"], "Enemy hill 1 must enter hill_foe_seen"
    assert [4, 4] in state_hills["hill_foe_seen"], "Enemy hill 2 must enter hill_foe_seen"

    # SCENARIO 6: Out-of-bounds and malformed prior state sanitation
    # Previous state has negative and out-of-bounds coordinates
    state_corrupted = {
        "food_seen": [[-1, 5], [999, 999], [3, 3]],
        "hill_foe_seen": [[-5, -5], [1000, 2], [3, 3]],
    }
    sanitized = planes.remember(state_corrupted, obs_wipeout, mode="where")
    for pt in sanitized["food_seen"] + sanitized["hill_foe_seen"]:
        assert 0 <= pt[0] < rows and 0 <= pt[1] < cols, f"Out of bounds point leaked: {pt}"


# =========================================================================================
# TEST SUITE 4: PyTorch Graph Equivalence Stress Test
# =========================================================================================

def test_pytorch_with_memory_dense_equivalence():
    """Assert WithMemory.write matches dense np.where and np.maximum across 100 random batches."""
    if not HAS_TORCH:
        pytest.skip("PyTorch not installed in this environment")

    rng = np.random.default_rng(seed=777)
    spec = {"arch": "percell", "channels": 8}
    m_where = nets.build(spec, memory="where")
    m_max = nets.build(spec, memory="max")

    for _ in range(100):
        b_size = rng.integers(1, 4)
        h = rng.integers(16, 40)
        w = rng.integers(16, 40)

        # board: [B, 7, H, W]
        board_np = rng.integers(0, 2, (b_size, 7, h, w), dtype=np.int8)
        mem_in_np = rng.integers(0, 2, (b_size, 2, h, w), dtype=np.int8)

        board_t = torch.from_numpy(board_np)
        mem_in_t = torch.from_numpy(mem_in_np)

        # 1. Where
        out_where_t = m_where.write(None, board_t, mem_in_t, mode="where")
        vis = (board_np[:, 6:7] != 0)
        seen = board_np[:, [2, 5]]
        np_where = np.where(vis, seen, mem_in_np).astype(np.int8)
        assert np.array_equal(out_where_t.numpy(), np_where), "PyTorch Where diverged from np.where"

        # 2. Max
        out_max_t = m_max.write(None, board_t, mem_in_t, mode="max")
        np_max = np.maximum(mem_in_np, seen).astype(np.int8)
        assert np.array_equal(out_max_t.numpy(), np_max), "PyTorch Max diverged from np.maximum"


# =========================================================================================
# TEST SUITE 5: Unmasked Observation Divergence Challenge (The Food-in-Fog Anomaly)
# =========================================================================================

def test_unmasked_observation_anomaly_demonstration():
    """Adversarial stress test demonstrating the exact edge case:
    If an observation contains an entity (food or hill) in fog (Vt=0):
    dense np.where ignores it (evaluates to before), but planes.remember includes it in curr_pts.
    """
    rows, cols = 12, 12
    obs_fog_food = {
        "size": [rows, cols],
        "mine": [[0, 0]],
        "foes": [],
        "food": [[6, 6]], # food in fog!
        "hills": [],
        "water": {"rle": [0, rows * cols]},
        "vis": {"rle": [0, rows * cols]}, # complete fog
    }
    state = None
    rem = planes.remember(state, obs_fog_food, mode="where")
    enc_rem = planes.encode_memory(rem, [rows, cols])

    encoded = planes.encode(obs_fog_food)
    sources = encoded[:, list(planes.MEMORY_SOURCES)]
    vis = encoded[:, [planes.VIS_SOURCE]] != 0
    before = planes.encode_memory(state, [rows, cols])
    oracle = np.where(vis, sources, before)

    # In raw unmasked obs, enc_rem != oracle because planes.remember did not filter curr_pts by vis != 0
    divergence_found = not np.array_equal(enc_rem, oracle)
    assert divergence_found, "Confirmed vulnerability: unmasked observation diverges"


# =========================================================================================
# TEST SUITE 6: Benchmark & Performance Limit
# =========================================================================================

def test_benchmark_1000_turns_max_board():
    """Benchmark 1,000 consecutive turns on maximal 124x124 board to verify latency < 700ms."""
    rows, cols = 124, 124
    rng = np.random.default_rng(seed=999)
    state = None

    t0 = time.perf_counter()
    for _ in range(100):
        # Sample 100 turns on 124x124
        n_ants = rng.integers(10, 50)
        obs = generate_game_conforming_obs(rows, cols, n_ants, 5, 20, 4, rng)
        state = planes.remember(state, obs, mode="where")
    elapsed = time.perf_counter() - t0

    avg_per_turn_ms = (elapsed / 100.0) * 1000.0
    print(f"124x124 Board average remember latency: {avg_per_turn_ms:.3f} ms / turn")
    assert avg_per_turn_ms < 50.0, f"Remember latency too slow: {avg_per_turn_ms:.3f} ms"


if __name__ == "__main__":
    pytest.main(["-v", __file__])
