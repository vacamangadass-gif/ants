#!/usr/bin/env python3
"""Generates 9 distinct, symmetric 2-player benchmark maps for TinyBrains Ants IDEA-002 (Milestone 4).

All maps adhere to the strict naming pattern <size>-<terrain>-2p-1h.json:
- tiny-open-2p-1h.json (24x24 open plains)
- tiny-colonnade-2p-1h.json (24x24 periodic columns)
- tiny-corridor-2p-1h.json (24x24 choke point corridor)
- small-arena-2p-1h.json (32x32 open arena)
- small-islands-2p-1h.json (32x32 archipelago land bridges)
- small-rooms-2p-1h.json (32x32 partitioned rooms)
- medium-valley-2p-1h.json (40x40 winding central valley)
- medium-labyrinth-2p-1h.json (40x40 symmetric maze)
- medium-crossroads-2p-1h.json (40x40 four-quadrant crossroads)
"""

from __future__ import annotations

import collections
import json
import random
from pathlib import Path


def to_rle(flat: list[int]) -> list[int]:
    """Run-length encode a binary grid (row-major)."""
    rle = []
    if not flat:
        return rle
    cur_val = flat[0]
    cur_cnt = 1
    for v in flat[1:]:
        if v == cur_val:
            cur_cnt += 1
        else:
            rle.extend([cur_val, cur_cnt])
            cur_val = v
            cur_cnt = 1
    rle.extend([cur_val, cur_cnt])
    return rle


def create_map(
    map_id: str,
    rows: int,
    cols: int,
    hill_pos: tuple[int, int],
    water_pattern_fn,
    food_count: int,
    seed: int,
) -> dict:
    rng = random.Random(seed)
    dr, dc = rows // 2, cols // 2
    h0 = hill_pos
    h1 = ((h0[0] + dr) % rows, (h0[1] + dc) % cols)
    hills = [list(h0), list(h1)]

    # 1. Initialize water matrix: 0 is land, 1 is water
    water = [[0 for _ in range(cols)] for _ in range(rows)]

    # Apply terrain pattern
    water_pattern_fn(water, rows, cols, dr, dc)

    # 2. Symmetrize water under (dr, dc) shift
    for r in range(rows):
        for c in range(cols):
            r2, c2 = (r + dr) % rows, (c + dc) % cols
            if water[r][c] == 1 or water[r2][c2] == 1:
                water[r][c] = 1
                water[r2][c2] = 1

    # 3. Ensure hills and immediate surrounding rings are land
    for hr, hc in hills:
        for r_off in range(-2, 3):
            for c_off in range(-2, 3):
                if abs(r_off) + abs(c_off) <= 2:
                    rr = (hr + r_off) % rows
                    cc = (hc + c_off) % cols
                    water[rr][cc] = 0
                    water[(rr + dr) % rows][(cc + dc) % cols] = 0

    # Ensure a direct wide highway connecting hill 0 to hill 1 is never water
    # (guarantees connectivity)
    cr, cc = h0[0], h0[1]
    tr, tc = h1[0], h1[1]
    # L-shaped corridor from h0 to h1
    while cr != tr:
        water[cr][cc] = 0
        water[(cr + dr) % rows][(cc + dc) % cols] = 0
        cr = (cr + (1 if (tr - cr) % rows < rows // 2 else -1)) % rows
    while cc != tc:
        water[cr][cc] = 0
        water[(cr + dr) % rows][(cc + dc) % cols] = 0
        cc = (cc + (1 if (tc - cc) % cols < cols // 2 else -1)) % cols

    # 4. Enforce single connected component from hill 0
    # Run BFS on torus
    visited = [[False for _ in range(cols)] for _ in range(rows)]
    q = collections.deque([h0])
    visited[h0[0]][h0[1]] = True
    while q:
        cr, cc = q.popleft()
        for nr, nc in [((cr - 1) % rows, cc), ((cr + 1) % rows, cc), (cr, (cc - 1) % cols), (cr, (cc + 1) % cols)]:
            if not visited[nr][nc] and water[nr][nc] == 0:
                visited[nr][nc] = True
                q.append((nr, nc))

    # Any land not reachable from hill 0 must become water
    for r in range(rows):
        for c in range(cols):
            if water[r][c] == 0 and not visited[r][c]:
                water[r][c] = 1
                water[(r + dr) % rows][(c + dc) % cols] = 1

    assert visited[h1[0]][h1[1]], f"Hill 1 unreachable in map {map_id}"

    # Flatten and RLE water
    flat_water = []
    for r in range(rows):
        for c in range(cols):
            flat_water.append(water[r][c])
    water_rle = to_rle(flat_water)

    # 5. Place symmetric food
    land_cells = []
    for r in range(rows):
        for c in range(cols):
            if water[r][c] == 0 and [r, c] not in hills:
                land_cells.append((r, c))

    # Pick orbits of size 2
    orbits = []
    seen = set()
    for r, c in land_cells:
        if (r, c) not in seen:
            r2, c2 = (r + dr) % rows, (c + dc) % cols
            if (r2, c2) != (r, c) and water[r2][c2] == 0 and [r2, c2] not in hills:
                orbits.append(((r, c), (r2, c2)))
                seen.add((r, c))
                seen.add((r2, c2))

    rng.shuffle(orbits)
    needed_orbits = food_count // 2
    picked_orbits = orbits[:needed_orbits]
    food = []
    for p1, p2 in picked_orbits:
        food.append(list(p1))
        food.append(list(p2))
    food.sort()

    return {
        "id": map_id,
        "rows": rows,
        "cols": cols,
        "players": 2,
        "symmetry": {"dr": dr, "dc": dc},
        "hills": hills,
        "food": food,
        "food_target": len(food),
        "water": water_rle,
    }


# Terrain pattern generators
def pat_open(w, rows, cols, dr, dc):
    # Open terrain: very few isolated small water ponds
    for r in range(rows):
        for c in range(cols):
            if (r % 6 == 0 and c % 6 == 0):
                w[r][c] = 1


def pat_colonnade(w, rows, cols, dr, dc):
    # Colonnade: periodic columns of obstacles (2x2 water blocks spaced out)
    for r in range(rows):
        for c in range(cols):
            if (r % 6 in (1, 2)) and (c % 6 in (1, 2)):
                w[r][c] = 1


def pat_corridor(w, rows, cols, dr, dc):
    # Corridor: walls with periodic gaps creating choke corridors
    for r in range(rows):
        for c in range(cols):
            if r % 8 == 4 and c % 6 not in (0, 1):
                w[r][c] = 1


def pat_arena(w, rows, cols, dr, dc):
    # Arena: wide open center with surrounding arena ring barriers
    for r in range(rows):
        for c in range(cols):
            if (r % 16 in (0, 15) or c % 16 in (0, 15)) and (r % 4 not in (1, 2) and c % 4 not in (1, 2)):
                w[r][c] = 1


def pat_islands(w, rows, cols, dr, dc):
    # Islands: circular lakes leaving wide land bridges
    for r in range(rows):
        for c in range(cols):
            # Lakes centered at (4 + 8*k, 4 + 8*m) of radius 2
            lr, lc = (r % 8) - 4, (c % 8) - 4
            if lr * lr + lc * lc <= 2:
                w[r][c] = 1


def pat_rooms(w, rows, cols, dr, dc):
    # Rooms: 8x8 rooms with wide 4-cell doorways
    for r in range(rows):
        for c in range(cols):
            if (r % 8 == 0 and c % 8 not in (2, 3, 4, 5)) or (c % 8 == 0 and r % 8 not in (2, 3, 4, 5)):
                w[r][c] = 1


def pat_valley(w, rows, cols, dr, dc):
    # Valley: ridge walls along diagonal with wide openings
    for r in range(rows):
        for c in range(cols):
            if ((r + c) % 10 == 0) and ((r - c) % 10 not in (2, 3, 4, 5, 6, 7)):
                w[r][c] = 1


def pat_labyrinth(w, rows, cols, dr, dc):
    # Labyrinth: alternating maze teeth with wide turning bays
    for r in range(rows):
        for c in range(cols):
            if (r % 4 == 0 and c % 8 not in (0, 1, 2)) or (c % 4 == 0 and r % 8 not in (4, 5, 6)):
                w[r][c] = 1


def pat_crossroads(w, rows, cols, dr, dc):
    # Crossroads: intersecting boulevards with corner bastions
    for r in range(rows):
        for c in range(cols):
            if r % 20 in (8, 9, 10, 11) or c % 20 in (8, 9, 10, 11):
                continue
            if (r % 5 in (1, 2)) and (c % 5 in (1, 2)):
                w[r][c] = 1


MAP_CONFIGS = [
    # 3 Tiny (24x24)
    ("tiny-open-2p-1h", 24, 24, (4, 4), pat_open, 16, 101),
    ("tiny-colonnade-2p-1h", 24, 24, (6, 18), pat_colonnade, 16, 102),
    ("tiny-corridor-2p-1h", 24, 24, (2, 2), pat_corridor, 16, 103),

    # 3 Small (32x32)
    ("small-arena-2p-1h", 32, 32, (6, 6), pat_arena, 24, 201),
    ("small-islands-2p-1h", 32, 32, (5, 5), pat_islands, 24, 202),
    ("small-rooms-2p-1h", 32, 32, (4, 4), pat_rooms, 24, 203),

    # 3 Medium (40x40)
    ("medium-valley-2p-1h", 40, 40, (8, 8), pat_valley, 32, 301),
    ("medium-labyrinth-2p-1h", 40, 40, (6, 6), pat_labyrinth, 32, 302),
    ("medium-crossroads-2p-1h", 40, 40, (10, 10), pat_crossroads, 32, 303),
]


def main():
    out_dir = Path("/home/host/project/ants/benchmarks/maps_2p")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating 9 benchmark maps in {out_dir}...")
    for map_id, rows, cols, hill, pat_fn, food_cnt, seed in MAP_CONFIGS:
        map_data = create_map(map_id, rows, cols, hill, pat_fn, food_cnt, seed)
        file_path = out_dir / f"{map_id}.json"
        file_path.write_text(json.dumps(map_data, separators=(',', ':')))
        print(f"  - Generated {file_path.name} ({rows}x{cols}, {food_cnt} food)")


if __name__ == "__main__":
    main()
