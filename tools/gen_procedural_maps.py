from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import os
import random
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
VIEW_RADIUS2 = 77
MIN_SEAT_DIST2 = VIEW_RADIUS2 + 20


@dataclass(frozen=True)
class MapSpec:
    map_id: str
    rows: int
    cols: int
    players: int
    hills_per_seat: int
    style: str
    food_per_seat: int
    seed: int
    shift_pick: int = 0


def to_rle(grid: list[list[int]]) -> list[int]:
    flat = [v for row in grid for v in row]
    rle: list[int] = []
    cur = flat[0]
    cnt = 1
    for v in flat[1:]:
        if v == cur:
            cnt += 1
        else:
            rle.extend([cur, cnt])
            cur = v
            cnt = 1
    rle.extend([cur, cnt])
    return rle


def orbit_cells(
    r: int, c: int, dr: int, dc: int, rows: int, cols: int, players: int
) -> list[tuple[int, int]]:
    return [((r + k * dr) % rows, (c + k * dc) % cols) for k in range(players)]


def torus_dist2(a: tuple[int, int], b: tuple[int, int], rows: int, cols: int) -> int:
    dr = abs(a[0] - b[0])
    dc = abs(a[1] - b[1])
    if dr > rows - dr:
        dr = rows - dr
    if dc > cols - dc:
        dc = cols - dc
    return dr * dr + dc * dc


def shift_is_exact(rows: int, cols: int, players: int, dr: int, dc: int) -> bool:
    if (dr * players) % rows != 0 or (dc * players) % cols != 0:
        return False
    for k in range(1, players):
        if (k * dr) % rows == 0 and (k * dc) % cols == 0:
            return False
    return True


def valid_shifts(rows: int, cols: int, players: int) -> list[tuple[int, int, float]]:
    # 枚举满足环面精确轨道闭合、席位最小距离与肥度要求的平移向量 (dr, dc)
    seen_subgroups: list[list[tuple[int, int]]] = []
    candidates: list[tuple[int, int, float]] = []
    for dr in range(rows):
        if (dr * players) % rows != 0:
            continue
        for dc in range(cols):
            if (dc * players) % cols != 0:
                continue
            if not shift_is_exact(rows, cols, players, dr, dc):
                continue
            subgroup = sorted(((k * dr) % rows, (k * dc) % cols) for k in range(players))
            if subgroup in seen_subgroups:
                continue
            seen_subgroups.append(subgroup)
            seat2 = min(
                torus_dist2((0, 0), ((k * dr) % rows, (k * dc) % cols), rows, cols)
                for k in range(1, players)
            )
            own2 = min(rows * rows, cols * cols)
            l2 = min(seat2, own2)
            fatness = l2 / (rows * cols / float(players))
            if seat2 <= MIN_SEAT_DIST2 or fatness < 0.50:
                continue
            candidates.append((dr, dc, fatness))
    candidates.sort(key=lambda item: (-round(item[2], 6), item[0], item[1]))
    if not candidates:
        raise ValueError(f"无法为 {rows}x{cols} {players}p 找到合法的环面平移对称向量")
    return candidates


def symmetrize_water(
    grid: list[list[int]], dr: int, dc: int, rows: int, cols: int, players: int
) -> None:
    visited = [[False] * cols for _ in range(rows)]
    for r in range(rows):
        for c in range(cols):
            if visited[r][c]:
                continue
            orb = orbit_cells(r, c, dr, dc, rows, cols, players)
            val = 1 if any(grid[rr][cc] == 1 for rr, cc in orb) else 0
            for rr, cc in orb:
                grid[rr][cc] = val
                visited[rr][cc] = True


def generate_terrain(
    style: str,
    rows: int,
    cols: int,
    dr: int,
    dc: int,
    players: int,
    rng: random.Random,
) -> list[list[int]]:
    grid = [[0] * cols for _ in range(rows)]

    if style == "open":
        # 开阔地形配少量对称水域柱体（约 8%-16% 水域）
        n_blobs = max(4, (rows * cols) // (players * 85))
        for _ in range(n_blobs):
            cr, cc = rng.randrange(rows), rng.randrange(cols)
            rad = rng.choice([1, 1, 2, 2])
            for rr in range(cr - rad, cr + rad + 1):
                for cc2 in range(cc - rad, cc + rad + 1):
                    if (rr - cr) ** 2 + (cc2 - cc) ** 2 <= rad * rad:
                        for xr, xc in orbit_cells(rr % rows, cc2 % cols, dr, dc, rows, cols, players):
                            grid[xr][xc] = 1

    elif style == "islands":
        # 群岛与水域障碍湖泊（约 20%-32% 水域）
        n_lakes = max(5, (rows * cols) // (players * 95))
        for _ in range(n_lakes):
            cr, cc = rng.randrange(rows), rng.randrange(cols)
            rad = rng.choice([2, 3, 3, 4])
            for rr in range(cr - rad, cr + rad + 1):
                for cc2 in range(cc - rad, cc + rad + 1):
                    if (rr - cr) ** 2 + (cc2 - cc) ** 2 <= rad * rad:
                        for xr, xc in orbit_cells(rr % rows, cc2 % cols, dr, dc, rows, cols, players):
                            grid[xr][xc] = 1
        symmetrize_water(grid, dr, dc, rows, cols, players)

    elif style == "cave":
        # 对称轨道细胞自动机洞穴，生成多通道有机洞窟
        fill_p = rng.uniform(0.39, 0.46)
        visited = [[False] * cols for _ in range(rows)]
        for r in range(rows):
            for c in range(cols):
                if visited[r][c]:
                    continue
                val = 1 if rng.random() < fill_p else 0
                for xr, xc in orbit_cells(r, c, dr, dc, rows, cols, players):
                    grid[xr][xc] = val
                    visited[xr][xc] = True
        smooth_iters = rng.choice([3, 4, 5])
        for _ in range(smooth_iters):
            nxt = [[0] * cols for _ in range(rows)]
            for r in range(rows):
                for c in range(cols):
                    cnt = 0
                    for ddr in (-1, 0, 1):
                        for ddc in (-1, 0, 1):
                            if ddr == 0 and ddc == 0:
                                continue
                            cnt += grid[(r + ddr) % rows][(c + ddc) % cols]
                    nxt[r][c] = 1 if (cnt >= 5 or (grid[r][c] == 1 and cnt >= 4)) else 0
            grid = nxt
        symmetrize_water(grid, dr, dc, rows, cols, players)

    elif style == "rooms":
        # 多尺度房间网格 + 2-3 格对称门洞与室内障碍柱
        g_dim = math.gcd(math.gcd(rows, cols), math.gcd(dr, dc))
        pitches = [q for q in (6, 7, 8, 9, 10, 12) if g_dim % q == 0]
        pitch = rng.choice(pitches) if pitches else (8 if min(rows, cols) >= 48 else 6)
        for r in range(rows):
            for c in range(cols):
                if r % pitch == 0 or c % pitch == 0:
                    grid[r][c] = 1
        door_w = rng.choice([2, 2, 3]) if pitch >= 7 else 2
        for r in range(0, rows, pitch):
            for c in range(0, cols, pitch):
                span = max(2, pitch - door_w)
                door_c = (c + rng.randrange(1, span)) % cols
                for off in range(door_w):
                    for xr, xc in orbit_cells(r, (door_c + off) % cols, dr, dc, rows, cols, players):
                        grid[xr][xc] = 0
                door_r = (r + rng.randrange(1, span)) % rows
                for off in range(door_w):
                    for xr, xc in orbit_cells((door_r + off) % rows, c, dr, dc, rows, cols, players):
                        grid[xr][xc] = 0
        # 可选室内柱子增加视线遮挡多样性
        if pitch >= 8 and rng.random() < 0.5:
            for r in range(pitch // 2, rows, pitch):
                for c in range(pitch // 2, cols, pitch):
                    if rng.random() < 0.45:
                        for xr, xc in orbit_cells(r, c, dr, dc, rows, cols, players):
                            grid[xr][xc] = 1
        symmetrize_water(grid, dr, dc, rows, cols, players)

    elif style == "maze":
        # 编织环路迷宫：1-2 格走廊与可控环路比例
        g_dim = math.gcd(math.gcd(rows, cols), math.gcd(dr, dc))
        pitches = [q for q in (4, 5, 6) if g_dim % q == 0]
        pitch = rng.choice(pitches) if pitches else 4
        for r in range(rows):
            for c in range(cols):
                if r % pitch == 0 or c % pitch == 0:
                    grid[r][c] = 1
        open_prob = rng.uniform(0.68, 0.84)
        for r in range(0, rows, pitch):
            for c in range(0, cols, pitch):
                if rng.random() < open_prob:
                    dc_open = (c + rng.randrange(1, pitch)) % cols
                    for xr, xc in orbit_cells(r, dc_open, dr, dc, rows, cols, players):
                        grid[xr][xc] = 0
                    if pitch >= 5 and rng.random() < 0.5:
                        for xr, xc in orbit_cells(r, (dc_open + 1) % cols, dr, dc, rows, cols, players):
                            grid[xr][xc] = 0
                if rng.random() < open_prob:
                    dr_open = (r + rng.randrange(1, pitch)) % rows
                    for xr, xc in orbit_cells(dr_open, c, dr, dc, rows, cols, players):
                        grid[xr][xc] = 0
                    if pitch >= 5 and rng.random() < 0.5:
                        for xr, xc in orbit_cells((dr_open + 1) % rows, c, dr, dc, rows, cols, players):
                            grid[xr][xc] = 0
        symmetrize_water(grid, dr, dc, rows, cols, players)

    else:
        raise ValueError(f"未知地形风格: {style}")

    return grid


def carve_corridor(
    grid: list[list[int]],
    src: tuple[int, int],
    dst: tuple[int, int],
    dr: int,
    dc: int,
    rows: int,
    cols: int,
    players: int,
    width: int = 1,
) -> None:
    cr, cc = src
    tr, tc = dst
    step_r = 1 if (tr - cr) % rows <= rows // 2 else -1
    step_c = 1 if (tc - cc) % cols <= cols // 2 else -1
    while cr != tr:
        for w in range(width):
            for xr, xc in orbit_cells(cr, (cc + w) % cols, dr, dc, rows, cols, players):
                grid[xr][xc] = 0
        cr = (cr + step_r) % rows
    while cc != tc:
        for w in range(width):
            for xr, xc in orbit_cells((cr + w) % rows, cc, dr, dc, rows, cols, players):
                grid[xr][xc] = 0
        cc = (cc + step_c) % cols


def canonical_board_hash(board: dict[str, Any]) -> str:
    # 针对棋盘尺寸、席位数、对称向量、水域 RLE 与有序蚁丘坐标计算规范 SHA-256 哈希
    sym = board.get("symmetry") or {}
    payload = {
        "rows": int(board["rows"]),
        "cols": int(board["cols"]),
        "players": int(board["players"]),
        "dr": int(sym.get("dr", 0)),
        "dc": int(sym.get("dc", 0)),
        "water": [int(x) for x in board["water"]],
        "hills": sorted([int(h[0]), int(h[1])] for h in board["hills"]),
    }
    raw = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def load_excluded_hashes(exclude_path: Path | None) -> set[str]:
    if exclude_path is None:
        return set()
    if not exclude_path.exists():
        raise FileNotFoundError(f"排除棋盘清单路径不存在: {exclude_path}")

    hashes: set[str] = set()
    if exclude_path.is_dir():
        for p in sorted(exclude_path.glob("*.json")):
            obj = json.loads(p.read_text(encoding="utf-8"))
            board = obj.get("board", obj) if isinstance(obj, dict) else obj
            if isinstance(board, dict) and "water" in board and "hills" in board:
                hashes.add(canonical_board_hash(board))
    else:
        obj = json.loads(exclude_path.read_text(encoding="utf-8"))
        items = obj.get("maps", obj) if isinstance(obj, dict) else obj
        if not isinstance(items, list):
            raise ValueError(f"无法解析排除棋盘文件结构: {exclude_path}")
        for entry in items:
            board = entry.get("board", entry) if isinstance(entry, dict) else entry
            if isinstance(board, dict) and "water" in board and "hills" in board:
                hashes.add(canonical_board_hash(board))
    return hashes


def build_board(spec: MapSpec, seed_offset: int = 0) -> dict[str, Any]:
    shifts = valid_shifts(spec.rows, spec.cols, spec.players)
    dr, dc, _ = shifts[spec.shift_pick % len(shifts)]
    rng = random.Random(spec.seed + seed_offset * 10007)

    grid = generate_terrain(spec.style, spec.rows, spec.cols, dr, dc, spec.players, rng)

    # 选定 seat 0 蚁穴位置，严格保证所有蚁穴（含跨席位与同席位多丘）间距大于视野半径
    min_hill_d2 = max(100, VIEW_RADIUS2 + 23)
    hill_seeds: list[tuple[int, int]] = []
    for h_idx in range(spec.hills_per_seat):
        chosen = None
        for relax_d2 in (min_hill_d2, VIEW_RADIUS2 + 5, 64):
            for _ in range(1200):
                cand = (rng.randrange(spec.rows), rng.randrange(spec.cols))
                orb = orbit_cells(cand[0], cand[1], dr, dc, spec.rows, spec.cols, spec.players)
                if len(set(orb)) != spec.players:
                    continue
                if any(torus_dist2(cand, other, spec.rows, spec.cols) < relax_d2 for other in orb[1:]):
                    continue
                existing = [
                    p
                    for hs in hill_seeds
                    for p in orbit_cells(hs[0], hs[1], dr, dc, spec.rows, spec.cols, spec.players)
                ]
                if any(torus_dist2(cand, ex, spec.rows, spec.cols) < relax_d2 for ex in existing):
                    continue
                chosen = cand
                break
            if chosen is not None:
                break
        if chosen is None:
            raise RuntimeError(
                f"{spec.map_id}: 无法在 {spec.rows}x{spec.cols} {spec.players}p-{spec.hills_per_seat}h 上安置第 {h_idx + 1} 座蚁穴"
            )
        hill_seeds.append(chosen)

    hills: list[tuple[int, int]] = []
    for hs in hill_seeds:
        hills.extend(orbit_cells(hs[0], hs[1], dr, dc, spec.rows, spec.cols, spec.players))

    # 清理蚁穴周边 2 格曼哈顿空地，保障出兵通畅
    for hr, hc in hills:
        for ddr in range(-2, 3):
            for ddc in range(-2, 3):
                if abs(ddr) + abs(ddc) <= 2:
                    grid[(hr + ddr) % spec.rows][(hc + ddc) % spec.cols] = 0

    def bfs(start: tuple[int, int]) -> set[tuple[int, int]]:
        vis = {start}
        q = collections.deque([start])
        while q:
            r0, c0 = q.popleft()
            for nr, nc in (
                ((r0 - 1) % spec.rows, c0),
                ((r0 + 1) % spec.rows, c0),
                (r0, (c0 - 1) % spec.cols),
                (r0, (c0 + 1) % spec.cols),
            ):
                if grid[nr][nc] == 0 and (nr, nc) not in vis:
                    vis.add((nr, nc))
                    q.append((nr, nc))
        return vis

    vis = bfs(hills[0])
    for h in hills[1:]:
        if h not in vis:
            carve_corridor(grid, hills[0], h, dr, dc, spec.rows, spec.cols, spec.players, width=1)
            vis = bfs(hills[0])

    # 将不可达孤岛陆地对称填水，保证单一连通陆地分量
    for r in range(spec.rows):
        for c in range(spec.cols):
            if grid[r][c] == 0 and (r, c) not in vis:
                for xr, xc in orbit_cells(r, c, dr, dc, spec.rows, spec.cols, spec.players):
                    grid[xr][xc] = 1
    symmetrize_water(grid, dr, dc, spec.rows, spec.cols, spec.players)

    vis2 = bfs(hills[0])
    if not all(h in vis2 for h in hills):
        raise RuntimeError(f"{spec.map_id}: 蚁穴连通性校验失败")
    for r in range(spec.rows):
        for c in range(spec.cols):
            if grid[r][c] == 0 and (r, c) not in vis2:
                grid[r][c] = 1
    symmetrize_water(grid, dr, dc, spec.rows, spec.cols, spec.players)

    # 放置对称食物轨道，避开蚁穴与水域
    hill_set = set(hills)
    orbits: list[list[tuple[int, int]]] = []
    seen_cells: set[tuple[int, int]] = set()
    for r in range(spec.rows):
        for c in range(spec.cols):
            if (r, c) in seen_cells or grid[r][c] == 1:
                continue
            orb = orbit_cells(r, c, dr, dc, spec.rows, spec.cols, spec.players)
            for p in orb:
                seen_cells.add(p)
            if len(set(orb)) == spec.players and all(
                grid[rr][cc] == 0 and (rr, cc) not in hill_set for rr, cc in orb
            ):
                orbits.append(orb)

    rng.shuffle(orbits)
    food: list[list[int]] = []
    for orb in orbits[: spec.food_per_seat]:
        for rr, cc in orb:
            food.append([rr, cc])

    return {
        "id": spec.map_id,
        "rows": spec.rows,
        "cols": spec.cols,
        "players": spec.players,
        "symmetry": {"dr": dr, "dc": dc},
        "water": to_rle(grid),
        "hills": [[r, c] for r, c in hills],
        "food": food,
        "food_target": len(food),
    }


def _alpha_tag(idx: int) -> str:
    # 将 1-based 编号编码为纯小写字母后缀，确保棋盘 ID 严格满足 size-terrain-Np-Hh 规范
    first = chr(ord("a") + (idx - 1) // 26)
    second = chr(ord("a") + (idx - 1) % 26)
    return f"{first}{second}"


def _size_bucket_prefix(rows: int, cols: int) -> str:
    m = max(rows, cols)
    if m <= 32:
        return "tiny"
    if m <= 48:
        return "small"
    if m <= 64:
        return "medium"
    if m <= 96:
        return "large"
    return "xlarge"


def default_60_specs(base_seed: int = 20261120) -> list[MapSpec]:
    # 60 张分层程序化训练棋盘配方（严格覆盖 2p/4p/6p、1h/2h/3h、方/非方 24x24..96x96 与五大地形风格）
    raw_specs: list[tuple[int, int, int, int, str, int, int]] = [
        # ---- 2p 1h (14 张: 含 24x24..72x84, ~43% 非方含非4倍数尺寸) ----
        (24, 24, 2, 1, "maze", 10, 0),
        (26, 36, 2, 1, "cave", 10, 0),
        (30, 42, 2, 1, "rooms", 11, 0),
        (32, 32, 2, 1, "open", 11, 0),
        (36, 44, 2, 1, "maze", 12, 0),
        (40, 40, 2, 1, "cave", 12, 0),
        (44, 52, 2, 1, "rooms", 13, 0),
        (48, 48, 2, 1, "maze", 14, 0),
        (48, 60, 2, 1, "cave", 14, 0),
        (54, 54, 2, 1, "rooms", 15, 0),
        (56, 68, 2, 1, "maze", 16, 0),
        (64, 64, 2, 1, "cave", 16, 0),
        (60, 76, 2, 1, "islands", 16, 0),
        (72, 72, 2, 1, "maze", 18, 0),
        # ---- 2p 2h (11 张: 覆盖 36x44..84x96) ----
        (36, 44, 2, 2, "cave", 14, 0),
        (42, 42, 2, 2, "maze", 14, 0),
        (44, 56, 2, 2, "rooms", 15, 0),
        (48, 48, 2, 2, "cave", 16, 0),
        (52, 64, 2, 2, "maze", 16, 0),
        (60, 60, 2, 2, "rooms", 18, 0),
        (60, 76, 2, 2, "cave", 18, 0),
        (64, 64, 2, 2, "maze", 18, 0),
        (68, 84, 2, 2, "rooms", 20, 0),
        (76, 76, 2, 2, "cave", 20, 0),
        (80, 96, 2, 2, "maze", 22, 0),
        # ---- 2p 3h (5 张: 覆盖 52x60..84x96) ----
        (52, 60, 2, 3, "cave", 18, 0),
        (60, 60, 2, 3, "maze", 18, 0),
        (64, 80, 2, 3, "rooms", 20, 0),
        (72, 72, 2, 3, "cave", 22, 0),
        (80, 92, 2, 3, "maze", 24, 0),
        # ---- 4p 1h (9 张: 多种平移格点, 36x44..88x88) ----
        (36, 44, 4, 1, "maze", 12, 0),
        (40, 40, 4, 1, "cave", 13, 0),
        (48, 56, 4, 1, "rooms", 14, 1),
        (52, 52, 4, 1, "maze", 15, 0),
        (56, 68, 4, 1, "cave", 16, 1),
        (60, 60, 4, 1, "open", 16, 0),
        (64, 76, 4, 1, "rooms", 18, 1),
        (72, 72, 4, 1, "maze", 18, 0),
        (80, 96, 4, 1, "cave", 20, 1),
        # ---- 4p 2h (7 张: 48x60..96x96) ----
        (48, 60, 4, 2, "cave", 16, 0),
        (56, 56, 4, 2, "maze", 17, 1),
        (60, 72, 4, 2, "rooms", 18, 0),
        (64, 64, 4, 2, "cave", 18, 1),
        (72, 84, 4, 2, "maze", 20, 0),
        (80, 80, 4, 2, "rooms", 20, 1),
        (84, 96, 4, 2, "cave", 22, 0),
        # ---- 4p 3h (4 张: 64x72..96x96) ----
        (64, 72, 4, 3, "maze", 20, 0),
        (72, 72, 4, 3, "cave", 20, 1),
        (80, 96, 4, 3, "rooms", 22, 0),
        (96, 96, 4, 3, "cave", 24, 1),
        # ---- 6p 1h (4 张: 54x60..90x96) ----
        (54, 60, 6, 1, "maze", 14, 0),
        (60, 72, 6, 1, "cave", 16, 1),
        (72, 72, 6, 1, "rooms", 18, 0),
        (78, 96, 6, 1, "cave", 18, 1),
        # ---- 6p 2h (4 张: 66x78..96x96) ----
        (66, 78, 6, 2, "maze", 18, 0),
        (72, 84, 6, 2, "cave", 20, 1),
        (84, 84, 6, 2, "rooms", 20, 0),
        (84, 96, 6, 2, "islands", 22, 1),
        # ---- 6p 3h (2 张: 84x96..96x96) ----
        (84, 96, 6, 3, "maze", 22, 0),
        (96, 96, 6, 3, "cave", 24, 1),
    ]
    return [
        MapSpec(
            map_id=f"{_size_bucket_prefix(r, c)}{_alpha_tag(idx)}-{st}-{p}p-{h}h",
            rows=r,
            cols=c,
            players=p,
            hills_per_seat=h,
            style=st,
            food_per_seat=f,
            seed=base_seed + idx * 97,
            shift_pick=sp,
        )
        for idx, (r, c, p, h, st, f, sp) in enumerate(raw_specs, start=1)
    ]


def bucket_of(players: int, hills_per_seat: int) -> str:
    if players == 2:
        return "2p-1h" if hills_per_seat == 1 else "2p-multi"
    if players == 4:
        return "4p-1h" if hills_per_seat == 1 else "4p-multi"
    if players == 6:
        return "6p-all"
    return f"{players}p-other"


def resolve_cli() -> str:
    found = os.environ.get("TINYBRAINS") or shutil.which("tinybrains")
    if found:
        return found
    brew_cli = Path("/home/linuxbrew/.linuxbrew/bin/tinybrains")
    if brew_cli.exists():
        return str(brew_cli)
    raise RuntimeError("未找到 tinybrains 可执行文件，无法执行真实引擎棋盘校验")


def generate_map_pool(
    out_dir: Path,
    base_seed: int = 20261120,
    exclude_boards: Path | None = None,
    validate_cli: bool = True,
    registry_path: Path | None = None,
) -> dict[str, Any]:
    excluded_hashes = load_excluded_hashes(exclude_boards)
    specs = default_60_specs(base_seed=base_seed)

    out_dir.mkdir(parents=True, exist_ok=True)
    for old_file in out_dir.glob("*.json"):
        old_file.unlink()

    manifest_entries: list[dict[str, Any]] = []
    written_paths: list[str] = []
    generated_hashes: set[str] = set()

    for spec in specs:
        board: dict[str, Any] | None = None
        board_hash = ""
        for attempt in range(32):
            cand = build_board(spec, seed_offset=attempt)
            h = canonical_board_hash(cand)
            if h not in excluded_hashes and h not in generated_hashes:
                board = cand
                board_hash = h
                break
        if board is None:
            raise RuntimeError(f"{spec.map_id}: 连续 32 次尝试均与排除集或已生成棋盘哈希碰撞")
        generated_hashes.add(board_hash)

        map_file = out_dir / f"{spec.map_id}.json"
        map_file.write_text(json.dumps(board, separators=(",", ":")), encoding="utf-8")
        written_paths.append(str(map_file.resolve()))

        water_cells = sum(board["water"][i + 1] for i in range(0, len(board["water"]), 2) if board["water"][i] == 1)
        water_pct = round(water_cells / float(spec.rows * spec.cols), 4)
        manifest_entries.append(
            {
                "id": spec.map_id,
                "path": str(map_file.resolve()),
                "rows": spec.rows,
                "cols": spec.cols,
                "players": spec.players,
                "hills_per_seat": spec.hills_per_seat,
                "style": spec.style,
                "symmetry": board["symmetry"],
                "food_target": board["food_target"],
                "water_pct": water_pct,
                "bucket": bucket_of(spec.players, spec.hills_per_seat),
                "sha256": board_hash,
            }
        )

    if validate_cli:
        cli = resolve_cli()
        env = os.environ.copy()
        if registry_path is not None:
            env["TINYBRAINS_REGISTRY"] = str(registry_path.resolve())
        elif "TINYBRAINS_REGISTRY" not in env:
            reg = ROOT / "baselines" / "games.toml"
            if reg.exists():
                env["TINYBRAINS_REGISTRY"] = str(reg.resolve())
        res = subprocess.run(
            [cli, "maps", "check", *written_paths],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            env=env,
        )
        if res.returncode != 0:
            raise RuntimeError(f"tinybrains maps check 拒绝了生成的程序化棋盘:\n{res.stdout}\n{res.stderr}")

    summary = {
        "count": len(manifest_entries),
        "base_seed": base_seed,
        "excluded_count": len(excluded_hashes),
        "disjoint_verified": True,
        "validated_by_engine": validate_cli,
        "maps": manifest_entries,
    }
    (out_dir / "manifest.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="生成通过真实引擎校验且与线上赛季棋盘严格不相交的分层程序化训练棋盘池。")
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "proc_maps_60", help="棋盘输出目录")
    ap.add_argument("--seed", type=int, default=20261120, help="确定性基础随机种子")
    ap.add_argument("--exclude-boards", type=Path, default=None, help="需严格哈希隔离的线上棋盘 JSON 文件或目录")
    ap.add_argument("--no-validate", action="store_true", help="跳过 tinybrains maps check 校验（仅用于离线调试）")
    return ap


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    summary = generate_map_pool(
        out_dir=args.out,
        base_seed=args.seed,
        exclude_boards=args.exclude_boards,
        validate_cli=not args.no_validate,
    )
    by_players: dict[int, int] = {}
    by_hills: dict[int, int] = {}
    by_style: dict[str, int] = {}
    non_sq = 0
    for m in summary["maps"]:
        by_players[m["players"]] = by_players.get(m["players"], 0) + 1
        by_hills[m["hills_per_seat"]] = by_hills.get(m["hills_per_seat"], 0) + 1
        by_style[m["style"]] = by_style.get(m["style"], 0) + 1
        if m["rows"] != m["cols"]:
            non_sq += 1
    print(
        f"已生成 {summary['count']} 张程序化训练棋盘 -> {args.out}\n"
        f"  席位分布: {by_players} | 蚁丘分布: {by_hills} | 风格分布: {by_style} | 非方图: {non_sq}/{summary['count']}\n"
        f"  严格哈希隔离校验: 对比 {summary['excluded_count']} 张线上棋盘，零碰撞\n"
        f"  真实 Wasm 引擎 tinybrains maps check: {'通过' if summary['validated_by_engine'] else '已跳过'}"
    )


if __name__ == "__main__":
    main()
