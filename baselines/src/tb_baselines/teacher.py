from __future__ import annotations
from enum import Enum

import numpy as np

from .planes import MOVES
from .telemetry import (
    FieldMetadata,
    RejectionCode,
    SOURCE_FOOD_MEMORY,
    SOURCE_FOOD_VISIBLE,
    SOURCE_FRONTIER_UNSEEN,
    SOURCE_HILL_FOE,
    SOURCE_NONE,
    TelemetryRecord,
    evaluate_ant_telemetry,
)

# 行优先偏移量，顺序与 `MOVES` 完全一致，因此索引可直接对应。
DELTAS = ((-1, 0), (0, 1), (1, 0), (0, -1))
HOLD = MOVES.index("-")

FAR = np.int32(1 << 29)


class TaskState(Enum):
    GATHER = "gather"
    EXPLORE = "explore"
    ASSAULT = "assault"
    FORTIFY = "fortify"



def extract_food_gathering_points(
    food_coords: list[tuple[int, int]] | list[list[int]] | set[tuple[int, int]] | np.ndarray | None,
    water_mask: np.ndarray,
    rows: int,
    cols: int,
) -> np.ndarray:
    """提取当前可见食物的正交相邻合法采集位集合 G_food (EXP-071A)。"""
    if food_coords is None or len(food_coords) == 0:
        return np.zeros((rows, cols), dtype=bool)

    food_mask = np.zeros((rows, cols), dtype=bool)
    if isinstance(food_coords, np.ndarray) and food_coords.size > 0:
        food_mask[food_coords[:, 0], food_coords[:, 1]] = True
    else:
        for item in food_coords:
            r, c = int(item[0]), int(item[1])
            if 0 <= r < rows and 0 <= c < cols:
                food_mask[r, c] = True

    if not food_mask.any():
        return np.zeros((rows, cols), dtype=bool)

    neighbors = (
        np.roll(food_mask, 1, 0)
        | np.roll(food_mask, -1, 0)
        | np.roll(food_mask, 1, 1)
        | np.roll(food_mask, -1, 1)
    )
    passable = (~water_mask) & (~food_mask)
    return neighbors & passable


def extract_memory_food_gathering_points(
    memory_food_coords: list[list[int]] | list[tuple[int, int]] | set[tuple[int, int]] | np.ndarray | None,
    water_mask: np.ndarray,
    visible_food_mask: np.ndarray,
    g_food: np.ndarray,
    rows: int,
    cols: int,
) -> np.ndarray:
    """提取历史记忆食物的合法采集位集合 G_food_memory (EXP-071B)。"""
    if memory_food_coords is None or len(memory_food_coords) == 0:
        return np.zeros((rows, cols), dtype=bool)

    mem_mask = np.zeros((rows, cols), dtype=bool)
    if isinstance(memory_food_coords, np.ndarray) and memory_food_coords.size > 0:
        for r, c in memory_food_coords:
            r, c = int(r), int(c)
            if 0 <= r < rows and 0 <= c < cols:
                if not visible_food_mask[r, c] and not water_mask[r, c]:
                    mem_mask[r, c] = True
    else:
        for item in memory_food_coords:
            r, c = int(item[0]), int(item[1])
            if 0 <= r < rows and 0 <= c < cols:
                if not visible_food_mask[r, c] and not water_mask[r, c]:
                    mem_mask[r, c] = True

    if not mem_mask.any():
        return np.zeros((rows, cols), dtype=bool)

    neighbors = (
        np.roll(mem_mask, 1, 0)
        | np.roll(mem_mask, -1, 0)
        | np.roll(mem_mask, 1, 1)
        | np.roll(mem_mask, -1, 1)
    )
    passable = (~water_mask) & (~visible_food_mask)
    return neighbors & passable & (~g_food)


def extract_exploration_frontier(
    seen_mask: np.ndarray,
    water_mask: np.ndarray,
    rows: int,
    cols: int,
    food_mask: np.ndarray | None = None,
) -> np.ndarray:
    """提取战争迷雾探索前沿集合 G_frontier (EXP-073B)。

    严格限定在已知通行陆地：seen_mask & (~water_mask) & (~food_mask)，
    且在环面四邻域中至少邻接一个未知迷雾单元格。
    """
    seen = np.asarray(seen_mask, dtype=bool)
    water = np.asarray(water_mask, dtype=bool)
    unseen = ~seen
    adjacent_to_fog = (
        np.roll(unseen, 1, 0)
        | np.roll(unseen, -1, 0)
        | np.roll(unseen, 1, 1)
        | np.roll(unseen, -1, 1)
    )
    passable_known = seen & (~water)
    if food_mask is not None:
        food = np.asarray(food_mask, dtype=bool)
        passable_known = passable_known & (~food)
    return adjacent_to_fog & passable_known


def compute_geodesic_potential_field(
    sources: np.ndarray | list[tuple[int, int]] | set[tuple[int, int]] | dict[int, np.ndarray] | None,
    passable_mask: np.ndarray,
    rows: int,
    cols: int,
    max_depth: int | None = None,
    return_meta: bool = False,
    source_grid: np.ndarray | None = None,
    initial_source_type: int = SOURCE_NONE,
) -> np.ndarray | tuple[np.ndarray, FieldMetadata]:
    """计算环面网格上的最短测地距离场与遥测元数据 (EXP-073A)。

    采用向量化广度优先搜索扩散，支持自然波前队列耗尽。
    """
    dist = np.full((rows, cols), FAR, dtype=np.int32)
    if source_grid is None:
        source_grid = np.zeros((rows, cols), dtype=np.int32)

    at_level: dict[int, np.ndarray] = {}

    if isinstance(sources, dict):
        for lvl, mask in sources.items():
            lvl_int = int(lvl)
            mask_bool = np.asarray(mask, dtype=bool)
            if lvl_int in at_level:
                at_level[lvl_int] = at_level[lvl_int] | mask_bool
            else:
                at_level[lvl_int] = mask_bool.copy()
    elif sources is not None:
        init_mask = np.zeros((rows, cols), dtype=bool)
        if isinstance(sources, np.ndarray) and sources.dtype == bool and sources.shape == (rows, cols):
            init_mask = sources.copy()
        elif isinstance(sources, np.ndarray) and sources.ndim == 2 and sources.shape[1] == 2:
            if sources.size > 0:
                for r, c in sources:
                    r_i, c_i = int(r), int(c)
                    if 0 <= r_i < rows and 0 <= c_i < cols:
                        init_mask[r_i, c_i] = True
        else:
            for item in sources:
                r_i, c_i = int(item[0]), int(item[1])
                if 0 <= r_i < rows and 0 <= c_i < cols:
                    init_mask[r_i, c_i] = True

        if init_mask.any():
            at_level[0] = init_mask

    if initial_source_type != SOURCE_NONE and 0 in at_level:
        valid_src = at_level[0] & passable_mask
        source_grid[valid_src & (source_grid == SOURCE_NONE)] = initial_source_type

    if not at_level:
        if return_meta:
            meta = FieldMetadata(
                frontier_size=0,
                frontier_nonempty_at_limit=False,
                source_grid=source_grid,
            )
            return dist, meta
        return dist

    wave = np.zeros((rows, cols), dtype=bool)
    exhausted_early = False
    current_depth = 0
    min_lvl = min(at_level.keys())
    if min_lvl > 0:
        current_depth = min_lvl

    while wave.any() or at_level:
        if max_depth is not None and current_depth > max_depth:
            break

        step = at_level.pop(current_depth, None)
        if wave.any():
            spread = (
                np.roll(wave, 1, 0)
                | np.roll(wave, -1, 0)
                | np.roll(wave, 1, 1)
                | np.roll(wave, -1, 1)
            )
            step = spread if step is None else (spread | step)

        if step is None:
            if at_level:
                current_depth = min(at_level.keys())
                continue
            break

        step &= passable_mask & (dist == FAR)
        if not step.any():
            if not at_level:
                exhausted_early = True
                break
            wave = np.zeros((rows, cols), dtype=bool)
            if at_level:
                current_depth = min(at_level.keys())
            continue

        dist[step] = current_depth

        if return_meta and wave.any():
            spread_source = np.maximum.reduce([
                np.roll(source_grid, 1, 0),
                np.roll(source_grid, -1, 0),
                np.roll(source_grid, 1, 1),
                np.roll(source_grid, -1, 1),
            ])
            mask_to_update = step & (source_grid == SOURCE_NONE)
            source_grid[mask_to_update] = spread_source[mask_to_update]

        wave = step
        current_depth += 1

    if return_meta:
        if exhausted_early or not wave.any():
            frontier_size = 0
            frontier_nonempty_at_limit = False
        else:
            frontier_size = int(np.count_nonzero(wave))
            frontier_nonempty_at_limit = bool(frontier_size > 0 or bool(at_level))
        meta = FieldMetadata(
            frontier_size=frontier_size,
            frontier_nonempty_at_limit=frontier_nonempty_at_limit,
            source_grid=source_grid,
        )
        return dist, meta
    return dist


def arbitrate_orders(
    mine_coords: list[list[int]] | list[tuple[int, int]] | np.ndarray,
    candidate_sequences: list[list[int]],
    potentials: list[int] | np.ndarray,
    stall_counters: dict[int, int] | None = None,
    fair_mode: bool = True,
    rows: int = 0,
    cols: int = 0,
    return_iteration_count: bool = False,
    ids: list[int] | None = None,
    squad_tiers: list[int] | None = None,
    rejections_out: dict[int, dict[int, str]] | list[dict[int, str]] | None = None,
) -> tuple[np.ndarray, dict[int, int]] | tuple[np.ndarray, dict[int, int], int]:
    """终点冲突仲裁与调度公平性核心算法（EXP-072A 与 EXP-072B）。"""
    total_ants = len(mine_coords)
    if total_ants == 0:
        empty_orders = np.zeros(0, dtype=np.int64)
        empty_stalls: dict[int, int] = {}
        if return_iteration_count:
            return empty_orders, empty_stalls, 0
        return empty_orders, empty_stalls

    mine_list: list[tuple[int, int]] = [
        (int(m[0]), int(m[1])) for m in mine_coords
    ]

    if rows <= 0 or cols <= 0:
        max_r = max(m[0] for m in mine_list)
        max_c = max(m[1] for m in mine_list)
        rows = max(rows, max_r + 1)
        cols = max(cols, max_c + 1)

    if ids is None:
        ant_ids = list(range(total_ants))
    else:
        ant_ids = [int(x) for x in ids]

    live_ids = set(ant_ids)
    if stall_counters is not None:
        current_stalls = {
            int(k): int(v) for k, v in stall_counters.items() if int(k) in live_ids
        }
    else:
        current_stalls = {}
    for aid in ant_ids:
        if aid not in current_stalls:
            current_stalls[aid] = 0

    pointers = [0] * total_ants

    def get_dest(ant_idx: int, action: int) -> tuple[int, int]:
        if action == HOLD:
            return mine_list[ant_idx]
        dr, dc = DELTAS[action]
        return ((mine_list[ant_idx][0] + dr) % rows, (mine_list[ant_idx][1] + dc) % cols)

    def get_priority(i: int) -> tuple:
        aid = ant_ids[i]
        pot = int(potentials[i])
        tier = squad_tiers[i] if squad_tiers is not None else 0
        stall = current_stalls.get(aid, 0)
        if fair_mode:
            if squad_tiers is not None:
                return (tier, -stall, pot, aid)
            return (-stall, pot, aid)
        else:
            if squad_tiers is not None:
                return (tier, pot, aid)
            return (pot, aid)

    max_rounds = 4 * total_ants + 1
    iteration_count = 0

    while True:
        iteration_count += 1
        if iteration_count > max_rounds:
            raise RuntimeError(
                f"仲裁算法迭代轮次超过理论上限 4N: {iteration_count} > {max_rounds}"
            )

        dest_claims: dict[tuple[int, int], list[int]] = {}
        for i in range(total_ants):
            action = candidate_sequences[i][pointers[i]]
            dest = get_dest(i, action)
            dest_claims.setdefault(dest, []).append(i)

        losers: list[int] = []
        for dest, claimants in dest_claims.items():
            if len(claimants) <= 1:
                continue

            holding_ants = [
                i for i in claimants
                if candidate_sequences[i][pointers[i]] == HOLD and mine_list[i] == dest
            ]

            if holding_ants:
                winner = holding_ants[0]
            else:
                winner = min(claimants, key=get_priority)

            for i in claimants:
                if i != winner:
                    losers.append(i)
                    rej_act = candidate_sequences[i][pointers[i]]
                    if rej_act != HOLD and rejections_out is not None:
                        if isinstance(rejections_out, dict):
                            rejections_out.setdefault(i, {})[rej_act] = RejectionCode.TAKEN.value
                        elif isinstance(rejections_out, list):
                            rejections_out[i][rej_act] = RejectionCode.TAKEN.value

        if not losers:
            break

        for i in losers:
            pointers[i] += 1
            if pointers[i] >= len(candidate_sequences[i]):
                raise AssertionError(f"蚂蚁 {i} 候选序列被非法耗尽，缺少保底 HOLD")

    orders = np.array(
        [candidate_sequences[i][pointers[i]] for i in range(total_ants)],
        dtype=np.int64,
    )

    updated_stalls: dict[int, int] = {}
    for i in range(total_ants):
        aid = ant_ids[i]
        final_act = orders[i]
        if final_act != HOLD:
            updated_stalls[aid] = 0
        else:
            if pointers[i] > 0:
                updated_stalls[aid] = current_stalls.get(aid, 0) + 1
            else:
                updated_stalls[aid] = 0

    if stall_counters is not None:
        stall_counters.clear()
        stall_counters.update(updated_stalls)

    if return_iteration_count:
        return orders, updated_stalls, iteration_count
    return orders, updated_stalls


def get_known_enemy_hills(
    obs: dict,
    water: np.ndarray,
    seen: np.ndarray,
    memory: dict | None,
) -> set[tuple[int, int]]:
    """从当前观测与记忆中提取所有已知且未确认被摧毁的敌方蚁丘坐标。"""
    rows, cols = obs["size"]
    visible = seen & ~water
    hills: set[tuple[int, int]] = set()

    for r, c, owner in obs.get("hills", []):
        if owner != 0:
            hills.add((int(r), int(c)))

    if memory:
        for r, c in memory.get("hill_foe_seen", ()):
            r_i, c_i = int(r), int(c)
            if 0 <= r_i < rows and 0 <= c_i < cols and not visible[r_i, c_i]:
                hills.add((r_i, c_i))

    return hills


def _dispatch_task_state_machine(
    mine: list[list[int]] | list[tuple[int, int]],
    obs: dict,
    water: np.ndarray,
    seen: np.ndarray,
    memory: dict | None,
    max_depth: int | None,
    min_raid_ants: int | None = None,
    raid_ratio: float = 0.5,
    collect_telemetry: bool = False,
    stall_counters: dict[int, int] | None = None,
    fair_mode: bool = True,
) -> tuple[np.ndarray, list[TelemetryRecord]]:
    """显式任务分配状态机与终点冲突仲裁调度核心实现 (EXP-073B)。"""
    total_ants = len(mine)
    if total_ants == 0:
        return np.zeros(0, dtype=np.int64), []

    rows, cols = obs["size"]
    ant_positions = {(int(a[0]), int(a[1])) for a in mine}
    ant_ids_raw = obs.get("ids")
    if ant_ids_raw is None:
        ant_ids = list(range(total_ants))
    else:
        ant_ids = [int(x) for x in ant_ids_raw]

    food_coords = obs.get("food", [])
    food_mask = np.zeros((rows, cols), dtype=bool)
    if food_coords:
        idx = np.asarray(food_coords, dtype=np.int64)
        food_mask[idx[:, 0], idx[:, 1]] = True

    blocked = food_mask
    passable = (~water) & (~food_mask)
    passable_known = seen & (~water) & (~food_mask)

    enemy_hills = get_known_enemy_hills(obs, water, seen, memory)
    friendly_hills_set = {
        (int(h[0]), int(h[1]))
        for h in obs.get("hills", [])
        if len(h) >= 3 and h[2] == 0
    }

    g_food = extract_food_gathering_points(food_coords, water, rows, cols)
    visible = seen & ~water
    mem_food_coords: list[tuple[int, int]] = []
    if memory:
        mem_food_coords = [
            (int(r), int(c))
            for r, c in memory.get("food_seen", ())
            if 0 <= int(r) < rows and 0 <= int(c) < cols and not visible[int(r), int(c)]
        ]
    g_food_mem = extract_memory_food_gathering_points(
        mem_food_coords, water, food_mask, g_food, rows, cols
    )
    active_food_count = len(food_coords) + len(mem_food_coords)

    g_frontier = extract_exploration_frontier(seen, water, rows, cols, food_mask)

    # 1. 构建采食势能场 V_food
    source_grid_food = np.zeros((rows, cols), dtype=np.int32)
    if g_food.any():
        source_grid_food[g_food] = SOURCE_FOOD_VISIBLE
    if g_food_mem.any():
        source_grid_food[g_food_mem & (source_grid_food == SOURCE_NONE)] = SOURCE_FOOD_MEMORY

    if g_food.any() or g_food_mem.any():
        sources_food = {0: g_food}
        if g_food_mem.any():
            sources_food[1] = g_food_mem
        dist_food, meta_food = compute_geodesic_potential_field(
            sources=sources_food,
            passable_mask=passable,
            rows=rows,
            cols=cols,
            max_depth=max_depth,
            return_meta=True,
            source_grid=source_grid_food,
        )
    else:
        dist_food = np.full((rows, cols), FAR, dtype=np.int32)
        meta_food = FieldMetadata(
            frontier_size=0,
            frontier_nonempty_at_limit=False,
            source_grid=source_grid_food,
        )

    # 2. 构建探索前沿势能场 V_frontier
    source_grid_frontier = np.zeros((rows, cols), dtype=np.int32)
    if g_frontier.any():
        source_grid_frontier[g_frontier] = SOURCE_FRONTIER_UNSEEN
        dist_frontier, meta_frontier = compute_geodesic_potential_field(
            sources={0: g_frontier},
            passable_mask=passable_known,
            rows=rows,
            cols=cols,
            max_depth=max_depth,
            return_meta=True,
            source_grid=source_grid_frontier,
            initial_source_type=SOURCE_FRONTIER_UNSEEN,
        )
    else:
        dist_frontier = np.full((rows, cols), FAR, dtype=np.int32)
        meta_frontier = FieldMetadata(
            frontier_size=0,
            frontier_nonempty_at_limit=False,
            source_grid=source_grid_frontier,
        )

    # 3. 构建攻坚势能场 V_hill
    source_grid_hill = np.zeros((rows, cols), dtype=np.int32)
    if enemy_hills:
        hills_mask = np.zeros((rows, cols), dtype=bool)
        for r, c in enemy_hills:
            if passable[r, c]:
                hills_mask[r, c] = True
                source_grid_hill[r, c] = SOURCE_HILL_FOE
        if hills_mask.any():
            dist_hill, meta_hill = compute_geodesic_potential_field(
                sources={0: hills_mask},
                passable_mask=passable,
                rows=rows,
                cols=cols,
                max_depth=max_depth,
                return_meta=True,
                source_grid=source_grid_hill,
                initial_source_type=SOURCE_HILL_FOE,
            )
        else:
            dist_hill = np.full((rows, cols), FAR, dtype=np.int32)
            meta_hill = FieldMetadata(
                frontier_size=0,
                frontier_nonempty_at_limit=False,
                source_grid=source_grid_hill,
            )
    else:
        dist_hill = np.full((rows, cols), FAR, dtype=np.int32)
        meta_hill = FieldMetadata(
            frontier_size=0,
            frontier_nonempty_at_limit=False,
            source_grid=source_grid_hill,
        )

    # 4. 构建守备势能场 V_fortify
    source_grid_fortify = np.zeros((rows, cols), dtype=np.int32)
    if friendly_hills_set:
        friendly_mask = np.zeros((rows, cols), dtype=bool)
        for r, c in friendly_hills_set:
            if passable[r, c]:
                friendly_mask[r, c] = True
        if friendly_mask.any():
            dist_fortify, meta_fortify = compute_geodesic_potential_field(
                sources={0: friendly_mask},
                passable_mask=passable,
                rows=rows,
                cols=cols,
                max_depth=max_depth,
                return_meta=True,
                source_grid=source_grid_fortify,
            )
        else:
            dist_fortify = np.full((rows, cols), FAR, dtype=np.int32)
            meta_fortify = FieldMetadata(
                frontier_size=0,
                frontier_nonempty_at_limit=False,
                source_grid=source_grid_fortify,
            )
    else:
        dist_fortify = np.full((rows, cols), FAR, dtype=np.int32)
        meta_fortify = FieldMetadata(
            frontier_size=0,
            frontier_nonempty_at_limit=False,
            source_grid=source_grid_fortify,
        )

    # 5. 任务状态机分配
    is_phase_b = (
        min_raid_ants is not None
        and len(enemy_hills) > 0
        and total_ants >= min_raid_ants
    )

    raid_set: set[int] = set()
    if is_phase_b:
        num_raid = max(1, min(total_ants - 1, int(round(total_ants * raid_ratio))))
        sorted_by_hill = sorted(
            range(total_ants),
            key=lambda idx: (dist_hill[int(mine[idx][0]), int(mine[idx][1])], idx),
        )
        for idx in sorted_by_hill:
            if len(raid_set) < num_raid and dist_hill[int(mine[idx][0]), int(mine[idx][1])] < FAR:
                raid_set.add(idx)

    non_raid_ants = [idx for idx in range(total_ants) if idx not in raid_set]

    k_gather = min(len(non_raid_ants), active_food_count)
    sorted_by_food = sorted(
        non_raid_ants,
        key=lambda idx: (dist_food[int(mine[idx][0]), int(mine[idx][1])], idx),
    )
    gather_set: set[int] = set()
    for idx in sorted_by_food:
        if len(gather_set) < k_gather and dist_food[int(mine[idx][0]), int(mine[idx][1])] < FAR:
            gather_set.add(idx)

    ant_tasks: dict[int, TaskState] = {}
    for idx in raid_set:
        ant_tasks[idx] = TaskState.ASSAULT
    for idx in gather_set:
        ant_tasks[idx] = TaskState.GATHER

    remaining_ants = [idx for idx in non_raid_ants if idx not in gather_set]
    for idx in remaining_ants:
        ar, ac = int(mine[idx][0]), int(mine[idx][1])
        if dist_frontier[ar, ac] < FAR:
            ant_tasks[idx] = TaskState.EXPLORE
        elif dist_hill[ar, ac] < FAR:
            ant_tasks[idx] = TaskState.ASSAULT
        elif friendly_hills_set and dist_fortify[ar, ac] < FAR:
            ant_tasks[idx] = TaskState.FORTIFY
        elif (dist_food < FAR).any():
            ant_tasks[idx] = TaskState.GATHER
        elif (dist_frontier < FAR).any():
            ant_tasks[idx] = TaskState.EXPLORE
        elif (dist_hill < FAR).any():
            ant_tasks[idx] = TaskState.ASSAULT
        elif friendly_hills_set and (dist_fortify < FAR).any():
            ant_tasks[idx] = TaskState.FORTIFY
        else:
            ant_tasks[idx] = TaskState.EXPLORE

    ant_dist_map: dict[int, tuple[np.ndarray, FieldMetadata]] = {}
    for i in range(total_ants):
        task = ant_tasks[i]
        if task == TaskState.ASSAULT:
            ant_dist_map[i] = (dist_hill, meta_hill)
        elif task == TaskState.GATHER:
            ant_dist_map[i] = (dist_food, meta_food)
        elif task == TaskState.EXPLORE:
            ant_dist_map[i] = (dist_frontier, meta_frontier)
        else:
            ant_dist_map[i] = (dist_fortify, meta_fortify)

    if is_phase_b:
        squad_tiers = [0 if i in raid_set else 1 for i in range(total_ants)]
    else:
        squad_tiers = None

    potentials = np.zeros(total_ants, dtype=np.int32)
    candidate_sequences: list[list[int]] = []
    candidate_rejections_map: dict[int, dict[int, str]] = {
        i: {} for i in range(total_ants)
    }

    for i in range(total_ants):
        ar, ac = int(mine[i][0]), int(mine[i][1])
        dist_field, _ = ant_dist_map[i]
        cur_d = int(dist_field[ar, ac])
        potentials[i] = cur_d
        rejections = candidate_rejections_map[i]
        valid_moves: list[tuple[int, int]] = []

        for k in range(4):
            dr, dc = DELTAS[k]
            nr, nc = (ar + dr) % rows, (ac + dc) % cols
            if water[nr, nc]:
                rejections[k] = RejectionCode.WATER.value
                continue
            if blocked[nr, nc]:
                rejections[k] = RejectionCode.FOOD.value
                continue
            if dist_field[nr, nc] >= cur_d:
                if (nr, nc) in ant_positions:
                    rejections[k] = RejectionCode.TAKEN.value
                else:
                    rejections[k] = RejectionCode.GRADIENT_NON_DECREASING.value
                continue
            valid_moves.append((int(dist_field[nr, nc]), k))

        valid_moves.sort(key=lambda item: (item[0], item[1]))
        cand_seq = [item[1] for item in valid_moves]
        cand_seq.append(HOLD)
        candidate_sequences.append(cand_seq)

    out, _ = arbitrate_orders(
        mine_coords=mine,
        candidate_sequences=candidate_sequences,
        potentials=potentials,
        stall_counters=stall_counters,
        fair_mode=fair_mode,
        rows=rows,
        cols=cols,
        ids=ant_ids,
        squad_tiers=squad_tiers,
        rejections_out=candidate_rejections_map,
    )

    telemetry_records: list[TelemetryRecord] = []
    if collect_telemetry:
        visible_food_set = {tuple(f) for f in food_coords}
        for i in range(total_ants):
            ant_id = ant_ids[i]
            ar, ac = int(mine[i][0]), int(mine[i][1])
            dist_field, meta_field = ant_dist_map[i]
            pot_self = int(dist_field[ar, ac])
            has_reachable = bool((dist_field < FAR).any())
            is_stale = (pot_self == int(FAR) and has_reachable)

            effective_friendly_hills = friendly_hills_set

            record = evaluate_ant_telemetry(
                ant_id=int(ant_id),
                pos=(ar, ac),
                dist=dist_field,
                field_meta=meta_field,
                candidate_rejections=candidate_rejections_map.get(i, {}),
                action=int(out[i]),
                visible_food=visible_food_set,
                friendly_hills=effective_friendly_hills,
                is_stale=is_stale,
            )
            telemetry_records.append(record)

    return out, telemetry_records


class Teacher:
    """每个进程维护一个实例即可；它在回合之间不保留任何状态，可安全复用。"""

    def __init__(self, hill_lead: int = 6, max_depth: int | None = None):
        if hill_lead < 0:
            raise ValueError(f"hill_lead must be non-negative, got {hill_lead}")
        if max_depth is not None and max_depth < 1:
            raise ValueError(f"max_depth must be positive, got {max_depth}")
        self.hill_lead = hill_lead
        self.max_depth = max_depth
        self.stall_counters: dict[int, int] = {}
        self._turn_telemetry: list[TelemetryRecord] = []

    def reset_stall_counters(self) -> None:
        self.stall_counters.clear()

    def reset(self) -> None:
        self.reset_stall_counters()

    def __repr__(self) -> str:
        return f"Teacher(hill_lead={self.hill_lead}, max_depth={self.max_depth})"

    # ---- 势能场构建 ----------------------------------------------------------------

    def field(
        self,
        obs: dict,
        water: np.ndarray,
        seen: np.ndarray,
        memory: dict | None = None,
        return_meta: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, FieldMetadata]:
        """计算到最近目标的距离场，跨越已知可行走单元格并在环面上无缝环绕。"""
        rows, cols = obs["size"]
        food_mask = np.zeros((rows, cols), dtype=bool)
        if obs.get("food"):
            idx = np.asarray(obs["food"], dtype=np.int64)
            food_mask[idx[:, 0], idx[:, 1]] = True
        passable = (~water) & (~food_mask)
        source_grid = np.zeros((rows, cols), dtype=np.int32)

        at_level: dict[int, np.ndarray] = {}

        def seed(mask: np.ndarray, level: int) -> None:
            if not mask.any():
                return
            if level in at_level:
                at_level[level] |= mask
            else:
                at_level[level] = mask.copy()

        hills = np.zeros((rows, cols), dtype=bool)
        for r, c, owner in obs.get("hills", []):
            if owner != 0:
                hills[r, c] = True
        if memory:
            visible = seen & ~water
            for r, c in memory.get("hill_foe_seen", ()):
                if not visible[r, c]:
                    hills[r, c] = True
        if hills.any():
            seed(hills & passable, 0)
            source_grid[(hills & passable)] = SOURCE_HILL_FOE

        g_food = extract_food_gathering_points(obs.get("food", []), water, rows, cols)
        if g_food.any():
            seed(g_food, self.hill_lead)
            source_grid[g_food & (source_grid == SOURCE_NONE)] = SOURCE_FOOD_VISIBLE

        if memory:
            visible = seen & ~water
            mem_food_coords = [
                (int(r), int(c))
                for r, c in memory.get("food_seen", ())
                if 0 <= int(r) < rows and 0 <= int(c) < cols and not visible[int(r), int(c)]
            ]
            g_food_mem = extract_memory_food_gathering_points(
                mem_food_coords, water, food_mask, g_food, rows, cols
            )
            if g_food_mem.any():
                seed(g_food_mem, self.hill_lead + 1)
                source_grid[g_food_mem & (source_grid == SOURCE_NONE)] = SOURCE_FOOD_MEMORY

        g_frontier = extract_exploration_frontier(seen, water, rows, cols, food_mask)
        if g_frontier.any():
            seed(g_frontier, self.hill_lead + 2)
            source_grid[g_frontier & (source_grid == SOURCE_NONE)] = SOURCE_FRONTIER_UNSEEN

        return compute_geodesic_potential_field(
            sources=at_level,
            passable_mask=passable,
            rows=rows,
            cols=cols,
            max_depth=self.max_depth,
            return_meta=return_meta,
            source_grid=source_grid,
        )

    def _get_known_enemy_hills(
        self,
        obs: dict,
        water: np.ndarray,
        seen: np.ndarray,
        memory: dict | None,
    ) -> set[tuple[int, int]]:
        return get_known_enemy_hills(obs, water, seen, memory)

    # ---- 动作决策 ------------------------------------------------------------------

    def orders(
        self,
        obs: dict,
        water: np.ndarray,
        seen: np.ndarray,
        memory: dict | None = None,
        collect_telemetry: bool = False,
        stall_counters: dict[int, int] | None = None,
        fair_mode: bool = True,
    ) -> np.ndarray:
        mine = obs.get("mine")
        if not mine:
            self._turn_telemetry = []
            return np.zeros(0, dtype=np.int64)

        rows, cols = obs["size"]
        if collect_telemetry:
            dist, field_meta = self.field(obs, water, seen, memory, return_meta=True)
        else:
            dist = self.field(obs, water, seen, memory, return_meta=False)
            field_meta = None

        blocked = np.zeros((rows, cols), dtype=bool)
        for r, c in obs.get("food", []):
            blocked[r, c] = True

        ant_positions = {(int(a[0]), int(a[1])) for a in mine}
        total_ants = len(mine)
        candidate_sequences: list[list[int]] = []
        candidate_rejections_map: dict[int, dict[int, str]] = {
            i: {} for i in range(total_ants)
        }
        potentials = np.zeros(total_ants, dtype=np.int32)

        for i in range(total_ants):
            ar, ac = int(mine[i][0]), int(mine[i][1])
            cur_d = int(dist[ar, ac])
            potentials[i] = cur_d
            rejections = candidate_rejections_map[i]
            valid_moves: list[tuple[int, int]] = []

            for k in range(4):
                dr, dc = DELTAS[k]
                nr, nc = (ar + dr) % rows, (ac + dc) % cols
                if water[nr, nc]:
                    rejections[k] = RejectionCode.WATER.value
                    continue
                if blocked[nr, nc]:
                    rejections[k] = RejectionCode.FOOD.value
                    continue
                if dist[nr, nc] >= cur_d:
                    if (nr, nc) in ant_positions:
                        rejections[k] = RejectionCode.TAKEN.value
                    else:
                        rejections[k] = RejectionCode.GRADIENT_NON_DECREASING.value
                    continue
                valid_moves.append((int(dist[nr, nc]), k))

            valid_moves.sort(key=lambda item: (item[0], item[1]))
            cand_seq = [item[1] for item in valid_moves]
            cand_seq.append(HOLD)
            candidate_sequences.append(cand_seq)

        ant_ids = obs.get("ids")
        out, _ = arbitrate_orders(
            mine_coords=mine,
            candidate_sequences=candidate_sequences,
            potentials=potentials,
            stall_counters=stall_counters,
            fair_mode=fair_mode,
            rows=rows,
            cols=cols,
            ids=ant_ids,
            rejections_out=candidate_rejections_map,
        )

        if collect_telemetry:
            visible_food_set = {tuple(f) for f in obs.get("food", [])}
            friendly_hills_set = {
                (int(h[0]), int(h[1]))
                for h in obs.get("hills", [])
                if len(h) >= 3 and h[2] == 0
            }
            has_reachable = bool((dist < FAR).any())

            telemetry_records: list[TelemetryRecord] = []
            for i in range(len(mine)):
                ant_id = ant_ids[i] if (ant_ids is not None and i < len(ant_ids)) else i
                ar, ac = mine[i]
                pot_self = int(dist[ar, ac])
                is_stale = (pot_self == int(FAR) and has_reachable)
                record = evaluate_ant_telemetry(
                    ant_id=int(ant_id),
                    pos=(ar, ac),
                    dist=dist,
                    field_meta=field_meta,
                    candidate_rejections=candidate_rejections_map.get(i, {}),
                    action=int(out[i]),
                    visible_food=visible_food_set,
                    friendly_hills=friendly_hills_set,
                    is_stale=is_stale,
                )
                telemetry_records.append(record)
            self._turn_telemetry = telemetry_records
        else:
            self._turn_telemetry = []

        return out

    def export_turn_telemetry(self) -> list[TelemetryRecord]:
        return list(self._turn_telemetry)

    def orders_with_telemetry(
        self,
        obs: dict,
        water: np.ndarray,
        seen: np.ndarray,
        memory: dict | None = None,
        stall_counters: dict[int, int] | None = None,
        fair_mode: bool = True,
    ) -> tuple[np.ndarray, list[TelemetryRecord]]:
        orders = self.orders(
            obs,
            water,
            seen,
            memory,
            collect_telemetry=True,
            stall_counters=stall_counters,
            fair_mode=fair_mode,
        )
        return orders, self.export_turn_telemetry()

    def act(
        self,
        obs: dict,
        memory: dict | None = None,
        collect_telemetry: bool = False,
        stall_counters: dict[int, int] | None = None,
        fair_mode: bool = True,
    ) -> list[int]:
        water, seen = known(obs)
        return self.orders(
            obs,
            water,
            seen,
            memory,
            collect_telemetry=collect_telemetry,
            stall_counters=stall_counters,
            fair_mode=fair_mode,
        ).tolist()

    def act_with_telemetry(
        self,
        obs: dict,
        memory: dict | None = None,
        stall_counters: dict[int, int] | None = None,
        fair_mode: bool = True,
    ) -> tuple[list[int], list[TelemetryRecord]]:
        water, seen = known(obs)
        orders = self.orders(
            obs,
            water,
            seen,
            memory,
            collect_telemetry=True,
            stall_counters=stall_counters,
            fair_mode=fair_mode,
        )
        return orders.tolist(), self.export_turn_telemetry()


class CompoundTeacher:
    """条件复合启发式教师：态势感知动态分工调度（IDEA-002 R2/EXP-014）。

    每个进程维护一个实例即可；它在回合之间不保留内部状态，可安全复用。
    """

    def __init__(
        self,
        min_raid_ants: int = 4,
        raid_ratio: float = 0.5,
        max_depth: int | None = None,
        hill_lead: int = 6,
    ):
        if min_raid_ants < 1:
            raise ValueError(f"min_raid_ants must be >= 1, got {min_raid_ants}")
        if not (0.0 < raid_ratio <= 1.0):
            raise ValueError(f"raid_ratio must be in (0.0, 1.0], got {raid_ratio}")
        if max_depth is not None and max_depth < 1:
            raise ValueError(f"max_depth must be >= 1, got {max_depth}")
        if hill_lead < 0:
            raise ValueError(f"hill_lead must be non-negative, got {hill_lead}")
        self.min_raid_ants = min_raid_ants
        self.raid_ratio = raid_ratio
        self.max_depth = max_depth
        self.hill_lead = hill_lead
        self.stall_counters: dict[int, int] = {}
        self._turn_telemetry: list[TelemetryRecord] = []

    def reset_stall_counters(self) -> None:
        self.stall_counters.clear()

    def reset(self) -> None:
        self.reset_stall_counters()

    def __repr__(self) -> str:
        return (
            f"CompoundTeacher(min_raid_ants={self.min_raid_ants}, "
            f"raid_ratio={self.raid_ratio}, max_depth={self.max_depth}, "
            f"hill_lead={self.hill_lead})"
        )

    # ---- 态势识别 ------------------------------------------------------------------

    def _get_known_enemy_hills(
        self,
        obs: dict,
        water: np.ndarray,
        seen: np.ndarray,
        memory: dict | None,
    ) -> set[tuple[int, int]]:
        return get_known_enemy_hills(obs, water, seen, memory)

    # ---- 势能场构建 ----------------------------------------------------------------

    def field_gather(
        self,
        obs: dict,
        water: np.ndarray,
        seen: np.ndarray,
        memory: dict | None,
        fallback_hills: set[tuple[int, int]] | None = None,
        return_meta: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, FieldMetadata]:
        """采食与探索势能场：食物优先级最高，其次为未探索前沿，最后为备用敌丘。"""
        rows, cols = obs["size"]
        food_mask = np.zeros((rows, cols), dtype=bool)
        if obs.get("food"):
            idx = np.asarray(obs["food"], dtype=np.int64)
            food_mask[idx[:, 0], idx[:, 1]] = True
        passable = (~water) & (~food_mask)
        source_grid = np.zeros((rows, cols), dtype=np.int32)
        at_level: dict[int, np.ndarray] = {}

        def seed(mask: np.ndarray, level: int) -> None:
            if not mask.any():
                return
            if level in at_level:
                at_level[level] |= mask
            else:
                at_level[level] = mask.copy()

        # 1. 视野内的食物合法采集位：0 级（最高优先）
        g_food = extract_food_gathering_points(obs.get("food", []), water, rows, cols)
        if g_food.any():
            seed(g_food, 0)
            source_grid[g_food & (source_grid == SOURCE_NONE)] = SOURCE_FOOD_VISIBLE

        # 2. 记忆中视野外的食物合法采集位：第 1 级
        if memory:
            visible = seen & ~water
            mem_food_coords = [
                (int(r), int(c))
                for r, c in memory.get("food_seen", ())
                if 0 <= int(r) < rows and 0 <= int(c) < cols and not visible[int(r), int(c)]
            ]
            g_food_mem = extract_memory_food_gathering_points(
                mem_food_coords, water, food_mask, g_food, rows, cols
            )
            if g_food_mem.any():
                seed(g_food_mem, 1)
                source_grid[g_food_mem & (source_grid == SOURCE_NONE)] = SOURCE_FOOD_MEMORY

        # 3. 未知探索前沿：第 2 级
        g_frontier = extract_exploration_frontier(seen, water, rows, cols, food_mask)
        if g_frontier.any():
            seed(g_frontier, 2)
            source_grid[g_frontier & (source_grid == SOURCE_NONE)] = SOURCE_FRONTIER_UNSEEN

        # 4. 备用敌丘目标：当无其他目标时作为独立兜底源点（0 级）
        if fallback_hills and not at_level:
            fh = np.zeros((rows, cols), dtype=bool)
            for r, c in fallback_hills:
                if passable[r, c]:
                    fh[r, c] = True
            if fh.any():
                seed(fh, 0)
                source_grid[fh & (source_grid == SOURCE_NONE)] = SOURCE_HILL_FOE

        return compute_geodesic_potential_field(
            sources=at_level,
            passable_mask=passable,
            rows=rows,
            cols=cols,
            max_depth=self.max_depth,
            return_meta=return_meta,
            source_grid=source_grid,
        )

    def field_raid(
        self,
        obs: dict,
        water: np.ndarray,
        enemy_hills: set[tuple[int, int]],
        return_meta: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, FieldMetadata]:
        """突击分队专用势能场：以已知敌方蚁丘为唯一初始源点向全图扩散。"""
        rows, cols = obs["size"]
        source_grid = np.zeros((rows, cols), dtype=np.int32)
        if not enemy_hills:
            dist = np.full((rows, cols), FAR, dtype=np.int32)
            if return_meta:
                meta = FieldMetadata(frontier_size=0, frontier_nonempty_at_limit=False, source_grid=source_grid)
                return dist, meta
            return dist

        food_mask = np.zeros((rows, cols), dtype=bool)
        if obs.get("food"):
            idx = np.asarray(obs["food"], dtype=np.int64)
            food_mask[idx[:, 0], idx[:, 1]] = True
        passable = (~water) & (~food_mask)
        hills = np.zeros((rows, cols), dtype=bool)
        for r, c in enemy_hills:
            if passable[r, c]:
                hills[r, c] = True
                source_grid[r, c] = SOURCE_HILL_FOE

        return compute_geodesic_potential_field(
            sources={0: hills},
            passable_mask=passable,
            rows=rows,
            cols=cols,
            max_depth=self.max_depth,
            return_meta=return_meta,
            source_grid=source_grid,
            initial_source_type=SOURCE_HILL_FOE,
        )

    # ---- 动作决策与无冲突仲裁 ------------------------------------------------------

    def orders(
        self,
        obs: dict,
        water: np.ndarray,
        seen: np.ndarray,
        memory: dict | None = None,
        collect_telemetry: bool = False,
        stall_counters: dict[int, int] | None = None,
        fair_mode: bool = True,
    ) -> np.ndarray:
        mine = obs.get("mine")
        if not mine:
            self._turn_telemetry = []
            return np.zeros(0, dtype=np.int64)

        orders, telemetry = _dispatch_task_state_machine(
            mine=mine,
            obs=obs,
            water=water,
            seen=seen,
            memory=memory,
            max_depth=self.max_depth,
            min_raid_ants=self.min_raid_ants,
            raid_ratio=self.raid_ratio,
            collect_telemetry=collect_telemetry,
            stall_counters=stall_counters,
            fair_mode=fair_mode,
        )
        self._turn_telemetry = telemetry
        return orders

    def export_turn_telemetry(self) -> list[TelemetryRecord]:
        return list(self._turn_telemetry)

    def orders_with_telemetry(
        self,
        obs: dict,
        water: np.ndarray,
        seen: np.ndarray,
        memory: dict | None = None,
        stall_counters: dict[int, int] | None = None,
        fair_mode: bool = True,
    ) -> tuple[np.ndarray, list[TelemetryRecord]]:
        orders = self.orders(
            obs,
            water,
            seen,
            memory,
            collect_telemetry=True,
            stall_counters=stall_counters,
            fair_mode=fair_mode,
        )
        return orders, self.export_turn_telemetry()

    def act(
        self,
        obs: dict,
        memory: dict | None = None,
        collect_telemetry: bool = False,
        stall_counters: dict[int, int] | None = None,
        fair_mode: bool = True,
    ) -> list[int]:
        water, seen = known(obs)
        return self.orders(
            obs,
            water,
            seen,
            memory,
            collect_telemetry=collect_telemetry,
            stall_counters=stall_counters,
            fair_mode=fair_mode,
        ).tolist()

    def act_with_telemetry(
        self,
        obs: dict,
        memory: dict | None = None,
        stall_counters: dict[int, int] | None = None,
        fair_mode: bool = True,
    ) -> tuple[list[int], list[TelemetryRecord]]:
        water, seen = known(obs)
        orders = self.orders(
            obs,
            water,
            seen,
            memory,
            collect_telemetry=True,
            stall_counters=stall_counters,
            fair_mode=fair_mode,
        )
        return orders.tolist(), self.export_turn_telemetry()


def known(obs: dict) -> tuple[np.ndarray, np.ndarray]:
    """从单次观测中提取已知水域以及曾经探索过的全部区域。

    `water` 是唯一自带记忆的字段，因此 1 表示见过的已知水域，0 表示*已知陆地*或者*从未探索过的单元格*。
    教师模型需要区分这两者，从单次观测中提取这一信息的严谨方法是计算己方蚂蚁周围的可见圆盘 ——
    这与第 6 个特征平面计算的内容相同，也是其存在的同一原因。
    """
    from .planes import VIEW_RADIUS2, Board

    rows, cols = obs["size"]
    b = Board(rows, cols)
    water = b.rle(obs["water"]["rle"]).astype(bool)
    if "vis" in obs and obs["vis"] and "rle" in obs["vis"]:
        vis_mask = b.rle(obs["vis"]["rle"]).astype(bool)
        if vis_mask.any():
            return water, water | vis_mask
    # 单元格若为已知水域或当前处于可见视野内，则视为“已探索”。这会略有漏算 —— 很久以前走过但现在处于迷雾中的陆地
    # 会被读为未探索 —— 但在此处这是完全良性的偏差：它激励蚁群持续向外探索，防止过早认定地图已探索完毕。
    visible = b.dilate(b.scatter(obs["mine"]), VIEW_RADIUS2).astype(bool)
    return water, water | visible
