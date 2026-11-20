from __future__ import annotations

import numpy as np

from .coordination import (
    DELTAS,
    FAR,
    HOLD,
    build_capacity_constrained_bipartite_graph,
    compute_batch_torus_geodesics,
    compute_batch_torus_geodesics_multibucket,
    compute_combined_delay,
    compute_corridor_width,
    compute_corridor_width_grid,
    compute_cost_matrix_bitmask,
    compute_cost_matrix_bitmask_multibucket,
    compute_dynamic_congestion_penalty,
    compute_enemy_threat_penalty,
    compute_local_ant_density,
    extract_cluster_gathering_positions,
    solve_mcmf_ssp,
)
from .teacher import (
    TaskState,
    arbitrate_orders,
    compute_geodesic_potential_field,
    extract_exploration_frontier,
    get_known_enemy_hills,
    known,
)
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


class MCMFTeacher:
    def __init__(
        self,
        hill_lead: int = 6,
        max_depth: int | None = None,
        min_raid_ants: int = 4,
        raid_ratio: float = 0.5,
        dynamic_congestion: bool = False,
        congestion_alpha: float = 10.0,
        congestion_beta: float = 1.5,
        adaptive_raid: bool = True,
        threat_avoidance: bool = True,
        focus_target: bool = True,
    ) -> None:
        if min_raid_ants < 1:
            raise ValueError(f"min_raid_ants 必须大于等于 1，实际为 {min_raid_ants}")
        if not (0.0 < raid_ratio <= 1.0):
            raise ValueError(f"raid_ratio 必须处于 (0.0, 1.0]，实际为 {raid_ratio}")
        if max_depth is not None and max_depth < 1:
            raise ValueError(f"max_depth 必须大于等于 1，实际为 {max_depth}")
        if hill_lead < 0:
            raise ValueError(f"hill_lead 必须为非负整数，实际为 {hill_lead}")

        self.hill_lead = hill_lead
        self.max_depth = max_depth
        self.min_raid_ants = min_raid_ants
        self.raid_ratio = raid_ratio
        self.dynamic_congestion = dynamic_congestion
        self.congestion_alpha = congestion_alpha
        self.congestion_beta = congestion_beta
        self.adaptive_raid = adaptive_raid
        self.threat_avoidance = threat_avoidance
        self.focus_target = focus_target
        self._last_target_hill: tuple[int, int] | None = None
        self.stall_counters: dict[int, int] = {}
        self._turn_telemetry: list[TelemetryRecord] = []
        self._last_ant_tasks: dict[int, TaskState] = {}
        self._last_winning_targets: list[tuple[int, int]] = []
        self._last_winning_fields: list[np.ndarray] = []

    def reset_stall_counters(self) -> None:
        self.stall_counters.clear()

    def reset(self) -> None:
        self.reset_stall_counters()
        self._turn_telemetry.clear()
        self._last_ant_tasks.clear()
        self._last_winning_targets.clear()
        self._last_winning_fields.clear()
        self._last_target_hill = None

    def get_last_tasks(self) -> dict[int, TaskState]:
        return dict(self._last_ant_tasks)

    def get_last_winning_flow(self) -> tuple[list[tuple[int, int]], list[np.ndarray]]:
        return list(self._last_winning_targets), list(self._last_winning_fields)

    def get_last_target_hill(self) -> tuple[int, int] | None:
        return self._last_target_hill

    def __repr__(self) -> str:
        return (
            f"MCMFTeacher(min_raid_ants={self.min_raid_ants}, "
            f"raid_ratio={self.raid_ratio}, max_depth={self.max_depth}, "
            f"hill_lead={self.hill_lead})"
        )

    def export_turn_telemetry(self) -> list[TelemetryRecord]:
        return list(self._turn_telemetry)

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
        total_ants = len(mine)
        ant_ids = [int(x) for x in obs["ids"]] if "ids" in obs else list(range(total_ants))
        ant_positions: list[tuple[int, int]] = [(int(m[0]), int(m[1])) for m in mine]
        ant_positions_set = set(ant_positions)

        food_mask = np.zeros((rows, cols), dtype=bool)
        if obs.get("food"):
            idx = np.asarray(obs["food"], dtype=np.int64)
            food_mask[idx[:, 0], idx[:, 1]] = True

        passable = (~water) & (~food_mask)
        passable_known = ~water

        friendly_hills_set = {
            (int(h[0]), int(h[1]))
            for h in obs.get("hills", [])
            if len(h) >= 3 and h[2] == 0
        }
        enemy_hills = get_known_enemy_hills(obs, water, seen, memory)

        visible_foes = obs.get("foes", [])

        # 基地蚁穴就近防守态势判定（仅收集威胁己方蚁穴的近身敌蚁，绝不回退至己方蚁穴格原地驻留）
        threatened_hills: set[tuple[int, int]] = set()
        hill_threat_counts: dict[tuple[int, int], int] = {}
        threat_foe_coords: list[tuple[int, int]] = []
        if friendly_hills_set and visible_foes:
            for hr, hc in friendly_hills_set:
                near_foes = 0
                for f in visible_foes:
                    fr, fc = int(f[0]), int(f[1])
                    diff_r = abs(hr - fr)
                    dr = min(diff_r, rows - diff_r)
                    diff_c = abs(hc - fc)
                    dc = min(diff_c, cols - diff_c)
                    if dr + dc <= 12 or (dr * dr + dc * dc) <= 100:
                        near_foes += 1
                        threat_foe_coords.append((fr, fc))
                if near_foes > 0:
                    threatened_hills.add((hr, hc))
                    hill_threat_counts[(hr, hc)] = near_foes

        defense_set: set[int] = set()
        if threatened_hills:
            for hr, hc in threatened_hills:
                num_threats = hill_threat_counts[(hr, hc)]
                needed_defenders = max(2, num_threats + 1)
                available = [i for i in range(total_ants) if i not in defense_set]
                sorted_for_hill = sorted(
                    available,
                    key=lambda idx_ant: (
                        min(abs(ant_positions[idx_ant][0] - hr), rows - abs(ant_positions[idx_ant][0] - hr))
                        + min(abs(ant_positions[idx_ant][1] - hc), cols - abs(ant_positions[idx_ant][1] - hc))
                    ),
                )
                assigned_count = 0
                for idx_ant in sorted_for_hill:
                    ar, ac = ant_positions[idx_ant]
                    d_h = (
                        min(abs(ar - hr), rows - abs(ar - hr))
                        + min(abs(ac - hc), cols - abs(ac - hc))
                    )
                    if d_h <= 16 or assigned_count < 2:
                        defense_set.add(idx_ant)
                        assigned_count += 1
                        if assigned_count >= needed_defenders or len(defense_set) >= max(1, total_ants // 2):
                            break

        # 敌方威胁与通道延时开销计算
        int_delay: np.ndarray | None = None
        width_map: np.ndarray | None = None
        if self.dynamic_congestion or (self.threat_avoidance and visible_foes):
            if not hasattr(self, "_cached_width_maps"):
                self._cached_width_maps: dict[bytes, tuple[np.ndarray, np.ndarray]] = {}
            water_key = water.tobytes()
            if water_key not in self._cached_width_maps:
                w_map = compute_corridor_width(water, rows, cols)
                w_grid = compute_corridor_width_grid(water)
                self._cached_width_maps[water_key] = (w_map, w_grid)
            width_map, _ = self._cached_width_maps[water_key]

        congestion_delay: np.ndarray | None = None
        if self.dynamic_congestion and width_map is not None:
            density_map = compute_local_ant_density(ant_positions, passable, rows, cols, radius=2)
            _, congestion_delay = compute_dynamic_congestion_penalty(
                width_map,
                density_map,
                passable,
                alpha=self.congestion_alpha,
                beta=self.congestion_beta,
                max_delay=4,
            )

        threat_delay: np.ndarray | None = None
        if self.threat_avoidance and visible_foes:
            foe_coords = [[int(f[0]), int(f[1])] for f in visible_foes]
            _, threat_delay = compute_enemy_threat_penalty(
                foe_positions=foe_coords,
                friendly_positions=ant_positions,
                passable_mask=passable,
                rows=rows,
                cols=cols,
                width_map=width_map,
                attack_radius_sq=5,
                max_delay=4,
            )

        if congestion_delay is not None and threat_delay is not None:
            int_delay = compute_combined_delay(congestion_delay, threat_delay, max_delay=4)
        elif congestion_delay is not None:
            int_delay = congestion_delay
        elif threat_delay is not None:
            int_delay = threat_delay

        # 集中突击目标定向突破判定
        target_enemy_hills = list(enemy_hills)
        if self.focus_target and len(enemy_hills) > 1:
            best_hill = None
            best_score = float("inf")
            ant_arr = np.asarray(ant_positions, dtype=np.int32)
            for hr, hc in enemy_hills:
                diff_r = np.abs(ant_arr[:, 0] - hr)
                dr = np.minimum(diff_r, rows - diff_r)
                diff_c = np.abs(ant_arr[:, 1] - hc)
                dc = np.minimum(diff_c, cols - diff_c)
                dists = dr + dc
                sorted_d = np.sort(dists)
                k_mean_d = float(np.mean(sorted_d[:min(5, len(sorted_d))]))

                defenders_count = 0
                for f in visible_foes:
                    fr, fc = int(f[0]), int(f[1])
                    d_f = min(abs(hr - fr), rows - abs(hr - fr)) + min(abs(hc - fc), cols - abs(hc - fc))
                    if d_f <= 6:
                        defenders_count += 1
                score = k_mean_d + 3.0 * defenders_count
                if score < best_score:
                    best_score = score
                    best_hill = (hr, hc)
            if best_hill is not None:
                target_enemy_hills = [best_hill]
                self._last_target_hill = best_hill
        elif len(enemy_hills) == 1:
            self._last_target_hill = target_enemy_hills[0]
        else:
            self._last_target_hill = None

        # 1. 突击分队判定与攻坚势能场
        source_grid_hill = np.zeros((rows, cols), dtype=np.int32)
        if target_enemy_hills:
            hills_mask = np.zeros((rows, cols), dtype=bool)
            for r, c in target_enemy_hills:
                if passable[r, c]:
                    hills_mask[r, c] = True
                    source_grid_hill[r, c] = SOURCE_HILL_FOE
            if hills_mask.any():
                dist_hill, meta_hill = compute_geodesic_potential_field(
                    sources={0: hills_mask},
                    passable_mask=passable,
                    rows=rows,
                    cols=cols,
                    max_depth=None,
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

        # 自适应态势感知突击调度比例计算（仅依赖可观测蚁群规模，消除失效的 turn 键与多席位突击饥饿）
        if self.adaptive_raid:
            if total_ants < self.min_raid_ants:
                current_raid_ratio = 0.0
            else:
                # 随兵力从 min_raid_ants 平滑提升至 raid_ratio，确保任意尺寸与多丘棋盘均保持稳定进攻压力
                ramp = min(1.0, (total_ants - self.min_raid_ants) / 20.0)
                base_floor = min(self.raid_ratio, 0.25)
                current_raid_ratio = base_floor + (self.raid_ratio - base_floor) * ramp
        else:
            current_raid_ratio = self.raid_ratio

        is_phase_b = (
            self.min_raid_ants is not None
            and len(target_enemy_hills) > 0
            and total_ants >= self.min_raid_ants
            and current_raid_ratio > 0.0
        )

        raid_set: set[int] = set()
        if is_phase_b:
            raid_candidates = [idx for idx in range(total_ants) if idx not in defense_set]
            max_raid_count = max(0, min(len(raid_candidates) - 1, int(round(total_ants * current_raid_ratio))))
            if max_raid_count > 0:
                sorted_by_hill = sorted(
                    raid_candidates,
                    key=lambda idx_ant: (dist_hill[ant_positions[idx_ant][0], ant_positions[idx_ant][1]], idx_ant),
                )
                for idx_ant in sorted_by_hill:
                    if len(raid_set) < max_raid_count and dist_hill[ant_positions[idx_ant][0], ant_positions[idx_ant][1]] < FAR:
                        raid_set.add(idx_ant)

        non_raid_ants = [idx_ant for idx_ant in range(total_ants) if idx_ant not in raid_set]
        squad_tiers = [0 if (i in raid_set or i in defense_set) else 1 for i in range(total_ants)] if (is_phase_b or defense_set) else None

        # 2. 食物提取（视野食物 + 迷雾记忆食物）
        visible = seen & ~water
        food_coords: list[tuple[int, int]] = [(int(f[0]), int(f[1])) for f in obs.get("food", [])]
        mem_food_coords: list[tuple[int, int]] = []
        if memory:
            mem_food_coords = [
                (int(r), int(c))
                for r, c in memory.get("food_seen", ())
                if 0 <= int(r) < rows and 0 <= int(c) < cols and not visible[int(r), int(c)] and not water[int(r), int(c)]
            ]
        all_food_coords = food_coords + mem_food_coords

        gathering_positions, food_to_gather, gather_to_food = extract_cluster_gathering_positions(
            all_food_coords, water, rows, cols
        )

        ant_tasks: dict[int, TaskState] = {}
        for idx_ant in raid_set:
            ant_tasks[idx_ant] = TaskState.ASSAULT

        ant_bound_target: dict[int, tuple[int, int]] = {}
        ant_dist_map: dict[int, tuple[np.ndarray, FieldMetadata]] = {}
        gather_set: set[int] = set()

        # 3. 两阶段容量约束 MCMF 多对多分配
        num_gather = len(gathering_positions)
        if num_gather > 0 and non_raid_ants:
            if self.max_depth is not None:
                effective_depth = self.max_depth
            else:
                # 严禁静默截断为 16 步；默认至少 32 步并按环面半周长自适应扩展
                effective_depth = max(32, max(rows, cols) // 2)

            non_raid_ant_positions = [ant_positions[idx_ant] for idx_ant in non_raid_ants]

            # 粗粒度环面曼哈顿距离预筛选
            ant_arr = np.asarray(non_raid_ant_positions, dtype=np.int32)
            gather_arr = np.asarray(gathering_positions, dtype=np.int32)
            num_foods = len(food_to_gather)

            diff_r = np.abs(ant_arr[:, 0:1] - gather_arr[None, :, 0])
            dr = np.minimum(diff_r, rows - diff_r)
            diff_c = np.abs(ant_arr[:, 1:2] - gather_arr[None, :, 1])
            dc = np.minimum(diff_c, cols - diff_c)
            dist_ant_g = dr + dc

            min_dist_to_any_ant = np.min(dist_ant_g, axis=0)
            reachable_g_mask = min_dist_to_any_ant <= effective_depth

            candidate_g_set: set[int] = set()
            if reachable_g_mask.any() and num_foods > 0:
                ant_food_dist = np.full((len(non_raid_ants), num_foods), 10**9, dtype=np.int32)
                for f_idx in range(num_foods):
                    g_list = food_to_gather[f_idx]
                    if g_list:
                        ant_food_dist[:, f_idx] = np.min(dist_ant_g[:, g_list], axis=1)

                if num_foods <= 10:
                    active_food_indices = set(range(num_foods))
                else:
                    k_per_ant = min(num_foods, 10)
                    top_k_foods = np.argpartition(ant_food_dist, k_per_ant - 1, axis=1)[:, :k_per_ant]
                    active_food_indices = set(top_k_foods.ravel().tolist())

                for f_idx in active_food_indices:
                    for g_idx in food_to_gather[f_idx]:
                        if reachable_g_mask[g_idx]:
                            candidate_g_set.add(g_idx)

            if candidate_g_set:
                filtered_g_indices = sorted(candidate_g_set)
                filtered_gathering_positions = [gathering_positions[idx] for idx in filtered_g_indices]
                filtered_gather_to_food = {
                    new_k: gather_to_food[old_g]
                    for new_k, old_g in enumerate(filtered_g_indices)
                }

                # 阶段 A: 通过超高速 Bitmask BFS 采样生成费用矩阵
                congestion_penalty_matrix: np.ndarray | None = None
                if int_delay is not None and (int_delay > 0).any():
                    delayed_cost_matrix = compute_cost_matrix_bitmask_multibucket(
                        ant_positions=non_raid_ant_positions,
                        gathering_positions=filtered_gathering_positions,
                        passable_mask=passable,
                        cell_delay=int_delay,
                        rows=rows,
                        cols=cols,
                        max_depth=effective_depth,
                        max_delay=4,
                    )
                    cost_matrix = delayed_cost_matrix
                    self.last_congestion_penalty_matrix = delayed_cost_matrix
                else:
                    cost_matrix = compute_cost_matrix_bitmask(
                        ant_positions=non_raid_ant_positions,
                        gathering_positions=filtered_gathering_positions,
                        passable_mask=passable,
                        rows=rows,
                        cols=cols,
                        max_depth=effective_depth,
                    )
                    self.last_congestion_penalty_matrix = None

                cluster_limit_val = min(num_foods, 10)
                graph, offsets = build_capacity_constrained_bipartite_graph(
                    ant_positions=non_raid_ant_positions,
                    food_positions=all_food_coords,
                    gathering_positions=filtered_gathering_positions,
                    geodesic_cost_matrix=cost_matrix,
                    congestion_penalty_matrix=congestion_penalty_matrix,
                    gather_to_food=filtered_gather_to_food,
                    rows=rows,
                    cols=cols,
                    max_candidates_per_ant=None,
                    cluster_limit=cluster_limit_val,
                    g_per_cluster=1,
                )

                max_flow_limit = min(len(non_raid_ants), len(all_food_coords))
                _, _, flow_dict = solve_mcmf_ssp(
                    graph,
                    offsets["source"],
                    offsets["sink"],
                    max_flow_limit=max_flow_limit,
                )

                winning_allocations: list[tuple[int, int, int, int]] = []
                for (u, v), flow in flow_dict.items():
                    if (
                        flow > 0
                        and offsets["u_offset"] <= u < offsets["g_in_offset"]
                        and offsets["g_in_offset"] <= v < offsets["g_out_offset"]
                    ):
                        local_ant_idx = u - offsets["u_offset"]
                        new_g_idx = v - offsets["g_in_offset"]
                        global_ant_idx = non_raid_ants[local_ant_idx]
                        old_g_idx = filtered_g_indices[new_g_idx]
                        winning_allocations.append((global_ant_idx, old_g_idx, local_ant_idx, new_g_idx))

                # 阶段 B: 仅为最终胜出的采集位计算高精度 2D 测地场
                if winning_allocations:
                    unique_winning_g_indices = sorted(list(set(old_g_idx for _, old_g_idx, _, _ in winning_allocations)))
                    winning_targets = [gathering_positions[old_g] for old_g in unique_winning_g_indices]

                    target_alloc_dists: dict[int, int] = {}
                    for _, old_g_idx, l_idx, new_g in winning_allocations:
                        dist_val = int(cost_matrix[l_idx, new_g])
                        if old_g_idx not in target_alloc_dists or dist_val > target_alloc_dists[old_g_idx]:
                            target_alloc_dists[old_g_idx] = dist_val

                    per_target_depths = [
                        min(effective_depth, target_alloc_dists.get(old_g, effective_depth) + 1)
                        for old_g in unique_winning_g_indices
                    ]
                    max_alloc_dist = max(per_target_depths) if per_target_depths else 1
                    depth_limit = min(effective_depth, max(max_alloc_dist, 1))

                    if int_delay is not None and (int_delay > 0).any():
                        winning_fields = compute_batch_torus_geodesics_multibucket(
                            winning_targets, passable, int_delay, rows, cols, max_depth=per_target_depths, max_delay=4
                        )
                    else:
                        winning_fields = compute_batch_torus_geodesics(
                            winning_targets, passable, rows, cols, max_depth=depth_limit
                        )
                    g_to_field = {old_g: winning_fields[k] for k, old_g in enumerate(unique_winning_g_indices)}
                    self._last_winning_targets = list(winning_targets)
                    self._last_winning_fields = list(winning_fields)

                    for global_ant_idx, old_g_idx, _, _ in winning_allocations:
                        gather_set.add(global_ant_idx)
                        ant_tasks[global_ant_idx] = TaskState.GATHER
                        target_tile = gathering_positions[old_g_idx]
                        ant_bound_target[global_ant_idx] = target_tile

                        is_visible_adjacent = False
                        for f_idx in gather_to_food.get(old_g_idx, []):
                            if f_idx < len(food_coords):
                                is_visible_adjacent = True
                                break
                        st = SOURCE_FOOD_VISIBLE if is_visible_adjacent else SOURCE_FOOD_MEMORY

                        source_grid_g = np.full((rows, cols), st, dtype=np.int32)
                        g_meta = FieldMetadata(
                            frontier_size=1,
                            frontier_nonempty_at_limit=False,
                            source_grid=source_grid_g,
                        )
                        ant_dist_map[global_ant_idx] = (g_to_field[old_g_idx], g_meta)
                else:
                    self._last_winning_targets = []
                    self._last_winning_fields = []

        # 4. 基地外围拦截势能场（严禁以己方蚁穴格自身为终点，杜绝产卵位堵塞）与探索/突击分流
        dist_fortify: np.ndarray | None = None
        meta_fortify: FieldMetadata | None = None
        if threatened_hills and threat_foe_coords:
            intercept_mask = np.zeros((rows, cols), dtype=bool)
            source_grid_fortify = np.zeros((rows, cols), dtype=np.int32)
            for fr, fc in threat_foe_coords:
                if passable[fr, fc] and (fr, fc) not in friendly_hills_set:
                    intercept_mask[fr, fc] = True
                else:
                    for dr, dc in DELTAS:
                        nr, nc = (fr + dr) % rows, (fc + dc) % cols
                        if passable[nr, nc] and (nr, nc) not in friendly_hills_set:
                            intercept_mask[nr, nc] = True
            if intercept_mask.any():
                dist_fortify, meta_fortify = compute_geodesic_potential_field(
                    sources={0: intercept_mask},
                    passable_mask=passable,
                    rows=rows,
                    cols=cols,
                    max_depth=16,
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

        remaining_ants = [idx_ant for idx_ant in non_raid_ants if idx_ant not in gather_set and idx_ant not in defense_set]

        dist_frontier: np.ndarray | None = None
        meta_frontier: FieldMetadata | None = None

        if remaining_ants or defense_set:
            g_frontier = extract_exploration_frontier(seen, water, rows, cols, food_mask)
            source_grid_frontier = np.zeros((rows, cols), dtype=np.int32)
            if g_frontier.any():
                source_grid_frontier[g_frontier] = SOURCE_FRONTIER_UNSEEN
                dist_frontier, meta_frontier = compute_geodesic_potential_field(
                    sources={0: g_frontier},
                    passable_mask=passable_known,
                    rows=rows,
                    cols=cols,
                    max_depth=None,
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

            frontier_has_target = bool((dist_frontier < FAR).any())
            hill_has_target = bool((dist_hill < FAR).any())

            for idx_ant in remaining_ants:
                ar, ac = ant_positions[idx_ant]
                if dist_frontier[ar, ac] < FAR:
                    ant_tasks[idx_ant] = TaskState.EXPLORE
                    ant_dist_map[idx_ant] = (dist_frontier, meta_frontier)
                elif hill_has_target and dist_hill[ar, ac] < FAR:
                    # 消除无前沿时回退 FORTIFY 驻守己方蚁穴的缺陷，改由已知敌丘牵引推进
                    ant_tasks[idx_ant] = TaskState.ASSAULT
                    ant_dist_map[idx_ant] = (dist_hill, meta_hill)
                elif frontier_has_target:
                    ant_tasks[idx_ant] = TaskState.EXPLORE
                    ant_dist_map[idx_ant] = (dist_frontier, meta_frontier)
                elif hill_has_target:
                    ant_tasks[idx_ant] = TaskState.ASSAULT
                    ant_dist_map[idx_ant] = (dist_hill, meta_hill)
                else:
                    ant_tasks[idx_ant] = TaskState.EXPLORE
                    ant_dist_map[idx_ant] = (dist_frontier, meta_frontier)

        for idx_ant in defense_set:
            ant_tasks[idx_ant] = TaskState.FORTIFY
            if dist_fortify is not None and dist_fortify[ant_positions[idx_ant][0], ant_positions[idx_ant][1]] < FAR:
                ant_dist_map[idx_ant] = (dist_fortify, meta_fortify)
            elif dist_frontier is not None and dist_frontier[ant_positions[idx_ant][0], ant_positions[idx_ant][1]] < FAR:
                ant_dist_map[idx_ant] = (dist_frontier, meta_frontier)
            else:
                ant_dist_map[idx_ant] = (dist_hill, meta_hill)

        for idx_ant in raid_set:
            ant_dist_map[idx_ant] = (dist_hill, meta_hill)

        self._last_ant_tasks = dict(ant_tasks)

        # 5. 生成候选移动动作序列与测地负梯度下降（己方蚁穴上的蚂蚁强制优先走离，杜绝阻塞产卵）
        potentials = np.zeros(total_ants, dtype=np.int32)
        candidate_sequences: list[list[int]] = []
        candidate_rejections_map: dict[int, dict[int, str]] = {i: {} for i in range(total_ants)}

        effective_tiers = list(squad_tiers) if squad_tiers is not None else [1] * total_ants
        has_own_hill_ant = False
        for i in range(total_ants):
            ar, ac = ant_positions[i]
            on_own_hill = (ar, ac) in friendly_hills_set
            if on_own_hill:
                effective_tiers[i] = 0
                has_own_hill_ant = True
            dist_field, _ = ant_dist_map[i]
            cur_d = int(dist_field[ar, ac])
            potentials[i] = cur_d
            rejections = candidate_rejections_map[i]

            if (not on_own_hill) and i in ant_bound_target and (ar, ac) == ant_bound_target[i]:
                candidate_sequences.append([HOLD])
                continue

            valid_moves: list[tuple[int, int, int]] = []
            fallback_off_hill: list[tuple[int, int, int]] = []
            for k in range(4):
                dr, dc = DELTAS[k]
                nr, nc = (ar + dr) % rows, (ac + dc) % cols
                if water[nr, nc]:
                    rejections[k] = RejectionCode.WATER.value
                    continue
                if food_mask[nr, nc]:
                    rejections[k] = RejectionCode.FOOD.value
                    continue
                step_d = int(dist_field[nr, nc])
                threat_cost = int(int_delay[nr, nc]) if int_delay is not None else 0
                step_onto_hill = 1 if ((not on_own_hill) and (nr, nc) in friendly_hills_set) else 0
                if step_d >= cur_d:
                    if on_own_hill:
                        fallback_off_hill.append((threat_cost, step_d, k))
                    if (nr, nc) in ant_positions_set:
                        rejections[k] = RejectionCode.TAKEN.value
                    else:
                        rejections[k] = RejectionCode.GRADIENT_NON_DECREASING.value
                    continue
                valid_moves.append((step_onto_hill, step_d + threat_cost * 2, k))

            valid_moves.sort(key=lambda item: (item[0], item[1], item[2]))
            cand_seq = [item[2] for item in valid_moves]
            if on_own_hill and fallback_off_hill:
                fallback_off_hill.sort(key=lambda item: (item[0], item[1], item[2]))
                cand_seq.extend(item[2] for item in fallback_off_hill)
            cand_seq.append(HOLD)
            candidate_sequences.append(cand_seq)

        # 6. 终点冲突仲裁保底（未显式传入 stall_counters 时严格无状态，杜绝跨对局状态泄漏）
        out, updated_stalls = arbitrate_orders(
            mine_coords=mine,
            candidate_sequences=candidate_sequences,
            potentials=potentials,
            stall_counters=stall_counters,
            fair_mode=fair_mode if stall_counters is not None else False,
            rows=rows,
            cols=cols,
            ids=ant_ids,
            squad_tiers=effective_tiers if (squad_tiers is not None or has_own_hill_ant) else None,
            rejections_out=candidate_rejections_map,
        )

        if stall_counters is not None:
            stall_counters.clear()
            stall_counters.update(updated_stalls)

        # 7. 遥测数据采集
        telemetry_records: list[TelemetryRecord] = []
        if collect_telemetry:
            visible_food_set = {tuple(f) for f in food_coords}
            for i in range(total_ants):
                ant_id = ant_ids[i]
                ar, ac = ant_positions[i]
                dist_field, meta_field = ant_dist_map[i]
                pot_self = int(dist_field[ar, ac])
                has_reachable = bool((dist_field < FAR).any())
                is_stale = (pot_self == int(FAR) and has_reachable)

                record = evaluate_ant_telemetry(
                    ant_id=int(ant_id),
                    pos=(ar, ac),
                    dist=dist_field,
                    field_meta=meta_field,
                    candidate_rejections=candidate_rejections_map.get(i, {}),
                    action=int(out[i]),
                    visible_food=visible_food_set,
                    friendly_hills=friendly_hills_set,
                    is_stale=is_stale,
                )
                telemetry_records.append(record)

        self._turn_telemetry = telemetry_records
        return out

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

    def step(
        self,
        obs: dict,
        water: np.ndarray,
        seen: np.ndarray,
        memory: dict | None = None,
        collect_telemetry: bool = False,
        stall_counters: dict[int, int] | None = None,
        fair_mode: bool = True,
    ) -> np.ndarray:
        return self.orders(
            obs,
            water,
            seen,
            memory,
            collect_telemetry=collect_telemetry,
            stall_counters=stall_counters,
            fair_mode=fair_mode,
        )
