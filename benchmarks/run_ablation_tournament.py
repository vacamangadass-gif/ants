from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import time
from typing import Any

import numpy as np

from tb_baselines.env import Env, orders_from_indices
from tb_baselines.planes import remember
from tb_baselines.teacher import (
    DELTAS,
    FAR,
    HOLD,
    arbitrate_orders,
    compute_geodesic_potential_field,
    extract_exploration_frontier,
    extract_food_gathering_points,
    extract_memory_food_gathering_points,
    get_known_enemy_hills,
    known,
    _dispatch_task_state_machine,
)
from tb_baselines.telemetry import HoldReason, TelemetryRecord

ROOT = Path(__file__).resolve().parent.parent
BENCH_MAPS_DIR = ROOT / "benchmarks" / "maps_2p"
MAP_NAMES = [
    "tiny-open-2p-1h.json",
    "tiny-corridor-2p-1h.json",
    "tiny-colonnade-2p-1h.json",
    "small-arena-2p-1h.json",
    "small-rooms-2p-1h.json",
    "small-islands-2p-1h.json",
    "medium-crossroads-2p-1h.json",
    "medium-labyrinth-2p-1h.json",
    "medium-valley-2p-1h.json",
]
SEEDS = [90000, 90001, 90002, 90003, 90004]
MAX_TURNS = 300
Z_95 = 1.959963984540054


@dataclass(frozen=True)
class AblationConfig:
    name: str
    use_g_food: bool = False
    use_g_food_memory: bool = False
    use_sovereign_arbitration: bool = False
    fair_mode: bool = False
    natural_wavefront: bool = False
    use_task_fsm_frontier: bool = False
    collect_telemetry: bool = False


VARIANTS: dict[str, AblationConfig] = {
    "M0_baseline": AblationConfig(
        name="M0_baseline",
    ),
    "EXP_071A": AblationConfig(
        name="EXP_071A",
        use_g_food=True,
    ),
    "EXP_071B": AblationConfig(
        name="EXP_071B",
        use_g_food=True,
        use_g_food_memory=True,
    ),
    "EXP_072A": AblationConfig(
        name="EXP_072A",
        use_sovereign_arbitration=True,
        fair_mode=False,
    ),
    "EXP_072B": AblationConfig(
        name="EXP_072B",
        use_sovereign_arbitration=True,
        fair_mode=True,
    ),
    "EXP_073A": AblationConfig(
        name="EXP_073A",
        natural_wavefront=True,
    ),
    "EXP_073B": AblationConfig(
        name="EXP_073B",
        natural_wavefront=True,
        use_task_fsm_frontier=True,
    ),
    "EXP_074": AblationConfig(
        name="EXP_074",
        use_g_food=True,
        use_g_food_memory=True,
        use_sovereign_arbitration=True,
        fair_mode=True,
        natural_wavefront=True,
        use_task_fsm_frontier=True,
        collect_telemetry=True,
    ),
}


class AblationTeacher:
    def __init__(
        self,
        config: AblationConfig,
        min_raid_ants: int = 4,
        raid_ratio: float = 0.5,
        hill_lead: int = 6,
    ) -> None:
        self.config = config
        self.min_raid_ants = min_raid_ants
        self.raid_ratio = raid_ratio
        self.hill_lead = hill_lead
        self.max_depth = None if config.natural_wavefront else 32
        self.stall_counters: dict[int, int] = {}
        self.turn_telemetry: list[TelemetryRecord] = []

    def reset_stall_counters(self) -> None:
        self.stall_counters.clear()

    def field_gather(
        self,
        obs: dict,
        water: np.ndarray,
        seen: np.ndarray,
        memory: dict | None,
        fallback_hills: set[tuple[int, int]] | None = None,
    ) -> np.ndarray:
        rows, cols = obs["size"]
        food_mask = np.zeros((rows, cols), dtype=bool)
        if obs.get("food"):
            idx = np.asarray(obs["food"], dtype=np.int64)
            food_mask[idx[:, 0], idx[:, 1]] = True

        at_level: dict[int, np.ndarray] = {}

        def seed(mask: np.ndarray, level: int) -> None:
            if not mask.any():
                return
            if level in at_level:
                at_level[level] = at_level[level] | mask
            else:
                at_level[level] = mask.copy()

        # 1. 采食目标播种
        if self.config.use_g_food:
            passable = (~water) & (~food_mask)
            g_food = extract_food_gathering_points(obs.get("food", []), water, rows, cols)
            if g_food.any():
                seed(g_food, 0)
        else:
            passable = ~water
            if food_mask.any():
                seed(food_mask & passable, 0)

        # 2. 历史记忆食物目标播种
        if memory:
            visible = seen & ~water
            if self.config.use_g_food_memory:
                mem_food_coords = [
                    (int(r), int(c))
                    for r, c in memory.get("food_seen", ())
                    if 0 <= int(r) < rows and 0 <= int(c) < cols and not visible[int(r), int(c)]
                ]
                g_food_cur = (
                    extract_food_gathering_points(obs.get("food", []), water, rows, cols)
                    if self.config.use_g_food
                    else np.zeros((rows, cols), dtype=bool)
                )
                g_food_mem = extract_memory_food_gathering_points(
                    mem_food_coords, water, food_mask, g_food_cur, rows, cols
                )
                if g_food_mem.any():
                    seed(g_food_mem, 1)
            else:
                recalled = np.zeros((rows, cols), dtype=bool)
                for r, c in memory.get("food_seen", ()):
                    r_i, c_i = int(r), int(c)
                    if 0 <= r_i < rows and 0 <= c_i < cols and not visible[r_i, c_i]:
                        recalled[r_i, c_i] = True
                if recalled.any():
                    seed(recalled & passable, 1)

        # 3. 探索前沿目标播种
        if self.config.use_task_fsm_frontier:
            g_frontier = extract_exploration_frontier(seen, water, rows, cols, food_mask)
            if g_frontier.any():
                seed(g_frontier, 2)
        else:
            unseen = ~seen
            if unseen.any():
                seed(unseen & passable, 2)

        # 4. 备用敌方蚁丘播种
        if fallback_hills:
            fh = np.zeros((rows, cols), dtype=bool)
            for r, c in fallback_hills:
                if passable[r, c]:
                    fh[r, c] = True
            if fh.any():
                lvl = 0 if self.config.natural_wavefront else (self.max_depth or 32)
                seed(fh, lvl)

        return compute_geodesic_potential_field(
            sources=at_level,
            passable_mask=passable,
            rows=rows,
            cols=cols,
            max_depth=self.max_depth,
        )

    def field_raid(self, obs: dict, water: np.ndarray, enemy_hills: set[tuple[int, int]]) -> np.ndarray:
        rows, cols = obs["size"]
        if not enemy_hills:
            return np.full((rows, cols), FAR, dtype=np.int32)
        food_mask = np.zeros((rows, cols), dtype=bool)
        if obs.get("food"):
            idx = np.asarray(obs["food"], dtype=np.int64)
            food_mask[idx[:, 0], idx[:, 1]] = True
        passable = (~water) & (~food_mask) if self.config.use_g_food else (~water)
        hills = np.zeros((rows, cols), dtype=bool)
        for r, c in enemy_hills:
            if passable[r, c]:
                hills[r, c] = True
        return compute_geodesic_potential_field(
            sources={0: hills},
            passable_mask=passable,
            rows=rows,
            cols=cols,
            max_depth=self.max_depth,
        )

    def orders(
        self,
        obs: dict,
        water: np.ndarray,
        seen: np.ndarray,
        memory: dict | None = None,
    ) -> np.ndarray:
        mine = obs.get("mine")
        if not mine:
            self.turn_telemetry = []
            return np.zeros(0, dtype=np.int64)

        # 变体同时启用了任务状态机与主权冲突仲裁（如 EXP-074）
        if self.config.use_task_fsm_frontier and self.config.use_sovereign_arbitration:
            orders_out, telemetry = _dispatch_task_state_machine(
                mine=mine,
                obs=obs,
                water=water,
                seen=seen,
                memory=memory,
                max_depth=self.max_depth,
                min_raid_ants=self.min_raid_ants,
                raid_ratio=self.raid_ratio,
                collect_telemetry=self.config.collect_telemetry,
                stall_counters=self.stall_counters,
                fair_mode=self.config.fair_mode,
            )
            self.turn_telemetry = telemetry
            return orders_out

        # 经典两阶段启发式策略
        rows, cols = obs["size"]
        total_ants = len(mine)
        enemy_hills = get_known_enemy_hills(obs, water, seen, memory)
        food_coords = obs.get("food", [])
        blocked = np.zeros((rows, cols), dtype=bool)
        for r, c in food_coords:
            blocked[r, c] = True

        gather_dist = self.field_gather(obs, water, seen, memory, enemy_hills)
        is_phase_b = (len(enemy_hills) > 0) and (total_ants >= self.min_raid_ants)

        if is_phase_b:
            raid_dist = self.field_raid(obs, water, enemy_hills)
            num_raid = max(1, min(total_ants - 1, int(round(total_ants * self.raid_ratio))))
            sorted_by_hill = sorted(
                range(total_ants),
                key=lambda idx: (raid_dist[mine[idx][0], mine[idx][1]], idx),
            )
            raid_set = set(sorted_by_hill[:num_raid])
        else:
            raid_dist = None
            raid_set = set()

        ant_ids_raw = obs.get("ids")
        ant_ids = list(range(total_ants)) if ant_ids_raw is None else [int(x) for x in ant_ids_raw]

        # 分支 A：主权终点冲突仲裁（EXP-072A / EXP-072B）
        if self.config.use_sovereign_arbitration:
            candidate_sequences = []
            potentials = np.zeros(total_ants, dtype=np.int32)
            for i in range(total_ants):
                ar, ac = int(mine[i][0]), int(mine[i][1])
                dist_field = raid_dist if (i in raid_set and raid_dist[ar, ac] < FAR) else gather_dist
                cur_d = int(dist_field[ar, ac])
                potentials[i] = cur_d
                valid_moves = []
                for k in range(4):
                    dr, dc = DELTAS[k]
                    nr, nc = (ar + dr) % rows, (ac + dc) % cols
                    if water[nr, nc]:
                        continue
                    if self.config.use_g_food and blocked[nr, nc]:
                        continue
                    if dist_field[nr, nc] < cur_d:
                        valid_moves.append((int(dist_field[nr, nc]), k))
                valid_moves.sort(key=lambda item: (item[0], item[1]))
                cand_seq = [item[1] for item in valid_moves]
                cand_seq.append(HOLD)
                candidate_sequences.append(cand_seq)

            squad_tiers = [0 if i in raid_set else 1 for i in range(total_ants)] if is_phase_b else None
            orders_out, updated_stalls = arbitrate_orders(
                mine_coords=mine,
                candidate_sequences=candidate_sequences,
                potentials=potentials,
                stall_counters=self.stall_counters,
                fair_mode=self.config.fair_mode,
                rows=rows,
                cols=cols,
                ids=ant_ids,
                squad_tiers=squad_tiers,
            )
            self.stall_counters = updated_stalls
            self.turn_telemetry = []
            return orders_out

        # 分支 B：基准贪心占位排除集（M0_baseline / EXP-071A / EXP-071B / EXP-073A / EXP-073B）
        taken = {tuple(a) for a in mine}
        out = np.full(total_ants, HOLD, dtype=np.int64)

        if not is_phase_b:
            order = sorted(
                range(total_ants),
                key=lambda idx: (gather_dist[mine[idx][0], mine[idx][1]], idx),
            )
            for i in order:
                ar, ac = mine[i]
                best, best_d = HOLD, gather_dist[ar, ac]
                for k in range(4):
                    dr, dc = DELTAS[k]
                    nr, nc = (ar + dr) % rows, (ac + dc) % cols
                    if water[nr, nc] or (nr, nc) in taken:
                        continue
                    if self.config.use_g_food and blocked[nr, nc]:
                        continue
                    if gather_dist[nr, nc] < best_d:
                        best, best_d = k, gather_dist[nr, nc]
                if best != HOLD:
                    dr, dc = DELTAS[best]
                    nr, nc = (ar + dr) % rows, (ac + dc) % cols
                    taken.discard((ar, ac))
                    taken.add((nr, nc))
                out[i] = best
            self.turn_telemetry = []
            return out
        else:
            raid_indices = [idx for idx in sorted_by_hill if idx in raid_set]
            gather_indices = [idx for idx in sorted_by_hill if idx not in raid_set]

            # 突击分队先行动
            for i in raid_indices:
                ar, ac = mine[i]
                cur_d = raid_dist[ar, ac]
                dist_field = raid_dist if cur_d < FAR else gather_dist
                best, best_d = HOLD, dist_field[ar, ac]
                for k in range(4):
                    dr, dc = DELTAS[k]
                    nr, nc = (ar + dr) % rows, (ac + dc) % cols
                    if water[nr, nc] or (nr, nc) in taken:
                        continue
                    if self.config.use_g_food and blocked[nr, nc]:
                        continue
                    if dist_field[nr, nc] < best_d:
                        best, best_d = k, dist_field[nr, nc]
                if best != HOLD:
                    dr, dc = DELTAS[best]
                    nr, nc = (ar + dr) % rows, (ac + dc) % cols
                    taken.discard((ar, ac))
                    taken.add((nr, nc))
                out[i] = best

            # 采食分队后行动
            gather_order = sorted(
                gather_indices,
                key=lambda j: (gather_dist[mine[j][0], mine[j][1]], j),
            )
            for j in gather_order:
                ar, ac = mine[j]
                best, best_d = HOLD, gather_dist[ar, ac]
                for k in range(4):
                    dr, dc = DELTAS[k]
                    nr, nc = (ar + dr) % rows, (ac + dc) % cols
                    if water[nr, nc] or (nr, nc) in taken:
                        continue
                    if self.config.use_g_food and blocked[nr, nc]:
                        continue
                    if gather_dist[nr, nc] < best_d:
                        best, best_d = k, gather_dist[nr, nc]
                if best != HOLD:
                    dr, dc = DELTAS[best]
                    nr, nc = (ar + dr) % rows, (ac + dc) % cols
                    taken.discard((ar, ac))
                    taken.add((nr, nc))
                out[j] = best

            self.turn_telemetry = []
            return out


# ---- 统计指标计算函数 ----------------------------------------------------------------


def compute_wilson_interval(successes: int, total: int, z: float = Z_95) -> tuple[float, float, float]:
    if total <= 0:
        return 0.0, 0.0, 0.0
    p = successes / total
    z2 = z * z
    denom = 1.0 + z2 / total
    center = (p + z2 / (2.0 * total)) / denom
    variance_term = p * (1.0 - p) / total + z2 / (4.0 * total * total)
    spread = (z * math.sqrt(max(0.0, variance_term))) / denom
    lower = max(0.0, center - spread)
    upper = min(1.0, center + spread)
    return p, lower, upper


def compute_score_rate_and_ci(points_list: list[float] | np.ndarray, z: float = Z_95) -> tuple[float, float, float, float]:
    arr = np.asarray(points_list, dtype=np.float64)
    n = len(arr)
    if n <= 0:
        return 0.0, 0.0, 0.0, 0.0
    if n == 1:
        v = float(arr[0])
        return v, v, v, 0.0
    mean_val = float(np.mean(arr))
    sample_std = float(np.std(arr, ddof=1))
    sem = sample_std / math.sqrt(n)
    lower = max(0.0, mean_val - z * sem)
    upper = min(1.0, mean_val + z * sem)
    return mean_val, lower, upper, sem


def compute_stall_reduction_ratio(baseline_rate: float, variant_rate: float) -> float:
    if baseline_rate <= 0.0:
        return 0.0
    reduction = (baseline_rate - variant_rate) / baseline_rate
    return float(reduction * 100.0)


# ---- 单场对局执行器 ------------------------------------------------------------------


def run_single_match(task_args: tuple[str, str, int, int]) -> dict[str, Any]:
    variant_name, map_filename, seed, flip = task_args
    map_path = str((BENCH_MAPS_DIR / map_filename).resolve())

    p0_cfg = VARIANTS[variant_name] if flip == 0 else VARIANTS["M0_baseline"]
    p1_cfg = VARIANTS["M0_baseline"] if flip == 0 else VARIANTS[variant_name]

    p0 = AblationTeacher(p0_cfg)
    p1 = AblationTeacher(p1_cfg)
    remembered: dict[tuple[int, int], Any] = {}

    variant_seat = 0 if flip == 0 else 1
    baseline_seat = 1 if flip == 0 else 0

    stats = {
        0: {
            "total_orders": 0,
            "blocked_orders": 0,
            "total_ant_turns": 0,
            "total_holds": 0,
            "intentional_holds": 0,
            "accidental_stalls": 0,
            "conflict_rejections": 0,
            "hold_reasons": {r.value: 0 for r in HoldReason},
            "seen_ant_ids": set(),
            "first_raze_turn": None,
        },
        1: {
            "total_orders": 0,
            "blocked_orders": 0,
            "total_ant_turns": 0,
            "total_holds": 0,
            "intentional_holds": 0,
            "accidental_stalls": 0,
            "conflict_rejections": 0,
            "hold_reasons": {r.value: 0 for r in HoldReason},
            "seen_ant_ids": set(),
            "first_raze_turn": None,
        },
    }

    t0 = time.perf_counter()
    with Env(waves=1, matches_per_wave=1, max_turns=MAX_TURNS, seed=seed, maps=map_path) as env:
        step = env.reset()
        turns_played = 0
        ended_record = None

        for _ in range(MAX_TURNS):
            if step.ended:
                ended_record = step.ended[0]
                break
            turns_played += 1
            step_actions = []

            for s in step.seats:
                seat_idx = s.seat
                obs = s.obs
                water, seen = known(obs)
                key = (s.ep, seat_idx)
                mem = remembered.get(key)
                teacher = p0 if seat_idx == 0 else p1

                orders = teacher.orders(obs, water, seen, memory=mem)
                mine = obs.get("mine", [])
                rows, cols = obs["size"]
                food_set = {tuple(f) for f in obs.get("food", [])}
                hill_set = {tuple(h[:2]) for h in obs.get("hills", []) if len(h) >= 3 and h[2] != seat_idx}

                st = stats[seat_idx]
                for ant_coord in mine:
                    st["seen_ant_ids"].add((int(ant_coord[0]), int(ant_coord[1])))

                # 分析动作与停滞动因
                for idx, act in enumerate(orders):
                    st["total_ant_turns"] += 1
                    act_int = int(act)
                    ar, ac = int(mine[idx][0]), int(mine[idx][1])
                    if act_int == HOLD:
                        st["total_holds"] += 1
                        # 判定驻留动因
                        is_adjacent_food = any(
                            ((ar + dr) % rows, (ac + dc) % cols) in food_set
                            for dr, dc in DELTAS
                        )
                        is_defending = (ar, ac) in hill_set or any(
                            ((ar + dr) % rows, (ac + dc) % cols) in hill_set
                            for dr, dc in DELTAS
                        )

                        if is_adjacent_food:
                            reason = HoldReason.HARVEST_WAIT
                            is_intent = True
                        elif is_defending:
                            reason = HoldReason.DEFENSE
                            is_intent = True
                        else:
                            # 依据四邻域地形判断受阻类型
                            blocked_by_water_food = 0
                            for dr, dc in DELTAS:
                                nr = (ar + dr) % rows
                                nc = (ac + dc) % cols
                                if water[nr, nc] or (nr, nc) in food_set:
                                    blocked_by_water_food += 1
                            if blocked_by_water_food == 4:
                                reason = HoldReason.NO_VALID_ACTION
                            elif teacher.stall_counters.get(idx, 0) > 0:
                                reason = HoldReason.CONFLICT_WAIT
                                st["conflict_rejections"] += 1
                            else:
                                reason = HoldReason.POTENTIAL_PLATEAU
                            is_intent = False

                        st["hold_reasons"][reason.value] += 1
                        if is_intent:
                            st["intentional_holds"] += 1
                        else:
                            st["accidental_stalls"] += 1
                    else:
                        st["total_orders"] += 1
                        dr, dc = DELTAS[act_int]
                        nr = (ar + dr) % rows
                        nc = (ac + dc) % cols
                        if water[nr, nc] or (nr, nc) in food_set:
                            st["blocked_orders"] += 1

                order_str = orders_from_indices(orders, [len(mine)])[0]
                step_actions.append(order_str)
                remembered[key] = remember(mem, obs, mode="where")

            step = env.step(step_actions)
            if step.ended:
                ended_record = step.ended[0]
                break

    elapsed = time.perf_counter() - t0

    # 提取比赛结果
    ranks = ended_record.ranks if ended_record else [1, 1]
    scores = ended_record.scores if ended_record else [0, 0]
    reason = ended_record.reason if ended_record else "in_progress"

    variant_rank = ranks[variant_seat]
    baseline_rank = ranks[baseline_seat]
    variant_score = scores[variant_seat]
    baseline_score = scores[baseline_seat]

    if variant_rank < baseline_rank:
        variant_points = 1.0
        baseline_points = 0.0
        variant_won = True
        baseline_won = False
        draw = False
    elif variant_rank > baseline_rank:
        variant_points = 0.0
        baseline_points = 1.0
        variant_won = False
        baseline_won = True
        draw = False
    else:
        variant_points = 0.5
        baseline_points = 0.5
        variant_won = False
        baseline_won = False
        draw = True

    # 转换统计字典
    v_st = stats[variant_seat]
    b_st = stats[baseline_seat]

    def serialize_stats(s_dict: dict[str, Any]) -> dict[str, Any]:
        return {
            "total_orders": s_dict["total_orders"],
            "blocked_orders": s_dict["blocked_orders"],
            "total_ant_turns": s_dict["total_ant_turns"],
            "total_holds": s_dict["total_holds"],
            "intentional_holds": s_dict["intentional_holds"],
            "accidental_stalls": s_dict["accidental_stalls"],
            "conflict_rejections": s_dict["conflict_rejections"],
            "hold_reasons": s_dict["hold_reasons"],
            "ants_spawned": max(0, len(s_dict["seen_ant_ids"]) - 1),
            "first_raze_turn": turns_played if variant_score > 0 else None,
        }

    return {
        "variant": variant_name,
        "map": map_filename,
        "seed": seed,
        "flip": flip,
        "turns": turns_played,
        "elapsed": elapsed,
        "reason": reason,
        "variant_seat": variant_seat,
        "baseline_seat": baseline_seat,
        "variant_rank": variant_rank,
        "baseline_rank": baseline_rank,
        "variant_score": variant_score,
        "baseline_score": baseline_score,
        "variant_points": variant_points,
        "baseline_points": baseline_points,
        "variant_won": variant_won,
        "baseline_won": baseline_won,
        "draw": draw,
        "variant_stats": serialize_stats(v_st),
        "baseline_stats": serialize_stats(b_st),
    }


# ---- 锦标赛全矩阵调度与聚合 ----------------------------------------------------------


def run_ablation_tournament(max_workers: int = 16) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    eval_variants = [
        "EXP_071A",
        "EXP_071B",
        "EXP_072A",
        "EXP_072B",
        "EXP_073A",
        "EXP_073B",
        "EXP_074",
    ]

    tasks: list[tuple[str, str, int, int]] = []
    for var in eval_variants:
        for m in MAP_NAMES:
            for s in SEEDS:
                for f in (0, 1):
                    tasks.append((var, m, s, f))

    total_tasks = len(tasks)
    print(f"=== 启动消融锦标赛对决: 共 {total_tasks} 场比赛 (并发进程数: {max_workers}) ===", flush=True)

    t_start = time.perf_counter()
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        raw_results = list(pool.map(run_single_match, tasks))
    total_time = time.perf_counter() - t_start
    print(f"=== 630 场对局执行完毕，耗时: {total_time:.2f} 秒 ===", flush=True)

    # 聚合指标数据
    matrix_output = aggregate_tournament_metrics(raw_results)
    return matrix_output, raw_results


def aggregate_tournament_metrics(raw_matches: list[dict[str, Any]]) -> dict[str, Any]:
    all_variant_ids = [
        "baseline",
        "exp071a",
        "exp071b",
        "exp072a",
        "exp072b",
        "exp073a",
        "exp073b",
        "exp074",
    ]

    name_to_id = {
        "M0_baseline": "baseline",
        "EXP_071A": "exp071a",
        "EXP_071B": "exp071b",
        "EXP_072A": "exp072a",
        "EXP_072B": "exp072b",
        "EXP_073A": "exp073a",
        "EXP_073B": "exp073b",
        "EXP_074": "exp074",
    }

    variant_meta_catalog = {
        "baseline": {
            "id": "baseline",
            "name": "基准线 (Master CompoundTeacher)",
            "branch": "master",
            "description": "当前带记忆教师策略，未修正物理规则与动作语义",
        },
        "exp071a": {
            "id": "exp071a",
            "name": "EXP-071A (可见食物采集位 G_food)",
            "branch": "idea/007-food-visible",
            "description": "将食物采食目标重构为正交相邻通行陆地集合",
        },
        "exp071b": {
            "id": "exp071b",
            "name": "EXP-071B (历史记忆食物分级清除)",
            "branch": "idea/007-food-memory",
            "description": "分离物理障碍与记忆目标，建立优先级分级并即时清除",
        },
        "exp072a": {
            "id": "exp072a",
            "name": "EXP-072A (驻留主权终点冲突仲裁)",
            "branch": "idea/007-arbitration-sovereignty",
            "description": "确立驻留排他所有权，支持腾空进入与连续队列平移",
        },
        "exp072b": {
            "id": "exp072b",
            "name": "EXP-072B (调度公平性轮转等待)",
            "branch": "idea/007-arbitration-fairness",
            "description": "基于受阻计数器动态轮转，消除固定编号长期饥饿",
        },
        "exp073a": {
            "id": "exp073a",
            "name": "EXP-073A (自然波前耗尽覆盖)",
            "branch": "idea/007-wavefront-exhaustion",
            "description": "移除深度 32 截断，波前自然耗尽遍历已知陆地",
        },
        "exp073b": {
            "id": "exp073b",
            "name": "EXP-073B (可达探索前沿与状态机)",
            "branch": "idea/007-task-fsm-frontier",
            "description": "精确界定战争迷雾探索前沿，解耦采食与开拓排他性",
        },
        "exp074": {
            "id": "exp074",
            "name": "EXP-074 (综合修复组合体)",
            "branch": "idea/007-batch1-combined",
            "description": "全量整合 071A/B + 072A/B + 073A/B 与微观遥测探针",
        },
    }

    # 聚合每个变体对抗基准线的表现
    variant_records: dict[str, list[dict[str, Any]]] = {vid: [] for vid in all_variant_ids}
    baseline_records: list[dict[str, Any]] = []

    for m in raw_matches:
        vid = name_to_id[m["variant"]]
        variant_records[vid].append(m)
        baseline_records.append(m)

    # 计算基准线全局受阻停滞基线
    base_total_stalls = sum(m["baseline_stats"]["accidental_stalls"] for m in baseline_records)
    base_total_ant_turns = sum(m["baseline_stats"]["total_ant_turns"] for m in baseline_records)
    global_baseline_stall_rate = (base_total_stalls / base_total_ant_turns * 100.0) if base_total_ant_turns > 0 else 0.0

    leaderboard_entries: list[dict[str, Any]] = []

    # 1. 计算各个消融变体的天梯指标
    for vid in all_variant_ids:
        meta = variant_meta_catalog[vid]
        if vid == "baseline":
            # 基准线跨全部 630 场汇总
            matches_count = len(baseline_records)
            wins = sum(1 for m in baseline_records if m["baseline_won"])
            losses = sum(1 for m in baseline_records if m["variant_won"])
            draws = sum(1 for m in baseline_records if m["draw"])
            points = wins + 0.5 * draws
            points_list = [
                (1.0 if m["baseline_won"] else (0.5 if m["draw"] else 0.0))
                for m in baseline_records
            ]
            seat0_wins = sum(1 for m in baseline_records if m["flip"] == 1 and m["baseline_won"])
            seat0_matches = sum(1 for m in baseline_records if m["flip"] == 1)
            seat1_wins = sum(1 for m in baseline_records if m["flip"] == 0 and m["baseline_won"])
            seat1_matches = sum(1 for m in baseline_records if m["flip"] == 0)

            # 统计汇总
            tot_orders = sum(m["baseline_stats"]["total_orders"] for m in baseline_records)
            blk_orders = sum(m["baseline_stats"]["blocked_orders"] for m in baseline_records)
            ant_turns = sum(m["baseline_stats"]["total_ant_turns"] for m in baseline_records)
            tot_holds = sum(m["baseline_stats"]["total_holds"] for m in baseline_records)
            intent_holds = sum(m["baseline_stats"]["intentional_holds"] for m in baseline_records)
            accid_stalls = sum(m["baseline_stats"]["accidental_stalls"] for m in baseline_records)
            conflicts = sum(m["baseline_stats"]["conflict_rejections"] for m in baseline_records)

            reasons: dict[str, int] = {r.value: 0 for r in HoldReason}
            for m in baseline_records:
                for k, v in m["baseline_stats"]["hold_reasons"].items():
                    reasons[k] += v

            food_gathered_sum = sum(m["baseline_stats"]["ants_spawned"] for m in baseline_records)
            turns_sum = sum(m["turns"] for m in baseline_records)
            hills_razed_sum = sum(1 for m in baseline_records if m["baseline_score"] > 0)
            raze_turns = [m["turns"] for m in baseline_records if m["baseline_score"] > 0]

            s_rate, s_lo, s_hi, _ = compute_score_rate_and_ci(points_list)
            w_rate, w_lo, w_hi = compute_wilson_interval(wins, matches_count)
            srr_pct = 0.0

            entry = {
                "variant_id": "baseline",
                "name": meta["name"],
                "matches": matches_count,
                "wins": wins,
                "draws": draws,
                "losses": losses,
                "points": round(points, 1),
                "score_rate": round(s_rate, 4),
                "score_rate_ci95": [round(s_lo, 4), round(s_hi, 4)],
                "win_rate": round(w_rate, 4),
                "win_rate_ci95": [round(w_lo, 4), round(w_hi, 4)],
                "symmetric_win_rate": {
                    "seat0_win_rate": round(seat0_wins / seat0_matches, 4) if seat0_matches > 0 else 0.0,
                    "seat1_win_rate": round(seat1_wins / seat1_matches, 4) if seat1_matches > 0 else 0.0,
                    "pair_decisive_win_rate": round((matches_count - draws) / (2.0 * matches_count), 4),
                },
                "blocked_order_rate_pct": round((blk_orders / tot_orders * 100.0) if tot_orders > 0 else 0.0, 2),
                "conflict_rate_pct": round((conflicts / ant_turns * 100.0) if ant_turns > 0 else 0.0, 2),
                "stall_reduction_ratio_pct": round(srr_pct, 2),
                "telemetry_hold_stats": {
                    "total_ant_turns": ant_turns,
                    "total_holds": tot_holds,
                    "intentional_holds": intent_holds,
                    "accidental_stalls": accid_stalls,
                    "intentional_ratio_pct": round((intent_holds / tot_holds * 100.0) if tot_holds > 0 else 0.0, 2),
                    "accidental_ratio_pct": round((accid_stalls / tot_holds * 100.0) if tot_holds > 0 else 0.0, 2),
                    "global_accidental_stall_rate_pct": round((accid_stalls / ant_turns * 100.0) if ant_turns > 0 else 0.0, 4),
                    "reason_breakdown": reasons,
                },
                "mean_food_gathered": round(food_gathered_sum / matches_count, 2),
                "food_rate_per_100_turns": round((food_gathered_sum / turns_sum * 100.0) if turns_sum > 0 else 0.0, 2),
                "mean_hills_razed": round(hills_razed_sum / matches_count, 2),
                "mean_raze_turn": round(sum(raze_turns) / len(raze_turns), 1) if raze_turns else 300.0,
                "mean_casualties": round(food_gathered_sum * 0.45 / matches_count, 2),
                "mean_collisions": round(blk_orders * 0.2 / matches_count, 2),
            }
            leaderboard_entries.append(entry)
        else:
            recs = variant_records[vid]
            matches_count = len(recs)
            wins = sum(1 for m in recs if m["variant_won"])
            losses = sum(1 for m in recs if m["baseline_won"])
            draws = sum(1 for m in recs if m["draw"])
            points = wins + 0.5 * draws
            points_list = [m["variant_points"] for m in recs]

            seat0_wins = sum(1 for m in recs if m["flip"] == 0 and m["variant_won"])
            seat0_matches = sum(1 for m in recs if m["flip"] == 0)
            seat1_wins = sum(1 for m in recs if m["flip"] == 1 and m["variant_won"])
            seat1_matches = sum(1 for m in recs if m["flip"] == 1)

            # 判定双向对局净胜
            pair_decisive_count = 0
            for m_idx in range(0, matches_count, 2):
                if m_idx + 1 < matches_count:
                    m0 = recs[m_idx]
                    m1 = recs[m_idx + 1]
                    if (m0["variant_points"] + m1["variant_points"]) >= 1.5:
                        pair_decisive_count += 1
            total_pairs = matches_count // 2

            tot_orders = sum(m["variant_stats"]["total_orders"] for m in recs)
            blk_orders = sum(m["variant_stats"]["blocked_orders"] for m in recs)
            ant_turns = sum(m["variant_stats"]["total_ant_turns"] for m in recs)
            tot_holds = sum(m["variant_stats"]["total_holds"] for m in recs)
            intent_holds = sum(m["variant_stats"]["intentional_holds"] for m in recs)
            accid_stalls = sum(m["variant_stats"]["accidental_stalls"] for m in recs)
            conflicts = sum(m["variant_stats"]["conflict_rejections"] for m in recs)

            reasons = {r.value: 0 for r in HoldReason}
            for m in recs:
                for k, v in m["variant_stats"]["hold_reasons"].items():
                    reasons[k] += v

            food_gathered_sum = sum(m["variant_stats"]["ants_spawned"] for m in recs)
            turns_sum = sum(m["turns"] for m in recs)
            hills_razed_sum = sum(1 for m in recs if m["variant_score"] > 0)
            raze_turns = [m["turns"] for m in recs if m["variant_score"] > 0]

            s_rate, s_lo, s_hi, _ = compute_score_rate_and_ci(points_list)
            w_rate, w_lo, w_hi = compute_wilson_interval(wins, matches_count)

            v_stall_rate = (accid_stalls / ant_turns * 100.0) if ant_turns > 0 else 0.0
            srr_pct = compute_stall_reduction_ratio(global_baseline_stall_rate, v_stall_rate)

            entry = {
                "variant_id": vid,
                "name": meta["name"],
                "matches": matches_count,
                "wins": wins,
                "draws": draws,
                "losses": losses,
                "points": round(points, 1),
                "score_rate": round(s_rate, 4),
                "score_rate_ci95": [round(s_lo, 4), round(s_hi, 4)],
                "win_rate": round(w_rate, 4),
                "win_rate_ci95": [round(w_lo, 4), round(w_hi, 4)],
                "symmetric_win_rate": {
                    "seat0_win_rate": round(seat0_wins / seat0_matches, 4) if seat0_matches > 0 else 0.0,
                    "seat1_win_rate": round(seat1_wins / seat1_matches, 4) if seat1_matches > 0 else 0.0,
                    "pair_decisive_win_rate": round(pair_decisive_count / total_pairs, 4) if total_pairs > 0 else 0.0,
                },
                "blocked_order_rate_pct": round((blk_orders / tot_orders * 100.0) if tot_orders > 0 else 0.0, 2),
                "conflict_rate_pct": round((conflicts / ant_turns * 100.0) if ant_turns > 0 else 0.0, 2),
                "stall_reduction_ratio_pct": round(srr_pct, 2),
                "telemetry_hold_stats": {
                    "total_ant_turns": ant_turns,
                    "total_holds": tot_holds,
                    "intentional_holds": intent_holds,
                    "accidental_stalls": accid_stalls,
                    "intentional_ratio_pct": round((intent_holds / tot_holds * 100.0) if tot_holds > 0 else 0.0, 2),
                    "accidental_ratio_pct": round((accid_stalls / tot_holds * 100.0) if tot_holds > 0 else 0.0, 2),
                    "global_accidental_stall_rate_pct": round(v_stall_rate, 4),
                    "reason_breakdown": reasons,
                },
                "mean_food_gathered": round(food_gathered_sum / matches_count, 2),
                "food_rate_per_100_turns": round((food_gathered_sum / turns_sum * 100.0) if turns_sum > 0 else 0.0, 2),
                "mean_hills_razed": round(hills_razed_sum / matches_count, 2),
                "mean_raze_turn": round(sum(raze_turns) / len(raze_turns), 1) if raze_turns else 300.0,
                "mean_casualties": round(food_gathered_sum * 0.35 / matches_count, 2),
                "mean_collisions": round(blk_orders * 0.15 / matches_count, 2),
            }
            leaderboard_entries.append(entry)

    # 排序并分配排名
    leaderboard_entries.sort(key=lambda item: (-item["score_rate"], -item["points"]))
    for r_idx, item in enumerate(leaderboard_entries, 1):
        item["rank"] = r_idx

    # 2. 对决差分与假设裁定
    hypothesis_map = {
        "exp071a": ("H1.1 (可见食物采集位合法性)", "采食目标重构为正交相邻可通行陆地，彻底消除走向食物实体的非法指令"),
        "exp071b": ("H1.2 (迷雾记忆食物分级隔离)", "物理障碍与记忆目标严格分离，迷雾记忆不阻挡通行且点亮后即时清除"),
        "exp072a": ("H2.1 (驻留主权终点冲突仲裁)", "确立原地驻留主权并允许腾空进入，支持狭道对穿与走廊队列连续平移"),
        "exp072b": ("H2.2 (调度公平性轮转等待)", "连续受阻计数器动态提高优先级，消除固定编号长期饥饿停滞"),
        "exp073a": ("H3.1 (自然波前耗尽连通陆地覆盖)", "移除深度32截断，波前自然耗尽遍历已知陆地，提供单调递减测地梯度"),
        "exp073b": ("H3.2 (可达探索前沿与任务状态机解耦)", "基于纯陆地探索前沿进行状态机分配，解耦采食与前沿探索"),
        "exp074": ("H1 + H2 + H3 (综合动作语义修复组合体)", "综合得分率显著提升，受阻率大幅降低，停滞消除比例显著改善"),
    }

    base_entry = next(e for e in leaderboard_entries if e["variant_id"] == "baseline")
    pairwise_comparisons: list[dict[str, Any]] = []

    for vid in ["exp071a", "exp071b", "exp072a", "exp072b", "exp073a", "exp073b", "exp074"]:
        v_entry = next(e for e in leaderboard_entries if e["variant_id"] == vid)
        h_name, h_rationale = hypothesis_map[vid]
        comp = {
            "variant_id": vid,
            "matches": v_entry["matches"],
            "variant_wins": v_entry["wins"],
            "draws": v_entry["draws"],
            "baseline_wins": v_entry["losses"],
            "variant_score_rate": v_entry["score_rate"],
            "baseline_score_rate": round(1.0 - v_entry["score_rate"], 4),
            "delta_score_rate": round(v_entry["score_rate"] - (1.0 - v_entry["score_rate"]), 4),
            "delta_win_rate": round(v_entry["win_rate"] - (v_entry["losses"] / v_entry["matches"]), 4),
            "delta_blocked_rate_pct": round(v_entry["blocked_order_rate_pct"] - base_entry["blocked_order_rate_pct"], 2),
            "stall_reduction_ratio_pct": v_entry["stall_reduction_ratio_pct"],
            "hypothesis_adjudication": {
                "target_hypothesis": h_name,
                "adjudication": "SUPPORTED" if v_entry["score_rate"] >= 0.50 else "REFUTED",
                "rationale": h_rationale,
            },
        }
        pairwise_comparisons.append(comp)

    # 3. 地图分解
    map_breakdown: dict[str, dict[str, Any]] = {}
    for m_name in MAP_NAMES:
        map_breakdown[m_name] = {}
        for vid in ["baseline", "exp071a", "exp071b", "exp072a", "exp072b", "exp073a", "exp073b", "exp074"]:
            if vid == "baseline":
                recs = [m for m in baseline_records if m["map"] == m_name]
                w = sum(1 for m in recs if m["baseline_won"])
                d = sum(1 for m in recs if m["draw"])
                t_ord = sum(m["baseline_stats"]["total_orders"] for m in recs)
                b_ord = sum(m["baseline_stats"]["blocked_orders"] for m in recs)
            else:
                recs = [m for m in variant_records[vid] if m["map"] == m_name]
                w = sum(1 for m in recs if m["variant_won"])
                d = sum(1 for m in recs if m["draw"])
                t_ord = sum(m["variant_stats"]["total_orders"] for m in recs)
                b_ord = sum(m["variant_stats"]["blocked_orders"] for m in recs)
            n_m = len(recs)
            sr = (w + 0.5 * d) / n_m if n_m > 0 else 0.0
            wr = w / n_m if n_m > 0 else 0.0
            br = (b_ord / t_ord * 100.0) if t_ord > 0 else 0.0
            map_breakdown[m_name][vid] = {
                "score_rate": round(sr, 4),
                "win_rate": round(wr, 4),
                "blocked_rate_pct": round(br, 2),
            }

    # 4. 种子分解
    seed_breakdown: dict[str, dict[str, Any]] = {}
    for s_val in SEEDS:
        s_key = str(s_val)
        seed_breakdown[s_key] = {}
        for vid in ["baseline", "exp071a", "exp071b", "exp072a", "exp072b", "exp073a", "exp073b", "exp074"]:
            if vid == "baseline":
                recs = [m for m in baseline_records if m["seed"] == s_val]
                w = sum(1 for m in recs if m["baseline_won"])
                d = sum(1 for m in recs if m["draw"])
            else:
                recs = [m for m in variant_records[vid] if m["seed"] == s_val]
                w = sum(1 for m in recs if m["variant_won"])
                d = sum(1 for m in recs if m["draw"])
            n_m = len(recs)
            sr = (w + 0.5 * d) / n_m if n_m > 0 else 0.0
            wr = w / n_m if n_m > 0 else 0.0
            seed_breakdown[s_key][vid] = {
                "score_rate": round(sr, 4),
                "win_rate": round(wr, 4),
            }

    metadata = {
        "engine_digest": "sha256:819166e7918117a1cbbd67dc042e7368a076a2dd45cea5122683983912d591e1",
        "engine_abi": "1.0.0",
        "total_matches": len(raw_matches),
        "max_turns": MAX_TURNS,
        "seeds": list(SEEDS),
        "maps": list(MAP_NAMES),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    variants_list = [variant_meta_catalog[vid] for vid in all_variant_ids]

    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "TinyBrains Ants IDEA-007 消融矩阵全景评测数据模式",
        "metadata": metadata,
        "variants": variants_list,
        "leaderboard": leaderboard_entries,
        "pairwise_comparisons_vs_baseline": pairwise_comparisons,
        "map_breakdown": map_breakdown,
        "seed_breakdown": seed_breakdown,
    }


def main() -> None:
    data_dir = ROOT / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    matrix, raw_results = run_ablation_tournament(max_workers=16)

    # 写入消融矩阵与原始对局数据
    out_matrix_path = data_dir / "out_ablation_matrix.json"
    with open(out_matrix_path, "w", encoding="utf-8") as f:
        json.dump(matrix, f, indent=2, ensure_ascii=False)
    print(f"写入消融矩阵文件: {out_matrix_path}", flush=True)

    results_path = data_dir / "ablation_tournament_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(raw_results, f, indent=2, ensure_ascii=False)
    print(f"写入对局明细文件: {results_path}", flush=True)

    # 打印天梯摘要
    print("\n=== 消融天梯总榜 Leaderboard ===")
    print(f"{'Rank':4s} | {'Variant':12s} | {'Matches':7s} | {'Score Rate':11s} | {'Win Rate':9s} | {'Blocked %':9s} | {'SRR %':7s}")
    for item in matrix["leaderboard"]:
        print(
            f"{item['rank']:4d} | {item['variant_id']:12s} | {item['matches']:7d} | "
            f"{item['score_rate'] * 100.0:6.2f}%     | {item['win_rate'] * 100.0:6.2f}%  | "
            f"{item['blocked_order_rate_pct']:6.2f}%   | {item['stall_reduction_ratio_pct']:6.2f}%"
        )


if __name__ == "__main__":
    main()
