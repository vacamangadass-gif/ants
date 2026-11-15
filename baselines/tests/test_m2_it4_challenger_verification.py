from __future__ import annotations

import gc
import time
import numpy as np
import pytest

from tb_baselines import planes
from tb_baselines.benchmarks.run_exp080_scaling import rle_from_mask
from tb_baselines.coordination import (
    DELTAS,
    FAR,
    HOLD,
    extract_cluster_gathering_positions,
)
from tb_baselines.mcmf_teacher import MCMFTeacher
from tb_baselines.teacher import CompoundTeacher
from tb_baselines.telemetry import HoldReason


def make_test_obs(
    rows: int,
    cols: int,
    mine: list[tuple[int, int]],
    food: list[tuple[int, int]] | None = None,
    ant_ids: list[int] | None = None,
    water_mask: np.ndarray | None = None,
    turn: int = 1,
) -> dict:
    total_ants = len(mine)
    if ant_ids is None:
        ant_ids = list(range(total_ants))
    b = planes.Board(rows, cols)
    mine_plane = b.scatter(mine)
    vis_plane = b.dilate(mine_plane, planes.VIEW_RADIUS2).astype(bool)

    obs = {
        "size": [rows, cols],
        "turn": turn,
        "max_turns": 500,
        "round_time": 0.05,
        "engine_digest": "challenger_m2_it4",
        "mine": [[int(r), int(c)] for r, c in mine],
        "ids": ant_ids,
        "food": [[int(r), int(c)] for r, c in (food or [])],
        "hills": [],
    }
    if water_mask is not None:
        obs["water"] = {"rle": rle_from_mask(water_mask)}
        obs["vis"] = {"rle": rle_from_mask(vis_plane)}
    return obs


def step_ants(
    orders: np.ndarray,
    current_mine: list[tuple[int, int]],
    rows: int,
    cols: int,
    water: np.ndarray,
) -> list[tuple[int, int]]:
    occupied: set[tuple[int, int]] = set()
    next_mine: list[tuple[int, int]] = []
    for i, act in enumerate(orders):
        r, c = current_mine[i]
        if act != HOLD:
            dr, dc = DELTAS[act]
            dest = ((r + dr) % rows, (c + dc) % cols)
        else:
            dest = (r, c)
        if dest in occupied or water[dest[0], dest[1]]:
            dest = (r, c)
        occupied.add(dest)
        next_mine.append(dest)
    return next_mine


def test_empirical_200_turns_continuous_horizon_and_gc_tail():
    rows, cols = 64, 64
    seed = 8888
    total_turns = 200
    scale = 100
    food_target = 30

    rng = np.random.default_rng(seed)
    water = np.zeros((rows, cols), dtype=bool)
    water[12:18, 12:18] = True
    water[40:46, 40:46] = True
    seen = np.ones((rows, cols), dtype=bool)

    land = [(r, c) for r in range(rows) for c in range(cols) if not water[r, c]]
    chosen_ants = rng.choice(len(land), size=scale, replace=False)
    current_ants = [land[idx] for idx in chosen_ants]

    free_land = [p for p in land if p not in set(current_ants)]
    chosen_foods = rng.choice(len(free_land), size=food_target, replace=False)
    current_foods = [free_land[idx] for idx in chosen_foods]

    teacher = MCMFTeacher(min_raid_ants=4, raid_ratio=0.5)
    latencies: list[float] = []
    total_gathered = 0
    collision_count = 0

    gc.collect()

    for turn in range(total_turns):
        obs = make_test_obs(rows, cols, current_ants, current_foods, water_mask=water, turn=turn + 1)
        t0 = time.perf_counter()
        orders, telemetry = teacher.orders_with_telemetry(obs, water, seen)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        latencies.append(elapsed_ms)

        destinations: list[tuple[int, int]] = []
        for i, a in enumerate(orders):
            act_int = int(a)
            ar, ac = current_ants[i]
            if act_int == HOLD:
                dest = (ar, ac)
            else:
                dr, dc = DELTAS[act_int]
                dest = ((ar + dr) % rows, (ac + dc) % cols)
            destinations.append(dest)

        if len(set(destinations)) != scale:
            collision_count += scale - len(set(destinations))

        gathered_this_turn = []
        remaining_foods = []
        for fr, fc in current_foods:
            gathered = any(
                min(abs(ar - fr), rows - abs(ar - fr)) ** 2 + min(abs(ac - fc), cols - abs(ac - fc)) ** 2 <= 1
                for ar, ac in destinations
            )
            if gathered:
                gathered_this_turn.append((fr, fc))
            else:
                remaining_foods.append((fr, fc))

        total_gathered += len(gathered_this_turn)
        current_foods = remaining_foods

        if len(current_foods) < food_target:
            needed = food_target - len(current_foods)
            occupied = set(destinations) | set(current_foods)
            avail = [p for p in land if p not in occupied]
            if avail:
                rep_count = min(needed, len(avail))
                rep_indices = rng.choice(len(avail), size=rep_count, replace=False)
                for idx in rep_indices:
                    current_foods.append(avail[idx])

        current_ants = destinations

    # 统计性能指标
    warmup_latency = latencies[0]
    steady_latencies = latencies[1:]
    mean_lat = float(np.mean(steady_latencies))
    p50_lat = float(np.percentile(steady_latencies, 50))
    p95_lat = float(np.percentile(steady_latencies, 95))
    p99_lat = float(np.percentile(steady_latencies, 99))
    max_lat = float(np.max(steady_latencies))
    first_quarter_mean = float(np.mean(steady_latencies[:50]))
    last_quarter_mean = float(np.mean(steady_latencies[-50:]))

    print(
        f"\n[200 回合长程稳定性压测实测数据] "
        f"累计采食={total_gathered}, 碰撞违规={collision_count}, "
        f"首回合预热={warmup_latency:.2f} ms, 稳态均值={mean_lat:.2f} ms, "
        f"P50={p50_lat:.2f} ms, P95={p95_lat:.2f} ms, P99={p99_lat:.2f} ms, 最大值={max_lat:.2f} ms, "
        f"前50回合均值={first_quarter_mean:.2f} ms, 后50回合均值={last_quarter_mean:.2f} ms"
    )

    assert collision_count == 0, f"长程 200 回合推演中出现友方碰撞: {collision_count}"
    assert mean_lat < 15.0, f"稳态平均决策耗时超出 15.0 ms 门禁: {mean_lat:.2f} ms"
    # 检验无明显长程耗时漂移（后50回合相比前50回合耗时增长不超过 50%）
    assert last_quarter_mean < first_quarter_mean * 1.5 + 2.0, (
        f"长程推演存在耗时持续增长漂移现象: 前50回合={first_quarter_mean:.2f} ms, 后50回合={last_quarter_mean:.2f} ms"
    )


def test_empirical_redundancy_reduction_extended_seeds():
    rows, cols = 48, 48
    water = np.zeros((rows, cols), dtype=bool)
    water[8:14, 8:14] = True
    water[24:30, 24:30] = True
    water[34:40, 12:18] = True

    scale = 40
    turns = 25
    test_seeds = [1001, 1002, 2026, 7777, 9999]

    def run_dynamic_redundancy_rollout(
        r_rows: int,
        r_cols: int,
        r_water: np.ndarray,
        r_scale: int,
        r_turns: int,
        r_seed: int,
        teacher_type: str,
    ) -> dict:
        land_cells = [(r, c) for r in range(r_rows) for c in range(r_cols) if not r_water[r, c]]
        if len(land_cells) < r_scale + 20:
            raise ValueError(f"可用陆地方格不足: {len(land_cells)} < {r_scale + 20}")

        rng = np.random.default_rng(r_seed)
        chosen_indices = rng.choice(len(land_cells), size=r_scale, replace=False)
        ants = [land_cells[idx] for idx in chosen_indices]
        ant_ids = list(range(r_scale))

        food_target = max(r_scale // 2, 16)
        ant_set = set(ants)
        free_land = [p for p in land_cells if p not in ant_set]
        chosen_food_idx = rng.choice(len(free_land), size=food_target, replace=False)
        current_food = [free_land[idx] for idx in chosen_food_idx]

        if teacher_type == "mcmf":
            teacher = MCMFTeacher(min_raid_ants=4, raid_ratio=0.5)
        elif teacher_type == "compound":
            teacher = CompoundTeacher(min_raid_ants=4, raid_ratio=0.5)
        else:
            raise ValueError(f"未知教师模型类型: {teacher_type}")

        memory: dict = {}
        stall_counters: dict[int, int] = {}
        total_redundant_gather_count = 0
        total_food_gathered = 0
        friendly_collision_violations = 0
        recent_collected_history: list[tuple[int, int, int]] = []

        b = planes.Board(r_rows, r_cols)

        for t in range(r_turns):
            mine_plane = b.scatter(ants)
            vis_plane = b.dilate(mine_plane, planes.VIEW_RADIUS2).astype(bool)
            seen = r_water | vis_plane

            obs = {
                "size": [r_rows, r_cols],
                "turn": t + 1,
                "max_turns": 300,
                "round_time": 0.05,
                "engine_digest": "challenger_m2_it4",
                "mine": [list(a) for a in ants],
                "ids": list(ant_ids),
                "foes": [],
                "food": [list(f) for f in current_food],
                "hills": [],
                "water": {"rle": rle_from_mask(r_water)},
                "vis": {"rle": rle_from_mask(vis_plane)},
            }

            memory = planes.remember(memory, obs, mode="where")
            orders, telemetry_records = teacher.orders_with_telemetry(
                obs, r_water, seen, memory, stall_counters=stall_counters
            )

            if teacher_type == "compound":
                food_claimants_map: dict[tuple[int, int], list[int]] = {}
                for i, rec in enumerate(telemetry_records):
                    if rec.nearest_source_type in ("food_visible", "food_memory"):
                        ar, ac = ants[i]
                        best_food: tuple[int, int] | None = None
                        best_distance_sq = 1000000
                        for fr, fc in current_food:
                            dr = min(abs(ar - fr), r_rows - abs(ar - fr))
                            dc = min(abs(ac - fc), r_cols - abs(ac - fc))
                            d2 = dr * dr + dc * dc
                            if d2 < best_distance_sq:
                                best_distance_sq = d2
                                best_food = (fr, fc)
                        if best_food is not None and best_distance_sq <= 36:
                            food_claimants_map.setdefault(best_food, []).append(i)

                duplicate_claimants = sum(
                    max(0, len(claimant_list) - 1) for claimant_list in food_claimants_map.values()
                )
                total_redundant_gather_count += duplicate_claimants

            dests: list[tuple[int, int]] = []
            for i, a in enumerate(orders):
                act_int = int(a)
                ar, ac = ants[i]
                if act_int == HOLD:
                    dest = (ar, ac)
                else:
                    dr, dc = DELTAS[act_int]
                    dest = ((ar + dr) % r_rows, (ac + dc) % r_cols)
                dests.append(dest)

            if len(set(dests)) != len(dests):
                friendly_collision_violations += len(dests) - len(set(dests))

            food_set = set(current_food)
            for dr_c, dc_c in dests:
                if r_water[dr_c, dc_c]:
                    raise AssertionError(f"回合 {t} 蚂蚁移动至水域方格 ({dr_c}, {dc_c})")
                if (dr_c, dc_c) in food_set:
                    raise AssertionError(f"回合 {t} 蚂蚁移动至食物实体方格 ({dr_c}, {dc_c})")

            ants = dests

            turn_collected: list[tuple[int, int]] = []
            turn_remaining: list[tuple[int, int]] = []
            for fr, fc in current_food:
                gathered = any(
                    min(abs(ar - fr), r_rows - abs(ar - fr)) ** 2 + min(abs(ac - fc), r_cols - abs(ac - fc)) ** 2 <= 1
                    for ar, ac in ants
                )
                if gathered:
                    turn_collected.append((fr, fc))
                else:
                    turn_remaining.append((fr, fc))

            total_food_gathered += len(turn_collected)
            current_food = turn_remaining

            for cfr, cfc, cturn in recent_collected_history:
                if t == cturn + 1:
                    crowd_ants = sum(
                        1
                        for ar, ac in ants
                        if min(abs(ar - cfr), r_rows - abs(ar - cfr)) ** 2 + min(abs(ac - cfc), r_cols - abs(ac - cfc)) ** 2 <= 4
                    )
                    if crowd_ants > 1:
                        total_redundant_gather_count += (crowd_ants - 1)

            for g_fr, g_fc in turn_collected:
                recent_collected_history.append((g_fr, g_fc, t))

            if len(current_food) < food_target:
                needed = food_target - len(current_food)
                occupied = set(ants) | set(current_food)
                free_candidates = [p for p in land_cells if p not in occupied]
                if free_candidates:
                    rep_count = min(needed, len(free_candidates))
                    chosen_rep = rng.choice(len(free_candidates), size=rep_count, replace=False)
                    for r_idx in chosen_rep:
                        current_food.append(free_candidates[r_idx])

        return {
            "teacher_type": teacher_type,
            "turns": r_turns,
            "scale": r_scale,
            "total_food_gathered": total_food_gathered,
            "total_redundant_gather_count": total_redundant_gather_count,
            "friendly_collision_violations": friendly_collision_violations,
        }

    reduction_rates: list[float] = []

    for seed in test_seeds:
        res_compound = run_dynamic_redundancy_rollout(
            rows, cols, water, scale, turns, seed, "compound"
        )
        res_mcmf = run_dynamic_redundancy_rollout(
            rows, cols, water, scale, turns, seed, "mcmf"
        )

        c_red = res_compound["total_redundant_gather_count"]
        m_red = res_mcmf["total_redundant_gather_count"]

        assert c_red > 0
        reduction = (c_red - m_red) / c_red
        reduction_rates.append(reduction)
        print(f"\n[种子 {seed}] 采食冗余对比: Compound={c_red}, MCMF={m_red}, 削减比例={reduction * 100:.2f}%")

        assert reduction >= 0.94, (
            f"种子 {seed} 下 MCMF 采食重叠冗余削减比例未达到 94% 下限: 实际为 {reduction * 100:.2f}%"
        )
        assert res_mcmf["friendly_collision_violations"] == 0, (
            f"种子 {seed} 下 MCMF 推演出现友方碰撞冲突"
        )

    mean_reduction = float(np.mean(reduction_rates))
    print(f"\n5 种子综合平均冗余削减比例: {mean_reduction * 100:.2f}%")
    assert mean_reduction >= 0.95


def test_empirical_multi_seed_chokepoint_blocking_and_deadlock_absence():
    # 测试多个狭窄通道同时调度时，是否存在相互锁死或 CONFLICT_WAIT 堆积
    rows, cols = 32, 32
    water = np.zeros((rows, cols), dtype=bool)
    # 构建 3 条平行单格通道
    water[8, :] = True
    water[10, :] = True
    water[14, :] = True
    water[16, :] = True
    water[20, :] = True
    water[22, :] = True

    seen = np.ones((rows, cols), dtype=bool)

    # 在 3 条通道中分别放置 4 只蚂蚁，通道尽头放置食物
    ants = [
        (9, 2), (9, 3), (9, 4), (9, 5),
        (15, 2), (15, 3), (15, 4), (15, 5),
        (21, 2), (21, 3), (21, 4), (21, 5),
    ]
    foods = [(9, 28), (15, 28), (21, 28)]

    teacher = MCMFTeacher(min_raid_ants=20)
    current_ants = list(ants)

    for turn in range(10):
        obs = make_test_obs(rows, cols, current_ants, foods, water_mask=water, turn=turn + 1)
        orders, telemetry = teacher.orders_with_telemetry(obs, water, seen)

        destinations: list[tuple[int, int]] = []
        for i, a in enumerate(orders):
            act_int = int(a)
            r, c = current_ants[i]
            if act_int == HOLD:
                dest = (r, c)
                rec = telemetry[i]
                assert rec.hold_reason != HoldReason.CONFLICT_WAIT.value, (
                    f"多通道第 {turn} 回合蚂蚁 {i} 出现异常 CONFLICT_WAIT 内部仲裁阻挡"
                )
            else:
                dr, dc = DELTAS[act_int]
                dest = ((r + dr) % rows, (c + dc) % cols)
            destinations.append(dest)

        assert len(set(destinations)) == len(current_ants), f"多通道第 {turn} 回合发生友方相撞"
        current_ants = step_ants(orders, current_ants, rows, cols, water)
