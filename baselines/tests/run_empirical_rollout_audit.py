from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from tb_baselines.env import Env, orders_from_indices
from tb_baselines.planes import remember
from tb_baselines.teacher import CompoundTeacher, known
from tb_baselines.telemetry import (
    FAR,
    HOLD,
    INVALID_GRADIENT,
    HoldReason,
    RejectionCode,
    SourceType,
    TelemetryCollector,
    TelemetryRecord,
)

ROOT = Path(__file__).resolve().parents[2]
BENCH_MAPS = ROOT / "benchmarks" / "maps_2p"

MAP_CONFIGS = [
    ("tiny-open-2p-1h.json", 100, 90000),
    ("tiny-corridor-2p-1h.json", 100, 90001),
    ("tiny-colonnade-2p-1h.json", 100, 90002),
    ("small-arena-2p-1h.json", 80, 90003),
    ("small-islands-2p-1h.json", 80, 90004),
    ("small-rooms-2p-1h.json", 80, 90000),
    ("medium-crossroads-2p-1h.json", 60, 90001),
    ("medium-labyrinth-2p-1h.json", 60, 90002),
    ("medium-valley-2p-1h.json", 60, 90003),
]


def run_audit() -> dict:
    ct = CompoundTeacher(min_raid_ants=3, raid_ratio=0.5, max_depth=32, hill_lead=6)

    total_ant_decisions = 0
    total_seat_turns = 0
    total_turns_executed = 0
    action_match_count = 0
    action_mismatch_count = 0

    hold_counts_by_reason: dict[str, int] = {}
    rejection_code_counts: dict[str, int] = {}
    intentionality_violations = 0
    pseudo_zero_gradient_violations = 0
    record_completeness_violations = 0
    collector_summary_mismatches = 0

    map_summaries = {}

    for map_file, max_turns, seed in MAP_CONFIGS:
        map_path = str((BENCH_MAPS / map_file).resolve())
        remembered: dict[tuple[int, int], dict] = {}
        map_ant_decisions = 0
        map_holds: dict[str, int] = {}

        with Env(
            waves=1,
            matches_per_wave=1,
            max_turns=max_turns,
            seed=seed,
            maps=map_path,
        ) as env:
            step = env.reset()
            for turn in range(max_turns):
                if not step.seats:
                    break
                total_turns_executed += 1
                step_actions = []

                for s in step.seats:
                    total_seat_turns += 1
                    obs = s.obs
                    water, seen = known(obs)
                    key = (s.ep, s.seat)
                    state = remembered.get(key)
                    mine = obs["mine"]
                    num_ants = len(mine)

                    # 1. 开启与关闭遥测动作一致性对比
                    orders_no_t = ct.orders(obs, water, seen, memory=state, collect_telemetry=False)
                    orders_t, records = ct.orders_with_telemetry(obs, water, seen, memory=state)

                    if np.array_equal(orders_no_t, orders_t):
                        action_match_count += num_ants
                    else:
                        action_mismatch_count += num_ants

                    # 2. 遥测记录数量与类型校验
                    if len(records) != num_ants:
                        record_completeness_violations += 1

                    collector = TelemetryCollector()
                    for idx, rec in enumerate(records):
                        total_ant_decisions += 1
                        map_ant_decisions += 1
                        collector.record_ant_decision(
                            ant_id=rec.ant_id,
                            pos=rec.pos,
                            potential_self=rec.potential_self,
                            potentials_neighbors=rec.potentials_neighbors,
                            frontier_size=rec.frontier_size,
                            frontier_nonempty_at_limit=rec.frontier_nonempty_at_limit,
                            nearest_source_type=rec.nearest_source_type,
                            candidate_rejections=rec.candidate_rejections,
                            hold_reason=rec.hold_reason,
                            is_intentional_hold=rec.is_intentional_hold,
                            valid_gradient_mask=rec.valid_gradient_mask,
                            potential_gradient=rec.potential_gradient,
                        )

                        for code in rec.candidate_rejections.values():
                            rejection_code_counts[code] = rejection_code_counts.get(code, 0) + 1

                        # 梯度有效性校验
                        for k in range(4):
                            if not rec.valid_gradient_mask[k]:
                                if rec.potential_gradient[k] == 0:
                                    pseudo_zero_gradient_violations += 1

                        # 意图驻留断言校验
                        if rec.hold_reason is not None:
                            hold_counts_by_reason[rec.hold_reason] = (
                                hold_counts_by_reason.get(rec.hold_reason, 0) + 1
                            )
                            map_holds[rec.hold_reason] = map_holds.get(rec.hold_reason, 0) + 1

                            is_intentional_reason = rec.hold_reason in (
                                HoldReason.HARVEST_WAIT.value,
                                HoldReason.DEFENSE.value,
                            )
                            if rec.is_intentional_hold != is_intentional_reason:
                                intentionality_violations += 1
                        else:
                            if rec.is_intentional_hold:
                                intentionality_violations += 1

                    # 3. 校验 TelemetryCollector 汇总一致性
                    summary = collector.compute_turn_summary()
                    if summary["total_ants"] != num_ants:
                        collector_summary_mismatches += 1
                    turn_holds = sum(1 for r in records if r.hold_reason is not None)
                    if summary["hold_count"] != turn_holds:
                        collector_summary_mismatches += 1

                    order_str = orders_from_indices(orders_no_t, [num_ants])[0]
                    step_actions.append(order_str)
                    remembered[key] = remember(state, obs, mode="where")

                step = env.step(step_actions)

        map_summaries[map_file] = {
            "ant_decisions": map_ant_decisions,
            "hold_reasons": map_holds,
        }

    return {
        "total_maps_tested": len(MAP_CONFIGS),
        "total_turns_executed": total_turns_executed,
        "total_seat_turns": total_seat_turns,
        "total_ant_decisions": total_ant_decisions,
        "action_match_count": action_match_count,
        "action_mismatch_count": action_mismatch_count,
        "hold_counts_by_reason": hold_counts_by_reason,
        "rejection_code_counts": rejection_code_counts,
        "intentionality_violations": intentionality_violations,
        "pseudo_zero_gradient_violations": pseudo_zero_gradient_violations,
        "record_completeness_violations": record_completeness_violations,
        "collector_summary_mismatches": collector_summary_mismatches,
        "map_summaries": map_summaries,
    }


if __name__ == "__main__":
    result = run_audit()
    print(json.dumps(result, indent=2))
