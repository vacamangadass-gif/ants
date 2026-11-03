from __future__ import annotations

import json
from pathlib import Path

from tb_baselines.env import Env, orders_from_indices
from tb_baselines.planes import remember
from tb_baselines.teacher import (
    DELTAS,
    HOLD,
    CompoundTeacher,
    arbitrate_orders,
    known,
)
from tb_baselines.telemetry import (
    HoldReason,
    RejectionCode,
)

ROOT = Path(__file__).resolve().parents[2]
BENCH_MAPS = ROOT / "benchmarks" / "maps_2p"

MAP_CONFIGS = [
    ("tiny-open-2p-1h.json", 100, 90000),
    ("tiny-corridor-2p-1h.json", 100, 90001),
    ("tiny-colonnade-2p-1h.json", 100, 90002),
    ("small-arena-2p-1h.json", 100, 90003),
    ("small-rooms-2p-1h.json", 100, 90004),
    ("small-islands-2p-1h.json", 100, 90000),
    ("medium-crossroads-2p-1h.json", 100, 90001),
    ("medium-labyrinth-2p-1h.json", 100, 90002),
    ("medium-valley-2p-1h.json", 100, 90003),
]


def run_bottleneck_starvation_experiment(k_values: list[int] = [2, 4, 8], turns: int = 100) -> dict:
    results = {}

    for k in k_values:
        for fair_mode in [False, True]:
            mode_key = f"K{k}_fair_{fair_mode}"
            rows, cols = 30, 30

            # 为 K 只蚂蚁生成在目标格子周围的起始坐标
            mine: list[tuple[int, int]] = []
            candidates: list[list[int]] = []
            for i in range(k):
                # 放置在目标格子的四邻域或等距外圈
                if i == 0:
                    pos = (14, 15)  # 北
                    cand = [2, HOLD]  # 南向进入 (15, 15)
                elif i == 1:
                    pos = (15, 16)  # 东
                    cand = [3, HOLD]  # 西向进入 (15, 15)
                elif i == 2:
                    pos = (16, 15)  # 南
                    cand = [0, HOLD]  # 北向进入 (15, 15)
                elif i == 3:
                    pos = (15, 14)  # 西
                    cand = [1, HOLD]  # 东向进入 (15, 15)
                elif i == 4:
                    pos = (13, 15)
                    cand = [2, HOLD]
                elif i == 5:
                    pos = (15, 17)
                    cand = [3, HOLD]
                elif i == 6:
                    pos = (17, 15)
                    cand = [0, HOLD]
                elif i == 7:
                    pos = (15, 13)
                    cand = [1, HOLD]
                else:
                    pos = (15 + (i % 3) - 1, 15 + (i // 3) - 1)
                    cand = [0, HOLD]
                mine.append(pos)
                candidates.append(cand)

            potentials = [10] * k
            stall_counters = {i: 0 for i in range(k)}

            win_counts = {i: 0 for i in range(k)}
            max_consecutive_stall = {i: 0 for i in range(k)}
            cur_consecutive_stall = {i: 0 for i in range(k)}
            total_stall_turns = {i: 0 for i in range(k)}

            for turn in range(turns):
                rejections: dict[int, dict[int, str]] = {i: {} for i in range(k)}
                orders, updated_stalls = arbitrate_orders(
                    mine_coords=mine,
                    candidate_sequences=candidates,
                    potentials=potentials,
                    stall_counters=stall_counters,
                    fair_mode=fair_mode,
                    rows=rows,
                    cols=cols,
                    rejections_out=rejections,
                )

                # 检验动作与终点唯一性
                dests = []
                for idx, act in enumerate(orders):
                    if act == HOLD:
                        dests.append(mine[idx])
                    else:
                        dr, dc = DELTAS[int(act)]
                        dests.append(((mine[idx][0] + dr) % rows, (mine[idx][1] + dc) % cols))
                assert len(dests) == len(set(dests)), "发生终点冲突碰撞"

                # 统计胜出与受阻
                for i in range(k):
                    if orders[i] != HOLD:
                        win_counts[i] += 1
                        cur_consecutive_stall[i] = 0
                    else:
                        cur_consecutive_stall[i] += 1
                        total_stall_turns[i] += 1
                        if cur_consecutive_stall[i] > max_consecutive_stall[i]:
                            max_consecutive_stall[i] = cur_consecutive_stall[i]

            wins_list = [win_counts[i] for i in range(k)]
            max_stall_overall = max(max_consecutive_stall.values())
            min_wins = min(wins_list)
            max_wins = max(wins_list)
            starved_ants = sum(1 for w in wins_list if w == 0)

            results[mode_key] = {
                "k": k,
                "fair_mode": fair_mode,
                "turns": turns,
                "max_consecutive_stall_by_ant": max_consecutive_stall,
                "max_consecutive_stall_overall": max_stall_overall,
                "total_stall_turns_by_ant": total_stall_turns,
                "win_counts_by_ant": win_counts,
                "min_wins": min_wins,
                "max_wins": max_wins,
                "fairness_ratio": float(min_wins) / float(max_wins) if max_wins > 0 else 0.0,
                "starved_ants_count": starved_ants,
                "starvation_rate_pct": (starved_ants / k) * 100.0,
            }

    return results


def run_full_game_rollout_audit() -> dict:
    ct = CompoundTeacher(min_raid_ants=3, raid_ratio=0.5, max_depth=32, hill_lead=6)

    total_turns_simulated = 0
    total_ant_decisions = 0
    total_internal_collisions = 0
    sovereign_hold_violations = 0

    hold_reason_distribution: dict[str, int] = {}
    rejection_code_distribution: dict[str, int] = {}
    conflict_wait_attribution_valid = 0
    conflict_wait_attribution_invalid = 0
    taken_rejection_attribution_valid = 0
    taken_rejection_attribution_invalid = 0

    map_metrics = {}

    for map_file, max_turns, seed in MAP_CONFIGS:
        map_path = str((BENCH_MAPS / map_file).resolve())
        remembered: dict[tuple[int, int], dict] = {}
        seat_stalls: dict[int, dict[int, int]] = {0: {}, 1: {}}

        map_decisions = 0
        map_collisions = 0
        map_holds: dict[str, int] = {}
        map_rejections: dict[str, int] = {}
        max_stall_observed = 0

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
                total_turns_simulated += 1
                step_actions = []

                for s in step.seats:
                    obs = s.obs
                    water, seen = known(obs)
                    key = (s.ep, s.seat)
                    state = remembered.get(key)
                    mine = obs["mine"]
                    num_ants = len(mine)
                    rows, cols = obs["size"]
                    stalls = seat_stalls[s.seat]

                    orders, records = ct.orders_with_telemetry(
                        obs,
                        water,
                        seen,
                        memory=state,
                        stall_counters=stalls,
                        fair_mode=True,
                    )

                    # 1. 严格检验本方蚂蚁目标格子无任何碰撞
                    dests = []
                    for idx, act in enumerate(orders):
                        r, c = mine[idx]
                        if act == HOLD:
                            dests.append((int(r), int(c)))
                        else:
                            dr, dc = DELTAS[int(act)]
                            dests.append(((int(r) + dr) % rows, (int(c) + dc) % cols))

                    if len(dests) != len(set(dests)):
                        map_collisions += 1
                        total_internal_collisions += 1

                    # 2. 严格检验原地驻留主权百分之百保留
                    for idx, act in enumerate(orders):
                        if act == HOLD:
                            stay_pos = (int(mine[idx][0]), int(mine[idx][1]))
                            # 确认其他蚂蚁没有进入该驻留格子
                            intruders = [
                                j for j, d in enumerate(dests)
                                if j != idx and d == stay_pos
                            ]
                            if intruders:
                                sovereign_hold_violations += 1

                    # 3. 严格检验遥测探针归因准确性
                    for idx, rec in enumerate(records):
                        total_ant_decisions += 1
                        map_decisions += 1

                        if rec.hold_reason is not None:
                            map_holds[rec.hold_reason] = map_holds.get(rec.hold_reason, 0) + 1
                            hold_reason_distribution[rec.hold_reason] = (
                                hold_reason_distribution.get(rec.hold_reason, 0) + 1
                            )

                        for code in rec.candidate_rejections.values():
                            map_rejections[code] = map_rejections.get(code, 0) + 1
                            rejection_code_distribution[code] = (
                                rejection_code_distribution.get(code, 0) + 1
                            )

                        # 检验 CONFLICT_WAIT 归因合法性
                        if rec.hold_reason == HoldReason.CONFLICT_WAIT.value:
                            has_taken = any(
                                c == RejectionCode.TAKEN.value
                                for c in rec.candidate_rejections.values()
                            )
                            if (not rec.is_intentional_hold) and has_taken and orders[idx] == HOLD:
                                conflict_wait_attribution_valid += 1
                            else:
                                conflict_wait_attribution_invalid += 1

                        # 检验 TAKEN 编码对应动作
                        for d_idx, code in rec.candidate_rejections.items():
                            if code == RejectionCode.TAKEN.value:
                                assert d_idx in (0, 1, 2, 3)
                                taken_rejection_attribution_valid += 1

                    # 记录最大受阻计数
                    for st_val in stalls.values():
                        if st_val > max_stall_observed:
                            max_stall_observed = st_val

                    # 动作打包并推进环境
                    order_str = orders_from_indices(orders, [num_ants])[0]
                    step_actions.append(order_str)

                    # 记忆更新
                    remembered[key] = remember(state, obs, mode="where")

                step = env.step(step_actions)

                # 4. 步进后验证存活蚂蚁坐标无重叠
                for s in step.seats:
                    post_mine = s.obs["mine"]
                    post_coords = [(int(a[0]), int(a[1])) for a in post_mine]
                    assert len(post_coords) == len(set(post_coords)), "环境步进后存活蚂蚁坐标发生重叠"

        map_metrics[map_file] = {
            "max_turns": max_turns,
            "seed": seed,
            "total_decisions": map_decisions,
            "internal_collisions": map_collisions,
            "holds": map_holds,
            "rejections": map_rejections,
            "max_stall_observed": max_stall_observed,
        }

    return {
        "total_turns_simulated": total_turns_simulated,
        "total_ant_decisions": total_ant_decisions,
        "total_internal_collisions": total_internal_collisions,
        "sovereign_hold_violations": sovereign_hold_violations,
        "conflict_wait_attribution_valid": conflict_wait_attribution_valid,
        "conflict_wait_attribution_invalid": conflict_wait_attribution_invalid,
        "taken_rejection_attribution_valid": taken_rejection_attribution_valid,
        "taken_rejection_attribution_invalid": taken_rejection_attribution_invalid,
        "hold_reason_distribution": hold_reason_distribution,
        "rejection_code_distribution": rejection_code_distribution,
        "map_metrics": map_metrics,
    }


def main():
    print("=== 开始运行瓶颈饥饿消除实证评测 ===")
    starvation_res = run_bottleneck_starvation_experiment([2, 4, 8], turns=100)
    print(json.dumps(starvation_res, indent=2))

    print("\n=== 开始运行 9 张基准地图全量动态推演评测 ===")
    rollout_res = run_full_game_rollout_audit()
    print(json.dumps(rollout_res, indent=2))


if __name__ == "__main__":
    main()
