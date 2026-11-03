from __future__ import annotations

from pathlib import Path
import pytest

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
    TelemetryRecord,
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


def test_starvation_elimination_k2_k4_direct_contention():
    for k in (2, 4):
        mine = [(14, 15), (15, 16), (16, 15), (15, 14)][:k]
        cands = [[2, HOLD], [3, HOLD], [0, HOLD], [1, HOLD]][:k]
        pots = [10] * k

        # 1. 验证 fair_mode=False 下编号较高蚂蚁发生严重饥饿停滞
        stalls_unfair = {i: 0 for i in range(k)}
        wins_unfair = {i: 0 for i in range(k)}
        max_stall_unfair = {i: 0 for i in range(k)}
        cur_stall_unfair = {i: 0 for i in range(k)}

        for _ in range(100):
            orders, _ = arbitrate_orders(
                mine_coords=mine,
                candidate_sequences=cands,
                potentials=pots,
                stall_counters=stalls_unfair,
                fair_mode=False,
                rows=30,
                cols=30,
            )
            for i in range(k):
                if orders[i] != HOLD:
                    wins_unfair[i] += 1
                    cur_stall_unfair[i] = 0
                else:
                    cur_stall_unfair[i] += 1
                    if cur_stall_unfair[i] > max_stall_unfair[i]:
                        max_stall_unfair[i] = cur_stall_unfair[i]

        assert wins_unfair[0] == 100
        for i in range(1, k):
            assert wins_unfair[i] == 0
            assert max_stall_unfair[i] == 100

        # 2. 验证 fair_mode=True 下最大连续停滞严格受限于 K - 1
        stalls_fair = {i: 0 for i in range(k)}
        wins_fair = {i: 0 for i in range(k)}
        max_stall_fair = {i: 0 for i in range(k)}
        cur_stall_fair = {i: 0 for i in range(k)}

        for _ in range(100):
            orders, _ = arbitrate_orders(
                mine_coords=mine,
                candidate_sequences=cands,
                potentials=pots,
                stall_counters=stalls_fair,
                fair_mode=True,
                rows=30,
                cols=30,
            )
            for i in range(k):
                if orders[i] != HOLD:
                    wins_fair[i] += 1
                    cur_stall_fair[i] = 0
                else:
                    cur_stall_fair[i] += 1
                    if cur_stall_fair[i] > max_stall_fair[i]:
                        max_stall_fair[i] = cur_stall_fair[i]

        expected_wins_per_ant = 100 // k
        for i in range(k):
            assert wins_fair[i] == expected_wins_per_ant
            assert max_stall_fair[i] <= k - 1


def test_starvation_elimination_k8_multi_lane_continuous_flow():
    # 模拟 8 只蚂蚁分布于 4 条行进队列竞争同一中心瓶颈格子的 100 回合推演
    turns = 100
    lane_queues_unfair = {0: [0, 4], 1: [1, 5], 2: [2, 6], 3: [3, 7]}
    stalls_unfair = {i: 0 for i in range(8)}
    wins_unfair = {i: 0 for i in range(8)}
    max_stall_unfair = {i: 0 for i in range(8)}
    cur_stall_unfair = {i: 0 for i in range(8)}

    for _ in range(turns):
        front_ants = [lane_queues_unfair[lane][0] for lane in range(4)]
        cands = [[2, HOLD], [3, HOLD], [0, HOLD], [1, HOLD]]
        pots = [5, 5, 5, 5]
        mine = [(9, 10), (10, 11), (11, 10), (10, 9)]
        sub_stalls = {aid: stalls_unfair[aid] for aid in front_ants}

        orders, updated = arbitrate_orders(
            mine_coords=mine,
            candidate_sequences=cands,
            potentials=pots,
            stall_counters=sub_stalls,
            fair_mode=False,
            ids=front_ants,
            rows=30,
            cols=30,
        )
        for aid in front_ants:
            stalls_unfair[aid] = updated[aid]

        for idx, aid in enumerate(front_ants):
            if orders[idx] != HOLD:
                wins_unfair[aid] += 1
                cur_stall_unfair[aid] = 0
                lane = aid % 4
                lane_queues_unfair[lane].pop(0)
                lane_queues_unfair[lane].append(aid)
            else:
                cur_stall_unfair[aid] += 1
                if cur_stall_unfair[aid] > max_stall_unfair[aid]:
                    max_stall_unfair[aid] = cur_stall_unfair[aid]

        for lane in range(4):
            back_ant = lane_queues_unfair[lane][1]
            cur_stall_unfair[back_ant] += 1
            if cur_stall_unfair[back_ant] > max_stall_unfair[back_ant]:
                max_stall_unfair[back_ant] = cur_stall_unfair[back_ant]

    # fair_mode=False 下第 0 队列严重垄断，其他队列蚂蚁停滞接近 100 回合
    assert wins_unfair[0] + wins_unfair[4] >= 90
    assert max(max_stall_unfair.values()) >= 95

    # fair_mode=True 下 8 只蚂蚁执行严格轮转，最大连续停滞严格受限于 8 - 1 = 7
    lane_queues_fair = {0: [0, 4], 1: [1, 5], 2: [2, 6], 3: [3, 7]}
    stalls_fair = {i: 0 for i in range(8)}
    wins_fair = {i: 0 for i in range(8)}
    max_stall_fair = {i: 0 for i in range(8)}
    cur_stall_fair = {i: 0 for i in range(8)}

    for _ in range(turns):
        front_ants = [lane_queues_fair[lane][0] for lane in range(4)]
        cands = [[2, HOLD], [3, HOLD], [0, HOLD], [1, HOLD]]
        pots = [5, 5, 5, 5]
        mine = [(9, 10), (10, 11), (11, 10), (10, 9)]
        sub_stalls = {aid: stalls_fair[aid] for aid in front_ants}

        orders, updated = arbitrate_orders(
            mine_coords=mine,
            candidate_sequences=cands,
            potentials=pots,
            stall_counters=sub_stalls,
            fair_mode=True,
            ids=front_ants,
            rows=30,
            cols=30,
        )
        for aid in front_ants:
            stalls_fair[aid] = updated[aid]

        for idx, aid in enumerate(front_ants):
            if orders[idx] != HOLD:
                wins_fair[aid] += 1
                cur_stall_fair[aid] = 0
                lane = aid % 4
                lane_queues_fair[lane].pop(0)
                lane_queues_fair[lane].append(aid)
            else:
                cur_stall_fair[aid] += 1
                if cur_stall_fair[aid] > max_stall_fair[aid]:
                    max_stall_fair[aid] = cur_stall_fair[aid]

        for lane in range(4):
            back_ant = lane_queues_fair[lane][1]
            cur_stall_fair[back_ant] += 1
            if cur_stall_fair[back_ant] > max_stall_fair[back_ant]:
                max_stall_fair[back_ant] = cur_stall_fair[back_ant]

        for aid in range(8):
            assert max_stall_fair[aid] <= 7

    assert max(max_stall_fair.values()) == 7
    for aid in range(8):
        assert 12 <= wins_fair[aid] <= 13


def test_dynamic_priority_overcomes_initial_potential_disparity():
    # 验证受阻等待计数能够克服初始势能微弱劣势，杜绝远近差异导致的永久饥饿
    mine = [(9, 10), (10, 9)]
    cands = [[2, HOLD], [1, HOLD]]
    # 蚂蚁 0 势能为 2，蚂蚁 1 势能为 3
    pots = [2, 3]

    # 第 1 回合：初始计数均为 0，蚂蚁 0 凭借更低势能胜出，蚂蚁 1 受阻自增
    stalls = {0: 0, 1: 0}
    orders_t1, stalls_t1 = arbitrate_orders(
        mine_coords=mine,
        candidate_sequences=cands,
        potentials=pots,
        stall_counters=stalls,
        fair_mode=True,
        rows=20,
        cols=20,
    )
    assert orders_t1[0] == 2
    assert orders_t1[1] == HOLD
    assert stalls_t1[0] == 0
    assert stalls_t1[1] == 1

    # 第 2 回合：蚂蚁 1 受阻计数为 1，优先级元组 (-1, 3, 1) 压制蚂蚁 0 的 (-0, 2, 0)
    orders_t2, stalls_t2 = arbitrate_orders(
        mine_coords=mine,
        candidate_sequences=cands,
        potentials=pots,
        stall_counters=stalls_t1,
        fair_mode=True,
        rows=20,
        cols=20,
    )
    assert orders_t2[0] == HOLD
    assert orders_t2[1] == 1
    assert stalls_t2[0] == 1
    assert stalls_t2[1] == 0


@pytest.mark.parametrize("map_file,max_turns,seed", MAP_CONFIGS)
def test_full_game_rollout_benchmark_maps_zero_collisions(map_file, max_turns, seed):
    map_path = str((BENCH_MAPS / map_file).resolve())
    ct = CompoundTeacher(min_raid_ants=3, raid_ratio=0.5, max_depth=32, hill_lead=6)

    remembered: dict[tuple[int, int], dict] = {}
    seat_stalls: dict[int, dict[int, int]] = {0: {}, 1: {}}

    total_decisions = 0
    total_internal_collisions = 0
    sovereign_hold_violations = 0
    conflict_wait_count = 0
    taken_rejection_count = 0

    with Env(
        waves=1,
        matches_per_wave=1,
        max_turns=max_turns,
        seed=seed,
        maps=map_path,
    ) as env:
        step = env.reset()
        for _ in range(max_turns):
            if not step.seats:
                break
            step_actions = []

            for s in step.seats:
                obs = s.obs
                water, seen = known(obs)
                key = (s.ep, s.seat)
                state = remembered.get(key)
                mine = obs["mine"]
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

                # 1. 验证本方全部蚂蚁目标坐标无重复
                dests = []
                for idx, act in enumerate(orders):
                    r, c = mine[idx]
                    if act == HOLD:
                        dests.append((int(r), int(c)))
                    else:
                        dr, dc = DELTAS[int(act)]
                        dests.append(((int(r) + dr) % rows, (int(c) + dc) % cols))

                if len(dests) != len(set(dests)):
                    total_internal_collisions += 1

                # 2. 验证原地驻留主权完整保留
                for idx, act in enumerate(orders):
                    if act == HOLD:
                        stay_pos = (int(mine[idx][0]), int(mine[idx][1]))
                        intruders = [
                            j for j, d in enumerate(dests)
                            if j != idx and d == stay_pos
                        ]
                        if intruders:
                            sovereign_hold_violations += 1

                # 3. 验证遥测探针归因准确性
                for idx, rec in enumerate(records):
                    total_decisions += 1
                    assert isinstance(rec, TelemetryRecord)

                    if rec.hold_reason == HoldReason.CONFLICT_WAIT.value:
                        conflict_wait_count += 1
                        assert rec.is_intentional_hold is False
                        assert orders[idx] == HOLD
                        has_taken = any(
                            c == RejectionCode.TAKEN.value
                            for c in rec.candidate_rejections.values()
                        )
                        assert has_taken, f"CONFLICT_WAIT 记录缺少 TAKEN 剔除编码: {rec.candidate_rejections}"

                    for d_idx, code in rec.candidate_rejections.items():
                        if code == RejectionCode.TAKEN.value:
                            taken_rejection_count += 1
                            assert d_idx in (0, 1, 2, 3)

                order_str = orders_from_indices(orders, [len(mine)])[0]
                step_actions.append(order_str)
                remembered[key] = remember(state, obs, mode="where")

            step = env.step(step_actions)

            # 4. 验证引擎步进后存活蚂蚁坐标绝对无重合
            for s in step.seats:
                post_coords = [(int(a[0]), int(a[1])) for a in s.obs["mine"]]
                assert len(post_coords) == len(set(post_coords))

    assert total_internal_collisions == 0
    assert sovereign_hold_violations == 0
    assert total_decisions > 0
