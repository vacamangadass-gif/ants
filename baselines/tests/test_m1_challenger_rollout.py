from __future__ import annotations

import os
import shutil
from pathlib import Path

import numpy as np
import pytest

from tb_baselines.env import Env, orders_from_indices
from tb_baselines.planes import remember
from tb_baselines.teacher import CompoundTeacher, Teacher, known
from tb_baselines.telemetry import (
    FAR,
    HOLD,
    INVALID_GRADIENT,
    FieldMetadata,
    HoldReason,
    RejectionCode,
    SourceType,
    TelemetryCollector,
    TelemetryRecord,
    classify_hold_reason,
    compute_gradient_and_mask,
    evaluate_ant_telemetry,
    is_intentional_hold_reason,
)

ROOT = Path(__file__).resolve().parents[2]
BENCH_MAPS = ROOT / "benchmarks" / "maps_2p"

pytestmark = pytest.mark.skipif(
    not (os.environ.get("TINYBRAINS") or shutil.which("tinybrains")),
    reason="环境缺少 tinybrains CLI 二进制文件",
)

VALID_HOLD_REASONS = {
    HoldReason.HARVEST_WAIT.value,
    HoldReason.DEFENSE.value,
    HoldReason.CONFLICT_WAIT.value,
    HoldReason.NO_VALID_ACTION.value,
    HoldReason.POTENTIAL_PLATEAU.value,
    HoldReason.NO_TARGET.value,
    HoldReason.STALE_TARGET.value,
}

VALID_SOURCE_TYPES = {
    SourceType.FOOD_VISIBLE.value,
    SourceType.FOOD_MEMORY.value,
    SourceType.FRONTIER_UNSEEN.value,
    SourceType.HILL_FOE.value,
    SourceType.NONE.value,
}

VALID_REJECTION_CODES = {
    RejectionCode.WATER.value,
    RejectionCode.FOOD.value,
    RejectionCode.TAKEN.value,
    RejectionCode.GRADIENT_NON_DECREASING.value,
}


def _validate_telemetry_records(
    obs: dict,
    orders_no_telem: np.ndarray,
    records: list[TelemetryRecord],
):
    # 验证单回合遥测记录对所有蚂蚁的完备性与契约一致性
    mine = obs["mine"]
    assert len(records) == len(mine), f"遥测记录数量 ({len(records)}) 与蚂蚁数量 ({len(mine)}) 不匹配"
    ant_ids = obs.get("ids")

    for i, rec in enumerate(records):
        assert isinstance(rec, TelemetryRecord)
        expected_id = ant_ids[i] if (ant_ids is not None and i < len(ant_ids)) else i
        assert rec.ant_id == expected_id
        assert rec.pos == (mine[i][0], mine[i][1])
        assert isinstance(rec.potential_self, int)
        assert isinstance(rec.potentials_neighbors, list) and len(rec.potentials_neighbors) == 4
        assert all(isinstance(p, int) for p in rec.potentials_neighbors)
        assert isinstance(rec.frontier_size, int) and rec.frontier_size >= 0
        assert isinstance(rec.frontier_nonempty_at_limit, bool)
        assert rec.nearest_source_type in VALID_SOURCE_TYPES
        assert isinstance(rec.candidate_rejections, dict)
        for k_dir, code in rec.candidate_rejections.items():
            assert k_dir in (0, 1, 2, 3)
            assert code in VALID_REJECTION_CODES

        assert isinstance(rec.valid_gradient_mask, list) and len(rec.valid_gradient_mask) == 4
        assert all(isinstance(v, bool) for v in rec.valid_gradient_mask)
        assert isinstance(rec.potential_gradient, list) and len(rec.potential_gradient) == 4
        assert all(isinstance(g, int) for g in rec.potential_gradient)

        # 验证梯度差分：有效时为差值，无效时为 INVALID_GRADIENT 且绝对不为 0
        for k in range(4):
            if rec.valid_gradient_mask[k]:
                assert rec.potential_gradient[k] == rec.potentials_neighbors[k] - rec.potential_self
            else:
                assert rec.potential_gradient[k] == INVALID_GRADIENT
                assert rec.potential_gradient[k] != 0

        # 核心契约断言：is_intentional_hold 为 True 当且仅当 hold_reason 为 harvest_wait 或 defense
        action = orders_no_telem[i]
        if action != HOLD:
            assert rec.hold_reason is None, f"移动蚂蚁动作非 HOLD 但存在 hold_reason: {rec.hold_reason}"
            assert rec.is_intentional_hold is False, "移动蚂蚁的 is_intentional_hold 必须为 False"
        else:
            assert rec.hold_reason in VALID_HOLD_REASONS, f"未知驻留动因: {rec.hold_reason}"
            if rec.hold_reason in (HoldReason.HARVEST_WAIT.value, HoldReason.DEFENSE.value):
                assert rec.is_intentional_hold is True, (
                    f"动因为 {rec.hold_reason} 时 is_intentional_hold 必须严格为 True"
                )
            else:
                assert rec.is_intentional_hold is False, (
                    f"非意图动因 {rec.hold_reason} 的 is_intentional_hold 必须严格为 False"
                )

        # 验证序列化导出支持完整 12 个键名
        rec_dict = rec.to_dict()
        assert len(rec_dict) == 12


@pytest.mark.parametrize(
    "map_filename,max_turns,seed",
    [
        ("tiny-open-2p-1h.json", 60, 42),
        ("tiny-corridor-2p-1h.json", 60, 42),
        ("tiny-colonnade-2p-1h.json", 60, 42),
        ("small-arena-2p-1h.json", 40, 101),
        ("medium-labyrinth-2p-1h.json", 30, 77),
    ],
)
def test_dynamic_rollout_compound_teacher_telemetry(map_filename, max_turns, seed):
    # 在真实基准地图上运行多回合动态推演，全面检验 CompoundTeacher 遥测探针
    map_path = str((BENCH_MAPS / map_filename).resolve())
    ct = CompoundTeacher(min_raid_ants=3, raid_ratio=0.5, max_depth=32, hill_lead=6)

    total_ant_decisions = 0
    hold_counts_by_reason: dict[str, int] = {}
    intentional_hold_count = 0
    unintentional_hold_count = 0

    remembered: dict[tuple[int, int], dict] = {}

    with Env(
        waves=1,
        matches_per_wave=1,
        max_turns=max_turns,
        seed=seed,
        maps=map_path,
    ) as env:
        step = env.reset()
        for turn_idx in range(max_turns):
            if not step.seats:
                break

            step_actions = []
            for s in step.seats:
                obs = s.obs
                water, seen = known(obs)
                key = (s.ep, s.seat)
                state = remembered.get(key)

                # 1. 对比遥测开启与关闭下的决策完全一致性
                orders_no_telem = ct.orders(obs, water, seen, memory=state, collect_telemetry=False)
                orders_telem = ct.orders(obs, water, seen, memory=state, collect_telemetry=True)
                np.testing.assert_array_equal(
                    orders_no_telem,
                    orders_telem,
                    err_msg=f"回合 {turn_idx} 座位 {s.seat} 开启/关闭遥测产生的动作不一致",
                )

                orders_convenience, records_convenience = ct.orders_with_telemetry(
                    obs, water, seen, memory=state
                )
                np.testing.assert_array_equal(
                    orders_no_telem,
                    orders_convenience,
                    err_msg=f"回合 {turn_idx} orders_with_telemetry 动作不一致",
                )

                act_no_t = ct.act(obs, memory=state, collect_telemetry=False)
                act_t = ct.act(obs, memory=state, collect_telemetry=True)
                assert act_no_t == act_t, f"回合 {turn_idx} act 动作不一致"

                act_convenience, records_act = ct.act_with_telemetry(obs, memory=state)
                assert act_no_t == act_convenience
                assert len(records_act) == len(obs["mine"])

                # 2. 验证导出的遥测记录完整性
                records = ct.export_turn_telemetry()
                assert len(records) == len(records_convenience)

                _validate_telemetry_records(obs, orders_no_telem, records)

                # 3. 统计指标累计
                for rec in records:
                    total_ant_decisions += 1
                    if rec.hold_reason is not None:
                        hold_counts_by_reason[rec.hold_reason] = (
                            hold_counts_by_reason.get(rec.hold_reason, 0) + 1
                        )
                        if rec.is_intentional_hold:
                            intentional_hold_count += 1
                        else:
                            unintentional_hold_count += 1

                # 4. 验证 TelemetryCollector 汇总准确性
                collector = TelemetryCollector()
                for rec in records:
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
                summary = collector.compute_turn_summary()
                assert summary["total_ants"] == len(records)
                turn_holds = sum(1 for r in records if r.hold_reason is not None)
                assert summary["hold_count"] == turn_holds
                turn_intentional = sum(1 for r in records if r.is_intentional_hold)
                assert summary["intentional_hold_count"] == turn_intentional
                assert summary["unintentional_hold_count"] == turn_holds - turn_intentional

                # 动作打包并推进环境
                order_str = orders_from_indices(orders_no_telem, [len(obs["mine"])])[0]
                step_actions.append(order_str)

                # 更新记忆
                remembered[key] = remember(state, obs, mode="where")

            step = env.step(step_actions)

    # 确保推演过程中有真实的蚂蚁行动
    assert total_ant_decisions > 100, f"推演蚂蚁决策样本过少: {total_ant_decisions}"


def test_rollout_without_memory_equivalence():
    # 验证在无记忆模式下 CompoundTeacher 遥测与动作的严格等价性
    map_path = str((BENCH_MAPS / "tiny-open-2p-1h.json").resolve())
    ct = CompoundTeacher(min_raid_ants=3, raid_ratio=0.5)

    with Env(waves=1, matches_per_wave=1, max_turns=30, seed=77, maps=map_path) as env:
        step = env.reset()
        for _ in range(30):
            if not step.seats:
                break
            step_actions = []
            for s in step.seats:
                obs = s.obs
                act_no_t = ct.act(obs, memory=None, collect_telemetry=False)
                act_t = ct.act(obs, memory=None, collect_telemetry=True)
                assert act_no_t == act_t

                act_conv, records = ct.act_with_telemetry(obs, memory=None)
                assert act_no_t == act_conv
                assert len(records) == len(obs["mine"])

                for r in records:
                    if r.is_intentional_hold:
                        assert r.hold_reason in (HoldReason.HARVEST_WAIT.value, HoldReason.DEFENSE.value)
                    else:
                        assert r.hold_reason not in (HoldReason.HARVEST_WAIT.value, HoldReason.DEFENSE.value)

                order_str = "".join("NESW-"[a] for a in act_no_t)
                step_actions.append(order_str)
            step = env.step(step_actions)


def test_adversarial_hold_reasons_and_intentional_hold_truth_table():
    # 对抗性全动因检验：构造对应所有 7 种动因与移动场景，断言 is_intentional_hold 严格互斥
    teacher = Teacher()

    # 1. HARVEST_WAIT: 蚂蚁邻接可见食物，且四周受阻驻留 -> intentional=True
    obs_harvest = {
        "size": [16, 16],
        "mine": [[5, 5]],
        "ids": [10],
        "foes": [],
        "food": [[5, 6]],
        "hills": [],
        "water": {"rle": [0, 256]},
        "vis": {"rle": [0, 256]},
    }
    water_h, seen_h = known(obs_harvest)
    water_h[4, 5] = True
    water_h[6, 5] = True
    water_h[5, 4] = True
    orders, records = teacher.orders_with_telemetry(obs_harvest, water_h, seen_h)
    assert orders[0] == HOLD
    assert records[0].hold_reason == HoldReason.HARVEST_WAIT.value
    assert records[0].is_intentional_hold is True

    # 2. DEFENSE: 蚂蚁位于己方蚁穴上且四周受阻驻留 -> intentional=True
    obs_defense = {
        "size": [16, 16],
        "mine": [[8, 8]],
        "ids": [20],
        "foes": [],
        "food": [],
        "hills": [[8, 8, 0]],
        "water": {"rle": [0, 256]},
        "vis": {"rle": [0, 256]},
    }
    water_d, seen_d = known(obs_defense)
    water_d[7, 8] = True
    water_d[9, 8] = True
    water_d[8, 7] = True
    water_d[8, 9] = True
    orders, records = teacher.orders_with_telemetry(obs_defense, water_d, seen_d)
    assert orders[0] == HOLD
    assert records[0].hold_reason == HoldReason.DEFENSE.value
    assert records[0].is_intentional_hold is True

    # 3. CONFLICT_WAIT: 存在更优梯度但目标方格被队友占用 -> intentional=False
    obs_conflict = {
        "size": [16, 16],
        "mine": [[5, 5], [5, 6]],
        "ids": [30, 31],
        "foes": [],
        "food": [[5, 8]],
        "hills": [],
        "water": {"rle": [0, 256]},
        "vis": {"rle": [0, 256]},
    }
    water_c, seen_c = known(obs_conflict)
    orders, records = teacher.orders_with_telemetry(obs_conflict, water_c, seen_c)
    conflict_records = [r for r in records if r.hold_reason == HoldReason.CONFLICT_WAIT.value]
    for r in conflict_records:
        assert r.is_intentional_hold is False

    # 4. NO_VALID_ACTION: 四周全为障碍（3水域 + 1等势能占位），无任何合法移动 -> intentional=False
    reason_no_valid = classify_hold_reason(
        is_holding=True,
        pos=(5, 5),
        potential_self=10,
        potentials_neighbors=[10, 10, 10, 10],
        candidate_rejections={
            0: RejectionCode.WATER.value,
            1: RejectionCode.WATER.value,
            2: RejectionCode.WATER.value,
            3: RejectionCode.TAKEN.value,
        },
        visible_food_coords=set(),
        rows=16,
        cols=16,
        is_defending=False,
        is_stale=False,
    )
    assert reason_no_valid == HoldReason.NO_VALID_ACTION
    assert is_intentional_hold_reason(reason_no_valid) is False

    # 5. NO_TARGET: 己方与四周均处于不可达 FAR 且无任何目标 -> intentional=False
    reason_no_target = classify_hold_reason(
        is_holding=True,
        pos=(5, 5),
        potential_self=int(FAR),
        potentials_neighbors=[int(FAR), int(FAR), int(FAR), int(FAR)],
        candidate_rejections={},
        visible_food_coords=set(),
        rows=16,
        cols=16,
        is_defending=False,
        is_stale=False,
    )
    assert reason_no_target == HoldReason.NO_TARGET
    assert is_intentional_hold_reason(reason_no_target) is False

    # 6. STALE_TARGET: 目标存在于地图某处，但当前蚂蚁处于不可达陆地 -> intentional=False
    obs_stale = {
        "size": [16, 16],
        "mine": [[2, 2]],
        "ids": [60],
        "foes": [],
        "food": [[12, 12]],
        "hills": [],
        "water": {"rle": [0, 256]},
        "vis": {"rle": [0, 256]},
    }
    water_s, seen_s = known(obs_stale)
    water_s[1, 2] = True
    water_s[3, 2] = True
    water_s[2, 1] = True
    water_s[2, 3] = True
    orders, records = teacher.orders_with_telemetry(obs_stale, water_s, seen_s)
    assert orders[0] == HOLD
    assert records[0].hold_reason == HoldReason.STALE_TARGET.value
    assert records[0].is_intentional_hold is False

    # 7. POTENTIAL_PLATEAU: 势能有效，但所有邻域势能均不低于自身（平坦势能面）-> intentional=False
    plateau_reason = classify_hold_reason(
        is_holding=True,
        pos=(5, 5),
        potential_self=10,
        potentials_neighbors=[10, 11, 10, 12],
        candidate_rejections={
            0: RejectionCode.GRADIENT_NON_DECREASING.value,
            1: RejectionCode.GRADIENT_NON_DECREASING.value,
            2: RejectionCode.GRADIENT_NON_DECREASING.value,
            3: RejectionCode.GRADIENT_NON_DECREASING.value,
        },
        visible_food_coords=set(),
        rows=16,
        cols=16,
        is_defending=False,
        is_stale=False,
    )
    assert plateau_reason == HoldReason.POTENTIAL_PLATEAU
    assert is_intentional_hold_reason(plateau_reason) is False

    # 8. 移动动作: action in (0, 1, 2, 3) 绝不标记为驻留
    ct_meta = FieldMetadata(0, False, np.zeros((16, 16), dtype=np.int32))
    for act in (0, 1, 2, 3):
        rec_move = evaluate_ant_telemetry(
            ant_id=1,
            pos=(5, 5),
            dist=np.zeros((16, 16), dtype=np.int32),
            field_meta=ct_meta,
            candidate_rejections={},
            action=act,
            visible_food=set(),
            friendly_hills=set(),
            is_stale=False,
        )
        assert rec_move.hold_reason is None
        assert rec_move.is_intentional_hold is False


def test_adversarial_gradient_mask_pseudo_zero_elimination():
    # 对抗性差分检验：不可达区域相减绝不允许出现 0 差分（防伪零梯度）
    pot_self = int(FAR)
    pot_neighbors = [int(FAR), int(FAR), int(FAR), int(FAR)]
    mask, grad = compute_gradient_and_mask(pot_self, pot_neighbors)
    assert mask == [False, False, False, False]
    assert 0 not in grad
    assert all(g == INVALID_GRADIENT for g in grad)

    # 自身有效但部分邻居不可达
    pot_self_2 = 5
    pot_neighbors_2 = [4, int(FAR), 6, int(FAR)]
    mask_2, grad_2 = compute_gradient_and_mask(pot_self_2, pot_neighbors_2)
    assert mask_2 == [True, False, True, False]
    assert grad_2[0] == -1
    assert grad_2[1] == INVALID_GRADIENT
    assert grad_2[2] == 1
    assert grad_2[3] == INVALID_GRADIENT
    assert grad_2[1] != 0
    assert grad_2[3] != 0


def test_stress_boundary_and_scale():
    # 极端边界与规模压力测试：空兵力、大规模群落、缺失 ids
    ct = CompoundTeacher(min_raid_ants=3, raid_ratio=0.5)

    # 1. 空兵力
    obs_empty = {
        "size": [16, 16],
        "mine": [],
        "foes": [],
        "food": [[5, 5]],
        "hills": [],
        "water": {"rle": [0, 256]},
        "vis": {"rle": [0, 256]},
    }
    water_e, seen_e = known(obs_empty)
    orders_e, records_e = ct.orders_with_telemetry(obs_empty, water_e, seen_e)
    assert len(orders_e) == 0
    assert len(records_e) == 0
    assert ct.export_turn_telemetry() == []
    act_e, records_act_e = ct.act_with_telemetry(obs_empty)
    assert len(act_e) == 0
    assert len(records_act_e) == 0

    # 2. 64 只密集蚂蚁群落
    ants_64 = [[r, c] for r in range(4, 12) for c in range(4, 12)]
    obs_dense = {
        "size": [20, 20],
        "mine": ants_64,
        "ids": list(range(100, 164)),
        "foes": [],
        "food": [[0, 0], [19, 19]],
        "hills": [],
        "water": {"rle": [0, 400]},
        "vis": {"rle": [0, 400]},
    }
    water_d, seen_d = known(obs_dense)
    orders_dense_no_t = ct.orders(obs_dense, water_d, seen_d, collect_telemetry=False)
    orders_dense_t, records_dense = ct.orders_with_telemetry(obs_dense, water_d, seen_d)
    np.testing.assert_array_equal(orders_dense_no_t, orders_dense_t)
    assert len(records_dense) == 64
    for idx, r in enumerate(records_dense):
        assert r.ant_id == 100 + idx
        assert r.pos == tuple(ants_64[idx])
        if r.hold_reason in (HoldReason.HARVEST_WAIT.value, HoldReason.DEFENSE.value):
            assert r.is_intentional_hold is True
        else:
            assert r.is_intentional_hold is False

    # 3. 缺失 ids 字段时整数下标回退
    obs_no_ids = {
        "size": [16, 16],
        "mine": [[2, 2], [3, 3]],
        "foes": [],
        "food": [[5, 5]],
        "hills": [],
        "water": {"rle": [0, 256]},
        "vis": {"rle": [0, 256]},
    }
    water_ni, seen_ni = known(obs_no_ids)
    orders_ni, records_ni = ct.orders_with_telemetry(obs_no_ids, water_ni, seen_ni)
    assert len(records_ni) == 2
    assert records_ni[0].ant_id == 0
    assert records_ni[1].ant_id == 1

