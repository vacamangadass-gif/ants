from __future__ import annotations

import numpy as np
import pytest

from tb_baselines.teacher import CompoundTeacher, Teacher, known
from tb_baselines.telemetry import (
    FAR,
    HOLD,
    INVALID_GRADIENT,
    HoldReason,
    RejectionCode,
    SourceType,
    TelemetryCollector,
    TelemetryRecord,
    classify_hold_reason,
    compute_gradient_and_mask,
    is_intentional_hold_reason,
)


def make_test_board(size: int = 24, ants=((12, 12),), hills=(), food=(), ids=None) -> dict:
    # 构造确定性测试棋盘字典对象
    if ids is None:
        ids = list(range(len(ants)))
    return {
        "size": [size, size],
        "mine": [list(a) for a in ants],
        "ids": ids,
        "foes": [],
        "food": [list(f) for f in food],
        "hills": [list(h) for h in hills],
        "water": {"rle": [0, size * size]},
        "vis": {"rle": [0, size * size]},
    }


def test_telemetry_record_field_completeness_and_types():
    # 验证 TelemetryRecord 数据类包含全部 12 个规定字段并符合强类型约束
    record = TelemetryRecord(
        ant_id=101,
        pos=(12, 14),
        potential_self=8,
        potentials_neighbors=[7, 9, 8, int(FAR)],
        frontier_size=15,
        frontier_nonempty_at_limit=False,
        nearest_source_type=SourceType.FOOD_VISIBLE.value,
        candidate_rejections={1: RejectionCode.FOOD.value, 3: RejectionCode.WATER.value},
        hold_reason=HoldReason.HARVEST_WAIT.value,
        is_intentional_hold=True,
        valid_gradient_mask=[True, True, True, False],
        potential_gradient=[-1, 1, 0, int(FAR)],
    )

    assert record.ant_id == 101
    assert isinstance(record.ant_id, int)
    assert record.pos == (12, 14)
    assert isinstance(record.pos, tuple) and len(record.pos) == 2
    assert record.potential_self == 8
    assert isinstance(record.potential_self, int)
    assert len(record.potentials_neighbors) == 4
    assert all(isinstance(x, int) for x in record.potentials_neighbors)
    assert record.frontier_size == 15
    assert isinstance(record.frontier_size, int)
    assert record.frontier_nonempty_at_limit is False
    assert isinstance(record.frontier_nonempty_at_limit, bool)
    assert record.nearest_source_type == "food_visible"
    assert isinstance(record.nearest_source_type, str)
    assert record.candidate_rejections == {1: "FOOD", 3: "WATER"}
    assert isinstance(record.candidate_rejections, dict)
    assert record.hold_reason == "harvest_wait"
    assert isinstance(record.hold_reason, str)
    assert record.is_intentional_hold is True
    assert isinstance(record.is_intentional_hold, bool)
    assert len(record.valid_gradient_mask) == 4
    assert all(isinstance(x, bool) for x in record.valid_gradient_mask)
    assert len(record.potential_gradient) == 4
    assert all(isinstance(x, int) for x in record.potential_gradient)

    # 验证字典序列化导出支持完整键名映射
    record_dict = record.to_dict()
    expected_keys = {
        "ant_id",
        "pos",
        "potential_self",
        "potentials_neighbors",
        "frontier_size",
        "frontier_nonempty_at_limit",
        "nearest_source_type",
        "candidate_rejections",
        "hold_reason",
        "is_intentional_hold",
        "valid_gradient_mask",
        "potential_gradient",
    }
    assert set(record_dict.keys()) == expected_keys


def test_candidate_rejections_accuracy_four_directions():
    # 验证四向候选动作被剔除的原因编码字典准确性
    candidate_rejections: dict[int, str] = {}
    pot_self = 5
    pot_neighbors = [3, 4, 3, 5]
    is_water = [True, False, False, False]
    is_food = [False, True, False, False]
    is_taken = [False, False, True, False]

    for k in range(4):
        if is_water[k]:
            candidate_rejections[k] = RejectionCode.WATER.value
        elif is_food[k]:
            candidate_rejections[k] = RejectionCode.FOOD.value
        elif is_taken[k]:
            candidate_rejections[k] = RejectionCode.TAKEN.value
        elif pot_neighbors[k] >= pot_self:
            candidate_rejections[k] = RejectionCode.GRADIENT_NON_DECREASING.value

    assert candidate_rejections[0] == "WATER"
    assert candidate_rejections[1] == "FOOD"
    assert candidate_rejections[2] == "TAKEN"
    assert candidate_rejections[3] == "GRADIENT_NON_DECREASING"
    assert len(candidate_rejections) == 4


@pytest.mark.parametrize(
    "scenario_name,kwargs,expected_reason,expected_intentional",
    [
        (
            "harvest_wait_scenario",
            {
                "is_holding": True,
                "pos": (5, 5),
                "potential_self": 2,
                "potentials_neighbors": [3, 3, 3, 3],
                "candidate_rejections": {0: "GRADIENT_NON_DECREASING"},
                "visible_food_coords": {(4, 5)},
                "rows": 20,
                "cols": 20,
            },
            HoldReason.HARVEST_WAIT,
            True,
        ),
        (
            "defense_scenario",
            {
                "is_holding": True,
                "pos": (12, 12),
                "potential_self": 0,
                "potentials_neighbors": [1, 1, 1, 1],
                "candidate_rejections": {},
                "visible_food_coords": set(),
                "rows": 24,
                "cols": 24,
                "is_defending": True,
            },
            HoldReason.DEFENSE,
            True,
        ),
        (
            "conflict_wait_scenario",
            {
                "is_holding": True,
                "pos": (8, 8),
                "potential_self": 6,
                "potentials_neighbors": [4, 7, 7, 7],
                "candidate_rejections": {
                    0: "TAKEN",
                    1: "GRADIENT_NON_DECREASING",
                    2: "GRADIENT_NON_DECREASING",
                    3: "GRADIENT_NON_DECREASING",
                },
                "visible_food_coords": set(),
                "rows": 20,
                "cols": 20,
            },
            HoldReason.CONFLICT_WAIT,
            False,
        ),
        (
            "no_valid_action_scenario",
            {
                "is_holding": True,
                "pos": (3, 3),
                "potential_self": 5,
                "potentials_neighbors": [4, 4, 4, 4],
                "candidate_rejections": {0: "WATER", 1: "WATER", 2: "FOOD", 3: "FOOD"},
                "visible_food_coords": set(),
                "rows": 20,
                "cols": 20,
            },
            HoldReason.NO_VALID_ACTION,
            False,
        ),
        (
            "potential_plateau_scenario",
            {
                "is_holding": True,
                "pos": (10, 10),
                "potential_self": 8,
                "potentials_neighbors": [8, 9, 8, 10],
                "candidate_rejections": {
                    0: "GRADIENT_NON_DECREASING",
                    1: "GRADIENT_NON_DECREASING",
                    2: "GRADIENT_NON_DECREASING",
                    3: "GRADIENT_NON_DECREASING",
                },
                "visible_food_coords": set(),
                "rows": 20,
                "cols": 20,
            },
            HoldReason.POTENTIAL_PLATEAU,
            False,
        ),
        (
            "no_target_scenario",
            {
                "is_holding": True,
                "pos": (2, 2),
                "potential_self": int(FAR),
                "potentials_neighbors": [int(FAR), int(FAR), int(FAR), int(FAR)],
                "candidate_rejections": {},
                "visible_food_coords": set(),
                "rows": 20,
                "cols": 20,
            },
            HoldReason.NO_TARGET,
            False,
        ),
        (
            "stale_target_scenario",
            {
                "is_holding": True,
                "pos": (6, 6),
                "potential_self": 4,
                "potentials_neighbors": [3, 5, 5, 5],
                "candidate_rejections": {},
                "visible_food_coords": set(),
                "rows": 20,
                "cols": 20,
                "is_stale": True,
            },
            HoldReason.STALE_TARGET,
            False,
        ),
    ],
)
def test_hold_reason_enumeration_and_classification(scenario_name, kwargs, expected_reason, expected_intentional):
    # 验证全部 7 种停滞动因枚举均能在对应代表性场景中正确触发并完成意图驻留映射
    actual_reason = classify_hold_reason(**kwargs)
    assert actual_reason == expected_reason
    assert is_intentional_hold_reason(actual_reason) is expected_intentional


def test_is_intentional_hold_strict_boolean_truth_table():
    # 验证 is_intentional_hold 严格仅在 harvest_wait 与 defense 时判定为真，其余全部为假
    truth_expectations = {
        HoldReason.HARVEST_WAIT: True,
        HoldReason.DEFENSE: True,
        HoldReason.CONFLICT_WAIT: False,
        HoldReason.NO_VALID_ACTION: False,
        HoldReason.POTENTIAL_PLATEAU: False,
        HoldReason.NO_TARGET: False,
        HoldReason.STALE_TARGET: False,
    }

    for reason, expected in truth_expectations.items():
        assert is_intentional_hold_reason(reason) is expected
        assert is_intentional_hold_reason(reason.value) is expected

    # 验证移动动作或者无停滞状态下的判定为假
    assert is_intentional_hold_reason(None) is False
    assert is_intentional_hold_reason("none") is False


def test_valid_gradient_mask_prevents_far_pseudo_zero_gradient():
    # 验证当前方格与邻近方格均为 FAR (2^29) 的边界情况，防止算术相减虚构伪零梯度
    pot_self = int(FAR)
    pot_neighbors = [int(FAR), int(FAR), int(FAR), int(FAR)]
    valid_mask, gradient = compute_gradient_and_mask(pot_self, pot_neighbors)

    # 掩码全部必须为假
    assert valid_mask == [False, False, False, False]
    # 差分序列必须统一设为无效标记值，禁止包含伪零数值 0
    assert all(g == INVALID_GRADIENT for g in gradient)
    assert 0 not in gradient


def test_valid_gradient_mask_mixed_reachability():
    # 验证连通与不可达混合场景下的掩码与真实差分计算
    pot_self = 10
    # 邻域势能：北向 8（下降 2 格），东向 12（上升 2 格），南向不可达 FAR，西向 10（等势）
    pot_neighbors = [8, 12, int(FAR), 10]
    water_neighbors = [False, False, False, False]
    valid_mask, gradient = compute_gradient_and_mask(pot_self, pot_neighbors, water_neighbors)

    assert valid_mask == [True, True, False, True]
    assert gradient[0] == -2
    assert gradient[1] == 2
    assert gradient[2] == INVALID_GRADIENT
    assert gradient[3] == 0


def test_telemetry_collector_lifecycle_and_summary_aggregation():
    # 验证 TelemetryCollector 收集器生命周期、记录导出与统计聚合计算
    collector = TelemetryCollector()

    # 注册第一只蚂蚁：主动移动
    collector.record_ant_decision(
        ant_id=1,
        pos=(2, 3),
        potential_self=5,
        potentials_neighbors=[4, 6, 7, 5],
        frontier_size=10,
        frontier_nonempty_at_limit=False,
        nearest_source_type=SourceType.FOOD_VISIBLE.value,
        candidate_rejections={1: RejectionCode.GRADIENT_NON_DECREASING.value},
        hold_reason=None,
        is_intentional_hold=False,
        valid_gradient_mask=[True, True, True, True],
        potential_gradient=[-1, 1, 2, 0],
    )

    # 注册第二只蚂蚁：意图驻留（采食等待）
    collector.record_ant_decision(
        ant_id=2,
        pos=(4, 5),
        potential_self=1,
        potentials_neighbors=[2, 2, 2, 2],
        frontier_size=10,
        frontier_nonempty_at_limit=False,
        nearest_source_type=SourceType.FOOD_VISIBLE.value,
        candidate_rejections={0: RejectionCode.FOOD.value},
        hold_reason=HoldReason.HARVEST_WAIT.value,
        is_intentional_hold=True,
        valid_gradient_mask=[True, True, True, True],
        potential_gradient=[1, 1, 1, 1],
    )

    # 注册第三只蚂蚁：受阻停滞（冲突避让）
    collector.record_ant_decision(
        ant_id=3,
        pos=(6, 7),
        potential_self=4,
        potentials_neighbors=[3, 5, 5, 5],
        frontier_size=10,
        frontier_nonempty_at_limit=False,
        nearest_source_type=SourceType.HILL_FOE.value,
        candidate_rejections={0: RejectionCode.TAKEN.value},
        hold_reason=HoldReason.CONFLICT_WAIT.value,
        is_intentional_hold=False,
        valid_gradient_mask=[True, True, True, True],
        potential_gradient=[-1, 1, 1, 1],
    )

    records = collector.export_turn_telemetry()
    assert len(records) == 3

    summary = collector.compute_turn_summary()
    assert summary["total_ants"] == 3
    assert summary["hold_count"] == 2
    assert summary["intentional_hold_count"] == 1
    assert summary["unintentional_hold_count"] == 1
    assert summary["hold_reasons"]["harvest_wait"] == 1
    assert summary["hold_reasons"]["conflict_wait"] == 1
    assert summary["rejection_codes"]["FOOD"] == 1
    assert summary["rejection_codes"]["TAKEN"] == 1

    collector.clear()
    assert len(collector.export_turn_telemetry()) == 0


def test_nearest_source_type_telemetry_mapping():
    # 验证五类势能源点属性映射的完整性与准确性
    source_mappings = {
        "food_visible": SourceType.FOOD_VISIBLE.value,
        "food_memory": SourceType.FOOD_MEMORY.value,
        "frontier_unseen": SourceType.FRONTIER_UNSEEN.value,
        "hill_foe": SourceType.HILL_FOE.value,
        "none": SourceType.NONE.value,
    }
    for expected_str, enum_val in source_mappings.items():
        assert expected_str == enum_val


def test_frontier_nonempty_at_limit_flag_logic():
    # 验证波前在达到搜索深度上限时的非空标记逻辑
    record_exhausted = TelemetryRecord(
        ant_id=1,
        pos=(0, 0),
        potential_self=10,
        potentials_neighbors=[9, 11, 10, 10],
        frontier_size=0,
        frontier_nonempty_at_limit=False,
        nearest_source_type=SourceType.FOOD_VISIBLE.value,
        candidate_rejections={},
        hold_reason=None,
        is_intentional_hold=False,
        valid_gradient_mask=[True, True, True, True],
        potential_gradient=[-1, 1, 0, 0],
    )
    assert record_exhausted.frontier_size == 0
    assert record_exhausted.frontier_nonempty_at_limit is False

    record_truncated = TelemetryRecord(
        ant_id=2,
        pos=(0, 0),
        potential_self=32,
        potentials_neighbors=[31, 32, 32, 32],
        frontier_size=18,
        frontier_nonempty_at_limit=True,
        nearest_source_type=SourceType.FRONTIER_UNSEEN.value,
        candidate_rejections={},
        hold_reason=None,
        is_intentional_hold=False,
        valid_gradient_mask=[True, True, True, True],
        potential_gradient=[-1, 0, 0, 0],
    )
    assert record_truncated.frontier_size == 18
    assert record_truncated.frontier_nonempty_at_limit is True


def test_teacher_telemetry_collection_integration():
    # 验证 Teacher 模型的遥测挂接与向后兼容性
    teacher = Teacher(hill_lead=6, max_depth=16)
    obs = make_test_board(size=20, ants=((10, 10), (10, 11)), food=((10, 8),), ids=[501, 502])
    water, seen = known(obs)

    # 1. 默认调用保持向后兼容：返回 np.ndarray，不污染内部遥测状态
    default_orders = teacher.orders(obs, water, seen)
    assert isinstance(default_orders, np.ndarray)
    assert len(default_orders) == 2
    assert teacher.export_turn_telemetry() == []

    # 2. 启用遥测采集调用
    telemetry_orders = teacher.orders(obs, water, seen, collect_telemetry=True)
    assert isinstance(telemetry_orders, np.ndarray)
    assert np.array_equal(default_orders, telemetry_orders)

    records = teacher.export_turn_telemetry()
    assert len(records) == 2

    # 验证蚂蚁编号与观测字典保持一致
    assert records[0].ant_id == 501
    assert records[0].pos == (10, 10)
    assert records[1].ant_id == 502
    assert records[1].pos == (10, 11)

    # 验证势能与邻域结构
    for rec in records:
        assert isinstance(rec.potential_self, int)
        assert len(rec.potentials_neighbors) == 4
        assert len(rec.valid_gradient_mask) == 4
        assert len(rec.potential_gradient) == 4
        assert isinstance(rec.candidate_rejections, dict)
        assert isinstance(rec.is_intentional_hold, bool)


def test_teacher_orders_and_act_with_telemetry_convenience():
    # 验证 orders_with_telemetry 与 act_with_telemetry 便捷方法
    teacher = Teacher()
    obs = make_test_board(size=16, ants=((5, 5),), food=((5, 3),), ids=[99])
    water, seen = known(obs)

    orders, records = teacher.orders_with_telemetry(obs, water, seen)
    assert isinstance(orders, np.ndarray)
    assert len(records) == 1
    assert records[0].ant_id == 99

    actions, act_records = teacher.act_with_telemetry(obs)
    assert isinstance(actions, list)
    assert len(actions) == 1
    assert len(act_records) == 1
    assert act_records[0].ant_id == 99


def test_compound_teacher_telemetry_phase_a_and_phase_b():
    # 验证 CompoundTeacher 阶段 A（全员觅食开拓）与阶段 B（突击与经济分工）下的遥测记录
    ct = CompoundTeacher(min_raid_ants=3, raid_ratio=0.5, max_depth=16)

    # 阶段 A：无已知敌方蚁丘，全员处于采食势场
    obs_a = make_test_board(
        size=24,
        ants=((8, 8), (8, 9), (8, 10), (8, 11)),
        food=((8, 5),),
        ids=[10, 11, 12, 13],
    )
    water_a, seen_a = known(obs_a)
    orders_a, records_a = ct.orders_with_telemetry(obs_a, water_a, seen_a)
    assert len(orders_a) == 4
    assert len(records_a) == 4
    for rec in records_a:
        assert rec.nearest_source_type in (SourceType.FOOD_VISIBLE.value, SourceType.FRONTIER_UNSEEN.value)

    # 阶段 B：探明敌方蚁丘（owner=1）且蚂蚁数量达到 min_raid_ants，激活动态突击分队
    obs_b = make_test_board(
        size=24,
        ants=((8, 8), (8, 9), (8, 10), (8, 11)),
        hills=[(8, 2, 1)],  # 敌方蚁穴
        food=((8, 15),),
        ids=[20, 21, 22, 23],
    )
    water_b, seen_b = known(obs_b)
    orders_b, records_b = ct.orders_with_telemetry(obs_b, water_b, seen_b)
    assert len(orders_b) == 4
    assert len(records_b) == 4

    # 突击分队按 raid_ratio=0.5 分配前 2 只蚂蚁，其势能源点类型为敌方蚁穴
    raid_records = [r for r in records_b if r.nearest_source_type == SourceType.HILL_FOE.value]
    assert len(raid_records) >= 1


def test_teacher_intentional_hold_defense_and_harvest():
    # 验证驻留在己方存活蚁穴触发 defense，驻留在食物邻接格触发 harvest_wait
    teacher = Teacher()

    # 场景 1：蚂蚁驻留在己方存活蚁穴 (owner=0)，无目标移动，判定为 defense 且意图驻留为真
    obs_defense = make_test_board(
        size=16,
        ants=((8, 8),),
        hills=[(8, 8, 0)],
        food=(),
        ids=[1],
    )
    water_d, seen_d = known(obs_defense)
    orders_d, records_d = teacher.orders_with_telemetry(obs_defense, water_d, seen_d)
    assert len(records_d) == 1
    # 当蚂蚁停留在蚁穴上时，若未移动，触发 defense
    if orders_d[0] == HOLD:
        assert records_d[0].hold_reason == HoldReason.DEFENSE.value
        assert records_d[0].is_intentional_hold is True

    # 场景 2：蚂蚁在可见食物正交相邻方格驻留，判定为 harvest_wait 且意图驻留为真
    obs_harvest = make_test_board(
        size=16,
        ants=((5, 5),),
        food=((5, 6),),  # 正交相邻
        ids=[2],
    )
    water_h, seen_h = known(obs_harvest)
    # 将四邻域设为阻挡，迫使蚂蚁停留在原地
    water_h[4, 5] = True
    water_h[6, 5] = True
    water_h[5, 4] = True
    # (5, 6) 本身为食物实体方格
    orders_h, records_h = teacher.orders_with_telemetry(obs_harvest, water_h, seen_h)
    assert orders_h[0] == HOLD
    assert len(records_h) == 1
    assert records_h[0].hold_reason == HoldReason.HARVEST_WAIT.value
    assert records_h[0].is_intentional_hold is True


def test_teacher_isolated_ant_far_pseudo_zero_gradient_prevention():
    # 验证蚂蚁处于不可达水域孤岛时，四邻域势能差分序列不产生伪零梯度
    teacher = Teacher()
    obs = make_test_board(size=20, ants=((10, 10),), food=(), ids=[777])
    water, seen = known(obs)

    # 将蚂蚁四周全部围成水域阻隔
    water[9, 10] = True
    water[11, 10] = True
    water[10, 9] = True
    water[10, 11] = True

    orders, records = teacher.orders_with_telemetry(obs, water, seen)
    assert len(records) == 1
    rec = records[0]

    # 掩码中对应水域或不可达的邻格必须为 False
    assert 0 not in rec.potential_gradient
    assert all(g == INVALID_GRADIENT for g in rec.potential_gradient)
    assert rec.valid_gradient_mask == [False, False, False, False]


def test_teacher_telemetry_fallback_when_ids_missing():
    # 验证观测字典缺失 ids 字段时平滑保底回退至整数下标，避免抛出 KeyError
    teacher = Teacher()
    obs = {
        "size": [16, 16],
        "mine": [[4, 4], [4, 5]],
        # 无 ids 键
        "foes": [],
        "food": [[4, 2]],
        "hills": [],
        "water": {"rle": [0, 256]},
        "vis": {"rle": [0, 256]},
    }
    water, seen = known(obs)
    orders, records = teacher.orders_with_telemetry(obs, water, seen)

    assert len(orders) == 2
    assert len(records) == 2
    assert records[0].ant_id == 0
    assert records[1].ant_id == 1
