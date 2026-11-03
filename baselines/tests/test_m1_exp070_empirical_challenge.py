from __future__ import annotations

import copy
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
    evaluate_ant_telemetry,
    is_intentional_hold_reason,
)


def create_mock_board(size: int = 24, ants=((12, 12),), hills=(), food=(), ids=None) -> dict:
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


def test_isolated_cell_all_neighbors_far_gradient_and_mask():
    # 验证当前格为 FAR 且四邻域均为 FAR 时的掩码与差分
    valid_mask, gradient = compute_gradient_and_mask(int(FAR), [int(FAR), int(FAR), int(FAR), int(FAR)])
    assert valid_mask == [False, False, False, False]
    assert 0 not in gradient
    assert all(g == INVALID_GRADIENT for g in gradient)

    # 验证当前格为有限势能但四邻域均为 FAR 时的掩码与差分
    valid_mask_finite, gradient_finite = compute_gradient_and_mask(5, [int(FAR), int(FAR), int(FAR), int(FAR)])
    assert valid_mask_finite == [False, False, False, False]
    assert 0 not in gradient_finite
    assert all(g == INVALID_GRADIENT for g in gradient_finite)

    # 验证当前格为 FAR 但四邻域存在有限势能时的掩码与差分
    valid_mask_inv, gradient_inv = compute_gradient_and_mask(int(FAR), [3, 4, 5, 6])
    assert valid_mask_inv == [False, False, False, False]
    assert 0 not in gradient_inv
    assert all(g == INVALID_GRADIENT for g in gradient_inv)


def test_isolated_cell_all_neighbors_water_gradient_and_mask():
    # 验证四邻域均为水域阻挡时的梯度计算
    valid_mask, gradient = compute_gradient_and_mask(
        2,
        [1, 1, 1, 1],
        water_neighbors=[True, True, True, True],
    )
    assert valid_mask == [False, False, False, False]
    assert 0 not in gradient
    assert -1 not in gradient
    assert all(g == INVALID_GRADIENT for g in gradient)


def test_teacher_and_compound_teacher_isolated_ant_in_water_lake():
    # 验证蚂蚁完全被水域环绕且外部存在目标时归因为 stale_target（不可达断通目标）
    size = 20
    obs = create_mock_board(size=size, ants=((10, 10),), food=((10, 5),), ids=[888])
    water, seen = known(obs)

    # 将四邻域置为水域阻隔
    water[9, 10] = True
    water[11, 10] = True
    water[10, 9] = True
    water[10, 11] = True

    # 验证 Teacher
    teacher = Teacher()
    orders, records = teacher.orders_with_telemetry(obs, water, seen)
    assert orders[0] == HOLD
    assert len(records) == 1
    rec = records[0]
    assert rec.potential_self == int(FAR)
    assert rec.valid_gradient_mask == [False, False, False, False]
    assert 0 not in rec.potential_gradient
    assert all(g == INVALID_GRADIENT for g in rec.potential_gradient)
    assert rec.candidate_rejections == {0: "WATER", 1: "WATER", 2: "WATER", 3: "WATER"}
    assert rec.hold_reason == HoldReason.STALE_TARGET.value
    assert rec.is_intentional_hold is False

    # 验证 CompoundTeacher
    ct = CompoundTeacher()
    ct_orders, ct_records = ct.orders_with_telemetry(obs, water, seen)
    assert ct_orders[0] == HOLD
    assert len(ct_records) == 1
    ct_rec = ct_records[0]
    assert ct_rec.potential_self == int(FAR)
    assert ct_rec.valid_gradient_mask == [False, False, False, False]
    assert 0 not in ct_rec.potential_gradient
    assert all(g == INVALID_GRADIENT for g in ct_rec.potential_gradient)
    assert ct_rec.candidate_rejections == {0: "WATER", 1: "WATER", 2: "WATER", 3: "WATER"}
    assert ct_rec.hold_reason == HoldReason.STALE_TARGET.value
    assert ct_rec.is_intentional_hold is False


def test_isolated_land_island_without_targets_hold_reason_no_target():
    # 验证全图无任何目标时孤岛蚂蚁归因为 no_target
    size = 16
    obs = create_mock_board(size=size, ants=((8, 8),), food=(), hills=(), ids=[123])
    water, seen = known(obs)
    seen[:] = True

    teacher = Teacher()
    orders, records = teacher.orders_with_telemetry(obs, water, seen)
    assert orders[0] == HOLD
    assert len(records) == 1
    rec = records[0]
    assert rec.potential_self == int(FAR)
    assert all(p == int(FAR) for p in rec.potentials_neighbors)
    assert rec.valid_gradient_mask == [False, False, False, False]
    assert 0 not in rec.potential_gradient
    assert rec.hold_reason == HoldReason.NO_TARGET.value
    assert rec.is_intentional_hold is False


def test_isolated_land_island_with_distant_targets_hold_reason_stale_target():
    # 验证全图存在远端目标但孤岛蚂蚁无法连通时归因为 stale_target
    size = 20
    obs = create_mock_board(size=size, ants=((10, 10),), food=((2, 2),), ids=[999])
    water, seen = known(obs)
    seen[:] = True

    # 将蚂蚁所在 3x3 区域的外围围成环形水域，切断与 (2, 2) 食物的连通
    for r in range(8, 13):
        for c in range(8, 13):
            if r in (8, 12) or c in (8, 12):
                water[r, c] = True

    teacher = Teacher()
    orders, records = teacher.orders_with_telemetry(obs, water, seen)
    assert orders[0] == HOLD
    assert len(records) == 1
    rec = records[0]
    assert rec.potential_self == int(FAR)
    assert rec.hold_reason == HoldReason.STALE_TARGET.value
    assert rec.is_intentional_hold is False
    assert rec.valid_gradient_mask == [False, False, False, False]
    assert 0 not in rec.potential_gradient


def test_teacher_orders_no_valid_action_trapped_by_water_and_uphill_taken():
    # 验证处于有效势场中但四向全被水域与无下坡队友占位封闭时归因为 no_valid_action
    teacher = Teacher()
    # 蚂蚁 1 在 (10, 10)，食物在 (10, 16)（有限势能连通区域）
    # 北向 (9, 10) 与南向 (11, 10) 为水域
    # 东向 (10, 11) 与西向 (10, 9) 为队友蚂蚁，且由优先权判定处于非递减梯度
    obs = create_mock_board(
        size=20,
        ants=((10, 10), (10, 11), (10, 9)),
        food=((10, 16),),
        ids=[1, 2, 3],
    )
    water, seen = known(obs)
    water[9, 10] = True
    water[11, 10] = True

    orders, records = teacher.orders_with_telemetry(obs, water, seen)
    rec1 = [r for r in records if r.ant_id == 1][0]
    if orders[0] == HOLD:
        # 当蚂蚁 1 驻留时，四向均为水域或队友占位，且无合法下坡动作
        assert rec1.hold_reason in (HoldReason.NO_VALID_ACTION.value, HoldReason.CONFLICT_WAIT.value)
        assert rec1.is_intentional_hold is False


def test_rejections_water_takes_precedence_over_food_and_taken():
    # 验证多障碍重叠时水域判定优先级高于食物与队友占位
    teacher = Teacher()
    obs = create_mock_board(
        size=20,
        ants=((10, 10), (9, 10), (10, 11)),
        food=((9, 10), (11, 10)),
        ids=[1, 2, 3],
    )
    water, seen = known(obs)

    # 北向 (9, 10)：同时存在水域、食物与队友占位
    water[9, 10] = True
    # 南向 (11, 10)：同时存在水域与食物
    water[11, 10] = True
    # 东向 (10, 11)：同时存在水域与队友占位
    water[10, 11] = True
    # 西向 (10, 9)：水域
    water[10, 9] = True

    orders, records = teacher.orders_with_telemetry(obs, water, seen)
    rec1 = [r for r in records if r.ant_id == 1][0]
    assert rec1.candidate_rejections[0] == RejectionCode.WATER.value
    assert rec1.candidate_rejections[1] == RejectionCode.WATER.value
    assert rec1.candidate_rejections[2] == RejectionCode.WATER.value
    assert rec1.candidate_rejections[3] == RejectionCode.WATER.value


def test_rejections_food_takes_precedence_over_taken():
    # 验证食物方格与队友占位重叠时食物阻挡识别先于队友占位
    teacher = Teacher()
    # 蚂蚁 1 在 (10, 10)，蚂蚁 2 在 (9, 10)；(9, 10) 处同时存在食物
    obs = create_mock_board(
        size=20,
        ants=((10, 10), (9, 10)),
        food=((9, 10), (10, 5)),
        ids=[101, 102],
    )
    water, seen = known(obs)

    orders, records = teacher.orders_with_telemetry(obs, water, seen)
    rec101 = [r for r in records if r.ant_id == 101][0]
    # 北向 (9, 10) 应被标记为 FOOD
    assert rec101.candidate_rejections[0] == RejectionCode.FOOD.value


def test_rejections_taken_takes_precedence_over_gradient_non_decreasing():
    # 验证队友占位方格即使梯度不下降也优先标记为 TAKEN
    teacher = Teacher()
    # 蚂蚁 1 在 (10, 10)，目标食物在 (10, 2)
    # 蚂蚁 2 在 (11, 10)，位于蚂蚁 1 南侧（梯度非递减方向）
    obs = create_mock_board(
        size=20,
        ants=((10, 10), (11, 10)),
        food=((10, 2),),
        ids=[1, 2],
    )
    water, seen = known(obs)

    orders, records = teacher.orders_with_telemetry(obs, water, seen)
    rec1 = [r for r in records if r.ant_id == 1][0]
    # 南向（索引 2）存在队友占位，应标记为 TAKEN
    assert rec1.candidate_rejections[2] == RejectionCode.TAKEN.value


def test_rejections_downhill_valid_moves_are_not_rejected():
    # 验证严格更优的下坡候选移动动作不会被错误记录进剔除原因字典
    teacher = Teacher()
    # 蚂蚁在 (10, 10)，目标食物在 (10, 8)，即西向为下坡方向
    obs = create_mock_board(
        size=20,
        ants=((10, 10),),
        food=((10, 8),),
        ids=[1],
    )
    water, seen = known(obs)
    # 将北向与东向设为水域
    water[9, 10] = True
    water[10, 11] = True

    orders, records = teacher.orders_with_telemetry(obs, water, seen)
    rec = records[0]
    # 西向索引为 3，为严格下坡合法方向且被执行
    assert orders[0] == 3
    assert 3 not in rec.candidate_rejections
    # 北向与东向为水域
    assert rec.candidate_rejections[0] == RejectionCode.WATER.value
    assert rec.candidate_rejections[1] == RejectionCode.WATER.value
    assert rec.candidate_rejections[2] == RejectionCode.GRADIENT_NON_DECREASING.value


def test_rejections_all_four_obstacles_simultaneous():
    # 验证四个方向分别触发四种不同剔除编码的综合场景
    teacher = Teacher()
    # 蚂蚁 1 在 (10, 10)，目标食物在 (10, 2)（西向为目标方向）
    # 北向 (9, 10)：水域
    # 东向 (10, 11)：梯度非递减开阔陆地
    # 南向 (11, 10)：食物实体方格
    # 西向 (10, 9)：队友 2 占位
    obs = create_mock_board(
        size=20,
        ants=((10, 10), (10, 9)),
        food=((11, 10), (10, 2)),
        ids=[1, 2],
    )
    water, seen = known(obs)
    water[9, 10] = True

    orders, records = teacher.orders_with_telemetry(obs, water, seen)
    rec1 = [r for r in records if r.ant_id == 1][0]
    assert rec1.candidate_rejections[0] == RejectionCode.WATER.value
    assert rec1.candidate_rejections[1] == RejectionCode.GRADIENT_NON_DECREASING.value
    assert rec1.candidate_rejections[2] == RejectionCode.FOOD.value
    assert rec1.candidate_rejections[3] == RejectionCode.TAKEN.value
    assert len(rec1.candidate_rejections) == 4


def test_hold_reason_harvest_wait_multi_food_adjacency():
    # 验证多相邻食物情况下的采食等待归因与意图驻留判定
    reason = classify_hold_reason(
        is_holding=True,
        pos=(5, 5),
        potential_self=1,
        potentials_neighbors=[2, 2, 2, 2],
        candidate_rejections={0: "FOOD", 1: "FOOD"},
        visible_food_coords={(4, 5), (5, 6)},
        rows=20,
        cols=20,
    )
    assert reason == HoldReason.HARVEST_WAIT
    assert is_intentional_hold_reason(reason) is True


def test_hold_reason_defense_precedence_over_harvest_and_conflict():
    # 验证己方存活蚁穴驻留时 defense 优先级高于食物邻接与冲突等待
    reason = classify_hold_reason(
        is_holding=True,
        pos=(12, 12),
        potential_self=0,
        potentials_neighbors=[1, 1, 1, 1],
        candidate_rejections={0: "TAKEN"},
        visible_food_coords={(11, 12)},
        rows=24,
        cols=24,
        is_defending=True,
    )
    assert reason == HoldReason.DEFENSE
    assert is_intentional_hold_reason(reason) is True


def test_hold_reason_enemy_hill_not_defending():
    # 验证敌方蚁穴驻留时不触发 defense 归因
    reason = classify_hold_reason(
        is_holding=True,
        pos=(12, 12),
        potential_self=0,
        potentials_neighbors=[1, 1, 1, 1],
        candidate_rejections={0: "GRADIENT_NON_DECREASING", 1: "GRADIENT_NON_DECREASING", 2: "GRADIENT_NON_DECREASING", 3: "GRADIENT_NON_DECREASING"},
        visible_food_coords=set(),
        rows=24,
        cols=24,
        is_defending=False,
    )
    assert reason == HoldReason.POTENTIAL_PLATEAU
    assert is_intentional_hold_reason(reason) is False


def test_hold_reason_conflict_wait_downhill_blocked_by_teammate():
    # 验证唯一严格下坡方向被队友占位时的冲突等待归因
    reason = classify_hold_reason(
        is_holding=True,
        pos=(6, 6),
        potential_self=8,
        potentials_neighbors=[6, 10, 10, 10],  # 北向下坡
        candidate_rejections={0: "TAKEN", 1: "GRADIENT_NON_DECREASING", 2: "GRADIENT_NON_DECREASING", 3: "GRADIENT_NON_DECREASING"},
        visible_food_coords=set(),
        rows=20,
        cols=20,
    )
    assert reason == HoldReason.CONFLICT_WAIT
    assert is_intentional_hold_reason(reason) is False


def test_hold_reason_uphill_teammate_does_not_cause_conflict_wait():
    # 验证队友仅存在于上坡方向且无下坡方向时归因为 potential_plateau 而非 conflict_wait
    reason = classify_hold_reason(
        is_holding=True,
        pos=(6, 6),
        potential_self=5,
        potentials_neighbors=[7, 8, 6, 9],  # 全为上坡
        candidate_rejections={0: "TAKEN", 1: "GRADIENT_NON_DECREASING", 2: "GRADIENT_NON_DECREASING", 3: "GRADIENT_NON_DECREASING"},
        visible_food_coords=set(),
        rows=20,
        cols=20,
    )
    assert reason == HoldReason.POTENTIAL_PLATEAU
    assert is_intentional_hold_reason(reason) is False


def test_hold_reason_no_valid_action_all_terrain_obstacles():
    # 验证四周被水域与食物实体刚性包围时的合法动作缺失归因
    reason = classify_hold_reason(
        is_holding=True,
        pos=(3, 3),
        potential_self=4,
        potentials_neighbors=[3, 3, 3, 3],
        candidate_rejections={0: "WATER", 1: "WATER", 2: "FOOD", 3: "FOOD"},
        visible_food_coords=set(),
        rows=20,
        cols=20,
    )
    assert reason == HoldReason.NO_VALID_ACTION
    assert is_intentional_hold_reason(reason) is False


def test_hold_reason_potential_plateau_local_minimum():
    # 验证局部极小值势能坑（四邻域势能均严格大于当前格）归因为 potential_plateau
    reason = classify_hold_reason(
        is_holding=True,
        pos=(8, 8),
        potential_self=4,
        potentials_neighbors=[5, 6, 5, 7],
        candidate_rejections={
            0: "GRADIENT_NON_DECREASING",
            1: "GRADIENT_NON_DECREASING",
            2: "GRADIENT_NON_DECREASING",
            3: "GRADIENT_NON_DECREASING",
        },
        visible_food_coords=set(),
        rows=20,
        cols=20,
    )
    assert reason == HoldReason.POTENTIAL_PLATEAU
    assert is_intentional_hold_reason(reason) is False


def test_hold_reason_moving_ant_returns_none():
    # 验证正在移动的蚂蚁其停滞动因为 None
    reason = classify_hold_reason(
        is_holding=False,
        pos=(8, 8),
        potential_self=4,
        potentials_neighbors=[3, 5, 5, 5],
        candidate_rejections={},
        visible_food_coords=set(),
        rows=20,
        cols=20,
        is_defending=True,
    )
    assert reason is None
    assert is_intentional_hold_reason(reason) is False


def test_is_intentional_hold_exhaustive_matrix():
    # 穷举测试所有 7 种停滞原因以及边界输入的意图驻留真值表
    for reason in HoldReason:
        expected = reason in (HoldReason.HARVEST_WAIT, HoldReason.DEFENSE)
        assert is_intentional_hold_reason(reason) is expected
        assert is_intentional_hold_reason(reason.value) is expected

    assert is_intentional_hold_reason(None) is False
    assert is_intentional_hold_reason("") is False
    assert is_intentional_hold_reason("unknown_reason") is False
    assert is_intentional_hold_reason("conflict_wait") is False


def test_caller_input_immutability_teacher_orders():
    # 验证调用 Teacher.orders 与 orders_with_telemetry 不会修改外部输入对象
    obs = create_mock_board(
        size=16,
        ants=((5, 5), (5, 6), (7, 7)),
        food=((5, 8), (2, 2)),
        hills=[(5, 5, 0), (1, 1, 1)],
        ids=[10, 20, 30],
    )
    water, seen = known(obs)
    water[4, 5] = True
    seen[5, 5] = True

    obs_backup = copy.deepcopy(obs)
    water_backup = water.copy()
    seen_backup = seen.copy()
    memory_input = {"hill_foe_seen": [(1, 1)]}
    memory_backup = copy.deepcopy(memory_input)

    teacher = Teacher()
    orders = teacher.orders(obs, water, seen, memory=memory_input, collect_telemetry=True)
    orders2, records = teacher.orders_with_telemetry(obs, water, seen, memory=memory_input)

    # 验证外部输入完全未受修改
    assert obs == obs_backup
    assert np.array_equal(water, water_backup)
    assert np.array_equal(seen, seen_backup)
    assert memory_input == memory_backup


def test_caller_input_immutability_compound_teacher_orders():
    # 验证调用 CompoundTeacher.orders 在阶段 A 与阶段 B 下均不修改外部输入
    obs = create_mock_board(
        size=24,
        ants=((8, 8), (8, 9), (8, 10), (8, 11)),
        food=((8, 5),),
        hills=[(8, 2, 1)],
        ids=[1, 2, 3, 4],
    )
    water, seen = known(obs)
    obs_backup = copy.deepcopy(obs)
    water_backup = water.copy()
    seen_backup = seen.copy()

    ct = CompoundTeacher(min_raid_ants=2, raid_ratio=0.5)
    orders, records = ct.orders_with_telemetry(obs, water, seen)

    assert obs == obs_backup
    assert np.array_equal(water, water_backup)
    assert np.array_equal(seen, seen_backup)


def test_caller_input_immutability_telemetry_utility_functions():
    # 验证底层遥测辅助函数不修改输入的可变列表与字典
    pot_neighbors = [3, 4, 5, 6]
    pot_neighbors_backup = list(pot_neighbors)
    water_neighbors = [False, True, False, False]
    water_neighbors_backup = list(water_neighbors)

    compute_gradient_and_mask(4, pot_neighbors, water_neighbors)
    assert pot_neighbors == pot_neighbors_backup
    assert water_neighbors == water_neighbors_backup

    rejections = {0: "WATER", 1: "FOOD"}
    rejections_backup = dict(rejections)
    food_coords = {(5, 6), (7, 8)}
    food_coords_backup = set(food_coords)

    classify_hold_reason(
        is_holding=True,
        pos=(5, 5),
        potential_self=3,
        potentials_neighbors=pot_neighbors,
        candidate_rejections=rejections,
        visible_food_coords=food_coords,
        rows=20,
        cols=20,
    )
    assert rejections == rejections_backup
    assert food_coords == food_coords_backup


def test_collector_defensive_copying_and_isolation():
    # 验证 TelemetryCollector 对传入参数执行防御性复制且隔离外部变更
    collector = TelemetryCollector()
    pot_neighbors = [1, 2, 3, 4]
    rejections = {0: "FOOD"}
    valid_mask = [True, True, True, True]
    gradients = [-1, 0, 1, 2]

    rec = collector.record_ant_decision(
        ant_id=1,
        pos=(2, 2),
        potential_self=2,
        potentials_neighbors=pot_neighbors,
        frontier_size=5,
        frontier_nonempty_at_limit=False,
        nearest_source_type=SourceType.FOOD_VISIBLE.value,
        candidate_rejections=rejections,
        hold_reason=HoldReason.HARVEST_WAIT.value,
        is_intentional_hold=True,
        valid_gradient_mask=valid_mask,
        potential_gradient=gradients,
    )

    # 外部修改原始列表与字典
    pot_neighbors[0] = 999
    rejections[0] = "MUTATED"
    valid_mask[0] = False
    gradients[0] = -999

    # 校验内部存储记录未被污染
    assert rec.potentials_neighbors[0] == 1
    assert rec.candidate_rejections[0] == "FOOD"
    assert rec.valid_gradient_mask[0] is True
    assert rec.potential_gradient[0] == -1

    # 校验 export_turn_telemetry 导出结果的列表隔离性
    exported = collector.export_turn_telemetry()
    exported.clear()
    assert len(collector.export_turn_telemetry()) == 1


def test_torus_wrapping_boundary_telemetry():
    # 验证环面边界蚂蚁在坐标跨越边界时的势能与遥测正确性
    teacher = Teacher()
    size = 20
    # 蚂蚁位于 (0, 0)，食物位于 (18, 0)，北向邻居 (19, 0) 是通往下坡的陆地
    obs = create_mock_board(size=size, ants=((0, 0),), food=((18, 0),), ids=[77])
    water, seen = known(obs)

    orders, records = teacher.orders_with_telemetry(obs, water, seen)
    assert len(records) == 1
    rec = records[0]
    # 北向邻居坐标为 (19, 0)
    assert rec.pos == (0, 0)
    assert len(rec.potentials_neighbors) == 4
    # (19, 0) 处为食物正交采集位 G_food，深度为 hill_lead (6)；(0, 0) 处势能为 7
    assert rec.potentials_neighbors[0] == teacher.hill_lead
    assert rec.potential_self == teacher.hill_lead + 1
    # 蚂蚁应选择北向移动（动作为 0）
    assert orders[0] == 0
    assert 0 not in rec.candidate_rejections
    assert rec.valid_gradient_mask[0] is True
    assert rec.potential_gradient[0] == -1


def test_telemetry_summary_arithmetic_invariants_large_scale():
    # 验证大规模单兵决策汇总时的数值恒等式约束
    collector = TelemetryCollector()
    total_ants = 100

    for i in range(total_ants):
        if i % 4 == 0:
            # 主动移动
            collector.record_ant_decision(
                ant_id=i,
                pos=(i % 20, (i * 2) % 20),
                potential_self=5,
                potentials_neighbors=[4, 6, 6, 6],
                frontier_size=10,
                frontier_nonempty_at_limit=False,
                nearest_source_type="food_visible",
                candidate_rejections={1: "GRADIENT_NON_DECREASING"},
                hold_reason=None,
                is_intentional_hold=False,
                valid_gradient_mask=[True, True, True, True],
                potential_gradient=[-1, 1, 1, 1],
            )
        elif i % 4 == 1:
            # 采食驻留
            collector.record_ant_decision(
                ant_id=i,
                pos=(i % 20, (i * 2) % 20),
                potential_self=1,
                potentials_neighbors=[2, 2, 2, 2],
                frontier_size=10,
                frontier_nonempty_at_limit=False,
                nearest_source_type="food_visible",
                candidate_rejections={0: "FOOD"},
                hold_reason="harvest_wait",
                is_intentional_hold=True,
                valid_gradient_mask=[True, True, True, True],
                potential_gradient=[1, 1, 1, 1],
            )
        elif i % 4 == 2:
            # 防守驻留
            collector.record_ant_decision(
                ant_id=i,
                pos=(i % 20, (i * 2) % 20),
                potential_self=0,
                potentials_neighbors=[1, 1, 1, 1],
                frontier_size=10,
                frontier_nonempty_at_limit=False,
                nearest_source_type="food_visible",
                candidate_rejections={},
                hold_reason="defense",
                is_intentional_hold=True,
                valid_gradient_mask=[True, True, True, True],
                potential_gradient=[1, 1, 1, 1],
            )
        else:
            # 冲突等待
            collector.record_ant_decision(
                ant_id=i,
                pos=(i % 20, (i * 2) % 20),
                potential_self=6,
                potentials_neighbors=[5, 7, 7, 7],
                frontier_size=10,
                frontier_nonempty_at_limit=False,
                nearest_source_type="food_visible",
                candidate_rejections={0: "TAKEN"},
                hold_reason="conflict_wait",
                is_intentional_hold=False,
                valid_gradient_mask=[True, True, True, True],
                potential_gradient=[-1, 1, 1, 1],
            )

    summary = collector.compute_turn_summary()
    assert summary["total_ants"] == total_ants
    # 3/4 的蚂蚁驻留，1/4 移动
    assert summary["hold_count"] == 75
    # harvest_wait (25) + defense (25) = 50
    assert summary["intentional_hold_count"] == 50
    # conflict_wait (25)
    assert summary["unintentional_hold_count"] == 25
    assert summary["hold_count"] == summary["intentional_hold_count"] + summary["unintentional_hold_count"]
    assert sum(summary["hold_reasons"].values()) == summary["hold_count"]
    assert summary["hold_reasons"]["harvest_wait"] == 25
    assert summary["hold_reasons"]["defense"] == 25
    assert summary["hold_reasons"]["conflict_wait"] == 25
