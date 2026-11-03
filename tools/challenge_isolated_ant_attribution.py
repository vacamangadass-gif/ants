import random
import numpy as np
from tb_baselines.teacher import CompoundTeacher, Teacher, known
from tb_baselines.telemetry import (
    FAR,
    HOLD,
    HoldReason,
    RejectionCode,
    TelemetryRecord,
)

# 构建测试棋盘数据结构
def build_test_board(
    rows: int,
    cols: int,
    ants: list[tuple[int, int]],
    hills: list[tuple[int, int, int]] | None = None,
    food: list[tuple[int, int]] | None = None,
    ids: list[int] | None = None,
) -> dict:
    if hills is None:
        hills = []
    if food is None:
        food = []
    if ids is None:
        ids = list(range(len(ants)))
    return {
        "size": [rows, cols],
        "mine": [list(a) for a in ants],
        "ids": ids,
        "foes": [],
        "food": [list(f) for f in food],
        "hills": [list(h) for h in hills],
        "water": {"rle": [0, rows * cols]},
        "vis": {"rle": [0, rows * cols]},
    }

# 场景 1：单兵水域十字环绕（Water Lake），全图无友方蚁穴
def verify_scenario_1_water_lake_configurations():
    size = 20
    test_cases = [
        {"food": [], "hills": [], "all_seen": True, "label": "无食物无敌丘全可见"},
        {"food": [(2, 2)], "hills": [], "all_seen": True, "label": "远端不可达食物全可见"},
        {"food": [], "hills": [(3, 3, 1)], "all_seen": True, "label": "远端不可达敌丘全可见"},
        {"food": [(2, 2), (5, 5)], "hills": [(3, 3, 1)], "all_seen": True, "label": "远端食物与敌丘全可见"},
        {"food": [(2, 2)], "hills": [(3, 3, 1)], "all_seen": False, "label": "远端食物与敌丘带迷雾"},
    ]

    for tc in test_cases:
        obs = build_test_board(
            rows=size,
            cols=size,
            ants=[(10, 10)],
            hills=tc["hills"],
            food=tc["food"],
            ids=[101],
        )
        water, seen = known(obs)
        if tc["all_seen"]:
            seen[:] = True
        else:
            seen[:] = False
            seen[9:12, 9:12] = True

        # 水域十字包围
        water[9, 10] = True
        water[11, 10] = True
        water[10, 9] = True
        water[10, 11] = True

        for teacher_cls in (Teacher, CompoundTeacher):
            agent = teacher_cls()
            orders, records = agent.orders_with_telemetry(obs, water, seen)
            assert len(orders) == 1, f"{tc['label']} {teacher_cls.__name__} 指令数量错误"
            assert orders[0] == HOLD, f"{tc['label']} {teacher_cls.__name__} 预期动作应为 HOLD"
            assert len(records) == 1, f"{tc['label']} {teacher_cls.__name__} 遥测记录数量错误"
            rec = records[0]

            assert rec.hold_reason != HoldReason.DEFENSE.value, (
                f"在无友方蚁穴时异常输出 DEFENSE: {tc['label']} {teacher_cls.__name__}"
            )
            assert rec.is_intentional_hold is False, (
                f"在无友方蚁穴孤立水域时 is_intentional_hold 异常为 True: {tc['label']} {teacher_cls.__name__}"
            )
            assert rec.hold_reason in (
                HoldReason.STALE_TARGET.value,
                HoldReason.NO_TARGET.value,
                HoldReason.NO_VALID_ACTION.value,
            ), f"未预期的停滞动因: {rec.hold_reason}"

# 场景 2：隔离陆地岛屿（Island Moat），多兵驻留与调度，全图无友方蚁穴
def verify_scenario_2_island_moat():
    size = 24
    obs = build_test_board(
        rows=size,
        cols=size,
        ants=[(11, 11), (11, 12), (12, 11)],
        hills=[(2, 2, 1)],
        food=[(3, 3), (4, 4)],
        ids=[201, 202, 203],
    )
    water, seen = known(obs)
    seen[:] = True

    # 在 [10..13, 10..13] 外围建立一圈闭合水域隔离带
    for r in range(9, 15):
        for c in range(9, 15):
            if r in (9, 14) or c in (9, 14):
                water[r, c] = True

    for teacher_cls in (Teacher, CompoundTeacher):
        agent = teacher_cls()
        orders, records = agent.orders_with_telemetry(obs, water, seen)
        assert len(orders) == 3
        assert len(records) == 3
        for i, rec in enumerate(records):
            assert rec.hold_reason != HoldReason.DEFENSE.value, (
                f"岛屿隔离多兵环境异常输出 DEFENSE: {teacher_cls.__name__}"
            )
            if orders[i] == HOLD:
                assert rec.is_intentional_hold is False, (
                    f"岛屿隔离多兵停滞时 is_intentional_hold 异常为 True: {teacher_cls.__name__}"
                )

# 场景 3：环面赤道/子午线完全切断的大陆桥阻隔，全图无友方蚁穴
def verify_scenario_3_torus_partition():
    size = 24
    obs = build_test_board(
        rows=size,
        cols=size,
        ants=[(3, 5), (4, 6), (5, 7)],
        hills=[(18, 18, 1)],
        food=[(15, 15), (16, 16)],
        ids=[301, 302, 303],
    )
    water, seen = known(obs)
    seen[:] = True

    # 将第 0 行与第 10 行全部填满水域，彻底切断上下半区连通
    water[0, :] = True
    water[10, :] = True

    for teacher_cls in (Teacher, CompoundTeacher):
        agent = teacher_cls()
        orders, records = agent.orders_with_telemetry(obs, water, seen)
        assert len(records) == 3
        for i, rec in enumerate(records):
            assert rec.hold_reason != HoldReason.DEFENSE.value, (
                f"环面切断环境异常输出 DEFENSE: {teacher_cls.__name__}"
            )
            if orders[i] == HOLD:
                assert rec.is_intentional_hold is False, (
                    f"环面切断环境 is_intentional_hold 异常为 True: {teacher_cls.__name__}"
                )

# 场景 4：随机拓扑对抗性模糊测试（100 组随机生成场景）
def verify_scenario_4_adversarial_fuzzing():
    rng = random.Random(1337)
    sizes = [14, 18, 22, 28, 32]

    for trial in range(100):
        board_size = rng.choice(sizes)
        ant_count = rng.randint(1, 4)
        center_r = rng.randint(4, board_size - 5)
        center_c = rng.randint(4, board_size - 5)

        ants: list[tuple[int, int]] = []
        for i in range(ant_count):
            ants.append((center_r + (i // 2), center_c + (i % 2)))

        # 远端放置 0..3 个敌方蚁穴（无己方蚁穴）
        enemy_hills = []
        for h in range(rng.randint(0, 3)):
            enemy_hills.append((h, h, 1))

        # 远端放置 0..5 个食物
        foods = []
        for f in range(rng.randint(0, 5)):
            foods.append((f, (board_size - 1 - f) % board_size))

        obs = build_test_board(
            rows=board_size,
            cols=board_size,
            ants=ants,
            hills=enemy_hills,
            food=foods,
            ids=[1000 + i for i in range(ant_count)],
        )
        water, seen = known(obs)

        # 随机可见度：50% 概率全图可见，50% 概率仅中心区域可见
        if rng.random() < 0.5:
            seen[:] = True
        else:
            seen[:] = False
            for ar, ac in ants:
                seen[max(0, ar - 2) : min(board_size, ar + 3), max(0, ac - 2) : min(board_size, ac + 3)] = True

        # 将蚂蚁群体所在局部区域外围用连续水域闭合圈完全阻断
        min_r = min(a[0] for a in ants) - 1
        max_r = max(a[0] for a in ants) + 1
        min_c = min(a[1] for a in ants) - 1
        max_c = max(a[1] for a in ants) + 1

        for r in range(min_r, max_r + 1):
            for c in range(min_c, max_c + 1):
                if r == min_r or r == max_r or c == min_c or c == max_c:
                    water[r % board_size, c % board_size] = True

        for teacher_cls in (Teacher, CompoundTeacher):
            agent = teacher_cls()
            orders, records = agent.orders_with_telemetry(obs, water, seen)
            assert len(records) == ant_count

            for i, rec in enumerate(records):
                assert rec.hold_reason != HoldReason.DEFENSE.value, (
                    f"Fuzzing Trial {trial}: {teacher_cls.__name__} 异常输出 DEFENSE"
                )
                if orders[i] == HOLD:
                    ar, ac = rec.pos
                    is_adjacent_food = False
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr = (ar + dr) % board_size
                        nc = (ac + dc) % board_size
                        if (nr, nc) in foods:
                            is_adjacent_food = True
                            break
                    if not is_adjacent_food:
                        assert rec.is_intentional_hold is False, (
                            f"Fuzzing Trial {trial}: {teacher_cls.__name__} is_intentional_hold 异常为 True"
                        )
                        assert rec.hold_reason in (
                            HoldReason.STALE_TARGET.value,
                            HoldReason.NO_TARGET.value,
                            HoldReason.NO_VALID_ACTION.value,
                            HoldReason.POTENTIAL_PLATEAU.value,
                            HoldReason.CONFLICT_WAIT.value,
                        ), f"Fuzzing Trial {trial}: 非法停滞原因 {rec.hold_reason}"

# 场景 5：友方蚁穴位于远端，孤立蚂蚁不产生虚假防御归因对比
def verify_scenario_5_remote_friendly_hill():
    size = 20
    # 己方蚁穴在 (0, 0)，孤立蚂蚁在 (10, 10) 被水域包围
    obs = build_test_board(
        rows=size,
        cols=size,
        ants=[(10, 10)],
        hills=[(0, 0, 0)],
        food=[(2, 2)],
        ids=[501],
    )
    water, seen = known(obs)
    seen[:] = True

    water[9, 10] = True
    water[11, 10] = True
    water[10, 9] = True
    water[10, 11] = True

    for teacher_cls in (Teacher, CompoundTeacher):
        agent = teacher_cls()
        orders, records = agent.orders_with_telemetry(obs, water, seen)
        assert len(records) == 1
        rec = records[0]
        # 蚂蚁位于 (10, 10)，并非位于 (0, 0) 蚁穴上，不可归因为 DEFENSE
        assert rec.hold_reason != HoldReason.DEFENSE.value, (
            f"远端友方蚁穴时孤立蚂蚁误判为 DEFENSE: {teacher_cls.__name__}"
        )
        assert rec.is_intentional_hold is False, (
            f"远端友方蚁穴时孤立蚂蚁 is_intentional_hold 应为 False: {teacher_cls.__name__}"
        )

    # 对照组：当蚂蚁确实位于己方蚁穴 (0, 0) 且原地驻留时，必须正确识别为 DEFENSE
    obs_defend = build_test_board(
        rows=size,
        cols=size,
        ants=[(0, 0)],
        hills=[(0, 0, 0)],
        food=[],
        ids=[502],
    )
    water_defend, seen_defend = known(obs_defend)
    seen_defend[:] = True
    for teacher_cls in (Teacher, CompoundTeacher):
        agent = teacher_cls()
        orders_def, records_def = agent.orders_with_telemetry(obs_defend, water_defend, seen_defend)
        rec_def = records_def[0]
        assert rec_def.hold_reason == HoldReason.DEFENSE.value, (
            f"位于己方蚁穴且无目标驻留时应判定为 DEFENSE: {teacher_cls.__name__}"
        )
        assert rec_def.is_intentional_hold is True, (
            f"位于己方蚁穴防御时 is_intentional_hold 应为 True: {teacher_cls.__name__}"
        )


if __name__ == "__main__":
    verify_scenario_1_water_lake_configurations()
    verify_scenario_2_island_moat()
    verify_scenario_3_torus_partition()
    verify_scenario_4_adversarial_fuzzing()
    verify_scenario_5_remote_friendly_hill()
    print("全部 5 组孤立蚂蚁对抗性挑战场景与 100 轮随机模糊测试执行通过")
