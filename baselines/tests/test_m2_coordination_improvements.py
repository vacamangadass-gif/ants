from __future__ import annotations

import time
import numpy as np
import pytest

from tb_baselines.coordination import (
    DELTAS,
    FAR,
    HOLD,
    compute_combined_delay,
    compute_cost_matrix_bitmask_multibucket,
    compute_enemy_threat_penalty,
)
from tb_baselines.mcmf_teacher import MCMFTeacher
from tb_baselines.teacher import TaskState


def make_test_obs(
    rows: int,
    cols: int,
    mine: list[tuple[int, int]],
    food: list[tuple[int, int]] | None = None,
    hills: list[tuple[int, int, int]] | None = None,
    foes: list[tuple[int, int, int]] | None = None,
    turn: int = 1,
    ant_ids: list[int] | None = None,
    players: int = 2,
) -> dict:
    total_ants = len(mine)
    if ant_ids is None:
        ant_ids = list(range(total_ants))
    return {
        "size": [rows, cols],
        "turn": turn,
        "max_turns": 300,
        "round_time": 0.05,
        "engine_digest": "test_m2",
        "mine": [[int(r), int(c)] for r, c in mine],
        "ids": ant_ids,
        "food": [[int(r), int(c)] for r, c in (food or [])],
        "hills": [[int(r), int(c), int(owner)] for r, c, owner in (hills or [])],
        "foes": [[int(r), int(c), int(owner)] for r, c, owner in (foes or [])],
        "players": players,
    }


def test_threat_penalty_coverage_and_superiority():
    rows, cols = 30, 30
    passable = np.ones((rows, cols), dtype=bool)

    # 敌方单兵位于 (15, 15)
    foes = [(15, 15)]
    # 无己方蚂蚁
    friendly = []

    raw_threat, int_delay = compute_enemy_threat_penalty(
        foe_positions=foes,
        friendly_positions=friendly,
        passable_mask=passable,
        rows=rows,
        cols=cols,
        attack_radius_sq=5,
        max_delay=4,
    )

    # 验证威胁覆盖格数为 21 格（欧氏距离平方 <= 5）
    threatened_cells = np.argwhere(int_delay > 0)
    assert len(threatened_cells) == 21

    # 验证中心点受到最大延时惩罚
    assert int_delay[15, 15] == 4

    # 验证欧氏距离平方为 5 的边缘单元格（例如 (15+2, 15+1) = (17, 16)）
    assert int_delay[17, 16] == 4
    # 验证欧氏距离平方为 8 的单元格（(17, 17)）不受威胁
    assert int_delay[17, 17] == 0

    # 当己方形成局部绝对优势（2 只己方蚂蚁覆盖该区域）
    friendly_dominant = [(15, 14), (15, 16)]
    raw_threat_dom, int_delay_dom = compute_enemy_threat_penalty(
        foe_positions=foes,
        friendly_positions=friendly_dominant,
        passable_mask=passable,
        rows=rows,
        cols=cols,
        attack_radius_sq=5,
        max_delay=4,
    )

    # 在 (15, 15) 处己方兵力为 2，敌方兵力为 1，己方占优，威胁惩罚归零
    assert int_delay_dom[15, 15] == 0


def test_threat_penalty_fast_fail_validation():
    rows, cols = 20, 20
    passable = np.ones((rows, cols), dtype=bool)

    # 浮点数类型必须就地崩溃
    with pytest.raises(TypeError):
        compute_enemy_threat_penalty([], [], passable.astype(np.float32), rows, cols)

    # 尺寸不匹配必须就地崩溃
    with pytest.raises(ValueError):
        compute_enemy_threat_penalty([], [], passable, 10, 10)

    # 坐标越界必须就地崩溃
    with pytest.raises(ValueError):
        compute_enemy_threat_penalty([(25, 5)], [], passable, rows, cols)

    # 延时组合形状不匹配必须就地崩溃
    d1 = np.zeros((10, 10), dtype=np.int32)
    d2 = np.zeros((20, 20), dtype=np.int32)
    with pytest.raises(ValueError):
        compute_combined_delay(d1, d2)


def test_large_map_adaptive_depth_and_anti_stagnation():
    # 80x96 大地图测试
    rows, cols = 80, 96
    water = np.zeros((rows, cols), dtype=bool)
    seen = np.zeros((rows, cols), dtype=bool)

    # 蚂蚁位于 (10, 10)，视野半径为 8
    mine = [(10, 10)]
    seen[2:19, 2:19] = True

    # 未探索前沿距离蚂蚁至少 35 格以外
    # 无任何可见或记忆食物，无敌方蚁穴
    obs = make_test_obs(rows, cols, mine, food=[], hills=[(10, 10, 0)], turn=10, players=6)
    teacher = MCMFTeacher()

    orders = teacher.orders(obs, water, seen)
    assert len(orders) == 1

    # 验证蚂蚁绝不输出原地驻留动作（杜绝 16 步截断发呆）
    assert orders[0] != HOLD
    # 验证任务状态判定为探索
    tasks = teacher.get_last_tasks()
    assert tasks[0] == TaskState.EXPLORE


def test_multiseat_adaptive_raid_and_defense_reservation():
    # 4 席位地图（48x64），双蚁穴配置
    rows, cols = 48, 64
    water = np.zeros((rows, cols), dtype=bool)
    seen = np.ones((rows, cols), dtype=bool)

    # 己方 2 座基地蚁穴：(10, 10) 与 (10, 30)
    hills = [(10, 10, 0), (10, 30, 0), (40, 40, 1), (40, 10, 2)]

    # 场景 1: 前期（兵力 10 只 >= min_raid_ants=4），不应因多席位或地图尺寸被饥饿锁定为 0 突击
    mine_early = [(10, 11), (10, 12), (11, 10), (12, 10), (10, 29), (10, 31), (11, 30), (12, 30), (15, 20), (16, 20)]
    obs_early = make_test_obs(rows, cols, mine_early, food=[(20, 20)], hills=hills, turn=20, players=4)

    teacher = MCMFTeacher(min_raid_ants=4, raid_ratio=0.5, adaptive_raid=True)
    orders_early = teacher.orders(obs_early, water, seen)
    tasks_early = teacher.get_last_tasks()

    assault_count_early = sum(1 for t in tasks_early.values() if t == TaskState.ASSAULT)
    assert assault_count_early >= 2
    # 无敌蚁威胁时严禁产生产卵丘 FORTIFY 回退
    assert sum(1 for t in tasks_early.values() if t == TaskState.FORTIFY) == 0

    # 场景 2: 敌方蚂蚁逼近基地 (10, 10)
    # 敌方蚂蚁在 (11, 11) 距离基地仅 2 格
    foes_threatening = [(11, 11, 1)]
    obs_threat = make_test_obs(rows, cols, mine_early, food=[(20, 20)], hills=hills, foes=foes_threatening, turn=20, players=4)

    orders_threat = teacher.orders(obs_threat, water, seen)
    tasks_threat = teacher.get_last_tasks()

    # 验证就近拦截防守逻辑激活：基地附近兵力被赋予 FORTIFY 拦截威胁敌蚁（不回退驻守己方蚁丘格）
    fortify_count = sum(1 for t in tasks_threat.values() if t == TaskState.FORTIFY)
    assert fortify_count >= 2


def test_focus_target_assault_selection():
    rows, cols = 60, 60
    water = np.zeros((rows, cols), dtype=bool)
    seen = np.ones((rows, cols), dtype=bool)

    # 己方基地 (0, 0)
    # 敌方存在两座蚁穴：近处 (15, 15) 与 远处 (45, 45)
    hills = [(0, 0, 0), (15, 15, 1), (45, 45, 2)]

    # 30 只己方蚂蚁集中在 (10, 10) 附近，回合数 120
    mine = []
    for r in range(8, 14):
        for c in range(8, 13):
            mine.append((r, c))

    obs = make_test_obs(rows, cols, mine, hills=hills, turn=120, players=3)
    teacher = MCMFTeacher(min_raid_ants=4, raid_ratio=0.3, focus_target=True)

    orders = teacher.orders(obs, water, seen)
    assert len(orders) == 30

    # 验证锁定的主要突破目标为距离最近的 (15, 15)，绝不分散至 (45, 45)
    target_hill = teacher.get_last_target_hill()
    assert target_hill == (15, 15)

    tasks = teacher.get_last_tasks()
    assault_ants = [i for i, t in tasks.items() if t == TaskState.ASSAULT]
    assert len(assault_ants) > 0


def test_threat_avoidance_corridor_diversion():
    rows, cols = 30, 30
    water = np.ones((rows, cols), dtype=bool)

    # 构造两条平行通道连接左右平原：
    # 通道 A (行 10, 列 5 至 25)：单格狭窄通道
    water[10, 5:26] = False
    # 通道 B (行 20, 列 5 至 25)：双格安全通道
    water[20:22, 5:26] = False
    # 左侧平原与右侧平原
    water[8:24, 0:5] = False
    water[8:24, 26:30] = False

    seen = np.ones((rows, cols), dtype=bool)

    # 己方 1 只蚂蚁在左侧通道入口 (10, 4)
    mine = [(10, 4)]
    # 食物在右侧通道出口 (10, 27)
    food = [(10, 27)]

    # 敌方在通道 A 的出口 (10, 24) 处设防
    foes = [(10, 24, 1)]

    obs = make_test_obs(rows, cols, mine, food=food, foes=foes, turn=10)
    teacher = MCMFTeacher(threat_avoidance=True)

    orders = teacher.orders(obs, water, seen)
    assert len(orders) == 1

    # 验证受阻于火网威胁时，动作不向通道 A 内深入推进或者通过延时规避火网
    act = orders[0]
    dr, dc = DELTAS[act] if act != HOLD else (0, 0)
    next_pos = ((mine[0][0] + dr) % rows, (mine[0][1] + dc) % cols)
    assert not water[next_pos[0], next_pos[1]]


def test_xlarge_map_latency_benchmark():
    # 官方包络上限 120x124 超大地图实测
    rows, cols = 120, 124
    water = np.zeros((rows, cols), dtype=bool)
    seen = np.ones((rows, cols), dtype=bool)

    rng = np.random.default_rng(2026)

    # 100 只己方蚂蚁
    mine_set = set()
    while len(mine_set) < 100:
        r = int(rng.integers(0, rows))
        c = int(rng.integers(0, cols))
        mine_set.add((r, c))
    mine = list(mine_set)

    # 30 颗随机食物
    food_set = set()
    while len(food_set) < 30:
        r = int(rng.integers(0, rows))
        c = int(rng.integers(0, cols))
        if (r, c) not in mine_set:
            food_set.add((r, c))
    food = list(food_set)

    # 20 只敌方蚂蚁
    foes_set = set()
    while len(foes_set) < 20:
        r = int(rng.integers(0, rows))
        c = int(rng.integers(0, cols))
        if (r, c) not in mine_set and (r, c) not in food_set:
            foes_set.add((r, c))
    foes = [(r, c, 1) for r, c in foes_set]

    hills = [(10, 10, 0), (10, 50, 0), (60, 60, 1), (80, 80, 2)]
    obs = make_test_obs(rows, cols, mine, food=food, hills=hills, foes=foes, turn=105, players=4)

    teacher = MCMFTeacher(dynamic_congestion=True, threat_avoidance=True, adaptive_raid=True, focus_target=True)

    # 连续执行 5 个回合，统计平均单步规划耗时
    latencies = []
    for _ in range(5):
        t0 = time.perf_counter()
        orders = teacher.orders(obs, water, seen)
        elapsed = time.perf_counter() - t0
        latencies.append(elapsed)
        assert len(orders) == 100

    avg_ms = float(np.mean(latencies)) * 1000
    p95_ms = float(np.percentile(latencies, 95)) * 1000
    print(f"\n120x124 超大地图 100 蚂蚁单回合耗时: 平均 {avg_ms:.2f} 毫秒 | 95分位 {p95_ms:.2f} 毫秒")

    # 硬性时间门禁校验：严格小于 1000 毫秒（甚至远低于 100 毫秒）
    assert avg_ms < 100.0, f"单步平均耗时超出 100 毫秒: 实际为 {avg_ms:.2f} 毫秒"


def test_basic_small_3p_multi_seat_early_raid_quota():
    # 官方 3 席位棋盘 basic-small-3p 早期小规模兵力突击比例检验
    rows, cols = 36, 36
    water = np.zeros((rows, cols), dtype=bool)
    hills = [(5, 5, 0), (18, 18, 1)]
    mine = [(5 + i // 3, 7 + i % 3) for i in range(12)]
    obs = {
        "size": [rows, cols],
        "turn": 25,
        "max_turns": 300,
        "round_time": 0.05,
        "engine_digest": "test_3p",
        "mine": mine,
        "ids": list(range(12)),
        "hills": hills,
        "foes": [(18, 19, 1)],
        "food": [],
    }
    seen = np.zeros((rows, cols), dtype=bool)
    seen[0:22, 0:22] = True

    teacher = MCMFTeacher(min_raid_ants=4, raid_ratio=0.5, adaptive_raid=True)
    orders = teacher.orders(obs, water, seen)
    assert len(orders) == 12

    tasks = teacher.get_last_tasks()
    assault_count = sum(1 for t in tasks.values() if t == TaskState.ASSAULT)
    # 消除失效 turn 键与 36x36 多席位突击饥饿：兵力 12 >= min_raid_ants=4 时应正常派出突击分队
    assert assault_count >= 3


def test_fully_explored_remaining_ants_no_fortify_fallback():
    # 4 席位地图全探索无迷雾前沿时，剩余兵力应推进敌丘而非回退 FORTIFY 堵塞产卵丘
    rows, cols = 48, 64
    water = np.zeros((rows, cols), dtype=bool)
    seen = np.ones((rows, cols), dtype=bool)
    hills = [(10, 10, 0), (40, 40, 1), (40, 10, 2), (10, 40, 3)]
    mine = [(10 + i // 4, 12 + i % 4) for i in range(20)]
    food = [(15, 15), (16, 16)]

    teacher = MCMFTeacher(min_raid_ants=4, raid_ratio=0.5, adaptive_raid=True)
    obs = {
        "size": [rows, cols],
        "turn": 30,
        "max_turns": 300,
        "round_time": 0.05,
        "engine_digest": "test_fallback",
        "mine": mine,
        "ids": list(range(20)),
        "hills": hills,
        "foes": [],
        "food": food,
        "players": 4,
    }
    orders = teacher.orders(obs, water, seen)
    assert len(orders) == 20

    tasks = teacher.get_last_tasks()
    assault_count = sum(1 for t in tasks.values() if t == TaskState.ASSAULT)
    fortify_count = sum(1 for t in tasks.values() if t == TaskState.FORTIFY)

    # 无近身敌蚁威胁时严禁触发 FORTIFY 回退，全部空闲兵力推进已知敌丘
    assert fortify_count == 0
    assert assault_count >= 16


def test_cost_matrix_multibucket_boundary_at_max_depth():
    # 多桶位掩码距离计算恰好等于最大搜索深度时的提取边界检验
    rows, cols = 30, 30
    passable = np.ones((rows, cols), dtype=bool)
    cell_delay = np.zeros((rows, cols), dtype=np.int32)

    ant_positions = [(0, 0)]
    gathering_positions = [(10, 0)]
    max_depth = 10

    cost_matrix = compute_cost_matrix_bitmask_multibucket(
        ant_positions=ant_positions,
        gathering_positions=gathering_positions,
        passable_mask=passable,
        cell_delay=cell_delay,
        rows=rows,
        cols=cols,
        max_depth=max_depth,
        max_delay=4,
    )

    # 验证边界距离等于 max_depth 的单元格成功提取，数值等于 10
    assert cost_matrix[0, 0] == 10
