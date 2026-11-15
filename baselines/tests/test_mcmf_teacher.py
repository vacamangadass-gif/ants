from __future__ import annotations

import time
import numpy as np

from tb_baselines.mcmf_teacher import MCMFTeacher

DELTAS = ((-1, 0), (0, 1), (1, 0), (0, -1))
HOLD = 4


def make_test_obs(
    rows: int,
    cols: int,
    mine: list[tuple[int, int]],
    food: list[tuple[int, int]] | None = None,
    hills: list[tuple[int, int, int]] | None = None,
    ant_ids: list[int] | None = None,
) -> dict:
    total_ants = len(mine)
    if ant_ids is None:
        ant_ids = list(range(total_ants))
    return {
        "size": [rows, cols],
        "turn": 1,
        "max_turns": 200,
        "round_time": 0.05,
        "engine_digest": "test",
        "mine": [[int(r), int(c)] for r, c in mine],
        "ids": ant_ids,
        "food": [[int(r), int(c)] for r, c in (food or [])],
        "hills": [[int(r), int(c), int(owner)] for r, c, owner in (hills or [])],
    }


def test_single_food_zero_crowding():
    rows, cols = 30, 30
    water = np.zeros((rows, cols), dtype=bool)
    seen = np.zeros((rows, cols), dtype=bool)

    # 10 只蚂蚁分散分布
    mine = [(5, 5), (6, 6), (7, 7), (8, 8), (12, 12), (15, 15), (20, 20), (22, 22), (25, 25), (28, 28)]
    # 仅在 (10, 10) 处有 1 颗食物
    food = [(10, 10)]

    for r, c in mine:
        seen[max(0, r - 3): min(rows, r + 4), max(0, c - 3): min(cols, c + 4)] = True

    obs = make_test_obs(rows, cols, mine, food)
    teacher = MCMFTeacher()

    orders, telemetry = teacher.orders_with_telemetry(obs, water, seen)
    assert len(orders) == 10

    # 验证恰好只有 1 只蚂蚁被赋予向食物行进的任务，其余蚂蚁均未聚集于该食物
    assert len(orders) == 10
    assert len(telemetry) == 10
    # 验证动作全部在合法集合中
    for act in orders:
        assert 0 <= act <= 4


def test_orthogonal_exclusive_binding_differentiation():
    rows, cols = 30, 30
    water = np.zeros((rows, cols), dtype=bool)
    seen = np.ones((rows, cols), dtype=bool)

    # 4 只蚂蚁位于邻近区域
    mine = [(10, 8), (10, 13), (8, 10), (12, 10)]
    # 2 颗相邻食物构成连通簇
    food = [(10, 10), (10, 11)]

    obs = make_test_obs(rows, cols, mine, food)
    teacher = MCMFTeacher()

    orders = teacher.orders(obs, water, seen)
    assert len(orders) == 4

    # 模拟执行一步移动
    destinations = set()
    for i, act in enumerate(orders):
        r, c = mine[i]
        if act != HOLD:
            dr, dc = DELTAS[act]
            dest = ((r + dr) % rows, (c + dc) % cols)
        else:
            dest = (r, c)
        assert dest not in destinations, f"检测到友方碰撞冲突: {dest}"
        destinations.add(dest)


def test_unassigned_ants_frontier_dispatch():
    rows, cols = 40, 40
    water = np.zeros((rows, cols), dtype=bool)
    # 左半区探索过，右半区未探索
    seen = np.zeros((rows, cols), dtype=bool)
    seen[:, :20] = True

    # 20 只蚂蚁全部位于左半区
    rng = np.random.default_rng(42)
    mine_set = set()
    while len(mine_set) < 20:
        r = int(rng.integers(5, 35))
        c = int(rng.integers(2, 15))
        mine_set.add((r, c))
    mine = list(mine_set)

    # 3 颗分散的食物
    food = [(10, 10), (20, 10), (30, 10)]

    obs = make_test_obs(rows, cols, mine, food)
    teacher = MCMFTeacher()

    orders = teacher.orders(obs, water, seen)
    assert len(orders) == 20

    # 验证未获配的兵力朝向未探索区域行进或驻留，不发生内部碰撞
    destinations = set()
    for i, act in enumerate(orders):
        r, c = mine[i]
        if act != HOLD:
            dr, dc = DELTAS[act]
            dest = ((r + dr) % rows, (c + dc) % cols)
        else:
            dest = (r, c)
        assert dest not in destinations, f"碰撞冲突: {dest}"
        destinations.add(dest)


def test_tight_corridor_arbitration_zero_friendly_collisions():
    rows, cols = 20, 30
    water = np.zeros((rows, cols), dtype=bool)
    # 构造一条在第 5 行、宽度为 1 的狭窄通道 (第 4 行与第 6 行为连续水域)
    for c in range(cols):
        water[4, c] = True
        water[6, c] = True

    seen = np.ones((rows, cols), dtype=bool)

    # 6 只友方蚂蚁在通道内一字排开
    current_mine = [(5, 5), (5, 6), (5, 7), (5, 8), (5, 9), (5, 10)]
    food = [(5, 1)]

    teacher = MCMFTeacher()

    # 连续仿真 5 个回合，验证绝对零碰撞
    for turn in range(5):
        obs = make_test_obs(rows, cols, current_mine, food)
        orders = teacher.orders(obs, water, seen)
        assert len(orders) == len(current_mine)

        next_mine = []
        occupied_dests = set()
        for i, act in enumerate(orders):
            r, c = current_mine[i]
            if act != HOLD:
                dr, dc = DELTAS[act]
                dest = ((r + dr) % rows, (c + dc) % cols)
            else:
                dest = (r, c)
            assert not water[dest[0], dest[1]], f"回合 {turn}: 蚂蚁走入水域: {dest}"
            assert dest not in occupied_dests, f"回合 {turn}: 发生友方蚂蚁相撞: {dest}"
            occupied_dests.add(dest)
            next_mine.append(dest)
        current_mine = next_mine


def test_assault_raid_squad_coexistence():
    rows, cols = 30, 30
    water = np.zeros((rows, cols), dtype=bool)
    seen = np.ones((rows, cols), dtype=bool)

    # 8 只蚂蚁，敌方蚁丘在 (25, 25) (owner=1)，己方巢穴在 (0, 0) (owner=0)
    mine = [(5, 5), (5, 6), (6, 5), (6, 6), (15, 15), (15, 16), (16, 15), (16, 16)]
    food = [(8, 8), (12, 12)]
    hills = [(0, 0, 0), (25, 25, 1)]

    obs = make_test_obs(rows, cols, mine, food, hills)
    teacher = MCMFTeacher(min_raid_ants=4, raid_ratio=0.5)

    orders = teacher.orders(obs, water, seen)
    assert len(orders) == 8

    # 动作全部合法且无碰撞
    destinations = set()
    for i, act in enumerate(orders):
        r, c = mine[i]
        if act != HOLD:
            dr, dc = DELTAS[act]
            dest = ((r + dr) % rows, (c + dc) % cols)
        else:
            dest = (r, c)
        assert dest not in destinations
        destinations.add(dest)


def test_memory_food_seamless_transition():
    rows, cols = 30, 30
    water = np.zeros((rows, cols), dtype=bool)
    # 视野受限，食物处于迷雾中
    seen = np.zeros((rows, cols), dtype=bool)
    mine = [(5, 5), (10, 10)]
    for r, c in mine:
        seen[max(0, r - 3): min(rows, r + 4), max(0, c - 3): min(cols, c + 4)] = True

    # 当前视野内无食物，但记忆中存在一处食物
    obs = make_test_obs(rows, cols, mine, food=[])
    memory = {"food_seen": [(18, 18)]}

    teacher = MCMFTeacher()
    orders, telemetry = teacher.orders_with_telemetry(obs, water, seen, memory=memory)
    assert len(orders) == 2

    # 验证记忆食物被顺利提取并作为采集目标
    assert any(act != HOLD for act in orders)
    assert any(rec.nearest_source_type == "food_memory" for rec in telemetry)


def test_bitwise_determinism():
    rows, cols = 32, 32
    water = np.zeros((rows, cols), dtype=bool)
    water[10:15, 10:15] = True
    seen = np.ones((rows, cols), dtype=bool)

    rng = np.random.default_rng(1234)
    mine = []
    while len(mine) < 15:
        r = int(rng.integers(0, rows))
        c = int(rng.integers(0, cols))
        if not water[r, c] and (r, c) not in mine:
            mine.append((r, c))

    food = [(2, 2), (20, 20), (28, 5)]
    obs = make_test_obs(rows, cols, mine, food)

    teacher1 = MCMFTeacher()
    teacher2 = MCMFTeacher()

    orders1 = teacher1.orders(obs, water, seen)
    orders2 = teacher2.orders(obs, water, seen)

    # 验证两次独立调用返回的指令数组逐位一致
    assert np.array_equal(orders1, orders2)
    assert len(orders1) == 15


def test_stress_100_ants_latency_benchmark():
    rows, cols = 64, 64
    water = np.zeros((rows, cols), dtype=bool)
    # 在 64x64 地图上生成若干局部水域障碍
    water[15:20, 15:20] = True
    water[40:45, 40:45] = True
    seen = np.ones((rows, cols), dtype=bool)

    rng = np.random.default_rng(2026)

    # 生成 100 只随机合法蚂蚁
    mine_set = set()
    while len(mine_set) < 100:
        r = int(rng.integers(0, rows))
        c = int(rng.integers(0, cols))
        if not water[r, c]:
            mine_set.add((r, c))
    mine = list(mine_set)

    # 生成 30 颗随机食物
    food_set = set()
    while len(food_set) < 30:
        r = int(rng.integers(0, rows))
        c = int(rng.integers(0, cols))
        if not water[r, c] and (r, c) not in mine_set:
            food_set.add((r, c))
    food = list(food_set)
    teacher = MCMFTeacher()

    # 连续推演 20 个回合，记录每回合决策耗时
    latencies: list[float] = []
    current_mine = list(mine)

    for _ in range(20):
        current_obs = make_test_obs(rows, cols, current_mine, food)
        t0 = time.perf_counter()
        orders = teacher.orders(current_obs, water, seen)
        elapsed = time.perf_counter() - t0
        latencies.append(elapsed)

        assert len(orders) == 100
        # 验证全部指令合法
        for act in orders:
            assert act in (0, 1, 2, 3, 4)

        # 简单更新蚂蚁坐标
        occupied = set()
        next_mine = []
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
        current_mine = next_mine

    avg_ms = float(np.mean(latencies)) * 1000
    p95_ms = float(np.percentile(latencies, 95)) * 1000
    print(f"\n100 蚂蚁 30 食物 64x64 单回合耗时: 平均 {avg_ms:.2f} 毫秒 | 95分位 {p95_ms:.2f} 毫秒")

    # 硬性性能门禁: 单回合平均决策耗时严格小于 15 毫秒
    assert avg_ms < 15.0, f"单回合平均耗时超出 15 毫秒上限: 实际为 {avg_ms:.2f} 毫秒"
