"""教师策略的记忆机制：被记住的敌方蚁穴或食物单元格如何对势场产生影响，以及何时不产生影响。
使用合成棋盘进行测试，因为这里关注的是规则逻辑而非真实对局。"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tb_baselines.teacher import DELTAS, FAR, HOLD, CompoundTeacher, Teacher, known  # noqa: E402


def board(size: int = 24, ant=(12, 12), hills=(), food=()) -> dict:
    """构造一个开阔棋盘：包含一只蚂蚁、无水域，以及测试显式指定的其他要素。"""
    return {
        "size": [size, size], "mine": [list(ant)], "ids": [0], "foes": [],
        "food": [list(f) for f in food], "hills": [list(h) for h in hills],
        "water": {"rle": [0, size * size]}, "vis": {"rle": [0, size * size]},
    }


def test_a_remembered_enemy_hill_out_of_sight_is_the_nearest_target():
    """距离 12 格已超出视野半径 (8.8)，因此无记忆教师在此处仅视其为探索前沿；
    而带记忆教师会将该蚁穴作为 0 级初始源点，优先级高于其他一切。"""
    t = Teacher()
    obs = board()
    water, seen = known(obs)
    plain = t.field(obs, water, seen)
    remembering = t.field(obs, water, seen, {"hill_foe_seen": [[0, 12]], "food_seen": []})
    assert plain[0, 12] == t.hill_lead + 2, "未被看见，因此具有探索前沿的等级"
    assert remembering[0, 12] == 0, "敌方蚁穴一旦被发现即作为攻击目标，直到亲眼确认其被摧毁"
    assert remembering[12, 12] < plain[12, 12], "蚂蚁所在位置因此距离目标更近"


def test_a_remembered_hill_whose_square_is_seen_empty_is_a_hill_razed():
    """距离 4 格处于视野范围内。视野并未显示该位置有蚁穴，说明记忆已过时，
    带记忆教师会结合视野校验记忆，避免走向一个已不复存在的蚁穴。"""
    t = Teacher()
    obs = board()
    water, seen = known(obs)
    remembering = t.field(obs, water, seen, {"hill_foe_seen": [[12, 16]], "food_seen": []})
    assert remembering[12, 16] != 0
    assert np.array_equal(remembering, t.field(obs, water, seen)), "被摧毁的蚁穴不应产生任何影响"


def test_remembered_food_out_of_sight_sits_between_food_in_view_and_the_frontier():
    t = Teacher()
    obs = board(food=[(12, 14)])
    water, seen = known(obs)
    f = t.field(obs, water, seen, {"hill_foe_seen": [], "food_seen": [[0, 12], [12, 14]]})
    assert f[12, 14] == t.hill_lead, "视野内的食物保持其优先级等级"
    assert f[0, 12] == t.hill_lead + 1, "视野外记住的食物优先级落后其一级"
    assert t.field(obs, water, seen)[0, 12] == t.hill_lead + 2, "探索前沿则再落后一级"


def test_without_a_memory_the_remembering_teacher_is_the_plain_one():
    t = Teacher()
    obs = board(hills=[(12, 2, 1)], food=[(3, 20)])
    water, seen = known(obs)
    assert np.array_equal(t.orders(obs, water, seen), t.orders(obs, water, seen, None))
    assert np.array_equal(t.orders(obs, water, seen), t.orders(obs, water, seen, {}))
    assert (t.field(obs, water, seen) != FAR).any()


def make_board(size: int = 24, ants=((12, 12),), hills=(), food=()) -> dict:
    """构造包含多只蚂蚁、指定蚁丘与食物的合成测试观测。"""
    return {
        "size": [size, size],
        "mine": [list(a) for a in ants],
        "ids": list(range(len(ants))),
        "foes": [],
        "food": [list(f) for f in food],
        "hills": [list(h) for h in hills],
        "water": {"rle": [0, size * size]},
        "vis": {"rle": [0, size * size]},
    }


def test_teacher_hill_lead_controls_hill_vs_food_priority():
    """Test 1: Teacher(hill_lead=12) 相比 Teacher(hill_lead=6) 优先选择进攻较远敌丘而非觅食。

    场景设计：
    - 蚂蚁位于 (12, 12)
    - 食物位于 (12, 14)，距离 d_food = 2 (正东方向，动作索引 1)
    - 敌丘位于 (12, 3)，距离 d_hill = 9 (正西方向，动作索引 3)

    判定原理：
    - hill_lead=6:  d_food + 6 = 8 < d_hill = 9  ==> 势能场指引向东采食 (1)
    - hill_lead=12: d_hill = 9 < d_food + 12 = 14 ==> 势能场指引向西拆丘 (3)
    """
    obs = make_board(size=24, ants=[(12, 12)], hills=[(12, 3, 1)], food=[(12, 14)])
    water, seen = known(obs)

    t6 = Teacher(hill_lead=6)
    order_t6 = t6.orders(obs, water, seen)
    assert order_t6[0] == 1, f"hill_lead=6 应向东 (1) 采食，实际为 {order_t6[0]}"

    t12 = Teacher(hill_lead=12)
    order_t12 = t12.orders(obs, water, seen)
    assert order_t12[0] == 3, f"hill_lead=12 应向西 (3) 拆丘，实际为 {order_t12[0]}"


def test_compound_teacher_dynamic_squad_switching():
    """Test 2: CompoundTeacher 态势感知与动态分工切换 (Phase A vs Phase B)。

    - Case 2a: 未探明敌丘 (hills 为空)，兵力 6 只 (>= 4) ==> 触发 Phase A，全员采食与探索。
    - Case 2b: 已探明敌丘，但存活兵力 2 只 (< 4) ==> 触发 Phase A，保持采食积累经济。
    - Case 2c: 已探明敌丘且存活兵力 6 只 (>= 4) ==> 触发 Phase B，动态分为突击分队与采食分队。
    """
    ct = CompoundTeacher(min_raid_ants=4, raid_ratio=0.5)

    # Case 2a: 6 只蚂蚁，无敌丘
    ants_6 = [(10, 10), (10, 11), (11, 10), (11, 11), (12, 10), (12, 11)]
    obs_2a = make_board(size=24, ants=ants_6, hills=[], food=[(12, 14)])
    w_2a, s_2a = known(obs_2a)
    orders_2a = ct.orders(obs_2a, w_2a, s_2a)
    default_2a = Teacher(hill_lead=6).orders(obs_2a, w_2a, s_2a)
    assert np.array_equal(orders_2a, default_2a), "无已知敌丘时，应处于 Phase A 全员采食探索"

    # Case 2b: 2 只蚂蚁，有已知敌丘与更远处的食物。传统 Teacher 因起步领先优势会孤军突击敌丘 (0)；
    # 但 CompoundTeacher 因兵力不足 min_raid_ants (4)，留在 Phase A 全员向东 (1) 采食积累经济。
    obs_2b = make_board(size=24, ants=[(12, 12), (12, 13)], hills=[(5, 12, 1)], food=[(12, 15)])
    w_2b, s_2b = known(obs_2b)
    orders_2b = ct.orders(obs_2b, w_2b, s_2b)
    assert (orders_2b == 1).all(), "兵力不足 min_raid_ants 时，应处于 Phase A 不贸然进攻"

    # Case 2c: 6 只蚂蚁，有已知敌丘 (5, 10) 与食物 (12, 14)
    obs_2c = make_board(size=24, ants=ants_6, hills=[(5, 10, 1)], food=[(12, 14)])
    w_2c, s_2c = known(obs_2c)
    orders_2c = ct.orders(obs_2c, w_2c, s_2c)
    # 距离敌丘最近的 3 只蚂蚁编入突击队向北 (0)，其余 3 只蚂蚁编入采食队向东 (1)
    assert (orders_2c[:3] == 0).all(), f"突击队应向北 (0) 直扑敌丘，实际为 {orders_2c[:3]}"
    assert (orders_2c[3:] == 1).all(), f"采食队应向东 (1) 维持经济，实际为 {orders_2c[3:]}"


def test_compound_teacher_collision_free_guarantee():
    """Test 3: CompoundTeacher 严格无碰撞保障 (零友军冲突、零步入水域、零步入食物阻挡)。

    场景设计：
    - 8 只蚂蚁密集排布在 (10..12, 10..12) 网格中。
    - 北侧设置水域阻挡，东侧设置食物，北侧远方设置敌方蚁丘。
    - 检验所有蚂蚁的下一回合目标方格完全互斥、不与水域重叠、不与食物重叠。
    """
    ct = CompoundTeacher(min_raid_ants=4, raid_ratio=0.5)
    dense_ants = [
        (10, 10), (10, 11), (10, 12),
        (11, 10), (11, 11), (11, 12),
        (12, 10), (12, 11),
    ]
    obs = make_board(size=24, ants=dense_ants, hills=[(5, 10, 1)], food=[(12, 14)])
    water, seen = known(obs)
    # 在北向通道加入水域障碍
    water[9, 10] = True
    water[9, 11] = True
    water[9, 12] = True

    orders = ct.orders(obs, water, seen)
    rows, cols = obs["size"]
    food_set = {tuple(f) for f in obs["food"]}

    destinations = []
    for (ar, ac), a in zip(obs["mine"], orders):
        if a == HOLD:
            destinations.append((ar, ac))
        else:
            dr, dc = DELTAS[a]
            destinations.append(((ar + dr) % rows, (ac + dc) % cols))

    assert len(destinations) == len(obs["mine"]), "指令输出长度必须与存活蚂蚁数严格一致"
    assert len(set(destinations)) == len(destinations), (
        f"必须保证零友军碰撞！重复方格数: {len(destinations) - len(set(destinations))}"
    )
    for r, c in destinations:
        assert not water[r, c], f"禁止蚂蚁步入水域: ({r}, {c})"
        assert (r, c) not in food_set, f"禁止蚂蚁直接移入食物阻挡格: ({r}, {c})"


def test_compound_teacher_determinism():
    """Test 4: CompoundTeacher 确定性保障 (相同观测与记忆产生严格逐位一致的动作输出)。"""
    ct = CompoundTeacher()
    obs = make_board(
        size=24,
        ants=[(10, 10), (10, 11), (11, 10), (11, 11), (12, 10), (12, 11)],
        hills=[(5, 10, 1)],
        food=[(12, 14), (15, 15)],
    )
    water, seen = known(obs)
    memory = {"hill_foe_seen": [[5, 10]], "food_seen": [[12, 14]]}

    orders_run1 = ct.orders(obs, water, seen, memory)
    orders_run2 = ct.orders(obs, water, seen, memory)
    assert np.array_equal(orders_run1, orders_run2), "同一实例连续调用必须输出完全一致的指令"

    ct_fresh = CompoundTeacher()
    orders_fresh = ct_fresh.orders(obs, water, seen, memory)
    assert np.array_equal(orders_run1, orders_fresh), "不同新实例化对象对相同输入必须产生相同指令"

