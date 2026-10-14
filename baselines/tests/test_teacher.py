"""教师策略的记忆机制：被记住的敌方蚁穴或食物单元格如何对势场产生影响，以及何时不产生影响。
使用合成棋盘进行测试，因为这里关注的是规则逻辑而非真实对局。"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tb_baselines.teacher import FAR, Teacher, known  # noqa: E402


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
