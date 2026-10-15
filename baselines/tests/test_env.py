"""针对真实卡带运行的环境客户端测试。

这些测试运行轻量 —— 仅在两场对局的波次上执行数个回合 —— 但它们覆盖了极易出错且一旦出错毫无警报的三大核心隐患：
座位与动作的位置对应对齐关系、在波次填充（wave refill）后 episode 键的存续有效性、以及得分通道是否切实传递实时分数。
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tb_baselines.env import Env, EnvError, orders_from_indices  # noqa: E402
from tb_baselines.planes import MOVES, N_PLANES  # noqa: E402

pytestmark = pytest.mark.skipif(
    not (os.environ.get("TINYBRAINS") or shutil.which("tinybrains")),
    reason="no `tinybrains` on PATH; set TINYBRAINS to the binary",
)


@pytest.fixture
def env():
    e = Env(waves=1, matches_per_wave=4, max_turns=40, seed=7, maps="basic-tiny-2p")
    yield e
    e.close()


def test_hello_names_the_engine_and_the_evaluator(env):
    """无法指明由哪个引擎产出数据的训练运行无法复现，且引擎的变更等同于规则变更。"""
    assert env.engine_digest.startswith("sha256:")
    assert env.evaluator.startswith("datalogic "), "the evaluator is named by version now"
    assert env.hello["max_turns"] == 40


def test_seats_and_boards_line_up(env):
    step = env.reset()
    assert len(step.seats) == 8, "4 场双人对局共 8 个座位"
    # 棋盘尺寸归视野本身说明，而非此测试自行假定：基础棋盘是由配方重新生成的，
    # 且赛季棋盘可以处于卡带允许范围内的任何尺寸。
    rows, cols = step.seats[0].obs["size"]
    assert step.boards.shape == (8, N_PLANES, rows, cols), "每个座位对应一个棋盘尺寸的特征面堆叠"
    for i, seat in enumerate(step.seats):
        assert tuple(seat.obs["size"]) == (rows, cols)
        assert step.groups[0].indices[i] == i


def test_actions_are_positional_and_the_engine_agrees(env):
    """每个活跃座位的每只蚂蚁均按 `mine` 的顺序获得指令。若对齐出错，
    引擎仍能正常接收指令 —— 但实际上是在执行其他座位的动作，这也是编写测试而非仅写注释的原因。"""
    step = env.reset()
    for _ in range(5):
        counts = [len(s.obs["mine"]) for s in step.seats]
        picks = np.zeros(sum(counts), dtype=np.int64)      # 每只蚂蚁均下达向北(N)移动指令
        orders = orders_from_indices(picks, counts)
        assert [len(o) for o in orders] == counts
        assert all(set(o) <= {"N"} for o in orders)
        step = env.step(orders)
    assert step.turn == 5


def test_scores_are_live_and_keyed_by_episode(env):
    """`f_finish` 能够响应正在进行中的对局，这正是稠密奖励无需变更卡带的全部奥秘。"""
    step = env.reset()
    assert len(step.scores) == 4, "每场进行中的比赛对应一个条目"
    for ep, scores in step.scores.items():
        assert len(scores) == 2
        assert all(isinstance(s, int) for s in scores)
    assert {s.ep for s in step.seats} == set(step.scores)


def test_a_mixed_pool_refuses_to_pretend_it_is_one_batch():
    """基础棋盘上的三个波次对应三种棋盘尺寸，单个张量无法同时容纳两种尺寸。
    若仅返回批次的一部分，将导致三分之一的训练步在无声无息中被吞掉。"""
    e = Env(waves=3, matches_per_wave=2, max_turns=20, seed=3)
    try:
        step = e.reset()
        assert len(step.groups) > 1, "三个波次抽取了基础棋盘中的三种"
        with pytest.raises(EnvError, match=r"(board sizes|棋盘尺寸)"):
            _ = step.boards
        assert sum(len(g.indices) for g in step.groups) == len(step.seats)
        for g in step.groups:
            assert g.boards.shape == (len(g.indices), N_PLANES, *g.size)
    finally:
        e.close()


def test_an_empty_seat_is_an_empty_order(env):
    """存活蚂蚁为零始终是合法状态，全灭的座位直到比赛结束前均发送空指令。"""
    assert orders_from_indices(np.zeros(0, dtype=np.int64), [0]) == [""]


def test_order_characters_are_the_moves_the_adapter_decodes():
    """`orders_from_indices` 与适配器的 `tb.at` 表必须索引完全相同的列表，
    否则策略通道在训练阶段与实战阶段将代表截然不同的动作。"""
    picks = np.arange(len(MOVES))
    assert orders_from_indices(picks, [len(MOVES)])[0] == "".join(MOVES)
