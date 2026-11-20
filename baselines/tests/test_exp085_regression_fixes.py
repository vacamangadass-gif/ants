from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT.resolve()) not in sys.path:
    sys.path.insert(0, str(_ROOT.resolve()))

from tb_baselines.collect import collect
from tb_baselines.coordination import DELTAS, HOLD, compute_enemy_threat_penalty
from tb_baselines.dagger import (
    InformationAlignmentGate,
    StudentDriver,
    TeacherOpponentDriver,
    build_teacher_by_name,
    rollout_iteration,
)
from tb_baselines.mcmf_teacher import MCMFTeacher
from tb_baselines.nets import policy_params
from tb_baselines.teacher import TacticalLocalTeacher, TaskState
from tools.gen_procedural_maps import (
    MapSpec,
    build_board,
    canonical_board_hash,
    default_60_specs,
    generate_map_pool,
    load_excluded_hashes,
)
from tools.run_exp085_pipeline import BUCKET_SHARES, compute_bucket_quotas


def test_mcmf_no_own_hill_fortify_fallback_and_spawn_clear():
    # 验证全探索地图上无敌蚁威胁时绝不回退为 FORTIFY，且位于己方蚁穴上的蚂蚁强制走离绝不输出 HOLD
    rows, cols = 48, 64
    water = np.zeros((rows, cols), dtype=bool)
    seen = np.ones((rows, cols), dtype=bool)
    hills = [[10, 10, 0], [10, 30, 0], [38, 50, 1], [38, 20, 2]]
    mine = [[10, 10], [10, 30], [12, 12], [12, 32], [14, 14], [14, 34]]
    obs = {
        "size": [rows, cols],
        "mine": mine,
        "ids": list(range(len(mine))),
        "hills": hills,
        "foes": [],
        "food": [[20, 20]],
        "water": {"rle": [0, rows * cols]},
        "vis": {"rle": [1, rows * cols]},
    }
    teacher = MCMFTeacher(min_raid_ants=4, raid_ratio=0.5, adaptive_raid=True)
    orders = teacher.orders(obs, water, seen)
    tasks = teacher.get_last_tasks()

    assert sum(1 for t in tasks.values() if t == TaskState.FORTIFY) == 0
    # 位于己方两座蚁穴 (10,10) 与 (10,30) 上的蚂蚁绝不输出 HOLD
    assert int(orders[0]) != HOLD
    assert int(orders[1]) != HOLD

    # 当敌蚁逼近基地 (10,10) 时，位于基地上的蚂蚁同样必须走离基地出兵拦截，严禁原地 HOLD 堵死产卵位
    obs_threat = dict(obs, foes=[[10, 13, 1]])
    orders_threat = teacher.orders(obs_threat, water, seen)
    assert int(orders_threat[0]) != HOLD


def test_mcmf_no_turn_dead_code_raid_starvation_and_no_depth16_truncation():
    # 验证引擎真实观测（不含 "turn" 键）在 >=36 与多蚁丘棋盘上不会发生突击饥饿，且 max_depth=None 不被截断为 16
    rows, cols = 48, 48
    water = np.zeros((rows, cols), dtype=bool)
    seen = np.ones((rows, cols), dtype=bool)
    hills = [[5, 5, 0], [5, 25, 0], [40, 40, 1]]
    mine = [[6 + i // 4, 6 + i % 4] for i in range(10)]
    # 食物位于距离蚂蚁 24 格处（> 16 步），验证 max_depth=None 能正常跨过 16 步完成 MCMF 分配
    food = [[30, 8]]
    obs = {
        "size": [rows, cols],
        "mine": mine,
        "ids": list(range(len(mine))),
        "hills": hills,
        "foes": [],
        "food": food,
        "water": {"rle": [0, rows * cols]},
        "vis": {"rle": [1, rows * cols]},
    }
    teacher = MCMFTeacher(min_raid_ants=4, raid_ratio=0.5, max_depth=None, adaptive_raid=True)
    orders = teacher.orders(obs, water, seen)
    tasks = teacher.get_last_tasks()

    assault_cnt = sum(1 for t in tasks.values() if t == TaskState.ASSAULT)
    gather_cnt = sum(1 for t in tasks.values() if t == TaskState.GATHER)
    assert assault_cnt >= 2, f"期望派出至少 2 只突击蚂蚁，实际为 {assault_cnt}"
    assert gather_cnt == 1, f"距离 24 格的食物未被 MCMF 分配（疑似 16 步截断），实际 gather_cnt={gather_cnt}"
    assert len(orders) == len(mine)


def test_threat_penalty_only_penalizes_strictly_outnumbered_cells():
    # 验证 1v1 等势或 2v1 优势交战区不产生斥力惩罚（int_delay == 0），仅 1v2 劣势区产生斥力
    rows, cols = 24, 24
    passable = np.ones((rows, cols), dtype=bool)
    width_map = np.ones((rows, cols), dtype=np.int32)
    foes = [(12, 12)]

    # 1v1 等势: 1 只己方蚂蚁在 (12, 11)
    _, delay_1v1 = compute_enemy_threat_penalty(
        foe_positions=foes,
        friendly_positions=[(12, 11)],
        passable_mask=passable,
        rows=rows,
        cols=cols,
        width_map=width_map,
        attack_radius_sq=5,
        max_delay=4,
    )
    assert delay_1v1[12, 12] == 0, "1v1 等势交战中心即使在窄道上也不应被施加威胁斥力"

    # 0v1 劣势: 无己方支援时必须施加最大斥力
    _, delay_0v1 = compute_enemy_threat_penalty(
        foe_positions=foes,
        friendly_positions=[],
        passable_mask=passable,
        rows=rows,
        cols=cols,
        width_map=width_map,
        attack_radius_sq=5,
        max_delay=4,
    )
    assert delay_0v1[12, 12] == 4


def test_mcmf_stateless_across_matches_no_stall_counter_leak():
    # 验证未显式传入 stall_counters 时 MCMFTeacher 保持严格无状态，跨对局调用不污染 self.stall_counters
    rows, cols = 24, 24
    water = np.zeros((rows, cols), dtype=bool)
    seen = np.ones((rows, cols), dtype=bool)
    obs = {
        "size": [rows, cols],
        "mine": [[5, 5], [5, 6]],
        "ids": [7, 8],
        "hills": [[2, 2, 0], [20, 20, 1]],
        "foes": [],
        "food": [[8, 5]],
    }
    teacher = MCMFTeacher()
    out1 = teacher.orders(obs, water, seen)
    assert len(teacher.stall_counters) == 0

    # 插入另一场使用相同 ant ID 的拥堵对局
    obs_jam = {
        "size": [rows, cols],
        "mine": [[10, 10], [10, 11], [11, 10]],
        "ids": [7, 8, 9],
        "hills": [[2, 2, 0]],
        "foes": [],
        "food": [],
    }
    for _ in range(5):
        teacher.orders(obs_jam, water, seen)
    assert len(teacher.stall_counters) == 0

    out2 = teacher.orders(obs, water, seen)
    assert np.array_equal(out1, out2)


def test_tactical_local_teacher_own_hill_clear_and_focus_battle():
    # 验证 TacticalLocalTeacher:
    # (1) 在己方蚁丘 (1h/2h/3h) 上绝不输出 HOLD
    # (2) 面对 1v2 劣势陷阱时优先安全撤离/侧移且不输出 HOLD
    # (3) 拥有 2v1 局部优势时主动前压围杀
    rows, cols = 32, 32
    water = np.zeros((rows, cols), dtype=bool)
    seen = np.ones((rows, cols), dtype=bool)
    teacher = TacticalLocalTeacher(hill_lead=8, max_depth=None, defense_radius=10)

    # (1) 3 座己方蚁丘上的蚂蚁同时出生（全探索无迷雾前沿、无食物、无敌丘）
    obs_3h = {
        "size": [rows, cols],
        "mine": [[4, 4], [12, 12], [20, 20]],
        "ids": [0, 1, 2],
        "hills": [[4, 4, 0], [12, 12, 0], [20, 20, 0]],
        "foes": [],
        "food": [],
    }
    mv_3h = teacher.orders(obs_3h, water, seen)
    assert all(int(a) != HOLD for a in mv_3h), f"己方 3 座蚁丘上的蚂蚁严禁 HOLD: {mv_3h}"

    # (2) 1v2 劣势规避: 己方单兵位于 (10, 10)，北方 (7, 10) 与 (7, 11) 有 2 只敌蚁守住食物 (6, 10)
    obs_1v2 = {
        "size": [rows, cols],
        "mine": [[10, 10]],
        "ids": [0],
        "hills": [[25, 25, 0]],
        "foes": [[7, 10, 1], [7, 11, 1]],
        "food": [[6, 10]],
    }
    mv_1v2 = teacher.orders(obs_1v2, water, seen)
    assert int(mv_1v2[0]) != 0, "1v2 劣势下严禁向北 (0=N) 走入双敌火网送死"
    assert int(mv_1v2[0]) != HOLD, "1v2 劣势下应选择安全侧移/后撤而非原地 HOLD"

    # (3) 2v1 优势围杀: 己方 2 只蚂蚁位于 (10, 10) 与 (10, 11)，北方 (7, 10) 仅有 1 只孤立敌蚁
    obs_2v1 = {
        "size": [rows, cols],
        "mine": [[10, 10], [10, 11]],
        "ids": [0, 1],
        "hills": [[25, 25, 0]],
        "foes": [[7, 10, 1]],
        "food": [[6, 10]],
    }
    mv_2v1 = teacher.orders(obs_2v1, water, seen)
    assert int(mv_2v1[0]) == 0 and int(mv_2v1[1]) == 0, f"2v1 绝对优势下两只己方蚂蚁均应向北前压击杀: {mv_2v1}"


def test_procedural_map_generator_disjointness_and_engine_validation():
    # 验证程序化棋盘生成器满足严格哈希隔离与分层配额要求（使用 gitignored runs/ 临时目录并自动清理）
    scratch_dir = _ROOT / "runs" / "test_exp085_regression_scratch_maps"
    if scratch_dir.exists():
        shutil.rmtree(scratch_dir)
    scratch_dir.mkdir(parents=True, exist_ok=True)
    try:
        spec_a = MapSpec("tinyaa-maze-2p-1h", 24, 24, 2, 1, "maze", 10, 101)
        board_a = build_board(spec_a, seed_offset=0)
        hash_a = canonical_board_hash(board_a)

        excl_file = scratch_dir / "excluded.json"
        excl_file.write_text(json.dumps({"maps": [{"board": board_a}]}), encoding="utf-8")
        loaded = load_excluded_hashes(excl_file)
        assert hash_a in loaded

        specs = default_60_specs(base_seed=20261120)
        assert len(specs) == 60
        players_set = {s.players for s in specs}
        hills_set = {s.hills_per_seat for s in specs}
        assert players_set == {2, 4, 6}
        assert hills_set == {1, 2, 3}
        assert sum(1 for s in specs if s.rows != s.cols) >= 24

        # 验证分层配额计算严格满足 >=50% 2h/3h 与 >=45% 4p/6p
        quotas = dict(compute_bucket_quotas(1000))
        assert sum(quotas.values()) == 1000
        multiseat_share = (quotas["4p-1h"] + quotas["4p-multi"] + quotas["6p-all"]) / 1000.0
        multihill_share = (
            quotas["2p-multi"] + quotas["4p-multi"] + 0.55 * quotas["6p-all"] + 0.5 * quotas["basic-3p-8p"]
        ) / 1000.0
        assert multiseat_share >= 0.45
        assert multihill_share >= 0.50
    finally:
        if scratch_dir.exists():
            shutil.rmtree(scratch_dir)


def test_collect_and_dagger_tactical_teacher_and_ch40_student():
    # 端到端真实引擎测试：验证 collect(--teacher tactical)、StudentDriver(ch=40,b=3) 与混合对手 rollout_iteration
    scratch_dir = _ROOT / "runs" / "test_exp085_regression_scratch_dagger"
    if scratch_dir.exists():
        shutil.rmtree(scratch_dir)
    scratch_dir.mkdir(parents=True, exist_ok=True)
    try:
        driver = StudentDriver(weights_path=None, device="cpu", cls="micro", channels=40, blocks=3)
        assert policy_params(driver.model) == 48445

        d0_out = scratch_dir / "tactical_d0.jsonl.gz"
        stats_d0 = collect(
            out=d0_out,
            seat_turns=8,
            waves=1,
            matches_per_wave=1,
            max_turns=10,
            seed=7,
            maps="basic-tiny-2p",
            remembers=True,
            teacher="tactical",
            hill_lead=8,
            memory_mode="where",
        )
        assert stats_d0["rows"] == 8
        assert stats_d0["teacher"] == "tactical"

        tactical_t = build_teacher_by_name("tactical", hill_lead=8)
        gate = InformationAlignmentGate(teacher=tactical_t, hill_lead=8)
        opponents = [TeacherOpponentDriver(tactical_t, name="tactical_opp"), driver]
        delta_out = scratch_dir / "tactical_iter1.jsonl.gz"
        stats_iter = rollout_iteration(
            student_driver=driver,
            gate=gate,
            out_path=delta_out,
            seat_turns=6,
            k_iter=1,
            waves=1,
            matches_per_wave=2,
            max_turns=10,
            seed=11,
            maps="basic-tiny-2p",
            opponent_drivers=opponents,
            teacher_type="tactical",
        )
        assert stats_iter["rows"] == 6
        assert 0.0 <= stats_iter["disagreement_rate"] <= 1.0
    finally:
        if scratch_dir.exists():
            shutil.rmtree(scratch_dir)
