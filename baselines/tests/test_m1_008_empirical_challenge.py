from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tb_baselines import planes
from tb_baselines.benchmarks.run_exp080_scaling import (
    compute_collision_blocking_rate,
    compute_corridor_congestion_index,
    compute_corridor_mask,
    compute_corridor_stalling_rate,
    compute_effective_gathering_throughput,
    rle_from_mask,
    run_scaling_simulation,
)
from tb_baselines.teacher import CompoundTeacher, DELTAS, HOLD
from tb_baselines.telemetry import HoldReason


def test_audit_telemetry_jsonl_integrity_and_math():
    telemetry_path = Path("runs/exp080_scaling/telemetry.jsonl")
    assert telemetry_path.is_file(), f"遥测文件不存在: {telemetry_path}"

    records = []
    with open(telemetry_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line_str = line.strip()
            if not line_str:
                continue
            item = json.loads(line_str)
            records.append((line_num, item))

    assert len(records) == 135, f"预期 135 场运行记录，实际解析得到 {len(records)} 条"

    expected_scales = {20, 40, 60, 80, 100}
    expected_seeds = {90000, 90001, 90002}
    expected_maps = {
        "medium-crossroads-2p-1h",
        "medium-labyrinth-2p-1h",
        "medium-valley-2p-1h",
        "small-arena-2p-1h",
        "small-islands-2p-1h",
        "small-rooms-2p-1h",
        "tiny-colonnade-2p-1h",
        "tiny-corridor-2p-1h",
        "tiny-open-2p-1h",
    }

    seen_keys = set()
    for line_num, rec in records:
        scale = rec["scale"]
        seed = rec["seed"]
        map_id = rec["map_id"]
        turns = rec["turns"]

        assert scale in expected_scales, f"行 {line_num} 异常规模: {scale}"
        assert seed in expected_seeds, f"行 {line_num} 异常随机数种子: {seed}"
        assert map_id in expected_maps, f"行 {line_num} 异常地图标识符: {map_id}"
        assert turns == 100, f"行 {line_num} 回合数不等于 100: {turns}"

        key = (scale, map_id, seed)
        assert key not in seen_keys, f"发现重复的评测配置: {key}"
        seen_keys.add(key)

        stalled_corridor_turns = rec["stalled_corridor_turns"]
        corridor_cell_count = rec["corridor_cell_count"]
        total_corridor_ant_turns = rec["total_corridor_ant_turns"]
        total_forced_holds = rec["total_forced_holds"]
        total_ant_turns = rec["total_ant_turns"]
        total_food_gathered = rec["total_food_gathered"]

        assert total_ant_turns == scale * turns, f"行 {line_num} 总蚂蚁回合数不匹配"

        expected_congestion_index = compute_corridor_congestion_index(
            stalled_corridor_turns, turns, corridor_cell_count
        )
        assert pytest.approx(rec["corridor_congestion_index"], abs=1e-6) == expected_congestion_index

        expected_stalling_rate = compute_corridor_stalling_rate(
            stalled_corridor_turns, total_corridor_ant_turns
        )
        assert pytest.approx(rec["corridor_stalling_rate"], abs=1e-6) == expected_stalling_rate

        expected_gathering_throughput = compute_effective_gathering_throughput(
            total_food_gathered, turns, float(scale)
        )
        assert pytest.approx(rec["effective_gathering_throughput"], abs=1e-6) == expected_gathering_throughput

        expected_collision_blocking_rate = compute_collision_blocking_rate(
            total_forced_holds, total_ant_turns
        )
        assert pytest.approx(rec["collision_blocking_rate"], abs=1e-6) == expected_collision_blocking_rate
        assert pytest.approx(rec["internal_collision_blocking_rate"], abs=1e-6) == expected_collision_blocking_rate

        lat = rec["latency_ms"]
        assert lat["min"] <= lat["p50"] <= lat["p95"] <= lat["max"], f"行 {line_num} 延迟分位数排序不合法"


def test_corridor_scaling_trend_analysis():
    telemetry_path = Path("runs/exp080_scaling/telemetry.jsonl")
    records = []
    with open(telemetry_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    scales = [20, 40, 60, 80, 100]

    for map_id in ["medium-labyrinth-2p-1h", "tiny-corridor-2p-1h"]:
        map_records = [r for r in records if r["map_id"] == map_id]
        assert len(map_records) == len(scales) * 3

        avg_congestion_by_scale = {}
        avg_collision_by_scale = {}
        for s in scales:
            s_recs = [r for r in map_records if r["scale"] == s]
            avg_congestion = float(np.mean([r["corridor_congestion_index"] for r in s_recs]))
            avg_collision = float(np.mean([r["collision_blocking_rate"] for r in s_recs]))
            avg_congestion_by_scale[s] = avg_congestion
            avg_collision_by_scale[s] = avg_collision

        for i in range(len(scales) - 1):
            s_curr = scales[i]
            s_next = scales[i + 1]
            c_curr = avg_congestion_by_scale[s_curr]
            c_next = avg_congestion_by_scale[s_next]
            assert c_next >= c_curr, (
                f"地图 {map_id} 走廊拥塞指数未呈现随规模增长单调递增趋势: "
                f"规模 {s_curr} 为 {c_curr:.5f}, 规模 {s_next} 为 {c_next:.5f}"
            )


def test_empirical_invariants_scale_100_all_9_maps():
    maps_dir = Path("benchmarks/maps_2p")
    map_files = sorted(list(maps_dir.glob("*.json")))
    assert len(map_files) == 9, f"必须存在 9 张官方基准地图，当前发现 {len(map_files)} 张"

    scale = 100
    turns = 100
    seed = 90000

    for map_file in map_files:
        with open(map_file, "r", encoding="utf-8") as f:
            board_data = json.load(f)

        rows = int(board_data["rows"])
        cols = int(board_data["cols"])
        b = planes.Board(rows, cols)
        water = b.rle(board_data["water"]).astype(bool)

        land_cells = [
            (r, c) for r in range(rows) for c in range(cols) if not water[r, c]
        ]
        assert len(land_cells) >= scale, f"地图 {map_file.stem} 陆地单元格数不足以容纳规模 100"

        rng = np.random.default_rng(seed)
        chosen_indices = rng.choice(len(land_cells), size=scale, replace=False)
        ants = [list(land_cells[idx]) for idx in chosen_indices]
        ant_ids = list(range(scale))

        hills_raw = board_data.get("hills", [])
        hills = [[int(h[0]), int(h[1]), idx] for idx, h in enumerate(hills_raw)]

        food_target = max(scale // 2, 16)
        ant_set = {tuple(a) for a in ants}
        initial_food_candidates = [
            [int(f[0]), int(f[1])]
            for f in board_data.get("food", [])
            if not water[int(f[0]), int(f[1])] and (int(f[0]), int(f[1])) not in ant_set
        ]
        current_food = list(initial_food_candidates)
        if len(current_food) < food_target:
            occupied = ant_set | {tuple(f) for f in current_food}
            free_land = [p for p in land_cells if p not in occupied]
            if free_land:
                need_count = min(food_target - len(current_food), len(free_land))
                chosen_food_idx = rng.choice(len(free_land), size=need_count, replace=False)
                for f_idx in chosen_food_idx:
                    current_food.append([free_land[f_idx][0], free_land[f_idx][1]])

        teacher = CompoundTeacher(min_raid_ants=4, raid_ratio=0.5)
        memory: dict = {}
        stall_counters: dict[int, int] = {}

        for t in range(turns):
            assert len(ants) == scale, f"回合 {t} 存活蚂蚁数量偏离目标规模 {scale}，实际数量为 {len(ants)}"
            assert len(ant_ids) == scale, f"回合 {t} 蚂蚁编号列表长度偏离目标规模 {scale}"

            unique_positions = set(tuple(p) for p in ants)
            assert len(unique_positions) == scale, (
                f"地图 {map_file.stem} 回合 {t} 出现友方重叠，唯一坐标数 {len(unique_positions)} 小于规模 {scale}"
            )

            mine_plane = b.scatter(ants)
            vis_plane = b.dilate(mine_plane, planes.VIEW_RADIUS2).astype(bool)
            seen = water | vis_plane

            obs = {
                "size": [rows, cols],
                "mine": ants,
                "ids": ant_ids,
                "foes": [],
                "food": [list(f) for f in current_food],
                "hills": hills,
                "water": {"rle": rle_from_mask(water)},
                "vis": {"rle": rle_from_mask(vis_plane)},
            }

            memory = planes.remember(memory, obs, mode="where")
            orders, telemetry_records = teacher.orders_with_telemetry(
                obs, water, seen, memory, stall_counters=stall_counters
            )

            dests: list[tuple[int, int]] = []
            for (ar, ac), a in zip(ants, orders):
                act_int = int(a)
                if act_int == HOLD:
                    dests.append((ar, ac))
                else:
                    dr, dc = DELTAS[act_int]
                    dests.append(((ar + dr) % rows, (ac + dc) % cols))

            assert len(dests) == scale, f"回合 {t} 目的地列表数量异常: {len(dests)}"
            assert len(set(dests)) == scale, (
                f"地图 {map_file.stem} 回合 {t} 检测到友方蚂蚁内部碰撞！目的地去重后数量为 {len(set(dests))}"
            )

            food_set = {tuple(f) for f in current_food}
            for idx_ant, (dr_c, dc_c) in enumerate(dests):
                assert not water[dr_c, dc_c], (
                    f"地图 {map_file.stem} 回合 {t} 蚂蚁 {idx_ant} 走入水域单元格 ({dr_c}, {dc_c})"
                )
                assert (dr_c, dc_c) not in food_set, (
                    f"地图 {map_file.stem} 回合 {t} 蚂蚁 {idx_ant} 走入食物单元格 ({dr_c}, {dc_c})"
                )

            ants = [list(p) for p in dests]

            remaining_food: list[list[int]] = []
            for fr, fc in current_food:
                gathered = False
                for ar, ac in ants:
                    dr = min(abs(ar - fr), rows - abs(ar - fr))
                    dc = min(abs(ac - fc), cols - abs(ac - fc))
                    if dr * dr + dc * dc <= 1:
                        gathered = True
                        break
                if not gathered:
                    remaining_food.append([fr, fc])
            current_food = remaining_food

            if len(current_food) < food_target:
                needed = food_target - len(current_food)
                occupied = {tuple(a) for a in ants} | {tuple(f) for f in current_food}
                free_land = [p for p in land_cells if p not in occupied]
                if free_land:
                    rep_count = min(needed, len(free_land))
                    chosen_rep_idx = rng.choice(len(free_land), size=rep_count, replace=False)
                    for r_idx in chosen_rep_idx:
                        current_food.append([free_land[r_idx][0], free_land[r_idx][1]])


def test_empirical_constant_scale_maintenance_across_all_scales():
    map_file = Path("benchmarks/maps_2p/tiny-corridor-2p-1h.json")
    with open(map_file, "r", encoding="utf-8") as f:
        board_data = json.load(f)

    for scale in [20, 40, 60, 80, 100]:
        res = run_scaling_simulation(
            board_data=board_data,
            scale=scale,
            turns=30,
            seed=90001,
            teacher_type="compound",
        )
        assert res["scale"] == scale
        assert res["total_ant_turns"] == scale * 30


def test_audit_telemetry_physical_reproducibility():
    telemetry_path = Path("runs/exp080_scaling/telemetry.jsonl")
    records = []
    with open(telemetry_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    # 抽样验证 5 种不同规模与不同地图的物理模拟确定性重现
    sample_targets = [
        (20, "tiny-corridor-2p-1h", 90000),
        (40, "medium-labyrinth-2p-1h", 90001),
        (60, "small-arena-2p-1h", 90002),
        (80, "tiny-colonnade-2p-1h", 90000),
        (100, "medium-crossroads-2p-1h", 90001),
    ]

    for scale, map_id, seed in sample_targets:
        match_recs = [
            r for r in records
            if r["scale"] == scale and r["map_id"] == map_id and r["seed"] == seed
        ]
        assert len(match_recs) == 1, f"未找到唯一对应的遥测记录: scale={scale}, map={map_id}, seed={seed}"
        logged = match_recs[0]

        map_path = Path(logged["map_path"])
        with open(map_path, "r", encoding="utf-8") as f:
            board_data = json.load(f)

        rerun = run_scaling_simulation(
            board_data=board_data,
            scale=scale,
            turns=logged["turns"],
            seed=seed,
            teacher_type=logged["teacher_type"],
        )

        assert rerun["total_food_gathered"] == logged["total_food_gathered"], (
            f"重跑采食总量不一致: 重跑={rerun['total_food_gathered']}, 记录={logged['total_food_gathered']}"
        )
        assert rerun["stalled_corridor_turns"] == logged["stalled_corridor_turns"], (
            f"重跑走廊停滞回合数不一致: 重跑={rerun['stalled_corridor_turns']}, 记录={logged['stalled_corridor_turns']}"
        )
        assert rerun["total_corridor_ant_turns"] == logged["total_corridor_ant_turns"], (
            f"重跑走廊总蚂蚁回合数不一致: 重跑={rerun['total_corridor_ant_turns']}, 记录={logged['total_corridor_ant_turns']}"
        )
        assert rerun["total_forced_holds"] == logged["total_forced_holds"], (
            f"重跑强制驻留数不一致: 重跑={rerun['total_forced_holds']}, 记录={logged['total_forced_holds']}"
        )
        assert pytest.approx(rerun["corridor_congestion_index"], abs=1e-6) == logged["corridor_congestion_index"]
        assert pytest.approx(rerun["corridor_stalling_rate"], abs=1e-6) == logged["corridor_stalling_rate"]
        assert pytest.approx(rerun["effective_gathering_throughput"], abs=1e-6) == logged["effective_gathering_throughput"]
        assert pytest.approx(rerun["collision_blocking_rate"], abs=1e-6) == logged["collision_blocking_rate"]


def test_adversarial_packed_corridor_arbitration():
    # 构造极窄 1 格宽单行走廊，6 只蚂蚁沿走廊紧挨排列，测试终点冲突仲裁保底机制
    rows, cols = 8, 8
    water = np.ones((rows, cols), dtype=bool)
    water[3, :] = False  # 第 3 行为唯一水平走廊

    b = planes.Board(rows, cols)
    ants = [[3, c] for c in range(6)]
    ant_ids = list(range(6))
    food = [[3, 6], [3, 7]]
    hills = [[3, 0, 0]]

    teacher = CompoundTeacher(min_raid_ants=4, raid_ratio=0.5)
    memory = {}
    stall_counters: dict[int, int] = {}

    for t in range(25):
        assert len(ants) == 6
        assert len(set(tuple(p) for p in ants)) == 6

        mine_plane = b.scatter(ants)
        vis_plane = b.dilate(mine_plane, planes.VIEW_RADIUS2).astype(bool)
        seen = water | vis_plane

        obs = {
            "size": [rows, cols],
            "mine": ants,
            "ids": ant_ids,
            "foes": [],
            "food": [list(f) for f in food],
            "hills": hills,
            "water": {"rle": rle_from_mask(water)},
            "vis": {"rle": rle_from_mask(vis_plane)},
        }

        memory = planes.remember(memory, obs, mode="where")
        orders, _ = teacher.orders_with_telemetry(obs, water, seen, memory, stall_counters=stall_counters)

        dests = []
        for (ar, ac), a in zip(ants, orders):
            act_int = int(a)
            if act_int == HOLD:
                dests.append((ar, ac))
            else:
                dr, dc = DELTAS[act_int]
                dests.append(((ar + dr) % rows, (ac + dc) % cols))

        # 验证物理不变量：绝无重合、绝不踏入水域、绝不踩上食物
        assert len(dests) == 6
        assert len(set(dests)) == 6, f"极端拥挤走廊第 {t} 回合发生友方碰撞: {dests}"
        food_set = {tuple(f) for f in food}
        for dr_c, dc_c in dests:
            assert not water[dr_c, dc_c], f"走廊测试第 {t} 回合蚂蚁进入水域: ({dr_c}, {dc_c})"
            assert (dr_c, dc_c) not in food_set, f"走廊测试第 {t} 回合蚂蚁踏入食物格: ({dr_c}, {dc_c})"

        ants = [list(p) for p in dests]


def test_adversarial_4way_surround_food():
    # 4 只蚂蚁紧密包围 1 个食物单元格，处于全部相邻采食位上
    rows, cols = 8, 8
    water = np.zeros((rows, cols), dtype=bool)
    b = planes.Board(rows, cols)

    food_center = (4, 4)
    ants = [
        [3, 4],  # 北
        [5, 4],  # 南
        [4, 3],  # 西
        [4, 5],  # 东
    ]
    ant_ids = [0, 1, 2, 3]
    food = [list(food_center)]
    hills = [[0, 0, 0]]

    teacher = CompoundTeacher(min_raid_ants=2, raid_ratio=0.5)
    memory = {}
    stall_counters: dict[int, int] = {}

    for t in range(20):
        assert len(ants) == 4
        assert len(set(tuple(p) for p in ants)) == 4

        mine_plane = b.scatter(ants)
        vis_plane = b.dilate(mine_plane, planes.VIEW_RADIUS2).astype(bool)
        seen = water | vis_plane

        obs = {
            "size": [rows, cols],
            "mine": ants,
            "ids": ant_ids,
            "foes": [],
            "food": [list(f) for f in food],
            "hills": hills,
            "water": {"rle": rle_from_mask(water)},
            "vis": {"rle": rle_from_mask(vis_plane)},
        }

        memory = planes.remember(memory, obs, mode="where")
        orders, _ = teacher.orders_with_telemetry(obs, water, seen, memory, stall_counters=stall_counters)

        dests = []
        for (ar, ac), a in zip(ants, orders):
            act_int = int(a)
            if act_int == HOLD:
                dests.append((ar, ac))
            else:
                dr, dc = DELTAS[act_int]
                dests.append(((ar + dr) % rows, (ac + dc) % cols))

        assert len(set(dests)) == 4, f"包围食物对抗场景第 {t} 回合碰撞: {dests}"
        for dr_c, dc_c in dests:
            assert not water[dr_c, dc_c]
            assert (dr_c, dc_c) != food_center, f"包围食物对抗场景第 {t} 回合蚂蚁踩入食物格: ({dr_c}, {dc_c})"

        ants = [list(p) for p in dests]


def test_adversarial_extreme_density_grid():
    # 6x6 环面网格放置 28 只蚂蚁（陆地密度 77.8%），食物 4 处，推演 25 回合
    rows, cols = 6, 6
    water = np.zeros((rows, cols), dtype=bool)
    b = planes.Board(rows, cols)

    scale = 28
    rng = np.random.default_rng(20261103)
    all_cells = [(r, c) for r in range(rows) for c in range(cols)]
    chosen = rng.choice(len(all_cells), size=scale, replace=False)
    ants = [list(all_cells[idx]) for idx in chosen]
    ant_ids = list(range(scale))

    remaining_cells = [p for p in all_cells if list(p) not in ants]
    food_chosen = rng.choice(len(remaining_cells), size=4, replace=False)
    food = [list(remaining_cells[idx]) for idx in food_chosen]
    hills = [[0, 0, 0]]

    teacher = CompoundTeacher(min_raid_ants=4, raid_ratio=0.5)
    memory = {}
    stall_counters: dict[int, int] = {}

    for t in range(25):
        assert len(ants) == scale
        assert len(set(tuple(p) for p in ants)) == scale

        mine_plane = b.scatter(ants)
        vis_plane = b.dilate(mine_plane, planes.VIEW_RADIUS2).astype(bool)
        seen = water | vis_plane

        obs = {
            "size": [rows, cols],
            "mine": ants,
            "ids": ant_ids,
            "foes": [],
            "food": [list(f) for f in food],
            "hills": hills,
            "water": {"rle": rle_from_mask(water)},
            "vis": {"rle": rle_from_mask(vis_plane)},
        }

        memory = planes.remember(memory, obs, mode="where")
        orders, _ = teacher.orders_with_telemetry(obs, water, seen, memory, stall_counters=stall_counters)

        dests = []
        for (ar, ac), a in zip(ants, orders):
            act_int = int(a)
            if act_int == HOLD:
                dests.append((ar, ac))
            else:
                dr, dc = DELTAS[act_int]
                dests.append(((ar + dr) % rows, (ac + dc) % cols))

        assert len(set(dests)) == scale, f"高密度极限压测第 {t} 回合友方碰撞"
        food_set = {tuple(f) for f in food}
        for dr_c, dc_c in dests:
            assert not water[dr_c, dc_c]
            assert (dr_c, dc_c) not in food_set, f"高密度极限压测第 {t} 回合踩入食物格"

        ants = [list(p) for p in dests]


def test_detailed_chokepoint_scaling_summary():
    telemetry_path = Path("runs/exp080_scaling/telemetry.jsonl")
    records = []
    with open(telemetry_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    scales = [20, 40, 60, 80, 100]
    chokepoints = ["tiny-corridor-2p-1h", "medium-labyrinth-2p-1h"]

    for map_id in chokepoints:
        recs_map = [r for r in records if r["map_id"] == map_id]
        print(f"\n=== 地图指标详报: {map_id} ===")
        print("规模 | 走廊拥塞指数 | 走廊停滞比率(%) | 采食吞吐率 | 碰撞阻挡率(%) | 平均耗时(ms)")
        for s in scales:
            s_recs = [r for r in recs_map if r["scale"] == s]
            c_idx = float(np.mean([r["corridor_congestion_index"] for r in s_recs]))
            c_stall = float(np.mean([r["corridor_stalling_rate"] for r in s_recs])) * 100.0
            g_rate = float(np.mean([r["effective_gathering_throughput"] for r in s_recs]))
            b_rate = float(np.mean([r["collision_blocking_rate"] for r in s_recs])) * 100.0
            lat = float(np.mean([r["mean_latency_ms"] for r in s_recs]))
            print(f"{s:4d} | {c_idx:12.4f} | {c_stall:14.2f}% | {g_rate:10.2f} | {b_rate:12.2f}% | {lat:11.2f}")


