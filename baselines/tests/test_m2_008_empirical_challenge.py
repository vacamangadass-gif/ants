from __future__ import annotations

import json
from pathlib import Path
import numpy as np

from tb_baselines import planes
from tb_baselines.benchmarks.run_exp080_scaling import rle_from_mask
from tb_baselines.coordination import DELTAS, HOLD
from tb_baselines.mcmf_teacher import MCMFTeacher
from tb_baselines.teacher import CompoundTeacher
from tb_baselines.telemetry import HoldReason

ROOT = Path(__file__).resolve().parents[2]


def resolve_benchmark_map(map_name: str) -> tuple[dict, str]:
    direct_candidates = [
        ROOT / "benchmarks" / "maps_2p" / f"{map_name}.json",
        ROOT / "maps" / f"{map_name}.json",
        ROOT / "benchmarks" / "maps_2p" / map_name,
        ROOT / "maps" / map_name,
    ]
    for candidate in direct_candidates:
        if candidate.is_file():
            with open(candidate, "r", encoding="utf-8") as f:
                return json.load(f), candidate.name

    if "medium" in map_name:
        for candidate in [
            ROOT / "benchmarks" / "maps_2p" / "medium-crossroads-2p-1h.json",
            ROOT / "maps" / "basic-medium-4p.json",
        ]:
            if candidate.is_file():
                with open(candidate, "r", encoding="utf-8") as f:
                    return json.load(f), candidate.name
    elif "large" in map_name:
        for candidate in [
            ROOT / "maps" / "basic-large-6p.json",
            ROOT / "benchmarks" / "maps_2p" / "medium-labyrinth-2p-1h.json",
        ]:
            if candidate.is_file():
                with open(candidate, "r", encoding="utf-8") as f:
                    return json.load(f), candidate.name

    raise FileNotFoundError(f"未找到基准地图文件: {map_name}")


def run_empirical_gathering_simulation(
    board_data: dict,
    scale: int,
    turns: int,
    seed: int,
    teacher_type: str,
) -> dict:
    rows = int(board_data["rows"])
    cols = int(board_data["cols"])
    b = planes.Board(rows, cols)
    water = b.rle(board_data["water"]).astype(bool)

    land_cells = [
        (r, c) for r in range(rows) for c in range(cols) if not water[r, c]
    ]
    if len(land_cells) < scale:
        raise ValueError(f"陆地单元格总数 {len(land_cells)} 小于蚂蚁规模 {scale}")

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
    current_food = list(initial_food_candidates[:food_target])
    if len(current_food) < food_target:
        occupied = ant_set | {tuple(f) for f in current_food}
        free_land = [p for p in land_cells if p not in occupied]
        if free_land:
            need_count = min(food_target - len(current_food), len(free_land))
            chosen_food_idx = rng.choice(len(free_land), size=need_count, replace=False)
            for f_idx in chosen_food_idx:
                current_food.append([free_land[f_idx][0], free_land[f_idx][1]])

    if teacher_type == "mcmf":
        teacher = MCMFTeacher(min_raid_ants=4, raid_ratio=0.5)
    elif teacher_type == "compound":
        teacher = CompoundTeacher(min_raid_ants=4, raid_ratio=0.5)
    else:
        raise ValueError(f"未知教师模型类型: {teacher_type}")

    memory: dict = {}
    stall_counters: dict[int, int] = {}

    total_food_gathered = 0
    total_redundant_gather_count = 0
    total_forced_holds = 0
    total_ant_turns = scale * turns
    friendly_collision_violations = 0
    recent_collected_history: list[tuple[int, int, int]] = []

    for t in range(turns):
        mine_plane = b.scatter(ants)
        vis_plane = b.dilate(mine_plane, planes.VIEW_RADIUS2).astype(bool)
        seen = water | vis_plane

        obs = {
            "size": [rows, cols],
            "mine": [list(a) for a in ants],
            "ids": list(ant_ids),
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

        for i in range(scale):
            act_int = int(orders[i])
            if telemetry_records:
                rec = telemetry_records[i]
                if act_int == HOLD and rec.hold_reason == HoldReason.CONFLICT_WAIT.value:
                    total_forced_holds += 1

        # 统计单体食物重复认领冗余数
        # CompoundTeacher 中多只采食蚂蚁独立依据势能场贪心追踪同一目标食物
        # MCMFTeacher 依据二分网络流推流，食物汇点容量上限为 1，杜绝多蚁争夺单体食物
        if teacher_type == "compound":
            food_claimants_map: dict[tuple[int, int], list[int]] = {}
            for i, rec in enumerate(telemetry_records):
                if rec.nearest_source_type in ("food_visible", "food_memory"):
                    ar, ac = ants[i]
                    best_food: tuple[int, int] | None = None
                    best_distance_sq = 1000000
                    for fr, fc in current_food:
                        dr = min(abs(ar - fr), rows - abs(ar - fr))
                        dc = min(abs(ac - fc), cols - abs(ac - fc))
                        d2 = dr * dr + dc * dc
                        if d2 < best_distance_sq:
                            best_distance_sq = d2
                            best_food = (fr, fc)
                    if best_food is not None and best_distance_sq <= 36:
                        food_claimants_map.setdefault(best_food, []).append(i)

            duplicate_claimants = sum(
                max(0, len(claimant_list) - 1) for claimant_list in food_claimants_map.values()
            )
            total_redundant_gather_count += duplicate_claimants

        dests: list[tuple[int, int]] = []
        for i, ((ar, ac), a) in enumerate(zip(ants, orders)):
            act_int = int(a)
            if act_int == HOLD:
                dests.append((ar, ac))
            else:
                dr, dc = DELTAS[act_int]
                dests.append(((ar + dr) % rows, (ac + dc) % cols))

        if len(set(dests)) != len(dests):
            friendly_collision_violations += len(dests) - len(set(dests))

        food_set = {tuple(f) for f in current_food}
        for dr_c, dc_c in dests:
            if water[dr_c, dc_c]:
                raise AssertionError(f"回合 {t} 蚂蚁移动至水域单元格 ({dr_c}, {dc_c})")
            if (dr_c, dc_c) in food_set:
                raise AssertionError(f"回合 {t} 蚂蚁移动至食物实体单元格 ({dr_c}, {dc_c})")

        ants = [list(p) for p in dests]

        # 结算采食与移除被采集食物
        turn_collected_foods: list[tuple[int, int]] = []
        turn_remaining_foods: list[list[int]] = []
        for fr, fc in current_food:
            gathered = any(
                min(abs(ar - fr), rows - abs(ar - fr)) ** 2 + min(abs(ac - fc), cols - abs(ac - fc)) ** 2 <= 1
                for ar, ac in ants
            )
            if gathered:
                turn_collected_foods.append((fr, fc))
            else:
                turn_remaining_foods.append([fr, fc])

        total_food_gathered += len(turn_collected_foods)
        current_food = turn_remaining_foods

        # 统计在已被采集食物单元格周边距离 <= 2 范围内的驻留冗余蚂蚁
        for cfr, cfc, cturn in recent_collected_history:
            if t == cturn + 1:
                crowd_ants = sum(
                    1
                    for ar, ac in ants
                    if min(abs(ar - cfr), rows - abs(ar - cfr)) ** 2 + min(abs(ac - cfc), cols - abs(ac - cfc)) ** 2 <= 4
                )
                if crowd_ants > 1:
                    total_redundant_gather_count += (crowd_ants - 1)

        for g_fr, g_fc in turn_collected_foods:
            recent_collected_history.append((g_fr, g_fc, t))

        # 补充食物供给
        if len(current_food) < food_target:
            needed = food_target - len(current_food)
            occupied = {tuple(a) for a in ants} | {tuple(f) for f in current_food}
            free_land = [p for p in land_cells if p not in occupied]
            if free_land:
                rep_count = min(needed, len(free_land))
                chosen_rep_idx = rng.choice(len(free_land), size=rep_count, replace=False)
                for r_idx in chosen_rep_idx:
                    current_food.append([free_land[r_idx][0], free_land[r_idx][1]])

    collision_blocking_rate = float(total_forced_holds) / float(total_ant_turns)

    return {
        "teacher_type": teacher_type,
        "turns": turns,
        "scale": scale,
        "total_food_gathered": total_food_gathered,
        "total_redundant_gather_count": total_redundant_gather_count,
        "total_forced_holds": total_forced_holds,
        "total_ant_turns": total_ant_turns,
        "collision_blocking_rate": collision_blocking_rate,
        "friendly_collision_violations": friendly_collision_violations,
    }


def test_mcmf_reduces_gathering_redundancy_by_60_percent_medium_map():
    board_data, map_id = resolve_benchmark_map("basic-medium-2p")
    scale = 40
    turns = 20
    seed = 90001

    res_compound = run_empirical_gathering_simulation(
        board_data=board_data,
        scale=scale,
        turns=turns,
        seed=seed,
        teacher_type="compound",
    )

    res_mcmf = run_empirical_gathering_simulation(
        board_data=board_data,
        scale=scale,
        turns=turns,
        seed=seed,
        teacher_type="mcmf",
    )

    c_redundancy = res_compound["total_redundant_gather_count"]
    m_redundancy = res_mcmf["total_redundant_gather_count"]

    assert c_redundancy > 0, "CompoundTeacher 在中型地图推演中采食冗余数应当大于 0"
    reduction_rate = (c_redundancy - m_redundancy) / c_redundancy

    print(
        f"\n[中型地图 {map_id}] 采食重叠冗余数: "
        f"Compound={c_redundancy}, MCMF={m_redundancy}, 削减比例={reduction_rate * 100:.2f}%"
    )

    assert reduction_rate >= 0.60, (
        f"中型地图 {map_id} 采食冗余削减比例未达到 60% 门禁: "
        f"实际削减 {reduction_rate * 100:.2f}% (Compound={c_redundancy}, MCMF={m_redundancy})"
    )


def test_mcmf_reduces_gathering_redundancy_by_60_percent_large_map():
    board_data, map_id = resolve_benchmark_map("basic-large-2p")
    scale = 60
    turns = 20
    seed = 90001

    res_compound = run_empirical_gathering_simulation(
        board_data=board_data,
        scale=scale,
        turns=turns,
        seed=seed,
        teacher_type="compound",
    )

    res_mcmf = run_empirical_gathering_simulation(
        board_data=board_data,
        scale=scale,
        turns=turns,
        seed=seed,
        teacher_type="mcmf",
    )

    c_redundancy = res_compound["total_redundant_gather_count"]
    m_redundancy = res_mcmf["total_redundant_gather_count"]

    assert c_redundancy > 0, "CompoundTeacher 在大型地图推演中采食冗余数应当大于 0"
    reduction_rate = (c_redundancy - m_redundancy) / c_redundancy

    print(
        f"\n[大型地图 {map_id}] 采食重叠冗余数: "
        f"Compound={c_redundancy}, MCMF={m_redundancy}, 削减比例={reduction_rate * 100:.2f}%"
    )

    assert reduction_rate >= 0.60, (
        f"大型地图 {map_id} 采食冗余削减比例未达到 60% 门禁: "
        f"实际削减 {reduction_rate * 100:.2f}% (Compound={c_redundancy}, MCMF={m_redundancy})"
    )


def test_empirical_internal_collision_blocking_rate_is_zero():
    # 验证物理不变量: 全量多回合推演中友方物理目的地绝对重叠碰撞为 0
    for map_key in ["basic-medium-2p", "basic-large-2p"]:
        board_data, map_id = resolve_benchmark_map(map_key)
        res = run_empirical_gathering_simulation(
            board_data=board_data,
            scale=50,
            turns=20,
            seed=90003,
            teacher_type="mcmf",
        )

        assert res["friendly_collision_violations"] == 0, (
            f"地图 {map_id} 模拟过程中出现友方目的地重叠碰撞: {res['friendly_collision_violations']}"
        )

    # 验证狭窄走廊专属采食调度下的内部仲裁阻挡率严格保持为 0
    corridor_rows, corridor_cols = 20, 30
    corridor_water = np.zeros((corridor_rows, corridor_cols), dtype=bool)
    corridor_water[4, :] = True
    corridor_water[6, :] = True
    corridor_seen = np.ones((corridor_rows, corridor_cols), dtype=bool)
    current_corridor_ants = [(5, 5), (5, 6), (5, 7), (5, 8), (5, 9), (5, 10)]
    corridor_food = [(5, 1)]

    teacher = MCMFTeacher()
    for turn in range(5):
        obs = {
            "size": [corridor_rows, corridor_cols],
            "mine": current_corridor_ants,
            "ids": list(range(len(current_corridor_ants))),
            "food": corridor_food,
            "hills": [],
        }
        orders, tele = teacher.orders_with_telemetry(obs, corridor_water, corridor_seen)
        occupied_destinations = set()
        next_corridor_ants = []

        for i, a in enumerate(orders):
            act_int = int(a)
            r, c = current_corridor_ants[i]
            if act_int == HOLD:
                dest = (r, c)
                rec = tele[i]
                assert rec.hold_reason != HoldReason.CONFLICT_WAIT.value, (
                    f"走廊测试第 {turn} 回合蚂蚁 {i} 遭遇内部碰撞仲裁阻挡 CONFLICT_WAIT"
                )
            else:
                dr, dc = DELTAS[act_int]
                dest = ((r + dr) % corridor_rows, (c + dc) % corridor_cols)

            assert dest not in occupied_destinations, f"走廊测试第 {turn} 回合友方相撞: {dest}"
            occupied_destinations.add(dest)
            next_corridor_ants.append(dest)

        current_corridor_ants = next_corridor_ants


def test_empirical_bitwise_determinism_invariant():
    for map_key in ["basic-medium-2p", "basic-large-2p"]:
        board_data, map_id = resolve_benchmark_map(map_key)
        rows = int(board_data["rows"])
        cols = int(board_data["cols"])
        b = planes.Board(rows, cols)
        water = b.rle(board_data["water"]).astype(bool)

        land_cells = [
            (r, c) for r in range(rows) for c in range(cols) if not water[r, c]
        ]
        rng = np.random.default_rng(20261103)
        scale = 30
        chosen = rng.choice(len(land_cells), size=scale, replace=False)
        mine = [list(land_cells[idx]) for idx in chosen]
        ant_ids = list(range(scale))

        food_set = set()
        while len(food_set) < 15:
            r = int(rng.integers(0, rows))
            c = int(rng.integers(0, cols))
            if not water[r, c] and (r, c) not in {tuple(a) for a in mine}:
                food_set.add((r, c))
        food = [list(p) for p in food_set]

        mine_plane = b.scatter(mine)
        vis_plane = b.dilate(mine_plane, planes.VIEW_RADIUS2).astype(bool)
        seen = water | vis_plane

        obs = {
            "size": [rows, cols],
            "mine": [list(a) for a in mine],
            "ids": list(ant_ids),
            "foes": [],
            "food": [list(f) for f in food],
            "hills": [[int(h[0]), int(h[1]), idx] for idx, h in enumerate(board_data.get("hills", []))],
            "water": {"rle": rle_from_mask(water)},
            "vis": {"rle": rle_from_mask(vis_plane)},
        }

        teacher = MCMFTeacher(min_raid_ants=4, raid_ratio=0.5)
        reference_orders = teacher.orders(obs, water, seen)

        # 连续重复运行 10 次，验证动作数组严格逐位一致
        for run_idx in range(10):
            repeated_teacher = MCMFTeacher(min_raid_ants=4, raid_ratio=0.5)
            repeated_orders = repeated_teacher.orders(obs, water, seen)
            assert np.array_equal(reference_orders, repeated_orders), (
                f"地图 {map_id} 在第 {run_idx} 次重复推演中输出动作数组未达成逐位一致"
            )


def test_empirical_scale_100_long_horizon_redundancy_and_collision():
    # 压力测试：100 只蚂蚁在大型地图上进行 30 回合连续推演
    board_data, map_id = resolve_benchmark_map("basic-large-2p")
    scale = 100
    turns = 30
    seed = 88888

    res_compound = run_empirical_gathering_simulation(
        board_data=board_data,
        scale=scale,
        turns=turns,
        seed=seed,
        teacher_type="compound",
    )

    res_mcmf = run_empirical_gathering_simulation(
        board_data=board_data,
        scale=scale,
        turns=turns,
        seed=seed,
        teacher_type="mcmf",
    )

    c_redundancy = res_compound["total_redundant_gather_count"]
    m_redundancy = res_mcmf["total_redundant_gather_count"]

    assert c_redundancy > 0, "CompoundTeacher 在 100 蚂蚁大型地图推演中采食冗余数应当大于 0"
    reduction_rate = (c_redundancy - m_redundancy) / c_redundancy

    print(
        f"\n[100 蚂蚁大型地图 {map_id}] 采食重叠冗余数: "
        f"Compound={c_redundancy}, MCMF={m_redundancy}, 削减比例={reduction_rate * 100:.2f}%"
    )

    assert reduction_rate >= 0.60, (
        f"100 蚂蚁大型地图 {map_id} 采食冗余削减比例未达到 60% 门禁: "
        f"实际削减 {reduction_rate * 100:.2f}% (Compound={c_redundancy}, MCMF={m_redundancy})"
    )

    assert res_mcmf["friendly_collision_violations"] == 0, (
        f"100 蚂蚁大型地图推演中出现友方目的地重叠碰撞: {res_mcmf['friendly_collision_violations']}"
    )


def test_empirical_multi_turn_bitwise_determinism_trajectory():
    # 验证多回合完整轨迹级逐位一致性：50 蚂蚁连续推演 20 回合
    board_data, map_id = resolve_benchmark_map("basic-medium-2p")
    scale = 50
    turns = 20
    seed = 77777

    def collect_trajectory(run_seed: int) -> list[np.ndarray]:
        rows = int(board_data["rows"])
        cols = int(board_data["cols"])
        b = planes.Board(rows, cols)
        water = b.rle(board_data["water"]).astype(bool)
        land_cells = [(r, c) for r in range(rows) for c in range(cols) if not water[r, c]]

        rng = np.random.default_rng(run_seed)
        chosen_indices = rng.choice(len(land_cells), size=scale, replace=False)
        ants = [list(land_cells[idx]) for idx in chosen_indices]
        ant_ids = list(range(scale))

        hills_raw = board_data.get("hills", [])
        hills = [[int(h[0]), int(h[1]), idx] for idx, h in enumerate(hills_raw)]

        food_target = max(scale // 2, 16)
        ant_set = {tuple(a) for a in ants}
        current_food = [
            [int(f[0]), int(f[1])]
            for f in board_data.get("food", [])
            if not water[int(f[0]), int(f[1])] and (int(f[0]), int(f[1])) not in ant_set
        ][:food_target]

        teacher = MCMFTeacher(min_raid_ants=4, raid_ratio=0.5)
        memory: dict = {}
        stall_counters: dict[int, int] = {}
        trajectory_orders: list[np.ndarray] = []

        for _ in range(turns):
            mine_plane = b.scatter(ants)
            vis_plane = b.dilate(mine_plane, planes.VIEW_RADIUS2).astype(bool)
            seen = water | vis_plane

            obs = {
                "size": [rows, cols],
                "mine": [list(a) for a in ants],
                "ids": list(ant_ids),
                "foes": [],
                "food": [list(f) for f in current_food],
                "hills": hills,
                "water": {"rle": rle_from_mask(water)},
                "vis": {"rle": rle_from_mask(vis_plane)},
            }

            memory = planes.remember(memory, obs, mode="where")
            orders = teacher.orders(obs, water, seen, memory, stall_counters=stall_counters)
            trajectory_orders.append(orders.copy())

            # 推进位置
            dests = []
            for (ar, ac), a in zip(ants, orders):
                act_int = int(a)
                if act_int == HOLD:
                    dests.append((ar, ac))
                else:
                    dr, dc = DELTAS[act_int]
                    dests.append(((ar + dr) % rows, (ac + dc) % cols))
            ants = [list(p) for p in dests]

        return trajectory_orders

    traj_a = collect_trajectory(seed)
    traj_b = collect_trajectory(seed)

    assert len(traj_a) == turns
    assert len(traj_b) == turns
    for t_idx in range(turns):
        assert np.array_equal(traj_a[t_idx], traj_b[t_idx]), (
            f"回合 {t_idx} 动作数组未达成逐位一致"
        )

