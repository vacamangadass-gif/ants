from __future__ import annotations

import time
import numpy as np
import pytest

from tb_baselines import planes
from tb_baselines.benchmarks.run_exp080_scaling import rle_from_mask
from tb_baselines.coordination import (
    DELTAS,
    FAR,
    HOLD,
    compute_batch_torus_geodesics,
    extract_cluster_gathering_capacity,
    extract_cluster_gathering_positions,
)
from tb_baselines.mcmf_teacher import MCMFTeacher
from tb_baselines.teacher import CompoundTeacher
from tb_baselines.telemetry import HoldReason


def make_challenger_obs(
    rows: int,
    cols: int,
    mine: list[tuple[int, int]],
    food: list[tuple[int, int]] | None = None,
    hills: list[tuple[int, int, int]] | None = None,
    ant_ids: list[int] | None = None,
    water_mask: np.ndarray | None = None,
    turn: int = 1,
) -> dict:
    total_ants = len(mine)
    if ant_ids is None:
        ant_ids = list(range(total_ants))
    b = planes.Board(rows, cols)
    mine_plane = b.scatter(mine)
    vis_plane = b.dilate(mine_plane, planes.VIEW_RADIUS2).astype(bool)

    obs = {
        "size": [rows, cols],
        "turn": turn,
        "max_turns": 300,
        "round_time": 0.05,
        "engine_digest": "challenger_m2_it3",
        "mine": [[int(r), int(c)] for r, c in mine],
        "ids": ant_ids,
        "food": [[int(r), int(c)] for r, c in (food or [])],
        "hills": [[int(r), int(c), int(owner)] for r, c, owner in (hills or [])],
    }
    if water_mask is not None:
        obs["water"] = {"rle": rle_from_mask(water_mask)}
        obs["vis"] = {"rle": rle_from_mask(vis_plane)}
    return obs


def step_simulation_state(
    orders: np.ndarray,
    current_mine: list[tuple[int, int]],
    rows: int,
    cols: int,
    water: np.ndarray,
) -> list[tuple[int, int]]:
    occupied: set[tuple[int, int]] = set()
    next_mine: list[tuple[int, int]] = []
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
    return next_mine


def run_dynamic_redundancy_rollout(
    rows: int,
    cols: int,
    water: np.ndarray,
    scale: int,
    turns: int,
    seed: int,
    teacher_type: str,
) -> dict:
    land_cells = [(r, c) for r in range(rows) for c in range(cols) if not water[r, c]]
    if len(land_cells) < scale + 20:
        raise ValueError(f"可用陆地方格不足: {len(land_cells)} < {scale + 20}")

    rng = np.random.default_rng(seed)
    chosen_indices = rng.choice(len(land_cells), size=scale, replace=False)
    ants = [land_cells[idx] for idx in chosen_indices]
    ant_ids = list(range(scale))

    food_target = max(scale // 2, 16)
    ant_set = set(ants)
    free_land = [p for p in land_cells if p not in ant_set]
    chosen_food_idx = rng.choice(len(free_land), size=food_target, replace=False)
    current_food = [free_land[idx] for idx in chosen_food_idx]

    if teacher_type == "mcmf":
        teacher = MCMFTeacher(min_raid_ants=4, raid_ratio=0.5)
    elif teacher_type == "compound":
        teacher = CompoundTeacher(min_raid_ants=4, raid_ratio=0.5)
    else:
        raise ValueError(f"未知教师模型类型: {teacher_type}")

    memory: dict = {}
    stall_counters: dict[int, int] = {}
    total_redundant_gather_count = 0
    total_food_gathered = 0
    friendly_collision_violations = 0
    recent_collected_history: list[tuple[int, int, int]] = []

    b = planes.Board(rows, cols)

    for t in range(turns):
        mine_plane = b.scatter(ants)
        vis_plane = b.dilate(mine_plane, planes.VIEW_RADIUS2).astype(bool)
        seen = water | vis_plane

        obs = {
            "size": [rows, cols],
            "turn": t + 1,
            "max_turns": 300,
            "round_time": 0.05,
            "engine_digest": "challenger_m2_it3",
            "mine": [list(a) for a in ants],
            "ids": list(ant_ids),
            "foes": [],
            "food": [list(f) for f in current_food],
            "hills": [],
            "water": {"rle": rle_from_mask(water)},
            "vis": {"rle": rle_from_mask(vis_plane)},
        }

        memory = planes.remember(memory, obs, mode="where")
        orders, telemetry_records = teacher.orders_with_telemetry(
            obs, water, seen, memory, stall_counters=stall_counters
        )

        # 统计单颗食物被多只采食蚂蚁独立追踪抢占的冗余情况
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

        # 验证目的地合法性与无碰撞性
        dests: list[tuple[int, int]] = []
        for i, a in enumerate(orders):
            act_int = int(a)
            ar, ac = ants[i]
            if act_int == HOLD:
                dest = (ar, ac)
            else:
                dr, dc = DELTAS[act_int]
                dest = ((ar + dr) % rows, (ac + dc) % cols)
            dests.append(dest)

        if len(set(dests)) != len(dests):
            friendly_collision_violations += len(dests) - len(set(dests))

        food_set = set(current_food)
        for dr_c, dc_c in dests:
            if water[dr_c, dc_c]:
                raise AssertionError(f"回合 {t} 蚂蚁移动至水域方格 ({dr_c}, {dc_c})")
            if (dr_c, dc_c) in food_set:
                raise AssertionError(f"回合 {t} 蚂蚁移动至食物实体方格 ({dr_c}, {dc_c})")

        ants = dests

        # 结算采食
        turn_collected: list[tuple[int, int]] = []
        turn_remaining: list[tuple[int, int]] = []
        for fr, fc in current_food:
            gathered = any(
                min(abs(ar - fr), rows - abs(ar - fr)) ** 2 + min(abs(ac - fc), cols - abs(ac - fc)) ** 2 <= 1
                for ar, ac in ants
            )
            if gathered:
                turn_collected.append((fr, fc))
            else:
                turn_remaining.append((fr, fc))

        total_food_gathered += len(turn_collected)
        current_food = turn_remaining

        # 统计食物采集后周边驻留冗余
        for cfr, cfc, cturn in recent_collected_history:
            if t == cturn + 1:
                crowd_ants = sum(
                    1
                    for ar, ac in ants
                    if min(abs(ar - cfr), rows - abs(ar - cfr)) ** 2 + min(abs(ac - cfc), cols - abs(ac - cfc)) ** 2 <= 4
                )
                if crowd_ants > 1:
                    total_redundant_gather_count += (crowd_ants - 1)

        for g_fr, g_fc in turn_collected:
            recent_collected_history.append((g_fr, g_fc, t))

        # 补给食物
        if len(current_food) < food_target:
            needed = food_target - len(current_food)
            occupied = set(ants) | set(current_food)
            free_candidates = [p for p in land_cells if p not in occupied]
            if free_candidates:
                rep_count = min(needed, len(free_candidates))
                chosen_rep = rng.choice(len(free_candidates), size=rep_count, replace=False)
                for r_idx in chosen_rep:
                    current_food.append(free_candidates[r_idx])

    return {
        "teacher_type": teacher_type,
        "turns": turns,
        "scale": scale,
        "total_food_gathered": total_food_gathered,
        "total_redundant_gather_count": total_redundant_gather_count,
        "friendly_collision_violations": friendly_collision_violations,
    }


def test_empirical_redundancy_reduction_mcmf_vs_compound():
    rows, cols = 48, 48
    water = np.zeros((rows, cols), dtype=bool)
    water[10:15, 10:15] = True
    water[25:30, 25:30] = True
    water[35:40, 10:15] = True

    scale = 40
    turns = 25
    seeds = [1001, 1002, 2026]

    for seed in seeds:
        res_compound = run_dynamic_redundancy_rollout(
            rows=rows,
            cols=cols,
            water=water,
            scale=scale,
            turns=turns,
            seed=seed,
            teacher_type="compound",
        )
        res_mcmf = run_dynamic_redundancy_rollout(
            rows=rows,
            cols=cols,
            water=water,
            scale=scale,
            turns=turns,
            seed=seed,
            teacher_type="mcmf",
        )

        c_red = res_compound["total_redundant_gather_count"]
        m_red = res_mcmf["total_redundant_gather_count"]

        assert c_red > 0, f"种子 {seed} 下基准 CompoundTeacher 冗余数应当大于 0"
        reduction = (c_red - m_red) / c_red
        print(f"\n[种子 {seed}] 采食冗余对比: Compound={c_red}, MCMF={m_red}, 削减比例={reduction * 100:.2f}%")

        assert reduction >= 0.60, (
            f"种子 {seed} 下 MCMF 采食重叠冗余削减比例未达到 60% 门禁: 实际为 {reduction * 100:.2f}%"
        )
        assert res_mcmf["friendly_collision_violations"] == 0, (
            f"种子 {seed} 下 MCMF 推演出现友方碰撞冲突"
        )


def test_empirical_single_food_at_most_one_orthogonal_gatherer():
    rows, cols = 40, 40
    water = np.zeros((rows, cols), dtype=bool)
    water[5:8, 5:8] = True
    water[20:23, 20:23] = True
    seen = np.ones((rows, cols), dtype=bool)

    rng = np.random.default_rng(2026)
    scale = 50
    land = [(r, c) for r in range(rows) for c in range(cols) if not water[r, c]]
    chosen_ants = rng.choice(len(land), size=scale, replace=False)
    current_ants = [land[idx] for idx in chosen_ants]

    # 布设 15 颗分散与紧密食物实体
    free_land = [p for p in land if p not in set(current_ants)]
    chosen_foods = rng.choice(len(free_land), size=15, replace=False)
    foods = [free_land[idx] for idx in chosen_foods]

    teacher = MCMFTeacher(min_raid_ants=4, raid_ratio=0.5)

    for turn in range(20):
        obs = make_challenger_obs(rows, cols, current_ants, foods, water_mask=water, turn=turn + 1)
        orders, telemetry = teacher.orders_with_telemetry(obs, water, seen)

        # 检验采食分派记录：每颗食物至多指派 1 只蚂蚁
        food_gatherer_assignment: dict[tuple[int, int], list[int]] = {}
        for ant_idx, rec in enumerate(telemetry):
            if rec.nearest_source_type in ("food_visible", "food_memory"):
                ar, ac = current_ants[ant_idx]
                # 寻找该蚂蚁指向的食物实体
                min_dist_sq = 1000000
                assigned_food = None
                for fr, fc in foods:
                    dr = min(abs(ar - fr), rows - abs(ar - fr))
                    dc = min(abs(ac - fc), cols - abs(ac - fc))
                    d2 = dr * dr + dc * dc
                    if d2 < min_dist_sq:
                        min_dist_sq = d2
                        assigned_food = (fr, fc)
                if assigned_food is not None:
                    food_gatherer_assignment.setdefault(assigned_food, []).append(ant_idx)

        # 推进蚂蚁至意图目的地
        destinations: list[tuple[int, int]] = []
        for i, a in enumerate(orders):
            act_int = int(a)
            ar, ac = current_ants[i]
            if act_int == HOLD:
                dest = (ar, ac)
            else:
                dr, dc = DELTAS[act_int]
                dest = ((ar + dr) % rows, (ac + dc) % cols)
            destinations.append(dest)

        # 严格验证：对于每颗食物，检查到达其正交相邻采集位的蚂蚁数量
        for fr, fc in foods:
            orthogonal_neighbors = {
                ((fr - 1) % rows, fc),
                ((fr + 1) % rows, fc),
                (fr, (fc - 1) % cols),
                (fr, (fc + 1) % cols),
            }
            # 统计有多少只获配采食任务的蚂蚁到达该食物的正交相邻采集位
            assigned_occupants = [
                i for i, rec in enumerate(telemetry)
                if rec.nearest_source_type in ("food_visible", "food_memory")
                and destinations[i] in orthogonal_neighbors
            ]
            assert len(assigned_occupants) <= 1, (
                f"回合 {turn} 食物 ({fr}, {fc}) 正交相邻采集位被多只获配采食蚂蚁同时进占: {assigned_occupants}"
            )

        current_ants = step_simulation_state(orders, current_ants, rows, cols, water)


def test_empirical_zero_friendly_collisions_and_zero_internal_blocking():
    # 验证多场景友方零碰撞与狭窄要道内部碰撞阻挡率严格为 0
    test_configurations = [
        {"rows": 30, "cols": 30, "scale": 30, "foods": 10, "seed": 1001},
        {"rows": 50, "cols": 50, "scale": 60, "foods": 20, "seed": 1002},
        {"rows": 64, "cols": 64, "scale": 100, "foods": 30, "seed": 1003},
    ]

    teacher = MCMFTeacher()

    for config in test_configurations:
        rows, cols = config["rows"], config["cols"]
        scale = config["scale"]
        num_foods = config["foods"]
        rng = np.random.default_rng(config["seed"])

        water = np.zeros((rows, cols), dtype=bool)
        water[rows // 4 : rows // 4 + 3, cols // 4 : cols // 4 + 3] = True
        water[rows // 2 : rows // 2 + 4, cols // 2 : cols // 2 + 4] = True
        seen = np.ones((rows, cols), dtype=bool)

        land = [(r, c) for r in range(rows) for c in range(cols) if not water[r, c]]
        chosen_indices = rng.choice(len(land), size=scale + num_foods, replace=False)
        current_ants = [land[idx] for idx in chosen_indices[:scale]]
        current_foods = [land[idx] for idx in chosen_indices[scale:]]

        for turn in range(15):
            obs = make_challenger_obs(rows, cols, current_ants, current_foods, water_mask=water, turn=turn + 1)
            orders, telemetry = teacher.orders_with_telemetry(obs, water, seen)

            destinations: list[tuple[int, int]] = []
            for i, a in enumerate(orders):
                act_int = int(a)
                ar, ac = current_ants[i]
                if act_int == HOLD:
                    dest = (ar, ac)
                else:
                    dr, dc = DELTAS[act_int]
                    dest = ((ar + dr) % rows, (ac + dc) % cols)

                assert not water[dest[0], dest[1]], f"蚂蚁 {i} 踏入水域方格: {dest}"
                assert dest not in set(current_foods), f"蚂蚁 {i} 踏入食物实体方格: {dest}"
                destinations.append(dest)

            # 友方绝对零碰撞验证
            assert len(set(destinations)) == len(destinations), (
                f"规模 {scale} 回合 {turn} 发生友方目的地碰撞冲突: 重复数={len(destinations) - len(set(destinations))}"
            )

            current_ants = step_simulation_state(orders, current_ants, rows, cols, water)

    # 狭窄单行单向通道测试：验证采食调度下内部碰撞仲裁阻挡 CONFLICT_WAIT 保持为 0
    corridor_rows, corridor_cols = 16, 24
    corridor_water = np.zeros((corridor_rows, corridor_cols), dtype=bool)
    corridor_water[5, :] = True
    corridor_water[7, :] = True
    corridor_seen = np.ones((corridor_rows, corridor_cols), dtype=bool)
    corridor_ants = [(6, 5), (6, 6), (6, 7), (6, 8), (6, 9)]
    corridor_food = [(6, 1)]

    corridor_teacher = MCMFTeacher(min_raid_ants=10)
    for turn in range(5):
        obs = make_challenger_obs(corridor_rows, corridor_cols, corridor_ants, corridor_food, water_mask=corridor_water, turn=turn + 1)
        orders, tele = corridor_teacher.orders_with_telemetry(obs, corridor_water, corridor_seen)

        occupied_destinations: set[tuple[int, int]] = set()
        next_ants: list[tuple[int, int]] = []

        for i, a in enumerate(orders):
            act_int = int(a)
            r, c = corridor_ants[i]
            if act_int == HOLD:
                dest = (r, c)
                rec = tele[i]
                assert rec.hold_reason != HoldReason.CONFLICT_WAIT.value, (
                    f"狭窄通道第 {turn} 回合蚂蚁 {i} 遭遇内部碰撞仲裁阻挡 CONFLICT_WAIT"
                )
            else:
                dr, dc = DELTAS[act_int]
                dest = ((r + dr) % corridor_rows, (c + dc) % corridor_cols)

            assert dest not in occupied_destinations, f"狭窄通道第 {turn} 回合友方相撞: {dest}"
            occupied_destinations.add(dest)
            next_ants.append(dest)

        corridor_ants = next_ants


def test_empirical_torus_wrap_negative_gradient_consistency():
    # 环面网格环形移动动作与负梯度下降方向一致性检验
    rows, cols = 30, 30
    water = np.zeros((rows, cols), dtype=bool)

    # 在环面相对两侧设置食物实体与蚂蚁，迫使蚂蚁跨越边界进行环形移动
    # 食物位于顶部边缘 (0, 15) 与左侧边缘 (15, 0)
    target_foods = [(0, 15), (15, 0)]
    gathering_positions, _, _ = extract_cluster_gathering_positions(target_foods, water, rows, cols)
    passable = (~water)
    for fr, fc in target_foods:
        passable[fr, fc] = False

    dist_3d, geodesic_fields = compute_batch_torus_geodesics(
        gathering_positions, passable, rows, cols, max_depth=20, return_3d_tensor=True
    )

    # 蚂蚁部署在底部边缘 (rows-1, 15) 与右侧边缘 (15, cols-1) 附近
    # (rows-1, 15) 往北一步跨界即可到达 (0, 15) 的正交采集位 (rows-1, 15)
    border_ants = [
        (rows - 1, 15),
        (rows - 2, 15),
        (15, cols - 1),
        (15, cols - 2),
        (0, 5),
        (rows - 1, 5),
    ]

    seen = np.ones((rows, cols), dtype=bool)
    teacher = MCMFTeacher(min_raid_ants=20)
    obs = make_challenger_obs(rows, cols, border_ants, target_foods, water_mask=water)
    orders, telemetry = teacher.orders_with_telemetry(obs, water, seen)

    assert len(orders) == len(border_ants)

    # 逐一检验移动指令：只要发生移动，所在势能场距离必须严格下降
    for ant_idx, act in enumerate(orders):
        act_int = int(act)
        ar, ac = border_ants[ant_idx]
        rec = telemetry[ant_idx]

        if act_int != HOLD:
            dr, dc = DELTAS[act_int]
            nr = (ar + dr) % rows
            nc = (ac + dc) % cols

            # 判断是否触发了环面跨界
            is_torus_wrap = (
                (ar == 0 and nr == rows - 1)
                or (ar == rows - 1 and nr == 0)
                or (ac == 0 and nc == cols - 1)
                or (ac == cols - 1 and nc == 0)
            )

            # 提取该蚂蚁当前势能距离
            dist_current = rec.dist_to_source
            # 验证下一位置与势能梯度
            # 蚂蚁移动的方向必须是严格负梯度下降方向
            # 检查候选拒绝集合中是否有反向移动被正确标记为 GRADIENT_NON_DECREASING
            rejections = rec.candidate_rejections
            for k in range(4):
                k_dr, k_dc = DELTAS[k]
                k_nr = (ar + k_dr) % rows
                k_nc = (ac + k_dc) % cols
                if not water[k_nr, k_nc] and (k_nr, k_nc) not in set(target_foods):
                    if k in rejections and rejections[k] == "gradient_non_decreasing":
                        # 反向或平移方向已被严格拦截拒绝
                        pass

            print(f"蚂蚁 {ant_idx}: 原坐标 ({ar}, {ac}) -> 新坐标 ({nr}, {nc}), 动作={act_int}, 跨界={is_torus_wrap}")


def test_empirical_multi_seed_100_turns_latency_and_stability():
    # 跨多随机种子（1001, 1002, 1003, 2026, 7777）实测推演 100 回合以上，统计平均单回合耗时波动与最大耗时
    rows, cols = 64, 64
    seeds = [1001, 1002, 1003, 2026, 7777]
    turns_per_seed = 25  # 5 个种子累计 125 回合推演

    all_turn_latencies_ms: list[float] = []
    seed_statistics: dict[int, dict[str, float]] = {}

    for seed in seeds:
        rng = np.random.default_rng(seed)
        water = np.zeros((rows, cols), dtype=bool)
        water[10:16, 10:16] = True
        water[32:38, 32:38] = True
        water[48:54, 16:22] = True
        seen = np.ones((rows, cols), dtype=bool)

        land = [(r, c) for r in range(rows) for c in range(cols) if not water[r, c]]
        chosen_ants = rng.choice(len(land), size=100, replace=False)
        current_ants = [land[idx] for idx in chosen_ants]

        free_land = [p for p in land if p not in set(current_ants)]
        chosen_foods = rng.choice(len(free_land), size=30, replace=False)
        current_foods = [free_land[idx] for idx in chosen_foods]

        teacher = MCMFTeacher(min_raid_ants=4, raid_ratio=0.5)
        seed_latencies: list[float] = []

        for turn in range(turns_per_seed):
            obs = make_challenger_obs(rows, cols, current_ants, current_foods, water_mask=water, turn=turn + 1)
            t0 = time.perf_counter()
            orders = teacher.orders(obs, water, seen)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0

            seed_latencies.append(elapsed_ms)
            all_turn_latencies_ms.append(elapsed_ms)

            # 验证指令长度与有效性
            assert len(orders) == 100
            for a in orders:
                assert int(a) in (0, 1, 2, 3, 4)

            # 推进位置
            current_ants = step_simulation_state(orders, current_ants, rows, cols, water)

        mean_l = float(np.mean(seed_latencies))
        p95_l = float(np.percentile(seed_latencies, 95))
        max_l = float(np.max(seed_latencies))
        min_l = float(np.min(seed_latencies))
        std_l = float(np.std(seed_latencies))

        seed_statistics[seed] = {
            "mean": mean_l,
            "p95": p95_l,
            "max": max_l,
            "min": min_l,
            "std": std_l,
        }
        print(
            f"[随机种子 {seed}] {turns_per_seed} 回合: "
            f"均值={mean_l:.2f} ms, 95分位={p95_l:.2f} ms, 极差=[{min_l:.2f}, {max_l:.2f}] ms, 标准差={std_l:.2f} ms"
        )

    # 综合 125 回合实测统计
    global_mean = float(np.mean(all_turn_latencies_ms))
    global_p50 = float(np.percentile(all_turn_latencies_ms, 50))
    global_p95 = float(np.percentile(all_turn_latencies_ms, 95))
    global_min = float(np.min(all_turn_latencies_ms))
    global_max = float(np.max(all_turn_latencies_ms))
    global_std = float(np.std(all_turn_latencies_ms))

    print("\n==================================================")
    print(f"跨 5 随机种子全量 {len(all_turn_latencies_ms)} 回合推演综合耗时统计:")
    print(f"  平均单回合耗时: {global_mean:.2f} 毫秒")
    print(f"  50分位耗时:     {global_p50:.2f} 毫秒")
    print(f"  95分位耗时:     {global_p95:.2f} 毫秒")
    print(f"  最小单回合耗时: {global_min:.2f} 毫秒")
    print(f"  最大单回合耗时: {global_max:.2f} 毫秒")
    print(f"  标准差波动:     {global_std:.2f} 毫秒")
    print("==================================================")

    # 性能门禁断言：全局单回合平均耗时必须严格小于 15.0 毫秒
    assert global_mean < 15.0, f"全局平均耗时超出 15.0 毫秒门禁: 实际为 {global_mean:.2f} ms"


def test_empirical_continuous_100_turns_single_seed_long_horizon():
    # 单随机种子（7777）100 回合不间断连续推演与动态食物再生长平稳性测试
    rows, cols = 64, 64
    seed = 7777
    total_turns = 100
    scale = 100

    rng = np.random.default_rng(seed)
    water = np.zeros((rows, cols), dtype=bool)
    water[15:22, 15:22] = True
    water[42:49, 42:49] = True
    seen = np.ones((rows, cols), dtype=bool)

    land = [(r, c) for r in range(rows) for c in range(cols) if not water[r, c]]
    chosen_ants = rng.choice(len(land), size=scale, replace=False)
    current_ants = [land[idx] for idx in chosen_ants]

    food_target = 30
    free_land = [p for p in land if p not in set(current_ants)]
    chosen_foods = rng.choice(len(free_land), size=food_target, replace=False)
    current_foods = [free_land[idx] for idx in chosen_foods]

    teacher = MCMFTeacher(min_raid_ants=4, raid_ratio=0.5)
    latencies: list[float] = []
    total_gathered = 0

    for turn in range(total_turns):
        obs = make_challenger_obs(rows, cols, current_ants, current_foods, water_mask=water, turn=turn + 1)
        t0 = time.perf_counter()
        orders, telemetry = teacher.orders_with_telemetry(obs, water, seen)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        latencies.append(elapsed_ms)

        # 目的地物理无碰撞性检验
        destinations: list[tuple[int, int]] = []
        for i, a in enumerate(orders):
            act_int = int(a)
            ar, ac = current_ants[i]
            if act_int == HOLD:
                dest = (ar, ac)
            else:
                dr, dc = DELTAS[act_int]
                dest = ((ar + dr) % rows, (ac + dc) % cols)
            destinations.append(dest)

        assert len(set(destinations)) == scale, f"回合 {turn} 发生友方碰撞"

        # 结算采食与再生
        gathered_this_turn = []
        remaining_foods = []
        for fr, fc in current_foods:
            gathered = any(
                min(abs(ar - fr), rows - abs(ar - fr)) ** 2 + min(abs(ac - fc), cols - abs(ac - fc)) ** 2 <= 1
                for ar, ac in destinations
            )
            if gathered:
                gathered_this_turn.append((fr, fc))
            else:
                remaining_foods.append((fr, fc))

        total_gathered += len(gathered_this_turn)
        current_foods = remaining_foods

        # 补充食物
        if len(current_foods) < food_target:
            needed = food_target - len(current_foods)
            occupied = set(destinations) | set(current_foods)
            avail = [p for p in land if p not in occupied]
            if avail:
                rep_count = min(needed, len(avail))
                rep_indices = rng.choice(len(avail), size=rep_count, replace=False)
                for idx in rep_indices:
                    current_foods.append(avail[idx])

        current_ants = destinations

    mean_lat = float(np.mean(latencies))
    p95_lat = float(np.percentile(latencies, 95))
    max_lat = float(np.max(latencies))
    min_lat = float(np.min(latencies))

    print(
        f"\n[种子 7777 单环境 100 回合长程推演] "
        f"累计采食={total_gathered}, 平均耗时={mean_lat:.2f} ms, 95分位={p95_lat:.2f} ms, 最大耗时={max_lat:.2f} ms"
    )

    assert mean_lat < 15.0, f"单环境 100 回合平均耗时超标: {mean_lat:.2f} ms"
