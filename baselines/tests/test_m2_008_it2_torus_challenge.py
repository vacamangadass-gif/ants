from __future__ import annotations

import numpy as np

from tb_baselines.coordination import (
    DELTAS,
    HOLD,
    extract_cluster_gathering_capacity,
)
from tb_baselines.mcmf_teacher import MCMFTeacher


def test_orthogonal_wrap_capacity_and_overscheduling_protection():
    # 环面垂直边界正交相邻食物：(0, 5) 与 (9, 5)
    rows, cols = 10, 10
    water = np.zeros((rows, cols), dtype=bool)
    food_coords = [(0, 5), (rows - 1, 5)]

    # 1. 验证开阔地形下物理容量
    pos_open, caps_open = extract_cluster_gathering_capacity(food_coords, water, rows, cols)
    # 两颗食物互为正交邻居，其余 6 个采集位均可通行
    assert len(pos_open) == 6
    assert len(caps_open) == 1
    assert list(caps_open.values())[0] == 2

    # 2. 施加水域约束，使采集位仅剩余 1 个
    # (0, 5) 邻居: (1, 5), (0, 4), (0, 6), (9, 5)[食物]
    # (9, 5) 邻居: (8, 5), (9, 4), (9, 6), (0, 5)[食物]
    water[1, 5] = True
    water[0, 4] = True
    water[0, 6] = True
    water[8, 5] = True
    water[9, 4] = True
    # 仅保留 (9, 6) 为合法通行陆地
    pos_choke, caps_choke = extract_cluster_gathering_capacity(food_coords, water, rows, cols)
    assert len(pos_choke) == 1
    assert pos_choke[0] == (9, 6)
    assert len(caps_choke) == 1
    assert list(caps_choke.values())[0] == 1

    # 3. 部署 6 只蚂蚁，验证 MCMFTeacher 不会出现过度调度
    ants = [(8, 6), (7, 6), (9, 7), (8, 7), (7, 7), (6, 6)]
    seen = np.ones((rows, cols), dtype=bool)
    obs = {
        "size": [rows, cols],
        "mine": ants,
        "ids": list(range(len(ants))),
        "food": food_coords,
        "hills": [],
    }

    teacher = MCMFTeacher(min_raid_ants=10)
    orders, telemetry_records = teacher.orders_with_telemetry(obs, water, seen)

    # 统计被分派为采食任务的蚂蚁数量
    gather_assigned_count = sum(
        1 for rec in telemetry_records if rec.nearest_source_type in ("food_visible", "food_memory")
    )
    assert gather_assigned_count == 1, (
        f"在物理采集容量为 1 的情况下，获配采食任务的蚂蚁数量必须严格等于 1，实际为 {gather_assigned_count}"
    )

    # 验证下一回合意图目的地无碰撞
    destinations = []
    for i, ((ar, ac), a) in enumerate(zip(ants, orders)):
        act_int = int(a)
        if act_int == HOLD:
            dest = (ar, ac)
        else:
            dr, dc = DELTAS[act_int]
            dest = ((ar + dr) % rows, (ac + dc) % cols)
        destinations.append(dest)

    assert len(set(destinations)) == len(destinations), "友方蚂蚁之间发生了目的地重叠冲突"
    # 验证没有蚂蚁走向水域或者食物实体
    food_set = set(food_coords)
    for dr_c, dc_c in destinations:
        assert not water[dr_c, dc_c], f"蚂蚁走向水域 ({dr_c}, {dc_c})"
        assert (dr_c, dc_c) not in food_set, f"蚂蚁走向食物实体 ({dr_c}, {dc_c})"


def test_diagonal_four_corners_torus_singularity():
    # 环面四个角点食物密集排列：(0, 0), (0, 11), (11, 0), (11, 11)
    rows, cols = 12, 12
    water = np.zeros((rows, cols), dtype=bool)
    corner_foods = [(0, 0), (0, cols - 1), (rows - 1, 0), (rows - 1, cols - 1)]

    # 1. 验证四个角点跨越双向环面边界合并为单一聚类簇
    pos_open, caps_open = extract_cluster_gathering_capacity(corner_foods, water, rows, cols)
    assert len(caps_open) == 1, f"四个环面角点食物应当合并为单一簇，实际簇数: {len(caps_open)}"
    assert len(pos_open) == 8, f"2x2 环面食物块外部采集位应当为 8，实际为 {len(pos_open)}"
    assert list(caps_open.values())[0] == 4, f"容量应当为 min(8, 4) = 4，实际为 {list(caps_open.values())[0]}"

    # 2. 水域严密包围测试情况 A：保留 2 个属于不同食物的独立采集位
    # 采集位包含: (0, 1)[邻接 (0,0)], (0, 10)[邻接 (0,11)], (11, 1), (11, 10), (10, 0), (1, 0), (1, 11), (10, 11)
    # 保留 (0, 1) 与 (0, 10)，分别属于食物 (0, 0) 与 (0, 11)
    water_a = np.zeros((rows, cols), dtype=bool)
    for r, c in [(11, 1), (11, 10), (10, 0), (1, 0), (1, 11), (10, 11)]:
        water_a[r, c] = True

    pos_choke_a, caps_choke_a = extract_cluster_gathering_capacity(corner_foods, water_a, rows, cols)
    assert len(pos_choke_a) == 2
    assert set(pos_choke_a) == {(0, 1), (0, 10)}
    assert len(caps_choke_a) == 1
    assert list(caps_choke_a.values())[0] == 2

    # 部署 12 只蚂蚁测试极限竞争（不同食物采集位）
    ants = [
        (0, 2), (0, 3), (1, 2), (2, 0), (3, 0), (2, 1),
        (4, 4), (5, 5), (6, 6), (7, 7), (8, 8), (9, 9),
    ]
    seen = np.ones((rows, cols), dtype=bool)
    obs_a = {
        "size": [rows, cols],
        "mine": ants,
        "ids": list(range(len(ants))),
        "food": corner_foods,
        "hills": [],
    }

    teacher = MCMFTeacher(min_raid_ants=20)
    orders_a, telemetry_records_a = teacher.orders_with_telemetry(obs_a, water_a, seen)

    gather_assigned_count_a = sum(
        1 for rec in telemetry_records_a if rec.nearest_source_type in ("food_visible", "food_memory")
    )
    assert gather_assigned_count_a == 2, (
        f"在 2 个采集位分别服务于 2 颗不同食物实体时，采食分派数量必须等于 2，实际为 {gather_assigned_count_a}"
    )

    # 3. 水域严密包围测试情况 B：保留 2 个采集位但它们共同属于同一颗食物实体 (0, 0)
    # 保留 (0, 1) 与 (1, 0)，其余 6 个采集位均设为水域
    # 此时簇虽有 4 颗食物和 2 个采集位，但因受限于同一食物汇点容量上限 1，物理匹配上限严格为 1
    water_b = np.zeros((rows, cols), dtype=bool)
    for r, c in [(0, 10), (11, 1), (11, 10), (10, 0), (1, 11), (10, 11)]:
        water_b[r, c] = True

    obs_b = {
        "size": [rows, cols],
        "mine": ants,
        "ids": list(range(len(ants))),
        "food": corner_foods,
        "hills": [],
    }
    orders_b, telemetry_records_b = teacher.orders_with_telemetry(obs_b, water_b, seen)
    gather_assigned_count_b = sum(
        1 for rec in telemetry_records_b if rec.nearest_source_type in ("food_visible", "food_memory")
    )
    assert gather_assigned_count_b == 1, (
        f"当 2 个采集位均连接至同一食物实体时，由于食物汇点容量为 1，采食分派蚂蚁数量必须严格受限为 1，杜绝重复认领，实际为 {gather_assigned_count_b}"
    )

    destinations = []
    for i, ((ar, ac), a) in enumerate(zip(ants, orders_a)):
        act_int = int(a)
        if act_int == HOLD:
            dest = (ar, ac)
        else:
            dr, dc = DELTAS[act_int]
            dest = ((ar + dr) % rows, (ac + dc) % cols)
        destinations.append(dest)

    assert len(set(destinations)) == len(destinations), "友方蚂蚁发生目的地碰撞"


def test_torus_ring_serpent_capacity_choke():
    # 跨越水平环面一整行食物带（14 颗食物闭合成环）
    rows, cols = 14, 14
    water = np.zeros((rows, cols), dtype=bool)
    food_coords = [(0, c) for c in range(cols)]

    # 1. 开阔地形下聚类验证
    pos_open, caps_open = extract_cluster_gathering_capacity(food_coords, water, rows, cols)
    assert len(caps_open) == 1
    assert len(pos_open) == cols * 2  # 上方 (rows-1) 行与下方 1 行各 cols 个采集位
    assert list(caps_open.values())[0] == cols

    # 2. 将第 1 行全部封死为水域，第 rows-1 行仅留 3 个通行方格
    water[1, :] = True
    water[rows - 1, :] = True
    water[rows - 1, 2] = False
    water[rows - 1, 6] = False
    water[rows - 1, 10] = False

    pos_choke, caps_choke = extract_cluster_gathering_capacity(food_coords, water, rows, cols)
    assert len(pos_choke) == 3
    assert set(pos_choke) == {(rows - 1, 2), (rows - 1, 6), (rows - 1, 10)}
    assert len(caps_choke) == 1
    assert list(caps_choke.values())[0] == 3

    # 3. 部署 20 只蚂蚁施加高密度调度压力
    ants = []
    for r in range(4, 9):
        for c in range(2, 6):
            ants.append((r, c))
    assert len(ants) == 20

    seen = np.ones((rows, cols), dtype=bool)
    obs = {
        "size": [rows, cols],
        "mine": ants,
        "ids": list(range(len(ants))),
        "food": food_coords,
        "hills": [],
    }

    teacher = MCMFTeacher(min_raid_ants=30)
    orders, telemetry_records = teacher.orders_with_telemetry(obs, water, seen)

    gather_assigned_count = sum(
        1 for rec in telemetry_records if rec.nearest_source_type in ("food_visible", "food_memory")
    )
    assert gather_assigned_count == 3, (
        f"整行环形食物受限为 3 个采集位时，采食分配数量必须严格等于 3，实际为 {gather_assigned_count}"
    )

    destinations = []
    for i, ((ar, ac), a) in enumerate(zip(ants, orders)):
        act_int = int(a)
        if act_int == HOLD:
            dest = (ar, ac)
        else:
            dr, dc = DELTAS[act_int]
            dest = ((ar + dr) % rows, (ac + dc) % cols)
        destinations.append(dest)

    assert len(set(destinations)) == len(destinations), "友方蚂蚁发生目的地碰撞"


def test_disjoint_torus_boundary_clusters_multi_turn_simulation():
    # 同时存在两个独立的跨环面边界食物簇
    # 簇 1：水平跨界 (0, 0) 与 (0, 9)
    # 簇 2：垂直跨界 (0, 5) 与 (9, 5)
    # 簇 3：中央孤立食物 (5, 5)
    rows, cols = 10, 10
    water = np.zeros((rows, cols), dtype=bool)
    food_coords = [(0, 0), (0, cols - 1), (0, 5), (rows - 1, 5), (5, 5)]

    positions, caps = extract_cluster_gathering_capacity(food_coords, water, rows, cols)
    # 簇 1 ((0,0) 与 (0,9))、簇 2 ((0,5) 与 (9,5))、簇 3 ((5,5)) 相互距离大于 1，必须为 3 个独立簇
    assert len(caps) == 3, f"必须分离为 3 个独立聚类簇，实际为 {len(caps)}"
    assert sorted(caps.values()) == [1, 2, 2]

    # 部署 15 只蚂蚁进行 5 回合连续推演，检验跨界多簇调度下的连续无碰撞与容量匹配
    rng = np.random.default_rng(20261103)
    land = [(r, c) for r in range(rows) for c in range(cols) if (r, c) not in set(food_coords)]
    chosen_indices = rng.choice(len(land), size=15, replace=False)
    current_ants = [land[idx] for idx in chosen_indices]

    teacher = MCMFTeacher(min_raid_ants=25)
    current_food = list(food_coords)

    for turn in range(5):
        seen = np.ones((rows, cols), dtype=bool)
        obs = {
            "size": [rows, cols],
            "mine": current_ants,
            "ids": list(range(len(current_ants))),
            "food": current_food,
            "hills": [],
        }

        orders, tele = teacher.orders_with_telemetry(obs, water, seen)

        destinations = []
        for i, ((ar, ac), a) in enumerate(zip(current_ants, orders)):
            act_int = int(a)
            if act_int == HOLD:
                dest = (ar, ac)
            else:
                dr, dc = DELTAS[act_int]
                dest = ((ar + dr) % rows, (ac + dc) % cols)
            destinations.append(dest)

        # 验证物理不变量：0 内部碰撞冲突
        assert len(set(destinations)) == len(destinations), (
            f"第 {turn} 回合友方蚂蚁发生目的地碰撞冲突: {destinations}"
        )

        # 验证没有蚂蚁走向当前食物实体
        food_set = set(current_food)
        for dest in destinations:
            assert dest not in food_set, f"第 {turn} 回合蚂蚁步入食物实体格 {dest}"

        current_ants = destinations


def test_diagonal_wrap_single_shared_gathering_cell():
    # 对角跨越环面左上角：(0, 1) 与 (rows - 1, 0)
    # (0, 0) 正好是两颗食物共同正交相邻的唯一重叠单元格
    rows, cols = 8, 8
    water = np.zeros((rows, cols), dtype=bool)
    food_coords = [(0, 1), (rows - 1, 0)]

    # 将除 (0, 0) 以外的所有潜在采集位均封闭为水域
    # (0, 1) 潜在采集位: (0, 0), (1, 1), (0, 2), (rows-1, 1)
    # (rows-1, 0) 潜在采集位: (rows-1, cols-1), (rows-2, 0), (0, 0), (rows-1, 1)
    for r, c in [(1, 1), (0, 2), (rows - 1, 1), (rows - 1, cols - 1), (rows - 2, 0)]:
        water[r, c] = True

    pos, caps = extract_cluster_gathering_capacity(food_coords, water, rows, cols)
    # 仅 (0, 0) 是唯一可通行的陆地采集位
    assert len(pos) == 1
    assert pos[0] == (0, 0)
    assert len(caps) == 1
    # 物理容量 C = min(|G(R)|, |R|) = min(1, 2) = 1
    assert list(caps.values())[0] == 1

    # 部署 5 只蚂蚁
    ants = [(2, 2), (3, 3), (4, 4), (5, 5), (6, 6)]
    seen = np.ones((rows, cols), dtype=bool)
    obs = {
        "size": [rows, cols],
        "mine": ants,
        "ids": list(range(len(ants))),
        "food": food_coords,
        "hills": [],
    }

    teacher = MCMFTeacher(min_raid_ants=10)
    orders, tele = teacher.orders_with_telemetry(obs, water, seen)

    gather_assigned_count = sum(
        1 for rec in tele if rec.nearest_source_type in ("food_visible", "food_memory")
    )
    assert gather_assigned_count == 1, (
        f"在只有 1 个共享采集位时，采食分派蚂蚁数量必须严格等于 1，实际为 {gather_assigned_count}"
    )

    destinations = []
    for i, ((ar, ac), a) in enumerate(zip(ants, orders)):
        act_int = int(a)
        if act_int == HOLD:
            dest = (ar, ac)
        else:
            dr, dc = DELTAS[act_int]
            dest = ((ar + dr) % rows, (ac + dc) % cols)
        destinations.append(dest)

    assert len(set(destinations)) == len(destinations), "友方相撞"
