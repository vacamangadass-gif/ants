from __future__ import annotations

from collections import deque
import numpy as np

from tb_baselines.coordination import (
    DELTAS,
    extract_cluster_gathering_capacity,
)


def independent_chebyshev_cluster_oracle(
    food_coords: list[tuple[int, int]],
    water_mask: np.ndarray,
    rows: int,
    cols: int,
) -> tuple[int, dict[int, int], dict[int, set[tuple[int, int]]]]:
    # 去重食物坐标
    unique_foods: list[tuple[int, int]] = []
    seen_f = set()
    for item in food_coords:
        pt = (int(item[0]) % rows, int(item[1]) % cols)
        if pt not in seen_f:
            seen_f.add(pt)
            unique_foods.append(pt)

    n_foods = len(unique_foods)
    if n_foods == 0:
        return 0, {}, {}

    passable = (~water_mask)
    for fr, fc in unique_foods:
        passable[fr, fc] = False

    # 计算每颗食物的正交采集位集合
    food_gatherings: list[set[tuple[int, int]]] = []
    for fr, fc in unique_foods:
        g_set = set()
        for dr, dc in DELTAS:
            nr = (fr + dr) % rows
            nc = (fc + dc) % cols
            if passable[nr, nc]:
                g_set.add((nr, nc))
        food_gatherings.append(g_set)

    # 建立邻接图：切比雪夫距离 <= 1 或共享采集位
    adj: list[list[int]] = [[] for _ in range(n_foods)]
    for i in range(n_foods):
        r1, c1 = unique_foods[i]
        for j in range(i + 1, n_foods):
            r2, c2 = unique_foods[j]
            dr = min(abs(r1 - r2), rows - abs(r1 - r2))
            dc = min(abs(c1 - c2), cols - abs(c1 - c2))
            is_chebyshev_adjacent = max(dr, dc) <= 1
            shares_gathering = bool(food_gatherings[i] & food_gatherings[j])
            if is_chebyshev_adjacent or shares_gathering:
                adj[i].append(j)
                adj[j].append(i)

    # 广度优先搜索求连通分量
    visited = [False] * n_foods
    clusters: list[list[int]] = []
    for i in range(n_foods):
        if not visited[i]:
            comp: list[int] = []
            queue = deque([i])
            visited[i] = True
            while queue:
                curr = queue.popleft()
                comp.append(curr)
                for neighbor in adj[curr]:
                    if not visited[neighbor]:
                        visited[neighbor] = True
                        queue.append(neighbor)
            clusters.append(comp)

    cluster_caps: dict[int, int] = {}
    cluster_gather_sets: dict[int, set[tuple[int, int]]] = {}
    for c_id, comp in enumerate(clusters):
        all_g = set()
        for f_idx in comp:
            all_g.update(food_gatherings[f_idx])
        cluster_caps[c_id] = min(len(all_g), len(comp))
        cluster_gather_sets[c_id] = all_g

    return len(clusters), cluster_caps, cluster_gather_sets


def test_stress_chebyshev_10x10_solid_dense_food_block():
    rows, cols = 30, 30
    water = np.zeros((rows, cols), dtype=bool)

    # 构造 10x10 紧密排列的 100 颗食物实体
    dense_foods = [(r, c) for r in range(10, 20) for c in range(10, 20)]
    assert len(dense_foods) == 100

    positions, caps = extract_cluster_gathering_capacity(dense_foods, water, rows, cols)

    # 1. 100 颗连通方块食物必须且只能合并为 1 个独立簇
    assert len(caps) == 1

    # 2. 外部正交边缘采集位精确为 40 个
    assert len(positions) == 40

    # 3. 簇物理容量必须为 40 (min(40, 100))
    cap_val = list(caps.values())[0]
    assert cap_val == 40


def test_stress_chebyshev_torus_wrapping_split_corners_100_foods():
    rows, cols = 30, 30
    water = np.zeros((rows, cols), dtype=bool)

    # 在环面四个对角处各部署 5x5 的食物块，总共 100 颗食物
    # 由于环面拓扑 (0 与 rows-1 相邻)，这四个角块在环面上拼接为一个连续的 10x10 密集食物块
    corner_foods = []
    for r in list(range(0, 5)) + list(range(25, 30)):
        for c in list(range(0, 5)) + list(range(25, 30)):
            corner_foods.append((r, c))
    assert len(corner_foods) == 100

    positions, caps = extract_cluster_gathering_capacity(corner_foods, water, rows, cols)

    # 1. 跨越环面边界全部通过切比雪夫距离合并为 1 个簇
    assert len(caps) == 1

    # 2. 环面外周采集位总数精确为 40 个
    assert len(positions) == 40

    # 3. 簇容量为 40
    assert list(caps.values())[0] == 40


def test_stress_chebyshev_100_foods_diagonal_chain():
    rows, cols = 120, 120
    water = np.zeros((rows, cols), dtype=bool)

    # 100 颗对角线排列食物：(i, i)，切比雪夫距离均为 1
    diag_foods = [(i, i) for i in range(100)]
    assert len(diag_foods) == 100

    positions, caps = extract_cluster_gathering_capacity(diag_foods, water, rows, cols)

    # 1. 全部合并为 1 个簇
    assert len(caps) == 1

    # 2. 容量不变式
    assert list(caps.values())[0] == min(len(positions), 100)
    assert list(caps.values())[0] > 0


def test_stress_chebyshev_random_packed_dense_clusters_fuzzing():
    # 随机模糊测试：检验 100 颗食物在高密度团簇分布下的聚类与容量不变式
    for seed in range(4001, 4016):
        rng = np.random.default_rng(seed)
        rows, cols = 64, 64
        water = rng.random((rows, cols)) < 0.10

        # 生成 5 个致密团簇种子，每个种子通过 8 邻域扩散生成 20 颗紧密排列食物
        all_foods_set: set[tuple[int, int]] = set()
        cluster_seeds = []
        while len(cluster_seeds) < 5:
            sr = int(rng.integers(0, rows))
            sc = int(rng.integers(0, cols))
            if not water[sr, sc] and (sr, sc) not in all_foods_set:
                cluster_seeds.append((sr, sc))
                all_foods_set.add((sr, sc))

        for sr, sc in cluster_seeds:
            local_cluster = [(sr, sc)]
            queue = deque([(sr, sc)])
            while len(local_cluster) < 20 and len(all_foods_set) < 100:
                if not queue:
                    break
                curr_r, curr_c = queue.popleft()
                # 随机打乱 8 邻域扩散方向
                offsets = [
                    (dr, dc)
                    for dr in (-1, 0, 1)
                    for dc in (-1, 0, 1)
                    if not (dr == 0 and dc == 0)
                ]
                rng.shuffle(offsets)
                for dr, dc in offsets:
                    nr = (curr_r + dr) % rows
                    nc = (curr_c + dc) % cols
                    if not water[nr, nc] and (nr, nc) not in all_foods_set:
                        all_foods_set.add((nr, nc))
                        local_cluster.append((nr, nc))
                        queue.append((nr, nc))
                        if len(local_cluster) == 20 or len(all_foods_set) == 100:
                            break

        # 若不足 100 颗，补充致密相邻点
        while len(all_foods_set) < 100:
            pivot = list(all_foods_set)[int(rng.integers(0, len(all_foods_set)))]
            dr = int(rng.integers(-1, 2))
            dc = int(rng.integers(-1, 2))
            cand = ((pivot[0] + dr) % rows, (pivot[1] + dc) % cols)
            if not water[cand[0], cand[1]]:
                all_foods_set.add(cand)

        dense_food_list = list(all_foods_set)
        assert len(dense_food_list) == 100

        # 运行被测实现
        positions, caps = extract_cluster_gathering_capacity(dense_food_list, water, rows, cols)

        # 运行独立验证 Oracle
        expected_n_clusters, expected_caps, expected_g_sets = independent_chebyshev_cluster_oracle(
            dense_food_list, water, rows, cols
        )

        # 1. 簇数量必须与独立 Oracle 严格一致
        assert len(caps) == expected_n_clusters, (
            f"种子 {seed}: 簇数量与 Oracle 不符 (实际 {len(caps)} vs 预期 {expected_n_clusters})"
        )

        # 2. 簇容量多重集合必须与独立 Oracle 严格一致
        actual_sorted_caps = sorted(caps.values())
        expected_sorted_caps = sorted(expected_caps.values())
        assert actual_sorted_caps == expected_sorted_caps, (
            f"种子 {seed}: 簇容量与 Oracle 不符 (实际 {actual_sorted_caps} vs 预期 {expected_sorted_caps})"
        )

        # 3. 总容量与采集位数量物理不变式
        total_cluster_capacity = sum(caps.values())
        assert total_cluster_capacity <= 100
        assert total_cluster_capacity <= len(positions)

        # 4. 采集位无遗漏不变式
        all_oracle_gatherings = set().union(*expected_g_sets.values()) if expected_g_sets else set()
        assert len(positions) == len(all_oracle_gatherings)
        assert set(positions) == all_oracle_gatherings
