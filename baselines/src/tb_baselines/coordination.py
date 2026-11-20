from __future__ import annotations

from enum import Enum
import hashlib
import heapq
import math
from typing import NamedTuple

import numpy as np

FAR = np.int32(1 << 29)
DELTAS = ((-1, 0), (0, 1), (1, 0), (0, -1))
ACTION_CHARS = ("N", "E", "S", "W", "-")
HOLD = 4


def _normalize_food_coords(
    food_coords: list[tuple[int, int]] | list[list[int]] | set[tuple[int, int]] | np.ndarray | None,
    rows: int,
    cols: int,
) -> list[tuple[int, int]]:
    if food_coords is None or len(food_coords) == 0:
        return []
    food_list: list[tuple[int, int]] = []
    if isinstance(food_coords, np.ndarray) and food_coords.size > 0:
        for item in food_coords:
            r, c = int(item[0]), int(item[1])
            if 0 <= r < rows and 0 <= c < cols:
                food_list.append((r, c))
    elif food_coords is not None:
        for item in food_coords:
            r, c = int(item[0]), int(item[1])
            if 0 <= r < rows and 0 <= c < cols:
                food_list.append((r, c))
    if not food_list:
        return []
    seen_food = set()
    unique_food_list: list[tuple[int, int]] = []
    for f in food_list:
        if f not in seen_food:
            seen_food.add(f)
            unique_food_list.append(f)
    return unique_food_list


def extract_cluster_gathering_positions(
    food_coords: list[tuple[int, int]] | list[list[int]] | set[tuple[int, int]] | np.ndarray | None,
    water_mask: np.ndarray,
    rows: int,
    cols: int,
) -> tuple[list[tuple[int, int]], list[list[int]], dict[int, list[int]]]:
    unique_food_list = _normalize_food_coords(food_coords, rows, cols)
    if not unique_food_list:
        return [], [], {}

    food_mask = np.zeros((rows, cols), dtype=bool)
    for r, c in unique_food_list:
        food_mask[r, c] = True

    passable = (~water_mask) & (~food_mask)

    gather_map: dict[tuple[int, int], int] = {}
    gathering_positions: list[tuple[int, int]] = []
    food_to_gather: list[list[int]] = [[] for _ in unique_food_list]
    gather_to_food: dict[int, list[int]] = {}

    for f_idx, (fr, fc) in enumerate(unique_food_list):
        for dr, dc in DELTAS:
            nr = (fr + dr) % rows
            nc = (fc + dc) % cols
            if passable[nr, nc]:
                pos = (nr, nc)
                if pos not in gather_map:
                    g_idx = len(gathering_positions)
                    gather_map[pos] = g_idx
                    gathering_positions.append(pos)
                    gather_to_food[g_idx] = []
                else:
                    g_idx = gather_map[pos]
                food_to_gather[f_idx].append(g_idx)
                gather_to_food[g_idx].append(f_idx)

    return gathering_positions, food_to_gather, gather_to_food


def extract_cluster_gathering_capacity(
    food_coords: list[tuple[int, int]] | list[list[int]] | set[tuple[int, int]] | np.ndarray | None,
    water_mask: np.ndarray,
    rows: int,
    cols: int,
) -> tuple[list[tuple[int, int]], dict[int, int]]:
    unique_food_list = _normalize_food_coords(food_coords, rows, cols)
    if not unique_food_list:
        return [], {}

    gathering_positions, food_to_gather, gather_to_food = extract_cluster_gathering_positions(
        unique_food_list, water_mask, rows, cols
    )
    if not gathering_positions:
        return [], {}

    # 构建食物连通簇：若两颗食物共享同一个采集位，或者切比雪夫距离 <= 1 则属于同一簇
    num_foods = len(unique_food_list)
    parent = list(range(num_foods))

    def find(i: int) -> int:
        path = []
        while parent[i] != i:
            path.append(i)
            i = parent[i]
        for p in path:
            parent[p] = i
        return i

    def union(i: int, j: int) -> None:
        root_i = find(i)
        root_j = find(j)
        if root_i != root_j:
            parent[root_i] = root_j

    # 1. 空间切比雪夫距离 <= 1（环面 8 邻域连通分量）合并
    food_map = {coord: idx for idx, coord in enumerate(unique_food_list)}
    for f_idx, (fr, fc) in enumerate(unique_food_list):
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nr = (fr + dr) % rows
                nc = (fc + dc) % cols
                other_idx = food_map.get((nr, nc))
                if other_idx is not None and other_idx > f_idx:
                    union(f_idx, other_idx)

    # 2. 共享采集位通道连通合并
    for g_idx, f_indices in gather_to_food.items():
        if len(f_indices) > 1:
            first = f_indices[0]
            for other in f_indices[1:]:
                union(first, other)

    # 3. 统计各个簇的食物集合与合法采集位并集
    cluster_foods: dict[int, list[int]] = {}
    for f_idx in range(num_foods):
        root = find(f_idx)
        cluster_foods.setdefault(root, []).append(f_idx)

    cluster_capacities: dict[int, int] = {}
    for cluster_id, f_indices in cluster_foods.items():
        g_set: set[int] = set()
        for f_idx in f_indices:
            for g_idx in food_to_gather[f_idx]:
                g_set.add(g_idx)
        # 容量公式：C_j = min(|G(R_j)|, |R_j|)
        cluster_capacities[cluster_id] = min(len(g_set), len(f_indices))

    return gathering_positions, cluster_capacities


def compute_single_source_torus_geodesic(
    target: tuple[int, int],
    passable_mask: np.ndarray,
    rows: int,
    cols: int,
) -> np.ndarray:
    dist = np.full((rows, cols), FAR, dtype=np.int32)
    tr, tc = int(target[0]), int(target[1])
    if not passable_mask[tr, tc]:
        return dist

    dist[tr, tc] = 0
    wave = np.zeros((rows, cols), dtype=bool)
    wave[tr, tc] = True
    current_depth = 0

    while wave.any():
        current_depth += 1
        spread = (
            np.roll(wave, 1, 0)
            | np.roll(wave, -1, 0)
            | np.roll(wave, 1, 1)
            | np.roll(wave, -1, 1)
        )
        step = spread & passable_mask & (dist == FAR)
        if not step.any():
            break
        dist[step] = current_depth
        wave = step

    return dist


def compute_batch_torus_geodesics(
    targets: list[tuple[int, int]],
    passable_mask: np.ndarray,
    rows: int,
    cols: int,
    max_depth: int | None = None,
    return_3d_tensor: bool = False,
) -> list[np.ndarray] | tuple[np.ndarray, list[np.ndarray]]:
    num_targets = len(targets)
    if num_targets == 0:
        return (np.empty((rows, cols, 0), dtype=np.int32), []) if return_3d_tensor else []

    dist = np.full((rows, cols, num_targets), FAR, dtype=np.int32)
    wave = np.zeros((rows, cols, num_targets), dtype=bool)

    for k, (tr, tc) in enumerate(targets):
        r, c = int(tr), int(tc)
        if passable_mask[r, c]:
            dist[r, c, k] = 0
            wave[r, c, k] = True

    visited = wave.copy()
    passable_broadcast = passable_mask[:, :, None]
    spread = np.empty_like(wave)

    current_depth = 0
    while wave.any() and (max_depth is None or current_depth < max_depth):
        current_depth += 1
        spread[:-1, :, :] = wave[1:, :, :]
        spread[-1, :, :] = wave[0, :, :]
        spread[1:, :, :] |= wave[:-1, :, :]
        spread[0, :, :] |= wave[-1, :, :]
        spread[:, :-1, :] |= wave[:, 1:, :]
        spread[:, -1, :] |= wave[:, 0, :]
        spread[:, 1:, :] |= wave[:, :-1, :]
        spread[:, 0, :] |= wave[:, -1, :]
        spread &= passable_broadcast

        step = spread & ~visited
        if not step.any():
            break
        visited |= step
        dist[step] = current_depth
        wave = step

    if return_3d_tensor:
        dist_list = [dist[:, :, k] for k in range(num_targets)]
        return dist, dist_list

    return [dist[:, :, k] for k in range(num_targets)]


ALL_ONES_U64 = np.uint64(0xFFFFFFFFFFFFFFFF)


def compute_cost_matrix_bitmask(
    ant_positions: list[tuple[int, int]] | np.ndarray,
    gathering_positions: list[tuple[int, int]],
    passable_mask: np.ndarray,
    rows: int,
    cols: int,
    max_depth: int = 16,
) -> np.ndarray:
    num_ants = len(ant_positions)
    num_gather = len(gathering_positions)
    if num_ants == 0 or num_gather == 0:
        return np.full((num_ants, num_gather), FAR, dtype=np.int32)

    cost_matrix = np.full((num_ants, num_gather), FAR, dtype=np.int32)

    ant_r = np.array([int(a[0]) for a in ant_positions], dtype=np.int32)
    ant_c = np.array([int(a[1]) for a in ant_positions], dtype=np.int32)

    num_blocks = (num_gather + 63) // 64
    wave = np.zeros((rows, cols, num_blocks), dtype=np.uint64)
    visited = np.zeros((rows, cols, num_blocks), dtype=np.uint64)

    for k, (tr, tc) in enumerate(gathering_positions):
        if passable_mask[tr, tc]:
            b = k // 64
            bit = np.uint64(1) << np.uint64(k % 64)
            wave[tr, tc, b] |= bit
            visited[tr, tc, b] |= bit

    wave_ants = wave[ant_r, ant_c, :]
    if wave_ants.any():
        for i in range(num_ants):
            for b in range(num_blocks):
                val = int(wave_ants[i, b])
                base_k = b * 64
                while val > 0:
                    lsb = val & -val
                    k = base_k + (lsb.bit_length() - 1)
                    if k < num_gather:
                        cost_matrix[i, k] = 0
                    val ^= lsb

    passable_3d = np.where(passable_mask[:, :, None], ALL_ONES_U64, np.uint64(0))
    spread = np.empty_like(wave)

    current_depth = 0
    while current_depth < max_depth:
        current_depth += 1
        spread[:-1, :, :] = wave[1:, :, :]
        spread[-1, :, :] = wave[0, :, :]
        spread[1:, :, :] |= wave[:-1, :, :]
        spread[0, :, :] |= wave[-1, :, :]
        spread[:, :-1, :] |= wave[:, 1:, :]
        spread[:, -1, :] |= wave[:, 0, :]
        spread[:, 1:, :] |= wave[:, :-1, :]
        spread[:, 0, :] |= wave[:, -1, :]
        spread &= passable_3d

        step = spread & ~visited
        if not step.any():
            break
        visited |= step
        wave = step

        step_ants = step[ant_r, ant_c, :]
        if step_ants.any():
            nonzero_ants, nonzero_blocks = np.nonzero(step_ants)
            for i, b in zip(nonzero_ants, nonzero_blocks):
                val = int(step_ants[i, b])
                base_k = b * 64
                while val > 0:
                    lsb = val & -val
                    k = base_k + (lsb.bit_length() - 1)
                    if k < num_gather and cost_matrix[i, k] == FAR:
                        cost_matrix[i, k] = current_depth
                    val ^= lsb

    return cost_matrix


class MCMFGraph:
    def __init__(self, num_nodes: int, max_edges: int = 50000) -> None:
        if num_nodes <= 0:
            raise ValueError(f"节点总数必须大于零，当前数值: {num_nodes}")
        self.n = num_nodes
        self.head = [-1] * num_nodes
        self.to = [0] * max_edges
        self.next_edge = [0] * max_edges
        self.cap = [0] * max_edges
        self.cost = [0] * max_edges
        self.rev = [0] * max_edges
        self.orig_cap = [0] * max_edges
        self.edge_count = 0

    def add_edge(self, u: int, v: int, cap: int, cost: int) -> None:
        if cap < 0:
            raise ValueError(f"弧容量不得为负数: ({u} -> {v}, cap={cap})")
        if u < 0 or u >= self.n or v < 0 or v >= self.n:
            raise ValueError(f"节点索引超出范围 [0, {self.n - 1}]: ({u} -> {v})")

        e1 = self.edge_count
        e2 = e1 + 1
        if e2 >= len(self.to):
            new_size = max(len(self.to) * 2, e2 + 1000)
            self.to.extend([0] * (new_size - len(self.to)))
            self.next_edge.extend([0] * (new_size - len(self.next_edge)))
            self.cap.extend([0] * (new_size - len(self.cap)))
            self.cost.extend([0] * (new_size - len(self.cost)))
            self.rev.extend([0] * (new_size - len(self.rev)))
            self.orig_cap.extend([0] * (new_size - len(self.orig_cap)))

        self.edge_count += 2

        self.to[e1] = v
        self.cap[e1] = cap
        self.orig_cap[e1] = cap
        self.cost[e1] = cost
        self.rev[e1] = e2
        self.next_edge[e1] = self.head[u]
        self.head[u] = e1

        self.to[e2] = u
        self.cap[e2] = 0
        self.orig_cap[e2] = 0
        self.cost[e2] = -cost
        self.rev[e2] = e1
        self.next_edge[e2] = self.head[v]
        self.head[v] = e2


def build_capacity_constrained_bipartite_graph(
    ant_positions: list[tuple[int, int]] | np.ndarray,
    food_positions: list[tuple[int, int]] | np.ndarray,
    gathering_positions: list[tuple[int, int]],
    geodesic_cost_matrix: np.ndarray,
    congestion_penalty_matrix: np.ndarray | None = None,
    gather_to_food: dict[int, list[int]] | None = None,
    rows: int | None = None,
    cols: int | None = None,
    max_candidates_per_ant: int | None = None,
    cluster_limit: int | None = None,
    g_per_cluster: int = 1,
) -> tuple[MCMFGraph, dict[str, int]]:
    num_ants = len(ant_positions)
    num_gather = len(gathering_positions)
    num_foods = len(food_positions)

    source = 0
    u_offset = 1
    g_in_offset = u_offset + num_ants
    g_out_offset = g_in_offset + num_gather
    f_offset = g_out_offset + num_gather
    sink = f_offset + num_foods
    total_nodes = sink + 1

    offsets = {
        "source": source,
        "sink": sink,
        "u_offset": u_offset,
        "g_in_offset": g_in_offset,
        "g_out_offset": g_out_offset,
        "f_offset": f_offset,
    }

    est_edges = (num_ants + num_ants * num_gather + num_gather + num_gather * 4 + num_foods) * 2 + 100
    graph = MCMFGraph(total_nodes, max_edges=max(est_edges, 1000))

    # 1. 源点至蚂蚁：容量 1，费用 0
    for i in range(num_ants):
        graph.add_edge(source, u_offset + i, 1, 0)

    # 几何推断采集位至食物映射（若未显式提供且具有尺寸）
    if gather_to_food is None and rows is not None and cols is not None and num_foods > 0:
        food_list = [(int(f[0]), int(f[1])) for f in food_positions]
        gather_to_food = {}
        for k, (gr, gc) in enumerate(gathering_positions):
            for f_idx, (fr, fc) in enumerate(food_list):
                dr = abs(gr - fr)
                dr = min(dr, rows - dr)
                dc = abs(gc - fc)
                dc = min(dc, cols - dc)
                if dr * dr + dc * dc <= 1:
                    gather_to_food.setdefault(k, []).append(f_idx)

    # 2. 蚂蚁至采集位入节点：容量 1，费用为测地距离 + 拥塞惩罚
    if cluster_limit is not None and gather_to_food is not None and num_foods > 0 and num_gather > 0:
        food_to_g: list[list[int]] = [[] for _ in range(num_foods)]
        for k, f_indices in gather_to_food.items():
            for f_idx in f_indices:
                if 0 <= f_idx < num_foods:
                    food_to_g[f_idx].append(k)

        ant_food_dist = np.full((num_ants, num_foods), FAR, dtype=np.int32)
        for f_idx in range(num_foods):
            g_list = food_to_g[f_idx]
            if g_list:
                ant_food_dist[:, f_idx] = np.min(geodesic_cost_matrix[:, g_list], axis=1)

        if num_foods <= cluster_limit:
            top_foods = np.tile(np.arange(num_foods), (num_ants, 1))
        else:
            top_foods = np.argpartition(ant_food_dist, cluster_limit, axis=1)[:, :cluster_limit]

        has_reachable = (geodesic_cost_matrix < FAR).any(axis=1)
        for i in range(num_ants):
            row_costs = geodesic_cost_matrix[i]

            candidate_set: set[int] = set()
            for f_idx in top_foods[i]:
                if ant_food_dist[i, f_idx] >= FAR:
                    continue
                g_list = food_to_g[f_idx]
                if len(g_list) <= g_per_cluster:
                    candidate_set.update(g_list)
                else:
                    best_g = sorted(g_list, key=lambda g: row_costs[g])[:g_per_cluster]
                    candidate_set.update(best_g)

            candidate_indices = [k for k in candidate_set if row_costs[k] < FAR]
            if len(candidate_indices) == 0 and has_reachable[i]:
                raise ValueError(f"蚂蚁 {i} 存在可达采集位但候选边集合为空，剪枝逻辑异常")

            for k in candidate_indices:
                base_cost = int(row_costs[k])
                penalty = 0
                if congestion_penalty_matrix is not None:
                    penalty = int(congestion_penalty_matrix[i, k])
                total_cost = base_cost + penalty
                graph.add_edge(u_offset + i, g_in_offset + k, 1, total_cost)

    else:
        if max_candidates_per_ant is not None and num_gather > max_candidates_per_ant:
            candidate_matrix = np.argpartition(
                geodesic_cost_matrix, max_candidates_per_ant, axis=1
            )[:, :max_candidates_per_ant]
        else:
            candidate_matrix = None

        has_reachable = (geodesic_cost_matrix < FAR).any(axis=1)

        for i in range(num_ants):
            row_costs = geodesic_cost_matrix[i]
            if candidate_matrix is not None:
                cand_k = candidate_matrix[i]
            else:
                cand_k = range(num_gather)

            candidate_indices = [k for k in cand_k if row_costs[k] < FAR]
            if len(candidate_indices) == 0 and has_reachable[i]:
                raise ValueError(f"蚂蚁 {i} 存在可达采集位但候选边集合为空，剪枝逻辑异常")

            for k in candidate_indices:
                base_cost = int(row_costs[k])
                penalty = 0
                if congestion_penalty_matrix is not None:
                    penalty = int(congestion_penalty_matrix[i, k])
                total_cost = base_cost + penalty
                graph.add_edge(u_offset + i, g_in_offset + k, 1, total_cost)

    # 3. 采集位内部约束弧 (G_in -> G_out)：容量 1，费用 0（实现方格独占物理排他）
    for k in range(num_gather):
        graph.add_edge(g_in_offset + k, g_out_offset + k, 1, 0)

    # 4. 采集位出节点到相邻食物节点：容量 1，费用 0
    if gather_to_food is not None:
        for k, f_indices in gather_to_food.items():
            for f_idx in f_indices:
                if 0 <= f_idx < num_foods:
                    graph.add_edge(g_out_offset + k, f_offset + f_idx, 1, 0)
    else:
        # 若未提供关联字典，则通过几何相邻关系推断
        if rows is not None and cols is not None:
            food_list = [(int(f[0]), int(f[1])) for f in food_positions]
            for k, (gr, gc) in enumerate(gathering_positions):
                for f_idx, (fr, fc) in enumerate(food_list):
                    dr = abs(gr - fr)
                    dr = min(dr, rows - dr)
                    dc = abs(gc - fc)
                    dc = min(dc, cols - dc)
                    if dr * dr + dc * dc <= 1:
                        graph.add_edge(g_out_offset + k, f_offset + f_idx, 1, 0)

    # 5. 食物节点到汇点：容量 1，费用 0（单个食物实体单回合仅由 1 只蚂蚁采集）
    for j in range(num_foods):
        graph.add_edge(f_offset + j, sink, 1, 0)

    return graph, offsets


_SSP_DIAL_BUCKETS: list[list[int]] = [[] for _ in range(512)]


def solve_mcmf_ssp(
    graph: MCMFGraph,
    source: int,
    sink: int,
    max_flow_limit: int | None = None,
) -> tuple[int, int, dict[tuple[int, int], int]]:
    n = graph.n
    head = graph.head
    to = graph.to
    cap = graph.cap
    cost = graph.cost
    rev = graph.rev
    next_edge = graph.next_edge
    edge_count = graph.edge_count

    total_flow = 0
    total_cost = 0
    INF = 10**9

    # 1. 快速初态负费用断言防线：线性步长 2 扁平扫描正向边
    for e in range(0, edge_count, 2):
        if cap[e] > 0 and cost[e] < 0:
            u = to[rev[e]]
            raise ValueError(
                f"初始有向弧存在负费用边，违背势能初值条件: ({u} -> {to[e]}, cost={cost[e]})"
            )

    potentials = [0] * n
    dist = [INF] * n
    visited = [0] * n
    curr_head = [0] * n
    token = 0

    MAX_BUCKET = 512
    buckets = _SSP_DIAL_BUCKETS
    for b in buckets:
        if b:
            b.clear()
    overflow_heap: list[int] = []

    # 2. 真多路增广 Dinic DFS（多分支连续推流 + 代次整数时间戳 + 残量出边剪枝）
    def dfs(u: int, push: int) -> int:
        if u == sink or push == 0:
            return push
        visited[u] = token
        pushed_total = 0
        e = curr_head[u]
        p_u = potentials[u]
        while e != -1:
            v = to[e]
            c = cap[e]
            if (
                c > 0
                and visited[v] != token
                and (curr_head[v] != -1 or v == sink)
                and (cost[e] + p_u - potentials[v] == 0)
            ):
                rem = push - pushed_total
                flow_in = rem if rem < c else c
                tr = dfs(v, flow_in)
                if tr > 0:
                    cap[e] -= tr
                    cap[rev[e]] += tr
                    pushed_total += tr
                    if cap[e] == 0:
                        curr_head[u] = next_edge[e]
                    else:
                        curr_head[u] = e
                    if pushed_total == push:
                        break
            e = next_edge[e]
            curr_head[u] = e
        return pushed_total

    try:
        while True:
            if max_flow_limit is not None and total_flow >= max_flow_limit:
                break

            for i in range(n):
                dist[i] = INF

            dist[source] = 0
            buckets[0].append(source)
            cur_d = 0
            max_d = 0

            # 3. Dial 桶队列最短路搜索 + 汇点就地出堆退出 + 整数打包溢出堆回退保底
            while True:
                while cur_d <= max_d:
                    b = buckets[cur_d]
                    if b:
                        break
                    cur_d += 1

                if cur_d <= max_d:
                    u = buckets[cur_d].pop()
                elif overflow_heap:
                    item = heapq.heappop(overflow_heap)
                    cur_d = item >> 20
                    u = item & 0xFFFFF
                else:
                    break

                if cur_d > dist[u]:
                    continue
                if u == sink:
                    for d in range(cur_d, min(max_d + 1, MAX_BUCKET)):
                        buckets[d].clear()
                    overflow_heap.clear()
                    break

                p_u = potentials[u]
                e = head[u]
                while e != -1:
                    if cap[e] > 0:
                        v = to[e]
                        rc = cost[e] + p_u - potentials[v]
                        if rc < 0:
                            raise ValueError(
                                f"势能更新后残量弧规约费用小于零，残量网络断言失败: ({u} -> {v}, red_cost={rc})"
                            )
                        nd = cur_d + rc
                        if nd < dist[v]:
                            dist[v] = nd
                            if nd < MAX_BUCKET:
                                buckets[nd].append(v)
                                if nd > max_d:
                                    max_d = nd
                            else:
                                heapq.heappush(overflow_heap, (nd << 20) | v)
                    e = next_edge[e]

            if dist[sink] == INF:
                for d in range(cur_d, min(max_d + 1, MAX_BUCKET)):
                    buckets[d].clear()
                overflow_heap.clear()
                break

            # 4. Johnson 势能更新与当前弧指针重置
            d_sink = dist[sink]
            for i in range(n):
                if dist[i] < d_sink:
                    potentials[i] += dist[i]
                else:
                    potentials[i] += d_sink
                curr_head[i] = head[i]

            # 5. 代次增广推进
            while max_flow_limit is None or total_flow < max_flow_limit:
                token += 1
                needed = INF if max_flow_limit is None else (max_flow_limit - total_flow)
                pushed = dfs(source, needed)
                if pushed == 0:
                    break
                total_flow += pushed
                total_cost += pushed * potentials[sink]
    finally:
        for b in buckets:
            if b:
                b.clear()

    # 6. 残量流字典线性提取
    flow_dict: dict[tuple[int, int], int] = {}
    orig_cap = graph.orig_cap
    for e in range(0, edge_count, 2):
        f = orig_cap[e] - cap[e]
        if f > 0:
            flow_dict[(to[rev[e]], to[e])] = f

    return total_flow, total_cost, flow_dict


def solve_mcmf_successive_shortest_path(
    nodes: int | list[int],
    source: int,
    sink: int,
    capacity_matrix: np.ndarray,
    cost_matrix: np.ndarray,
) -> tuple[int, dict[tuple[int, int], int]]:
    num_nodes = nodes if isinstance(nodes, int) else len(nodes)
    if capacity_matrix.shape != (num_nodes, num_nodes):
        raise ValueError(
            f"容量矩阵维度不匹配: 期望 ({num_nodes}, {num_nodes})，实际为 {capacity_matrix.shape}"
        )
    if cost_matrix.shape != (num_nodes, num_nodes):
        raise ValueError(
            f"费用矩阵维度不匹配: 期望 ({num_nodes}, {num_nodes})，实际为 {cost_matrix.shape}"
        )
    if (capacity_matrix < 0).any():
        raise ValueError("容量矩阵中存在负数容量，违背物理约束")

    us, vs = np.where(capacity_matrix > 0)
    edge_count = len(us)

    graph = MCMFGraph(num_nodes, max_edges=edge_count * 2 + 10)
    caps = capacity_matrix[us, vs]
    costs = cost_matrix[us, vs]

    for i in range(edge_count):
        graph.add_edge(int(us[i]), int(vs[i]), int(caps[i]), int(costs[i]))

    _, total_cost, flow_dict = solve_mcmf_ssp(graph, source, sink)
    return total_cost, flow_dict


def assign_ants_to_exclusive_gathering(
    flow_result: dict[tuple[int, int], int] | dict[int, int],
    ant_positions: list[tuple[int, int]] | np.ndarray,
    gathering_positions: list[tuple[int, int]],
    geodesic_fields: list[np.ndarray] | None = None,
    passable_mask: np.ndarray | None = None,
    rows: int | None = None,
    cols: int | None = None,
    return_candidates: bool = False,
) -> dict[int, int] | tuple[dict[int, list[int]], dict[int, int]]:
    num_ants = len(ant_positions)
    num_gather = len(gathering_positions)

    # 提取分配映射 ant_idx -> g_idx
    ant_to_gather: dict[int, int] = {}
    if flow_result:
        first_key = next(iter(flow_result.keys()))
        if isinstance(first_key, tuple):
            # 格式为 edge (u, v) -> flow
            u_offset = 1
            g_in_offset = u_offset + num_ants
            g_out_offset = g_in_offset + num_gather
            for (u, v), flow in flow_result.items():
                if flow > 0 and u_offset <= u < g_in_offset and g_in_offset <= v < g_out_offset:
                    ant_idx = u - u_offset
                    g_idx = v - g_in_offset
                    ant_to_gather[ant_idx] = g_idx
        else:
            # 格式为 ant_idx -> g_idx
            for k_int, v_int in flow_result.items():
                ant_to_gather[int(k_int)] = int(v_int)

    if not return_candidates:
        return ant_to_gather

    # 若需要同时生成负梯度动作候选序列
    if geodesic_fields is None or passable_mask is None or rows is None or cols is None:
        raise ValueError("生成动作候选序列需要提供完整的 geodesic_fields、passable_mask、rows 与 cols")

    candidate_sequences: dict[int, list[int]] = {}
    for ant_idx, g_idx in ant_to_gather.items():
        ar, ac = int(ant_positions[ant_idx][0]), int(ant_positions[ant_idx][1])
        target_pos = gathering_positions[g_idx]
        dist_field = geodesic_fields[g_idx]

        if (ar, ac) == target_pos:
            candidate_sequences[ant_idx] = [HOLD]
            continue

        cur_d = dist_field[ar, ac]
        valid_moves: list[tuple[int, int]] = []
        for a_idx, (dr, dc) in enumerate(DELTAS):
            nr = (ar + dr) % rows
            nc = (ac + dc) % cols
            if passable_mask[nr, nc]:
                step_d = dist_field[nr, nc]
                if step_d < cur_d:
                    valid_moves.append((step_d, a_idx))

        valid_moves.sort(key=lambda item: item[0])
        cand_seq = [act for _, act in valid_moves]
        cand_seq.append(HOLD)
        candidate_sequences[ant_idx] = cand_seq

    return candidate_sequences, ant_to_gather


def compute_corridor_width(
    water: np.ndarray,
    rows: int,
    cols: int,
    max_search_width: int = 8,
) -> np.ndarray:
    if not isinstance(water, np.ndarray):
        raise TypeError(f"water 必须为 numpy.ndarray 类型，实际为 {type(water)}")
    if np.issubdtype(water.dtype, np.floating):
        raise TypeError(f"water 严禁使用浮点数类型，实际为 {water.dtype}")
    if rows <= 0 or cols <= 0:
        raise ValueError(f"棋盘尺寸不合法: rows={rows}, cols={cols}")
    if water.shape != (rows, cols):
        raise ValueError(f"water 形状不匹配: 期望 ({rows}, {cols})，实际为 {water.shape}")
    if max_search_width < 1:
        raise ValueError(f"max_search_width 必须大于等于 1，实际为 {max_search_width}")

    passable = ~water
    if not passable.any():
        return np.zeros((rows, cols), dtype=np.int32)

    span_v = np.ones((rows, cols), dtype=np.int32)
    span_h = np.ones((rows, cols), dtype=np.int32)

    active_up = passable.copy()
    active_down = passable.copy()
    active_left = passable.copy()
    active_right = passable.copy()

    for _ in range(1, max_search_width):
        active_up = np.roll(active_up, -1, axis=0) & passable
        active_down = np.roll(active_down, 1, axis=0) & passable
        span_v += active_up.astype(np.int32) + active_down.astype(np.int32)

        active_left = np.roll(active_left, -1, axis=1) & passable
        active_right = np.roll(active_right, 1, axis=1) & passable
        span_h += active_left.astype(np.int32) + active_right.astype(np.int32)

    min_span = np.clip(np.minimum(span_v, span_h), 0, max_search_width)
    return np.where(passable, min_span, 0)


_CORRIDOR_WIDTH_CACHE: dict[tuple[int, int, int], np.ndarray] = {}


def clear_corridor_width_cache() -> None:
    _CORRIDOR_WIDTH_CACHE.clear()


def compute_corridor_width_grid(water: np.ndarray, use_cache: bool = True) -> np.ndarray:
    if not isinstance(water, np.ndarray):
        raise TypeError(f"water 必须为 numpy.ndarray 类型，实际为 {type(water)}")
    if np.issubdtype(water.dtype, np.floating):
        raise TypeError(f"water 严禁使用浮点数类型，实际为 {water.dtype}")
    if water.ndim != 2:
        raise ValueError(f"水域网格维度必须为 2，当前维度为 {water.ndim}")

    rows, cols = water.shape
    if rows <= 0 or cols <= 0:
        raise ValueError(f"棋盘尺寸不合法: rows={rows}, cols={cols}")

    cache_key = (id(water), rows, cols)
    if use_cache and cache_key in _CORRIDOR_WIDTH_CACHE:
        return _CORRIDOR_WIDTH_CACHE[cache_key].copy()

    water_bool = water.astype(bool)
    span_h = np.zeros((rows, cols), dtype=np.int32)
    span_v = np.zeros((rows, cols), dtype=np.int32)

    for r in range(rows):
        water_cols = np.where(water_bool[r])[0]
        if len(water_cols) == 0:
            span_h[r, :] = cols
        elif len(water_cols) == cols:
            span_h[r, :] = 0
        else:
            m = len(water_cols)
            for idx in range(m):
                w_curr = int(water_cols[idx])
                w_next = int(water_cols[(idx + 1) % m])
                length = (w_next - w_curr - 1) if idx + 1 < m else ((w_next + cols) - w_curr - 1)
                if length > 0:
                    if idx + 1 < m:
                        span_h[r, w_curr + 1 : w_next] = length
                    else:
                        span_h[r, w_curr + 1 :] = length
                        span_h[r, :w_next] = length

    for c in range(cols):
        water_rows = np.where(water_bool[:, c])[0]
        if len(water_rows) == 0:
            span_v[:, c] = rows
        elif len(water_rows) == rows:
            span_v[:, c] = 0
        else:
            m = len(water_rows)
            for idx in range(m):
                w_curr = int(water_rows[idx])
                w_next = int(water_rows[(idx + 1) % m])
                length = (w_next - w_curr - 1) if idx + 1 < m else ((w_next + rows) - w_curr - 1)
                if length > 0:
                    if idx + 1 < m:
                        span_v[w_curr + 1 : w_next, c] = length
                    else:
                        span_v[w_curr + 1 :, c] = length
                        span_v[:w_next, c] = length

    width_grid = np.minimum(span_h, span_v)
    width_grid = np.maximum(width_grid, 1)
    width_grid[water_bool] = 0

    if use_cache:
        _CORRIDOR_WIDTH_CACHE[cache_key] = width_grid
    return width_grid


def compute_local_ant_density(
    ant_positions: list[tuple[int, int]] | np.ndarray,
    passable_mask: np.ndarray,
    rows: int,
    cols: int,
    radius: int = 2,
) -> np.ndarray:
    if not isinstance(passable_mask, np.ndarray):
        raise TypeError(f"passable_mask 必须为 numpy.ndarray 类型，实际为 {type(passable_mask)}")
    if np.issubdtype(passable_mask.dtype, np.floating):
        raise TypeError(f"passable_mask 严禁使用浮点数类型，实际为 {passable_mask.dtype}")
    if rows <= 0 or cols <= 0:
        raise ValueError(f"棋盘尺寸不合法: rows={rows}, cols={cols}")
    if passable_mask.shape != (rows, cols):
        raise ValueError(f"passable_mask 形状不匹配: 期望 ({rows}, {cols})，实际为 {passable_mask.shape}")
    if radius < 0:
        raise ValueError(f"radius 必须为非负整数，实际为 {radius}")

    ant_grid = np.zeros((rows, cols), dtype=np.float32)
    for pos in ant_positions:
        r, c = int(pos[0]), int(pos[1])
        if r < 0 or r >= rows or c < 0 or c >= cols:
            raise ValueError(f"蚂蚁坐标 ({r}, {c}) 超出有效范围 ({rows}, {cols})")
        ant_grid[r, c] += 1.0

    temp_ant = np.zeros_like(ant_grid)
    for dr in range(-radius, radius + 1):
        temp_ant += np.roll(ant_grid, dr, axis=0)
    ant_count = np.zeros_like(temp_ant)
    for dc in range(-radius, radius + 1):
        ant_count += np.roll(temp_ant, dc, axis=1)

    passable_float = passable_mask.astype(np.float32)
    temp_pass = np.zeros_like(passable_float)
    for dr in range(-radius, radius + 1):
        temp_pass += np.roll(passable_float, dr, axis=0)
    passable_count = np.zeros_like(temp_pass)
    for dc in range(-radius, radius + 1):
        passable_count += np.roll(temp_pass, dc, axis=1)

    density = np.where(passable_mask, ant_count / np.maximum(1.0, passable_count), 0.0)
    return density


def compute_local_density_field(
    ant_positions: list[tuple[int, int]] | np.ndarray,
    rows: int,
    cols: int,
    passable_mask: np.ndarray | None = None,
    radius: int = 1,
) -> np.ndarray:
    if rows <= 0 or cols <= 0:
        raise ValueError(f"棋盘尺寸不合法: rows={rows}, cols={cols}")
    if radius < 0:
        raise ValueError(f"radius 必须为非负整数，实际为 {radius}")
    if passable_mask is not None and passable_mask.shape != (rows, cols):
        raise ValueError(f"passable_mask 形状不匹配: 期望 ({rows}, {cols})，实际为 {passable_mask.shape}")

    grid = np.zeros((rows, cols), dtype=np.float32)
    for pos in ant_positions:
        r, c = int(pos[0]), int(pos[1])
        if r < 0 or r >= rows or c < 0 or c >= cols:
            raise ValueError(f"蚂蚁坐标 ({r}, {c}) 超出有效范围 ({rows}, {cols})")
        grid[r, c] += 1.0

    if radius == 0:
        density = grid.copy()
    else:
        # 5 邻域十字平滑
        density = grid.copy()
        density += np.roll(grid, 1, axis=0)
        density += np.roll(grid, -1, axis=0)
        density += np.roll(grid, 1, axis=1)
        density += np.roll(grid, -1, axis=1)

    if passable_mask is not None:
        density[~passable_mask] = 0.0

    return density


def compute_dynamic_congestion_penalty(
    width_map: np.ndarray,
    density_map: np.ndarray,
    passable_mask: np.ndarray,
    alpha: float = 10.0,
    beta: float = 1.5,
    max_delay: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    if width_map.shape != density_map.shape or width_map.shape != passable_mask.shape:
        raise ValueError("width_map、density_map 与 passable_mask 形状必须一致")
    if alpha < 0.0:
        raise ValueError(f"alpha 必须为非负数，实际为 {alpha}")
    if beta <= 0.0:
        raise ValueError(f"beta 必须大于零，实际为 {beta}")
    if max_delay < 0:
        raise ValueError(f"max_delay 必须为非负整数，实际为 {max_delay}")

    raw_penalty = np.where(
        passable_mask,
        alpha * np.power(density_map / np.maximum(1, width_map), beta),
        0.0,
    )
    int_delay = np.clip(np.round(raw_penalty).astype(np.int32), 0, max_delay)
    return raw_penalty, int_delay


def compute_enemy_threat_penalty(
    foe_positions: list[tuple[int, int] | list[int]] | np.ndarray,
    friendly_positions: list[tuple[int, int] | list[int]] | np.ndarray,
    passable_mask: np.ndarray,
    rows: int,
    cols: int,
    width_map: np.ndarray | None = None,
    attack_radius_sq: int = 5,
    max_delay: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    if not isinstance(passable_mask, np.ndarray):
        raise TypeError(f"passable_mask 必须为 numpy.ndarray 类型，实际为 {type(passable_mask)}")
    if np.issubdtype(passable_mask.dtype, np.floating):
        raise TypeError(f"passable_mask 严禁使用浮点数类型，实际为 {passable_mask.dtype}")
    if rows <= 0 or cols <= 0:
        raise ValueError(f"棋盘尺寸不合法: rows={rows}, cols={cols}")
    if passable_mask.shape != (rows, cols):
        raise ValueError(f"passable_mask 形状不匹配: 期望 ({rows}, {cols})，实际为 {passable_mask.shape}")
    if width_map is not None and width_map.shape != (rows, cols):
        raise ValueError(f"width_map 形状不匹配: 期望 ({rows}, {cols})，实际为 {width_map.shape}")
    if attack_radius_sq < 0:
        raise ValueError(f"attack_radius_sq 必须为非负整数，实际为 {attack_radius_sq}")
    if max_delay < 0:
        raise ValueError(f"max_delay 必须为非负整数，实际为 {max_delay}")

    raw_threat = np.zeros((rows, cols), dtype=np.float32)
    int_delay = np.zeros((rows, cols), dtype=np.int32)
    if len(foe_positions) == 0:
        return raw_threat, int_delay

    foe_grid = np.zeros((rows, cols), dtype=np.int32)
    for pos in foe_positions:
        r, c = int(pos[0]), int(pos[1])
        if r < 0 or r >= rows or c < 0 or c >= cols:
            raise ValueError(f"敌方坐标 ({r}, {c}) 超出有效范围 ({rows}, {cols})")
        foe_grid[r, c] += 1

    friendly_grid = np.zeros((rows, cols), dtype=np.int32)
    for pos in friendly_positions:
        r, c = int(pos[0]), int(pos[1])
        if r < 0 or r >= rows or c < 0 or c >= cols:
            raise ValueError(f"己方坐标 ({r}, {c}) 超出有效范围 ({rows}, {cols})")
        friendly_grid[r, c] += 1

    max_offset = int(np.floor(np.sqrt(attack_radius_sq)))
    offsets = [
        (dr, dc)
        for dr in range(-max_offset, max_offset + 1)
        for dc in range(-max_offset, max_offset + 1)
        if dr * dr + dc * dc <= attack_radius_sq
    ]

    foe_coverage = np.zeros((rows, cols), dtype=np.int32)
    friendly_coverage = np.zeros((rows, cols), dtype=np.int32)
    for dr, dc in offsets:
        foe_coverage += np.roll(np.roll(foe_grid, dr, axis=0), dc, axis=1)
        friendly_coverage += np.roll(np.roll(friendly_grid, dr, axis=0), dc, axis=1)

    has_foe = (foe_coverage > 0) & passable_mask
    # 仅在严格以少打多（敌方火力覆盖 > 己方火力覆盖）时施加威胁斥力；
    # 1v1 等势或己方占优交战绝不施加斥力，防止策略畏战退缩。
    outnumbered = has_foe & (foe_coverage > friendly_coverage)

    raw_threat[outnumbered] = float(max_delay)

    if width_map is not None:
        chokepoint_threat = outnumbered & (width_map <= 1)
        raw_threat[chokepoint_threat] = float(max_delay)

    raw_threat[~passable_mask] = 0.0
    int_delay = np.clip(np.round(raw_threat).astype(np.int32), 0, max_delay)
    return raw_threat, int_delay


def compute_combined_delay(
    congestion_delay: np.ndarray,
    threat_delay: np.ndarray,
    max_delay: int = 4,
) -> np.ndarray:
    if congestion_delay.shape != threat_delay.shape:
        raise ValueError(f"延迟矩阵形状不匹配: {congestion_delay.shape} 与 {threat_delay.shape}")
    if max_delay < 0:
        raise ValueError(f"max_delay 必须为非负整数，实际为 {max_delay}")
    return np.clip(congestion_delay + threat_delay, 0, max_delay).astype(np.int32)


def trace_geodesic_allocation_path(
    start_pos: tuple[int, int] | list[int] | np.ndarray,
    target_pos: tuple[int, int] | list[int] | np.ndarray,
    dist_field: np.ndarray,
    passable_mask: np.ndarray,
    rows: int,
    cols: int,
) -> list[tuple[int, int]]:
    sr, sc = int(start_pos[0]), int(start_pos[1])
    tr, tc = int(target_pos[0]), int(target_pos[1])
    if not (0 <= sr < rows and 0 <= sc < cols):
        raise ValueError(f"start_pos ({sr}, {sc}) 超出棋盘有效范围 ({rows}, {cols})")
    if not (0 <= tr < rows and 0 <= tc < cols):
        raise ValueError(f"target_pos ({tr}, {tc}) 超出棋盘有效范围 ({rows}, {cols})")

    path: list[tuple[int, int]] = [(sr, sc)]
    cur = (sr, sc)
    if dist_field[sr, sc] >= FAR:
        return path

    max_steps = int(dist_field[sr, sc]) + 5

    for _ in range(max_steps):
        if cur == (tr, tc):
            break
        cr, cc = cur
        cur_d = dist_field[cr, cc]
        best_next = None
        best_d = cur_d

        for dr, dc in DELTAS:
            nr = (cr + dr) % rows
            nc = (cc + dc) % cols
            if passable_mask[nr, nc] and dist_field[nr, nc] < best_d:
                best_d = dist_field[nr, nc]
                best_next = (nr, nc)

        if best_next is None:
            break
        cur = best_next
        path.append(cur)

    return path


def compute_path_congestion_penalties(
    ant_positions: list[tuple[int, int]] | np.ndarray,
    gathering_positions: list[tuple[int, int]],
    geodesic_fields: list[np.ndarray] | dict[tuple[int, int], np.ndarray],
    width_grid: np.ndarray,
    density_grid: np.ndarray,
    passable_mask: np.ndarray,
    rows: int,
    cols: int,
    alpha: float = 2.0,
    beta: float = 1.5,
) -> np.ndarray:
    num_ants = len(ant_positions)
    num_gather = len(gathering_positions)
    penalty_matrix = np.zeros((num_ants, num_gather), dtype=np.int32)
    if num_ants == 0 or num_gather == 0:
        return penalty_matrix

    safe_width = np.maximum(width_grid, 1).astype(np.float32)
    unit_cost_field = np.zeros((rows, cols), dtype=np.float32)
    pos_mask = (density_grid > 0) & passable_mask
    unit_cost_field[pos_mask] = (density_grid[pos_mask] / safe_width[pos_mask]) ** beta

    for k in range(num_gather):
        target = gathering_positions[k]
        if isinstance(geodesic_fields, dict):
            dist_field = geodesic_fields[target]
        else:
            dist_field = geodesic_fields[k]

        for i in range(num_ants):
            ar, ac = int(ant_positions[i][0]), int(ant_positions[i][1])
            if dist_field[ar, ac] >= FAR or (ar, ac) == target:
                continue

            path = trace_geodesic_allocation_path((ar, ac), target, dist_field, passable_mask, rows, cols)
            if len(path) <= 1:
                continue

            accumulated_penalty = 0.0
            for step_r, step_c in path[1:]:
                accumulated_penalty += float(unit_cost_field[step_r, step_c])

            penalty_matrix[i, k] = max(0, int(round(alpha * accumulated_penalty)))

    return penalty_matrix


def compute_cost_matrix_bitmask_multibucket(
    ant_positions: list[tuple[int, int]] | np.ndarray,
    gathering_positions: list[tuple[int, int]],
    passable_mask: np.ndarray,
    cell_delay: np.ndarray,
    rows: int,
    cols: int,
    max_depth: int = 16,
    max_delay: int = 4,
) -> np.ndarray:
    num_ants = len(ant_positions)
    num_gather = len(gathering_positions)
    if num_ants == 0 or num_gather == 0:
        return np.full((num_ants, num_gather), FAR, dtype=np.int32)
    if passable_mask.shape != (rows, cols) or cell_delay.shape != (rows, cols):
        raise ValueError("passable_mask 与 cell_delay 形状必须为 (rows, cols)")

    cost_matrix = np.full((num_ants, num_gather), FAR, dtype=np.int32)
    ant_r = np.array([int(a[0]) for a in ant_positions], dtype=np.int32)
    ant_c = np.array([int(a[1]) for a in ant_positions], dtype=np.int32)

    num_blocks = (num_gather + 63) // 64
    num_buckets = max(2, max_delay + 2)

    buckets = [np.zeros((rows, cols, num_blocks), dtype=np.uint64) for _ in range(num_buckets)]
    visited = np.zeros((rows, cols, num_blocks), dtype=np.uint64)

    for k, (tr, tc) in enumerate(gathering_positions):
        if passable_mask[tr, tc]:
            b = k // 64
            bit = np.uint64(1) << np.uint64(k % 64)
            buckets[0][tr, tc, b] |= bit
            visited[tr, tc, b] |= bit

    delay_r, delay_c = np.nonzero(cell_delay > 0)
    has_any_delay = len(delay_r) > 0

    if has_any_delay:
        delay_clipped = np.clip(cell_delay, 0, max_delay)[:, :, None]
        delay_masks = [
            np.where(delay_clipped == d, ALL_ONES_U64, np.uint64(0)) for d in range(num_buckets)
        ]
        active_delays = [d for d in range(num_buckets) if (delay_clipped == d).any()]
    else:
        delay_masks = []
        active_delays = []

    passable_3d = np.where(passable_mask[:, :, None], ALL_ONES_U64, np.uint64(0))
    spread = np.empty((rows, cols, num_blocks), dtype=np.uint64)

    current_cost = 0
    max_cost_bound = max_depth
    empty_bucket_streak = 0

    while current_cost <= max_cost_bound:
        bucket_idx = current_cost % num_buckets
        wave = buckets[bucket_idx]

        if wave.any():
            empty_bucket_streak = 0
            step_ants = wave[ant_r, ant_c, :]
            if step_ants.any():
                nonzero_ants, nonzero_blocks = np.nonzero(step_ants)
                for i, b in zip(nonzero_ants, nonzero_blocks):
                    val = int(step_ants[i, b])
                    base_k = b * 64
                    while val > 0:
                        lsb = val & -val
                        k = base_k + (lsb.bit_length() - 1)
                        if k < num_gather and cost_matrix[i, k] == FAR:
                            cost_matrix[i, k] = current_cost
                        val ^= lsb

            spread[:-1, :, :] = wave[1:, :, :]
            spread[-1, :, :] = wave[0, :, :]
            spread[1:, :, :] |= wave[:-1, :, :]
            spread[0, :, :] |= wave[-1, :, :]
            spread[:, :-1, :] |= wave[:, 1:, :]
            spread[:, -1, :] |= wave[:, 0, :]
            spread[:, 1:, :] |= wave[:, :-1, :]
            spread[:, 0, :] |= wave[:, -1, :]
            spread &= passable_3d

            new_step = spread & ~visited
            if new_step.any():
                visited |= new_step
                target_b0 = (current_cost + 1) % num_buckets
                if has_any_delay and new_step[delay_r, delay_c, :].any():
                    for d in active_delays:
                        target_b = (current_cost + 1 + d) % num_buckets
                        target_wave = new_step & delay_masks[d]
                        if target_wave.any():
                            buckets[target_b] |= target_wave
                else:
                    buckets[target_b0] |= new_step

            buckets[bucket_idx].fill(np.uint64(0))
        else:
            empty_bucket_streak += 1
            if empty_bucket_streak >= num_buckets:
                break

        current_cost += 1

    return cost_matrix


def compute_batch_torus_geodesics_multibucket(
    targets: list[tuple[int, int]],
    passable_mask: np.ndarray,
    cell_delay: np.ndarray,
    rows: int,
    cols: int,
    max_depth: int | list[int] | np.ndarray = 16,
    max_delay: int = 4,
) -> list[np.ndarray]:
    num_targets = len(targets)
    if num_targets == 0:
        return []

    dist = np.full((rows, cols, num_targets), FAR, dtype=np.int32)
    num_buckets = max(2, max_delay + 2)
    buckets = [np.zeros((rows, cols, num_targets), dtype=bool) for _ in range(num_buckets)]
    visited = np.zeros((rows, cols, num_targets), dtype=bool)

    if isinstance(max_depth, (list, tuple, np.ndarray)):
        limits = np.asarray(max_depth, dtype=np.int32)
        if len(limits) != num_targets:
            raise ValueError(
                f"max_depth 长度 ({len(limits)}) 与 targets 数量 ({num_targets}) 不匹配"
            )
        max_cost_bound = int(np.max(limits)) if num_targets > 0 else 0
    else:
        limits = np.full(num_targets, max_depth, dtype=np.int32)
        max_cost_bound = int(max_depth)

    for k, (tr, tc) in enumerate(targets):
        r, c = int(tr), int(tc)
        if passable_mask[r, c]:
            dist[r, c, k] = 0
            if limits[k] > 0:
                buckets[0][r, c, k] = True
            visited[r, c, k] = True

    passable_3d = passable_mask[:, :, None]
    delay_clipped = np.clip(cell_delay, 0, max_delay)[:, :, None]
    delay_masks = [(delay_clipped == d) for d in range(num_buckets)]
    delay_r, delay_c = np.nonzero(cell_delay > 0)
    has_any_delay = len(delay_r) > 0
    active_delays = [d for d in range(num_buckets) if delay_masks[d].any()]

    spread = np.empty_like(visited)

    current_cost = 0
    empty_bucket_streak = 0

    while current_cost < max_cost_bound:
        b_idx = current_cost % num_buckets
        wave = buckets[b_idx]

        if wave.any():
            empty_bucket_streak = 0
            spread[:-1, :, :] = wave[1:, :, :]
            spread[-1, :, :] = wave[0, :, :]
            spread[1:, :, :] |= wave[:-1, :, :]
            spread[0, :, :] |= wave[-1, :, :]
            spread[:, :-1, :] |= wave[:, 1:, :]
            spread[:, -1, :] |= wave[:, 0, :]
            spread[:, 1:, :] |= wave[:, :-1, :]
            spread[:, 0, :] |= wave[:, -1, :]
            spread &= passable_3d

            new_step = spread & ~visited
            if new_step.any():
                visited |= new_step
                cost_next = current_cost + 1
                dist[new_step] = cost_next

                can_expand = (limits > cost_next)[None, None, :]
                expand_step = new_step & can_expand

                if expand_step.any():
                    target_b0 = cost_next % num_buckets
                    if has_any_delay and expand_step[delay_r, delay_c, :].any():
                        for d in active_delays:
                            target_b = (cost_next + d) % num_buckets
                            target_wave = expand_step & delay_masks[d]
                            if target_wave.any():
                                buckets[target_b] |= target_wave
                    else:
                        buckets[target_b0] |= expand_step

            buckets[b_idx].fill(False)
        else:
            empty_bucket_streak += 1
            if empty_bucket_streak >= num_buckets:
                break

        current_cost += 1

    return [dist[:, :, k] for k in range(num_targets)]


RF_RADIUS = 15
RF_DIAMETER = 31

_GRID_R, _GRID_C = np.indices((RF_DIAMETER, RF_DIAMETER), dtype=np.int32)
_OFFSET_DR = _GRID_R - RF_RADIUS
_OFFSET_DC = _GRID_C - RF_RADIUS
_DIST2 = _OFFSET_DR * _OFFSET_DR + _OFFSET_DC * _OFFSET_DC

_QUAD_N = (_OFFSET_DR < 0) & (np.abs(_OFFSET_DR) >= np.abs(_OFFSET_DC))
_QUAD_E = (_OFFSET_DC > 0) & (np.abs(_OFFSET_DC) > np.abs(_OFFSET_DR))
_QUAD_S = (_OFFSET_DR > 0) & (np.abs(_OFFSET_DR) >= np.abs(_OFFSET_DC))
_QUAD_W = (_OFFSET_DC < 0) & (np.abs(_OFFSET_DC) > np.abs(_OFFSET_DR))
_QUAD_MASKS = [_QUAD_N, _QUAD_E, _QUAD_S, _QUAD_W]

_NEAR_MASK = (_DIST2 >= 4) & (_DIST2 <= 25)
_NEAR_QUADS = [np.where(m & _NEAR_MASK) for m in _QUAD_MASKS]

_FAR_MASK = (_DIST2 >= 36) & (_DIST2 <= 225)
_FAR_QUADS = [np.where(m & _FAR_MASK) for m in _QUAD_MASKS]

_CARDINAL_R = np.array([RF_RADIUS - 1, RF_RADIUS, RF_RADIUS + 1, RF_RADIUS], dtype=np.int32)
_CARDINAL_C = np.array([RF_RADIUS, RF_RADIUS + 1, RF_RADIUS, RF_RADIUS - 1], dtype=np.int32)

_D4_ACTION_PERM = np.array([
    [0, 1, 2, 3, 4],
    [3, 0, 1, 2, 4],
    [2, 3, 0, 1, 4],
    [1, 2, 3, 0, 4],
    [0, 3, 2, 1, 4],
    [3, 2, 1, 0, 4],
    [2, 1, 0, 3, 4],
    [1, 0, 3, 2, 4],
], dtype=np.int32)


def extract_receptive_field_slice(
    board_planes: np.ndarray,
    ant_pos: tuple[int, int] | list[int] | np.ndarray,
    radius: int = RF_RADIUS,
    channel_indices: tuple[int, ...] | list[int] | None = None,
) -> np.ndarray:
    if not isinstance(board_planes, np.ndarray):
        raise TypeError(f"board_planes 必须为 numpy.ndarray 类型，实际为 {type(board_planes)}")
    if np.issubdtype(board_planes.dtype, np.floating):
        raise TypeError(f"board_planes 严禁使用浮点数类型，实际为 {board_planes.dtype}")
    if radius != RF_RADIUS:
        raise ValueError(f"radius 必须严格为 {RF_RADIUS}，实际为 {radius}")

    tensor = board_planes
    if tensor.ndim == 4:
        if tensor.shape[0] != 1:
            raise ValueError(f"四维张量批次维度必须为 1，实际为 {tensor.shape[0]}")
        tensor = tensor[0]

    if tensor.ndim != 3:
        raise ValueError(f"特征平面张量必须为 3 维，实际为 {tensor.ndim} 维")

    if tensor.shape[0] <= 16 and tensor.shape[1] >= 10 and tensor.shape[2] >= 10:
        c_dim, rows, cols = tensor.shape
    elif tensor.shape[2] <= 16 and tensor.shape[0] >= 10 and tensor.shape[1] >= 10:
        rows, cols, c_dim = tensor.shape
        tensor = np.transpose(tensor, (2, 0, 1))
    else:
        raise ValueError(f"无法推断通道排列维度，张量尺寸为 {tensor.shape}")

    if len(ant_pos) != 2:
        raise ValueError(f"ant_pos 必须包含两个坐标分量，实际长度为 {len(ant_pos)}")
    ar, ac = int(ant_pos[0]), int(ant_pos[1])
    if ar < 0 or ar >= rows or ac < 0 or ac >= cols:
        raise ValueError(f"ant_pos ({ar}, {ac}) 超出棋盘有效范围 ({rows}, {cols})")

    if channel_indices is not None:
        tensor = tensor[channel_indices]

    dr = np.arange(-radius, radius + 1, dtype=np.int32)
    dc = np.arange(-radius, radius + 1, dtype=np.int32)
    idx_r = (ar + dr) % rows
    idx_c = (ac + dc) % cols

    slice_c_hw = tensor[:, idx_r[:, None], idx_c[None, :]]
    return np.transpose(slice_c_hw, (1, 2, 0))


def extract_batch_receptive_field_slices(
    board_planes: np.ndarray,
    ant_positions: list[tuple[int, int]] | np.ndarray,
    radius: int = RF_RADIUS,
    channel_indices: tuple[int, ...] | list[int] | None = None,
) -> np.ndarray:
    if not isinstance(board_planes, np.ndarray):
        raise TypeError(f"board_planes 必须为 numpy.ndarray 类型，实际为 {type(board_planes)}")
    if np.issubdtype(board_planes.dtype, np.floating):
        raise TypeError(f"board_planes 严禁使用浮点数类型，实际为 {board_planes.dtype}")
    if radius != RF_RADIUS:
        raise ValueError(f"radius 必须严格为 {RF_RADIUS}，实际为 {radius}")

    tensor = board_planes
    if tensor.ndim == 4:
        if tensor.shape[0] != 1:
            raise ValueError(f"四维张量批次维度必须为 1，实际为 {tensor.shape[0]}")
        tensor = tensor[0]

    if tensor.ndim != 3:
        raise ValueError(f"特征平面张量必须为 3 维，实际为 {tensor.ndim} 维")

    if tensor.shape[0] <= 16 and tensor.shape[1] >= 10 and tensor.shape[2] >= 10:
        c_dim, rows, cols = tensor.shape
    elif tensor.shape[2] <= 16 and tensor.shape[0] >= 10 and tensor.shape[1] >= 10:
        rows, cols, c_dim = tensor.shape
        tensor = np.transpose(tensor, (2, 0, 1))
    else:
        raise ValueError(f"无法推断通道排列维度，张量尺寸为 {tensor.shape}")

    if channel_indices is not None:
        tensor = tensor[channel_indices]
    final_c = tensor.shape[0]

    if len(ant_positions) == 0:
        return np.empty((0, 2 * radius + 1, 2 * radius + 1, final_c), dtype=board_planes.dtype)

    ants_arr = np.asarray(ant_positions, dtype=np.int32)
    dr = np.arange(-radius, radius + 1, dtype=np.int32)
    dc = np.arange(-radius, radius + 1, dtype=np.int32)

    all_idx_r = (ants_arr[:, 0, None] + dr[None, :]) % rows
    all_idx_c = (ants_arr[:, 1, None] + dc[None, :]) % cols

    batch_c_n_hw = tensor[:, all_idx_r[:, :, None], all_idx_c[:, None, :]]
    return np.transpose(batch_c_n_hw, (1, 2, 3, 0))


def compute_single_patch_hash(patch_hwc: np.ndarray) -> int:
    if not isinstance(patch_hwc, np.ndarray):
        raise TypeError(f"patch_hwc 必须为 numpy.ndarray 类型，实际为 {type(patch_hwc)}")
    if patch_hwc.ndim != 3:
        raise ValueError(f"patch_hwc 维度必须为 3，实际为 {patch_hwc.ndim}")
    if patch_hwc.shape[0] <= 16 and patch_hwc.shape[1] == RF_DIAMETER and patch_hwc.shape[2] == RF_DIAMETER:
        patch_hwc = np.transpose(patch_hwc, (1, 2, 0))
    if patch_hwc.shape[0] != RF_DIAMETER or patch_hwc.shape[1] != RF_DIAMETER:
        raise ValueError(f"patch_hwc 空间尺寸必须为 ({RF_DIAMETER}, {RF_DIAMETER})，实际为 {patch_hwc.shape[:2]}")

    c_mine = 0
    c_foes = 1
    c_food = 2
    c_water = 3
    c_vis = 6 if patch_hwc.shape[2] > 6 else (patch_hwc.shape[2] - 1)

    w_bit = (patch_hwc[_CARDINAL_R, _CARDINAL_C, c_water] > 0).astype(np.int64)
    f_bit = (patch_hwc[_CARDINAL_R, _CARDINAL_C, c_food] > 0).astype(np.int64)
    m_bit = (patch_hwc[_CARDINAL_R, _CARDINAL_C, c_mine] > 0).astype(np.int64)
    e_bit = (patch_hwc[_CARDINAL_R, _CARDINAL_C, c_foes] > 0).astype(np.int64)

    s1_bits = (
        (w_bit[0]) | (w_bit[1] << 1) | (w_bit[2] << 2) | (w_bit[3] << 3)
        | (f_bit[0] << 4) | (f_bit[1] << 5) | (f_bit[2] << 6) | (f_bit[3] << 7)
        | (m_bit[0] << 8) | (m_bit[1] << 9) | (m_bit[2] << 10) | (m_bit[3] << 11)
        | (e_bit[0] << 12) | (e_bit[1] << 13) | (e_bit[2] << 14) | (e_bit[3] << 15)
    )

    s2_bits = 0
    shift = 16
    for qr, qc in _NEAR_QUADS:
        has_food = int((patch_hwc[qr, qc, c_food] > 0).any())
        has_foes = int((patch_hwc[qr, qc, c_foes] > 0).any())
        has_mine = int((patch_hwc[qr, qc, c_mine] > 0).any())
        s2_bits |= (has_food << shift) | (has_foes << (shift + 1)) | (has_mine << (shift + 2))
        shift += 3

    s3_bits = 0
    shift = 28
    for qr, qc in _FAR_QUADS:
        has_food = int((patch_hwc[qr, qc, c_food] > 0).any())
        has_foes = int((patch_hwc[qr, qc, c_foes] > 0).any())
        has_unseen = int((patch_hwc[qr, qc, c_vis] == 0).any())
        s3_bits |= (has_food << shift) | (has_foes << (shift + 1)) | (has_unseen << (shift + 2))
        shift += 3

    water_density = int(np.sum(patch_hwc[:, :, c_water] > 0))
    density_level = min(water_density // 64, 3)
    s4_bits = density_level << 40

    return int(s1_bits | s2_bits | s3_bits | s4_bits)


def compute_discrete_feature_hash(
    rf_slice: np.ndarray,
    canonicalize: bool = True,
    use_blake2b: bool = False,
) -> int:
    if not isinstance(rf_slice, np.ndarray):
        raise TypeError(f"rf_slice 必须为 numpy.ndarray 类型，实际为 {type(rf_slice)}")
    if rf_slice.ndim != 3:
        raise ValueError(f"rf_slice 维度必须为 3，实际为 {rf_slice.ndim}")
    if rf_slice.shape[0] <= 16 and rf_slice.shape[1] == RF_DIAMETER and rf_slice.shape[2] == RF_DIAMETER:
        rf_slice = np.transpose(rf_slice, (1, 2, 0))
    if rf_slice.shape[0] != RF_DIAMETER or rf_slice.shape[1] != RF_DIAMETER:
        raise ValueError(f"rf_slice 空间尺寸必须为 ({RF_DIAMETER}, {RF_DIAMETER})，实际为 {rf_slice.shape[:2]}")

    if not canonicalize:
        h = compute_single_patch_hash(rf_slice)
        if use_blake2b:
            digest = hashlib.blake2b(h.to_bytes(8, "little"), digest_size=8).digest()
            return int.from_bytes(digest, "little")
        return h

    min_hash: int | None = None
    for flip in (False, True):
        p_base = np.fliplr(rf_slice) if flip else rf_slice
        for k in range(4):
            p_rot = np.rot90(p_base, k=k, axes=(0, 1))
            h = compute_single_patch_hash(p_rot)
            if min_hash is None or h < min_hash:
                min_hash = h
    assert min_hash is not None
    if use_blake2b:
        digest = hashlib.blake2b(min_hash.to_bytes(8, "little"), digest_size=8).digest()
        return int.from_bytes(digest, "little")
    return min_hash


def canonicalize_receptive_field_and_action(
    rf_slice: np.ndarray,
    action: int,
) -> tuple[int, int]:
    if not isinstance(rf_slice, np.ndarray):
        raise TypeError(f"rf_slice 必须为 numpy.ndarray 类型，实际为 {type(rf_slice)}")
    if rf_slice.ndim != 3:
        raise ValueError(f"rf_slice 维度必须为 3，实际为 {rf_slice.ndim}")
    if rf_slice.shape[0] != RF_DIAMETER or rf_slice.shape[1] != RF_DIAMETER:
        raise ValueError(f"rf_slice 空间尺寸必须为 ({RF_DIAMETER}, {RF_DIAMETER})，实际为 {rf_slice.shape[:2]}")
    if not isinstance(action, (int, np.integer)) or action < 0 or action > 4:
        raise ValueError(f"action 编号必须在 [0, 4] 范围内，实际为 {action}")

    act_int = int(action)
    transforms: list[tuple[int, int]] = []
    t_idx = 0
    for flip in (False, True):
        p_base = np.fliplr(rf_slice) if flip else rf_slice
        for k in range(4):
            p_rot = np.rot90(p_base, k=k, axes=(0, 1))
            h = compute_single_patch_hash(p_rot)
            mapped_action = int(_D4_ACTION_PERM[t_idx, act_int])
            transforms.append((h, mapped_action))
            t_idx += 1

    min_h = min(item[0] for item in transforms)
    candidate_actions = [act for h, act in transforms if h == min_h]
    best_action = min(candidate_actions)
    return min_h, best_action


class JointCondition(NamedTuple):
    feature_hash: int
    role: int
    memory_state: int


def make_joint_condition(
    feature_hash: int,
    role: int,
    memory_state: int = 0,
) -> JointCondition:
    if not isinstance(feature_hash, (int, np.integer)):
        raise TypeError(f"feature_hash 必须为整数类型，实际类型为: {type(feature_hash)}")
    if not isinstance(role, (int, np.integer)) or role < 0:
        raise ValueError(f"role 必须为非负整数，实际数值为: {role}")
    if not isinstance(memory_state, (int, np.integer)) or memory_state < 0:
        raise ValueError(f"memory_state 必须为非负整数，实际数值为: {memory_state}")
    return JointCondition(int(feature_hash), int(role), int(memory_state))


class EmpiricalEntropyResult(float):
    def __new__(cls, value: float, metadata: dict):
        obj = super().__new__(cls, value)
        obj.metadata = metadata
        return obj

    def __iter__(self):
        yield float(self)
        yield self.metadata

    def __getitem__(self, index):
        if index == 0:
            return float(self)
        elif index == 1:
            return self.metadata
        raise IndexError("EmpiricalEntropyResult 仅包含 (entropy, metadata) 两个元素")


def compute_empirical_conditional_entropy(
    actions: np.ndarray | list[int],
    conditions: list[JointCondition] | list[int] | np.ndarray | list[tuple],
    roles: list[int] | np.ndarray | None = None,
    memories: list[int] | np.ndarray | None = None,
    optimal_action_sets: list[set[int]] | list[tuple[int, ...]] | None = None,
    min_category_samples: int = 10,
    optimal_equivalent_actions: list[set[int]] | None = None,
) -> EmpiricalEntropyResult:
    total_samples = len(actions)
    if total_samples != len(conditions):
        raise ValueError(
            f"动作序列长度 ({total_samples}) 与条件序列长度 ({len(conditions)}) 不一致"
        )
    if total_samples == 0:
        raise ValueError("输入样本序列为空，无法计算经验条件熵")
    if min_category_samples < 1:
        raise ValueError(f"最小样本支持门限必须大于等于 1，实际数值: {min_category_samples}")

    equiv_actions = optimal_action_sets if optimal_action_sets is not None else optimal_equivalent_actions
    if equiv_actions is not None and len(equiv_actions) != total_samples:
        raise ValueError(
            f"等价动作序列长度 ({len(equiv_actions)}) 与样本长度 ({total_samples}) 不一致"
        )

    # 规范化条件序列：支持 JointCondition、tuple 以及离散哈希整型列表
    cond_list: list[Any] = []
    if len(conditions) > 0 and isinstance(conditions[0], (int, np.integer)):
        for idx in range(total_samples):
            fh = int(conditions[idx])
            r = int(roles[idx]) if roles is not None else 0
            m = int(memories[idx]) if memories is not None else 0
            cond_list.append(JointCondition(fh, r, m))
    else:
        cond_list = list(conditions)

    category_counts: dict[Any, int] = {}
    action_counts: dict[Any, dict[int, int]] = {}

    for idx in range(total_samples):
        cond = cond_list[idx]
        if not isinstance(cond, (JointCondition, tuple, int, np.integer)):
            raise TypeError(f"条件元素必须为 JointCondition、tuple 或整数，实际为 {type(cond)}")
        act = int(actions[idx])
        if act < 0 or act > 4:
            raise ValueError(f"动作编号非法: {act}")

        if equiv_actions is not None:
            equiv_set = equiv_actions[idx]
            if act in equiv_set:
                act = min(equiv_set)

        category_counts[cond] = category_counts.get(cond, 0) + 1
        act_dist = action_counts.setdefault(cond, {})
        act_dist[act] = act_dist.get(act, 0) + 1

    valid_categories = {
        cond: count for cond, count in category_counts.items()
        if count >= min_category_samples
    }
    valid_samples_count = sum(valid_categories.values())

    if valid_samples_count == 0:
        raise ValueError("经过最小样本门限过滤后，无任何有效类别留存，样本规模不足")

    entropy_total = 0.0
    for cond, n_c in valid_categories.items():
        p_c = float(n_c) / float(valid_samples_count)
        act_dist = action_counts[cond]
        h_c = 0.0
        for act, n_ca in act_dist.items():
            p_a_given_c = float(n_ca) / float(n_c)
            if p_a_given_c > 0.0:
                h_c -= p_a_given_c * np.log2(p_a_given_c)
        entropy_total += p_c * h_c

    coverage_ratio = float(valid_samples_count) / float(total_samples)
    metadata = {
        "total_samples": total_samples,
        "valid_samples": valid_samples_count,
        "total_categories": len(category_counts),
        "valid_categories": len(valid_categories),
        "coverage_ratio": coverage_ratio,
        "min_category_samples": min_category_samples,
    }
    return EmpiricalEntropyResult(float(entropy_total), metadata)


class GateStatus(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"


def evaluate_gate3_1(
    conditional_entropy: float,
    threshold: float = 0.50,
    coverage_ratio: float = 1.0,
    min_coverage_ratio: float = 0.70,
    metadata: dict | None = None,
) -> tuple[GateStatus, dict]:
    if conditional_entropy < 0.0:
        raise ValueError(f"经验条件熵数值不得为负数，实际数值: {conditional_entropy}")
    if threshold <= 0.0:
        raise ValueError(f"门禁判定阈值必须为正数，实际数值: {threshold}")
    if not (0.0 < min_coverage_ratio <= 1.0):
        raise ValueError(f"最小样本覆盖率门限必须处于 (0.0, 1.0] 区间，实际数值: {min_coverage_ratio}")

    meta = metadata or {}
    coverage = float(meta.get("coverage_ratio", coverage_ratio))
    is_coverage_ok = coverage >= min_coverage_ratio
    is_entropy_ok = conditional_entropy <= threshold

    if is_entropy_ok and is_coverage_ok:
        status = GateStatus.PASS
        triggered = False
        action_msg = "条件熵符合局部可学习性标准，准许推进至后续流程"
    else:
        status = GateStatus.FAIL
        triggered = True
        action_msg = "经验条件熵超标或样本覆盖率不足，阻断流程并触发特征投影补偿机制"

    verdict_report = {
        "gate_id": "Gate 3-1",
        "status": status.value,
        "empirical_conditional_entropy": conditional_entropy,
        "threshold": threshold,
        "coverage_ratio": coverage,
        "min_coverage_ratio": min_coverage_ratio,
        "projection_compensation_triggered": triggered,
        "action_directive": action_msg,
        "telemetry_details": meta,
    }
    return status, verdict_report


def generate_flow_projection_plane(
    winning_targets: list[tuple[int, int]],
    winning_fields: list[np.ndarray],
    rows: int,
    cols: int,
    decay_tau: float = 6.0,
) -> np.ndarray:
    if rows <= 0 or cols <= 0:
        raise ValueError(f"棋盘尺寸不合法: rows={rows}, cols={cols}")
    if decay_tau <= 0.0:
        raise ValueError(f"衰减常数必须大于零，实际数值: {decay_tau}")
    if len(winning_targets) != len(winning_fields):
        raise ValueError(
            f"目标列表长度 ({len(winning_targets)}) 与测地场列表长度 ({len(winning_fields)}) 不匹配"
        )

    projection_plane = np.zeros((rows, cols), dtype=np.float32)
    for field in winning_fields:
        if field.shape != (rows, cols):
            raise ValueError(f"测地场尺寸不匹配: 期望 ({rows}, {cols})，实际为 {field.shape}")
        valid_mask = field < FAR
        if valid_mask.any():
            projection_plane[valid_mask] += np.exp(
                -field[valid_mask].astype(np.float32) / decay_tau
            )

    max_val = float(np.max(projection_plane))
    if max_val > 0.0:
        projection_plane /= max_val
    return projection_plane


def generate_feature_projection_pheromone_plane(
    winning_allocations: list[tuple],
    geodesic_fields: dict[tuple[int, int], np.ndarray] | list[np.ndarray],
    passable_mask: np.ndarray,
    rows: int,
    cols: int,
    decay_tau: float = 8.0,
    spread_radius: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    if rows <= 0 or cols <= 0:
        raise ValueError(f"棋盘尺寸不合法: rows={rows}, cols={cols}")
    if decay_tau <= 0.0:
        raise ValueError(f"decay_tau 必须大于零，实际数值: {decay_tau}")
    if spread_radius < 0:
        raise ValueError(f"spread_radius 必须为非负整数，实际数值: {spread_radius}")

    pheromone_plane = np.zeros((rows, cols), dtype=np.float32)

    for item in winning_allocations:
        if len(item) == 3:
            # (ant_idx, ant_pos, target_pos)
            _, ant_pos, target_pos = item
        elif len(item) == 2:
            # (ant_pos, target_pos)
            ant_pos, target_pos = item
        else:
            raise ValueError(f"不支持的 winning_allocation 元组格式: {item}")

        target_tuple = (int(target_pos[0]), int(target_pos[1]))
        if isinstance(geodesic_fields, dict):
            if target_tuple not in geodesic_fields:
                continue
            dist_field = geodesic_fields[target_tuple]
        else:
            if isinstance(target_pos, int) and target_pos < len(geodesic_fields):
                dist_field = geodesic_fields[target_pos]
            else:
                continue

        ar, ac = int(ant_pos[0]), int(ant_pos[1])
        if dist_field[ar, ac] >= FAR:
            continue

        path = trace_geodesic_allocation_path((ar, ac), target_tuple, dist_field, passable_mask, rows, cols)
        if len(path) == 0:
            continue

        for pr, pc in path:
            d_to_target = dist_field[pr, pc]
            base_intensity = float(np.exp(-d_to_target / decay_tau))

            for dr in range(-spread_radius, spread_radius + 1):
                for dc in range(-spread_radius, spread_radius + 1):
                    d2 = dr * dr + dc * dc
                    if d2 <= spread_radius * spread_radius:
                        nr = (pr + dr) % rows
                        nc = (pc + dc) % cols
                        if passable_mask[nr, nc]:
                            spatial_weight = float(np.exp(-d2 / 2.0))
                            val = base_intensity * spatial_weight
                            if val > pheromone_plane[nr, nc]:
                                pheromone_plane[nr, nc] = val

    max_val = float(np.max(pheromone_plane))
    if max_val > 0.0:
        pheromone_plane /= max_val

    gradient_directions = np.full((rows, cols), HOLD, dtype=np.int32)
    for r in range(rows):
        for c in range(cols):
            if not passable_mask[r, c] or pheromone_plane[r, c] == 0.0:
                continue
            cur_val = pheromone_plane[r, c]
            best_diff = 1e-4
            best_dir = HOLD

            for a_idx, (dr, dc) in enumerate(DELTAS):
                nr = (r + dr) % rows
                nc = (c + dc) % cols
                if passable_mask[nr, nc]:
                    diff = pheromone_plane[nr, nc] - cur_val
                    if diff > best_diff:
                        best_diff = diff
                        best_dir = a_idx

            gradient_directions[r, c] = best_dir

    return pheromone_plane, gradient_directions


