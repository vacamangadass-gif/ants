from __future__ import annotations

import multiprocessing as mp
import time
import numpy as np
import pytest

from tb_baselines.coordination import (
    DELTAS,
    HOLD,
)
from tb_baselines.mcmf_teacher import MCMFTeacher


def _worker_stress_task(worker_id: int, num_rounds: int, map_size: int, num_ants: int, num_foods: int) -> list[float]:
    rows, cols = map_size, map_size
    rng = np.random.default_rng(10000 + worker_id * 100)

    water = np.zeros((rows, cols), dtype=bool)
    for _ in range(6):
        wr = int(rng.integers(2, rows - 10))
        wc = int(rng.integers(2, cols - 10))
        water[wr : wr + 4, wc : wc + 4] = True
    seen = np.ones((rows, cols), dtype=bool)

    land_cells = [(r, c) for r in range(rows) for c in range(cols) if not water[r, c]]
    if len(land_cells) < num_ants + num_foods:
        raise ValueError(f"可用陆地方格数量不足: {len(land_cells)} < {num_ants + num_foods}")

    chosen_indices = rng.choice(len(land_cells), size=num_ants + num_foods, replace=False)
    current_ants = [land_cells[idx] for idx in chosen_indices[:num_ants]]
    foods = [land_cells[idx] for idx in chosen_indices[num_ants:]]

    teacher = MCMFTeacher(min_raid_ants=4, raid_ratio=0.5)
    latencies: list[float] = []

    for _ in range(num_rounds):
        obs = {
            "size": [rows, cols],
            "turn": 1,
            "max_turns": 300,
            "round_time": 0.05,
            "engine_digest": f"concurrency_worker_{worker_id}",
            "mine": [[int(r), int(c)] for r, c in current_ants],
            "ids": list(range(num_ants)),
            "food": [[int(r), int(c)] for r, c in foods],
            "hills": [],
        }

        t0 = time.perf_counter()
        orders = teacher.orders(obs, water, seen)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        latencies.append(elapsed_ms)

        if len(orders) != num_ants:
            raise ValueError(f"返回动作数量异常: {len(orders)} != {num_ants}")

        # 模拟环面网格移动推进
        occupied: set[tuple[int, int]] = set()
        next_ants: list[tuple[int, int]] = []
        for i, act in enumerate(orders):
            ar, ac = current_ants[i]
            if act != HOLD:
                dr, dc = DELTAS[act]
                dest = ((ar + dr) % rows, (ac + dc) % cols)
            else:
                dest = (ar, ac)
            if dest in occupied or water[dest[0], dest[1]]:
                dest = (ar, ac)
            occupied.add(dest)
            next_ants.append(dest)
        current_ants = next_ants

    return latencies


def test_concurrent_4_processes_cpu_contention_stress():
    # 验证 4 进程并行高负载竞争下的单回合决策延迟门禁 (< 15.0 ms)
    num_workers = 4
    rounds_per_worker = 30
    map_size = 64
    num_ants = 100
    num_foods = 30

    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=num_workers) as pool:
        args = [
            (wid, rounds_per_worker, map_size, num_ants, num_foods)
            for wid in range(num_workers)
        ]
        results = pool.starmap(_worker_stress_task, args)

    all_latencies: list[float] = []
    worker_means: list[float] = []
    for wid, lats in enumerate(results):
        w_mean = float(np.mean(lats))
        w_p95 = float(np.percentile(lats, 95))
        worker_means.append(w_mean)
        all_latencies.extend(lats)
        print(f"\n[4进程并发 进程 {wid}] 平均耗时: {w_mean:.2f} ms | 95分位: {w_p95:.2f} ms")

    global_mean = float(np.mean(all_latencies))
    global_p95 = float(np.percentile(all_latencies, 95))
    global_max = float(np.max(all_latencies))

    print(f"\n[4 进程全量 {len(all_latencies)} 轮综合统计] 100 蚂蚁 30 食物 64x64:")
    print(f"  全局平均耗时: {global_mean:.2f} ms")
    print(f"  全局 95分位耗时: {global_p95:.2f} ms")
    print(f"  全局最大单次耗时: {global_max:.2f} ms")

    assert global_mean < 15.0, f"4 进程并发全局平均耗时超出 15.0 ms 门禁: 实际为 {global_mean:.2f} ms"


def test_concurrent_8_processes_cpu_contention_stress():
    # 验证 8 进程高并发 CPU 竞争背景下的单回合决策延迟门禁 (< 15.0 ms)
    num_workers = 8
    rounds_per_worker = 30
    map_size = 64
    num_ants = 100
    num_foods = 30

    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=num_workers) as pool:
        args = [
            (wid, rounds_per_worker, map_size, num_ants, num_foods)
            for wid in range(num_workers)
        ]
        results = pool.starmap(_worker_stress_task, args)

    all_latencies: list[float] = []
    worker_means: list[float] = []
    for wid, lats in enumerate(results):
        w_mean = float(np.mean(lats))
        w_p95 = float(np.percentile(lats, 95))
        worker_means.append(w_mean)
        all_latencies.extend(lats)
        print(f"\n[8进程并发 进程 {wid}] 平均耗时: {w_mean:.2f} ms | 95分位: {w_p95:.2f} ms")

    global_mean = float(np.mean(all_latencies))
    global_p95 = float(np.percentile(all_latencies, 95))
    global_max = float(np.max(all_latencies))

    print(f"\n[8 进程全量 {len(all_latencies)} 轮综合统计] 100 蚂蚁 30 食物 64x64:")
    print(f"  全局平均耗时: {global_mean:.2f} ms")
    print(f"  全局 95分位耗时: {global_p95:.2f} ms")
    print(f"  全局最大单次耗时: {global_max:.2f} ms")

    assert global_mean < 15.0, f"8 进程并发全局平均耗时超出 15.0 ms 门禁: 实际为 {global_mean:.2f} ms"


def test_concurrent_8_processes_80x80_large_map_stress():
    # 验证 8 进程在 80x80 大尺寸地图下的单回合平均决策延迟门禁 (< 15.0 ms)
    num_workers = 8
    rounds_per_worker = 15
    map_size = 80
    num_ants = 100
    num_foods = 30

    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=num_workers) as pool:
        args = [
            (wid, rounds_per_worker, map_size, num_ants, num_foods)
            for wid in range(num_workers)
        ]
        results = pool.starmap(_worker_stress_task, args)

    all_latencies: list[float] = []
    for wid, lats in enumerate(results):
        w_mean = float(np.mean(lats))
        w_p95 = float(np.percentile(lats, 95))
        all_latencies.extend(lats)
        print(f"\n[8进程 80x80 进程 {wid}] 平均耗时: {w_mean:.2f} ms | 95分位: {w_p95:.2f} ms")

    global_mean = float(np.mean(all_latencies))
    global_p95 = float(np.percentile(all_latencies, 95))
    global_max = float(np.max(all_latencies))

    print(f"\n[8 进程 80x80 全量 {len(all_latencies)} 轮综合统计] 100 蚂蚁 30 食物 80x80:")
    print(f"  全局平均耗时: {global_mean:.2f} ms")
    print(f"  全局 95分位耗时: {global_p95:.2f} ms")
    print(f"  全局最大单次耗时: {global_max:.2f} ms")

    assert global_mean < 15.0, f"8 进程 80x80 大地图并发全局平均耗时超出 15.0 ms 门禁: 实际为 {global_mean:.2f} ms"


def test_memory_stability_and_zero_growth_over_long_horizon():
    # 验证长程推演过程中内存稳定性，杜绝内存持续泄漏与数组无限膨胀
    import tracemalloc

    rows, cols = 64, 64
    rng = np.random.default_rng(99999)

    water = np.zeros((rows, cols), dtype=bool)
    water[10:18, 10:18] = True
    water[35:43, 35:43] = True
    seen = np.ones((rows, cols), dtype=bool)

    land_cells = [(r, c) for r in range(rows) for c in range(cols) if not water[r, c]]
    chosen_indices = rng.choice(len(land_cells), size=130, replace=False)
    current_ants = [land_cells[idx] for idx in chosen_indices[:100]]
    foods = [land_cells[idx] for idx in chosen_indices[100:]]

    teacher = MCMFTeacher(min_raid_ants=4, raid_ratio=0.5)

    # 预热 20 回合使内部缓冲区达到稳定状态
    for _ in range(20):
        obs = {
            "size": [rows, cols],
            "turn": 1,
            "max_turns": 300,
            "round_time": 0.05,
            "engine_digest": "mem_warmup",
            "mine": [[int(r), int(c)] for r, c in current_ants],
            "ids": list(range(100)),
            "food": [[int(r), int(c)] for r, c in foods],
            "hills": [],
        }
        orders = teacher.orders(obs, water, seen)

    tracemalloc.start()
    snapshot_before = tracemalloc.take_snapshot()

    # 连续运行 150 回合长程推演
    for turn in range(150):
        obs = {
            "size": [rows, cols],
            "turn": turn + 1,
            "max_turns": 300,
            "round_time": 0.05,
            "engine_digest": "mem_stress",
            "mine": [[int(r), int(c)] for r, c in current_ants],
            "ids": list(range(100)),
            "food": [[int(r), int(c)] for r, c in foods],
            "hills": [],
        }
        orders = teacher.orders(obs, water, seen)
        if len(orders) != 100:
            raise ValueError("指令长度异常")

    import gc
    gc.collect()
    snapshot_after = tracemalloc.take_snapshot()
    tracemalloc.stop()

    top_stats = snapshot_after.compare_to(snapshot_before, "lineno")
    total_diff_kb = sum(stat.size_diff for stat in top_stats) / 1024.0

    print(f"\n[150 回合长程内存增长统计] 内存差异: {total_diff_kb:.2f} KB")
    for stat in top_stats[:10]:
        print(f"  {stat}")

    # 验证停滞计数器与内部集合规模受控，杜绝无界增长
    assert len(teacher.stall_counters) <= 100, f"停滞计数器发生膨胀: {len(teacher.stall_counters)}"
    assert total_diff_kb < 512.0, f"长程推演存在显著内存泄漏: 增长 {total_diff_kb:.2f} KB"


def _cpu_burner_process(stop_event: mp.Event) -> None:
    import os
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    # 模拟外部独立进程 CPU 密集运算任务，每个进程占用 1 个 CPU 核心
    x = 0
    while not stop_event.is_set():
        for i in range(10000):
            x = (x + i * 3) ^ 0x5A5A5A5A



def test_continuous_external_cpu_burn_contention_stress():
    # 启动 8 个后台高负荷 CPU 争用进程，在 100% 算力挤压下实测单回合决策耗时是否满足 < 15.0 ms
    num_burners = 8
    ctx = mp.get_context("spawn")
    stop_event = ctx.Event()

    burners = [
        ctx.Process(target=_cpu_burner_process, args=(stop_event,))
        for _ in range(num_burners)
    ]
    for p in burners:
        p.start()

    try:
        # 在高负荷竞争下运行 50 回合端到端决策
        rows, cols = 64, 64
        rng = np.random.default_rng(88888)

        water = np.zeros((rows, cols), dtype=bool)
        water[12:20, 12:20] = True
        water[40:48, 40:48] = True
        seen = np.ones((rows, cols), dtype=bool)

        land_cells = [(r, c) for r in range(rows) for c in range(cols) if not water[r, c]]
        chosen_indices = rng.choice(len(land_cells), size=130, replace=False)
        current_ants = [land_cells[idx] for idx in chosen_indices[:100]]
        foods = [land_cells[idx] for idx in chosen_indices[100:]]

        teacher = MCMFTeacher(min_raid_ants=4, raid_ratio=0.5)
        latencies: list[float] = []

        for turn in range(50):
            obs = {
                "size": [rows, cols],
                "turn": turn + 1,
                "max_turns": 300,
                "round_time": 0.05,
                "engine_digest": "cpu_burn_stress",
                "mine": [[int(r), int(c)] for r, c in current_ants],
                "ids": list(range(100)),
                "food": [[int(r), int(c)] for r, c in foods],
                "hills": [],
            }

            t0 = time.perf_counter()
            orders = teacher.orders(obs, water, seen)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            latencies.append(elapsed_ms)

            if len(orders) != 100:
                raise ValueError("指令生成数量异常")

            occupied: set[tuple[int, int]] = set()
            next_ants: list[tuple[int, int]] = []
            for i, act in enumerate(orders):
                ar, ac = current_ants[i]
                if act != HOLD:
                    dr, dc = DELTAS[act]
                    dest = ((ar + dr) % rows, (ac + dc) % cols)
                else:
                    dest = (ar, ac)
                if dest in occupied or water[dest[0], dest[1]]:
                    dest = (ar, ac)
                occupied.add(dest)
                next_ants.append(dest)
            current_ants = next_ants

        mean_ms = float(np.mean(latencies))
        p95_ms = float(np.percentile(latencies, 95))
        max_ms = float(np.max(latencies))

        print(f"\n[8 进程外部高负荷 CPU 争用背景下 50 回合推演]")
        print(f"  平均单回合耗时: {mean_ms:.2f} ms")
        print(f"  95分位耗时:     {p95_ms:.2f} ms")
        print(f"  最大单回合耗时: {max_ms:.2f} ms")

        assert mean_ms < 15.0, f"8 核心 CPU 争用背景下平均耗时超出 15.0 ms 门禁: 实际为 {mean_ms:.2f} ms"
    finally:
        stop_event.set()
        for p in burners:
            p.join(timeout=2.0)
            if p.is_alive():
                p.terminate()

