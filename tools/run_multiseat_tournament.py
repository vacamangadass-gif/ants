from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import subprocess
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CLI = os.environ.get("TINYBRAINS") or shutil.which("tinybrains") or "/home/linuxbrew/.linuxbrew/bin/tinybrains"
DEFAULT_REGISTRY = os.environ.get("TINYBRAINS_REGISTRY") or str((ROOT / "baselines" / "games.toml").resolve())
EXTRACTOR = ROOT / "benchmarks" / "extract_frames.mjs"

MODEL_SPECS: dict[str, str] = {
    "exp085": "多席位 MCMF 协同优化模型 (Micro 28 通道)",
    "exp083": "前代 MCMF 蒸馏模型 (Micro 28 通道)",
    "exp033": "DAgger 8 轮聚合天梯标杆 (Micro 28 通道)",
    "exp021": "激进进攻型基准 (Micro 28 通道)",
    "exp011": "防守采食型基准 (Nano 14 通道)",
}

BOARDS: dict[str, dict[str, Any]] = {
    "2p": {
        "seat_count": 2,
        "name": "basic-tiny-2p",
        "path": ROOT / "maps" / "basic-tiny-2p.json",
        "base_lineup": ["exp085", "exp033"],
    },
    "3p": {
        "seat_count": 3,
        "name": "basic-small-3p",
        "path": ROOT / "maps" / "basic-small-3p.json",
        "base_lineup": ["exp085", "exp033", "exp021"],
    },
    "4p": {
        "seat_count": 4,
        "name": "basic-medium-4p",
        "path": ROOT / "maps" / "basic-medium-4p.json",
        "base_lineup": ["exp085", "exp033", "exp021", "exp011"],
    },
    "6p": {
        "seat_count": 6,
        "name": "basic-large-6p",
        "path": ROOT / "maps" / "basic-large-6p.json",
        "base_lineup": ["exp085", "exp033", "exp021", "exp011", "exp083", "exp033"],
    },
    "8p": {
        "seat_count": 8,
        "name": "basic-xlarge-8p",
        "path": ROOT / "maps" / "basic-xlarge-8p.json",
        "base_lineup": ["exp085", "exp033", "exp021", "exp011", "exp085", "exp033", "exp021", "exp083"],
    },
}


def build_models_registry(models_root: Path | None = None) -> dict[str, dict[str, Any]]:
    base = (models_root or (ROOT / "models")).resolve()
    return {
        name: {
            "name": name,
            "description": desc,
            "weights": base / name / "model.onnx",
            "manifest": base / name / "manifest.json",
        }
        for name, desc in MODEL_SPECS.items()
    }


MODELS = build_models_registry()


def _torus_manhattan(a: tuple[int, int], b: tuple[int, int], rows: int, cols: int) -> int:
    dr = abs(a[0] - b[0])
    dc = abs(a[1] - b[1])
    return min(dr, rows - dr) + min(dc, cols - dc)


def compute_board_neighbor_offsets(board_path: Path) -> list[int]:
    # 计算棋盘上席位 0 几何最近邻的相对席位偏移集合，用于构造邻接均衡的席位排列
    data = json.loads(board_path.read_text(encoding="utf-8"))
    rows = int(data["rows"])
    cols = int(data["cols"])
    players = int(data["players"])
    hills = data["hills"]
    by_seat: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for idx, pos in enumerate(hills):
        by_seat[idx % players].append((int(pos[0]), int(pos[1])))

    dists: dict[int, int] = {}
    for s in range(1, players):
        dists[s] = min(
            _torus_manhattan(h0, hs, rows, cols)
            for h0 in by_seat[0]
            for hs in by_seat[s]
        )

    min_d = min(dists.values())
    # 纳入距离在最近邻 1.15 倍以内的全部直接交火邻接偏移（如 8p 棋盘中的 44 与 46 步两档共 4 个紧邻）
    threshold = math.ceil(min_d * 1.15)
    return sorted(s for s, d in dists.items() if d <= threshold)


def _lineup_neighbor_counts(lineup: list[str], neighbor_offsets: list[int]) -> Counter[tuple[str, str]]:
    n = len(lineup)
    counts: Counter[tuple[str, str]] = Counter()
    for i, m_a in enumerate(lineup):
        for off in neighbor_offsets:
            m_b = lineup[(i + off) % n]
            counts[(m_a, m_b)] += 1
    return counts


def generate_neighbor_balanced_lineups(
    base_lineup: list[str],
    rounds: int,
    seed: int,
    neighbor_offsets: list[int],
) -> list[list[str]]:
    # 通过贪心均衡排列消除固定循环移位导致的永久邻接偏差，使各模型对每个对手的近邻暴露频次均衡
    n = len(base_lineup)
    if rounds <= 0:
        raise ValueError(f"rounds 必须大于等于 1，实际为 {rounds}")
    if n <= 3:
        return [list(base_lineup) for _ in range(rounds)]

    rng = random.Random(seed)
    counts_in_lineup = Counter(base_lineup)
    unique_models = sorted(counts_in_lineup.keys())
    deg = len(neighbor_offsets)

    # 理想单轮邻接期望：每个 m_a 席位的 deg 个近邻按剩余席位比例均匀分布给各 m_b
    ideal_per_round: dict[tuple[str, str], float] = {}
    for m_a in unique_models:
        c_a = counts_in_lineup[m_a]
        for m_b in unique_models:
            rem_b = counts_in_lineup[m_b] - (1 if m_a == m_b else 0)
            ideal_per_round[(m_a, m_b)] = c_a * deg * (rem_b / max(n - 1, 1))

    # 生成候选排列池：包含基准排列、两两交换排列与确定性随机排列（固定 0 号位消除纯循环同构）
    seen_canonical: set[tuple[str, ...]] = set()
    candidates: list[list[str]] = []

    def add_cand(seq: list[str]) -> None:
        key = tuple(seq)
        if key not in seen_canonical:
            seen_canonical.add(key)
            candidates.append(seq)

    add_cand(list(base_lineup))
    for i in range(n):
        for j in range(i + 1, n):
            if base_lineup[i] != base_lineup[j]:
                swapped = list(base_lineup)
                swapped[i], swapped[j] = swapped[j], swapped[i]
                add_cand(swapped)
    for _ in range(400):
        perm = list(base_lineup)
        rng.shuffle(perm)
        add_cand(perm)

    cand_adj = [_lineup_neighbor_counts(c, neighbor_offsets) for c in candidates]
    cum_adj: Counter[tuple[str, str]] = Counter()
    chosen: list[list[str]] = []
    used_indices: set[int] = set()

    for r in range(1, rounds + 1):
        best_idx = 0
        best_loss = float("inf")
        for idx, adj in enumerate(cand_adj):
            # 优先选择尚未重复使用过的相对排列，同时最小化累计近邻频次与理想均匀分布的平方误差
            repeat_penalty = 0.05 if idx in used_indices else 0.0
            loss = repeat_penalty
            for m_a in unique_models:
                c_a = counts_in_lineup[m_a]
                for m_b in unique_models:
                    actual = cum_adj[(m_a, m_b)] + adj[(m_a, m_b)]
                    target = ideal_per_round[(m_a, m_b)] * r
                    diff = (actual - target) / max(c_a, 1)
                    loss += diff * diff
            if loss < best_loss:
                best_loss = loss
                best_idx = idx
        chosen.append(list(candidates[best_idx]))
        used_indices.add(best_idx)
        cum_adj.update(cand_adj[best_idx])

    return chosen


def create_seat_descriptor(
    seat_index: int,
    model_key: str,
    models_registry: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    reg = models_registry or MODELS
    model_info = reg[model_key]
    weights_path = model_info["weights"].resolve()
    manifest_path = model_info["manifest"].resolve()
    if not weights_path.exists():
        raise FileNotFoundError(f"未找到权重文件: {weights_path}")
    if not manifest_path.exists():
        raise FileNotFoundError(f"未找到清单文件: {manifest_path}")
    return {
        "seat": seat_index,
        "label": model_info["name"],
        "weights": str(weights_path),
        "manifest": str(manifest_path),
    }


def create_cyclic_shift_seats(
    base_models: list[str],
    shift: int,
    models_registry: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    seat_count = len(base_models)
    seats = []
    for base_seat, model_key in enumerate(base_models):
        seat_idx = (base_seat + shift) % seat_count
        seats.append(create_seat_descriptor(seat_idx, model_key, models_registry=models_registry))
    seats.sort(key=lambda s: s["seat"])
    return seats


def build_multiseat_tournament_rows(
    rounds: int = 4,
    base_seed: int = 85000,
    models_root: Path | None = None,
) -> list[dict[str, Any]]:
    reg = build_models_registry(models_root) if models_root is not None else MODELS
    rows: list[dict[str, Any]] = []

    balanced_by_board: dict[str, list[list[str]]] = {}
    for seat_key, board_info in BOARDS.items():
        offsets = compute_board_neighbor_offsets(board_info["path"])
        balanced_by_board[seat_key] = generate_neighbor_balanced_lineups(
            base_lineup=board_info["base_lineup"],
            rounds=rounds,
            seed=base_seed + board_info["seat_count"] * 97,
            neighbor_offsets=offsets,
        )

    for r in range(rounds):
        round_seed = base_seed + r * 1000
        for seat_key, board_info in BOARDS.items():
            board_path = str(board_info["path"].resolve())
            seat_count = board_info["seat_count"]
            lineup = balanced_by_board[seat_key][r]
            for shift in range(seat_count):
                match_id = f"m3-multi-{seat_key}-r{r}-s{round_seed + shift}-shift{shift}"
                rows.append({
                    "id": match_id,
                    "seed": round_seed + shift,
                    "map": board_path,
                    "seat_count": seat_count,
                    "seats": create_cyclic_shift_seats(lineup, shift, models_registry=reg),
                })

    return rows


def _execute_worker_batch(args: tuple[str, int, list[dict[str, Any]], Path, int, str]) -> tuple[int, float]:
    seat_key, worker_id, chunk_rows, target_out_dir, max_turns, registry = args
    batch_file = target_out_dir / f"_batch_{seat_key}_{worker_id}.json"
    batch_file.write_text(json.dumps({
        "game": "ants",
        "vars": {"max_turns": max_turns},
        "rows": chunk_rows,
    }, ensure_ascii=False))

    env = os.environ.copy()
    env["TINYBRAINS_REGISTRY"] = registry
    t0 = time.perf_counter()
    res = subprocess.run(
        [CLI, str(batch_file), "--out", str(target_out_dir)],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        env=env,
    )
    duration = time.perf_counter() - t0
    batch_file.unlink(missing_ok=True)
    if res.returncode != 0:
        raise RuntimeError(f"工作进程 {seat_key}_{worker_id} 执行对局失败:\n{res.stderr or res.stdout}")
    return len(chunk_rows), duration


def run_parallel_tournament(
    rows: list[dict[str, Any]],
    replays_dir: Path,
    workers: int = 16,
    max_turns: int = 1000,
    registry: str = DEFAULT_REGISTRY,
) -> None:
    replays_dir.mkdir(parents=True, exist_ok=True)
    for old_json in replays_dir.glob("*.json"):
        old_json.unlink(missing_ok=True)

    by_seat_count: dict[int, list[dict[str, Any]]] = {}
    for r in rows:
        by_seat_count.setdefault(r["seat_count"], []).append(r)

    tasks = []
    task_id = 0
    for sc, sc_rows in by_seat_count.items():
        chunk_size = max(1, math.ceil(len(sc_rows) / max(1, workers // max(1, len(by_seat_count)))))
        for i in range(0, len(sc_rows), chunk_size):
            chunk = sc_rows[i : i + chunk_size]
            tasks.append((f"{sc}p", task_id, chunk, replays_dir, max_turns, registry))
            task_id += 1

    actual_workers = min(workers, len(tasks), max(os.cpu_count() or 1, 1))
    print(f"启动 {actual_workers} 个并发进程执行 {len(tasks)} 个分组批次任务 (总计 {len(rows)} 场对局)...")
    t_start = time.perf_counter()
    with ProcessPoolExecutor(max_workers=actual_workers) as pool:
        results = list(pool.map(_execute_worker_batch, tasks))

    total_done = sum(r[0] for r in results)
    total_time = time.perf_counter() - t_start
    print(f"全部 {total_done} 场多席位对局执行完毕，耗时 {total_time:.2f} 秒 (平均 {total_done / max(total_time, 1e-6):.1f} 场/秒)")


def compute_mean_and_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    mean_val = sum(values) / len(values)
    if len(values) < 2:
        return mean_val, 0.0
    var = sum((x - mean_val) ** 2 for x in values) / (len(values) - 1)
    return mean_val, math.sqrt(var)


def _new_bucket() -> dict[str, Any]:
    return {
        "scores": [],
        "ranks": [],
        "rank_pcts": [],
        "first_places": 0,
        "zero_scores": 0,
        "ants_spawned": [],
        "hills_razed": [],
        "hills_lost": [],
        "combat_deaths": 0.0,
        "combat_kills": 0.0,
        "collisions": [],
        "total_orders": 0,
        "blocked_orders": 0,
        "all_orders": 0,
        "hold_orders": 0,
        "on_own_hill_steps": 0,
        "on_own_hill_hold_orders": 0,
        "idle3_steps": 0,
        "idle3_eligible": 0,
        "osc2_steps": 0,
        "osc2_eligible": 0,
        "infer_ms_max": [],
        "strikes": 0,
        "forfeits": 0,
    }


def _record_seat(bucket: dict[str, Any], s: dict[str, Any], n_seats: int, ranks: list[int]) -> None:
    score = float(s.get("score", 0))
    rank = float(s.get("rank", 1))
    if "rank_pct" in s:
        rank_pct = float(s["rank_pct"])
    else:
        better = sum(1 for r in ranks if r > rank)
        tied = max(0, sum(1 for r in ranks if r == rank) - 1)
        rank_pct = (better + 0.5 * tied) / max(n_seats - 1, 1)

    bucket["scores"].append(score)
    bucket["ranks"].append(rank)
    bucket["rank_pcts"].append(rank_pct)
    if rank == 1:
        bucket["first_places"] += 1
    if score == 0:
        bucket["zero_scores"] += 1
    bucket["ants_spawned"].append(float(s.get("ants_spawned", 0)))
    bucket["hills_razed"].append(float(s.get("hills_razed", 0)))
    bucket["hills_lost"].append(float(s.get("hills_lost", 0)))
    bucket["combat_deaths"] += float(s.get("combat_deaths", 0))
    bucket["combat_kills"] += float(s.get("combat_kills", 0))
    bucket["collisions"].append(float(s.get("collisions", 0)))

    # 修复原脚本读取 s.get("orders") 的键名缺陷：extract_frames.mjs 输出字段为 total_orders
    bucket["total_orders"] += int(s.get("total_orders", 0))
    bucket["blocked_orders"] += int(s.get("blocked_orders", 0))
    all_ord = int(s.get("all_orders", int(s.get("total_orders", 0)) + int(s.get("hold_orders", 0))))
    bucket["all_orders"] += all_ord
    bucket["hold_orders"] += int(s.get("hold_orders", 0))
    bucket["on_own_hill_steps"] += int(s.get("on_own_hill_steps", 0))
    bucket["on_own_hill_hold_orders"] += int(s.get("on_own_hill_hold_orders", 0))
    bucket["idle3_steps"] += int(s.get("idle3_steps", 0))
    bucket["idle3_eligible"] += int(s.get("idle3_eligible", 0))
    bucket["osc2_steps"] += int(s.get("osc2_steps", 0))
    bucket["osc2_eligible"] += int(s.get("osc2_eligible", 0))
    bucket["infer_ms_max"].append(float(s.get("infer_us_max", 0)) / 1000.0)
    bucket["strikes"] += int(s.get("strikes", 0))
    if s.get("forfeit"):
        bucket["forfeits"] += 1


def _summarize_bucket(b: dict[str, Any]) -> dict[str, Any]:
    count = len(b["scores"])
    mean_score, std_score = compute_mean_and_std(b["scores"])
    mean_rank, std_rank = compute_mean_and_std(b["ranks"])
    mean_rank_pct, _ = compute_mean_and_std(b["rank_pcts"])
    mean_ants, _ = compute_mean_and_std(b["ants_spawned"])
    mean_razed, _ = compute_mean_and_std(b["hills_razed"])
    mean_infer, _ = compute_mean_and_std(b["infer_ms_max"])
    max_infer = max(b["infer_ms_max"]) if b["infer_ms_max"] else 0.0

    blocked_rate = b["blocked_orders"] / b["total_orders"] if b["total_orders"] > 0 else 0.0
    hold_rate = b["hold_orders"] / b["all_orders"] if b["all_orders"] > 0 else 0.0
    own_hill_hold_rate = (
        b["on_own_hill_hold_orders"] / b["on_own_hill_steps"] if b["on_own_hill_steps"] > 0 else 0.0
    )
    idle3_rate = b["idle3_steps"] / b["idle3_eligible"] if b["idle3_eligible"] > 0 else 0.0
    osc2_rate = b["osc2_steps"] / b["osc2_eligible"] if b["osc2_eligible"] > 0 else 0.0
    kd_ratio = b["combat_kills"] / max(b["combat_deaths"], 1.0) if b["combat_deaths"] > 0 else 0.0

    return {
        "matches_participated": count,
        "rank_pct": round(mean_rank_pct * 100.0, 2),
        "mean_score": round(mean_score, 2),
        "std_score": round(std_score, 2),
        "mean_rank": round(mean_rank, 2),
        "std_rank": round(std_rank, 2),
        "first_place_rate": round(b["first_places"] / max(count, 1), 4),
        "zero_score_rate": round(b["zero_scores"] / max(count, 1), 4),
        "mean_ants_spawned": round(mean_ants, 1),
        "total_ants_spawned": int(sum(b["ants_spawned"])),
        "mean_hills_razed": round(mean_razed, 2),
        "total_hills_razed": int(sum(b["hills_razed"])),
        "total_hills_lost": int(sum(b["hills_lost"])),
        "combat_deaths_mean": round(b["combat_deaths"] / max(count, 1), 1),
        "kd_ratio": round(kd_ratio, 3),
        "total_orders": b["total_orders"],
        "blocked_orders": b["blocked_orders"],
        "blocked_order_rate": round(blocked_rate, 6),
        "hold_rate": round(hold_rate, 4),
        "on_own_hill_hold_rate": round(own_hill_hold_rate, 4),
        "idle3_rate": round(idle3_rate, 4),
        "osc2_rate": round(osc2_rate, 4),
        "mean_infer_ms": round(mean_infer, 2),
        "max_infer_ms": round(max_infer, 2),
        "strikes": b["strikes"],
        "forfeits": b["forfeits"],
    }


def analyze_match_records(
    match_records: list[dict[str, Any]],
    models_involved: list[str] | None = None,
) -> dict[str, Any]:
    if not match_records:
        raise RuntimeError("未提供任何回放记录")

    if models_involved is None:
        discovered = sorted({
            str(s.get("name"))
            for m in match_records
            for s in m.get("seats", [])
            if s.get("name")
        })
        ordered = [m for m in MODEL_SPECS.keys() if m in discovered]
        models_involved = ordered + [m for m in discovered if m not in ordered]

    overall_buckets: dict[str, dict[str, Any]] = {m: _new_bucket() for m in models_involved}
    per_board_buckets: dict[str, dict[str, dict[str, Any]]] = defaultdict(lambda: defaultdict(_new_bucket))
    total_seats_evaluated = 0

    for match in match_records:
        map_id = str(match.get("map_id", ""))
        board_key = None
        for b_k, b_info in BOARDS.items():
            if b_info["name"] in map_id:
                board_key = b_k
                break
        if not board_key:
            board_key = map_id or f"{len(match.get('seats', []))}p"

        seats = match.get("seats", [])
        ranks = [int(s.get("rank", 1)) for s in seats]
        n_seats = len(seats)
        for s in seats:
            m_name = str(s.get("name", ""))
            if m_name not in overall_buckets:
                continue
            total_seats_evaluated += 1
            _record_seat(overall_buckets[m_name], s, n_seats=n_seats, ranks=ranks)
            _record_seat(per_board_buckets[board_key][m_name], s, n_seats=n_seats, ranks=ranks)

    summary_by_model: dict[str, Any] = {}
    for m in models_involved:
        if overall_buckets[m]["scores"]:
            summary_by_model[m] = _summarize_bucket(overall_buckets[m])

    board_summary: dict[str, Any] = {}
    for b_k in sorted(per_board_buckets.keys()):
        board_summary[b_k] = {}
        for m in models_involved:
            b_data = per_board_buckets[b_k].get(m)
            if b_data and b_data["scores"]:
                bs = _summarize_bucket(b_data)
                total_m_seats = summary_by_model[m]["matches_participated"]
                board_summary[b_k][m] = {
                    "count": bs["matches_participated"],
                    "seat_share_pct": round(100.0 * bs["matches_participated"] / max(total_m_seats, 1), 2),
                    "rank_pct": bs["rank_pct"],
                    "mean_score": bs["mean_score"],
                    "mean_rank": bs["mean_rank"],
                    "zero_score_rate": bs["zero_score_rate"],
                    "mean_ants": bs["mean_ants_spawned"],
                    "total_razed": bs["total_hills_razed"],
                    "total_lost": bs["total_hills_lost"],
                    "kd_ratio": bs["kd_ratio"],
                    "blocked_order_rate": bs["blocked_order_rate"],
                    "hold_rate": bs["hold_rate"],
                    "on_own_hill_hold_rate": bs["on_own_hill_hold_rate"],
                    "idle3_rate": bs["idle3_rate"],
                }

    # 计算等权棋盘均值（每张棋盘权重相等，防止单一 8p 棋盘席位占比过高掩盖其余 4 张棋盘的失利）
    unweighted_map_summary: dict[str, Any] = {}
    for m in models_involved:
        played_boards = [b_k for b_k, bm in board_summary.items() if m in bm]
        if not played_boards:
            continue
        k = len(played_boards)
        unweighted_map_summary[m] = {
            "boards_played": k,
            "unweighted_rank_pct": round(sum(board_summary[b][m]["rank_pct"] for b in played_boards) / k, 2),
            "unweighted_mean_rank": round(sum(board_summary[b][m]["mean_rank"] for b in played_boards) / k, 2),
            "unweighted_mean_score": round(sum(board_summary[b][m]["mean_score"] for b in played_boards) / k, 2),
            "unweighted_zero_score_rate": round(sum(board_summary[b][m]["zero_score_rate"] for b in played_boards) / k, 4),
            "unweighted_blocked_order_rate": round(sum(board_summary[b][m]["blocked_order_rate"] for b in played_boards) / k, 6),
            "unweighted_hold_rate": round(sum(board_summary[b][m]["hold_rate"] for b in played_boards) / k, 4),
            "unweighted_on_own_hill_hold_rate": round(sum(board_summary[b][m]["on_own_hill_hold_rate"] for b in played_boards) / k, 4),
        }

    # 聚合一致性审计：输出每对模型在共同参战棋盘上的逐图胜负计数，防止辛普森悖论误导结论
    board_h2h: dict[str, dict[str, int]] = {}
    for i, m_a in enumerate(models_involved):
        for m_b in models_involved[i + 1 :]:
            shared = [b_k for b_k, bm in board_summary.items() if m_a in bm and m_b in bm]
            if not shared:
                continue
            a_better = sum(1 for b in shared if board_summary[b][m_a]["rank_pct"] > board_summary[b][m_b]["rank_pct"])
            b_better = sum(1 for b in shared if board_summary[b][m_b]["rank_pct"] > board_summary[b][m_a]["rank_pct"])
            board_h2h[f"{m_a}_vs_{m_b}"] = {
                "shared_boards": len(shared),
                f"{m_a}_boards_won": a_better,
                f"{m_b}_boards_won": b_better,
                "ties": len(shared) - a_better - b_better,
            }

    return {
        "total_matches": len(match_records),
        "total_seats_evaluated": total_seats_evaluated,
        "models": summary_by_model,
        "unweighted_map_summary": unweighted_map_summary,
        "per_board": board_summary,
        "aggregation_audit": {
            "seat_share_by_board": {
                m: {b: bm[m]["seat_share_pct"] for b, bm in board_summary.items() if m in bm}
                for m in summary_by_model.keys()
            },
            "per_board_rank_comparison": board_h2h,
        },
    }


def analyze_multiseat_replays(
    replays_dir: Path,
    dist_dir: Path | None = None,
) -> dict[str, Any]:
    if not EXTRACTOR.exists():
        raise FileNotFoundError(f"未找到度量提取脚本: {EXTRACTOR}")

    env = os.environ.copy()
    if dist_dir is not None:
        env["ANTS_DIST_DIR"] = str(dist_dir.resolve())

    node_cmd = ["node", str(EXTRACTOR.resolve()), str(replays_dir.resolve())]
    node_res = subprocess.run(
        node_cmd,
        cwd=str(ROOT / "benchmarks"),
        capture_output=True,
        text=True,
        env=env,
    )
    if node_res.returncode != 0:
        raise RuntimeError(f"提取回放度量指标失败:\n{node_res.stderr}")

    match_records: list[dict[str, Any]] = json.loads(node_res.stdout)
    return analyze_match_records(match_records)


def main() -> None:
    parser = argparse.ArgumentParser(description="多席位全谱系邻接均衡循环锦标赛执行与指标分析器。")
    parser.add_argument("--rounds", type=int, default=4, help="锦标赛轮数（默认 4）")
    parser.add_argument("--workers", type=int, default=16, help="并行工作进程数（默认 16）")
    parser.add_argument("--max-turns", type=int, default=1000, help="每场最大回合数（默认 1000，与线上一致）")
    parser.add_argument("--base-seed", type=int, default=85000, help="随机数基准种子（默认 85000）")
    parser.add_argument("--models-root", type=Path, default=None, help="模型根目录（默认 <repo>/models）")
    parser.add_argument("--registry", type=str, default=DEFAULT_REGISTRY, help="TinyBrains 注册表路径")
    parser.add_argument("--dist-dir", type=Path, default=None, help="可选 dist 目录（用于 worktree 复用）")
    parser.add_argument(
        "--replays",
        type=Path,
        default=ROOT / "runs" / "m3_multiseat_tournament",
        help="回放保存目录",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "runs" / "m3_multiseat_evaluation_report.json",
        help="评测报告输出路径",
    )
    args = parser.parse_args()

    rows = build_multiseat_tournament_rows(
        rounds=args.rounds,
        base_seed=args.base_seed,
        models_root=args.models_root,
    )
    print(f"已构建 {len(rows)} 场多席位邻接均衡锦标赛配置 (覆盖 2p, 3p, 4p, 6p, 8p 全谱系)。")

    run_parallel_tournament(
        rows=rows,
        replays_dir=args.replays,
        workers=args.workers,
        max_turns=args.max_turns,
        registry=args.registry,
    )

    print("正在解算回放帧度量指标...")
    report = analyze_multiseat_replays(args.replays, dist_dir=args.dist_dir)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\n评测分析报告已写入: {args.out}")

    print("\n================== 多席位锦标赛综合对抗战报 (席位加权) ==================")
    print(
        f"{'模型标识':<10} {'席位':<6} {'名次分%':<9} {'均分':<7} {'均名次':<7} "
        f"{'阻塞%':<8} {'HOLD%':<8} {'占穴HOLD%':<10} {'停滞3t%':<9} {'K/D':<6} {'峰值ms':<8}"
    )
    print("-" * 102)
    for m, info in report["models"].items():
        print(
            f"{m:<10} {info['matches_participated']:<6} {info['rank_pct']:<9.2f} "
            f"{info['mean_score']:<7.2f} {info['mean_rank']:<7.2f} "
            f"{info['blocked_order_rate'] * 100.0:<8.3f} {info['hold_rate'] * 100.0:<8.2f} "
            f"{info['on_own_hill_hold_rate'] * 100.0:<10.2f} {info['idle3_rate'] * 100.0:<9.2f} "
            f"{info['kd_ratio']:<6.3f} {info['max_infer_ms']:<8.1f}"
        )
    print("\n================== 等权棋盘均值 (消除单图席位加权偏差) ==================")
    for m, uinfo in report["unweighted_map_summary"].items():
        print(
            f"  {m:<10} boards={uinfo['boards_played']} "
            f"unweighted_rank_pct={uinfo['unweighted_rank_pct']:.2f}% "
            f"unweighted_score={uinfo['unweighted_mean_score']:.2f} "
            f"unweighted_hold={uinfo['unweighted_hold_rate'] * 100.0:.2f}% "
            f"unweighted_own_hill_hold={uinfo['unweighted_on_own_hill_hold_rate'] * 100.0:.2f}%"
        )
    print("=" * 102)


if __name__ == "__main__":
    main()
