from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path("/home/host/project/ants")
CLI = os.environ.get("TINYBRAINS") or shutil.which("tinybrains") or "/home/linuxbrew/.linuxbrew/bin/tinybrains"
REGISTRY = str((ROOT / "baselines" / "games.toml").resolve())
EXTRACTOR = ROOT / "benchmarks" / "extract_frames.mjs"

MODELS = {
    "exp085": {
        "name": "exp085",
        "description": "多席位 MCMF 协同优化模型 (Micro 28 通道)",
        "weights": ROOT / "models" / "exp085" / "model.onnx",
        "manifest": ROOT / "models" / "exp085" / "manifest.json",
    },
    "exp083": {
        "name": "exp083",
        "description": "前代 MCMF 蒸馏模型 (Micro 28 通道)",
        "weights": ROOT / "models" / "exp083" / "model.onnx",
        "manifest": ROOT / "models" / "exp083" / "manifest.json",
    },
    "exp033": {
        "name": "exp033",
        "description": "DAgger 8 轮聚合天梯标杆 (Micro 28 通道)",
        "weights": ROOT / "models" / "exp033" / "model.onnx",
        "manifest": ROOT / "models" / "exp033" / "manifest.json",
    },
    "exp021": {
        "name": "exp021",
        "description": "激进进攻型基准 (Micro 28 通道)",
        "weights": ROOT / "models" / "exp021" / "model.onnx",
        "manifest": ROOT / "models" / "exp021" / "manifest.json",
    },
    "exp011": {
        "name": "exp011",
        "description": "防守采食型基准 (Nano 14 通道)",
        "weights": ROOT / "models" / "exp011" / "model.onnx",
        "manifest": ROOT / "models" / "exp011" / "manifest.json",
    },
}

BOARDS = {
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


def create_seat_descriptor(seat_index: int, model_key: str) -> dict[str, Any]:
    model_info = MODELS[model_key]
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


def create_cyclic_shift_seats(base_models: list[str], shift: int) -> list[dict[str, Any]]:
    seat_count = len(base_models)
    seats = []
    for base_seat, model_key in enumerate(base_models):
        seat_idx = (base_seat + shift) % seat_count
        seats.append(create_seat_descriptor(seat_idx, model_key))
    seats.sort(key=lambda s: s["seat"])
    return seats


def build_multiseat_tournament_rows(
    rounds: int = 4,
    base_seed: int = 85000,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for r in range(rounds):
        round_seed = base_seed + r * 1000

        for seat_key, board_info in BOARDS.items():
            board_path = str(board_info["path"].resolve())
            seat_count = board_info["seat_count"]
            lineup = board_info["base_lineup"]

            for shift in range(seat_count):
                match_id = f"m3-multi-{seat_key}-r{r}-s{round_seed + shift}-shift{shift}"
                rows.append({
                    "id": match_id,
                    "seed": round_seed + shift,
                    "map": board_path,
                    "seat_count": seat_count,
                    "seats": create_cyclic_shift_seats(lineup, shift),
                })

    return rows


def _execute_worker_batch(args: tuple[str, int, list[dict[str, Any]], Path, int]) -> tuple[int, float]:
    seat_key, worker_id, chunk_rows, target_out_dir, max_turns = args
    batch_file = target_out_dir / f"_batch_{seat_key}_{worker_id}.json"
    batch_file.write_text(json.dumps({
        "game": "ants",
        "vars": {"max_turns": max_turns},
        "rows": chunk_rows,
    }, ensure_ascii=False))

    env = os.environ.copy()
    env["TINYBRAINS_REGISTRY"] = REGISTRY
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
    max_turns: int = 300,
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
        chunk_size = max(1, math.ceil(len(sc_rows) / max(1, workers // len(by_seat_count))))
        for i in range(0, len(sc_rows), chunk_size):
            chunk = sc_rows[i : i + chunk_size]
            tasks.append((f"{sc}p", task_id, chunk, replays_dir, max_turns))
            task_id += 1

    actual_workers = min(workers, len(tasks), max(os.cpu_count() or 1, 1))
    print(f"启动 {actual_workers} 个并发进程执行 {len(tasks)} 个分组批次任务 (总计 {len(rows)} 场对局)...")
    t_start = time.perf_counter()
    with ProcessPoolExecutor(max_workers=actual_workers) as pool:
        results = list(pool.map(_execute_worker_batch, tasks))

    total_done = sum(r[0] for r in results)
    total_time = time.perf_counter() - t_start
    print(f"全部 {total_done} 场多席位对局执行完毕，耗时 {total_time:.2f} 秒 (平均 {total_done / total_time:.1f} 场/秒)")


def compute_mean_and_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    mean_val = sum(values) / len(values)
    if len(values) < 2:
        return mean_val, 0.0
    var = sum((x - mean_val) ** 2 for x in values) / (len(values) - 1)
    return mean_val, math.sqrt(var)


def analyze_multiseat_replays(replays_dir: Path) -> dict[str, Any]:
    if not EXTRACTOR.exists():
        raise FileNotFoundError(f"未找到度量提取脚本: {EXTRACTOR}")

    node_cmd = ["node", str(EXTRACTOR.resolve()), str(replays_dir.resolve())]
    node_res = subprocess.run(
        node_cmd,
        cwd=str(ROOT / "benchmarks"),
        capture_output=True,
        text=True,
    )
    if node_res.returncode != 0:
        raise RuntimeError(f"提取回放度量指标失败:\n{node_res.stderr}")

    match_records: list[dict[str, Any]] = json.loads(node_res.stdout)
    if not match_records:
        raise RuntimeError(f"未提取到任何回放记录: {replays_dir}")

    models_involved = list(MODELS.keys())
    stats: dict[str, dict[str, list[float]]] = {
        m: {
            "scores": [],
            "ranks": [],
            "ants_spawned": [],
            "hills_razed": [],
            "combat_deaths": [],
            "blocked_rates": [],
            "infer_ms_max": [],
        }
        for m in models_involved
    }

    per_board_stats: dict[str, dict[str, dict[str, list[float]]]] = {
        b: {
            m: {"scores": [], "ranks": [], "ants_spawned": [], "hills_razed": []}
            for m in models_involved
        }
        for b in BOARDS.keys()
    }

    zero_timeout_strikes = 0
    total_seats_evaluated = 0

    for match in match_records:
        map_id = match.get("map_id", "")
        board_key = None
        for b_k, b_info in BOARDS.items():
            if b_info["name"] in map_id:
                board_key = b_k
                break
        if not board_key:
            board_key = "unknown"

        seats = match.get("seats", [])
        for s in seats:
            m_name = s.get("name")
            if m_name not in stats:
                continue

            total_seats_evaluated += 1
            score = float(s.get("score", 0))
            rank = float(s.get("rank", 1))
            ants = float(s.get("ants_spawned", 0))
            razed = float(s.get("hills_razed", 0))
            deaths = float(s.get("combat_deaths", 0))
            orders = s.get("orders", 0)
            blocked = s.get("blocked_orders", 0)
            blocked_rate = (blocked / max(orders, 1)) if orders > 0 else 0.0
            infer_max = float(s.get("infer_us_max", 0)) / 1000.0

            stats[m_name]["scores"].append(score)
            stats[m_name]["ranks"].append(rank)
            stats[m_name]["ants_spawned"].append(ants)
            stats[m_name]["hills_razed"].append(razed)
            stats[m_name]["combat_deaths"].append(deaths)
            stats[m_name]["blocked_rates"].append(blocked_rate)
            stats[m_name]["infer_ms_max"].append(infer_max)

            if board_key in per_board_stats and m_name in per_board_stats[board_key]:
                per_board_stats[board_key][m_name]["scores"].append(score)
                per_board_stats[board_key][m_name]["ranks"].append(rank)
                per_board_stats[board_key][m_name]["ants_spawned"].append(ants)
                per_board_stats[board_key][m_name]["hills_razed"].append(razed)

    summary_by_model: dict[str, Any] = {}
    for m in models_involved:
        data = stats[m]
        count = len(data["scores"])
        if count == 0:
            continue
        mean_score, std_score = compute_mean_and_std(data["scores"])
        mean_rank, std_rank = compute_mean_and_std(data["ranks"])
        mean_ants, std_ants = compute_mean_and_std(data["ants_spawned"])
        mean_razed, std_razed = compute_mean_and_std(data["hills_razed"])
        mean_blocked, _ = compute_mean_and_std(data["blocked_rates"])
        max_infer = max(data["infer_ms_max"]) if data["infer_ms_max"] else 0.0
        mean_infer, _ = compute_mean_and_std(data["infer_ms_max"])

        summary_by_model[m] = {
            "matches_participated": count,
            "mean_score": round(mean_score, 2),
            "std_score": round(std_score, 2),
            "mean_rank": round(mean_rank, 2),
            "std_rank": round(std_rank, 2),
            "mean_ants_spawned": round(mean_ants, 1),
            "total_ants_spawned": int(sum(data["ants_spawned"])),
            "mean_hills_razed": round(mean_razed, 2),
            "total_hills_razed": int(sum(data["hills_razed"])),
            "blocked_order_rate": round(mean_blocked, 4),
            "mean_infer_ms": round(mean_infer, 2),
            "max_infer_ms": round(max_infer, 2),
        }

    board_summary: dict[str, Any] = {}
    for b_k, b_models in per_board_stats.items():
        board_summary[b_k] = {}
        for m, m_data in b_models.items():
            if m_data["scores"]:
                board_summary[b_k][m] = {
                    "count": len(m_data["scores"]),
                    "mean_score": round(sum(m_data["scores"]) / len(m_data["scores"]), 2),
                    "mean_rank": round(sum(m_data["ranks"]) / len(m_data["ranks"]), 2),
                    "mean_ants": round(sum(m_data["ants_spawned"]) / len(m_data["ants_spawned"]), 1),
                    "total_razed": int(sum(m_data["hills_razed"])),
                }

    return {
        "total_matches": len(match_records),
        "total_seats_evaluated": total_seats_evaluated,
        "models": summary_by_model,
        "per_board": board_summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="多席位全谱系循环锦标赛执行与指标分析器。")
    parser.add_argument("--rounds", type=int, default=4, help="锦标赛轮数（默认 4）")
    parser.add_argument("--workers", type=int, default=16, help="并行工作进程数（默认 16）")
    parser.add_argument("--max-turns", type=int, default=300, help="每场最大回合数（默认 300）")
    parser.add_argument("--base-seed", type=int, default=85000, help="随机数基准种子（默认 85000）")
    parser.add_argument(
        "--replays",
        type=Path,
        default=ROOT / "replays" / "m3_multiseat_tournament",
        help="回放保存目录",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "runs" / "m3_multiseat_evaluation_report.json",
        help="评测报告输出路径",
    )
    args = parser.parse_args()

    rows = build_multiseat_tournament_rows(rounds=args.rounds, base_seed=args.base_seed)
    print(f"已构建 {len(rows)} 场多席位全对称循环锦标赛配置 (覆盖 2p, 3p, 4p, 6p, 8p 全谱系)。")

    run_parallel_tournament(
        rows=rows,
        replays_dir=args.replays,
        workers=args.workers,
        max_turns=args.max_turns,
    )

    print("正在解算回放帧度量指标...")
    report = analyze_multiseat_replays(args.replays)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\n评测分析报告已写入: {args.out}")

    print("\n================== 多席位锦标赛综合对抗战报 ==================")
    print(f"{'模型标识':<10} {'参战席位':<10} {'场均得分':<12} {'场均排名':<12} {'场均繁衍':<12} {'摧毁敌穴':<10} {'单步峰值ms':<12}")
    print("-" * 82)
    for m, info in report["models"].items():
        print(
            f"{m:<10} {info['matches_participated']:<10} "
            f"{info['mean_score']:<12} {info['mean_rank']:<12} "
            f"{info['mean_ants_spawned']:<12} {info['total_hills_razed']:<10} "
            f"{info['max_infer_ms']:<12}"
        )
    print("=" * 82)


if __name__ == "__main__":
    main()
