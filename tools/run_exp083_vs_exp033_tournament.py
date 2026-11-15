import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path("/home/host/project/ants")
CLI = os.environ.get("TINYBRAINS") or shutil.which("tinybrains") or "/home/linuxbrew/.linuxbrew/bin/tinybrains"
REGISTRY = str((ROOT / "baselines" / "games.toml").resolve())
EXTRACTOR = ROOT / "benchmarks" / "extract_frames.mjs"

@dataclass
class Entrant:
    name: str
    weights: Path
    manifest: Path

    def seat(self, index: int) -> dict[str, Any]:
        return {
            "seat": index,
            "label": self.name,
            "weights": str(self.weights.resolve()),
            "manifest": str(self.manifest.resolve()),
        }

def _execute_worker_batch(args: tuple[int, list[dict[str, Any]], Path, int]) -> tuple[int, float]:
    worker_id, chunk_rows, target_out_dir, max_turns = args
    batch_file = target_out_dir / f"_batch_{worker_id}.json"
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
    if res.returncode != 0:
        raise RuntimeError(f"工作进程 {worker_id} 执行对局失败:\n{res.stderr or res.stdout}")
    batch_file.unlink(missing_ok=True)
    return len(chunk_rows), duration

def calculate_std(values: list[float], mean_val: float) -> float:
    if len(values) < 2:
        return 0.0
    variance = sum((x - mean_val) ** 2 for x in values) / (len(values) - 1)
    return math.sqrt(variance)

def run_tournament(
    model_a_dir: Path,
    model_b_dir: Path,
    maps_dir: Path,
    out_dir: Path,
    rounds: int = 50,
    max_turns: int = 300,
    workers: int = 16,
    base_seed: int = 95000,
) -> dict[str, Any]:
    entrant_a = Entrant(
        name=model_a_dir.name,
        weights=model_a_dir / "model.onnx",
        manifest=model_a_dir / "manifest.json",
    )
    entrant_b = Entrant(
        name=model_b_dir.name,
        weights=model_b_dir / "model.onnx",
        manifest=model_b_dir / "manifest.json",
    )

    if not entrant_a.weights.exists():
        raise FileNotFoundError(f"模型 A 权重不存在: {entrant_a.weights}")
    if not entrant_b.weights.exists():
        raise FileNotFoundError(f"模型 B 权重不存在: {entrant_b.weights}")

    map_files = sorted(list(maps_dir.glob("*.json")))
    if len(map_files) != 9:
        raise ValueError(f"预期 9 张基准地图，实际找到: {len(map_files)}")

    rows: list[dict[str, Any]] = []
    labels: dict[str, tuple[str, str]] = {}

    for r_idx in range(rounds):
        round_seed = base_seed + r_idx * 1000
        for b_idx, map_file in enumerate(map_files):
            map_name = map_file.stem
            for flip in (0, 1):
                first = entrant_a if flip == 0 else entrant_b
                second = entrant_b if flip == 0 else entrant_a
                match_id = f"h2h-r{r_idx:03d}-s{flip}-{first.name}-vs-{second.name}-{map_name}"
                match_seed = round_seed + b_idx * 10 + flip
                rows.append({
                    "id": match_id,
                    "seed": match_seed,
                    "map": str(map_file.resolve()),
                    "seat_count": 2,
                    "seats": [first.seat(0), second.seat(1)],
                })
                labels[match_id] = (first.name, second.name)

    total_matches = len(rows)
    print(f"生成的对局总数: {total_matches} (轮数: {rounds}, 地图数: {len(map_files)}, 双向席位: 2)")

    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    num_workers = min(workers, total_matches)
    chunk_size = (total_matches + num_workers - 1) // num_workers
    chunks = [rows[i:i + chunk_size] for i in range(0, total_matches, chunk_size)]
    tasks = [(i, chunk, out_dir, max_turns) for i, chunk in enumerate(chunks)]

    print(f"开始执行对战，工作进程数: {num_workers}...")
    t_start = time.perf_counter()
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = list(executor.map(_execute_worker_batch, tasks))
    total_wall_time = time.perf_counter() - t_start
    print(f"对战执行完成，总耗时: {total_wall_time:.2f} 秒，吞吐量: {total_matches / total_wall_time:.2f} 对局/秒")

    print("开始提取官方帧度量指标...")
    node_cmd = ["node", str(EXTRACTOR), str(out_dir.resolve())]
    node_res = subprocess.run(node_cmd, cwd=str(ROOT / "benchmarks"), capture_output=True, text=True)
    if node_res.returncode != 0:
        raise RuntimeError(f"提取指标失败:\n{node_res.stderr or node_res.stdout}")
    frame_metrics = json.loads(node_res.stdout)
    if len(frame_metrics) != total_matches:
        raise ValueError(f"提取的对局记录数量 ({len(frame_metrics)}) 与期望数量 ({total_matches}) 不一致")

    name_a = entrant_a.name
    name_b = entrant_b.name

    stats = {
        name_a: {
            "wins": 0, "draws": 0, "losses": 0, "points": 0.0,
            "total_score": 0, "scores": [],
            "score_frequencies": Counter(),
            "seat0_matches": 0, "seat0_wins": 0, "seat0_draws": 0, "seat0_losses": 0, "seat0_points": 0.0, "seat0_score": 0,
            "seat1_matches": 0, "seat1_wins": 0, "seat1_draws": 0, "seat1_losses": 0, "seat1_points": 0.0, "seat1_score": 0,
            "hills_razed": 0, "raze_turns": [],
            "early_raze_count": 0,
            "total_orders": 0, "blocked_orders": 0, "collisions": 0, "combat_deaths": 0, "ants_spawned": 0,
            "infer_us_total": 0, "infer_us_max": 0,
        },
        name_b: {
            "wins": 0, "draws": 0, "losses": 0, "points": 0.0,
            "total_score": 0, "scores": [],
            "score_frequencies": Counter(),
            "seat0_matches": 0, "seat0_wins": 0, "seat0_draws": 0, "seat0_losses": 0, "seat0_points": 0.0, "seat0_score": 0,
            "seat1_matches": 0, "seat1_wins": 0, "seat1_draws": 0, "seat1_losses": 0, "seat1_points": 0.0, "seat1_score": 0,
            "hills_razed": 0, "raze_turns": [],
            "early_raze_count": 0,
            "total_orders": 0, "blocked_orders": 0, "collisions": 0, "combat_deaths": 0, "ants_spawned": 0,
            "infer_us_total": 0, "infer_us_max": 0,
        },
    }

    map_stats: dict[str, dict[str, Any]] = {
        m.stem: {
            "matches": 0,
            name_a: {
                "wins": 0, "draws": 0, "losses": 0, "points": 0.0,
                "score": 0, "scores": [], "hills_razed": 0, "raze_turns": [],
                "blocked": 0, "orders": 0, "collisions": 0, "combat_deaths": 0, "spawned": 0
            },
            name_b: {
                "wins": 0, "draws": 0, "losses": 0, "points": 0.0,
                "score": 0, "scores": [], "hills_razed": 0, "raze_turns": [],
                "blocked": 0, "orders": 0, "collisions": 0, "combat_deaths": 0, "spawned": 0
            },
            "turns": [],
        }
        for m in map_files
    }

    termination_reasons: Counter[str] = Counter()
    total_turns_list: list[int] = []
    score_margins: list[int] = []
    margin_frequencies: Counter[int] = Counter()

    for match in frame_metrics:
        map_id = match["map_id"]
        reason = match.get("reason", "unknown")
        termination_reasons[reason] += 1
        turns = match.get("turns", 0)
        total_turns_list.append(turns)

        seats = match["seats"]
        seat_a = seats[0] if seats[0]["name"] == name_a else seats[1]
        seat_b = seats[0] if seats[0]["name"] == name_b else seats[1]

        score_a = seat_a["score"]
        score_b = seat_b["score"]
        rank_a = seat_a["rank"]
        rank_b = seat_b["rank"]

        diff = score_a - score_b
        score_margins.append(diff)
        margin_frequencies[diff] += 1

        stats[name_a]["total_score"] += score_a
        stats[name_a]["scores"].append(score_a)
        stats[name_a]["score_frequencies"][score_a] += 1

        stats[name_b]["total_score"] += score_b
        stats[name_b]["scores"].append(score_b)
        stats[name_b]["score_frequencies"][score_b] += 1

        map_entry = map_stats[map_id]
        map_entry["matches"] += 1
        map_entry["turns"].append(turns)
        map_entry[name_a]["score"] += score_a
        map_entry[name_a]["scores"].append(score_a)
        map_entry[name_b]["score"] += score_b
        map_entry[name_b]["scores"].append(score_b)

        is_a_seat0 = (seat_a["seat"] == 0)
        if is_a_seat0:
            stats[name_a]["seat0_matches"] += 1
            stats[name_a]["seat0_score"] += score_a
            stats[name_b]["seat1_matches"] += 1
            stats[name_b]["seat1_score"] += score_b
        else:
            stats[name_a]["seat1_matches"] += 1
            stats[name_a]["seat1_score"] += score_a
            stats[name_b]["seat0_matches"] += 1
            stats[name_b]["seat0_score"] += score_b

        if rank_a < rank_b:
            stats[name_a]["wins"] += 1
            stats[name_a]["points"] += 1.0
            stats[name_b]["losses"] += 1
            map_entry[name_a]["wins"] += 1
            map_entry[name_a]["points"] += 1.0
            map_entry[name_b]["losses"] += 1
            if is_a_seat0:
                stats[name_a]["seat0_wins"] += 1
                stats[name_a]["seat0_points"] += 1.0
                stats[name_b]["seat1_losses"] += 1
            else:
                stats[name_a]["seat1_wins"] += 1
                stats[name_a]["seat1_points"] += 1.0
                stats[name_b]["seat0_losses"] += 1
        elif rank_a > rank_b:
            stats[name_a]["losses"] += 1
            stats[name_b]["wins"] += 1
            stats[name_b]["points"] += 1.0
            map_entry[name_a]["losses"] += 1
            map_entry[name_b]["wins"] += 1
            map_entry[name_b]["points"] += 1.0
            if is_a_seat0:
                stats[name_a]["seat0_losses"] += 1
                stats[name_b]["seat1_wins"] += 1
                stats[name_b]["seat1_points"] += 1.0
            else:
                stats[name_a]["seat1_losses"] += 1
                stats[name_b]["seat0_wins"] += 1
                stats[name_b]["seat0_points"] += 1.0
        else:
            stats[name_a]["draws"] += 1
            stats[name_a]["points"] += 0.5
            stats[name_b]["draws"] += 1
            stats[name_b]["points"] += 0.5
            map_entry[name_a]["draws"] += 1
            map_entry[name_a]["points"] += 0.5
            map_entry[name_b]["draws"] += 1
            map_entry[name_b]["points"] += 0.5
            if is_a_seat0:
                stats[name_a]["seat0_draws"] += 1
                stats[name_a]["seat0_points"] += 0.5
                stats[name_b]["seat1_draws"] += 1
                stats[name_b]["seat1_points"] += 0.5
            else:
                stats[name_a]["seat1_draws"] += 1
                stats[name_a]["seat1_points"] += 0.5
                stats[name_b]["seat0_draws"] += 1
                stats[name_b]["seat0_points"] += 0.5

        for mod_name, s_data in [(name_a, seat_a), (name_b, seat_b)]:
            orders = s_data.get("total_orders", 0)
            blocked = s_data.get("blocked_orders", 0)
            colls = s_data.get("collisions", 0)
            deaths = s_data.get("combat_deaths", 0)
            spawned = s_data.get("ants_spawned", 0)
            infer_us = s_data.get("infer_us_total", 0)

            stats[mod_name]["total_orders"] += orders
            stats[mod_name]["blocked_orders"] += blocked
            stats[mod_name]["collisions"] += colls
            stats[mod_name]["combat_deaths"] += deaths
            stats[mod_name]["ants_spawned"] += spawned
            stats[mod_name]["infer_us_total"] += infer_us
            if s_data.get("infer_us_max", 0) > stats[mod_name]["infer_us_max"]:
                stats[mod_name]["infer_us_max"] = s_data["infer_us_max"]

            if s_data.get("hills_razed", 0) > 0:
                stats[mod_name]["hills_razed"] += s_data["hills_razed"]
                map_entry[mod_name]["hills_razed"] += s_data["hills_razed"]
                first_turn = s_data.get("first_hill_destroy_turn")
                if first_turn is not None:
                    stats[mod_name]["raze_turns"].append(first_turn)
                    map_entry[mod_name]["raze_turns"].append(first_turn)
                    if first_turn <= 50:
                        stats[mod_name]["early_raze_count"] += 1

            map_entry[mod_name]["orders"] += orders
            map_entry[mod_name]["blocked"] += blocked
            map_entry[mod_name]["collisions"] += colls
            map_entry[mod_name]["combat_deaths"] += deaths
            map_entry[mod_name]["spawned"] += spawned

    mean_score_a = stats[name_a]["total_score"] / total_matches
    mean_score_b = stats[name_b]["total_score"] / total_matches
    std_score_a = calculate_std(stats[name_a]["scores"], mean_score_a)
    std_score_b = calculate_std(stats[name_b]["scores"], mean_score_b)

    score_rate_a = stats[name_a]["points"] / total_matches
    score_rate_b = stats[name_b]["points"] / total_matches

    win_rate_a = stats[name_a]["wins"] / total_matches
    win_rate_b = stats[name_b]["wins"] / total_matches

    mean_margin = sum(score_margins) / len(score_margins)
    std_margin = calculate_std([float(m) for m in score_margins], mean_margin)

    ci_margin_95 = 1.96 * (std_margin / math.sqrt(total_matches))
    ci_score_rate_a_95 = 1.96 * math.sqrt((score_rate_a * (1.0 - score_rate_a)) / total_matches)

    mean_turns = sum(total_turns_list) / len(total_turns_list)

    report: dict[str, Any] = {
        "metadata": {
            "title": "EXP-083 对决 EXP-033 官方正式比赛环境基准对抗赛",
            "model_a": name_a,
            "model_b": name_b,
            "rounds": rounds,
            "maps_count": len(map_files),
            "total_matches": total_matches,
            "max_turns": max_turns,
            "base_seed": base_seed,
            "wall_clock_seconds": round(total_wall_time, 2),
            "throughput_matches_per_sec": round(total_matches / total_wall_time, 2),
        },
        "summary": {
            name_a: {
                "wins": stats[name_a]["wins"],
                "draws": stats[name_a]["draws"],
                "losses": stats[name_a]["losses"],
                "points": stats[name_a]["points"],
                "win_rate": round(win_rate_a, 4),
                "score_rate": round(score_rate_a, 4),
                "score_rate_95ci": [round(score_rate_a - ci_score_rate_a_95, 4), round(score_rate_a + ci_score_rate_a_95, 4)],
                "total_score": stats[name_a]["total_score"],
                "mean_score": round(mean_score_a, 4),
                "std_score": round(std_score_a, 4),
                "median_score": sorted(stats[name_a]["scores"])[len(stats[name_a]["scores"]) // 2],
                "score_distribution": dict(sorted(stats[name_a]["score_frequencies"].items())),
                "hills_razed": stats[name_a]["hills_razed"],
                "raze_rate": round(stats[name_a]["hills_razed"] / total_matches, 4),
                "mean_raze_turn": round(sum(stats[name_a]["raze_turns"]) / len(stats[name_a]["raze_turns"]), 2) if stats[name_a]["raze_turns"] else None,
                "early_raze_rate": round(stats[name_a]["early_raze_count"] / total_matches, 4),
                "ants_spawned": stats[name_a]["ants_spawned"],
                "combat_deaths": stats[name_a]["combat_deaths"],
                "collisions": stats[name_a]["collisions"],
                "kd_ratio": round(stats[name_b]["combat_deaths"] / max(stats[name_a]["combat_deaths"], 1), 4),
                "total_orders": stats[name_a]["total_orders"],
                "blocked_orders": stats[name_a]["blocked_orders"],
                "blocked_order_rate": round(stats[name_a]["blocked_orders"] / max(stats[name_a]["total_orders"], 1), 6),
                "blocked_orders_per_thousand": round(stats[name_a]["blocked_orders"] / max(stats[name_a]["total_orders"], 1) * 1000.0, 3),
                "collision_rate": round(stats[name_a]["collisions"] / max(stats[name_a]["total_orders"], 1), 6),
                "infer_us_avg_per_match": round(stats[name_a]["infer_us_total"] / total_matches, 2),
                "infer_us_max": stats[name_a]["infer_us_max"],
            },
            name_b: {
                "wins": stats[name_b]["wins"],
                "draws": stats[name_b]["draws"],
                "losses": stats[name_b]["losses"],
                "points": stats[name_b]["points"],
                "win_rate": round(win_rate_b, 4),
                "score_rate": round(score_rate_b, 4),
                "total_score": stats[name_b]["total_score"],
                "mean_score": round(mean_score_b, 4),
                "std_score": round(std_score_b, 4),
                "median_score": sorted(stats[name_b]["scores"])[len(stats[name_b]["scores"]) // 2],
                "score_distribution": dict(sorted(stats[name_b]["score_frequencies"].items())),
                "hills_razed": stats[name_b]["hills_razed"],
                "raze_rate": round(stats[name_b]["hills_razed"] / total_matches, 4),
                "mean_raze_turn": round(sum(stats[name_b]["raze_turns"]) / len(stats[name_b]["raze_turns"]), 2) if stats[name_b]["raze_turns"] else None,
                "early_raze_rate": round(stats[name_b]["early_raze_count"] / total_matches, 4),
                "ants_spawned": stats[name_b]["ants_spawned"],
                "combat_deaths": stats[name_b]["combat_deaths"],
                "collisions": stats[name_b]["collisions"],
                "kd_ratio": round(stats[name_a]["combat_deaths"] / max(stats[name_b]["combat_deaths"], 1), 4),
                "total_orders": stats[name_b]["total_orders"],
                "blocked_orders": stats[name_b]["blocked_orders"],
                "blocked_order_rate": round(stats[name_b]["blocked_orders"] / max(stats[name_b]["total_orders"], 1), 6),
                "blocked_orders_per_thousand": round(stats[name_b]["blocked_orders"] / max(stats[name_b]["total_orders"], 1) * 1000.0, 3),
                "collision_rate": round(stats[name_b]["collisions"] / max(stats[name_b]["total_orders"], 1), 6),
                "infer_us_avg_per_match": round(stats[name_b]["infer_us_total"] / total_matches, 2),
                "infer_us_max": stats[name_b]["infer_us_max"],
            },
            "comparison": {
                "score_difference_total": stats[name_a]["total_score"] - stats[name_b]["total_score"],
                "score_margin_mean": round(mean_margin, 4),
                "score_margin_std": round(std_margin, 4),
                "score_margin_95ci": [round(mean_margin - ci_margin_95, 4), round(mean_margin + ci_margin_95, 4)],
                "score_ratio_a_over_b": round(stats[name_a]["total_score"] / max(stats[name_b]["total_score"], 1), 4),
                "margin_frequencies": dict(sorted(margin_frequencies.items())),
                "blocked_order_reduction_rate": round(1.0 - (stats[name_a]["blocked_orders"] / max(stats[name_b]["blocked_orders"], 1)), 4),
                "mean_turns": round(mean_turns, 2),
                "termination_reasons": dict(termination_reasons),
            },
        },
        "seat_analysis": {
            "seat0_is_model_a": {
                "matches": stats[name_a]["seat0_matches"],
                f"{name_a}_wins": stats[name_a]["seat0_wins"],
                f"{name_a}_draws": stats[name_a]["seat0_draws"],
                f"{name_a}_losses": stats[name_a]["seat0_losses"],
                f"{name_a}_points": stats[name_a]["seat0_points"],
                f"{name_a}_score_rate": round(stats[name_a]["seat0_points"] / max(stats[name_a]["seat0_matches"], 1), 4),
                f"{name_a}_total_score": stats[name_a]["seat0_score"],
                f"{name_a}_mean_score": round(stats[name_a]["seat0_score"] / max(stats[name_a]["seat0_matches"], 1), 4),
                f"{name_b}_total_score": stats[name_b]["seat1_score"],
                f"{name_b}_mean_score": round(stats[name_b]["seat1_score"] / max(stats[name_b]["seat1_matches"], 1), 4),
            },
            "seat0_is_model_b": {
                "matches": stats[name_a]["seat1_matches"],
                f"{name_a}_wins": stats[name_a]["seat1_wins"],
                f"{name_a}_draws": stats[name_a]["seat1_draws"],
                f"{name_a}_losses": stats[name_a]["seat1_losses"],
                f"{name_a}_points": stats[name_a]["seat1_points"],
                f"{name_a}_score_rate": round(stats[name_a]["seat1_points"] / max(stats[name_a]["seat1_matches"], 1), 4),
                f"{name_a}_total_score": stats[name_a]["seat1_score"],
                f"{name_a}_mean_score": round(stats[name_a]["seat1_score"] / max(stats[name_a]["seat1_matches"], 1), 4),
                f"{name_b}_total_score": stats[name_b]["seat0_score"],
                f"{name_b}_mean_score": round(stats[name_b]["seat0_score"] / max(stats[name_b]["seat0_matches"], 1), 4),
            },
            "seat_symmetry_delta": {
                f"{name_a}_score_rate_seat0_minus_seat1": round(
                    (stats[name_a]["seat0_points"] / max(stats[name_a]["seat0_matches"], 1)) -
                    (stats[name_a]["seat1_points"] / max(stats[name_a]["seat1_matches"], 1)), 4
                ),
            },
        },
        "map_breakdown": {},
    }

    for map_stem, m_data in sorted(map_stats.items()):
        m_matches = m_data["matches"]
        pts_a = m_data[name_a]["points"]
        pts_b = m_data[name_b]["points"]
        sc_a = m_data[name_a]["score"]
        sc_b = m_data[name_b]["score"]
        std_sc_a = calculate_std(m_data[name_a]["scores"], sc_a / m_matches)
        std_sc_b = calculate_std(m_data[name_b]["scores"], sc_b / m_matches)

        report["map_breakdown"][map_stem] = {
            "matches": m_matches,
            f"{name_a}_wins": m_data[name_a]["wins"],
            f"{name_a}_draws": m_data[name_a]["draws"],
            f"{name_a}_losses": m_data[name_a]["losses"],
            f"{name_a}_score_rate": round(pts_a / m_matches, 4),
            f"{name_a}_total_score": sc_a,
            f"{name_a}_mean_score": round(sc_a / m_matches, 4),
            f"{name_a}_std_score": round(std_sc_a, 4),
            f"{name_b}_total_score": sc_b,
            f"{name_b}_mean_score": round(sc_b / m_matches, 4),
            f"{name_b}_std_score": round(std_sc_b, 4),
            "score_differential_mean": round((sc_a - sc_b) / m_matches, 4),
            f"{name_a}_hills_razed": m_data[name_a]["hills_razed"],
            f"{name_b}_hills_razed": m_data[name_b]["hills_razed"],
            f"{name_a}_mean_raze_turn": round(sum(m_data[name_a]["raze_turns"]) / len(m_data[name_a]["raze_turns"]), 2) if m_data[name_a]["raze_turns"] else None,
            f"{name_b}_mean_raze_turn": round(sum(m_data[name_b]["raze_turns"]) / len(m_data[name_b]["raze_turns"]), 2) if m_data[name_b]["raze_turns"] else None,
            f"{name_a}_blocked_rate": round(m_data[name_a]["blocked"] / max(m_data[name_a]["orders"], 1), 6),
            f"{name_b}_blocked_rate": round(m_data[name_b]["blocked"] / max(m_data[name_b]["orders"], 1), 6),
            f"{name_a}_spawned": m_data[name_a]["spawned"],
            f"{name_b}_spawned": m_data[name_b]["spawned"],
            f"{name_a}_combat_deaths": m_data[name_a]["combat_deaths"],
            f"{name_b}_combat_deaths": m_data[name_b]["combat_deaths"],
            "mean_turns": round(sum(m_data["turns"]) / len(m_data["turns"]), 2),
        }

    return report

def main() -> None:
    parser = argparse.ArgumentParser(description="EXP-083 对决 EXP-033 官方环境大规模对抗赛执行工具")
    parser.add_argument("--model-a", type=Path, default=ROOT / "models" / "exp083", help="模型 A 目录")
    parser.add_argument("--model-b", type=Path, default=ROOT / "models" / "exp033", help="模型 B 目录")
    parser.add_argument("--maps", type=Path, default=ROOT / "benchmarks" / "maps_2p", help="官方双人基准地图目录")
    parser.add_argument("--rounds", type=int, default=50, help="随机种子循环轮数（每轮 18 场，默认 50 轮共 900 场）")
    parser.add_argument("--workers", type=int, default=16, help="并行工作进程数（默认 16）")
    parser.add_argument("--max-turns", type=int, default=300, help="单场对局最大回合数（默认 300）")
    parser.add_argument("--seed", type=int, default=95000, help="基准随机数种子")
    parser.add_argument("--replays", type=Path, default=ROOT / "replays" / "h2h_exp083_vs_exp033", help="回放存储目录")
    parser.add_argument("--out", type=Path, default=ROOT / "runs" / "h2h_exp083_vs_exp033" / "evaluation_report.json", help="评测分析报告输出路径")
    args = parser.parse_args()

    report = run_tournament(
        model_a_dir=args.model_a,
        model_b_dir=args.model_b,
        maps_dir=args.maps,
        out_dir=args.replays,
        rounds=args.rounds,
        max_turns=args.max_turns,
        workers=args.workers,
        base_seed=args.seed,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"评测结果报告已写入: {args.out}")

if __name__ == "__main__":
    main()
