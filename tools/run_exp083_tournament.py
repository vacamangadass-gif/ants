from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CLI = os.environ.get("TINYBRAINS") or shutil.which("tinybrains") or "/home/linuxbrew/.linuxbrew/bin/tinybrains"
REGISTRY = os.environ.get("TINYBRAINS_REGISTRY") or str((ROOT / "baselines" / "games.toml").resolve())
EXTRACTOR = ROOT / "benchmarks" / "extract_frames.mjs"


def aggregate_exp083_metrics(
    frame_metrics: list[dict[str, Any]],
    target_name: str,
    wall_clock: float = 0.0,
) -> dict[str, Any]:
    total_matches = 0
    wins = 0
    draws = 0
    losses = 0
    points = 0.0
    score_for = 0
    score_against = 0
    food_gathered = 0
    ants_spawned_total = 0
    total_turns = 0
    total_orders = 0
    blocked_orders = 0
    all_orders = 0
    hold_orders = 0
    on_own_hill_steps = 0
    on_own_hill_hold_orders = 0
    idle3_steps = 0
    idle3_eligible = 0
    collisions = 0
    combat_deaths = 0
    combat_kills = 0.0
    hills_razed = 0
    hills_lost = 0
    raze_turns: list[int] = []

    h2h_records: dict[str, dict[str, Any]] = {}
    map_records: dict[str, dict[str, Any]] = {}

    for match in frame_metrics:
        seats = match["seats"]
        map_id = match["map_id"]
        turns = int(match.get("turns", 0))

        my_seat = None
        rival_seat = None
        for s in seats:
            if s["name"] == target_name and my_seat is None:
                my_seat = s
            else:
                rival_seat = s

        if my_seat is None or rival_seat is None:
            continue

        rival_name = rival_seat["name"]
        if rival_name not in h2h_records:
            h2h_records[rival_name] = {
                "matches": 0, "wins": 0, "draws": 0, "losses": 0, "points": 0.0,
                "score_for": 0, "score_against": 0, "blocked_orders": 0, "total_orders": 0,
                "ants_spawned": 0,
            }

        if map_id not in map_records:
            map_records[map_id] = {
                "matches": 0, "wins": 0, "draws": 0, "losses": 0, "points": 0.0,
                "score_for": 0, "score_against": 0, "food_gathered": 0, "total_turns": 0,
            }

        total_matches += 1
        h2h_records[rival_name]["matches"] += 1
        map_records[map_id]["matches"] += 1

        my_rank = my_seat["rank"]
        rival_rank = rival_seat["rank"]
        my_score = int(my_seat["score"])
        rival_score = int(rival_seat["score"])
        # 修复原脚本将 my_score (蚁穴得分) 误计入 food_gathered 的口径混淆缺陷：
        # 按引擎帧统计真实繁衍蚂蚁数 ants_spawned 度量采食转化通量，并与蚁穴得分严格分列
        my_ants_spawned = int(my_seat.get("ants_spawned", 0))

        score_for += my_score
        score_against += rival_score
        h2h_records[rival_name]["score_for"] += my_score
        h2h_records[rival_name]["score_against"] += rival_score
        h2h_records[rival_name]["ants_spawned"] += my_ants_spawned
        map_records[map_id]["score_for"] += my_score
        map_records[map_id]["score_against"] += rival_score

        food_gathered += my_ants_spawned
        ants_spawned_total += my_ants_spawned
        map_records[map_id]["food_gathered"] += my_ants_spawned
        total_turns += turns
        map_records[map_id]["total_turns"] += turns

        t_ord = int(my_seat.get("total_orders", 0))
        b_ord = int(my_seat.get("blocked_orders", 0))
        h_ord = int(my_seat.get("hold_orders", 0))
        a_ord = int(my_seat.get("all_orders", t_ord + h_ord))

        total_orders += t_ord
        blocked_orders += b_ord
        all_orders += a_ord
        hold_orders += h_ord
        on_own_hill_steps += int(my_seat.get("on_own_hill_steps", 0))
        on_own_hill_hold_orders += int(my_seat.get("on_own_hill_hold_orders", 0))
        idle3_steps += int(my_seat.get("idle3_steps", 0))
        idle3_eligible += int(my_seat.get("idle3_eligible", 0))
        h2h_records[rival_name]["total_orders"] += t_ord
        h2h_records[rival_name]["blocked_orders"] += b_ord

        collisions += int(my_seat.get("collisions", 0))
        combat_deaths += int(my_seat.get("combat_deaths", 0))
        combat_kills += float(my_seat.get("combat_kills", 0.0))
        hills_lost += int(my_seat.get("hills_lost", 0))
        if my_seat.get("hills_razed", 0) > 0:
            hills_razed += int(my_seat["hills_razed"])
            if my_seat.get("first_hill_destroy_turn") is not None:
                raze_turns.append(int(my_seat["first_hill_destroy_turn"]))

        if my_rank < rival_rank:
            wins += 1
            points += 1.0
            h2h_records[rival_name]["wins"] += 1
            h2h_records[rival_name]["points"] += 1.0
            map_records[map_id]["wins"] += 1
            map_records[map_id]["points"] += 1.0
        elif my_rank > rival_rank:
            losses += 1
            h2h_records[rival_name]["losses"] += 1
            map_records[map_id]["losses"] += 1
        else:
            draws += 1
            points += 0.5
            h2h_records[rival_name]["draws"] += 1
            h2h_records[rival_name]["points"] += 0.5
            map_records[map_id]["draws"] += 1
            map_records[map_id]["points"] += 0.5

    score_rate = points / total_matches if total_matches > 0 else 0.0
    blocked_rate = (blocked_orders / total_orders) if total_orders > 0 else 0.0
    hold_rate = (hold_orders / all_orders) if all_orders > 0 else 0.0
    own_hill_hold_rate = (on_own_hill_hold_orders / on_own_hill_steps) if on_own_hill_steps > 0 else 0.0
    idle3_rate = (idle3_steps / idle3_eligible) if idle3_eligible > 0 else 0.0
    food_rate_per_100_turns = (food_gathered / total_turns * 100.0) if total_turns > 0 else 0.0
    collision_rate = (collisions / total_orders) if total_orders > 0 else 0.0
    kd_ratio = (combat_kills / combat_deaths) if combat_deaths > 0 else 0.0

    gate3_2_pass = (score_rate >= 0.6574) and (blocked_rate <= 0.0080)

    return {
        "model": target_name,
        "gate3_2_pass": gate3_2_pass,
        "criteria": {
            "score_rate_threshold": 0.6574,
            "blocked_order_rate_threshold": 0.0080,
            "score_rate_achieved": round(score_rate, 4),
            "blocked_order_rate_achieved": round(blocked_rate, 6),
            "food_rate_per_100_turns": round(food_rate_per_100_turns, 2),
        },
        "overall": {
            "matches": total_matches,
            "wins": wins,
            "draws": draws,
            "losses": losses,
            "points": points,
            "score_rate": round(score_rate, 4),
            "score_for": score_for,
            "score_against": score_against,
            "hill_score_for": score_for,
            "ants_spawned": ants_spawned_total,
            "food_gathered": food_gathered,
            "total_turns": total_turns,
            "total_orders": total_orders,
            "blocked_orders": blocked_orders,
            "blocked_order_rate_pct": round(blocked_rate * 100.0, 4),
            "collision_rate_pct": round(collision_rate * 100.0, 4),
            "hold_rate_pct": round(hold_rate * 100.0, 2),
            "on_own_hill_hold_rate_pct": round(own_hill_hold_rate * 100.0, 2),
            "idle3_rate_pct": round(idle3_rate * 100.0, 2),
            "kd_ratio": round(kd_ratio, 3),
            "hills_razed": hills_razed,
            "hills_lost": hills_lost,
            "mean_raze_turn": round(sum(raze_turns) / len(raze_turns), 2) if raze_turns else None,
            "wall_clock_seconds": round(wall_clock, 2),
        },
        "head_to_head": {
            r_name: {
                "matches": r_stats["matches"],
                "wins": r_stats["wins"],
                "draws": r_stats["draws"],
                "losses": r_stats["losses"],
                "points": r_stats["points"],
                "score_rate": round(r_stats["points"] / r_stats["matches"], 4) if r_stats["matches"] > 0 else 0.0,
                "score_for": r_stats["score_for"],
                "score_against": r_stats["score_against"],
                "ants_spawned": r_stats["ants_spawned"],
                "blocked_order_rate_pct": round(r_stats["blocked_orders"] / max(r_stats["total_orders"], 1) * 100.0, 4),
            }
            for r_name, r_stats in sorted(h2h_records.items())
        },
        "map_breakdown": {
            m_id: {
                "matches": m_stats["matches"],
                "wins": m_stats["wins"],
                "draws": m_stats["draws"],
                "losses": m_stats["losses"],
                "points": m_stats["points"],
                "score_rate": round(m_stats["points"] / m_stats["matches"], 4) if m_stats["matches"] > 0 else 0.0,
                "score_for": m_stats["score_for"],
                "score_against": m_stats["score_against"],
                "food_gathered": m_stats["food_gathered"],
                "food_rate_per_100_turns": round(m_stats["food_gathered"] / max(m_stats["total_turns"], 1) * 100.0, 2),
            }
            for m_id, m_stats in sorted(map_records.items())
        },
    }


def run_tournament_evaluation(
    model_dir: Path,
    competitors: list[Path],
    maps_dir: Path,
    out_replays_dir: Path,
    seed: int = 90000,
    workers: int = 16,
    max_turns: int = 300,
    dist_dir: Path | None = None,
) -> dict[str, Any]:
    out_replays_dir.mkdir(parents=True, exist_ok=True)

    entrants = [model_dir] + competitors
    entrant_args = [str(e.resolve()) for e in entrants]

    runner_script = ROOT / "benchmarks" / "tournament_runner_p2.py"
    cmd = [
        sys.executable,
        str(runner_script),
        *entrant_args,
        "--maps", str(maps_dir.resolve()),
        "--boards", "9",
        "--max-turns", str(max_turns),
        "--seed", str(seed),
        "--workers", str(workers),
        "--out", str(out_replays_dir.resolve()),
        "--json",
    ]

    env = os.environ.copy()
    env["TINYBRAINS_REGISTRY"] = REGISTRY
    t0 = time.perf_counter()
    res = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, env=env)
    wall_clock = time.perf_counter() - t0

    if res.returncode != 0:
        raise RuntimeError(f"锦标赛执行失败:\n{res.stderr or res.stdout}")

    json_start = res.stdout.find("{")
    if json_start == -1:
        raise RuntimeError(f"锦标赛未输出有效 JSON:\n{res.stdout}")
    json_end = res.stdout.rfind("}") + 1
    _ = json.loads(res.stdout[json_start:json_end])

    node_env = os.environ.copy()
    if dist_dir is not None:
        node_env["ANTS_DIST_DIR"] = str(dist_dir.resolve())
    node_cmd = ["node", str(EXTRACTOR), str(out_replays_dir.resolve())]
    node_res = subprocess.run(node_cmd, cwd=str(ROOT / "benchmarks"), capture_output=True, text=True, env=node_env)
    if node_res.returncode != 0:
        raise RuntimeError(f"帧度量指标提取失败:\n{node_res.stderr or node_res.stdout}")

    frame_metrics = json.loads(node_res.stdout)
    return aggregate_exp083_metrics(frame_metrics, target_name=model_dir.name, wall_clock=wall_clock)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="EXP-083 Gate 3-2 锦标赛评测与门禁裁决器。")
    ap.add_argument("--model", type=Path, default=ROOT / "models" / "exp083", help="待评测模型目录（默认 models/exp083）")
    ap.add_argument("--maps", type=Path, default=ROOT / "benchmarks" / "maps_2p", help="基准地图目录（默认 benchmarks/maps_2p）")
    ap.add_argument("--replays", type=Path, default=ROOT / "runs" / "tournament_exp083", help="回放输出目录")
    ap.add_argument("--out", type=Path, default=ROOT / "runs" / "exp083" / "tournament_evaluation.json", help="评测结果输出文件")
    ap.add_argument("--workers", type=int, default=16, help="并行工作进程数（默认 16）")
    ap.add_argument("--seed", type=int, default=90000, help="随机种子（默认 90000）")
    ap.add_argument("--dist-dir", type=Path, default=None, help="可选 dist 目录（用于 worktree 复用）")
    return ap


def main() -> None:
    args = build_parser().parse_args()
    model_dir = args.model
    if not (model_dir / "model.onnx").exists():
        raise FileNotFoundError(f"待评测模型缺失: {model_dir / 'model.onnx'}")

    models_root = model_dir.resolve().parent
    competitors = [
        models_root / "exp033",
        models_root / "exp033_iter8",
        models_root / "exp033_iter7",
        models_root / "exp033_iter4",
        models_root / "exp033_iter1",
        models_root / "exp031",
        models_root / "exp031_iter8",
        models_root / "exp031_iter4",
        models_root / "exp031_iter1",
        models_root / "exp011",
    ]

    for comp in competitors:
        if not (comp / "model.onnx").exists():
            raise FileNotFoundError(f"竞争对手模型缺失: {comp / 'model.onnx'}")

    print(f"=== 开始 EXP-083 Gate 3-2 锦标赛评测 ({model_dir.name} 对决 {len(competitors)} 款基准模型) ===")
    report = run_tournament_evaluation(
        model_dir=model_dir,
        competitors=competitors,
        maps_dir=args.maps,
        out_replays_dir=args.replays,
        seed=args.seed,
        workers=args.workers,
        dist_dir=args.dist_dir,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print("\n=== EXP-083 Gate 3-2 门禁裁决结果 ===")
    print(f"综合得分率 (Score Rate): {report['criteria']['score_rate_achieved']:.2%} (阈值 >= 65.74%)")
    print(f"通道拥塞阻挡率 (Blocked Rate): {report['criteria']['blocked_order_rate_achieved']:.2%} (阈值 <= 0.80%)")
    print(f"百回合采食转化通量 (Ants Spawned / 100 turns): {report['criteria']['food_rate_per_100_turns']:.2f}")
    print(f"Gate 3-2 门禁判定状态: {'PASS (通过)' if report['gate3_2_pass'] else 'FAIL (未通过)'}")
    print(f"评测结果已写入: {args.out}")


if __name__ == "__main__":
    main()
