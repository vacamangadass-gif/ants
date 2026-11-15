import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path("/home/host/project/ants")
CLI = os.environ.get("TINYBRAINS") or shutil.which("tinybrains") or "/home/linuxbrew/.linuxbrew/bin/tinybrains"
REGISTRY = str((ROOT / "baselines" / "games.toml").resolve())
ANALYZER = ROOT / "tools" / "analyze_behavior.mjs"

REPLAYS_DIR = ROOT / "replays" / "challenge_m2_2"
REPORT_FILE = ROOT / "runs" / "challenge_exp084_behavior_report.json"

TARGET_DIR = ROOT / "models" / "exp084"
OPPONENTS = [
    ROOT / "models" / "exp083",
    ROOT / "models" / "exp033",
    ROOT / "models" / "exp031",
    ROOT / "models" / "exp011",
]

MAP_FILES = [
    ROOT / "benchmarks" / "maps_2p" / "tiny-corridor-2p-1h.json",
    ROOT / "benchmarks" / "maps_2p" / "medium-labyrinth-2p-1h.json",
]

SEEDS = [84001, 84002, 84003, 84004, 84005]


def verify_engine_invariants():
    status_res = subprocess.run(
        ["git", "status", "-s", "engine/"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )
    if status_res.returncode != 0:
        raise RuntimeError(f"执行 git status 失败: {status_res.stderr}")
    if status_res.stdout.strip():
        raise RuntimeError(f"engine/ 目录存在修改差异:\n{status_res.stdout}")

    diff_res = subprocess.run(
        ["git", "diff", "--exit-code", "engine/"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )
    if diff_res.returncode != 0:
        raise RuntimeError(f"engine/ 目录存在差异代码:\n{diff_res.stdout}")

    deny_res = subprocess.run(
        ["./tools/deny.sh"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )
    if deny_res.returncode != 0:
        raise RuntimeError(f"tools/deny.sh 检测未通过:\n{deny_res.stderr or deny_res.stdout}")


def build_match_rows():
    rows = []
    for map_path in MAP_FILES:
        map_name = map_path.stem
        # 针对历史基准模型进行双向席位互换测试
        for opp_dir in OPPONENTS:
            opp_name = opp_dir.name
            for seed in SEEDS:
                # 席位 0 为 exp084，席位 1 为对手
                row_id_1 = f"live-exp084-vs-{opp_name}-{map_name}-s{seed}-seat0"
                rows.append({
                    "id": row_id_1,
                    "seed": seed,
                    "map": str(map_path.resolve()),
                    "seat_count": 2,
                    "seats": [
                        {
                            "seat": 0,
                            "label": "exp084",
                            "weights": str((TARGET_DIR / "model.onnx").resolve()),
                            "manifest": str((TARGET_DIR / "manifest.json").resolve()),
                        },
                        {
                            "seat": 1,
                            "label": opp_name,
                            "weights": str((opp_dir / "model.onnx").resolve()),
                            "manifest": str((opp_dir / "manifest.json").resolve()),
                        },
                    ],
                })

                # 席位 0 为对手，席位 1 为 exp084（席位互换）
                row_id_2 = f"live-{opp_name}-vs-exp084-{map_name}-s{seed}-seat1"
                rows.append({
                    "id": row_id_2,
                    "seed": seed + 500,
                    "map": str(map_path.resolve()),
                    "seat_count": 2,
                    "seats": [
                        {
                            "seat": 0,
                            "label": opp_name,
                            "weights": str((opp_dir / "model.onnx").resolve()),
                            "manifest": str((opp_dir / "manifest.json").resolve()),
                        },
                        {
                            "seat": 1,
                            "label": "exp084",
                            "weights": str((TARGET_DIR / "model.onnx").resolve()),
                            "manifest": str((TARGET_DIR / "manifest.json").resolve()),
                        },
                    ],
                })

        # 自博弈（测试同质高密度狭窄通道对抗与避免死锁能力）
        for seed in SEEDS:
            row_id_self = f"live-exp084-vs-exp084-{map_name}-s{seed}-selfplay"
            rows.append({
                "id": row_id_self,
                "seed": seed + 1000,
                "map": str(map_path.resolve()),
                "seat_count": 2,
                "seats": [
                    {
                        "seat": 0,
                        "label": "exp084",
                        "weights": str((TARGET_DIR / "model.onnx").resolve()),
                        "manifest": str((TARGET_DIR / "manifest.json").resolve()),
                    },
                    {
                        "seat": 1,
                        "label": "exp084",
                        "weights": str((TARGET_DIR / "model.onnx").resolve()),
                        "manifest": str((TARGET_DIR / "manifest.json").resolve()),
                    },
                ],
            })

    return rows


def execute_batch(batch_args):
    worker_id, chunk_rows, out_dir = batch_args
    batch_file = out_dir / f"_batch_{worker_id}.json"
    batch_data = {
        "game": "ants",
        "vars": {"max_turns": 300},
        "rows": chunk_rows,
    }
    batch_file.write_text(json.dumps(batch_data, ensure_ascii=False))

    env = os.environ.copy()
    env["TINYBRAINS_REGISTRY"] = REGISTRY

    t0 = time.perf_counter()
    res = subprocess.run(
        [CLI, str(batch_file), "--out", str(out_dir)],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        env=env,
    )
    duration = time.perf_counter() - t0

    if res.returncode != 0:
        raise RuntimeError(f"工作进程 {worker_id} 运行对局失败:\n{res.stderr or res.stdout}")

    try:
        batch_file.unlink(missing_ok=True)
    except Exception:
        pass

    return len(chunk_rows), duration


def run_all_matches(rows, workers=8):
    if REPLAYS_DIR.exists():
        shutil.rmtree(REPLAYS_DIR)
    REPLAYS_DIR.mkdir(parents=True, exist_ok=True)

    chunks = [[] for _ in range(workers)]
    for idx, row in enumerate(rows):
        chunks[idx % workers].append(row)

    chunks = [c for c in chunks if c]
    batch_args = [(i, chunk, REPLAYS_DIR) for i, chunk in enumerate(chunks)]

    with ProcessPoolExecutor(max_workers=len(chunks)) as executor:
        results = list(executor.map(execute_batch, batch_args))

    total_matches = sum(r[0] for r in results)
    total_time = sum(r[1] for r in results)
    return total_matches, total_time


def analyze_matches(target_name, target_dir):
    node_cmd = ["node", str(ANALYZER), target_name, str(target_dir.resolve())]
    node_res = subprocess.run(
        node_cmd,
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )
    if node_res.returncode != 0:
        raise RuntimeError(f"行为仿真分析失败 ({target_name}):\n{node_res.stderr or node_res.stdout}")

    try:
        data = json.loads(node_res.stdout)
    except Exception as e:
        raise RuntimeError(f"解析分析结果 JSON 异常: {e}\n输出内容:\n{node_res.stdout}")

    return data


def extract_match_outcomes(replays_dir, target_name="exp084"):
    outcomes = []
    for p in sorted(replays_dir.glob("*.json")):
        if p.name.startswith("_batch_"):
            continue
        try:
            content = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue

        seats = content.get("seats", [])
        ranks = content.get("engine_ranks", [])
        scores = content.get("scores", [])
        map_id = content.get("map_id", "")
        turns = content.get("turns", 0)
        reason = content.get("reason", "")
        match_id = content.get("match_id", p.stem)

        target_seat = None
        rival_seat = None
        rival_name = None

        for s in seats:
            lbl = s.get("label", "")
            s_idx = s.get("seat", 0)
            if lbl == target_name and target_seat is None:
                target_seat = s_idx
            else:
                rival_seat = s_idx
                rival_name = lbl

        is_self_play = (rival_name == target_name)
        if target_seat is not None and len(ranks) >= 2:
            t_rank = ranks[target_seat]
            r_rank = ranks[rival_seat] if rival_seat is not None else ranks[1 - target_seat]
            t_score = scores[target_seat] if len(scores) > target_seat else 0
            r_score = scores[rival_seat] if rival_seat is not None and len(scores) > rival_seat else 0

            if is_self_play:
                outcome = "self_play"
            elif t_rank < r_rank:
                outcome = "win"
            elif t_rank > r_rank:
                outcome = "loss"
            else:
                outcome = "draw"

            outcomes.append({
                "match_id": match_id,
                "map_id": map_id,
                "target_seat": target_seat,
                "rival_name": rival_name,
                "is_self_play": is_self_play,
                "outcome": outcome,
                "target_rank": t_rank,
                "rival_rank": r_rank,
                "target_score": t_score,
                "rival_score": r_score,
                "turns": turns,
                "reason": reason,
            })
    return outcomes


def aggregate_behavior_metrics(match_records):
    total_matches = len(match_records)
    total_orders = sum(m["total_orders"] for m in match_records)
    blocked_orders = sum(m["blocked_orders"] for m in match_records)
    blocked_rate = (blocked_orders / total_orders) if total_orders > 0 else 0.0

    total_consecutive_blocked = sum(m["consecutive_blocked_episodes"] for m in match_records)
    total_mutual_blocks = sum(m["mutual_head_on_blocks"] for m in match_records)
    total_oscillations = sum(m["total_oscillation_episodes"] for m in match_records)
    max_oscillation_streak = max((m["max_oscillation_streak"] for m in match_records), default=0)
    oscillating_ants_total = sum(m["oscillating_ants_count"] for m in match_records)
    total_ants_tracked = sum(m["total_ants_tracked"] for m in match_records)

    total_food_collected = sum(m["food_collected"] for m in match_records)
    total_turns = sum(m["turns"] for m in match_records)
    food_per_100_turns = (total_food_collected / total_turns * 100.0) if total_turns > 0 else 0.0

    hills_razed = sum(1 for m in match_records if m["enemy_hill_razed"])
    hills_reached = sum(1 for m in match_records if m["enemy_hill_reached"])
    close_to_enemy_hill = sum(1 for m in match_records if m["min_enemy_hill_dist"] <= 2)

    by_map = {}
    for m in match_records:
        mid = m["map_id"]
        if mid not in by_map:
            by_map[mid] = []
        by_map[mid].append(m)

    map_summaries = {}
    for mid, mlist in by_map.items():
        m_orders = sum(x["total_orders"] for x in mlist)
        m_blocked = sum(x["blocked_orders"] for x in mlist)
        m_turns = sum(x["turns"] for x in mlist)
        m_food = sum(x["food_collected"] for x in mlist)
        map_summaries[mid] = {
            "matches": len(mlist),
            "total_orders": m_orders,
            "blocked_orders": m_blocked,
            "blocked_rate": round((m_blocked / m_orders) if m_orders > 0 else 0.0, 6),
            "blocked_rate_pct": round((m_blocked / m_orders * 100.0) if m_orders > 0 else 0.0, 4),
            "consecutive_blocked_episodes": sum(x["consecutive_blocked_episodes"] for x in mlist),
            "mutual_head_on_blocks": sum(x["mutual_head_on_blocks"] for x in mlist),
            "oscillation_episodes": sum(x["total_oscillation_episodes"] for x in mlist),
            "max_oscillation_streak": max((x["max_oscillation_streak"] for x in mlist), default=0),
            "food_collected": m_food,
            "food_per_100_turns": round((m_food / m_turns * 100.0) if m_turns > 0 else 0.0, 2),
            "hills_razed": sum(1 for x in mlist if x["enemy_hill_razed"]),
            "close_to_enemy_hill": sum(1 for x in mlist if x["min_enemy_hill_dist"] <= 2),
            "avg_min_enemy_hill_dist": round(sum(x["min_enemy_hill_dist"] for x in mlist) / len(mlist), 2),
        }

    return {
        "total_matches": total_matches,
        "total_orders": total_orders,
        "blocked_orders": blocked_orders,
        "blocked_rate": round(blocked_rate, 6),
        "blocked_rate_pct": round(blocked_rate * 100.0, 4),
        "consecutive_blocked_episodes": total_consecutive_blocked,
        "mutual_head_on_blocks": total_mutual_blocks,
        "total_oscillation_episodes": total_oscillations,
        "max_oscillation_streak": max_oscillation_streak,
        "oscillating_ants_total": oscillating_ants_total,
        "total_ants_tracked": total_ants_tracked,
        "food_collected": total_food_collected,
        "total_turns": total_turns,
        "food_per_100_turns": round(food_per_100_turns, 2),
        "hills_razed": hills_razed,
        "hills_reached": hills_reached,
        "close_to_enemy_hill": close_to_enemy_hill,
        "by_map": map_summaries,
    }


def main():
    verify_engine_invariants()

    rows = build_match_rows()
    total_matches, total_time = run_all_matches(rows, workers=8)

    live_records_exp084 = analyze_matches("exp084", REPLAYS_DIR)
    live_summary_exp084 = aggregate_behavior_metrics(live_records_exp084)

    # 提取比赛胜负判定
    outcomes = extract_match_outcomes(REPLAYS_DIR, target_name="exp084")
    h2h_stats = {}
    for opp_dir in OPPONENTS:
        h2h_stats[opp_dir.name] = {"matches": 0, "wins": 0, "draws": 0, "losses": 0, "points": 0.0}

    for oc in outcomes:
        if oc["is_self_play"]:
            continue
        rival = oc["rival_name"]
        if rival in h2h_stats:
            h2h_stats[rival]["matches"] += 1
            if oc["outcome"] == "win":
                h2h_stats[rival]["wins"] += 1
                h2h_stats[rival]["points"] += 1.0
            elif oc["outcome"] == "draw":
                h2h_stats[rival]["draws"] += 1
                h2h_stats[rival]["points"] += 0.5
            elif oc["outcome"] == "loss":
                h2h_stats[rival]["losses"] += 1

    for rival, st in h2h_stats.items():
        st["score_rate"] = round(st["points"] / st["matches"], 4) if st["matches"] > 0 else 0.0

    total_h2h_matches = sum(st["matches"] for st in h2h_stats.values())
    total_h2h_wins = sum(st["wins"] for st in h2h_stats.values())
    total_h2h_draws = sum(st["draws"] for st in h2h_stats.values())
    total_h2h_losses = sum(st["losses"] for st in h2h_stats.values())
    total_h2h_points = sum(st["points"] for st in h2h_stats.values())
    total_h2h_score_rate = round(total_h2h_points / total_h2h_matches, 4) if total_h2h_matches > 0 else 0.0

    # 对手基准模型在相同对局中的行为统计分析（作为严谨的对比对照组）
    baseline_comparisons = {}
    for opp_dir in OPPONENTS:
        opp_name = opp_dir.name
        opp_records = analyze_matches(opp_name, REPLAYS_DIR)
        if opp_records:
            opp_summary = aggregate_behavior_metrics(opp_records)
            baseline_comparisons[opp_name] = {
                "matches": opp_summary["total_matches"],
                "total_orders": opp_summary["total_orders"],
                "blocked_orders": opp_summary["blocked_orders"],
                "blocked_rate_pct": opp_summary["blocked_rate_pct"],
                "food_per_100_turns": opp_summary["food_per_100_turns"],
                "hills_razed": opp_summary["hills_razed"],
                "oscillation_episodes": opp_summary["total_oscillation_episodes"],
                "mutual_head_on_blocks": opp_summary["mutual_head_on_blocks"],
            }

    # 交叉核验锦标赛中 9 图全量回放
    tournament_replays_dir = ROOT / "replays" / "tournament_exp084"
    tournament_records = []
    if tournament_replays_dir.exists():
        all_tourney = analyze_matches("exp084", tournament_replays_dir)
        tournament_records = [
            r for r in all_tourney
            if r["map_id"] in ("tiny-corridor-2p-1h", "medium-labyrinth-2p-1h")
        ]
    tournament_summary = aggregate_behavior_metrics(tournament_records) if tournament_records else None

    # 门禁核验判据：
    # 1. 窄通道受阻率必须 <= 0.80% (0.0080)
    # 2. 必须具备主动采食能力 (food_collected > 0)
    # 3. 必须具备向敌方蚁穴突进/摧毁能力 (close_to_enemy_hill > 0 or hills_razed > 0)
    # 4. 底座确定性与纯整数门禁必须通过
    blocked_rate_pass = (live_summary_exp084["blocked_rate"] <= 0.0080)
    food_seeking_pass = (live_summary_exp084["food_collected"] > 0)
    hill_advance_pass = (live_summary_exp084["close_to_enemy_hill"] > 0 or live_summary_exp084["hills_razed"] > 0)
    zero_engine_diff_pass = True

    overall_pass = (
        blocked_rate_pass and
        food_seeking_pass and
        hill_advance_pass and
        zero_engine_diff_pass
    )

    final_report = {
        "verdict": "APPROVE" if overall_pass else "REJECT",
        "criteria_checks": {
            "blocked_rate_le_0_8_pct": {
                "threshold": 0.0080,
                "achieved": live_summary_exp084["blocked_rate"],
                "achieved_pct": live_summary_exp084["blocked_rate_pct"],
                "passed": blocked_rate_pass,
            },
            "active_food_seeking": {
                "food_collected": live_summary_exp084["food_collected"],
                "food_per_100_turns": live_summary_exp084["food_per_100_turns"],
                "passed": food_seeking_pass,
            },
            "advance_towards_enemy_hills": {
                "hills_razed": live_summary_exp084["hills_razed"],
                "close_to_enemy_hill": live_summary_exp084["close_to_enemy_hill"],
                "passed": hill_advance_pass,
            },
            "corridor_movement_stability": {
                "oscillation_episodes": live_summary_exp084["total_oscillation_episodes"],
                "max_oscillation_streak": live_summary_exp084["max_oscillation_streak"],
                "mutual_head_on_blocks": live_summary_exp084["mutual_head_on_blocks"],
                "consecutive_blocked_episodes": live_summary_exp084["consecutive_blocked_episodes"],
            },
            "base_engine_invariants": {
                "engine_zero_git_diff": True,
                "tools_deny_sh_exit_0": True,
                "passed": True,
            },
        },
        "live_head_to_head": {
            "total_matches": total_h2h_matches,
            "wins": total_h2h_wins,
            "draws": total_h2h_draws,
            "losses": total_h2h_losses,
            "points": total_h2h_points,
            "score_rate": total_h2h_score_rate,
            "opponents": h2h_stats,
        },
        "baseline_comparisons_in_same_matches": baseline_comparisons,
        "live_test_summary": live_summary_exp084,
        "tournament_cross_check_summary": tournament_summary,
        "live_records_exp084": live_records_exp084,
    }

    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
    REPORT_FILE.write_text(json.dumps(final_report, indent=2, ensure_ascii=False))

    print(json.dumps({
        "verdict": final_report["verdict"],
        "total_live_matches": total_matches,
        "h2h_matches": total_h2h_matches,
        "h2h_record": f"{total_h2h_wins}W-{total_h2h_draws}D-{total_h2h_losses}L",
        "h2h_score_rate": total_h2h_score_rate,
        "exp084_blocked_rate_pct": live_summary_exp084["blocked_rate_pct"],
        "exp084_food_collected": live_summary_exp084["food_collected"],
        "exp084_food_per_100_turns": live_summary_exp084["food_per_100_turns"],
        "exp084_hills_razed": live_summary_exp084["hills_razed"],
        "exp084_hills_reached": live_summary_exp084["hills_reached"],
        "exp084_close_to_enemy_hill": live_summary_exp084["close_to_enemy_hill"],
        "oscillation_episodes": live_summary_exp084["total_oscillation_episodes"],
        "mutual_head_on_blocks": live_summary_exp084["mutual_head_on_blocks"],
        "consecutive_blocked_episodes": live_summary_exp084["consecutive_blocked_episodes"],
    }, indent=2))


if __name__ == "__main__":
    main()
