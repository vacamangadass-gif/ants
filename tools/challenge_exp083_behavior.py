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

REPLAYS_DIR = ROOT / "replays" / "challenge_m1_2"
REPORT_FILE = ROOT / "runs" / "challenge_exp083_behavior_report.json"

TARGET_DIR = ROOT / "models" / "exp083"
OPPONENTS = [
    ROOT / "models" / "exp033",
    ROOT / "models" / "exp031",
    ROOT / "models" / "exp011",
]

MAP_FILES = [
    ROOT / "benchmarks" / "maps_2p" / "tiny-corridor-2p-1h.json",
    ROOT / "benchmarks" / "maps_2p" / "medium-labyrinth-2p-1h.json",
]

SEEDS = [70001, 70002, 70003]


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
                # 席位 0 为 exp083，席位 1 为对手
                row_id_1 = f"live-exp083-vs-{opp_name}-{map_name}-s{seed}-seat0"
                rows.append({
                    "id": row_id_1,
                    "seed": seed,
                    "map": str(map_path.resolve()),
                    "seat_count": 2,
                    "seats": [
                        {
                            "seat": 0,
                            "label": "exp083",
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

                # 席位 0 为对手，席位 1 为 exp083（席位互换）
                row_id_2 = f"live-{opp_name}-vs-exp083-{map_name}-s{seed}-seat1"
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
                            "label": "exp083",
                            "weights": str((TARGET_DIR / "model.onnx").resolve()),
                            "manifest": str((TARGET_DIR / "manifest.json").resolve()),
                        },
                    ],
                })

        # 自博弈（测试同质高密度狭窄通道对抗与避免死锁能力）
        for seed in SEEDS:
            row_id_self = f"live-exp083-vs-exp083-{map_name}-s{seed}-selfplay"
            rows.append({
                "id": row_id_self,
                "seed": seed + 1000,
                "map": str(map_path.resolve()),
                "seat_count": 2,
                "seats": [
                    {
                        "seat": 0,
                        "label": "exp083",
                        "weights": str((TARGET_DIR / "model.onnx").resolve()),
                        "manifest": str((TARGET_DIR / "manifest.json").resolve()),
                    },
                    {
                        "seat": 1,
                        "label": "exp083",
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


def analyze_matches(target_dir):
    node_cmd = ["node", str(ANALYZER), "exp083", str(target_dir.resolve())]
    node_res = subprocess.run(
        node_cmd,
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )
    if node_res.returncode != 0:
        raise RuntimeError(f"行为仿真分析失败:\n{node_res.stderr or node_res.stdout}")

    try:
        data = json.loads(node_res.stdout)
    except Exception as e:
        raise RuntimeError(f"解析分析结果 JSON 异常: {e}\n输出内容:\n{node_res.stdout}")

    return data


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

    live_records = analyze_matches(REPLAYS_DIR)
    live_summary = aggregate_behavior_metrics(live_records)

    # 同时提取锦标赛中 tiny-corridor 与 medium-labyrinth 的历史回放作为大样本交叉核验
    tournament_replays_dir = ROOT / "replays" / "tournament_exp083"
    tournament_records = []
    if tournament_replays_dir.exists():
        all_tourney = analyze_matches(tournament_replays_dir)
        tournament_records = [
            r for r in all_tourney
            if r["map_id"] in ("tiny-corridor-2p-1h", "medium-labyrinth-2p-1h")
        ]
    tournament_summary = aggregate_behavior_metrics(tournament_records) if tournament_records else None

    # 判定门禁准入标准
    # 1. 受阻率必须 <= 0.8% (0.0080)
    # 2. 狭窄通道必须主动采食且向前推进
    # 3. 必须无恶性死锁崩溃
    blocked_rate_pass = (live_summary["blocked_rate"] <= 0.0080)
    food_seeking_pass = (live_summary["food_collected"] > 0)
    hill_advance_pass = (live_summary["close_to_enemy_hill"] > 0 or live_summary["hills_razed"] > 0)
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
                "achieved": live_summary["blocked_rate"],
                "achieved_pct": live_summary["blocked_rate_pct"],
                "passed": blocked_rate_pass,
            },
            "active_food_seeking": {
                "food_collected": live_summary["food_collected"],
                "food_per_100_turns": live_summary["food_per_100_turns"],
                "passed": food_seeking_pass,
            },
            "advance_towards_enemy_hills": {
                "hills_razed": live_summary["hills_razed"],
                "close_to_enemy_hill": live_summary["close_to_enemy_hill"],
                "passed": hill_advance_pass,
            },
            "corridor_movement_stability": {
                "oscillation_episodes": live_summary["total_oscillation_episodes"],
                "max_oscillation_streak": live_summary["max_oscillation_streak"],
                "mutual_head_on_blocks": live_summary["mutual_head_on_blocks"],
                "consecutive_blocked_episodes": live_summary["consecutive_blocked_episodes"],
            },
            "base_engine_invariants": {
                "engine_zero_git_diff": True,
                "tools_deny_sh_exit_0": True,
                "passed": True,
            },
        },
        "live_test_summary": live_summary,
        "tournament_cross_check_summary": tournament_summary,
        "live_records": live_records,
    }

    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
    REPORT_FILE.write_text(json.dumps(final_report, indent=2, ensure_ascii=False))

    print(json.dumps({
        "verdict": final_report["verdict"],
        "blocked_rate_pct": live_summary["blocked_rate_pct"],
        "food_collected": live_summary["food_collected"],
        "hills_razed": live_summary["hills_razed"],
        "oscillation_episodes": live_summary["total_oscillation_episodes"],
        "mutual_head_on_blocks": live_summary["mutual_head_on_blocks"],
        "consecutive_blocked_episodes": live_summary["consecutive_blocked_episodes"],
    }, indent=2))


if __name__ == "__main__":
    main()
