import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path("/home/host/project/ants")
CLI = os.environ.get("TINYBRAINS") or shutil.which("tinybrains") or "/home/linuxbrew/.linuxbrew/bin/tinybrains"
REGISTRY = str((ROOT / "baselines" / "games.toml").resolve())
EXTRACTOR = ROOT / "benchmarks" / "extract_frames.mjs"

MODELS = {
    "exp083": {
        "name": "exp083",
        "description": "MCMF 蒸馏模型 (Micro 28 通道)",
        "weights": ROOT / "models" / "exp083" / "model.onnx",
        "manifest": ROOT / "models" / "exp083" / "manifest.json",
    },
    "exp033": {
        "name": "exp033",
        "description": "DAgger 聚合标杆 (Micro 28 通道)",
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

MAPS = {
    "4p": {
        "seat_count": 4,
        "name": "basic-medium-4p",
        "path": ROOT / "maps" / "basic-medium-4p.json",
    },
    "6p": {
        "seat_count": 6,
        "name": "basic-large-6p",
        "path": ROOT / "maps" / "basic-large-6p.json",
    },
    "8p": {
        "seat_count": 8,
        "name": "basic-xlarge-8p",
        "path": ROOT / "maps" / "basic-xlarge-8p.json",
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

def safe_clean_directory(target_dir: Path) -> None:
    resolved = target_dir.resolve()
    root_resolved = ROOT.resolve()
    if not resolved.is_relative_to(root_resolved) or resolved == root_resolved:
        raise RuntimeError(f"目录清理操作被拒绝: 目标路径超出安全边界: {resolved}")
    forbidden_directories = {
        "engine", "baselines", "maps", "models", "tools",
        "benchmarks", "viz", "tinybrain", ".git", ".agents",
    }
    if resolved.parent == root_resolved and resolved.name in forbidden_directories:
        raise RuntimeError(f"目录清理操作被拒绝: 禁止清理重要源码目录: {resolved}")
    if "runs" not in resolved.parts and "replays" not in resolved.parts:
        raise RuntimeError(f"目录清理操作被拒绝: 仅允许清理回放与运行输出目录: {resolved}")
    if resolved.exists():
        shutil.rmtree(resolved)

def build_reproduction_suite_rows(
    maps: str = "all",
    rounds: int = 1,
    seed_offset: int = 0,
) -> dict[str, list[dict[str, Any]]]:
    if maps not in {"4p", "6p", "8p", "all"}:
        raise ValueError(f"不支持的地图规格: {maps}")
    if rounds < 1:
        raise ValueError(f"轮数必须大于等于 1: {rounds}")

    selected_maps = ["4p", "6p", "8p"] if maps == "all" else [maps]
    suite_by_seat_count: dict[str, list[dict[str, Any]]] = {}

    # 1. 4 席位对局 (地图: basic-medium-4p, 48x64)
    if "4p" in selected_maps:
        map_4p_path = str(MAPS["4p"]["path"].resolve())
        base_models_4p = ["exp083", "exp033", "exp021", "exp011"]
        rows_4p: list[dict[str, Any]] = []
        for r in range(rounds):
            seed = 101 + seed_offset + r
            for shift in range(4):
                rows_4p.append({
                    "id": f"repro-4p-s{seed}-shift{shift}",
                    "seed": seed,
                    "map": map_4p_path,
                    "seat_count": 4,
                    "seats": create_cyclic_shift_seats(base_models_4p, shift),
                })
        suite_by_seat_count["4p"] = rows_4p

    # 2. 6 席位对局 (地图: basic-large-6p, 80x96)
    if "6p" in selected_maps:
        map_6p_path = str(MAPS["6p"]["path"].resolve())
        base_models_6p = ["exp083", "exp033", "exp011", "exp021", "exp021", "exp011"]
        rows_6p: list[dict[str, Any]] = []
        for r in range(rounds):
            seed = 201 + seed_offset + r
            for shift in range(6):
                rows_6p.append({
                    "id": f"repro-6p-s{seed}-shift{shift}",
                    "seed": seed,
                    "map": map_6p_path,
                    "seat_count": 6,
                    "seats": create_cyclic_shift_seats(base_models_6p, shift),
                })
        suite_by_seat_count["6p"] = rows_6p

    # 3. 8 席位对局 (地图: basic-xlarge-8p, 120x124)
    if "8p" in selected_maps:
        map_8p_path = str(MAPS["8p"]["path"].resolve())
        base_models_8p = ["exp083", "exp021", "exp033", "exp011", "exp083", "exp021", "exp033", "exp011"]
        rows_8p: list[dict[str, Any]] = []
        for r in range(rounds):
            seeds_8p_round = [301 + seed_offset + r * 2, 302 + seed_offset + r * 2]
            for seed in seeds_8p_round:
                for shift in range(4):
                    rows_8p.append({
                        "id": f"repro-8p-s{seed}-shift{shift}",
                        "seed": seed,
                        "map": map_8p_path,
                        "seat_count": 8,
                        "seats": create_cyclic_shift_seats(base_models_8p, shift),
                    })
        suite_by_seat_count["8p"] = rows_8p

    return suite_by_seat_count

def run_match_batches(
    suite: dict[str, list[dict[str, Any]]],
    replays_dir: Path,
    max_turns: int = 300,
) -> None:
    replays_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["TINYBRAINS_REGISTRY"] = REGISTRY

    for group_key, rows in suite.items():
        batch_file = replays_dir / f"_batch_{group_key}.json"
        batch_file.write_text(json.dumps({
            "game": "ants",
            "vars": {"max_turns": max_turns},
            "rows": rows,
        }, ensure_ascii=False))

        start_time = time.perf_counter()
        print(f"正在推进 {group_key} 批次对局 (场次: {len(rows)} 场)...")
        command = [CLI, str(batch_file), "--out", str(replays_dir)]
        execution_result = subprocess.run(
            command,
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            env=env,
        )
        elapsed_seconds = time.perf_counter() - start_time
        print(f"{group_key} 批次执行完成，耗时: {elapsed_seconds:.2f} 秒")

        batch_file.unlink(missing_ok=True)
        if execution_result.returncode != 0:
            raise RuntimeError(f"批次 {group_key} 对局执行失败:\n{execution_result.stderr or execution_result.stdout}")

def compute_mean_and_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    mean_val = sum(values) / len(values)
    if len(values) < 2:
        return mean_val, 0.0
    variance = sum((x - mean_val) ** 2 for x in values) / (len(values) - 1)
    return mean_val, math.sqrt(variance)

def analyze_replays(replays_dir: Path) -> dict[str, Any]:
    if not EXTRACTOR.exists():
        raise FileNotFoundError(f"未找到度量提取脚本: {EXTRACTOR}")

    node_command = ["node", str(EXTRACTOR.resolve()), str(replays_dir.resolve())]
    node_result = subprocess.run(
        node_command,
        cwd=str(ROOT / "benchmarks"),
        capture_output=True,
        text=True,
    )
    if node_result.returncode != 0:
        raise RuntimeError(f"提取回放度量指标失败:\n{node_result.stderr}")

    match_records: list[dict[str, Any]] = json.loads(node_result.stdout)
    if not match_records:
        raise RuntimeError(f"未在目录中提取到任何回放记录: {replays_dir}")

    # 读取原始回放 JSON 文件以获取 seat_turns 精确计算单步推理毫秒
    raw_seat_turns_map: dict[str, dict[int, int]] = {}
    for replay_file in replays_dir.glob("*.json"):
        if replay_file.name.startswith("_batch_"):
            continue
        with open(replay_file, "r", encoding="utf-8") as f:
            content = json.load(f)
        mid = content.get("match_id")
        if mid and "seats" in content:
            raw_seat_turns_map[mid] = {
                s.get("seat", idx): s.get("seat_turns", content.get("turns", 1))
                for idx, s in enumerate(content["seats"])
            }

    # 统计汇总容器
    model_keys = ["exp083", "exp033", "exp021", "exp011"]
    metrics_by_model: dict[str, dict[str, list[float]]] = {
        m: {
            "scores": [],
            "ranks": [],
            "ants_spawned": [],
            "hills_razed": [],
            "combat_deaths": [],
            "blocked_rates": [],
            "mean_infer_ms": [],
            "max_infer_ms": [],
        }
        for m in model_keys
    }

    # 按地图规模细分统计
    metrics_by_map: dict[str, dict[str, dict[str, list[float]]]] = {
        "basic-medium-4p": {m: {"scores": [], "ranks": [], "ants_spawned": [], "hills_razed": []} for m in model_keys},
        "basic-large-6p": {m: {"scores": [], "ranks": [], "ants_spawned": [], "hills_razed": []} for m in model_keys},
        "basic-xlarge-8p": {m: {"scores": [], "ranks": [], "ants_spawned": [], "hills_razed": []} for m in model_keys},
    }

    total_seats_evaluated = 0
    for match in match_records:
        mid = match.get("match_id", "")
        map_id = match.get("map_id", "")
        turns = match.get("turns", 1)
        seat_turns_dict = raw_seat_turns_map.get(mid, {})

        for seat_info in match.get("seats", []):
            model_name = seat_info.get("name")
            if model_name not in metrics_by_model:
                continue

            total_seats_evaluated += 1
            seat_index = seat_info.get("seat", 0)
            seat_turns = seat_turns_dict.get(seat_index, turns)
            if seat_turns <= 0:
                seat_turns = 1

            score = float(seat_info.get("score", 0.0))
            rank = float(seat_info.get("rank", 1.0))
            spawned = float(seat_info.get("ants_spawned", 0.0))
            razed = float(seat_info.get("hills_razed", 0.0))
            deaths = float(seat_info.get("combat_deaths", 0.0))
            blocked_rate = float(seat_info.get("blocked_rate", 0.0))

            infer_us_total = float(seat_info.get("infer_us_total", 0.0))
            infer_us_max = float(seat_info.get("infer_us_max", 0.0))
            mean_infer_ms = (infer_us_total / 1000.0) / float(seat_turns)
            max_infer_ms = infer_us_max / 1000.0

            m_dict = metrics_by_model[model_name]
            m_dict["scores"].append(score)
            m_dict["ranks"].append(rank)
            m_dict["ants_spawned"].append(spawned)
            m_dict["hills_razed"].append(razed)
            m_dict["combat_deaths"].append(deaths)
            m_dict["blocked_rates"].append(blocked_rate)
            m_dict["mean_infer_ms"].append(mean_infer_ms)
            m_dict["max_infer_ms"].append(max_infer_ms)

            if map_id in metrics_by_map and model_name in metrics_by_map[map_id]:
                map_m_dict = metrics_by_map[map_id][model_name]
                map_m_dict["scores"].append(score)
                map_m_dict["ranks"].append(rank)
                map_m_dict["ants_spawned"].append(spawned)
                map_m_dict["hills_razed"].append(razed)

    # 计算聚合均值与标准差
    summary_by_model: dict[str, dict[str, Any]] = {}
    for m in model_keys:
        d = metrics_by_model[m]
        seats_count = len(d["scores"])
        score_mean, score_std = compute_mean_and_std(d["scores"])
        rank_mean, rank_std = compute_mean_and_std(d["ranks"])
        spawn_mean, spawn_std = compute_mean_and_std(d["ants_spawned"])
        deaths_mean, deaths_std = compute_mean_and_std(d["combat_deaths"])
        blocked_mean, _ = compute_mean_and_std(d["blocked_rates"])
        infer_mean, _ = compute_mean_and_std(d["mean_infer_ms"])
        infer_max = max(d["max_infer_ms"]) if d["max_infer_ms"] else 0.0
        total_razed = int(sum(d["hills_razed"]))

        summary_by_model[m] = {
            "model_name": m,
            "description": MODELS[m]["description"],
            "seats_count": seats_count,
            "score_mean": round(score_mean, 2),
            "score_std": round(score_std, 2),
            "rank_mean": round(rank_mean, 2),
            "rank_std": round(rank_std, 2),
            "ants_spawned_mean": round(spawn_mean, 1),
            "ants_spawned_std": round(spawn_std, 1),
            "hills_razed_total": total_razed,
            "combat_deaths_mean": round(deaths_mean, 1),
            "combat_deaths_std": round(deaths_std, 1),
            "blocked_rate_mean_percent": round(blocked_mean * 100.0, 3),
            "mean_infer_latency_ms": round(infer_mean, 2),
            "max_infer_latency_ms": round(infer_max, 2),
        }

    # 地图细分汇总
    summary_by_map: dict[str, dict[str, Any]] = {}
    for map_name, map_data in metrics_by_map.items():
        summary_by_map[map_name] = {}
        for m in model_keys:
            md = map_data[m]
            s_mean, _ = compute_mean_and_std(md["scores"])
            r_mean, _ = compute_mean_and_std(md["ranks"])
            sp_mean, _ = compute_mean_and_std(md["ants_spawned"])
            rz_total = int(sum(md["hills_razed"]))
            summary_by_map[map_name][m] = {
                "seats_count": len(md["scores"]),
                "score_mean": round(s_mean, 2),
                "rank_mean": round(r_mean, 2),
                "ants_spawned_mean": round(sp_mean, 1),
                "hills_razed_total": rz_total,
            }

    return {
        "suite_total_matches": len(match_records),
        "suite_total_seats": total_seats_evaluated,
        "models_summary": summary_by_model,
        "maps_summary": summary_by_map,
        "raw_matches": match_records,
    }

def print_terminal_comparison_table(summary: dict[str, Any]) -> None:
    models_summary = summary["models_summary"]
    maps_summary = summary["maps_summary"]

    print("\n" + "=" * 110)
    print("EXP-083 多席位多样化对抗复现实验评测结果汇总")
    print("=" * 110)
    header = f"{'模型名称':<10} {'席位数':<8} {'平均得分':<12} {'平均排名':<12} {'平均繁衍':<14} {'摧毁敌穴':<10} {'平均阵亡':<12} {'受阻率(%)':<12} {'单步均耗(ms)':<14}"
    print(header)
    print("-" * 110)
    for model_key in ["exp083", "exp033", "exp021", "exp011"]:
        info = models_summary[model_key]
        score_str = f"{info['score_mean']:.2f}±{info['score_std']:.2f}"
        rank_str = f"{info['rank_mean']:.2f}±{info['rank_std']:.2f}"
        spawn_str = f"{info['ants_spawned_mean']:.1f}±{info['ants_spawned_std']:.1f}"
        deaths_str = f"{info['combat_deaths_mean']:.1f}±{info['combat_deaths_std']:.1f}"
        blocked_str = f"{info['blocked_rate_mean_percent']:.3f}%"
        latency_str = f"{info['mean_infer_latency_ms']:.2f}"

        row_str = (
            f"{info['model_name']:<10} "
            f"{info['seats_count']:<8} "
            f"{score_str:<12} "
            f"{rank_str:<12} "
            f"{spawn_str:<14} "
            f"{info['hills_razed_total']:<10} "
            f"{deaths_str:<12} "
            f"{blocked_str:<12} "
            f"{latency_str:<14}"
        )
        print(row_str)
    print("-" * 110)

    print("\n按地图规模细分对战表现:")
    for map_key, map_desc in [
        ("basic-medium-4p", "4 席位编织迷宫 basic-medium-4p (48x64)"),
        ("basic-large-6p", "6 席位狭窄走廊 basic-large-6p (80x96)"),
        ("basic-xlarge-8p", "8 席位超大洞穴 basic-xlarge-8p (120x124)"),
    ]:
        if map_key not in maps_summary:
            continue
        has_seats = any(maps_summary[map_key][m]["seats_count"] > 0 for m in ["exp083", "exp033", "exp021", "exp011"])
        if not has_seats:
            continue
        print(f"\n[{map_desc}]")
        sub_header = f"{'模型名称':<10} {'参战席位':<10} {'平均得分':<12} {'平均排名':<12} {'平均繁衍':<14} {'摧毁敌穴':<10}"
        print(sub_header)
        for model_key in ["exp083", "exp033", "exp021", "exp011"]:
            minfo = maps_summary[map_key][model_key]
            sub_row = (
                f"{model_key:<10} "
                f"{minfo['seats_count']:<10} "
                f"{minfo['score_mean']:<12.2f} "
                f"{minfo['rank_mean']:<12.2f} "
                f"{minfo['ants_spawned_mean']:<14.1f} "
                f"{minfo['hills_razed_total']:<10}"
            )
            print(sub_row)
    print("=" * 110 + "\n")

def main() -> None:
    parser = argparse.ArgumentParser(description="EXP-083 多席位多样化对手对抗复现评测工具")
    parser.add_argument("--maps", type=str, default="all", choices=["4p", "6p", "8p", "all"], help="指定待评估的地图规模 (4p, 6p, 8p 或 all)")
    parser.add_argument("--rounds", type=int, default=1, help="随机种子循环轮数")
    parser.add_argument("--out-dir", type=str, default="runs/reproduce_exp083_gap/replays", help="回放文件存放目录")
    parser.add_argument("--json", type=str, default="runs/reproduce_exp083_gap/summary.json", help="结构化结果输出 JSON 路径")
    parser.add_argument("--turns", type=int, default=300, help="每场对局最大回合数限制")
    parser.add_argument("--seed-offset", type=int, default=0, help="随机种子偏移量")
    parser.add_argument("--skip-run", action="store_true", help="跳过对局仿真推演，直接分析已有回放文件")

    args = parser.parse_args()

    if args.turns < 1:
        parser.error(f"--turns 必须大于等于 1，当前输入参数为: {args.turns}")
    if args.rounds < 1:
        parser.error(f"--rounds 必须大于等于 1，当前输入参数为: {args.rounds}")
    if args.maps not in {"4p", "6p", "8p", "all"}:
        parser.error(f"--maps 参数不合法，仅支持 4p, 6p, 8p 或 all，当前输入参数为: {args.maps}")

    replays_dir = (ROOT / args.out_dir).resolve()
    json_path = (ROOT / args.json).resolve()

    suite = build_reproduction_suite_rows(
        maps=args.maps,
        rounds=args.rounds,
        seed_offset=args.seed_offset,
    )

    if not args.skip_run:
        safe_clean_directory(replays_dir)
        replays_dir.mkdir(parents=True, exist_ok=True)
        total_matches = sum(len(rows) for rows in suite.values())
        print(f"初始化实验套件: 包含 {len(suite)} 个地图组别，累计 {total_matches} 场多方混战")
        run_match_batches(suite, replays_dir=replays_dir, max_turns=args.turns)

    summary_results = analyze_replays(replays_dir=replays_dir)

    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary_results, f, ensure_ascii=False, indent=2)
    print(f"结构化实验汇总数据已保存至: {json_path}")

    print_terminal_comparison_table(summary_results)

if __name__ == "__main__":
    main()
