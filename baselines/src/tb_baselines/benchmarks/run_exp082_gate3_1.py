from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any

import numpy as np

from tb_baselines import planes
from tb_baselines.coordination import (
    DELTAS,
    FAR,
    HOLD,
    GateStatus,
    JointCondition,
    canonicalize_receptive_field_and_action,
    compute_empirical_conditional_entropy,
    evaluate_gate3_1,
    extract_batch_receptive_field_slices,
    generate_flow_projection_plane,
    make_joint_condition,
)
from tb_baselines.mcmf_teacher import MCMFTeacher


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="EXP-082 Gate 3-1 局部可学习性门禁与特征投影补偿自动化评估"
    )
    parser.add_argument(
        "--maps",
        type=str,
        default="benchmarks/maps_2p",
        help="基准地图目录或具体地图文件路径",
    )
    parser.add_argument(
        "--scales",
        type=str,
        default="20,40,60,80,100",
        help="逗号分隔的存活蚂蚁规模序列",
    )
    parser.add_argument(
        "--turns",
        type=int,
        default=100,
        help="每个对局用例的模拟推进回合数",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default="42,43,44",
        help="逗号分隔的随机数种子序列",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.50,
        help="Gate 3-1 经验条件熵判定阈值（单位：bit）",
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=10,
        help="长尾过滤的单类别最小支持样本数门限",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="runs/exp082_gate3_1/",
        help="遥测记录与报告输出目录路径",
    )
    return parser.parse_args(args)


def rle_from_mask(mask: np.ndarray) -> list[int]:
    flat = mask.ravel().astype(np.int8)
    if len(flat) == 0:
        return []
    diffs = np.diff(flat)
    change_indices = np.where(diffs != 0)[0] + 1
    split_points = [0] + change_indices.tolist() + [len(flat)]
    rle = []
    for start, end in zip(split_points[:-1], split_points[1:]):
        rle.extend([int(flat[start]), int(end - start)])
    return rle


def run_single_simulation_case(
    board_data: dict,
    scale: int,
    turns: int,
    seed: int,
    min_samples: int = 10,
) -> dict[str, Any]:
    rows = int(board_data["rows"])
    cols = int(board_data["cols"])
    b = planes.Board(rows, cols)
    water = b.rle(board_data["water"]).astype(bool)

    land_cells = [
        (r, c) for r in range(rows) for c in range(cols) if not water[r, c]
    ]
    if len(land_cells) < scale:
        raise ValueError(f"地图合法陆地数量 ({len(land_cells)}) 小于蚂蚁规模 {scale}")

    rng = np.random.default_rng(seed)
    chosen_indices = rng.choice(len(land_cells), size=scale, replace=False)
    ants = [list(land_cells[idx]) for idx in chosen_indices]
    ant_ids = list(range(scale))

    hills_raw = board_data.get("hills", [])
    hills = [[int(h[0]), int(h[1]), idx] for idx, h in enumerate(hills_raw)]

    food_target = max(scale // 2, 16)
    ant_set = {tuple(a) for a in ants}
    initial_foods = [
        [int(f[0]), int(f[1])]
        for f in board_data.get("food", [])
        if not water[int(f[0]), int(f[1])] and (int(f[0]), int(f[1])) not in ant_set
    ]
    current_food = list(initial_foods)

    teacher = MCMFTeacher(
        max_depth=16,
        dynamic_congestion=True,
        congestion_alpha=10.0,
        congestion_beta=1.5,
    )
    memory: dict = {}

    all_actions: list[int] = []
    all_raw_conditions: list[JointCondition] = []
    all_turn_contexts: list[dict] = []

    t_start = time.perf_counter()

    for t in range(turns):
        ant_positions = [(int(a[0]), int(a[1])) for a in ants]
        ant_positions_set = set(ant_positions)

        # 补给食物点位
        if len(current_food) < food_target:
            occupied = ant_positions_set | {tuple(f) for f in current_food}
            free_land = [p for p in land_cells if p not in occupied]
            if free_land:
                need_count = min(food_target - len(current_food), len(free_land))
                chosen_f_idx = rng.choice(len(free_land), size=need_count, replace=False)
                for f_idx in chosen_f_idx:
                    current_food.append([free_land[f_idx][0], free_land[f_idx][1]])

        mine_plane = b.scatter(ants)
        vis_plane = b.dilate(mine_plane, planes.VIEW_RADIUS2).astype(bool)
        seen = water | vis_plane

        obs = {
            "size": [rows, cols],
            "mine": ants,
            "foes": [],
            "food": [list(f) for f in current_food],
            "water": {"rle": board_data["water"] if "water" in board_data else rle_from_mask(water)},
            "vis": {"rle": rle_from_mask(vis_plane)},
            "hills": hills,
            "ids": ant_ids,
        }

        memory = planes.remember(memory, obs, mode="where")

        # 生成 7 特征观测平面
        obs_tensor = planes.encode(obs)  # (1, 7, rows, cols)

        # 教师执行决策推演
        orders = teacher.orders(obs, water, seen, memory)
        tasks = teacher.get_last_tasks()
        winning_targets, winning_fields = teacher.get_last_winning_flow()

        # 提取 31x31 感受野切片并执行 D4 规范化
        slices = extract_batch_receptive_field_slices(obs_tensor, ant_positions, radius=15)

        turn_ant_samples = []
        task_role_map = {
            "assault": 0,
            "gather": 1,
            "explore": 2,
            "fortify": 3,
        }
        for i in range(len(ants)):
            act = int(orders[i]) if i < len(orders) else HOLD
            rf_slice = slices[i]
            min_hash, best_action = canonicalize_receptive_field_and_action(rf_slice, act)
            task_item = tasks.get(i)
            task_str = task_item.value if hasattr(task_item, "value") else str(task_item)
            role = task_role_map.get(task_str, 0)
            mem_state = 0
            cond = make_joint_condition(min_hash, role, mem_state)

            all_actions.append(best_action)
            all_raw_conditions.append(cond)
            turn_ant_samples.append({
                "ant_idx": i,
                "pos": ant_positions[i],
                "base_hash": min_hash,
                "role": role,
                "action": best_action,
            })

        all_turn_contexts.append({
            "turn": t,
            "ant_samples": turn_ant_samples,
            "winning_targets": winning_targets,
            "winning_fields": winning_fields,
            "rows": rows,
            "cols": cols,
        })

        # 推进物理世界移动与采食
        new_ants = []
        for i, a in enumerate(ants):
            act = int(orders[i]) if i < len(orders) else HOLD
            if act < 4:
                dr, dc = DELTAS[act]
                nr = (a[0] + dr) % rows
                nc = (a[1] + dc) % cols
                if not water[nr, nc]:
                    new_ants.append([nr, nc])
                else:
                    new_ants.append(a)
            else:
                new_ants.append(a)
        ants = new_ants

        # 食物被采食判定
        remaining_food = []
        new_ant_set = {tuple(a) for a in ants}
        for f in current_food:
            if tuple(f) not in new_ant_set:
                remaining_food.append(f)
        current_food = remaining_food

    elapsed_total_ms = (time.perf_counter() - t_start) * 1000.0

    return {
        "scale": scale,
        "turns": turns,
        "seed": seed,
        "total_time_ms": elapsed_total_ms,
        "avg_turn_latency_ms": elapsed_total_ms / float(turns),
        "actions": all_actions,
        "raw_conditions": all_raw_conditions,
        "turn_contexts": all_turn_contexts,
    }


def execute_evaluation_pipeline(
    map_files: list[Path],
    scales: list[int],
    seeds: list[int],
    turns: int,
    threshold: float,
    min_samples: int,
    out_dir: Path,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / "telemetry.jsonl"
    summary_path = out_dir / "telemetry_summary.json"
    report_path = out_dir / "gate3_1_report.md"

    all_case_records = []
    global_actions: list[int] = []
    global_raw_conditions: list[JointCondition] = []
    global_comp_conditions: list[JointCondition] = []

    with open(jsonl_path, "w", encoding="utf-8") as f_jsonl:
        for map_file in map_files:
            with open(map_file, "r", encoding="utf-8") as f_map:
                board_data = json.load(f_map)
            map_name = board_data.get("id", map_file.stem)

            for scale in scales:
                for seed in seeds:
                    case_res = run_single_simulation_case(
                        board_data=board_data,
                        scale=scale,
                        turns=turns,
                        seed=seed,
                        min_samples=min_samples,
                    )

                    case_actions = case_res["actions"]
                    case_raw_conds = case_res["raw_conditions"]
                    turn_contexts = case_res["turn_contexts"]

                    # 尝试计算单用例局部条件熵
                    case_entropy = 0.0
                    case_coverage = 0.0
                    valid_categories = 0
                    if len(case_actions) > 0:
                        category_counter: dict[JointCondition, int] = {}
                        for c_item in case_raw_conds:
                            category_counter[c_item] = category_counter.get(c_item, 0) + 1
                        has_valid = any(cnt >= min_samples for cnt in category_counter.values())
                        if has_valid:
                            case_entropy, case_meta = compute_empirical_conditional_entropy(
                                case_actions, case_raw_conds, min_category_samples=min_samples
                            )
                            case_coverage = case_meta["coverage_ratio"]
                            valid_categories = case_meta["valid_categories"]

                    global_actions.extend(case_actions)
                    global_raw_conditions.extend(case_raw_conds)

                    # 构建补偿条件序列
                    case_comp_conds: list[JointCondition] = []
                    for ctx in turn_contexts:
                        w_targets = ctx["winning_targets"]
                        w_fields = ctx["winning_fields"]
                        c_rows = ctx["rows"]
                        c_cols = ctx["cols"]
                        if w_targets and w_fields:
                            proj_plane = generate_flow_projection_plane(w_targets, w_fields, c_rows, c_cols)
                        else:
                            proj_plane = np.zeros((c_rows, c_cols), dtype=np.float32)

                        for samp in ctx["ant_samples"]:
                            ar, ac = samp["pos"]
                            # 计算局部势能反向梯度符号
                            gn = proj_plane[(ar - 1) % c_rows, ac]
                            gs = proj_plane[(ar + 1) % c_rows, ac]
                            gw = proj_plane[ar, (ac - 1) % c_cols]
                            ge = proj_plane[ar, (ac + 1) % c_cols]

                            # 4 个正交方向导数离散符号压缩为 4 比特编码
                            code_n = 1 if gn > proj_plane[ar, ac] else 0
                            code_s = 1 if gs > proj_plane[ar, ac] else 0
                            code_w = 1 if gw > proj_plane[ar, ac] else 0
                            code_e = 1 if ge > proj_plane[ar, ac] else 0
                            flow_code = (code_n) | (code_s << 1) | (code_w << 2) | (code_e << 3)

                            # 注入联合特征哈希
                            comp_hash = (samp["base_hash"] << 4) | flow_code
                            case_comp_conds.append(make_joint_condition(comp_hash, samp["role"], 0))

                    global_comp_conditions.extend(case_comp_conds)

                    record = {
                        "map": map_name,
                        "scale": scale,
                        "seed": seed,
                        "turns": turns,
                        "total_samples": len(case_actions),
                        "valid_categories": valid_categories,
                        "empirical_entropy": float(case_entropy),
                        "coverage_ratio": float(case_coverage),
                        "avg_turn_latency_ms": float(case_res["avg_turn_latency_ms"]),
                    }
                    all_case_records.append(record)
                    f_jsonl.write(json.dumps(record, ensure_ascii=False) + "\n")
                    f_jsonl.flush()

    # 全局经验条件熵评估
    raw_entropy, raw_meta = compute_empirical_conditional_entropy(
        global_actions, global_raw_conditions, min_category_samples=min_samples
    )
    raw_status, raw_verdict = evaluate_gate3_1(
        raw_entropy, threshold=threshold, min_coverage_ratio=0.70, metadata=raw_meta
    )

    # 特征投影补偿条件熵评估
    comp_entropy, comp_meta = compute_empirical_conditional_entropy(
        global_actions, global_comp_conditions, min_category_samples=min_samples
    )
    comp_status, comp_verdict = evaluate_gate3_1(
        comp_entropy, threshold=threshold, min_coverage_ratio=0.70, metadata=comp_meta
    )

    summary = {
        "gate_id": "Gate 3-1",
        "threshold": threshold,
        "min_category_samples": min_samples,
        "total_aggregated_samples": len(global_actions),
        "raw_evaluation": {
            "status": raw_status.value,
            "empirical_conditional_entropy": raw_entropy,
            "coverage_ratio": raw_meta["coverage_ratio"],
            "valid_categories": raw_meta["valid_categories"],
            "total_categories": raw_meta["total_categories"],
            "verdict": raw_verdict,
        },
        "compensated_evaluation": {
            "status": comp_status.value,
            "empirical_conditional_entropy": comp_entropy,
            "coverage_ratio": comp_meta["coverage_ratio"],
            "valid_categories": comp_meta["valid_categories"],
            "total_categories": comp_meta["total_categories"],
            "verdict": comp_verdict,
        },
        "entropy_reduction": float(raw_entropy - comp_entropy),
        "case_records": all_case_records,
    }

    with open(summary_path, "w", encoding="utf-8") as f_sum:
        json.dump(summary, f_sum, indent=2, ensure_ascii=False)

    # 输出学术报告
    with open(report_path, "w", encoding="utf-8") as f_rep:
        f_rep.write("# EXP-082 Gate 3-1 局部可学习性门禁与特征投影补偿评估报告\n\n")
        f_rep.write("## 1. 核心裁决结论\n\n")
        f_rep.write(f"- **门禁标识**: Gate 3-1 (局部可学习性门禁)\n")
        f_rep.write(f"- **判定阈值**: {threshold:.2f} bit\n")
        f_rep.write(f"- **原始局部观测条件熵 $H(A \\mid C)$**: {raw_entropy:.4f} bit\n")
        f_rep.write(f"- **原始判定分支**: {raw_status.value}\n")
        f_rep.write(f"- **特征投影补偿条件熵 $H(A \\mid C')$**: {comp_entropy:.4f} bit\n")
        f_rep.write(f"- **补偿判定分支**: {comp_status.value}\n")
        f_rep.write(f"- **条件熵净消减量**: {raw_entropy - comp_entropy:.4f} bit\n\n")

        f_rep.write("## 2. 统计样本分布与覆盖度量\n\n")
        f_rep.write(f"- **总采样规模**: {len(global_actions):,} 条单体动作记录\n")
        f_rep.write(f"- **原始离散条件类别总数**: {raw_meta['total_categories']:,}\n")
        f_rep.write(f"- **有效支持类别数 ($N_c \\ge {min_samples}$)**: {raw_meta['valid_categories']:,}\n")
        f_rep.write(f"- **有效样本覆盖率**: {raw_meta['coverage_ratio'] * 100.0:.2f}%\n\n")

        f_rep.write("## 3. 分规模分用例遥测矩阵\n\n")
        f_rep.write("| 地图名称 | 规模 | 回合 | 样本数 | 有效类别 | 经验条件熵 (bit) | 覆盖率 (%) | 单回合耗时 (ms) |\n")
        f_rep.write("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n")
        for rec in all_case_records:
            f_rep.write(
                f"| {rec['map']} | {rec['scale']} | {rec['turns']} | {rec['total_samples']:,} | "
                f"{rec['valid_categories']} | {rec['empirical_entropy']:.4f} | "
                f"{rec['coverage_ratio'] * 100.0:.1f}% | {rec['avg_turn_latency_ms']:.2f} |\n"
            )

    return summary


def main(args: list[str] | None = None) -> None:
    parsed = parse_args(args)
    scales = [int(s.strip()) for s in parsed.scales.split(",") if s.strip()]
    seeds = [int(s.strip()) for s in parsed.seeds.split(",") if s.strip()]
    turns = int(parsed.turns)
    threshold = float(parsed.threshold)
    min_samples = int(parsed.min_samples)
    out_dir = Path(parsed.out)

    maps_path = Path(parsed.maps)
    if maps_path.is_file():
        map_files = [maps_path]
    elif maps_path.is_dir():
        map_files = sorted(list(maps_path.glob("*.json")))
        if not map_files:
            raise FileNotFoundError(f"目录 {maps_path} 中未发现任何 JSON 地图文件")
    else:
        raise FileNotFoundError(f"指定的地图路径不存在: {maps_path}")

    summary = execute_evaluation_pipeline(
        map_files=map_files,
        scales=scales,
        seeds=seeds,
        turns=turns,
        threshold=threshold,
        min_samples=min_samples,
        out_dir=out_dir,
    )
    print("\n=== EXP-082 Gate 3-1 评估执行完毕 ===")
    print(f"原始条件熵: {summary['raw_evaluation']['empirical_conditional_entropy']:.4f} bit -> 裁决: {summary['raw_evaluation']['status']}")
    print(f"补偿条件熵: {summary['compensated_evaluation']['empirical_conditional_entropy']:.4f} bit -> 裁决: {summary['compensated_evaluation']['status']}")
    print(f"报告输出路径: {out_dir / 'gate3_1_report.md'}")


if __name__ == "__main__":
    main()
