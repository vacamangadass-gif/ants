from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

from tb_baselines import planes
from tb_baselines.teacher import CompoundTeacher, DELTAS, HOLD, Teacher
from tb_baselines.telemetry import HoldReason


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


def compute_corridor_mask(water: np.ndarray, max_corridor_width: int = 2) -> np.ndarray:
    if water.ndim != 2:
        raise ValueError(f"水域网格维度必须为 2，当前维度为 {water.ndim}")
    if max_corridor_width < 1:
        raise ValueError(f"最大走廊宽度必须大于等于 1，当前数值为 {max_corridor_width}")

    rows, cols = water.shape
    corridor_mask = np.zeros((rows, cols), dtype=bool)

    for r in range(rows):
        for c in range(cols):
            if water[r, c]:
                continue

            # 沿水平方向检索连续陆地跨度，环面边界取模循环
            if not water[r].any():
                span_h = cols
            else:
                steps_east = 0
                while steps_east < cols:
                    nc = (c + 1 + steps_east) % cols
                    if water[r, nc]:
                        break
                    steps_east += 1

                steps_west = 0
                while steps_west < cols:
                    nc = (c - 1 - steps_west) % cols
                    if water[r, nc]:
                        break
                    steps_west += 1

                span_h = 1 + steps_east + steps_west

            # 沿垂直方向检索连续陆地跨度，环面边界取模循环
            if not water[:, c].any():
                span_v = rows
            else:
                steps_south = 0
                while steps_south < rows:
                    nr = (r + 1 + steps_south) % rows
                    if water[nr, c]:
                        break
                    steps_south += 1

                steps_north = 0
                while steps_north < rows:
                    nr = (r - 1 - steps_north) % rows
                    if water[nr, c]:
                        break
                    steps_north += 1

                span_v = 1 + steps_south + steps_north

            width = min(span_h, span_v)
            if width <= max_corridor_width:
                corridor_mask[r, c] = True

    return corridor_mask


def compute_corridor_congestion_index(
    stalled_turns_in_corridor: int, turns: int, corridor_cell_count: int
) -> float:
    if turns <= 0 or corridor_cell_count <= 0:
        return 0.0
    denominator = float(turns * corridor_cell_count)
    return float(stalled_turns_in_corridor) / denominator


def compute_corridor_stalling_rate(
    stalled_turns_in_corridor: int, total_corridor_ant_turns: int
) -> float:
    if total_corridor_ant_turns <= 0:
        return 0.0
    return float(stalled_turns_in_corridor) / float(total_corridor_ant_turns)


def compute_effective_gathering_throughput(
    total_food_gathered: int, turns: int, average_alive_ants: float
) -> float:
    if turns <= 0 or average_alive_ants <= 0.0:
        return 0.0
    numerator = float(total_food_gathered * 100)
    denominator = float(turns * average_alive_ants)
    return numerator / denominator


def compute_collision_blocking_rate(
    total_forced_holds: int, total_ant_turns: int
) -> float:
    if total_ant_turns <= 0:
        return 0.0
    return float(total_forced_holds) / float(total_ant_turns)


def run_scaling_simulation(
    board_data: dict,
    scale: int,
    turns: int,
    seed: int,
    teacher_type: str = "compound",
    max_corridor_width: int = 2,
) -> dict:
    if scale <= 0:
        raise ValueError(f"蚂蚁规模必须大于 0，当前输入为 {scale}")
    if turns <= 0:
        raise ValueError(f"模拟推进回合数必须大于 0，当前输入为 {turns}")

    rows = int(board_data["rows"])
    cols = int(board_data["cols"])
    b = planes.Board(rows, cols)
    water = b.rle(board_data["water"]).astype(bool)

    corridor_mask = compute_corridor_mask(water, max_corridor_width=max_corridor_width)
    corridor_cell_count = int(corridor_mask.sum())

    land_cells = [
        (r, c) for r in range(rows) for c in range(cols) if not water[r, c]
    ]
    if len(land_cells) < scale:
        raise ValueError(
            f"地图合法陆地数量不足: 陆地单元格数量 {len(land_cells)} 小于蚂蚁规模 {scale}"
        )

    rng = np.random.default_rng(seed)
    chosen_indices = rng.choice(len(land_cells), size=scale, replace=False)
    ants = [list(land_cells[idx]) for idx in chosen_indices]
    ant_ids = list(range(scale))

    hills_raw = board_data.get("hills", [])
    hills = [[int(h[0]), int(h[1]), idx] for idx, h in enumerate(hills_raw)]

    food_target = max(scale // 2, 16)
    ant_set = {tuple(a) for a in ants}
    initial_food_candidates = [
        [int(f[0]), int(f[1])]
        for f in board_data.get("food", [])
        if not water[int(f[0]), int(f[1])] and (int(f[0]), int(f[1])) not in ant_set
    ]

    current_food = list(initial_food_candidates)
    if len(current_food) < food_target:
        occupied = ant_set | {tuple(f) for f in current_food}
        free_land = [p for p in land_cells if p not in occupied]
        if free_land:
            need_count = min(food_target - len(current_food), len(free_land))
            chosen_food_idx = rng.choice(len(free_land), size=need_count, replace=False)
            for f_idx in chosen_food_idx:
                current_food.append([free_land[f_idx][0], free_land[f_idx][1]])

    if teacher_type == "compound":
        teacher = CompoundTeacher(min_raid_ants=4, raid_ratio=0.5)
    elif teacher_type == "baseline":
        teacher = Teacher(hill_lead=6)
    else:
        raise ValueError(f"未知教师模型类型: {teacher_type}")

    memory: dict = {}
    stall_counters: dict[int, int] = {}

    total_food_gathered = 0
    stalled_corridor_turns = 0
    total_corridor_ant_turns = 0
    total_forced_holds = 0
    latencies_ms: list[float] = []

    for t in range(turns):
        mine_plane = b.scatter(ants)
        vis_plane = b.dilate(mine_plane, planes.VIEW_RADIUS2).astype(bool)
        seen = water | vis_plane

        obs = {
            "size": [rows, cols],
            "mine": ants,
            "ids": ant_ids,
            "foes": [],
            "food": [list(f) for f in current_food],
            "hills": hills,
            "water": {"rle": rle_from_mask(water)},
            "vis": {"rle": rle_from_mask(vis_plane)},
        }

        memory = planes.remember(memory, obs, mode="where")

        t_start = time.perf_counter()
        if teacher_type == "compound":
            orders, telemetry_records = teacher.orders_with_telemetry(
                obs, water, seen, memory, stall_counters=stall_counters
            )
        else:
            orders = teacher.orders(obs, water, seen)
            telemetry_records = []
        turn_latency = (time.perf_counter() - t_start) * 1000.0
        latencies_ms.append(turn_latency)

        turn_corridor_ants = 0
        turn_corridor_stalls = 0
        turn_forced_holds = 0

        for i in range(scale):
            ar, ac = ants[i]
            act = int(orders[i])
            in_corridor = bool(corridor_mask[ar, ac])

            if in_corridor:
                turn_corridor_ants += 1
                if act == HOLD:
                    turn_corridor_stalls += 1

            if telemetry_records:
                rec = telemetry_records[i]
                if act == HOLD and rec.hold_reason == HoldReason.CONFLICT_WAIT.value:
                    turn_forced_holds += 1
            else:
                if act == HOLD:
                    turn_forced_holds += 1

        total_corridor_ant_turns += turn_corridor_ants
        stalled_corridor_turns += turn_corridor_stalls
        total_forced_holds += turn_forced_holds

        dests: list[tuple[int, int]] = []
        for (ar, ac), a in zip(ants, orders):
            act_int = int(a)
            if act_int == HOLD:
                dests.append((ar, ac))
            else:
                dr, dc = DELTAS[act_int]
                dests.append(((ar + dr) % rows, (ac + dc) % cols))

        if len(set(dests)) != len(dests):
            raise AssertionError(f"回合 {t} 检测到友方蚂蚁内部碰撞: {dests}")

        food_set = {tuple(f) for f in current_food}
        for dr_c, dc_c in dests:
            if water[dr_c, dc_c]:
                raise AssertionError(f"回合 {t} 蚂蚁走入水域: ({dr_c}, {dc_c})")
            if (dr_c, dc_c) in food_set:
                raise AssertionError(f"回合 {t} 蚂蚁走入食物方格: ({dr_c}, {dc_c})")

        ants = [list(p) for p in dests]

        remaining_food: list[list[int]] = []
        gathered_this_turn = 0
        for fr, fc in current_food:
            gathered = False
            for ar, ac in ants:
                dr = min(abs(ar - fr), rows - abs(ar - fr))
                dc = min(abs(ac - fc), cols - abs(ac - fc))
                if dr * dr + dc * dc <= 1:
                    gathered = True
                    break
            if gathered:
                gathered_this_turn += 1
            else:
                remaining_food.append([fr, fc])

        total_food_gathered += gathered_this_turn
        current_food = remaining_food

        if len(current_food) < food_target:
            needed = food_target - len(current_food)
            occupied = {tuple(a) for a in ants} | {tuple(f) for f in current_food}
            free_land = [p for p in land_cells if p not in occupied]
            if free_land:
                rep_count = min(needed, len(free_land))
                chosen_rep_idx = rng.choice(len(free_land), size=rep_count, replace=False)
                for r_idx in chosen_rep_idx:
                    current_food.append([free_land[r_idx][0], free_land[r_idx][1]])

    total_ant_turns = scale * turns
    congestion_index = compute_corridor_congestion_index(
        stalled_corridor_turns, turns, corridor_cell_count
    )
    corridor_stalling_rate = compute_corridor_stalling_rate(
        stalled_corridor_turns, total_corridor_ant_turns
    )
    gathering_throughput = compute_effective_gathering_throughput(
        total_food_gathered, turns, float(scale)
    )
    collision_blocking_rate = compute_collision_blocking_rate(
        total_forced_holds, total_ant_turns
    )

    lat_arr = np.array(latencies_ms, dtype=np.float64)

    assert 0.0 <= corridor_stalling_rate <= 1.0, f"走廊停滞比率越界: {corridor_stalling_rate}"
    assert congestion_index >= 0.0, "走廊单位面积拥塞指数必须非负"
    assert gathering_throughput >= 0.0, "有效采食速率必须非负"
    assert np.isfinite(gathering_throughput), "有效采食速率必须为有限数值"
    assert 0.0 <= collision_blocking_rate <= 1.0, f"强制驻留阻挡率越界: {collision_blocking_rate}"
    assert (
        float(np.min(lat_arr))
        <= float(np.percentile(lat_arr, 50))
        <= float(np.percentile(lat_arr, 95))
        <= float(np.max(lat_arr))
    ), "计算延迟分位数单调性违规"

    mean_lat = float(np.mean(lat_arr))
    p50_lat = float(np.percentile(lat_arr, 50))
    p95_lat = float(np.percentile(lat_arr, 95))
    max_lat = float(np.max(lat_arr))
    min_lat = float(np.min(lat_arr))
    std_lat = float(np.std(lat_arr))

    return {
        "scale": scale,
        "turns": turns,
        "seed": seed,
        "teacher_type": teacher_type,
        "grid_shape": [rows, cols],
        "total_land_cells": len(land_cells),
        "corridor_cell_count": corridor_cell_count,
        "total_corridor_ant_turns": total_corridor_ant_turns,
        "stalled_corridor_turns": stalled_corridor_turns,
        "corridor_congestion_index": congestion_index,
        "corridor_stalling_rate": corridor_stalling_rate,
        "total_food_gathered": total_food_gathered,
        "effective_gathering_throughput": gathering_throughput,
        "total_ant_turns": total_ant_turns,
        "total_forced_holds": total_forced_holds,
        "collision_blocking_rate": collision_blocking_rate,
        "internal_collision_blocking_rate": collision_blocking_rate,
        "mean_latency_ms": mean_lat,
        "p50_latency_ms": p50_lat,
        "p95_latency_ms": p95_lat,
        "max_latency_ms": max_lat,
        "min_latency_ms": min_lat,
        "std_latency_ms": std_lat,
        "total_latency_ms": float(np.sum(lat_arr)),
        "latency_ms": {
            "mean": mean_lat,
            "p50": p50_lat,
            "p95": p95_lat,
            "max": max_lat,
            "min": min_lat,
            "std": std_lat,
        },
    }


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="EXP-080 教师规模压力自动化评测基准运行程序"
    )
    parser.add_argument(
        "--scales",
        type=str,
        default="20,40,60,80,100",
        help="逗号分隔的蚂蚁规模序列，例如 20,40,60,80,100",
    )
    parser.add_argument(
        "--maps",
        type=str,
        default="benchmarks/maps_2p",
        help="基准地图目录或具体地图文件路径",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default="90000,90001,90002",
        help="逗号分隔的随机数种子序列，例如 90000,90001,90002",
    )
    parser.add_argument(
        "--turns",
        type=int,
        default=100,
        help="每个评测用例的模拟推进回合数",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="runs/exp080_scaling/",
        help="遥测结果与评估报告输出目录路径",
    )
    parser.add_argument(
        "--teacher",
        type=str,
        default="compound",
        choices=["compound", "baseline"],
        help="待评估教师模型类型",
    )
    return parser.parse_args(args)


def main(args: list[str] | None = None) -> None:
    parsed = parse_args(args)

    scales = [int(s.strip()) for s in parsed.scales.split(",") if s.strip()]
    if not scales:
        raise ValueError("解析得到的蚂蚁规模序列为空")
    for s in scales:
        if s <= 0:
            raise ValueError(f"蚂蚁规模数值必须为正整数，当前输入为 {s}")

    seeds = [int(s.strip()) for s in parsed.seeds.split(",") if s.strip()]
    if not seeds:
        raise ValueError("解析得到的随机数种子序列为空")

    turns = int(parsed.turns)
    if turns <= 0:
        raise ValueError(f"模拟推进回合数必须大于 0，当前输入为 {turns}")

    maps_path = Path(parsed.maps)
    if maps_path.is_file():
        map_files = [maps_path]
    elif maps_path.is_dir():
        map_files = sorted(list(maps_path.glob("*.json")))
        if not map_files:
            raise FileNotFoundError(f"在目录 {maps_path} 中未发现任何 JSON 地图文件")
    else:
        raise FileNotFoundError(f"指定的地图路径不存在: {maps_path}")

    out_dir = Path(parsed.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / "telemetry.jsonl"
    summary_path = out_dir / "telemetry_summary.json"
    report_path = out_dir / "scaling_report.md"

    all_records: list[dict] = []

    with open(jsonl_path, "w", encoding="utf-8") as jsonl_file:
        for scale in scales:
            for map_file in map_files:
                with open(map_file, "r", encoding="utf-8") as f:
                    board_data = json.load(f)

                map_id = board_data.get("id", map_file.stem)

                for seed in seeds:
                    record = run_scaling_simulation(
                        board_data=board_data,
                        scale=scale,
                        turns=turns,
                        seed=seed,
                        teacher_type=parsed.teacher,
                    )
                    record["map_id"] = map_id
                    record["map_path"] = str(map_file)

                    all_records.append(record)
                    jsonl_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                    jsonl_file.flush()

    summary_payload = {
        "metadata": {
            "scales": scales,
            "seeds": seeds,
            "turns": turns,
            "teacher": parsed.teacher,
            "map_count": len(map_files),
            "total_runs": len(all_records),
        },
        "records": all_records,
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, ensure_ascii=False, indent=2)

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# EXP-080 教师规模压力基准评测报告\n\n")
        f.write(f"- 评估教师模型: {parsed.teacher}\n")
        f.write(f"- 蚂蚁规模阶梯: {scales}\n")
        f.write(f"- 地图数量: {len(map_files)}\n")
        f.write(f"- 随机种子数量: {len(seeds)}\n")
        f.write(f"- 每场模拟推进回合数: {turns}\n\n")
        f.write("| 规模 | 地图 | 走廊单元格数 | 拥塞指数 | 走廊受阻率 (%) | 有效采食速率 | 内部碰撞阻挡率 (%) | 平均耗时 (ms) | P95 耗时 (ms) |\n")
        f.write("|---|---|---|---|---|---|---|---|---|\n")
        for rec in all_records:
            f.write(
                f"| {rec['scale']} | {rec['map_id']} | {rec['corridor_cell_count']} | "
                f"{rec['corridor_congestion_index']:.4f} | "
                f"{rec['corridor_stalling_rate'] * 100:.2f}% | "
                f"{rec['effective_gathering_throughput']:.2f} | "
                f"{rec['collision_blocking_rate'] * 100:.2f}% | "
                f"{rec['mean_latency_ms']:.2f} | "
                f"{rec['p95_latency_ms']:.2f} |\n"
            )


if __name__ == "__main__":
    main()
