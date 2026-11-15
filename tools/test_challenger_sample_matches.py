import itertools
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
EXTRACTOR = ROOT / "benchmarks" / "extract_frames.mjs"
OUT_REPLAYS = ROOT / "replays" / "challenger_m2_sample"

MAP_DIR = ROOT / "benchmarks" / "maps_2p"
MAP_NAMES = [
    "tiny-corridor-2p-1h.json",
    "tiny-colonnade-2p-1h.json",
    "tiny-open-2p-1h.json",
    "small-rooms-2p-1h.json",
    "small-arena-2p-1h.json",
    "small-islands-2p-1h.json",
    "medium-crossroads-2p-1h.json",
    "medium-labyrinth-2p-1h.json",
    "medium-valley-2p-1h.json",
]

TARGET = ROOT / "models" / "exp084"
OPPONENTS = [
    ROOT / "models" / "exp083",
    ROOT / "models" / "exp033",
    ROOT / "models" / "exp031",
    ROOT / "models" / "exp011",
]

if OUT_REPLAYS.exists():
    shutil.rmtree(OUT_REPLAYS)
OUT_REPLAYS.mkdir(parents=True, exist_ok=True)

# 构造双向席位互换对弈配置
rows = []
for map_file in MAP_NAMES:
    map_path = str((MAP_DIR / map_file).resolve())
    map_stem = Path(map_file).stem
    for opp_dir in OPPONENTS:
        opp_name = opp_dir.name
        # 席位 0: exp084, 席位 1: 对手
        mid_0 = f"exp084-vs-{opp_name}-{map_stem}-s0"
        rows.append({
            "id": mid_0,
            "seed": 95000 + len(rows),
            "map": map_path,
            "seat_count": 2,
            "seats": [
                {
                    "seat": 0,
                    "label": "exp084",
                    "weights": str((TARGET / "model.onnx").resolve()),
                    "manifest": str((TARGET / "manifest.json").resolve()),
                },
                {
                    "seat": 1,
                    "label": opp_name,
                    "weights": str((opp_dir / "model.onnx").resolve()),
                    "manifest": str((opp_dir / "manifest.json").resolve()),
                },
            ],
        })

        # 席位 0: 对手, 席位 1: exp084 (席位互换)
        mid_1 = f"{opp_name}-vs-exp084-{map_stem}-s1"
        rows.append({
            "id": mid_1,
            "seed": 95000 + len(rows),
            "map": map_path,
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
                    "weights": str((TARGET / "model.onnx").resolve()),
                    "manifest": str((TARGET / "manifest.json").resolve()),
                },
            ],
        })

print(f"总计生成抽样对战场次: {len(rows)} 场 (涵盖全部 9 张基准地图与 4 种基线模型)")

# 多进程并行执行对局
def run_batch(worker_id: int, chunk: list[dict]):
    batch_file = OUT_REPLAYS / f"_batch_{worker_id}.json"
    batch_file.write_text(json.dumps({
        "game": "ants",
        "vars": {"max_turns": 300},
        "rows": chunk,
    }))
    env = dict(os.environ)
    env["TINYBRAINS_REGISTRY"] = REGISTRY
    res = subprocess.run(
        [CLI, str(batch_file), "--out", str(OUT_REPLAYS)],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        env=env,
    )
    if res.returncode != 0:
        raise RuntimeError(f"批次 {worker_id} 运行失败: {res.stderr}")
    batch_file.unlink(missing_ok=True)
    return len(chunk)

workers = 8
chunks = [[] for _ in range(workers)]
for idx, r in enumerate(rows):
    chunks[idx % workers].append(r)

chunks = [c for c in chunks if c]
print(f"启动并行对战推演，工作进程数: {len(chunks)}...")
t_start = time.perf_counter()
with ProcessPoolExecutor(max_workers=len(chunks)) as executor:
    futures = [executor.submit(run_batch, i, c) for i, c in enumerate(chunks)]
    for f in futures:
        f.result()
t_elapsed = time.perf_counter() - t_start
print(f"对战推演全部完成，耗时: {t_elapsed:.2f} 秒")

# 使用官方 Wasm 重放解析器抽取逐帧指标
print("运行官方 Wasm 重放度量提取器...")
node_res = subprocess.run(
    ["node", str(EXTRACTOR), str(OUT_REPLAYS.resolve())],
    cwd=str(ROOT / "benchmarks"),
    capture_output=True,
    text=True,
)
assert node_res.returncode == 0, f"提取帧度量失败: {node_res.stderr}"
frame_metrics = json.loads(node_res.stdout)

# 汇总 exp084 对战得分率与移动受阻率指标
total_matches = 0
wins = 0
draws = 0
losses = 0
points = 0.0
total_orders = 0
blocked_orders = 0
collisions = 0
hills_razed = 0
total_turns = 0

h2h = {opp.name: {"matches": 0, "wins": 0, "draws": 0, "losses": 0, "points": 0.0, "orders": 0, "blocked": 0} for opp in OPPONENTS}
map_stats = {Path(m).stem: {"matches": 0, "wins": 0, "points": 0.0, "orders": 0, "blocked": 0} for m in MAP_NAMES}

for m in frame_metrics:
    seats = m["seats"]
    target_seat = None
    rival_seat = None
    for s in seats:
        if s["name"] == "exp084" and target_seat is None:
            target_seat = s
        else:
            rival_seat = s

    assert target_seat is not None and rival_seat is not None, "未找到对局席位信息"
    t_rank = target_seat["rank"]
    r_rank = rival_seat["rank"]
    rival_name = rival_seat["name"]
    
    total_matches += 1
    total_turns += m.get("turns", 0)
    t_orders = target_seat["total_orders"]
    t_blocked = target_seat["blocked_orders"]
    t_coll = target_seat["collisions"]

    total_orders += t_orders
    blocked_orders += t_blocked
    collisions += t_coll

    h2h[rival_name]["matches"] += 1
    h2h[rival_name]["orders"] += t_orders
    h2h[rival_name]["blocked"] += t_blocked

    map_id = m["map_id"]
    if map_id in map_stats:
        map_stats[map_id]["matches"] += 1
        map_stats[map_id]["orders"] += t_orders
        map_stats[map_id]["blocked"] += t_blocked

    if t_rank < r_rank:
        wins += 1
        points += 1.0
        h2h[rival_name]["wins"] += 1
        h2h[rival_name]["points"] += 1.0
        if map_id in map_stats:
            map_stats[map_id]["wins"] += 1
            map_stats[map_id]["points"] += 1.0
    elif t_rank == r_rank:
        draws += 1
        points += 0.5
        h2h[rival_name]["draws"] += 1
        h2h[rival_name]["points"] += 0.5
        if map_id in map_stats:
            map_stats[map_id]["points"] += 0.5
    else:
        losses += 1
        h2h[rival_name]["losses"] += 1

score_rate = points / total_matches if total_matches > 0 else 0.0
blocked_rate = blocked_orders / total_orders if total_orders > 0 else 0.0
blocked_rate_pct = blocked_rate * 100.0

print("\n========== 对抗性抽样对战实测结果 ==========")
print(f"总场次: {total_matches}")
print(f"战绩: {wins} 胜 / {draws} 平 / {losses} 负")
print(f"积分: {points:.1f} / {total_matches}")
print(f"实测综合得分率: {score_rate * 100.0:.2f}% (理论门禁标准 >= 65.74%)")
print(f"总移动指令数: {total_orders}")
print(f"受阻指令数: {blocked_orders}")
print(f"实测通道移动受阻率: {blocked_rate_pct:.4f}% (理论门禁标准 <= 0.8000%)")
print(f"实测碰撞次数: {collisions}")

print("\n--- 历史模型交手明细 (Head-to-Head) ---")
for opp_name, stats in h2h.items():
    opp_sr = stats["points"] / stats["matches"] if stats["matches"] > 0 else 0.0
    opp_br = stats["blocked"] / stats["orders"] * 100.0 if stats["orders"] > 0 else 0.0
    print(f"vs {opp_name:7s}: {stats['wins']}胜 {stats['draws']}平 {stats['losses']}负 (得分率: {opp_sr * 100.0:.1f}%, 受阻率: {opp_br:.4f}%)")

print("\n--- 9 张基准地图表现明细 ---")
for mid, mstats in map_stats.items():
    m_sr = mstats["points"] / mstats["matches"] if mstats["matches"] > 0 else 0.0
    m_br = mstats["blocked"] / mstats["orders"] * 100.0 if mstats["orders"] > 0 else 0.0
    print(f"{mid:30s}: 得分率 {m_sr * 100.0:.1f}%, 受阻率 {m_br:.4f}%")

report_data = {
    "total_matches": total_matches,
    "wins": wins,
    "draws": draws,
    "losses": losses,
    "points": points,
    "score_rate": score_rate,
    "total_orders": total_orders,
    "blocked_orders": blocked_orders,
    "blocked_rate": blocked_rate,
    "blocked_rate_pct": blocked_rate_pct,
    "h2h": h2h,
    "map_stats": map_stats,
}
report_path = ROOT / "runs" / "challenger_exp084_sampled_matches.json"
report_path.write_text(json.dumps(report_data, indent=2, ensure_ascii=False))
print(f"\n实测数据已保存至: {report_path}")
