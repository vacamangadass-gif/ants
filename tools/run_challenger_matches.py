import itertools
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

ROOT = Path("/home/host/project/ants")
CLI = os.environ.get("TINYBRAINS") or shutil.which("tinybrains") or "/home/linuxbrew/.linuxbrew/bin/tinybrains"
REGISTRY = str((ROOT / "baselines" / "games.toml").resolve())
EXTRACTOR = ROOT / "benchmarks" / "extract_frames.mjs"

@dataclass
class Entrant:
    name: str
    weights: Path
    manifest: Path

    def seat(self, index: int) -> dict:
        return {
            "seat": index,
            "label": self.name,
            "weights": str(self.weights.resolve()),
            "manifest": str(self.manifest.resolve()),
        }

target = Entrant(
    name="exp084",
    weights=ROOT / "models" / "exp084" / "model.onnx",
    manifest=ROOT / "models" / "exp084" / "manifest.json",
)

opponents = [
    Entrant(
        name="exp083",
        weights=ROOT / "models" / "exp083" / "model.onnx",
        manifest=ROOT / "models" / "exp083" / "manifest.json",
    ),
    Entrant(
        name="exp033",
        weights=ROOT / "models" / "exp033" / "model.onnx",
        manifest=ROOT / "models" / "exp033" / "manifest.json",
    ),
    Entrant(
        name="exp031",
        weights=ROOT / "models" / "exp031" / "model.onnx",
        manifest=ROOT / "models" / "exp031" / "manifest.json",
    ),
    Entrant(
        name="exp011",
        weights=ROOT / "models" / "exp011" / "model.onnx",
        manifest=ROOT / "models" / "exp011" / "manifest.json",
    ),
]

maps_dir = ROOT / "benchmarks" / "maps_2p"
map_files = sorted(list(maps_dir.glob("*.json")))
assert len(map_files) == 9, f"基准地图数量非 9: {len(map_files)}"

rows = []
base_seed = 88800
for b_idx, map_file in enumerate(map_files):
    map_name = map_file.stem
    for opp in opponents:
        for flip in (0, 1):
            s0 = target if flip == 0 else opp
            s1 = opp if flip == 0 else target
            match_id = f"challenger-s{flip}-{target.name}-vs-{opp.name}-{map_name}-{b_idx}"
            rows.append({
                "id": match_id,
                "seed": base_seed + len(rows),
                "map": str(map_file.resolve()),
                "seat_count": 2,
                "seats": [s0.seat(0), s1.seat(1)],
            })

out_dir = ROOT / "replays" / "challenger_m2_empirical"
if out_dir.exists():
    shutil.rmtree(out_dir)
out_dir.mkdir(parents=True, exist_ok=True)

workers = 16
num_workers = min(workers, len(rows))
chunk_size = (len(rows) + num_workers - 1) // num_workers
chunks = [rows[i:i + chunk_size] for i in range(0, len(rows), chunk_size)]

def _execute_worker_batch(args):
    worker_id, chunk_rows, target_out_dir = args
    batch_file = target_out_dir / f"_batch_{worker_id}.json"
    batch_file.write_text(json.dumps({
        "game": "ants",
        "vars": {"max_turns": 300},
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
    assert res.returncode == 0, f"Worker {worker_id} 运行失败: {res.stderr or res.stdout}"
    batch_file.unlink(missing_ok=True)
    return len(chunk_rows), duration

t_start = time.perf_counter()
with ProcessPoolExecutor(max_workers=num_workers) as executor:
    worker_tasks = [(i, chunk, out_dir) for i, chunk in enumerate(chunks)]
    results = list(executor.map(_execute_worker_batch, worker_tasks))
total_wall_time = time.perf_counter() - t_start

print(f"成功完成 {len(rows)} 场对局，耗时: {total_wall_time:.2f} 秒")

node_cmd = ["node", str(EXTRACTOR), str(out_dir.resolve())]
node_res = subprocess.run(node_cmd, cwd=str(ROOT / "benchmarks"), capture_output=True, text=True)
assert node_res.returncode == 0, f"提取指标失败: {node_res.stderr or node_res.stdout}"
frame_metrics = json.loads(node_res.stdout)

target_name = "exp084"
matches_count = 0
wins = 0
draws = 0
losses = 0
points = 0.0
total_orders = 0
blocked_orders = 0
score_for = 0
score_against = 0

h2h_records = {opp.name: {"matches": 0, "wins": 0, "draws": 0, "losses": 0, "points": 0.0, "orders": 0, "blocked": 0} for opp in opponents}
map_records = {m.stem: {"matches": 0, "wins": 0, "draws": 0, "losses": 0, "points": 0.0, "orders": 0, "blocked": 0} for m in map_files}

for match in frame_metrics:
    seats = match["seats"]
    map_id = match["map_id"]
    t_seat = None
    o_seat = None
    for s in seats:
        if s["name"] == target_name:
            t_seat = s
        else:
            o_seat = s
    assert t_seat is not None and o_seat is not None, f"未匹配到对战双方: {seats}"

    matches_count += 1
    total_orders += t_seat["total_orders"]
    blocked_orders += t_seat["blocked_orders"]
    score_for += t_seat["score"]
    score_against += o_seat["score"]

    opp_name = o_seat["name"]
    h2h = h2h_records[opp_name]
    m_rec = map_records[map_id]

    h2h["matches"] += 1
    h2h["orders"] += t_seat["total_orders"]
    h2h["blocked"] += t_seat["blocked_orders"]
    m_rec["matches"] += 1
    m_rec["orders"] += t_seat["total_orders"]
    m_rec["blocked"] += t_seat["blocked_orders"]

    if t_seat["won"]:
        wins += 1
        points += 1.0
        h2h["wins"] += 1
        h2h["points"] += 1.0
        m_rec["wins"] += 1
        m_rec["points"] += 1.0
    elif t_seat["drawn"]:
        draws += 1
        points += 0.5
        h2h["draws"] += 1
        h2h["points"] += 0.5
        m_rec["draws"] += 1
        m_rec["points"] += 0.5
    else:
        losses += 1
        h2h["losses"] += 1
        m_rec["losses"] += 1

score_rate = points / matches_count
blocked_rate = (blocked_orders / total_orders) if total_orders > 0 else 0.0

summary = {
    "total_matches": matches_count,
    "wins": wins,
    "draws": draws,
    "losses": losses,
    "points": points,
    "score_rate": score_rate,
    "total_orders": total_orders,
    "blocked_orders": blocked_orders,
    "blocked_rate": blocked_rate,
    "blocked_rate_pct": blocked_rate * 100.0,
    "score_for": score_for,
    "score_against": score_against,
    "h2h": h2h_records,
    "maps": map_records,
}

print(json.dumps(summary, indent=2, ensure_ascii=False))

summary_path = ROOT / "runs" / "exp084_challenger_empirical_summary.json"
summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
print(f"实测结果已输出至 {summary_path}")
