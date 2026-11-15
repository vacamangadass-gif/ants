import json
import os
import shutil
import subprocess
import time
from pathlib import Path

ROOT = Path("/home/host/project/ants")
CLI = os.environ.get("TINYBRAINS") or shutil.which("tinybrains") or "/home/linuxbrew/.linuxbrew/bin/tinybrains"
REGISTRY = str((ROOT / "baselines" / "games.toml").resolve())
EXTRACTOR = ROOT / "benchmarks" / "extract_frames.mjs"
OUT_REPLAYS = ROOT / "replays" / "challenger_m2_corridor_stress"

if OUT_REPLAYS.exists():
    shutil.rmtree(OUT_REPLAYS)
OUT_REPLAYS.mkdir(parents=True, exist_ok=True)

TARGET = ROOT / "models" / "exp084"
corridor_map = str((ROOT / "benchmarks" / "maps_2p" / "tiny-corridor-2p-1h.json").resolve())

rows = []
for i in range(10):
    rows.append({
        "id": f"selfplay-corridor-stress-seed-{88000 + i}",
        "seed": 88000 + i,
        "map": corridor_map,
        "seat_count": 2,
        "seats": [
            {
                "seat": 0,
                "label": "exp084_p0",
                "weights": str((TARGET / "model.onnx").resolve()),
                "manifest": str((TARGET / "manifest.json").resolve()),
            },
            {
                "seat": 1,
                "label": "exp084_p1",
                "weights": str((TARGET / "model.onnx").resolve()),
                "manifest": str((TARGET / "manifest.json").resolve()),
            },
        ],
    })

batch_file = OUT_REPLAYS / "_batch_stress.json"
batch_file.write_text(json.dumps({
    "game": "ants",
    "vars": {"max_turns": 300},
    "rows": rows,
}))

env = dict(os.environ)
env["TINYBRAINS_REGISTRY"] = REGISTRY
t0 = time.perf_counter()
res = subprocess.run(
    [CLI, str(batch_file), "--out", str(OUT_REPLAYS)],
    cwd=str(ROOT),
    capture_output=True,
    text=True,
    env=env,
)
assert res.returncode == 0, f"通道自博弈压力测试失败: {res.stderr}"
batch_file.unlink(missing_ok=True)
elapsed = time.perf_counter() - t0
print(f"通道狭窄瓶颈 10 局同质自博弈推演完成，耗时: {elapsed:.2f} 秒")

node_res = subprocess.run(
    ["node", str(EXTRACTOR), str(OUT_REPLAYS.resolve())],
    cwd=str(ROOT / "benchmarks"),
    capture_output=True,
    text=True,
)
assert node_res.returncode == 0, f"提取帧度量失败: {node_res.stderr}"
frame_metrics = json.loads(node_res.stdout)

total_turns = sum(m.get("turns", 0) for m in frame_metrics)
total_orders_all = sum(s["total_orders"] for m in frame_metrics for s in m["seats"])
blocked_orders_all = sum(s["blocked_orders"] for m in frame_metrics for s in m["seats"])
collisions_all = sum(s["collisions"] for m in frame_metrics for s in m["seats"])
razed_hills_all = sum(len(m.get("razed_hills", [])) for m in frame_metrics)

blocked_rate_pct = (blocked_orders_all / total_orders_all * 100.0) if total_orders_all > 0 else 0.0

print(f"10 局通道狭窄瓶颈压力实测:")
print(f"平均回合数: {total_turns / len(frame_metrics):.1f} / 300")
print(f"总移动指令: {total_orders_all}")
print(f"受阻移动指令: {blocked_orders_all} (受阻率: {blocked_rate_pct:.4f}%)")
print(f"总碰撞次数: {collisions_all}")
print(f"蚁穴推平局数: {razed_hills_all} / {len(frame_metrics)}")
assert blocked_rate_pct <= 0.8, f"通道狭窄压力下受阻率超标: {blocked_rate_pct:.4f}% > 0.8%"
print("[通过] 通道狭窄瓶颈压力测试通过")
