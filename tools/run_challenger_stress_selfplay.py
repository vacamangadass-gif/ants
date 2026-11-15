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

target_weights = ROOT / "models" / "exp084" / "model.onnx"
target_manifest = ROOT / "models" / "exp084" / "manifest.json"

corridor_map = ROOT / "benchmarks" / "maps_2p" / "tiny-corridor-2p-1h.json"
labyrinth_map = ROOT / "benchmarks" / "maps_2p" / "medium-labyrinth-2p-1h.json"

stress_rows = []
for idx, m_path in enumerate([corridor_map, labyrinth_map]):
    for seed in [99001, 99002, 99003, 99004, 99005]:
        stress_rows.append({
            "id": f"stress-selfplay-{m_path.stem}-s{seed}",
            "seed": seed,
            "map": str(m_path.resolve()),
            "seat_count": 2,
            "seats": [
                {
                    "seat": 0,
                    "label": "exp084_seat0",
                    "weights": str(target_weights.resolve()),
                    "manifest": str(target_manifest.resolve()),
                },
                {
                    "seat": 1,
                    "label": "exp084_seat1",
                    "weights": str(target_weights.resolve()),
                    "manifest": str(target_manifest.resolve()),
                },
            ],
        })

out_dir = ROOT / "replays" / "challenger_stress_selfplay"
if out_dir.exists():
    shutil.rmtree(out_dir)
out_dir.mkdir(parents=True, exist_ok=True)

batch_file = out_dir / "_batch_stress.json"
batch_file.write_text(json.dumps({
    "game": "ants",
    "vars": {"max_turns": 300},
    "rows": stress_rows,
}, ensure_ascii=False))

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
assert res.returncode == 0, f"压力对局运行失败: {res.stderr or res.stdout}"
batch_file.unlink(missing_ok=True)

print(f"自博弈狭窄通道压力测试 10 局完成，耗时: {duration:.2f} 秒")

node_cmd = ["node", str(EXTRACTOR), str(out_dir.resolve())]
node_res = subprocess.run(node_cmd, cwd=str(ROOT / "benchmarks"), capture_output=True, text=True)
assert node_res.returncode == 0, f"提取指标失败: {node_res.stderr or node_res.stdout}"
frame_metrics = json.loads(node_res.stdout)

total_turns = sum(m["turns"] for m in frame_metrics)
total_razed = sum(len(m.get("razed_hills", [])) for m in frame_metrics)
print(f"10 局对局平均回合数: {total_turns / len(frame_metrics):.1f}, 摧毁蚁穴总数: {total_razed}")

for m in frame_metrics:
    s0 = m["seats"][0]
    s1 = m["seats"][1]
    print(f"对局 {m['match_id']}: 回合 {m['turns']}, 原因 {m['reason']}, 座位0指令 {s0['total_orders']} (受阻 {s0['blocked_orders']}), 座位1指令 {s1['total_orders']} (受阻 {s1['blocked_orders']})")
