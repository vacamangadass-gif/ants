import json
import os
import subprocess

model_path = "models/exp084/model.onnx"
manifest_path = "models/exp084/manifest.json"

env = dict(os.environ)
env["TINYBRAINS_REGISTRY"] = "baselines/games.toml"

probe_medians = []
for i in range(10):
    res = subprocess.run(
        ["tinybrains", "check", model_path, manifest_path, "--json"],
        capture_output=True,
        text=True,
        env=env,
    )
    assert res.returncode == 0, f"tinybrains check 失败: {res.stderr}"
    data = json.loads(res.stdout)
    assert data["ok"] is True, f"准入检验失败: {data}"
    median_ms = data["probe"]["median_ms"]
    probe_medians.append(median_ms)
    print(f"轮次 {i + 1}: median_ms = {median_ms:.3f} ms")

sorted_medians = sorted(probe_medians)
overall_median = sorted_medians[len(sorted_medians) // 2]
print(f"总体统计: 最小值={min(probe_medians):.3f} ms, 中位数={overall_median:.3f} ms, 最大值={max(probe_medians):.3f} ms, 均值={sum(probe_medians)/len(probe_medians):.3f} ms")
