from __future__ import annotations

import datetime
import json
import os
import shutil
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_BASELINES_SRC = _ROOT / "baselines" / "src"
if str(_BASELINES_SRC.resolve()) not in sys.path:
    sys.path.insert(0, str(_BASELINES_SRC.resolve()))

from tb_baselines.export import main as export_main, verdict
from tb_baselines.train import bc

PROGRESS_FILE = Path(
    "/home/host/.gemini/antigravity-cli/scratch/teamwork/d5655af1-3615-437e-911e-1079b0953b1c/.agents/teamwork/teamwork_worker_m3/progress.md"
)


def update_progress_heartbeat(phase: str, details: str = "") -> None:
    now_str = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    content = f"""# Progress — Worker M3

Last visited: {now_str}
Status: IN_PROGRESS

## Current Phase: {phase}
{details}
"""
    if PROGRESS_FILE.parent.exists():
        PROGRESS_FILE.write_text(content, encoding="utf-8")


def main() -> None:
    data_path = _ROOT / "data" / "exp085_all.jsonl.gz"
    if not data_path.exists():
        raise FileNotFoundError(f"未找到全量融合数据集: {data_path}")

    out_run = _ROOT / "runs" / "exp085_unified"
    out_run.mkdir(parents=True, exist_ok=True)
    out_model = _ROOT / "models" / "exp085"

    print(f"=== 开始全谱系融合策略网络训练 (750,000 样本, 5 轮次, batch 128) ===")
    update_progress_heartbeat("Unified Training", "Training policy on 750,000 multi-scale samples...")

    cmd_args = [
        "--class", "micro",
        "--memory",
        "--data", str(data_path.resolve()),
        "--epochs", "5",
        "--batch", "128",
        "--lr", "0.002",
        "--seed", "42",
        "--out", str(out_run),
    ]

    t0 = time.time()
    bc.main(cmd_args)
    elapsed = time.time() - t0
    print(f"全谱系策略网络训练完成，耗时: {elapsed:.1f} 秒")

    best_pt = out_run / "best.pt"
    if not best_pt.exists():
        raise FileNotFoundError(f"未生成最优权重检查点: {best_pt}")

    print(f"=== 导出融合优化模型构件 models/exp085 ===")
    update_progress_heartbeat("Model Export", "Exporting unified models/exp085...")
    export_args = [
        "--class", "micro",
        "--memory", "where",
        "--weights", str(best_pt.resolve()),
        "--out", str(out_model),
        "--method", "EXP-085: unified full-spectrum multi-seat MCMF DAgger + Where memory Micro BC",
    ]
    export_main(export_args)

    model_path = out_model / "model.onnx"
    manifest_path = out_model / "manifest.json"
    metrics_path = out_model / "metrics.json"

    assert model_path.exists(), f"模型缺失: {model_path}"
    assert manifest_path.exists(), f"清单缺失: {manifest_path}"
    assert metrics_path.exists(), f"指标缺失: {metrics_path}"

    said = verdict(model_path.resolve(), manifest_path.resolve())
    assert said.get("ok"), f"平台准入拒绝: {said.get('reason')}"
    print("模型平台准入校验通过！")
    print(f"指标体积: {said.get('size_metric_bytes')} 字节, 峰值推理: {said.get('infer_us_max')} 微秒")

    update_progress_heartbeat("Training & Export Complete", f"Artifact size: {said.get('size_metric_bytes')} bytes")


if __name__ == "__main__":
    main()
