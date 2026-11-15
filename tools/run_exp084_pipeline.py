from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str((ROOT / "baselines" / "src").resolve()))

from tb_baselines import nets
from tb_baselines.env import Env
from tb_baselines.export import classes, export, recipe
from tb_baselines.train.bc import device
from tb_baselines.train.mappo import (
    MAPPOActorCritic,
    MAPPORunner,
    RewardConfig,
    update_mappo,
)


def verify_engine_clean():
    res = subprocess.run(["git", "status", "-s", "engine/"], cwd=str(ROOT), capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"检查 engine/ 目录状态失败: {res.stderr}")
    if res.stdout.strip():
        raise RuntimeError(f"engine/ 目录存在修改差异:\n{res.stdout}")

    deny_res = subprocess.run(["./tools/deny.sh"], cwd=str(ROOT), capture_output=True, text=True)
    if deny_res.returncode != 0:
        raise RuntimeError(f"tools/deny.sh 检测未通过:\n{deny_res.stderr or deny_res.stdout}")


def run_mappo_training(
    warm_start_path: Path,
    out_dir: Path,
    iters: int = 50,
    rollout_steps: int = 32,
    waves: int = 2,
    matches_per_wave: int = 4,
    max_turns: int = 150,
    lr: float = 1e-4,
    epochs: int = 2,
    minibatch: int = 16,
    clip: float = 0.2,
    entropy: float = 0.01,
    seed: int = 84000,
) -> dict:
    verify_engine_clean()
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = device()

    spec = classes()["micro"]
    trunk = nets.build(spec, memory="where")

    if not warm_start_path.is_file():
        raise FileNotFoundError(f"未找到热启动权重文件: {warm_start_path}")

    state = torch.load(warm_start_path, map_location="cpu", weights_only=False)
    trunk.load_state_dict(state.get("trunk", state))
    print(f"成功加载 EXP-083 蒸馏热启动权重: {warm_start_path}，Actor 策略参数量: {nets.policy_params(trunk):,}")

    model = MAPPOActorCritic(trunk, privileged_dim=8).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    cfg = RewardConfig(gamma=0.997, lam=0.95, shaping_weight=1.0, cf_weight=0.5)

    maps_dir = ROOT / "benchmarks" / "maps_2p"
    map_files = sorted(list(maps_dir.glob("*.json")))
    if not map_files:
        raise RuntimeError(f"未在 {maps_dir} 找到基准地图")
    map_arg = ",".join(str(m.resolve()) for m in map_files)

    history = []
    best_return = -1e9
    best_ckpt_path = out_dir / "best.pt"

    t_train_start = time.perf_counter()
    with Env(waves=waves, matches_per_wave=matches_per_wave, max_turns=max_turns, maps=map_arg, seed=seed) as env:
        runner = MAPPORunner(model, env, dev, cfg, seed=seed)
        print(f"底座引擎摘要: {env.engine_digest[:20]}，开始 MAPPO 迭代训练 (总计 {iters} 轮)...")

        for it in range(1, iters + 1):
            t0 = time.perf_counter()
            rolled = runner.collect(rollout_steps)
            batch = runner.harvest()
            t1 = time.perf_counter()

            stats = update_mappo(
                model,
                opt,
                batch,
                cfg,
                dev,
                epochs=epochs,
                clip=clip,
                entropy=entropy,
                minibatch=minibatch,
            )
            t2 = time.perf_counter()

            mean_ret = rolled["mean_return"]
            row = {
                "iter": it,
                "episodes": rolled["episodes"],
                "mean_return": round(mean_ret, 4),
                "policy_loss": round(stats.get("policy", 0.0), 6),
                "value_loss": round(stats.get("value", 0.0), 6),
                "entropy": round(stats.get("entropy", 0.0), 4),
                "clipped": round(stats.get("clipped", 0.0), 4),
                "adv_mean": round(stats.get("adv_mean", 0.0), 6),
                "adv_std": round(stats.get("adv_std", 0.0), 6),
                "samples": stats.get("samples", 0),
                "collect_sec": round(t1 - t0, 2),
                "update_sec": round(t2 - t1, 2),
                "total_sec": round(t2 - t0, 2),
            }
            history.append(row)

            if it % 5 == 0 or it == 1 or it == iters:
                print(
                    f"  Iter {it:3d}/{iters} | 回报: {mean_ret:+7.2f} ({rolled['episodes']:2d} 场) | "
                    f"策略损失: {row['policy_loss']:+.4f} | 价值误差: {row['value_loss']:.4f} | "
                    f"熵: {row['entropy']:.3f} | 优势差分差: {row['adv_std']:.4f} | "
                    f"样本: {row['samples']:4d} | 耗时: {row['total_sec']:.1f}s"
                )

            if rolled["episodes"] > 0 and mean_ret > best_return:
                best_return = mean_ret
                torch.save({
                    "trunk": trunk.state_dict(),
                    "class": "micro",
                    "memory": "where",
                    "engine_digest": env.engine_digest,
                    "best_return": best_return,
                    "iter": it,
                }, best_ckpt_path)

    if not best_ckpt_path.is_file():
        torch.save({
            "trunk": trunk.state_dict(),
            "class": "micro",
            "memory": "where",
            "engine_digest": env.engine_digest,
            "best_return": best_return,
            "iter": iters,
        }, best_ckpt_path)

    total_training_sec = time.perf_counter() - t_train_start
    telemetry_payload = {
        "class": "micro",
        "method": "mappo",
        "reward_config": {
            "gamma": cfg.gamma,
            "lam": cfg.lam,
            "shaping_weight": cfg.shaping_weight,
            "cf_weight": cfg.cf_weight,
            "score": cfg.score,
            "rival": cfg.rival,
            "ants": cfg.ants,
            "win": cfg.win,
        },
        "total_training_seconds": round(total_training_sec, 2),
        "iters": history,
    }
    telemetry_file = out_dir / "mappo_telemetry.json"
    telemetry_file.write_text(json.dumps(telemetry_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"训练完成，遥测记录已写入: {telemetry_file} (总耗时: {total_training_sec:.1f} 秒)")
    return telemetry_payload


def export_and_certify_model(
    checkpoint_path: Path,
    export_dir: Path,
) -> dict:
    export_dir.mkdir(parents=True, exist_ok=True)
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    spec = classes()["micro"]
    trunk = nets.build(spec, memory="where")
    trunk.load_state_dict(state.get("trunk", state))

    argv_sim = [
        "--class", "micro",
        "--memory", "where",
        "--weights", str(checkpoint_path.resolve()),
        "--out", str(export_dir.resolve()),
        "--method", "mappo",
    ]
    repro_cmd = recipe(checkpoint_path, argv_sim)
    metrics = export(
        trunk,
        name="micro",
        out=export_dir,
        method="mappo",
        repro=repro_cmd,
        summary="A micro-class Ants MAPPO cooperative reinforcement learning policy.",
        notes="Warm-started from EXP-083 distilled weights; trained with centralized critic and counterfactual baseline credit assignment; strictly stripped critic value head on export.",
    )

    onnx_file = export_dir / "model.onnx"
    manifest_file = export_dir / "manifest.json"

    check_cmd = [
        "tinybrains", "check",
        str(onnx_file.resolve()),
        str(manifest_file.resolve()),
        "--json",
    ]
    env = os.environ.copy()
    env["TINYBRAINS_REGISTRY"] = str((ROOT / "baselines" / "games.toml").resolve())
    check_res = subprocess.run(check_cmd, cwd=str(ROOT), capture_output=True, text=True, env=env)
    if check_res.returncode != 0:
        raise RuntimeError(f"tinybrains check 执行失败:\n{check_res.stderr or check_res.stdout}")

    check_data = json.loads(check_res.stdout)
    if not check_data.get("ok"):
        raise RuntimeError(f"平台准入校验未通过: {check_data.get('reason')}")

    artifact_bytes = check_data["artifact_bytes"]
    median_ms = check_data["probe"]["median_ms"]
    infer_us_max = check_data["infer_us_max"]
    operators = check_data["graph"]["operators"]

    if artifact_bytes > 131072:
        raise RuntimeError(f"ONNX 固件体积超出 128 KB 配额: {artifact_bytes} 字节")
    if median_ms > 30.0:
        raise RuntimeError(f"单步推理耗时超出 30.0 毫秒时限: {median_ms} 毫秒")

    print(f"平台准入校验成功: 体积 {artifact_bytes} 字节 <= 128 KB，探针推理耗时 {median_ms:.2f} ms <= 30.0 ms")
    print(f"算子集合: {operators} (确认无任何价值头残存)")
    return check_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--rollout-steps", type=int, default=24)
    parser.add_argument("--waves", type=int, default=2)
    parser.add_argument("--matches-per-wave", type=int, default=4)
    parser.add_argument("--max-turns", type=int, default=120)
    parser.add_argument("--seed", type=int, default=84000)
    args = parser.parse_args()

    warm_start_path = ROOT / "runs" / "exp083" / "k8" / "best.pt"
    if not warm_start_path.is_file():
        warm_start_path = ROOT / "runs" / "exp083" / "best.pt"

    out_run_dir = ROOT / "runs" / "exp084"
    export_dir = ROOT / "models" / "exp084"

    run_mappo_training(
        warm_start_path=warm_start_path,
        out_dir=out_run_dir,
        iters=args.iters,
        rollout_steps=args.rollout_steps,
        waves=args.waves,
        matches_per_wave=args.matches_per_wave,
        max_turns=args.max_turns,
        seed=args.seed,
    )

    best_ckpt = out_run_dir / "best.pt"
    export_and_certify_model(
        checkpoint_path=best_ckpt,
        export_dir=export_dir,
    )
    verify_engine_clean()


if __name__ == "__main__":
    main()
