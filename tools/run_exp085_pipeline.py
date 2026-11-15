from __future__ import annotations

import argparse
import datetime
import gzip
import hashlib
import json
import os
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[1]
_BASELINES_SRC = _ROOT / "baselines" / "src"
if str(_BASELINES_SRC.resolve()) not in sys.path:
    sys.path.insert(0, str(_BASELINES_SRC.resolve()))

from tb_baselines.dagger import (
    DatasetAggregator,
    InformationAlignmentGate,
    STRIDE,
    StateSignature,
    StateTracker,
    StudentDriver,
)
from tb_baselines.env import Env
from tb_baselines.export import classes, main as export_main, verdict
from tb_baselines.mcmf_teacher import MCMFTeacher
from tb_baselines.planes import MEMORY, encode, remember
from tb_baselines.train import bc

EXPECTED_HEX_DIGEST = "819166e7918117a1cbbd67dc042e7368a076a2dd45cea5122683983912d591e1"
EXPECTED_DIGEST = f"sha256:{EXPECTED_HEX_DIGEST}"
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


@dataclass(frozen=True)
class PipelineConfig:
    exp_name: str = "exp085"
    weight_class: str = "micro"
    start_iter: int = 1
    end_iter: int = 4
    d0_samples: int = 100000
    samples_per_iter: int = 50000
    pretrain_epochs: int = 5
    epochs: int = 5
    batch_size: int = 128
    lr: float = 2e-3
    device: str | None = "cuda"
    seed: int = 42
    hill_lead: int = 6
    max_depth: int | None = None
    adaptive_raid: bool = True
    threat_avoidance: bool = True
    focus_target: bool = True
    dynamic_congestion: bool = True
    congestion_alpha: float = 10.0
    congestion_beta: float = 1.5
    waves: int = 4
    matches_per_wave: int = 16
    max_turns: int = 300
    project_root: Path = _ROOT


def build_full_spectrum_maps(project_root: Path) -> str:
    base_maps = sorted(list((project_root / "maps").glob("*.json")))
    if len(base_maps) != 5:
        raise RuntimeError(f"基础棋盘数量不匹配！期望 5 张，实际找到 {len(base_maps)} 张")

    bench_maps = sorted(list((project_root / "benchmarks" / "maps_2p").glob("*.json")))
    if len(bench_maps) != 9:
        raise RuntimeError(f"2p 基准棋盘数量不匹配！期望 9 张，实际找到 {len(bench_maps)} 张")

    all_maps = [str(p.resolve()) for p in base_maps + bench_maps]
    return ",".join(all_maps)


def verify_engine_digest(wasm_path: Path) -> None:
    if not wasm_path.exists():
        raise FileNotFoundError(f"未找到底座引擎构件: {wasm_path}")
    with open(wasm_path, "rb") as f:
        actual_hash = hashlib.sha256(f.read()).hexdigest()
    if actual_hash != EXPECTED_HEX_DIGEST:
        raise RuntimeError(
            f"Wasm 引擎散列摘要不匹配！期望 {EXPECTED_HEX_DIGEST}，实际为 {actual_hash}"
        )


def check_prerequisites(project_root: Path) -> None:
    wasm_path = project_root / "dist" / "tb-ants.wasm"
    verify_engine_digest(wasm_path)

    games_toml = project_root / "baselines" / "games.toml"
    if not games_toml.exists():
        raise FileNotFoundError(f"未找到 TinyBrains 注册表文件: {games_toml}")
    os.environ["TINYBRAINS_REGISTRY"] = str(games_toml.resolve())

    if shutil.which("tinybrains") is None and not Path("/home/linuxbrew/.linuxbrew/bin/tinybrains").exists():
        raise RuntimeError("环境变量 PATH 中未找到 tinybrains 可执行文件")


def collect_mcmf_d0(
    out_path: Path,
    seat_turns: int = 100000,
    waves: int = 4,
    matches_per_wave: int = 16,
    max_turns: int = 300,
    seed: int = 42,
    maps: str | None = None,
    hill_lead: int = 6,
    max_depth: int | None = None,
    adaptive_raid: bool = True,
    threat_avoidance: bool = True,
    focus_target: bool = True,
    dynamic_congestion: bool = True,
    congestion_alpha: float = 10.0,
    congestion_beta: float = 1.5,
) -> dict[str, Any]:
    teacher = MCMFTeacher(
        hill_lead=hill_lead,
        max_depth=max_depth,
        adaptive_raid=adaptive_raid,
        threat_avoidance=threat_avoidance,
        focus_target=focus_target,
        dynamic_congestion=dynamic_congestion,
        congestion_alpha=congestion_alpha,
        congestion_beta=congestion_beta,
    )
    gate = InformationAlignmentGate(teacher=teacher, hill_lead=hill_lead)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    episodes = 0
    reasons: dict[str, int] = {}
    scores: list[int] = []
    started = time.time()
    last_heartbeat_time = started

    with Env(waves=waves, matches_per_wave=matches_per_wave, max_turns=max_turns,
             seed=seed, maps=maps, cwd=_ROOT) as env, gzip.open(out_path, "wt") as f:
        header = {
            "kind": "teacher-rollout",
            "engine_digest": env.engine_digest,
            "evaluator": env.evaluator,
            "maps": [m["id"] for m in env.hello["maps"]],
            "max_turns": max_turns,
            "env_seed": seed,
            "teacher": "MCMF coordination teacher (max_depth=None, adaptive_raid=True, threat_avoidance=True, focus_target=True), remembering",
            "teacher_type": "mcmf",
            "teacher_remembers": True,
            "hill_lead": hill_lead,
            "memory_mode": "where",
            "memory": [m.name for m in MEMORY],
        }
        f.write(json.dumps(header) + "\n")

        memories: dict[tuple[int, int], dict[str, Any]] = {}
        step = env.reset()

        while written < seat_turns:
            orders = []
            for s in step.seats:
                key = (s.ep, s.seat)
                mem = memories.get(key)
                moves, order_str = gate.label(s.obs, mem)
                orders.append(order_str)

                if s.obs["mine"] and written < seat_turns:
                    row = {
                        "o": s.obs,
                        "k": [s.ep, s.seat, s.turn],
                        "m": mem or {},
                        "a": order_str,
                    }
                    f.write(json.dumps(row, separators=(",", ":")) + "\n")
                    written += 1

                memories[key] = remember(mem, s.obs, mode="where")

            step = env.step(orders)
            for e in step.ended:
                episodes += 1
                reasons[e.reason] = reasons.get(e.reason, 0) + 1
                scores.append(max(e.scores))
                for seat in range(e.seat_count):
                    memories.pop((e.ep, seat), None)

            if not step.seats:
                break

            now = time.time()
            if now - last_heartbeat_time > 60:
                last_heartbeat_time = now
                rate = written / max(now - started, 1e-6)
                update_progress_heartbeat(
                    "D0 Collection",
                    f"- Samples collected: {written:,} / {seat_turns:,} ({written / seat_turns:.1%})\n"
                    f"- Episodes completed: {episodes}\n"
                    f"- Speed: {rate:.1f} rows/s\n"
                    f"- Elapsed: {now - started:.1f}s"
                )
                print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] D0 进度: {written:,} / {seat_turns:,} ({rate:.1f} rows/s)")

    elapsed = time.time() - started
    stats = {
        "rows": written,
        "episodes": episodes,
        "reasons": reasons,
        "mean_best_score": round(sum(scores) / len(scores), 2) if scores else None,
        "seconds": round(elapsed, 1),
        "rows_per_second": round(written / max(elapsed, 1e-9)),
    }
    stats_path = out_path.with_suffix("").with_suffix(".stats.json")
    stats_path.write_text(json.dumps(stats, indent=2) + "\n")
    return stats


def rollout_iteration_m3(
    student_driver: StudentDriver,
    gate: InformationAlignmentGate,
    out_path: Path | str,
    seat_turns: int,
    k_iter: int,
    offset_stride: int = 0,
    waves: int = 4,
    matches_per_wave: int = 16,
    max_turns: int = 300,
    seed: int = 1,
    maps: str | None = None,
    baseline_signatures: set[tuple] | None = None,
    prior_state_hashes: set[bytes] | None = None,
) -> dict[str, Any]:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    episode_offset = k_iter * offset_stride

    written = 0
    episodes = 0
    scores: list[int] = []
    reasons: dict[str, int] = {}
    started = time.time()
    last_heartbeat_time = started

    tracker = StateTracker(baseline_signatures=baseline_signatures)
    iteration_unique_states: set[bytes] = set()
    novel_states: set[bytes] = set()

    with Env(
        waves=waves,
        matches_per_wave=matches_per_wave,
        max_turns=max_turns,
        seed=seed,
        maps=maps,
        cwd=_ROOT,
    ) as env, gzip.open(out_path, "wt") as f:
        header = {
            "kind": "dagger-rollout",
            "engine_digest": env.engine_digest,
            "evaluator": env.evaluator,
            "maps": [m["id"] for m in env.hello["maps"]],
            "max_turns": max_turns,
            "env_seed": seed,
            "teacher": "MCMF coordination teacher (max_depth=None, adaptive_raid=True, threat_avoidance=True, focus_target=True), remembering",
            "teacher_type": "mcmf",
            "teacher_remembers": True,
            "hill_lead": gate.teacher.hill_lead,
            "memory_mode": "where",
            "memory": [m.name for m in MEMORY],
            "dagger_iteration": k_iter,
            "episode_offset": episode_offset,
        }
        f.write(json.dumps(header) + "\n")

        student_memory: dict[tuple[int, int], dict[str, Any]] = {}
        step = env.reset()

        while written < seat_turns:
            student_orders, student_moves = student_driver.predict_step(step, student_memory)

            for seat_idx, s in enumerate(step.seats):
                key = (s.ep, s.seat)
                M_t = student_memory.get(key)
                teacher_moves, teacher_order = gate.label(s.obs, M_t)

                if s.obs["mine"]:
                    n_ants = len(s.obs["mine"])
                    s_moves = student_moves[seat_idx]
                    if len(s_moves) == n_ants and len(teacher_moves) == n_ants:
                        tracker.record_step(s.obs, M_t, s_moves, teacher_moves)

                    fp = hashlib.sha256(encode(s.obs).tobytes()).digest()[:8]
                    iteration_unique_states.add(fp)
                    if prior_state_hashes is not None and fp not in prior_state_hashes:
                        novel_states.add(fp)

                    if written < seat_turns:
                        row = {
                            "o": s.obs,
                            "k": [s.ep + episode_offset, s.seat, s.turn],
                            "m": M_t or {},
                            "a": teacher_order,
                        }
                        f.write(json.dumps(row, separators=(",", ":")) + "\n")
                        written += 1

                student_memory[key] = remember(M_t, s.obs, mode="where")

            step = env.step(student_orders)
            for e in step.ended:
                episodes += 1
                reasons[e.reason] = reasons.get(e.reason, 0) + 1
                scores.append(max(e.scores))
                for seat in range(e.seat_count):
                    student_memory.pop((e.ep, seat), None)

            if not step.seats:
                break

            now = time.time()
            if now - last_heartbeat_time > 60:
                last_heartbeat_time = now
                rate = written / max(now - started, 1e-6)
                update_progress_heartbeat(
                    f"DAgger Iteration {k_iter} Rollout",
                    f"- Samples collected: {written:,} / {seat_turns:,} ({written / seat_turns:.1%})\n"
                    f"- Episodes completed: {episodes}\n"
                    f"- Disagreement rate: {tracker.disagreement_rate:.4f}\n"
                    f"- Speed: {rate:.1f} rows/s\n"
                    f"- Elapsed: {now - started:.1f}s"
                )
                print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] DAgger {k_iter} 推演进度: {written:,} / {seat_turns:,} ({rate:.1f} rows/s, 歧异率={tracker.disagreement_rate:.4f})")

    elapsed = time.time() - started
    disagreement_rate = tracker.disagreement_rate
    novel_ratio = tracker.novel_state_ratio

    stats = {
        "iteration": k_iter,
        "rows": written,
        "ant_decisions": tracker.total_ants,
        "disagreed_ants": tracker.disagree_ants,
        "disagreement_rate": round(disagreement_rate, 4),
        "unique_states": len(tracker.delta_signatures),
        "novel_states": tracker.novel_steps,
        "novel_state_ratio": round(novel_ratio, 4),
        "episodes": episodes,
        "reasons": reasons,
        "mean_best_score": round(sum(scores) / len(scores), 2) if scores else None,
        "seconds": round(elapsed, 1),
        "rows_per_second": round(written / max(elapsed, 1e-9)),
        "out": str(out_path),
    }
    stats_file = out_path.with_suffix("").with_suffix(".stats.json")
    stats_file.write_text(json.dumps(stats, indent=2) + "\n")
    return stats


def train_iteration_model(
    cls: str,
    data_path: Path,
    out_dir: Path,
    epochs: int = 5,
    batch: int = 128,
    lr: float = 2e-3,
    seed: int = 42,
    device: str | None = None,
) -> tuple[Path, dict[str, Any]]:
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if device is not None:
        dev_str = device.strip().lower()
        if dev_str == "cpu":
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
        elif dev_str == "cuda":
            os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    cmd_args = [
        "--class", cls,
        "--memory",
        "--data", str(data_path.resolve()),
        "--epochs", str(epochs),
        "--batch", str(batch),
        "--lr", str(lr),
        "--seed", str(seed),
        "--out", str(out_dir),
    ]

    bc.main(cmd_args)

    best_pt = out_dir / "best.pt"
    history_json = out_dir / "history.json"

    if not best_pt.exists():
        raise FileNotFoundError(f"最优模型检查点文件不存在: {best_pt}")
    if not history_json.exists():
        raise FileNotFoundError(f"训练历史记录文件不存在: {history_json}")

    history_data = json.loads(history_json.read_text(encoding="utf-8"))
    return best_pt, history_data


def validate_admission_verdict(said: dict[str, Any], cls: str) -> None:
    if not said.get("ok"):
        raise AssertionError(f"平台准入校验拒绝了构件: {said.get('reason')}")

    if said.get("engine_digest") != EXPECTED_DIGEST:
        raise AssertionError(
            f"平台校验返回的引擎散列摘要不匹配！期望 {EXPECTED_DIGEST}，实际为 {said.get('engine_digest')}"
        )

    size_bytes = said.get("size_metric_bytes", 0)
    micro_limit = 128 * 1024
    if size_bytes >= micro_limit:
        raise AssertionError(
            f"Micro 构件指标体积 {size_bytes} 字节超出 128 KB（{micro_limit} 字节）上限！"
        )

    infer_us = said.get("infer_us_max", 0)
    if infer_us > 700_000:
        raise AssertionError(
            f"单回合推理峰值时延 {infer_us} 微秒超出 700 毫秒上限！"
        )

    mem_info = said.get("memory")
    if mem_info and "round_trip" in mem_info:
        failed_count = mem_info["round_trip"].get("failed", 0)
        if failed_count != 0:
            raise AssertionError(f"记忆往返一致性测试存在 {failed_count} 次失败！")


def export_and_verify_model(
    cls: str,
    weights_path: Path,
    out_dir: Path,
    method: str = "dagger",
) -> dict[str, Any]:
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    export_args = [
        "--class", cls,
        "--memory", "where",
        "--weights", str(weights_path.resolve()),
        "--out", str(out_dir),
        "--method", method,
    ]
    export_main(export_args)

    model_path = out_dir / "model.onnx"
    manifest_path = out_dir / "manifest.json"
    metrics_path = out_dir / "metrics.json"

    if not model_path.exists():
        raise FileNotFoundError(f"导出的模型文件缺失: {model_path}")
    if not manifest_path.exists():
        raise FileNotFoundError(f"导出的清单文件缺失: {manifest_path}")
    if not metrics_path.exists():
        raise FileNotFoundError(f"导出的指标文件缺失: {metrics_path}")

    said = verdict(model_path.resolve(), manifest_path.resolve())
    validate_admission_verdict(said, cls=cls)
    return said


def extract_signatures_from_file(file_path: Path) -> set[tuple]:
    if not file_path.exists():
        raise FileNotFoundError(f"数据文件不存在: {file_path}")
    sigs: set[tuple] = set()
    with gzip.open(file_path, "rt", encoding="utf-8") as f:
        header_line = f.readline()
        if header_line and not header_line.endswith("\n"):
            raise ValueError(f"数据文件 {file_path} 头部数据损坏")
        for line in f:
            if line.endswith("\n"):
                row = json.loads(line)
                sigs.add(StateSignature.compute(row["o"], row.get("m")))
    return sigs


class Exp085Pipeline:
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.project_root = config.project_root
        self.aggregator = DatasetAggregator(stride=STRIDE)
        self.all_maps_str = build_full_spectrum_maps(self.project_root)

    def validate_environment(self) -> None:
        check_prerequisites(project_root=self.project_root)

    def get_or_create_d0(self) -> Path:
        d0_path = self.project_root / "data" / f"{self.config.exp_name}_d0.jsonl.gz"
        if d0_path.exists():
            print(f"找到已存在的 D_0 数据集: {d0_path}")
            return d0_path

        print(f"\n=== 开始采集 EXP-085 初始全谱系专家数据集 D_0 ({self.config.d0_samples:,} 席位步数) ===")
        update_progress_heartbeat("D0 Collection Started", f"Target: {self.config.d0_samples:,} seat turns across 14 maps")
        collect_mcmf_d0(
            out_path=d0_path,
            seat_turns=self.config.d0_samples,
            waves=self.config.waves,
            matches_per_wave=self.config.matches_per_wave,
            max_turns=self.config.max_turns,
            seed=self.config.seed,
            maps=self.all_maps_str,
            hill_lead=self.config.hill_lead,
            max_depth=self.config.max_depth,
            adaptive_raid=self.config.adaptive_raid,
            threat_avoidance=self.config.threat_avoidance,
            focus_target=self.config.focus_target,
            dynamic_congestion=self.config.dynamic_congestion,
            congestion_alpha=self.config.congestion_alpha,
            congestion_beta=self.config.congestion_beta,
        )
        update_progress_heartbeat("D0 Collection Finished", f"Saved to {d0_path.name}")
        return d0_path

    def run_pretrain(self, d0_path: Path) -> Path:
        pretrain_run = self.project_root / "runs" / self.config.exp_name / "pretrain"
        best_pt = pretrain_run / "best.pt"
        if best_pt.exists():
            print(f"找到已存在的预训练模型检查点: {best_pt}")
            return best_pt

        print(f"\n=== 开始预训练初始策略网络 pi_0 ({self.config.pretrain_epochs} 轮次, batch {self.config.batch_size}) ===")
        update_progress_heartbeat("Pretrain Started", f"Epochs: {self.config.pretrain_epochs}, batch: {self.config.batch_size}")
        best_pt, _ = train_iteration_model(
            cls=self.config.weight_class,
            data_path=d0_path,
            out_dir=pretrain_run,
            epochs=self.config.pretrain_epochs,
            batch=self.config.batch_size,
            lr=self.config.lr,
            seed=self.config.seed,
            device=self.config.device,
        )
        pretrain_model = self.project_root / "models" / f"{self.config.exp_name}_pretrain"
        export_and_verify_model(
            cls=self.config.weight_class,
            weights_path=best_pt,
            out_dir=pretrain_model,
            method="EXP-085: pretrain Micro on multi-seat MCMF D_0",
        )
        update_progress_heartbeat("Pretrain Finished", f"Saved to {pretrain_model.name}")
        return best_pt

    def get_initial_weights(self, d0_path: Path) -> Path:
        if self.config.start_iter == 1:
            return self.run_pretrain(d0_path)
        prev_weights = self.project_root / "runs" / self.config.exp_name / f"k{self.config.start_iter - 1}" / "best.pt"
        if not prev_weights.exists():
            raise FileNotFoundError(f"前置迭代 {self.config.start_iter - 1} 检查点缺失: {prev_weights}")
        return prev_weights

    def load_or_init_telemetry(self, baseline_sigs_count: int) -> dict[str, Any]:
        telemetry_path = self.project_root / "runs" / self.config.exp_name / "dagger_telemetry.json"
        if telemetry_path.exists():
            return json.loads(telemetry_path.read_text(encoding="utf-8"))
        telemetry: dict[str, Any] = {
            "baseline_signatures_d0": baseline_sigs_count,
            "disagreement_rates": {},
            "expansion_factors": {},
            "cumulative_signatures": {},
            "novel_signatures_vs_d0": {},
        }
        return telemetry

    def update_telemetry_record(
        self,
        telemetry: dict[str, Any],
        k: int,
        rollout_stats: dict[str, Any],
        cumulative_sigs_count: int,
        baseline_sigs_count: int,
        novel_vs_d0_count: int,
    ) -> dict[str, Any]:
        telemetry["disagreement_rates"][f"d{k}"] = rollout_stats["disagreement_rate"]
        telemetry["expansion_factors"][f"e{k}"] = round(
            cumulative_sigs_count / max(baseline_sigs_count, 1), 4
        )
        telemetry["cumulative_signatures"][f"k{k}"] = cumulative_sigs_count
        telemetry["novel_signatures_vs_d0"][f"k{k}"] = novel_vs_d0_count
        telemetry[f"k{k}"] = rollout_stats
        return telemetry

    def save_telemetry(self, telemetry: dict[str, Any]) -> Path:
        telemetry_path = self.project_root / "runs" / self.config.exp_name / "dagger_telemetry.json"
        telemetry_path.parent.mkdir(parents=True, exist_ok=True)
        telemetry_path.write_text(json.dumps(telemetry, indent=2) + "\n", encoding="utf-8")
        return telemetry_path

    def run(self) -> None:
        self.validate_environment()

        d0_path = self.get_or_create_d0()
        print("正在提取 D_0 宏观状态签名空间...")
        baseline_sigs = extract_signatures_from_file(d0_path)
        cumulative_sigs = set(baseline_sigs)
        print(f"D_0 宏观特征签名数量: {len(baseline_sigs):,}")

        for prior_k in range(1, self.config.start_iter):
            prior_file = self.project_root / "data" / f"{self.config.exp_name}_iter{prior_k}.jsonl.gz"
            if prior_file.exists():
                cumulative_sigs.update(extract_signatures_from_file(prior_file))

        current_weights = self.get_initial_weights(d0_path)
        telemetry = self.load_or_init_telemetry(len(baseline_sigs))
        spec = classes()[self.config.weight_class]

        for k in range(self.config.start_iter, self.config.end_iter + 1):
            iter_seed = self.config.seed + k * 100
            delta_path = self.project_root / "data" / f"{self.config.exp_name}_iter{k}.jsonl.gz"

            print(f"\n========================================================")
            print(f"=== 开始 EXP-085 DAgger 迭代 {k} / {self.config.end_iter} ===")
            print(f"========================================================")
            update_progress_heartbeat(f"DAgger Iteration {k}/{self.config.end_iter}", "Starting rollout...")

            if not delta_path.exists():
                student_driver = StudentDriver(
                    weights_path=current_weights,
                    spec=spec,
                    device=self.config.device,
                )
                teacher = MCMFTeacher(
                    hill_lead=self.config.hill_lead,
                    max_depth=self.config.max_depth,
                    adaptive_raid=self.config.adaptive_raid,
                    threat_avoidance=self.config.threat_avoidance,
                    focus_target=self.config.focus_target,
                    dynamic_congestion=self.config.dynamic_congestion,
                    congestion_alpha=self.config.congestion_alpha,
                    congestion_beta=self.config.congestion_beta,
                )
                gate = InformationAlignmentGate(teacher=teacher, hill_lead=self.config.hill_lead)

                rollout_stats = rollout_iteration_m3(
                    student_driver=student_driver,
                    gate=gate,
                    out_path=delta_path,
                    seat_turns=self.config.samples_per_iter,
                    k_iter=k,
                    waves=self.config.waves,
                    matches_per_wave=self.config.matches_per_wave,
                    max_turns=self.config.max_turns,
                    seed=iter_seed,
                    maps=self.all_maps_str,
                    baseline_signatures=baseline_sigs,
                )
            else:
                stats_file = delta_path.with_suffix("").with_suffix(".stats.json")
                if stats_file.exists():
                    rollout_stats = json.loads(stats_file.read_text(encoding="utf-8"))
                else:
                    rollout_stats = {"iteration": k, "disagreement_rate": 0.0, "novel_state_ratio": 0.0}

            delta_sigs = extract_signatures_from_file(delta_path)
            cumulative_sigs.update(delta_sigs)

            accum_out = self.project_root / "data" / f"{self.config.exp_name}_k{k}.jsonl.gz"
            parts = [d0_path] + [
                self.project_root / "data" / f"{self.config.exp_name}_iter{i}.jsonl.gz" for i in range(1, k + 1)
            ]
            if not accum_out.exists():
                print(f"--- 聚合全量数据集 D_{k} -> {accum_out.name} ---")
                self.aggregator.merge(
                    parts=parts,
                    out=accum_out,
                    iteration=k,
                    stats_extra={
                        "disagreement_rate": rollout_stats.get("disagreement_rate", 0.0),
                        "novel_state_ratio": rollout_stats.get("novel_state_ratio", 0.0),
                    },
                )

            iter_run_dir = self.project_root / "runs" / self.config.exp_name / f"k{k}"
            iter_best_pt = iter_run_dir / "best.pt"
            if not iter_best_pt.exists():
                print(f"--- 训练策略网络 pi_{k} ({self.config.epochs} 轮次, batch {self.config.batch_size}) ---")
                update_progress_heartbeat(f"DAgger Iteration {k}/{self.config.end_iter}", f"Training pi_{k}...")
                best_weights, _ = train_iteration_model(
                    cls=self.config.weight_class,
                    data_path=accum_out,
                    out_dir=iter_run_dir,
                    epochs=self.config.epochs,
                    batch=self.config.batch_size,
                    lr=self.config.lr,
                    seed=self.config.seed,
                    device=self.config.device,
                )
            else:
                best_weights = iter_best_pt

            iter_model_dir = self.project_root / "models" / f"{self.config.exp_name}_iter{k}"
            if not (iter_model_dir / "model.onnx").exists():
                print(f"--- 导出迭代模型固件 -> {iter_model_dir.name} ---")
                export_and_verify_model(
                    cls=self.config.weight_class,
                    weights_path=best_weights,
                    out_dir=iter_model_dir,
                    method=f"EXP-085: Micro multi-seat DAgger iteration {k}",
                )

            current_weights = best_weights

            self.update_telemetry_record(
                telemetry=telemetry,
                k=k,
                rollout_stats=rollout_stats,
                cumulative_sigs_count=len(cumulative_sigs),
                baseline_sigs_count=len(baseline_sigs),
                novel_vs_d0_count=len(delta_sigs - baseline_sigs),
            )
            self.save_telemetry(telemetry)
            update_progress_heartbeat(
                f"DAgger Iteration {k}/{self.config.end_iter} Complete",
                f"Disagreement rate: {rollout_stats.get('disagreement_rate', 0.0):.4f}\n"
                f"Cumulative signatures: {len(cumulative_sigs):,}"
            )

        print("\n========================================================")
        print("=== 导出 EXP-085 最终模型构件 models/exp085 ===")
        print("========================================================")
        update_progress_heartbeat("Final Model Export", "Exporting models/exp085...")
        final_model_dir = self.project_root / "models" / self.config.exp_name
        said = export_and_verify_model(
            cls=self.config.weight_class,
            weights_path=current_weights,
            out_dir=final_model_dir,
            method="EXP-085: 4-iteration multi-seat MCMF DAgger aggregation + Where memory Micro BC",
        )
        runs_exp_dir = self.project_root / "runs" / self.config.exp_name
        runs_exp_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(current_weights, runs_exp_dir / "best.pt")
        print("EXP-085 全部迭代训练与模型导出完成！平台准入核验通过！")
        update_progress_heartbeat(
            "Model Pipeline Complete",
            f"models/exp085 exported successfully.\n"
            f"- Artifact bytes: {said.get('artifact_bytes')}\n"
            f"- Size metric bytes: {said.get('size_metric_bytes')}\n"
            f"- Infer us max: {said.get('infer_us_max')}"
        )


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="EXP-085 全谱系多席位 MCMF 协同教师与 DAgger 交互聚合训练控制器。")
    ap.add_argument("--start-iter", type=int, default=1, help="起始迭代轮次（默认 1）")
    ap.add_argument("--end-iter", type=int, default=4, help="结束迭代轮次（默认 4）")
    ap.add_argument("--d0-samples", type=int, default=100000, help="D0 专家样本席位步数配额（默认 100,000）")
    ap.add_argument("--samples-per-iter", type=int, default=50000, help="每轮迭代席位步数配额（默认 50,000）")
    ap.add_argument("--pretrain-epochs", type=int, default=5, help="预训练轮次（默认 5）")
    ap.add_argument("--epochs", type=int, default=5, help="每轮迭代训练轮次（默认 5）")
    ap.add_argument("--batch", type=int, default=128, help="批处理大小（默认 128）")
    ap.add_argument("--lr", type=float, default=2e-3, help="初始学习率（默认 2e-3）")
    ap.add_argument("--device", default="cuda", help="计算设备（默认 cuda）")
    ap.add_argument("--seed", type=int, default=42, help="随机种子（默认 42）")
    return ap


def main() -> None:
    args = build_parser().parse_args()
    config = PipelineConfig(
        start_iter=args.start_iter,
        end_iter=args.end_iter,
        d0_samples=args.d0_samples,
        samples_per_iter=args.samples_per_iter,
        pretrain_epochs=args.pretrain_epochs,
        epochs=args.epochs,
        batch_size=args.batch,
        lr=args.lr,
        device=args.device,
        seed=args.seed,
    )
    pipeline = Exp085Pipeline(config)
    pipeline.run()


if __name__ == "__main__":
    main()
