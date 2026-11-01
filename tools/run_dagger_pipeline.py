from __future__ import annotations

import argparse
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

# 若当前 Python 环境未包含虚拟环境依赖，切换至专用虚拟环境执行
_ROOT = Path(__file__).resolve().parents[1]
_VENV_PY = _ROOT / "baselines" / ".venv" / "bin" / "python"
if _VENV_PY.exists() and not sys.executable.startswith(str((_ROOT / "baselines" / ".venv").absolute())):
    os.execv(str(_VENV_PY.absolute()), [str(_VENV_PY.absolute())] + sys.argv)

_BASELINES_SRC = _ROOT / "baselines" / "src"
if str(_BASELINES_SRC.resolve()) not in sys.path:
    sys.path.insert(0, str(_BASELINES_SRC.resolve()))

import numpy as np
import torch

from tb_baselines.dagger import (
    DatasetAggregator,
    InformationAlignmentGate,
    STRIDE,
    StateSignature,
    StateTracker,
    StudentDriver,
    rollout_iteration,
)
from tb_baselines.export import classes, export as export_model, main as export_main, verdict
from tb_baselines.teacher import Teacher
from tb_baselines.train import bc

EXPECTED_DIGEST = "sha256:819166e7918117a1cbbd67dc042e7368a076a2dd45cea5122683983912d591e1"
EXPECTED_HEX_DIGEST = "819166e7918117a1cbbd67dc042e7368a076a2dd45cea5122683983912d591e1"


@dataclass(frozen=True)
class PipelineConfig:
    route: str
    weight_class: str
    start_iter: int
    end_iter: int
    pretrain_epochs: int
    epochs: int
    batch_size: int
    lr: float
    device: str | None
    seed: int
    samples_per_iter: int | None
    hill_lead: int = 6
    max_depth: int = 32
    waves: int = 4
    matches_per_wave: int = 16
    max_turns: int = 300
    maps: str | None = None
    project_root: Path = _ROOT

    @property
    def exp_name(self) -> str:
        if self.route == "route_a":
            return "exp031"
        return "exp033"


def get_quota_samples(route: str, k: int, explicit: int | None = None) -> int:
    if explicit is not None:
        return explicit
    if route == "route_a":
        if k in (3, 4, 8):
            return 83_334
        return 83_333
    return 83_333


def get_expected_cumulative_rows(route: str, k: int, explicit: int | None = None) -> int:
    if explicit is not None:
        if route == "route_a":
            return 500_000 + (k - 3) * explicit
        return 250_000 + k * explicit

    if route == "route_a":
        mapping = {
            1: 333_333,
            2: 416_666,
            3: 500_000,
            4: 583_334,
            5: 666_667,
            6: 750_000,
            7: 833_333,
            8: 916_667,
        }
        if k in mapping:
            return mapping[k]
        return 500_000 + (k - 3) * 83_333

    mapping_b = {
        1: 333_333,
        2: 416_666,
        3: 499_999,
        4: 583_332,
        5: 666_665,
        6: 749_998,
        7: 833_331,
        8: 916_664,
    }
    if k in mapping_b:
        return mapping_b[k]
    return 250_000 + k * 83_333


def verify_engine_digest(wasm_path: Path) -> None:
    if not wasm_path.exists():
        raise FileNotFoundError(f"未找到 Wasm 游戏引擎文件: {wasm_path}")
    with open(wasm_path, "rb") as f:
        actual_hash = hashlib.sha256(f.read()).hexdigest()
    if actual_hash != EXPECTED_HEX_DIGEST:
        raise RuntimeError(
            f"Wasm 引擎散列摘要不匹配！期望 {EXPECTED_HEX_DIGEST}，实际为 {actual_hash}"
        )


def verify_file_row_count(file_path: Path, expected: int) -> int:
    if not file_path.exists():
        raise FileNotFoundError(f"待校验行数的数据文件不存在: {file_path}")
    count = 0
    with gzip.open(file_path, "rt", encoding="utf-8") as f:
        f.readline()
        for line in f:
            if line.endswith("\n"):
                count += 1
    if count != expected:
        raise RuntimeError(
            f"数据文件 {file_path} 实际样本行数 {count} 与期望行数 {expected} 不匹配！"
        )
    return count


def check_prerequisites(route: str, cls: str, start_iter: int, project_root: Path) -> None:
    wasm_path = project_root / "dist" / "tb-ants.wasm"
    verify_engine_digest(wasm_path)

    games_toml = project_root / "baselines" / "games.toml"
    if not games_toml.exists():
        raise FileNotFoundError(f"未找到 TinyBrains 注册表文件: {games_toml}")
    os.environ["TINYBRAINS_REGISTRY"] = str(games_toml.resolve())

    if shutil.which("tinybrains") is None:
        raise RuntimeError("环境变量 PATH 中未找到 tinybrains 可执行文件")

    d0_path = project_root / "data" / "exp011.jsonl.gz"
    if not d0_path.exists():
        raise FileNotFoundError(f"未找到基础专家数据集文件: {d0_path}")

    if route == "route_a":
        if start_iter < 4:
            raise ValueError(f"路线 A 起始轮次 start_iter 必须大于等于 4，收到: {start_iter}")
        for i in range(1, start_iter):
            part_file = project_root / "data" / f"exp031_iter{i}.jsonl.gz"
            if not part_file.exists():
                raise FileNotFoundError(f"路线 A 前置增量数据文件缺失: {part_file}")
        if start_iter == 4:
            base_weights = project_root / "runs" / "exp031" / "best.pt"
            if not base_weights.exists():
                raise FileNotFoundError(f"路线 A 初始模型检查点缺失: {base_weights}")
        else:
            prev_weights = project_root / "runs" / "exp031" / f"k{start_iter - 1}" / "best.pt"
            if not prev_weights.exists():
                raise FileNotFoundError(f"路线 A 迭代 {start_iter - 1} 检查点缺失: {prev_weights}")
    else:
        if start_iter < 1:
            raise ValueError(f"路线 B 起始轮次 start_iter 必须大于等于 1，收到: {start_iter}")
        if start_iter > 1:
            for i in range(1, start_iter):
                part_file = project_root / "data" / f"exp033_iter{i}.jsonl.gz"
                if not part_file.exists():
                    raise FileNotFoundError(f"路线 B 前置增量数据文件缺失: {part_file}")
            prev_weights = project_root / "runs" / "exp033" / f"k{start_iter - 1}" / "best.pt"
            if not prev_weights.exists():
                raise FileNotFoundError(f"路线 B 迭代 {start_iter - 1} 检查点缺失: {prev_weights}")


def train_iteration_model(
    cls: str,
    data_path: Path,
    out_dir: Path,
    epochs: int = 5,
    batch: int = 256,
    lr: float = 2e-3,
    seed: int = 0,
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
        elif dev_str.startswith("cuda:"):
            cuda_id = dev_str.split(":", 1)[1]
            os.environ["CUDA_VISIBLE_DEVICES"] = cuda_id

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
        raise FileNotFoundError(f"行为克隆训练未生成最优模型检查点: {best_pt}")
    if not history_json.exists():
        raise FileNotFoundError(f"行为克隆训练未生成指标历史记录: {history_json}")

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
    if cls == "nano":
        nano_limit = int(12.8 * 1024)
        if size_bytes >= nano_limit:
            raise AssertionError(
                f"Nano 构件指标体积 {size_bytes} 字节超出 12.8 KB（{nano_limit} 字节）上限！"
            )
    elif cls == "micro":
        micro_limit = 128 * 1024
        if size_bytes >= micro_limit:
            raise AssertionError(
                f"Micro 构件指标体积 {size_bytes} 字节超出 128 KB（{micro_limit} 字节）上限！"
            )
    else:
        raise ValueError(f"未知的规格等级: {cls}，仅支持 nano 或 micro")

    infer_us = said.get("infer_us_max", 0)
    if infer_us > 700_000:
        raise AssertionError(
            f"单回合推理峰值时延 {infer_us} 微秒超出 700 毫秒（700000 微秒）上限！"
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
        raise FileNotFoundError(f"导出的度量指标文件缺失: {metrics_path}")

    said = verdict(model_path.resolve(), manifest_path.resolve())
    validate_admission_verdict(said, cls=cls)
    return said


def extract_signatures_from_file(file_path: Path) -> set[tuple]:
    if not file_path.exists():
        raise FileNotFoundError(f"待提取状态指纹的数据文件不存在: {file_path}")
    sigs: set[tuple] = set()
    with gzip.open(file_path, "rt", encoding="utf-8") as f:
        header_line = f.readline()
        if header_line and not header_line.endswith("\n"):
            raise ValueError(f"数据文件 {file_path} 头部数据损坏（未以换行符结尾）！")
        for line_no, line in enumerate(f, start=2):
            if not line.endswith("\n"):
                raise ValueError(
                    f"数据文件 {file_path} 第 {line_no} 行格式异常（未以换行符结尾），数据流可能已损坏或截断！"
                )
            row = json.loads(line)
            sigs.add(StateSignature.compute(row["o"], row.get("m")))
    return sigs


class DaggerPipeline:
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.project_root = config.project_root
        self.aggregator = DatasetAggregator(stride=STRIDE)

    def validate_environment(self) -> None:
        check_prerequisites(
            route=self.config.route,
            cls=self.config.weight_class,
            start_iter=self.config.start_iter,
            project_root=self.project_root,
        )

    def load_baseline_signatures(self) -> set[tuple]:
        telemetry_path = self.project_root / "runs" / self.config.exp_name / "dagger_telemetry.json"
        d0_path = self.project_root / "data" / "exp011.jsonl.gz"
        sigs = extract_signatures_from_file(d0_path)
        return sigs

    def run_pretrain(self) -> Path:
        pretrain_run = self.project_root / "runs" / "exp033" / "pretrain"
        d0_path = self.project_root / "data" / "exp011.jsonl.gz"
        best_pt, _ = train_iteration_model(
            cls="micro",
            data_path=d0_path,
            out_dir=pretrain_run,
            epochs=self.config.pretrain_epochs,
            batch=self.config.batch_size,
            lr=self.config.lr,
            seed=self.config.seed,
            device=self.config.device,
        )
        pretrain_model = self.project_root / "models" / "exp033_pretrain"
        export_and_verify_model(
            cls="micro",
            weights_path=best_pt,
            out_dir=pretrain_model,
            method="bc_pretrain",
        )
        return best_pt

    def get_initial_weights(self) -> Path:
        if self.config.route == "route_a":
            if self.config.start_iter == 4:
                return self.project_root / "runs" / "exp031" / "best.pt"
            return self.project_root / "runs" / "exp031" / f"k{self.config.start_iter - 1}" / "best.pt"

        if self.config.start_iter == 1:
            pretrain_weights = self.project_root / "runs" / "exp033" / "pretrain" / "best.pt"
            if pretrain_weights.exists():
                return pretrain_weights
            return self.run_pretrain()
        return self.project_root / "runs" / "exp033" / f"k{self.config.start_iter - 1}" / "best.pt"

    def load_or_init_telemetry(self, baseline_sigs_count: int) -> dict[str, Any]:
        telemetry_path = self.project_root / "runs" / self.config.exp_name / "dagger_telemetry.json"
        if telemetry_path.exists():
            return json.loads(telemetry_path.read_text(encoding="utf-8"))
        telemetry = {
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

        baseline_sigs = self.load_baseline_signatures()
        cumulative_sigs = set(baseline_sigs)

        if self.config.route == "route_a":
            for prior_k in range(1, self.config.start_iter):
                prior_file = self.project_root / "data" / f"exp031_iter{prior_k}.jsonl.gz"
                if prior_file.exists():
                    cumulative_sigs.update(extract_signatures_from_file(prior_file))
        else:
            for prior_k in range(1, self.config.start_iter):
                prior_file = self.project_root / "data" / f"exp033_iter{prior_k}.jsonl.gz"
                if prior_file.exists():
                    cumulative_sigs.update(extract_signatures_from_file(prior_file))

        current_weights = self.get_initial_weights()
        telemetry = self.load_or_init_telemetry(len(baseline_sigs))

        d0_path = self.project_root / "data" / "exp011.jsonl.gz"
        spec = classes()[self.config.weight_class]

        for k in range(self.config.start_iter, self.config.end_iter + 1):
            samples_quota = get_quota_samples(self.config.route, k, explicit=self.config.samples_per_iter)
            iter_seed = self.config.seed + k * 100
            delta_path = self.project_root / "data" / f"{self.config.exp_name}_iter{k}.jsonl.gz"

            student_driver = StudentDriver(
                weights_path=current_weights,
                spec=spec,
                device=self.config.device,
            )
            teacher = Teacher(hill_lead=self.config.hill_lead, max_depth=self.config.max_depth)
            gate = InformationAlignmentGate(teacher=teacher, hill_lead=self.config.hill_lead)

            rollout_stats = rollout_iteration(
                student_driver=student_driver,
                gate=gate,
                out_path=delta_path,
                seat_turns=samples_quota,
                k_iter=k,
                waves=self.config.waves,
                matches_per_wave=self.config.matches_per_wave,
                max_turns=self.config.max_turns,
                seed=iter_seed,
                maps=self.config.maps,
                baseline_signatures=baseline_sigs,
            )

            verify_file_row_count(delta_path, samples_quota)

            delta_sigs = extract_signatures_from_file(delta_path)
            cumulative_sigs.update(delta_sigs)

            if self.config.route == "route_a":
                parts = [d0_path] + [
                    self.project_root / "data" / f"exp031_iter{i}.jsonl.gz" for i in range(1, k + 1)
                ]
            else:
                parts = [d0_path] + [
                    self.project_root / "data" / f"exp033_iter{i}.jsonl.gz" for i in range(1, k + 1)
                ]

            accum_out = self.project_root / "data" / f"{self.config.exp_name}_k{k}.jsonl.gz"
            self.aggregator.merge(
                parts=parts,
                out=accum_out,
                iteration=k,
                stats_extra={
                    "disagreement_rate": rollout_stats["disagreement_rate"],
                    "novel_state_ratio": rollout_stats["novel_state_ratio"],
                    "ant_decisions": rollout_stats["ant_decisions"],
                    "disagreed_ants": rollout_stats["disagreed_ants"],
                },
            )

            expected_accum_rows = get_expected_cumulative_rows(
                self.config.route, k, explicit=self.config.samples_per_iter
            )
            verify_file_row_count(accum_out, expected_accum_rows)

            iter_run_dir = self.project_root / "runs" / self.config.exp_name / f"k{k}"
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

            iter_model_dir = self.project_root / "models" / f"{self.config.exp_name}_iter{k}"
            export_and_verify_model(
                cls=self.config.weight_class,
                weights_path=best_weights,
                out_dir=iter_model_dir,
                method="dagger",
            )
            for artifact_name in ("model.onnx", "manifest.json", "metrics.json", "card.md"):
                src_art = iter_model_dir / artifact_name
                if src_art.exists():
                    shutil.copy2(src_art, iter_run_dir / artifact_name)

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

        final_model_dir = self.project_root / "models" / self.config.exp_name
        export_and_verify_model(
            cls=self.config.weight_class,
            weights_path=current_weights,
            out_dir=final_model_dir,
            method="dagger_final",
        )
        shutil.copy2(current_weights, self.project_root / "runs" / self.config.exp_name / "best.pt")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="TinyBrains Ants DAgger 双轨数据集聚合闭环训练流水线控制器。")
    ap.add_argument("--route", choices=["route_a", "route_b"], required=True,
                    help="执行路线：route_a（Nano 持续聚合轨）或 route_b（Micro 独立聚合轨）")
    ap.add_argument("--class", dest="weight_class", choices=["nano", "micro"], default=None,
                    help="网络规格等级：nano 或 micro（默认依据路线自动匹配）")
    ap.add_argument("--start-iter", type=int, default=None,
                    help="起始迭代轮次（路线 A 默认 4，路线 B 默认 1）")
    ap.add_argument("--end-iter", type=int, default=8,
                    help="终止迭代轮次（默认 8）")
    ap.add_argument("--pretrain-epochs", type=int, default=5,
                    help="路线 B 专家数据预训练轮数（默认 5）")
    ap.add_argument("--epochs", type=int, default=5,
                    help="每轮聚合训练轮数（默认 5）")
    ap.add_argument("--batch", type=int, default=256,
                    help="训练批处理尺寸（默认 256）")
    ap.add_argument("--lr", type=float, default=2e-3,
                    help="初始学习率（默认 2e-3）")
    ap.add_argument("--device", default=None,
                    help="计算设备标识（'cpu', 'cuda', 'cuda:0' 等）")
    ap.add_argument("--seed", type=int, default=1,
                    help="随机种子（默认 1）")
    ap.add_argument("--samples-per-iter", type=int, default=None,
                    help="每轮增量采样席位步数配额（缺省时严格按路线与轮次配额执行）")
    ap.add_argument("--hill-lead", type=int, default=6,
                    help="启发式教师模型起步领先步数（默认 6）")
    ap.add_argument("--max-depth", type=int, default=32,
                    help="启发式教师模型泛洪最大深度（默认 32）")
    ap.add_argument("--waves", type=int, default=4,
                    help="环境推演波次数量（默认 4）")
    ap.add_argument("--matches-per-wave", type=int, default=16,
                    help="每波次对局数量（默认 16）")
    ap.add_argument("--max-turns", type=int, default=300,
                    help="每对局最大回合数（默认 300）")
    ap.add_argument("--maps", default=None,
                    help="推演地图标识或目录（默认使用发布包基础棋盘）")
    return ap


def validate_cli_arguments(args: argparse.Namespace) -> argparse.Namespace:
    if args.route not in ("route_a", "route_b"):
        raise ValueError(f"未知路线选项: {args.route}，必须为 route_a 或 route_b")

    if args.route == "route_a":
        if args.weight_class is not None and args.weight_class != "nano":
            raise ValueError(f"路线 A 仅允许使用 nano 规格等级，收到非法的规格: {args.weight_class}")
        args.weight_class = "nano"
        if args.start_iter is None:
            args.start_iter = 4
        elif args.start_iter < 4:
            raise ValueError(f"路线 A 起始轮次 start_iter 必须大于等于 4，收到: {args.start_iter}")
    else:
        if args.weight_class is not None and args.weight_class != "micro":
            raise ValueError(f"路线 B 仅允许使用 micro 规格等级，收到非法的规格: {args.weight_class}")
        args.weight_class = "micro"
        if args.start_iter is None:
            args.start_iter = 1
        elif args.start_iter < 1:
            raise ValueError(f"路线 B 起始轮次 start_iter 必须大于等于 1，收到: {args.start_iter}")
    if args.end_iter < args.start_iter:
        raise ValueError(
            f"终止轮次 end_iter ({args.end_iter}) 必须大于等于起始轮次 start_iter ({args.start_iter})"
        )
    if args.samples_per_iter is not None and args.samples_per_iter <= 0:
        raise ValueError(f"采样步数配额 samples_per_iter 必须为正整数，收到: {args.samples_per_iter}")
    if args.epochs <= 0:
        raise ValueError(f"训练轮数 epochs 必须为正整数，收到: {args.epochs}")
    if args.batch <= 0:
        raise ValueError(f"批处理尺寸 batch 必须为正整数，收到: {args.batch}")
    if args.lr <= 0:
        raise ValueError(f"学习率 lr 必须大于 0，收到: {args.lr}")

    return args


def main(argv: list[str] | None = None) -> None:
    ap = build_parser()
    args = ap.parse_args(argv)
    args = validate_cli_arguments(args)

    config = PipelineConfig(
        route=args.route,
        weight_class=args.weight_class,
        start_iter=args.start_iter,
        end_iter=args.end_iter,
        pretrain_epochs=args.pretrain_epochs,
        epochs=args.epochs,
        batch_size=args.batch,
        lr=args.lr,
        device=args.device,
        seed=args.seed,
        samples_per_iter=args.samples_per_iter,
        hill_lead=args.hill_lead,
        max_depth=args.max_depth,
        waves=args.waves,
        matches_per_wave=args.matches_per_wave,
        max_turns=args.max_turns,
        maps=args.maps,
    )

    pipeline = DaggerPipeline(config)
    pipeline.run()


if __name__ == "__main__":
    main()
