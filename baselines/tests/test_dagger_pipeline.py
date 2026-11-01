from __future__ import annotations

import gzip
import json
import os
from pathlib import Path
import re
import shutil
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pytest
import torch

from tb_baselines.dagger import (
    DatasetAggregator,
    InformationAlignmentGate,
    InformationAlignmentViolation,
    STRIDE,
    StateSignature,
)
from tb_baselines.teacher import Teacher
from tools.run_dagger_pipeline import (
    EXPECTED_DIGEST,
    EXPECTED_HEX_DIGEST,
    DaggerPipeline,
    PipelineConfig,
    build_parser,
    check_prerequisites,
    export_and_verify_model,
    extract_signatures_from_file,
    get_expected_cumulative_rows,
    get_quota_samples,
    train_iteration_model,
    validate_admission_verdict,
    validate_cli_arguments,
    verify_engine_digest,
    verify_file_row_count,
)

SCRATCH_DIR = ROOT / "runs" / "test_scratch"


@pytest.fixture(autouse=True)
def scratch_environment():
    if SCRATCH_DIR.exists():
        shutil.rmtree(SCRATCH_DIR, ignore_errors=True)
    SCRATCH_DIR.mkdir(parents=True, exist_ok=True)
    os.environ["TINYBRAINS_REGISTRY"] = str((ROOT / "baselines" / "games.toml").resolve())
    try:
        yield SCRATCH_DIR
    finally:
        if SCRATCH_DIR.exists():
            shutil.rmtree(SCRATCH_DIR, ignore_errors=True)


def create_synthetic_dataset(
    out_path: Path,
    num_rows: int = 20,
    stride_offset: int = 0,
    iteration: int = 1,
) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    header = {
        "kind": "dagger-rollout" if iteration > 1 else "teacher-rollout",
        "engine_digest": EXPECTED_DIGEST,
        "evaluator": "datalogic 5.7",
        "teacher_remembers": True,
        "memory_mode": "where",
        "memory": ["food_seen", "hill_foe_seen"],
        "dagger_iteration": iteration,
        "episode_offset": stride_offset,
    }
    with gzip.open(out_path, "wt", encoding="utf-8") as f:
        f.write(json.dumps(header) + "\n")
        for i in range(num_rows):
            row = {
                "o": {
                    "size": [24, 24],
                    "mine": [[5, 5]],
                    "foes": [],
                    "food": [[6, 5]],
                    "hills": [],
                    "water": {"rle": [0, 576]},
                    "vis": {"rle": [0, 576]},
                    "ids": [i + 1],
                },
                "k": [i // 2 + stride_offset, i % 2, 1],
                "m": {"food_seen": [[6, 5]], "hill_foe_seen": []},
                "a": "N" if i % 2 == 0 else "E",
            }
            f.write(json.dumps(row, separators=(",", ":")) + "\n")
    return out_path


# ==============================================================================
# 1. 无特权门禁测试集
# ==============================================================================


class TestInformationAlignmentGate:
    def test_gate_accepts_valid_local_observation_and_memory(self):
        gate = InformationAlignmentGate(Teacher(hill_lead=6, max_depth=32))
        obs = {
            "size": [24, 24],
            "mine": [[5, 5], [10, 10]],
            "foes": [],
            "food": [[6, 5]],
            "hills": [],
            "water": {"rle": [0, 576]},
            "vis": {"rle": [0, 576]},
            "ids": [1, 2],
        }
        mem = {"food_seen": [[6, 5]], "hill_foe_seen": []}
        moves, order_str = gate.label(obs, mem)
        assert len(moves) == 2
        assert len(order_str) == 2
        assert all(0 <= m <= 4 for m in moves)
        assert all(c in ("N", "E", "S", "W", "-") for c in order_str)

    @pytest.mark.parametrize("priv_key", [
        "score", "turn_scores", "true_hills", "global_map", "actor_critic_extra", "secret"
    ])
    def test_gate_rejects_privileged_observation_keys(self, priv_key: str):
        gate = InformationAlignmentGate(Teacher(hill_lead=6, max_depth=32))
        obs = {
            "size": [24, 24],
            "mine": [[5, 5]],
            "foes": [],
            "food": [],
            "hills": [],
            "water": {"rle": [0, 576]},
            "vis": {"rle": [0, 576]},
            "ids": [1],
            priv_key: "forbidden_privileged_data",
        }
        mem = {"food_seen": [], "hill_foe_seen": []}
        with pytest.raises(InformationAlignmentViolation):
            gate.label(obs, mem)

    @pytest.mark.parametrize("unauth_mem_key", [
        "enemy_base", "secret_channel", "global_water_map"
    ])
    def test_gate_rejects_unauthorized_memory_keys(self, unauth_mem_key: str):
        gate = InformationAlignmentGate(Teacher(hill_lead=6, max_depth=32))
        obs = {
            "size": [24, 24],
            "mine": [[5, 5]],
            "foes": [],
            "food": [],
            "hills": [],
            "water": {"rle": [0, 576]},
            "vis": {"rle": [0, 576]},
            "ids": [1],
        }
        mem = {"food_seen": [], "hill_foe_seen": [], unauth_mem_key: [[1, 1]]}
        with pytest.raises(InformationAlignmentViolation):
            gate.label(obs, mem)

    def test_gate_handles_zero_ants_gracefully(self):
        gate = InformationAlignmentGate(Teacher(hill_lead=6, max_depth=32))
        obs = {
            "size": [24, 24],
            "mine": [],
            "foes": [],
            "food": [],
            "hills": [],
            "water": {"rle": [0, 576]},
            "vis": {"rle": [0, 576]},
            "ids": [],
        }
        mem = {"food_seen": [], "hill_foe_seen": []}
        moves, order_str = gate.label(obs, mem)
        assert len(moves) == 0
        assert order_str == ""

    def test_gate_teacher_configuration_parameters(self):
        teacher = Teacher(hill_lead=12, max_depth=16)
        gate = InformationAlignmentGate(teacher=teacher, hill_lead=12)
        assert gate.teacher.hill_lead == 12
        assert gate.teacher.max_depth == 16


# ==============================================================================
# 2. 参数解析与快速失败测试集
# ==============================================================================


class TestPipelineArgumentParsing:
    def test_route_a_default_and_custom_arguments(self):
        ap = build_parser()
        args = ap.parse_args(["--route", "route_a"])
        args = validate_cli_arguments(args)
        assert args.route == "route_a"
        assert args.weight_class == "nano"
        assert args.start_iter == 4
        assert args.end_iter == 8

    def test_route_b_default_and_custom_arguments(self):
        ap = build_parser()
        args = ap.parse_args(["--route", "route_b", "--pretrain-epochs", "6"])
        args = validate_cli_arguments(args)
        assert args.route == "route_b"
        assert args.weight_class == "micro"
        assert args.start_iter == 1
        assert args.end_iter == 8
        assert args.pretrain_epochs == 6

    def test_fast_fail_on_route_class_mismatch(self):
        ap = build_parser()
        args_a = ap.parse_args(["--route", "route_a", "--class", "micro"])
        with pytest.raises(ValueError, match="仅允许使用 nano"):
            validate_cli_arguments(args_a)

        args_b = ap.parse_args(["--route", "route_b", "--class", "nano"])
        with pytest.raises(ValueError, match="仅允许使用 micro"):
            validate_cli_arguments(args_b)

    def test_fast_fail_on_invalid_iteration_ranges(self):
        ap = build_parser()
        args_zero = ap.parse_args(["--route", "route_b", "--start-iter", "0"])
        with pytest.raises(ValueError, match="必须大于等于 1"):
            validate_cli_arguments(args_zero)

        args_inverted = ap.parse_args(["--route", "route_a", "--start-iter", "6", "--end-iter", "5"])
        with pytest.raises(ValueError, match="必须大于等于起始轮次"):
            validate_cli_arguments(args_inverted)

    def test_fast_fail_on_route_a_start_iter_less_than_four(self):
        ap = build_parser()
        for invalid_iter in (1, 2, 3):
            args = ap.parse_args(["--route", "route_a", "--start-iter", str(invalid_iter)])
            with pytest.raises(ValueError, match="路线 A 起始轮次 start_iter 必须大于等于 4"):
                validate_cli_arguments(args)

    def test_fast_fail_on_invalid_hyperparameters(self):
        ap = build_parser()
        with pytest.raises(ValueError, match="必须为正整数"):
            validate_cli_arguments(ap.parse_args(["--route", "route_a", "--samples-per-iter", "-1"]))

        with pytest.raises(ValueError, match="必须为正整数"):
            validate_cli_arguments(ap.parse_args(["--route", "route_a", "--epochs", "0"]))

        with pytest.raises(ValueError, match="必须为正整数"):
            validate_cli_arguments(ap.parse_args(["--route", "route_a", "--batch", "-256"]))

        with pytest.raises(ValueError, match="必须大于 0"):
            validate_cli_arguments(ap.parse_args(["--route", "route_a", "--lr", "0"]))

    def test_quota_calculations_route_a_and_b(self):
        assert get_quota_samples("route_a", 4) == 83_334
        assert get_quota_samples("route_a", 5) == 83_333
        assert get_quota_samples("route_a", 6) == 83_333
        assert get_quota_samples("route_a", 7) == 83_333
        assert get_quota_samples("route_a", 8) == 83_334
        assert get_expected_cumulative_rows("route_a", 4) == 583_334
        assert get_expected_cumulative_rows("route_a", 8) == 916_667

        for k in range(1, 9):
            assert get_quota_samples("route_b", k) == 83_333
        assert get_expected_cumulative_rows("route_b", 1) == 333_333
        assert get_expected_cumulative_rows("route_b", 8) == 916_664

    def test_engine_digest_verification_fast_fail(self):
        wasm_path = ROOT / "dist" / "tb-ants.wasm"
        verify_engine_digest(wasm_path)

        fake_wasm = SCRATCH_DIR / "fake_engine.wasm"
        fake_wasm.write_bytes(b"corrupted_wasm_content")
        with pytest.raises(RuntimeError, match="Wasm 引擎散列摘要不匹配"):
            verify_engine_digest(fake_wasm)


# ==============================================================================
# 3. 前置条件检查测试集
# ==============================================================================


class TestPrerequisitesChecking:
    def _setup_mock_env(self, env_root: Path) -> None:
        dist_dir = env_root / "dist"
        dist_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / "dist" / "tb-ants.wasm", dist_dir / "tb-ants.wasm")
        games_toml = env_root / "baselines" / "games.toml"
        games_toml.parent.mkdir(parents=True, exist_ok=True)
        games_toml.write_text("[registry]\n", encoding="utf-8")
        create_synthetic_dataset(env_root / "data" / "exp011.jsonl.gz", num_rows=2)

    def test_prerequisites_route_a_start_iter_less_than_four_fails(self):
        env_root = SCRATCH_DIR / "prereq_a_invalid_start"
        self._setup_mock_env(env_root)
        with pytest.raises(ValueError, match="路线 A 起始轮次 start_iter 必须大于等于 4"):
            check_prerequisites("route_a", "nano", 3, env_root)

    def test_prerequisites_route_a_missing_initial_checkpoint(self):
        env_root = SCRATCH_DIR / "prereq_a_initial"
        self._setup_mock_env(env_root)
        for i in range(1, 4):
            create_synthetic_dataset(env_root / "data" / f"exp031_iter{i}.jsonl.gz", num_rows=2)

        with pytest.raises(FileNotFoundError, match="路线 A 初始模型检查点缺失"):
            check_prerequisites("route_a", "nano", 4, env_root)

    def test_prerequisites_route_a_resume_missing_intermediate_data(self):
        env_root = SCRATCH_DIR / "prereq_a_intermediate"
        self._setup_mock_env(env_root)
        for i in [1, 2, 3, 5]:
            create_synthetic_dataset(env_root / "data" / f"exp031_iter{i}.jsonl.gz", num_rows=2)
        (env_root / "runs" / "exp031" / "k5").mkdir(parents=True, exist_ok=True)
        (env_root / "runs" / "exp031" / "k5" / "best.pt").write_bytes(b"dummy_weights")

        with pytest.raises(FileNotFoundError, match="路线 A 前置增量数据文件缺失"):
            check_prerequisites("route_a", "nano", 6, env_root)

    def test_prerequisites_route_a_resume_missing_previous_checkpoint(self):
        env_root = SCRATCH_DIR / "prereq_a_prev_ckpt"
        self._setup_mock_env(env_root)
        for i in range(1, 6):
            create_synthetic_dataset(env_root / "data" / f"exp031_iter{i}.jsonl.gz", num_rows=2)

        with pytest.raises(FileNotFoundError, match="路线 A 迭代 5 检查点缺失"):
            check_prerequisites("route_a", "nano", 6, env_root)

    def test_prerequisites_route_b_start_iter_less_than_one_fails(self):
        env_root = SCRATCH_DIR / "prereq_b_invalid_start"
        self._setup_mock_env(env_root)
        with pytest.raises(ValueError, match="路线 B 起始轮次 start_iter 必须大于等于 1"):
            check_prerequisites("route_b", "micro", 0, env_root)

    def test_prerequisites_route_b_resume_missing_checkpoint_fails_fast(self):
        env_root = SCRATCH_DIR / "prereq_b_resume"
        self._setup_mock_env(env_root)
        create_synthetic_dataset(env_root / "data" / "exp033_iter1.jsonl.gz", num_rows=2)
        (env_root / "runs" / "exp033" / "pretrain").mkdir(parents=True, exist_ok=True)
        (env_root / "runs" / "exp033" / "pretrain" / "best.pt").write_bytes(b"dummy_pretrain")

        with pytest.raises(FileNotFoundError, match="路线 B 迭代 1 检查点缺失"):
            check_prerequisites("route_b", "micro", 2, env_root)

    def test_prerequisites_route_b_resume_missing_intermediate_data(self):
        env_root = SCRATCH_DIR / "prereq_b_intermediate"
        self._setup_mock_env(env_root)
        create_synthetic_dataset(env_root / "data" / "exp033_iter1.jsonl.gz", num_rows=2)
        (env_root / "runs" / "exp033" / "k2").mkdir(parents=True, exist_ok=True)
        (env_root / "runs" / "exp033" / "k2" / "best.pt").write_bytes(b"dummy_k2")

        with pytest.raises(FileNotFoundError, match="路线 B 前置增量数据文件缺失"):
            check_prerequisites("route_b", "micro", 3, env_root)


# ==============================================================================
# 3. 数据累加与步长偏移测试集
# ==============================================================================


class TestDatasetAggregationAndOffsets:
    def test_dataset_aggregation_streaming_and_offsets(self):
        part_paths = []
        raw_rows_by_part = []

        for p_idx in range(3):
            p_file = SCRATCH_DIR / f"test_part_{p_idx}.jsonl.gz"
            create_synthetic_dataset(p_file, num_rows=4, stride_offset=0, iteration=p_idx + 1)
            part_paths.append(p_file)

            with gzip.open(p_file, "rt", encoding="utf-8") as f:
                f.readline()
                raw_rows_by_part.append([json.loads(line) for line in f])

        out_path = SCRATCH_DIR / "test_merged.jsonl.gz"
        aggregator = DatasetAggregator(stride=STRIDE)
        stats = aggregator.merge(part_paths, out_path, iteration=2)

        assert stats["rows"] == 12
        assert stats["parts"] == 3

        with gzip.open(out_path, "rt", encoding="utf-8") as f:
            merged_header = json.loads(f.readline())
            merged_rows = [json.loads(line) for line in f]

        assert len(merged_header["parts"]) == 3
        for i, p_info in enumerate(merged_header["parts"]):
            assert p_info["episode_offset"] == i * STRIDE
            assert p_info["file"] == part_paths[i].name

        for i in range(3):
            expected_offset = i * STRIDE
            slice_rows = merged_rows[i * 4 : (i + 1) * 4]
            for r_idx, r in enumerate(slice_rows):
                raw_ep = raw_rows_by_part[i][r_idx]["k"][0]
                assert r["k"][0] == raw_ep + expected_offset

    def test_serialization_compatibility_with_size_regex(self):
        test_file = SCRATCH_DIR / "regex_check.jsonl.gz"
        create_synthetic_dataset(test_file, num_rows=6)

        pattern = re.compile(rb'"size":\[(\d+),(\d+)\]')
        with gzip.open(test_file, "rb") as f:
            f.readline()
            for line in f:
                assert pattern.search(line) is not None

    def test_verify_file_row_count_assertion_and_fast_fail(self):
        test_file = SCRATCH_DIR / "row_count_check.jsonl.gz"
        create_synthetic_dataset(test_file, num_rows=10)

        assert verify_file_row_count(test_file, 10) == 10
        with pytest.raises(RuntimeError, match="实际样本行数 10 与期望行数 12 不匹配"):
            verify_file_row_count(test_file, 12)

    def test_extract_signatures_truncated_line_fast_fail(self):
        test_file = SCRATCH_DIR / "truncated_check.jsonl.gz"
        test_file.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(test_file, "wt", encoding="utf-8") as f:
            f.write(json.dumps({"kind": "teacher-rollout"}) + "\n")
            f.write('{"o":{"size":[24,24],"mine":[[5,5]],"foes":[],"food":[],"hills":[],"water":{"rle":[0,576]},"vis":{"rle":[0,576]},"ids":[1]},"k":[0,0,1],"a":"N"}')
        with pytest.raises(ValueError, match="未以换行符结尾"):
            extract_signatures_from_file(test_file)


# ==============================================================================
# 4. 训练集成冒烟测试集
# ==============================================================================


class TestBCTrainingIntegration:
    def test_fast_bc_training_execution(self):
        synth_data = SCRATCH_DIR / "train_synth.jsonl.gz"
        create_synthetic_dataset(synth_data, num_rows=24)

        train_out = SCRATCH_DIR / "train_output"
        best_pt, history = train_iteration_model(
            cls="nano",
            data_path=synth_data,
            out_dir=train_out,
            epochs=1,
            batch=8,
            lr=1e-3,
            seed=42,
            device="cpu",
        )

        assert best_pt.exists()
        assert best_pt.stat().st_size > 0
        assert (train_out / "history.json").exists()
        assert "epochs" in history
        assert len(history["epochs"]) == 1
        assert "train_loss" in history["epochs"][0]
        assert "val_loss" in history["epochs"][0]


# ==============================================================================
# 5. 导出与平台准入断言测试集
# ==============================================================================


class TestModelExportAndPlatformAdmission:
    def test_model_export_and_platform_admission(self):
        synth_data = SCRATCH_DIR / "export_synth.jsonl.gz"
        create_synthetic_dataset(synth_data, num_rows=24)

        train_out = SCRATCH_DIR / "train_weights"
        best_pt, _ = train_iteration_model(
            cls="nano",
            data_path=synth_data,
            out_dir=train_out,
            epochs=1,
            batch=8,
            lr=1e-3,
            seed=42,
            device="cpu",
        )

        model_pkg = SCRATCH_DIR / "model_pkg"
        said = export_and_verify_model(
            cls="nano",
            weights_path=best_pt,
            out_dir=model_pkg,
            method="dagger_test",
        )

        assert (model_pkg / "model.onnx").exists()
        assert (model_pkg / "manifest.json").exists()
        assert (model_pkg / "metrics.json").exists()
        assert (model_pkg / "card.md").exists()

        assert said["ok"] is True
        assert said["engine_digest"] == EXPECTED_DIGEST
        assert said["size_metric_bytes"] < int(12.8 * 1024)
        assert said["infer_us_max"] <= 700_000
        assert said["memory"]["round_trip"]["failed"] == 0

# ==============================================================================
# 6. 准入裁决独立函数验证测试集
# ==============================================================================


class TestAdmissionVerdictVerification:
    def test_admission_nano_size_negative_boundary(self):
        said = {
            "ok": True,
            "engine_digest": EXPECTED_DIGEST,
            "size_metric_bytes": int(12.8 * 1024),
            "infer_us_max": 20_000,
            "memory": {"round_trip": {"failed": 0}},
        }
        with pytest.raises(AssertionError, match="超出 12.8 KB"):
            validate_admission_verdict(said, cls="nano")

    def test_admission_nano_size_positive_boundary(self):
        said = {
            "ok": True,
            "engine_digest": EXPECTED_DIGEST,
            "size_metric_bytes": int(12.8 * 1024) - 1,
            "infer_us_max": 20_000,
            "memory": {"round_trip": {"failed": 0}},
        }
        validate_admission_verdict(said, cls="nano")

    def test_admission_micro_size_negative_boundary(self):
        said = {
            "ok": True,
            "engine_digest": EXPECTED_DIGEST,
            "size_metric_bytes": 128 * 1024,
            "infer_us_max": 20_000,
            "memory": {"round_trip": {"failed": 0}},
        }
        with pytest.raises(AssertionError, match="超出 128 KB"):
            validate_admission_verdict(said, cls="micro")

    def test_admission_micro_size_positive_boundary(self):
        said = {
            "ok": True,
            "engine_digest": EXPECTED_DIGEST,
            "size_metric_bytes": 128 * 1024 - 1,
            "infer_us_max": 20_000,
            "memory": {"round_trip": {"failed": 0}},
        }
        validate_admission_verdict(said, cls="micro")

    def test_admission_latency_negative_boundary(self):
        said = {
            "ok": True,
            "engine_digest": EXPECTED_DIGEST,
            "size_metric_bytes": 10_000,
            "infer_us_max": 700_001,
            "memory": {"round_trip": {"failed": 0}},
        }
        with pytest.raises(AssertionError, match="超出 700 毫秒"):
            validate_admission_verdict(said, cls="nano")

    def test_admission_latency_positive_boundary(self):
        said = {
            "ok": True,
            "engine_digest": EXPECTED_DIGEST,
            "size_metric_bytes": 10_000,
            "infer_us_max": 700_000,
            "memory": {"round_trip": {"failed": 0}},
        }
        validate_admission_verdict(said, cls="nano")

    def test_admission_engine_digest_mismatch(self):
        said = {
            "ok": True,
            "engine_digest": "sha256:corrupted_engine_digest_hash_value",
            "size_metric_bytes": 10_000,
            "infer_us_max": 20_000,
            "memory": {"round_trip": {"failed": 0}},
        }
        with pytest.raises(AssertionError, match="平台校验返回的引擎散列摘要不匹配"):
            validate_admission_verdict(said, cls="nano")

    def test_admission_rejection_reason(self):
        said = {
            "ok": False,
            "reason": "Platform validator rejected component",
            "engine_digest": EXPECTED_DIGEST,
            "size_metric_bytes": 10_000,
            "infer_us_max": 20_000,
            "memory": {"round_trip": {"failed": 0}},
        }
        with pytest.raises(AssertionError, match="平台准入校验拒绝了构件"):
            validate_admission_verdict(said, cls="nano")

    def test_admission_memory_round_trip_failure(self):
        said = {
            "ok": True,
            "engine_digest": EXPECTED_DIGEST,
            "size_metric_bytes": 10_000,
            "infer_us_max": 20_000,
            "memory": {"round_trip": {"failed": 3}},
        }
        with pytest.raises(AssertionError, match="记忆往返一致性测试存在 3 次失败"):
            validate_admission_verdict(said, cls="nano")

    def test_admission_unknown_class_failure(self):
        said = {
            "ok": True,
            "engine_digest": EXPECTED_DIGEST,
            "size_metric_bytes": 10_000,
            "infer_us_max": 20_000,
            "memory": {"round_trip": {"failed": 0}},
        }
        with pytest.raises(ValueError, match="未知的规格等级"):
            validate_admission_verdict(said, cls="unknown")


# ==============================================================================
# 7. 遥测生产类与模式兼容性测试集
# ==============================================================================


class TestTelemetrySchemaCompatibility:
    def test_telemetry_initialization_when_file_absent(self):
        env_root = SCRATCH_DIR / "telemetry_env_empty"
        env_root.mkdir(parents=True, exist_ok=True)
        config = PipelineConfig(
            route="route_a",
            weight_class="nano",
            start_iter=4,
            end_iter=8,
            pretrain_epochs=5,
            epochs=5,
            batch_size=256,
            lr=2e-3,
            device="cpu",
            seed=1,
            samples_per_iter=83334,
            project_root=env_root,
        )
        pipeline = DaggerPipeline(config)
        telemetry = pipeline.load_or_init_telemetry(baseline_sigs_count=77171)

        assert telemetry["baseline_signatures_d0"] == 77171
        assert telemetry["disagreement_rates"] == {}
        assert telemetry["expansion_factors"] == {}
        assert telemetry["cumulative_signatures"] == {}
        assert telemetry["novel_signatures_vs_d0"] == {}

    def test_telemetry_loading_existing_file_preserving_history(self):
        env_root = SCRATCH_DIR / "telemetry_env_existing"
        exp_runs = env_root / "runs" / "exp031"
        exp_runs.mkdir(parents=True, exist_ok=True)
        ref_file = ROOT / "runs" / "exp031" / "dagger_telemetry.json"
        assert ref_file.exists()
        shutil.copy2(ref_file, exp_runs / "dagger_telemetry.json")

        config = PipelineConfig(
            route="route_a",
            weight_class="nano",
            start_iter=4,
            end_iter=8,
            pretrain_epochs=5,
            epochs=5,
            batch_size=256,
            lr=2e-3,
            device="cpu",
            seed=1,
            samples_per_iter=83334,
            project_root=env_root,
        )
        pipeline = DaggerPipeline(config)
        loaded = pipeline.load_or_init_telemetry(baseline_sigs_count=77171)

        assert loaded["baseline_signatures_d0"] == 77171
        assert "d1" in loaded["disagreement_rates"]
        assert "d2" in loaded["disagreement_rates"]
        assert "d3" in loaded["disagreement_rates"]
        assert "k1" in loaded["cumulative_signatures"]
        assert "k2" in loaded["cumulative_signatures"]
        assert "k3" in loaded["cumulative_signatures"]

    def test_telemetry_iteration_update_and_persistence(self):
        env_root = SCRATCH_DIR / "telemetry_env_update"
        exp_runs = env_root / "runs" / "exp031"
        exp_runs.mkdir(parents=True, exist_ok=True)
        ref_file = ROOT / "runs" / "exp031" / "dagger_telemetry.json"
        shutil.copy2(ref_file, exp_runs / "dagger_telemetry.json")

        config = PipelineConfig(
            route="route_a",
            weight_class="nano",
            start_iter=4,
            end_iter=8,
            pretrain_epochs=5,
            epochs=5,
            batch_size=256,
            lr=2e-3,
            device="cpu",
            seed=1,
            samples_per_iter=83334,
            project_root=env_root,
        )
        pipeline = DaggerPipeline(config)
        telemetry = pipeline.load_or_init_telemetry(baseline_sigs_count=77171)

        mock_k4_stats = {
            "iteration": 4,
            "rows": 83334,
            "ant_decisions": 950000,
            "disagreed_ants": 190000,
            "disagreement_rate": 0.2000,
            "unique_states": 14000,
            "novel_states": 9500,
            "novel_state_ratio": 0.1140,
            "episodes": 85,
            "reasons": {"rank_stabilized": 60, "extermination": 5, "turn_limit": 20},
            "mean_best_score": 2.65,
            "seconds": 108.5,
            "rows_per_second": 768,
            "out": "data/exp031_iter4.jsonl.gz",
        }

        updated = pipeline.update_telemetry_record(
            telemetry=telemetry,
            k=4,
            rollout_stats=mock_k4_stats,
            cumulative_sigs_count=108950,
            baseline_sigs_count=77171,
            novel_vs_d0_count=9500,
        )
        saved_file = pipeline.save_telemetry(updated)

        assert saved_file.exists()
        reloaded = json.loads(saved_file.read_text(encoding="utf-8"))

        assert reloaded["baseline_signatures_d0"] == 77171
        assert reloaded["disagreement_rates"]["d4"] == 0.2000
        assert reloaded["expansion_factors"]["e4"] == round(108950 / 77171, 4)
        assert reloaded["cumulative_signatures"]["k4"] == 108950
        assert reloaded["novel_signatures_vs_d0"]["k4"] == 9500
        assert reloaded["k4"]["iteration"] == 4
        assert reloaded["k4"]["rows"] == 83334
        assert reloaded["k4"]["ant_decisions"] == 950000
        assert "d1" in reloaded["disagreement_rates"]

    def test_telemetry_route_b_exp033_isolation(self):
        env_root = SCRATCH_DIR / "telemetry_env_route_b"
        config = PipelineConfig(
            route="route_b",
            weight_class="micro",
            start_iter=1,
            end_iter=8,
            pretrain_epochs=5,
            epochs=5,
            batch_size=256,
            lr=2e-3,
            device="cpu",
            seed=1,
            samples_per_iter=83333,
            project_root=env_root,
        )
        pipeline = DaggerPipeline(config)
        assert config.exp_name == "exp033"
        telemetry = pipeline.load_or_init_telemetry(baseline_sigs_count=77171)
        saved_file = pipeline.save_telemetry(telemetry)
        assert saved_file == env_root / "runs" / "exp033" / "dagger_telemetry.json"
        assert not (env_root / "runs" / "exp031" / "dagger_telemetry.json").exists()
