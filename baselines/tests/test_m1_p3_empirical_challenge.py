import collections
import gzip
import hashlib
import json
import os
import subprocess
from pathlib import Path

import numpy as np
import onnx
from onnx.reference import ReferenceEvaluator
import pytest
import torch

BASELINES_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = BASELINES_ROOT.parent
MODELS_DIR = PROJECT_ROOT / "models"
RUNS_DIR = PROJECT_ROOT / "runs"
DATA_DIR = PROJECT_ROOT / "data"

EXPECTED_ENGINE_DIGEST = "sha256:819166e7918117a1cbbd67dc042e7368a076a2dd45cea5122683983912d591e1"


class TestDatasetGzipStreamingAndSchema:

    @pytest.mark.parametrize(
        ("filename", "expected_rows", "expected_kind"),
        [
            ("exp033_iter7.jsonl.gz", 83334, "dagger-rollout"),
            ("exp033_k7.jsonl.gz", 833332, "dagger-aggregated"),
            ("exp033_iter8.jsonl.gz", 83334, "dagger-rollout"),
            ("exp033_k8.jsonl.gz", 916665, "dagger-aggregated"),
        ],
    )
    def test_gzip_streaming_integrity_and_sample_count(self, filename: str, expected_rows: int, expected_kind: str):
        file_path = DATA_DIR / filename
        assert file_path.exists(), f"文件不存在: {file_path}"

        row_count = 0
        head_records = []
        tail_records = collections.deque(maxlen=1000)

        with gzip.open(file_path, "rt", encoding="utf-8") as gz_file:
            for line in gz_file:
                if row_count == 0:
                    meta = json.loads(line)
                    assert meta.get("kind") == expected_kind
                    assert meta.get("engine_digest") == EXPECTED_ENGINE_DIGEST
                    assert meta.get("memory_mode") == "where"
                else:
                    if len(head_records) < 1000:
                        head_records.append(line)
                    tail_records.append(line)
                row_count += 1

        assert row_count == expected_rows, f"{filename} 行数不匹配: 实际 {row_count}, 期望 {expected_rows}"
        assert len(head_records) == 1000, f"头部样本数量不足 1000: {len(head_records)}"
        assert len(tail_records) == 1000, f"尾部样本数量不足 1000: {len(tail_records)}"

        required_obs_keys = {"foes", "food", "hills", "ids", "mine", "size", "vis", "water"}
        valid_actions = {"N", "E", "S", "W", "-"}

        for line in head_records:
            record = json.loads(line)
            assert "o" in record and "k" in record and "m" in record and "a" in record
            obs = record["o"]
            assert required_obs_keys.issubset(obs.keys())
            assert len(obs["size"]) == 2
            h, w = obs["size"]
            assert isinstance(obs["vis"], dict) and "rle" in obs["vis"]
            assert sum(obs["vis"]["rle"][1::2]) == h * w
            assert isinstance(obs["water"], dict) and "rle" in obs["water"]
            assert sum(obs["water"]["rle"][1::2]) == h * w
            k = record["k"]
            assert isinstance(k, list) and len(k) == 3
            act_str = record["a"]
            assert isinstance(act_str, str)
            for ch in act_str:
                assert ch in valid_actions

        for line in tail_records:
            record = json.loads(line)
            assert "o" in record and "k" in record and "m" in record and "a" in record
            obs = record["o"]
            assert required_obs_keys.issubset(obs.keys())
            assert len(obs["size"]) == 2
            h, w = obs["size"]
            assert isinstance(obs["vis"], dict) and "rle" in obs["vis"]
            assert sum(obs["vis"]["rle"][1::2]) == h * w
            assert isinstance(obs["water"], dict) and "rle" in obs["water"]
            assert sum(obs["water"]["rle"][1::2]) == h * w
            k = record["k"]
            assert isinstance(k, list) and len(k) == 3
            act_str = record["a"]
            assert isinstance(act_str, str)
            for ch in act_str:
                assert ch in valid_actions


class TestONNXModelInferenceAndShapeContract:

    @pytest.mark.parametrize(
        "model_dir_name",
        [
            "exp033_iter7",
            "exp033_iter8",
            "exp033",
        ],
    )
    def test_onnx_graph_structure_and_where_operator(self, model_dir_name: str):
        onnx_file = MODELS_DIR / model_dir_name / "model.onnx"
        assert onnx_file.exists(), f"ONNX 文件不存在: {onnx_file}"

        model = onnx.load(str(onnx_file))
        onnx.checker.check_model(model)

        input_names = [inp.name for inp in model.graph.input]
        output_names = [out.name for out in model.graph.output]
        assert "board" in input_names
        assert "memory_in" in input_names
        assert "policy" in output_names
        assert "memory" in output_names

        operators = {node.op_type for node in model.graph.node}
        assert "Where" in operators, f"模型中必须包含 Where 算子: {operators}"
        assert "Max" not in operators, f"模型中不得包含 Max 算子: {operators}"

    @pytest.mark.parametrize(
        "model_dir_name",
        [
            "exp033_iter7",
            "exp033_iter8",
            "exp033",
        ],
    )
    @pytest.mark.parametrize(
        ("height", "width"),
        [
            (24, 24),
            (32, 32),
            (64, 64),
            (128, 128),
            (24, 32),
            (32, 40),
            (48, 64),
            (25, 25),
            (33, 33),
            (25, 35),
        ],
    )
    def test_onnx_inference_dynamic_spatial_dimensions(self, model_dir_name: str, height: int, width: int):
        onnx_file = MODELS_DIR / model_dir_name / "model.onnx"
        evaluator = ReferenceEvaluator(str(onnx_file))

        board = np.random.randn(1, 7, height, width).astype(np.float32)
        memory_in = np.zeros((1, 2, height, width), dtype=np.int8)

        policy, memory = evaluator.run(None, {"board": board, "memory_in": memory_in})

        assert policy.shape == (1, 5, height, width), f"策略输出维度错误: {policy.shape}"
        assert memory.shape == (1, 2, height, width), f"记忆输出维度错误: {memory.shape}"
        assert not np.isnan(policy).any(), "策略输出包含 NaN"
        assert not np.isinf(policy).any(), "策略输出包含 Inf"
        assert not np.isnan(memory).any(), "记忆输出包含 NaN"
        assert not np.isinf(memory).any(), "记忆输出包含 Inf"

    @pytest.mark.parametrize(
        "model_dir_name",
        [
            "exp033_iter7",
            "exp033_iter8",
            "exp033",
        ],
    )
    def test_onnx_inference_adversarial_inputs(self, model_dir_name: str):
        onnx_file = MODELS_DIR / model_dir_name / "model.onnx"
        evaluator = ReferenceEvaluator(str(onnx_file))
        h, w = 32, 32

        zero_board = np.zeros((1, 7, h, w), dtype=np.float32)
        zero_memory = np.zeros((1, 2, h, w), dtype=np.int8)
        p_zero, m_zero = evaluator.run(None, {"board": zero_board, "memory_in": zero_memory})
        assert not np.isnan(p_zero).any()
        assert not np.isinf(p_zero).any()
        assert not np.isnan(m_zero).any()

        ones_board = np.ones((1, 7, h, w), dtype=np.float32)
        ones_memory = np.ones((1, 2, h, w), dtype=np.int8)
        p_ones, m_ones = evaluator.run(None, {"board": ones_board, "memory_in": ones_memory})
        assert not np.isnan(p_ones).any()
        assert not np.isinf(p_ones).any()
        assert not np.isnan(m_ones).any()

        extreme_board = (np.random.randn(1, 7, h, w) * 100.0).astype(np.float32)
        extreme_memory = np.random.randint(-128, 127, size=(1, 2, h, w), dtype=np.int8)
        p_ext, m_ext = evaluator.run(None, {"board": extreme_board, "memory_in": extreme_memory})
        assert not np.isnan(p_ext).any()
        assert not np.isinf(p_ext).any()
        assert not np.isnan(m_ext).any()

    @pytest.mark.parametrize(
        "model_dir_name",
        [
            "exp033_iter7",
            "exp033_iter8",
            "exp033",
        ],
    )
    def test_onnx_inference_batched_forward(self, model_dir_name: str):
        onnx_file = MODELS_DIR / model_dir_name / "model.onnx"
        evaluator = ReferenceEvaluator(str(onnx_file))
        batch_size = 2
        h, w = 24, 24

        board = np.random.randn(batch_size, 7, h, w).astype(np.float32)
        memory_in = np.zeros((batch_size, 2, h, w), dtype=np.int8)

        policy, memory = evaluator.run(None, {"board": board, "memory_in": memory_in})
        assert policy.shape == (batch_size, 5, h, w)
        assert memory.shape == (batch_size, 2, h, w)
        assert not np.isnan(policy).any()
        assert not np.isinf(policy).any()


class TestModelEquivalenceIter8AndFinal:

    def test_iter8_and_exp033_onnx_file_digest_and_size(self):
        onnx8_file = MODELS_DIR / "exp033_iter8" / "model.onnx"
        onnx_final_file = MODELS_DIR / "exp033" / "model.onnx"

        bytes8 = onnx8_file.read_bytes()
        bytes_final = onnx_final_file.read_bytes()

        assert len(bytes8) == len(bytes_final), "模型文件大小不一致"
        sha8 = hashlib.sha256(bytes8).hexdigest()
        sha_final = hashlib.sha256(bytes_final).hexdigest()
        assert sha8 == sha_final, f"模型文件散列不一致: iter8={sha8}, final={sha_final}"

    def test_iter8_and_exp033_checkpoint_digest(self):
        pt8_file = RUNS_DIR / "exp033" / "k8" / "best.pt"
        pt_final_file = RUNS_DIR / "exp033" / "best.pt"

        bytes8 = pt8_file.read_bytes()
        bytes_final = pt_final_file.read_bytes()

        assert len(bytes8) == len(bytes_final), "权重检查点文件大小不一致"
        sha8 = hashlib.sha256(bytes8).hexdigest()
        sha_final = hashlib.sha256(bytes_final).hexdigest()
        assert sha8 == sha_final, f"检查点散列不一致: k8={sha8}, best={sha_final}"

    def test_iter8_and_exp033_inference_numerical_equivalence(self):
        eval8 = ReferenceEvaluator(str(MODELS_DIR / "exp033_iter8" / "model.onnx"))
        eval_final = ReferenceEvaluator(str(MODELS_DIR / "exp033" / "model.onnx"))

        np.random.seed(1337)
        for shape in [(24, 24), (32, 40), (64, 64)]:
            h, w = shape
            board = np.random.randn(1, 7, h, w).astype(np.float32)
            memory_in = np.random.randint(-128, 127, size=(1, 2, h, w), dtype=np.int8)

            p8, m8 = eval8.run(None, {"board": board, "memory_in": memory_in})
            p_final, m_final = eval_final.run(None, {"board": board, "memory_in": memory_in})

            np.testing.assert_array_equal(p8, p_final)
            np.testing.assert_array_equal(m8, m_final)


class TestDaggerTelemetryMetricsReasonableness:

    def test_telemetry_json_structure_and_monotonicity(self):
        telemetry_file = RUNS_DIR / "exp033" / "dagger_telemetry.json"
        assert telemetry_file.exists(), f"遥测文件不存在: {telemetry_file}"

        data = json.loads(telemetry_file.read_text(encoding="utf-8"))

        d0 = data["baseline_signatures_d0"]
        assert d0 == 77171, f"基线指纹总数不匹配: {d0}"

        expansions = data["expansion_factors"]
        cumulative = data["cumulative_signatures"]
        disagreements = data["disagreement_rates"]

        prev_cum = d0
        prev_exp = 1.0

        for idx in range(1, 9):
            k_key = f"k{idx}"
            e_key = f"e{idx}"
            d_key = f"d{idx}"

            assert k_key in cumulative
            assert e_key in expansions
            assert d_key in disagreements

            cum_val = cumulative[k_key]
            exp_val = expansions[e_key]
            dis_val = disagreements[d_key]

            assert cum_val > prev_cum, f"累积指纹非单调递增: {k_key} ({cum_val}) <= 前值 ({prev_cum})"
            assert exp_val > prev_exp, f"扩展倍率非单调递增: {e_key} ({exp_val}) <= 前值 ({prev_exp})"

            expected_exp = round(cum_val / d0, 4)
            assert exp_val == expected_exp, f"扩展倍率数学计算不匹配: {exp_val} != {expected_exp}"

            prev_cum = cum_val
            prev_exp = exp_val

            assert 0.05 <= dis_val <= 0.15, f"分歧率超出合理范围: {d_key}={dis_val}"

            detail = data[k_key]
            assert detail["rows"] == 83333
            assert detail["ant_decisions"] > 0
            assert detail["disagreed_ants"] > 0
            computed_rate = round(detail["disagreed_ants"] / detail["ant_decisions"], 4)
            assert detail["disagreement_rate"] == computed_rate

        assert disagreements["d8"] < disagreements["d1"] * 0.60, "第 8 轮分歧率未显著下降至第 1 轮的 60% 以下"


class TestPlatformGateCertification:

    @pytest.mark.parametrize(
        "model_dir_name",
        [
            "exp033_iter7",
            "exp033_iter8",
            "exp033",
        ],
    )
    def test_tinybrains_check_platform_admission(self, model_dir_name: str):
        model_dir = MODELS_DIR / model_dir_name
        onnx_file = model_dir / "model.onnx"
        manifest_file = model_dir / "manifest.json"

        env = os.environ.copy()
        env["TINYBRAINS_REGISTRY"] = str(BASELINES_ROOT / "games.toml")

        res = subprocess.run(
            ["tinybrains", "check", str(onnx_file), str(manifest_file), "--json"],
            capture_output=True,
            text=True,
            env=env,
            cwd=str(PROJECT_ROOT),
        )

        assert res.returncode == 0, f"tinybrains check 退出码非零: {res.stderr}"
        report = json.loads(res.stdout)

        assert report["ok"] is True, f"平台准入门禁未通过: {report.get('reason')}"
        assert report["engine_digest"] == EXPECTED_ENGINE_DIGEST, "引擎散列摘要不匹配"
        assert report["size_metric_bytes"] < 131072, f"模型包体积超过 128 KB: {report['size_metric_bytes']}"
        assert report["infer_us_max"] < 700000, f"推理峰值时延超过 700 毫秒: {report['infer_us_max']}"
        assert report["memory"]["round_trip"]["failed"] == 0, "记忆往返测试存在失败"
