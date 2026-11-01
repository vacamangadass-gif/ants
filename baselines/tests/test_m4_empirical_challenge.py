import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper
import pytest

REPO_ROOT = Path("/home/host/project/ants")
DIST_WASM = REPO_ROOT / "dist" / "tb-ants.wasm"
MODELS_DIR = REPO_ROOT / "models"
GAMES_TOML = REPO_ROOT / "baselines" / "games.toml"

EXP031_MODELS = [f"exp031_iter{i}" for i in range(1, 9)]
EXP033_MODELS = [f"exp033_iter{i}" for i in range(1, 9)]
ALL_16_MODELS = EXP031_MODELS + EXP033_MODELS

EXPECTED_ENGINE_DIGEST = (
    "sha256:819166e7918117a1cbbd67dc042e7368a076a2dd45cea5122683983912d591e1"
)
EXPECTED_WASM_SHA256 = (
    "819166e7918117a1cbbd67dc042e7368a076a2dd45cea5122683983912d591e1"
)

MAX_LATENCY_US = 700_000
MICRO_SIZE_LIMIT_BYTES = 128 * 1024
NANO_SIZE_LIMIT_BYTES = int(12.8 * 1024)


def test_wasm_engine_digest():
    assert DIST_WASM.exists()
    assert DIST_WASM.is_file()
    data = DIST_WASM.read_bytes()
    computed_sha256 = hashlib.sha256(data).hexdigest()
    assert computed_sha256 == EXPECTED_WASM_SHA256


def run_tinybrains_check(model_rel_path: str):
    target_dir = MODELS_DIR / model_rel_path
    onnx_file = target_dir / "model.onnx"
    manifest_file = target_dir / "manifest.json"

    assert onnx_file.exists()
    assert manifest_file.exists()

    cmd = [
        "tinybrains",
        "check",
        str(onnx_file),
        str(manifest_file),
        "--json",
    ]
    env = {"TINYBRAINS_REGISTRY": str(GAMES_TOML)}
    res = subprocess.run(
        cmd,
        cwd=str(REPO_ROOT),
        env={**dict(subprocess.os.environ), **env},
        capture_output=True,
        text=True,
    )
    assert res.returncode == 0
    parsed = json.loads(res.stdout)
    return parsed


def test_tinybrains_check_exp033():
    result = run_tinybrains_check("exp033")
    assert result["ok"] is True
    assert result["engine_digest"] == EXPECTED_ENGINE_DIGEST
    assert result["size_metric_bytes"] < MICRO_SIZE_LIMIT_BYTES
    assert result["artifact_bytes"] < MICRO_SIZE_LIMIT_BYTES
    assert result["infer_us_max"] < MAX_LATENCY_US


def test_tinybrains_check_exp033_iter7():
    result = run_tinybrains_check("exp033_iter7")
    assert result["ok"] is True
    assert result["engine_digest"] == EXPECTED_ENGINE_DIGEST
    assert result["size_metric_bytes"] < MICRO_SIZE_LIMIT_BYTES
    assert result["artifact_bytes"] < MICRO_SIZE_LIMIT_BYTES
    assert result["infer_us_max"] < MAX_LATENCY_US


def test_tinybrains_check_exp033_iter8():
    result = run_tinybrains_check("exp033_iter8")
    assert result["ok"] is True
    assert result["engine_digest"] == EXPECTED_ENGINE_DIGEST
    assert result["size_metric_bytes"] < MICRO_SIZE_LIMIT_BYTES
    assert result["artifact_bytes"] < MICRO_SIZE_LIMIT_BYTES
    assert result["infer_us_max"] < MAX_LATENCY_US


def test_tinybrains_check_exp031():
    model_file = MODELS_DIR / "exp031" / "model.onnx"
    assert model_file.stat().st_size < NANO_SIZE_LIMIT_BYTES

    result = run_tinybrains_check("exp031")
    assert result["ok"] is True
    assert result["engine_digest"] == EXPECTED_ENGINE_DIGEST
    assert result["size_metric_bytes"] < NANO_SIZE_LIMIT_BYTES
    assert result["artifact_bytes"] < NANO_SIZE_LIMIT_BYTES
    assert result["infer_us_max"] < MAX_LATENCY_US


@pytest.mark.parametrize("model_name", ALL_16_MODELS)
def test_model_directory_and_json_validity(model_name: str):
    model_dir = MODELS_DIR / model_name
    assert model_dir.exists()
    assert model_dir.is_dir()

    onnx_path = model_dir / "model.onnx"
    assert onnx_path.exists()
    assert onnx_path.stat().st_size > 0

    manifest_path = model_dir / "manifest.json"
    assert manifest_path.exists()
    manifest_data = json.loads(manifest_path.read_text())
    assert manifest_data.get("abi") == "orion:model@1.0.0"
    assert manifest_data.get("format") == "onnx"
    assert "name" in manifest_data
    assert len(manifest_data.get("inputs", [])) == 2
    assert len(manifest_data.get("outputs", [])) == 2

    metrics_path = model_dir / "metrics.json"
    assert metrics_path.exists()
    metrics_data = json.loads(metrics_path.read_text())
    assert isinstance(metrics_data, dict)
    assert metrics_data.get("class") in ("nano", "micro")
    assert metrics_data.get("engine_digest") == EXPECTED_ENGINE_DIGEST
    assert "weights_hash" in metrics_data
    assert metrics_data["weights_hash"].startswith("sha256:")


@pytest.mark.parametrize("model_name", ALL_16_MODELS)
def test_all_16_models_tinybrains_check(model_name: str):
    res = run_tinybrains_check(model_name)
    assert res["ok"] is True
    assert res["engine_digest"] == EXPECTED_ENGINE_DIGEST
    assert res["infer_us_max"] < MAX_LATENCY_US
    if model_name.startswith("exp031"):
        assert res["size_metric_bytes"] < NANO_SIZE_LIMIT_BYTES
        assert res["artifact_bytes"] < NANO_SIZE_LIMIT_BYTES
    else:
        assert res["size_metric_bytes"] < MICRO_SIZE_LIMIT_BYTES
        assert res["artifact_bytes"] < MICRO_SIZE_LIMIT_BYTES


def test_all_16_models_weights_distinctness():
    hashes = {}
    for model_name in ALL_16_MODELS:
        onnx_path = MODELS_DIR / model_name / "model.onnx"
        sha256_val = hashlib.sha256(onnx_path.read_bytes()).hexdigest()
        hashes[model_name] = sha256_val

    assert len(hashes) == 16
    unique_hashes = set(hashes.values())
    assert len(unique_hashes) == 16


@pytest.mark.parametrize("model_name", ALL_16_MODELS)
def test_model_numerical_integrity(model_name: str):
    onnx_path = MODELS_DIR / model_name / "model.onnx"
    model_proto = onnx.load(str(onnx_path))
    assert len(model_proto.graph.initializer) > 0

    for init in model_proto.graph.initializer:
        arr = numpy_helper.to_array(init)
        assert not np.isnan(arr).any()
        assert not np.isinf(arr).any()


def test_final_iteration_model_equivalence():
    exp031_iter8_sha = hashlib.sha256(
        (MODELS_DIR / "exp031_iter8" / "model.onnx").read_bytes()
    ).hexdigest()
    exp031_sha = hashlib.sha256(
        (MODELS_DIR / "exp031" / "model.onnx").read_bytes()
    ).hexdigest()
    assert exp031_sha == exp031_iter8_sha

    exp033_iter8_sha = hashlib.sha256(
        (MODELS_DIR / "exp033_iter8" / "model.onnx").read_bytes()
    ).hexdigest()
    exp033_sha = hashlib.sha256(
        (MODELS_DIR / "exp033" / "model.onnx").read_bytes()
    ).hexdigest()
    assert exp033_sha == exp033_iter8_sha
