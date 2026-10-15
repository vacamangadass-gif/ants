"""Empirical Adversarial Stress Test Suite for Milestone 3 (EXP-010 ~ EXP-014).

Author: teamwork_preview_challenger_m3_2
Mission:
1. Stress-test ONNX inference across diverse board shapes (tiny, medium, large, xlarge, non-square, odd) and batch sizes.
2. Benchmark single-turn CPU inference latency under load: assert latency remains strictly <= 700 ms across 100+ turns.
3. Multi-turn sequential rollout stress test:
   - Feed sequential observation streams into each model, pass output memory back into input memory for 100+ turns.
   - Assert numerical stability: memory values stay within valid bounds (int8: 0, 1), no shape mismatches, zero memory leaks.
4. Operator verification:
   - Directly inspect ONNX graph nodes: confirm Max in EXP-010/012 and Where in EXP-011/013/014.
5. Platform certification compliance check via tinybrains check.
"""

from __future__ import annotations

import gc
import json
import os
import resource
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest

# Ensure baselines/src is on path
BASELINES_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = BASELINES_ROOT.parent
SRC_ROOT = BASELINES_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

try:
    import onnx
    import onnxruntime as ort
    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False

MODELS_DIR = PROJECT_ROOT / "models"
EXPERIMENTS = ["exp010", "exp011", "exp012", "exp013", "exp014"]


@pytest.mark.skipif(not HAS_ONNX, reason="onnx or onnxruntime not installed")
class TestArtifactsAndOperatorIntegrity:
    """Audit artifact layout, sizes, and ONNX graph operator nodes."""

    @pytest.mark.parametrize("exp", EXPERIMENTS)
    def test_artifact_files_and_size_limits(self, exp: str):
        model_dir = MODELS_DIR / exp
        assert model_dir.is_dir(), f"Model directory {model_dir} does not exist"

        onnx_file = model_dir / "model.onnx"
        manifest_file = model_dir / "manifest.json"
        metrics_file = model_dir / "metrics.json"
        card_file = model_dir / "card.md"

        for f in [onnx_file, manifest_file, metrics_file, card_file]:
            assert f.is_file(), f"Required artifact {f} missing"

        onnx_bytes = onnx_file.stat().st_size
        manifest_bytes = manifest_file.stat().st_size
        metric_size = onnx_bytes + manifest_bytes

        # Strict 16 KB (16,384 bytes) platform limit
        assert onnx_bytes <= 16384, f"{exp}: model.onnx exceeds 16KB limit: {onnx_bytes} bytes"
        assert metric_size <= 16384, f"{exp}: metric size (model+manifest) exceeds 16KB limit: {metric_size} bytes"

        # Check manifest schema
        manifest = json.loads(manifest_file.read_text())
        assert manifest.get("abi") == "orion:model@1.0.0"
        assert manifest.get("format") == "onnx"
        assert manifest.get("inputs")[0]["name"] == "board"
        assert manifest.get("inputs")[1]["name"] == "memory_in"
        assert manifest.get("outputs")[0]["name"] == "policy"
        assert manifest.get("outputs")[1]["name"] == "memory"

    @pytest.mark.parametrize("exp", EXPERIMENTS)
    def test_operator_verification(self, exp: str):
        model_path = MODELS_DIR / exp / "model.onnx"
        model = onnx.load(str(model_path))

        op_types = set(node.op_type for node in model.graph.node)

        # Invariant check from IDEA-002:
        # EXP-010 and EXP-012 use legacy Max memory
        # EXP-011, EXP-013, EXP-014 use Where memory
        if exp in ("exp010", "exp012"):
            assert "Max" in op_types, f"{exp} must contain Max operator node"
            assert "Where" not in op_types, f"{exp} must NOT contain Where operator node"
        elif exp in ("exp011", "exp013", "exp014"):
            assert "Where" in op_types, f"{exp} must contain Where operator node"
            assert "Max" not in op_types, f"{exp} must NOT contain Max operator node"
            assert "Equal" in op_types, f"{exp} must contain Equal operator node for visibility test"
            assert "Not" in op_types, f"{exp} must contain Not operator node for visibility inversion"

        # Check opset
        opset_version = model.opset_import[0].version
        assert 13 <= opset_version <= 19, f"Opset version {opset_version} outside allowed range [13, 19]"

        # Check parameter count is within Nano limits (~3.1k params)
        total_params = 0
        for init in model.graph.initializer:
            dims = init.dims
            params = 1
            for d in dims:
                params *= d
            total_params += params
        assert total_params < 4000, f"{exp} parameters {total_params} exceeds Nano limit 4,000"


@pytest.mark.skipif(not HAS_ONNX, reason="onnx or onnxruntime not installed")
class TestDynamicBoardShapesAndBatching:
    """Stress-test ONNX inference across diverse board shapes and batch sizes."""

    SHAPES = [
        (24, 24),   # tiny (basic-tiny-2p)
        (36, 36),   # small (basic-small-3p)
        (48, 64),   # medium (basic-medium-4p)
        (80, 96),   # large (basic-large-6p)
        (120, 124), # xlarge (basic-xlarge-8p)
        (128, 128), # probe dimensions
        (25, 27),   # odd dimensions
        (24, 124),  # extreme aspect ratio (wide)
        (120, 24),  # extreme aspect ratio (tall)
    ]

    BATCH_SIZES = [1, 2, 4, 8, 16]

    @pytest.mark.parametrize("exp", EXPERIMENTS)
    @pytest.mark.parametrize("h,w", SHAPES)
    def test_diverse_board_shapes(self, exp: str, h: int, w: int):
        model_path = str(MODELS_DIR / exp / "model.onnx")
        session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])

        b = 1
        board = np.zeros((b, 7, h, w), dtype=np.int8)
        memory_in = np.zeros((b, 2, h, w), dtype=np.int8)

        # Seed random visibility and ant
        board[0, 6, :, :] = 1  # all visible
        board[0, 0, 0, 0] = 1  # ant at (0, 0)

        outputs = session.run(None, {"board": board, "memory_in": memory_in})
        policy, memory_out = outputs[0], outputs[1]

        assert policy.shape == (b, 5, h, w), f"Policy shape mismatch: got {policy.shape}"
        assert memory_out.shape == (b, 2, h, w), f"Memory shape mismatch: got {memory_out.shape}"
        assert policy.dtype == np.float32, f"Policy dtype mismatch: got {policy.dtype}"
        assert memory_out.dtype == np.int8, f"Memory dtype mismatch: got {memory_out.dtype}"
        assert np.isfinite(policy).all(), "Policy contains NaN or Inf"
        assert np.all((memory_out == 0) | (memory_out == 1)), "Memory out has values outside {0, 1}"

    @pytest.mark.parametrize("exp", EXPERIMENTS)
    @pytest.mark.parametrize("batch_size", BATCH_SIZES)
    def test_batch_sizes(self, exp: str, batch_size: int):
        model_path = str(MODELS_DIR / exp / "model.onnx")
        session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])

        h, w = 32, 32
        board = np.zeros((batch_size, 7, h, w), dtype=np.int8)
        memory_in = np.zeros((batch_size, 2, h, w), dtype=np.int8)

        outputs = session.run(None, {"board": board, "memory_in": memory_in})
        policy, memory_out = outputs[0], outputs[1]

        assert policy.shape == (batch_size, 5, h, w)
        assert memory_out.shape == (batch_size, 2, h, w)
        assert np.isfinite(policy).all()


@pytest.mark.skipif(not HAS_ONNX, reason="onnx or onnxruntime not installed")
class TestLatencyBenchmarkUnderLoad:
    """Assert single-turn CPU latency strictly <= 700 ms across 100+ simulated turns."""

    @pytest.mark.parametrize("exp", EXPERIMENTS)
    def test_latency_under_load_100_turns(self, exp: str):
        model_path = str(MODELS_DIR / exp / "model.onnx")
        session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])

        # Test on maximum board size: xlarge (120, 124)
        h, w = 120, 124
        board = np.random.randint(0, 2, size=(1, 7, h, w), dtype=np.int8)
        memory = np.zeros((1, 2, h, w), dtype=np.int8)

        # Warm up 5 turns
        for _ in range(5):
            _, memory = session.run(None, {"board": board, "memory_in": memory})

        latencies = []
        num_turns = 120  # strictly > 100 turns

        for t in range(num_turns):
            # Dynamic board updates
            board[0, 6] = np.random.randint(0, 2, size=(h, w), dtype=np.int8)
            board[0, 2] = np.random.randint(0, 2, size=(h, w), dtype=np.int8)

            t0 = time.perf_counter()
            _, memory = session.run(None, {"board": board, "memory_in": memory})
            dt_ms = (time.perf_counter() - t0) * 1000.0
            latencies.append(dt_ms)

        max_latency = max(latencies)
        mean_latency = np.mean(latencies)
        p95_latency = np.percentile(latencies, 95)
        p99_latency = np.percentile(latencies, 99)

        print(f"\n[{exp}] 120 turns on (120, 124): Mean={mean_latency:.2f}ms, Max={max_latency:.2f}ms, P95={p95_latency:.2f}ms, P99={p99_latency:.2f}ms")

        # Core constraint assertion: <= 700 ms per turn
        assert max_latency <= 700.0, f"{exp}: Maximum latency {max_latency:.2f}ms exceeded limit 700ms"
        # Sanity check: on modern CPU, it should comfortably run within 50ms
        assert mean_latency <= 50.0, f"{exp}: Mean latency {mean_latency:.2f}ms suspiciously high"


@pytest.mark.skipif(not HAS_ONNX, reason="onnx or onnxruntime not installed")
class TestMultiTurnRolloutAndMemoryStability:
    """Multi-turn sequential rollout stress test:

    - Feed sequential observation streams into each model for 150 turns.
    - Pass output memory back into input memory.
    - Assert numerical stability: memory values in {0, 1}, no shape drift, no memory leak.
    """

    @pytest.mark.parametrize("exp", EXPERIMENTS)
    def test_sequential_rollout_stability_and_memory_bounds(self, exp: str):
        model_path = str(MODELS_DIR / exp / "model.onnx")
        session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])

        h, w = 48, 64
        num_turns = 150
        rng = np.random.default_rng(42)

        # Initial zero memory
        memory = np.zeros((1, 2, h, w), dtype=np.int8)

        # Track RSS before rollout
        gc.collect()
        mem_rss_start = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

        for turn in range(num_turns):
            # Construct realistic observation planes:
            # Ch 0: mine, Ch 1: foes, Ch 2: food, Ch 3: water, Ch 4: hill_mine, Ch 5: hill_foe, Ch 6: vis
            board = np.zeros((1, 7, h, w), dtype=np.int8)

            # Random visibility patches
            vis = rng.binomial(1, 0.4, size=(h, w)).astype(np.int8)
            board[0, 6] = vis

            # Food appearing in visible cells
            food = rng.binomial(1, 0.05, size=(h, w)).astype(np.int8) * vis
            board[0, 2] = food

            # Foe hills appearing in visible cells
            if turn % 20 < 10:
                hill_foe = np.zeros((h, w), dtype=np.int8)
                hill_foe[15, 20] = 1
                board[0, 5] = hill_foe * vis

            # Forward model
            outputs = session.run(None, {"board": board, "memory_in": memory})
            policy, memory = outputs[0], outputs[1]

            # Numerical stability assertions:
            assert memory.shape == (1, 2, h, w), f"Turn {turn}: memory shape drifted to {memory.shape}"
            assert memory.dtype == np.int8, f"Turn {turn}: memory dtype drifted to {memory.dtype}"
            assert np.all((memory == 0) | (memory == 1)), f"Turn {turn}: memory contains values outside {{0, 1}}"
            assert np.isfinite(policy).all(), f"Turn {turn}: policy contains non-finite values"

        # Check memory leak after 150 turns
        gc.collect()
        mem_rss_end = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        rss_growth_kb = mem_rss_end - mem_rss_start

        # ru_maxrss is in KB on Linux. Growth should be bounded (< 50MB)
        assert rss_growth_kb < 50000, f"Potential memory leak detected: RSS grew by {rss_growth_kb} KB"

    def test_ghost_hill_elimination_divergence_oracle(self):
        """Adversarial oracle test contrasting Max (EXP-010/012) vs Where (EXP-011/013/014).

        Scenario:
        1. Turn 1: Enemy hill is observed at (10, 10). Cell is visible.
        2. Turn 2-5: Cell is in fog. Both models must remember the hill.
        3. Turn 6: Hill is destroyed! Cell (10, 10) is visible, but contains no hill.
           - Max models MUST still have hill=1 (demonstrating the ghost hill bug).
           - Where models MUST clear hill to 0 (demonstrating ghost hill elimination).
        4. Turn 7: Cell returns to fog.
           - Max models stay 1.
           - Where models stay 0.
        """
        h, w = 32, 32

        sessions = {exp: ort.InferenceSession(str(MODELS_DIR / exp / "model.onnx"), providers=["CPUExecutionProvider"])
                    for exp in EXPERIMENTS}

        memories = {exp: np.zeros((1, 2, h, w), dtype=np.int8) for exp in EXPERIMENTS}

        # --- Turn 1: Discover hill at (10, 10) ---
        board_t1 = np.zeros((1, 7, h, w), dtype=np.int8)
        board_t1[0, 6, 10, 10] = 1  # vis = 1
        board_t1[0, 5, 10, 10] = 1  # hill_foe = 1
        for exp, sess in sessions.items():
            _, memories[exp] = sess.run(None, {"board": board_t1, "memory_in": memories[exp]})
            assert memories[exp][0, 1, 10, 10] == 1, f"{exp} failed to record discovered hill at turn 1"

        # --- Turn 2: Cell enters fog (vis = 0, hill_foe = 0) ---
        board_t2 = np.zeros((1, 7, h, w), dtype=np.int8)  # all fog
        for exp, sess in sessions.items():
            _, memories[exp] = sess.run(None, {"board": board_t2, "memory_in": memories[exp]})
            assert memories[exp][0, 1, 10, 10] == 1, f"{exp} lost memory of hill in fog at turn 2"

        # --- Turn 6: Hill razed! Cell is visible (vis = 1), but no hill (hill_foe = 0) ---
        board_t6 = np.zeros((1, 7, h, w), dtype=np.int8)
        board_t6[0, 6, 10, 10] = 1  # vis = 1
        board_t6[0, 5, 10, 10] = 0  # hill razed!

        for exp, sess in sessions.items():
            _, memories[exp] = sess.run(None, {"board": board_t6, "memory_in": memories[exp]})

        # Oracle assertion of IDEA-002:
        # EXP-010 and EXP-012 use Max -> still 1 (ghost hill bug)
        assert memories["exp010"][0, 1, 10, 10] == 1, "EXP-010 Max memory should retain ghost hill"
        assert memories["exp012"][0, 1, 10, 10] == 1, "EXP-012 Max memory should retain ghost hill"

        # EXP-011, EXP-013, EXP-014 use Where -> must be 0 (cleared!)
        assert memories["exp011"][0, 1, 10, 10] == 0, "EXP-011 Where memory failed to eliminate ghost hill!"
        assert memories["exp013"][0, 1, 10, 10] == 0, "EXP-013 Where memory failed to eliminate ghost hill!"
        assert memories["exp014"][0, 1, 10, 10] == 0, "EXP-014 Where memory failed to eliminate ghost hill!"

        # --- Turn 7: Cell returns to fog (vis = 0) ---
        board_t7 = np.zeros((1, 7, h, w), dtype=np.int8)
        for exp, sess in sessions.items():
            _, memories[exp] = sess.run(None, {"board": board_t7, "memory_in": memories[exp]})

        # EXP-010/012: still ghost hill
        assert memories["exp010"][0, 1, 10, 10] == 1
        assert memories["exp012"][0, 1, 10, 10] == 1
        # EXP-011/013/014: remains cleared
        assert memories["exp011"][0, 1, 10, 10] == 0
        assert memories["exp013"][0, 1, 10, 10] == 0
        assert memories["exp014"][0, 1, 10, 10] == 0


class TestPlatformCertification:
    """Run tinybrains check CLI to ensure official platform certification passes."""

    @pytest.mark.parametrize("exp", EXPERIMENTS)
    def test_tinybrains_check_pass(self, exp: str):
        tinybrains_bin = os.environ.get("TINYBRAINS") or shutil.which("tinybrains") or "/home/linuxbrew/.linuxbrew/bin/tinybrains"
        if not Path(tinybrains_bin).is_file():
            pytest.skip("tinybrains binary not found")

        model_dir = MODELS_DIR / exp
        onnx_file = model_dir / "model.onnx"
        manifest_file = model_dir / "manifest.json"

        env = dict(os.environ)
        env["TINYBRAINS_REGISTRY"] = str(PROJECT_ROOT / "baselines" / "games.toml")

        cmd = [tinybrains_bin, "check", str(onnx_file), str(manifest_file), "--json"]
        proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
        assert proc.returncode == 0, f"tinybrains check failed: {proc.stderr}"

        data = json.loads(proc.stdout)
        assert data.get("ok") is True, f"{exp}: platform check not ok: {data.get('reason')}"
        round_trip = data.get("memory", {}).get("round_trip", {})
        assert round_trip.get("failed") == 0, f"{exp}: round trip failed cases: {round_trip.get('failed')}"
        assert data.get("graph", {}).get("opset") == 17
