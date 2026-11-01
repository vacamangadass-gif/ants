"""Empirical Challenger & Adversarial Stress Suite for Milestone 1: Micro Class ONNX Export & Platform Certification.

Author: teamwork_preview_challenger_p2_m1_2
Mission:
1. Export a Micro model with Where memory via `export.to_onnx` + `export.halve` + `export.shorten`.
2. Inspect the ONNX graph directly:
   - Check opset version (17).
   - Check operator set: assert `Where` is present, assert `Max` is absent.
   - Assert all operators belong to the TinyBrains allowlist.
3. Validate with `tinybrains check <model.onnx> <manifest.json> --json`:
   - Assert `ok == True`.
   - Assert `round_trip.failed == 0`.
   - Assert total artifact metric size <= 131,072 bytes (128KB limit).
   - Assert inference latency <= 700ms.
4. Adversarial stress & boundary validation:
   - Operator rejection oracle: verify `tinybrains check` rejects non-allowlist operators (e.g. GreaterOrEqual).
   - Dynamic board topology stress: (24, 24) to (128, 128) including non-square boards.
   - Where memory bitwise equivalence & razed-hill elimination under adversarial perturbations.
   - 150-turn autoregressive rollout stability: verify zero memory leakage, numerical stability, and valid binary state.
   - Full CLI export pipeline integration (`python -m tb_baselines.export --class micro --memory`).
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

# Ensure baselines/src is on path
BASELINES_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = BASELINES_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import onnx
import torch
from tb_baselines import adapters, export, nets
from tb_baselines.planes import MEMORY_SOURCES, N_MEMORY, N_PLANES


@pytest.fixture(scope="module")
def exported_micro_challenger(tmp_path_factory):
    """Export and optimize Micro model with Where memory for empirical testing."""
    tmp_dir = tmp_path_factory.mktemp("challenger_micro")
    spec = export.classes()["micro"]
    model = nets.build(spec, memory="where")
    onnx_path = tmp_dir / "model.onnx"
    manifest_path = tmp_dir / "manifest.json"
    manifest_path.write_text(adapters.dumps(memory=True))

    export.to_onnx(model, onnx_path, planes=N_PLANES)
    export.halve(onnx_path)
    export.shorten(onnx_path)
    return onnx_path, manifest_path


class TestMicroONNXExportAndIntegrity:
    """Task 1 & 2: Direct inspection of ONNX graph, opset, operators, and size limits."""

    def test_opset_version_is_17(self, exported_micro_challenger):
        onnx_path, _ = exported_micro_challenger
        model = onnx.load(str(onnx_path))
        onnx.checker.check_model(model)
        opset_imports = {entry.domain: entry.version for entry in model.opset_import}
        default_opset = opset_imports.get("", opset_imports.get("ai.onnx"))
        assert default_opset == 17, f"Expected ONNX opset 17, got {default_opset} (imports: {opset_imports})"

    def test_where_operator_present_and_max_absent(self, exported_micro_challenger):
        onnx_path, _ = exported_micro_challenger
        model = onnx.load(str(onnx_path))
        ops = set(n.op_type for n in model.graph.node)
        assert "Where" in ops, f"'Where' operator must be present in graph nodes, found: {ops}"
        assert "Max" not in ops, f"'Max' operator must be absent from graph nodes, found: {ops}"

    def test_all_operators_belong_to_tinybrains_allowlist(self, exported_micro_challenger):
        """Assert all graph operators are within TinyBrains admission allowlist."""
        onnx_path, manifest_path = exported_micro_challenger
        if not (shutil.which("tinybrains") or os.environ.get("TINYBRAINS")):
            pytest.skip("tinybrains binary not available")

        cmd = [export.cli(), "check", str(onnx_path), str(manifest_path), "--json"]
        res = subprocess.run(cmd, capture_output=True, text=True, cwd=BASELINES_ROOT)
        assert res.returncode == 0, f"tinybrains check exited with code {res.returncode}: {res.stderr}"
        data = json.loads(res.stdout)

        surface = data.get("surface", {})
        assert surface.get("ok") is True, f"Surface check failed: {surface}"
        assert surface.get("op_not_allowed") == [], f"Non-allowlist operators detected: {surface.get('op_not_allowed')}"
        assert surface.get("opset_ok") is True, f"Opset check failed: {surface}"

        graph_ops = data.get("graph", {}).get("operators", [])
        assert "Where" in graph_ops, f"'Where' not listed in graph.operators: {graph_ops}"
        assert "Max" not in graph_ops, f"'Max' unexpectedly listed in graph.operators: {graph_ops}"

    def test_artifact_sizes_strictly_within_micro_limit(self, exported_micro_challenger):
        """Micro limit: 131,072 bytes (128KB). Nano limit: 16,384 bytes (16KB)."""
        onnx_path, manifest_path = exported_micro_challenger
        onnx_bytes = onnx_path.stat().st_size
        manifest_bytes = manifest_path.stat().st_size
        total_metric_bytes = onnx_bytes + manifest_bytes

        assert total_metric_bytes <= 131072, (
            f"Total metric size {total_metric_bytes} bytes exceeds Micro 128KB limit (131,072 bytes)"
        )
        assert total_metric_bytes > 16384, (
            f"Total metric size {total_metric_bytes} bytes must exceed Nano 16KB limit (16,384 bytes)"
        )
        # Expected around 53KB
        assert 45000 <= total_metric_bytes <= 65000, (
            f"Unexpected size range: {total_metric_bytes} bytes (expected ~53KB for Micro fp16)"
        )

    def test_halve_quantization_and_shorten_minification(self, exported_micro_challenger):
        """Verify fp16 cast nodes and minified node/tensor names."""
        onnx_path, _ = exported_micro_challenger
        model = onnx.load(str(onnx_path))

        # Halve check: fp16 initializers with Cast nodes
        init_dtypes = {init.name: init.data_type for init in model.graph.initializer}
        fp16_inits = [name for name, dtype in init_dtypes.items() if dtype == onnx.TensorProto.FLOAT16]
        assert len(fp16_inits) > 0, "Expected float16 initializers after export.halve"
        cast_nodes = [n for n in model.graph.node if n.op_type == "Cast"]
        assert len(cast_nodes) > 0, "Expected Cast nodes from export.halve"

        # Shorten check: all intermediate nodes have empty names or short names
        for node in model.graph.node:
            assert node.name == "", f"export.shorten should strip node names, found: {node.name}"


class TestMicroPlatformCertification:
    """Task 3: tinybrains check <model.onnx> <manifest.json> --json certification verification."""

    def test_tinybrains_check_metrics_and_verdict(self, exported_micro_challenger):
        onnx_path, manifest_path = exported_micro_challenger
        if not (shutil.which("tinybrains") or os.environ.get("TINYBRAINS")):
            pytest.skip("tinybrains binary not available")

        cmd = [export.cli(), "check", str(onnx_path), str(manifest_path), "--json"]
        res = subprocess.run(cmd, capture_output=True, text=True, cwd=BASELINES_ROOT)
        assert res.returncode == 0, f"tinybrains check command failed: {res.stderr}"

        v = json.loads(res.stdout)
        # 1. ok == True
        assert v.get("ok") is True, f"Platform admission rejected: {v.get('reason')}"

        # 2. round_trip.failed == 0
        mem = v.get("memory", {})
        rt = mem.get("round_trip", {})
        assert rt.get("failed") == 0, (
            f"Memory round trip failed {rt.get('failed')} times out of {rt.get('checked')} test cases. "
            f"Failing case: {rt.get('failing_case')}, Reason: {rt.get('reason')}"
        )
        assert rt.get("checked") > 150, f"Expected > 150 round trip test cases, got {rt.get('checked')}"

        # 3. total artifact metric size <= 131,072 bytes
        size_metric = v.get("size_metric_bytes")
        assert size_metric <= 131072, f"size_metric_bytes {size_metric} exceeds 131,072 limit"

        # 4. inference latency <= 700ms (700,000 us)
        infer_us = v.get("infer_us_max")
        assert infer_us <= 700000, f"Inference latency {infer_us} us exceeds 700ms limit"
        # Micro latency typically ~15-35 ms
        assert infer_us < 100000, f"Inference latency unexpectedly high: {infer_us / 1000:.2f} ms"

        # 5. Probe test pass on 128x128 board
        probe = v.get("probe", {})
        assert probe.get("ok") is True, f"Probe run failed: {probe}"
        assert probe.get("dims") == {"H": 128, "W": 128}

        # 6. certify helper returns no problems
        problems = export.certify("micro", v)
        assert problems == [], f"export.certify reported problems: {problems}"


class TestAdversarialOperatorDetection:
    """Adversarial Challenge: Verify oracle rejects disallowed operators and detects Max."""

    def test_oracle_rejects_greater_or_equal(self, exported_micro_challenger, tmp_path):
        """Adversarially inject forbidden GreaterOrEqual and assert tinybrains check fails."""
        if not (shutil.which("tinybrains") or os.environ.get("TINYBRAINS")):
            pytest.skip("tinybrains binary not available")

        onnx_path, manifest_path = exported_micro_challenger
        m = onnx.load(str(onnx_path))
        bad_onnx = tmp_path / "bad_op.onnx"

        # Inject disallowed operator GreaterOrEqual
        fake_node = onnx.helper.make_node(
            "GreaterOrEqual",
            inputs=[m.graph.input[0].name, m.graph.input[0].name],
            outputs=["dummy_bad_out"],
        )
        m.graph.node.append(fake_node)
        onnx.save(m, str(bad_onnx))

        cmd = [export.cli(), "check", str(bad_onnx), str(manifest_path), "--json"]
        res = subprocess.run(cmd, capture_output=True, text=True, cwd=BASELINES_ROOT)
        # return code will be 1 because check failed
        data = json.loads(res.stdout) if res.stdout.strip() else {}
        assert data.get("ok") is False, "tinybrains check must reject graph containing GreaterOrEqual"
        surface = data.get("surface", {})
        assert "GreaterOrEqual" in surface.get("op_not_allowed", []), (
            f"Expected GreaterOrEqual in op_not_allowed, got: {surface}"
        )

    def test_max_memory_ablation_detection(self, tmp_path):
        """Build with memory='max' (ablation mode) and confirm Max operator is present."""
        spec = export.classes()["micro"]
        model_max = nets.build(spec, memory="max")
        onnx_max = tmp_path / "model_max.onnx"
        export.to_onnx(model_max, onnx_max, planes=N_PLANES)

        m = onnx.load(str(onnx_max))
        ops = set(n.op_type for n in m.graph.node)
        assert "Max" in ops, "Model built with memory='max' must contain Max operator"
        assert "Where" not in ops, "Model built with memory='max' must NOT contain Where operator"


class TestWhereMemoryAdversarialSemantics:
    """Stress-test Where memory logic under edge cases and adversarial scenarios."""

    def test_multi_quadrant_where_semantics(self):
        spec = export.classes()["micro"]
        model = nets.build(spec, memory="where").eval()

        H, W = 32, 32
        board = torch.zeros(1, N_PLANES, H, W, dtype=torch.int8)
        mem_in = torch.zeros(1, N_MEMORY, H, W, dtype=torch.int8)

        # Quadrant 1 (top-left): visible, enemy hill present (should write 1)
        board[0, 6, 0:16, 0:16] = 1   # vis = 1
        board[0, 5, 0:16, 0:16] = 1   # enemy hill = 1

        # Quadrant 2 (top-right): visible, enemy hill was razed (obs=0, mem_in=1) -> must clear to 0
        board[0, 6, 0:16, 16:32] = 1   # vis = 1
        board[0, 5, 0:16, 16:32] = 0   # enemy hill = 0
        mem_in[0, 1, 0:16, 16:32] = 1  # old memory = 1

        # Quadrant 3 (bottom-left): fog, had enemy hill in memory -> must retain 1
        board[0, 6, 16:32, 0:16] = 0   # vis = 0 (fog)
        board[0, 5, 16:32, 0:16] = 0   # obs = 0
        mem_in[0, 1, 16:32, 0:16] = 1  # old memory = 1

        # Quadrant 4 (bottom-right): fog, no enemy hill -> must remain 0
        board[0, 6, 16:32, 16:32] = 0  # vis = 0 (fog)
        board[0, 5, 16:32, 16:32] = 0  # obs = 0
        mem_in[0, 1, 16:32, 16:32] = 0 # old memory = 0

        with torch.no_grad():
            _, mem_out = model(board, mem_in)

        # Q1: Newly observed hill
        assert torch.all(mem_out[0, 1, 0:16, 0:16] == 1), "Visible enemy hill must be recorded as 1"
        # Q2: Razed hill cleared! (Ghost hill eliminated)
        assert torch.all(mem_out[0, 1, 0:16, 16:32] == 0), "Razed hill in view must be cleared to 0"
        # Q3: Fog hill retained
        assert torch.all(mem_out[0, 1, 16:32, 0:16] == 1), "Fog hill must be retained from memory"
        # Q4: Empty fog remains empty
        assert torch.all(mem_out[0, 1, 16:32, 16:32] == 0), "Empty fog must remain 0"

    def test_non_binary_and_boundary_board_inputs(self):
        """Ensure non-binary visibility values (e.g. 2, 127, -1) behave strictly as boolean on valid board size (32x32)."""
        spec = export.classes()["micro"]
        model = nets.build(spec, memory="where").eval()

        H, W = 32, 32
        board = torch.zeros(1, N_PLANES, H, W, dtype=torch.int8)
        mem_in = torch.ones(1, N_MEMORY, H, W, dtype=torch.int8)

        # Extreme non-zero values on visibility plane
        board[0, 6, 0, 0] = 127
        board[0, 6, 1, 1] = -1
        board[0, 6, 2, 2] = 2

        with torch.no_grad():
            _, mem_out = model(board, mem_in)

        # Since obs on enemy hill plane is 0, any non-zero visibility cell must clear memory
        assert mem_out[0, 1, 0, 0] == 0
        assert mem_out[0, 1, 1, 1] == 0
        assert mem_out[0, 1, 2, 2] == 0
        # Fog cell (vis=0) must retain old memory (1)
        assert mem_out[0, 1, 31, 31] == 1


class TestDynamicShapesAndRolloutStress:
    """Stress-test model over dynamic board sizes and multi-turn autoregressive rollout."""

    @pytest.mark.parametrize("shape", [(24, 24), (36, 36), (48, 64), (64, 96), (128, 128), (32, 48)])
    def test_dynamic_board_forward_shapes(self, shape):
        H, W = shape
        spec = export.classes()["micro"]
        model = nets.build(spec, memory="where").eval()

        b = torch.randint(0, 2, (1, N_PLANES, H, W), dtype=torch.int8)
        m = torch.randint(0, 2, (1, N_MEMORY, H, W), dtype=torch.int8)

        with torch.no_grad():
            policy, mem_out = model(b, m)

        assert policy.shape == (1, 5, H, W), f"Expected policy shape (1, 5, {H}, {W}), got {policy.shape}"
        assert mem_out.shape == (1, N_MEMORY, H, W), f"Expected memory shape (1, {N_MEMORY}, {H}, {W}), got {mem_out.shape}"
        assert not torch.isnan(policy).any(), "Policy output contains NaN"
        assert not torch.isinf(policy).any(), "Policy output contains Inf"

    def test_150_turn_autoregressive_rollout_stability(self):
        """Feed sequential turns, passing mem_out back into mem_in for 150 turns without degradation."""
        spec = export.classes()["micro"]
        model = nets.build(spec, memory="where").eval()

        H, W = 48, 64
        mem = torch.zeros(1, N_MEMORY, H, W, dtype=torch.int8)

        for turn in range(150):
            board = torch.zeros(1, N_PLANES, H, W, dtype=torch.int8)
            # Simulate a moving agent revealing visibility circle
            cr, cc = (turn * 3) % H, (turn * 5) % W
            r_min, r_max = max(0, cr - 5), min(H, cr + 6)
            c_min, c_max = max(0, cc - 5), min(W, cc + 6)
            board[0, 6, r_min:r_max, c_min:c_max] = 1

            if turn == 10:
                # Enemy hill spotted at (20, 20): agent is looking at it (vis=1)
                board[0, 6, 20, 20] = 1
                board[0, 5, 20, 20] = 1
            elif turn == 50:
                # Enemy hill razed at (20, 20) while agent is still looking at it
                board[0, 6, 20, 20] = 1
                board[0, 5, 20, 20] = 0

            with torch.no_grad():
                policy, mem = model(board, mem)

            # Assert invariants at every turn
            assert mem.dtype == torch.int8, f"Turn {turn}: Memory dtype altered to {mem.dtype}"
            assert mem.shape == (1, N_MEMORY, H, W), f"Turn {turn}: Memory shape changed to {mem.shape}"
            assert torch.all((mem == 0) | (mem == 1)), f"Turn {turn}: Memory contains non-binary values"

            if 10 <= turn < 50:
                assert mem[0, 1, 20, 20] == 1, f"Turn {turn}: Expected enemy hill to be retained in memory"
            elif turn >= 50:
                assert mem[0, 1, 20, 20] == 0, f"Turn {turn}: Expected razed enemy hill to stay cleared"


class TestCLIExportPipelineIntegration:
    """Verify standard CLI export command executes and outputs valid model package."""

    def test_cli_export_command_and_check(self, tmp_path):
        out_dir = tmp_path / "micro_cli_export"
        cmd = [
            sys.executable, "-m", "tb_baselines.export",
            "--class", "micro",
            "--memory",  # Bare flag: defaults to where
            "--out", str(out_dir),
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, cwd=BASELINES_ROOT)
        assert res.returncode == 0, f"CLI export failed with code {res.returncode}: {res.stderr}\n{res.stdout}"

        # Assert all 4 artifact files exist
        assert (out_dir / "model.onnx").is_file(), "model.onnx missing from export directory"
        assert (out_dir / "manifest.json").is_file(), "manifest.json missing from export directory"
        assert (out_dir / "metrics.json").is_file(), "metrics.json missing from export directory"
        assert (out_dir / "card.md").is_file(), "card.md missing from export directory"

        # Assert metrics content
        metrics = json.loads((out_dir / "metrics.json").read_text())
        assert metrics.get("class") == "micro"
        assert metrics.get("size_metric_bytes") <= 131072
        assert metrics.get("infer_us_max") <= 700000
        assert "Where" in metrics.get("operators", [])
        assert "Max" not in metrics.get("operators", [])
