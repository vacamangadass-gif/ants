"""Verification suite for Milestone 1: Micro Class Architecture Support & Parity Gate.

Covers 7 comprehensive test suites:
1. test_micro_specification: Validate classes.toml micro spec (channels=28, blocks=3, arch=trunk, 128KB limit, target_params=24001).
2. test_micro_parameter_counts: Exact verification of 24,001 params (no memory) and 24,505 params (with Where memory), plus layer breakdown.
3. test_micro_receptive_field: Verify dilations [1, 2, 4, 8], reach 15, 31x31 span, and empirical gradient backward test on 64x64 grid.
4. test_micro_dynamic_board_shapes: Forward pass across sizes 24x24 to 128x128.
5. test_micro_where_memory_semantics: Verify Where clears razed hills and retains fog memory; bitwise numpy equivalence.
6. test_micro_onnx_export_and_quantization: Verify export with halve and shorten, opset 17, Where op present, Max op absent, size <= 131,072 bytes.
7. test_micro_platform_certification: Verify tinybrains check passes (ok == True, round_trip.failed == 0, latency <= 700ms).
"""

from __future__ import annotations

import argparse
import os
import shutil
import warnings

import numpy as np
import pytest

from tb_baselines import adapters, export, nets
from tb_baselines.train import bc

# Filter non-actionable environment warnings (e.g. sandbox GPU access probe, torch legacy exporter)
warnings.filterwarnings("ignore", category=UserWarning, message=".*CUDA initialization.*")
warnings.filterwarnings("ignore", category=DeprecationWarning, message=".*legacy TorchScript-based ONNX export.*")


# ---- Fixtures (module-scoped at top level to prevent pytest deprecation warnings) ----

@pytest.fixture(scope="module")
def exported_micro(tmp_path_factory):
    """Export and optimize a Micro class model with Where memory for ONNX and platform tests."""
    pytest.importorskip("torch")
    pytest.importorskip("onnx")

    tmp_dir = tmp_path_factory.mktemp("micro_export")
    spec = export.classes()["micro"]
    model = nets.build(spec, memory="where")
    onnx_path = tmp_dir / "model.onnx"
    manifest_path = tmp_dir / "manifest.json"
    manifest_path.write_text(adapters.dumps(memory=True))

    export.to_onnx(model, onnx_path, planes=7)
    export.halve(onnx_path)
    export.shorten(onnx_path)
    return onnx_path, manifest_path


# ---- Suite 1: Specification Verification ----

class TestMicroSpecification:
    def test_classes_toml_micro_entry(self):
        table = export.classes()
        assert "micro" in table, "micro class missing from classes.toml"
        spec = table["micro"]
        assert spec["arch"] == "trunk", f"Expected arch 'trunk', got '{spec.get('arch')}'"
        assert spec["channels"] == 28, f"Expected 28 channels, got {spec.get('channels')}"
        assert spec["blocks"] == 3, f"Expected 3 blocks, got {spec.get('blocks')}"
        assert spec["max_bytes"] == 131072, f"Expected max_bytes 131072, got {spec.get('max_bytes')}"
        assert spec["target_params"] == 24001, f"Expected target_params 24001, got {spec.get('target_params')}"

    def test_micro_budget_fraction(self):
        b = export.budget("micro")
        assert b["max_bytes"] == 131072
        assert b["fraction"] == 0.70
        assert b["target_bytes"] == int(131072 * 0.70)


# ---- Suite 2: Parameter Counts Verification ----

class TestMicroParameterCount:
    def test_micro_plain_parameter_count(self):
        pytest.importorskip("torch")
        spec = export.classes()["micro"]
        net = nets.build(spec, memory=False)
        params = nets.policy_params(net)
        assert params == 24001, f"Expected exactly 24,001 parameters without memory, got {params}"

    def test_micro_where_memory_parameter_count(self):
        pytest.importorskip("torch")
        spec = export.classes()["micro"]
        net = nets.build(spec, memory="where")
        params = nets.policy_params(net)
        assert params == 24505, f"Expected exactly 24,505 parameters with Where memory, got {params}"

    def test_micro_layer_by_layer_breakdown(self):
        pytest.importorskip("torch")
        spec = export.classes()["micro"]
        net = nets.build(spec, memory="where")
        convs = net.trunk.stage.convs
        assert len(convs) == 4, f"Expected 4 stage conv layers, got {len(convs)}"
        # Layer 0: 9 -> 28, 3x3 conv
        w0 = convs[0].weight.numel() + convs[0].bias.numel()
        assert w0 == 28 * 9 * 9 + 28, f"Expected conv0 to have 2,296 parameters, got {w0}"
        assert w0 == 2296

        # Layers 1-3: 28 -> 28, 3x3 conv
        for i, c in enumerate(convs[1:], start=1):
            wi = c.weight.numel() + c.bias.numel()
            assert wi == 28 * 28 * 9 + 28, f"Expected conv{i} to have 7,084 parameters, got {wi}"
            assert wi == 7084

        # Mix: 28 -> 28, 1x1 conv
        wmix = net.trunk.mix.weight.numel() + net.trunk.mix.bias.numel()
        assert wmix == 28 * 28 * 1 + 28, f"Expected mix to have 812 parameters, got {wmix}"
        assert wmix == 812

        # Head: 28 -> 5, 1x1 conv
        whead = net.trunk.head.weight.numel() + net.trunk.head.bias.numel()
        assert whead == 5 * 28 * 1 + 5, f"Expected head to have 145 parameters, got {whead}"
        assert whead == 145

        total = w0 + 3 * 7084 + wmix + whead
        assert total == 24505, f"Sum of layer parameters must equal 24,505, got {total}"


# ---- Suite 3: Receptive Field Verification ----

class TestMicroReceptiveField:
    def test_analytical_reach_and_window(self):
        pytest.importorskip("torch")
        spec = export.classes()["micro"]
        net = nets.build(spec, memory="where")
        stage = net.trunk.stage
        assert stage.dilations == [1, 2, 4, 8], f"Expected dilations [1, 2, 4, 8], got {stage.dilations}"
        assert stage.reach == 15, f"Expected stage reach 15, got {stage.reach}"
        rf_window = 2 * stage.reach + 1
        assert rf_window == 31, f"Receptive field must be 31x31, got {rf_window}x{rf_window}"

    def test_empirical_gradient_receptive_field(self):
        torch = pytest.importorskip("torch")
        spec = export.classes()["micro"]
        model = nets.build(spec, memory="where").eval()
        H, W = 64, 64
        x = torch.randn(1, 9, H, W, requires_grad=True, device="cpu")
        feat = model.trunk.mix(model.trunk.stage(x))

        # Center target pixel at (32, 32)
        target = feat[0, :, 32, 32].sum()
        target.backward()

        grad_mag = x.grad[0].abs().sum(dim=0)
        active_r, active_c = torch.where(grad_mag > 1e-9)

        min_r, max_r = active_r.min().item(), active_r.max().item()
        min_c, max_c = active_c.min().item(), active_c.max().item()

        assert min_r == 17 and max_r == 47, f"Row active bounds must be [17, 47], got [{min_r}, {max_r}]"
        assert min_c == 17 and max_c == 47, f"Col active bounds must be [17, 47], got [{min_c}, {max_c}]"

        max_r_reach = max(32 - min_r, max_r - 32)
        max_c_reach = max(32 - min_c, max_c - 32)
        assert max_r_reach == 15, f"Expected row reach 15, got {max_r_reach}"
        assert max_c_reach == 15, f"Expected col reach 15, got {max_c_reach}"

        span_r = max_r - min_r + 1
        span_c = max_c - min_c + 1
        assert span_r == 31, f"Expected row span 31, got {span_r}"
        assert span_c == 31, f"Expected col span 31, got {span_c}"


# ---- Suite 4: Dynamic Board Shapes Verification ----

class TestMicroDynamicBoardShapes:
    @pytest.mark.parametrize("H,W", [(24, 24), (36, 36), (48, 64), (64, 96), (128, 128)])
    def test_forward_pass_shapes(self, H: int, W: int):
        torch = pytest.importorskip("torch")
        spec = export.classes()["micro"]
        model = nets.build(spec, memory="where").eval()
        b = torch.randint(0, 2, (1, 7, H, W), dtype=torch.int8)
        m = torch.randint(0, 2, (1, 2, H, W), dtype=torch.int8)
        with torch.no_grad():
            pol, mem_out = model(b, m)
        assert pol.shape == (1, 5, H, W), f"Expected policy shape (1, 5, {H}, {W}), got {pol.shape}"
        assert mem_out.shape == (1, 2, H, W), f"Expected memory shape (1, 2, {H}, {W}), got {mem_out.shape}"
        assert mem_out.dtype == torch.int8, f"Expected int8 memory dtype, got {mem_out.dtype}"


# ---- Suite 5: Where Memory Semantics Verification ----

class TestMicroWhereMemorySemantics:
    def test_where_memory_clears_razed_hill_in_view(self):
        torch = pytest.importorskip("torch")
        spec = export.classes()["micro"]
        model = nets.build(spec, memory="where").eval()

        # [1, 7, 16, 16] board:
        # cell (5, 5) visible (plane 6=1) and empty of enemy hills (plane 5=0)
        # cell (12, 12) in fog (plane 6=0)
        b = torch.zeros(1, 7, 16, 16, dtype=torch.int8)
        b[0, 6, 5, 5] = 1    # visible
        b[0, 6, 12, 12] = 0  # fog

        # memory_in has enemy hills at (5, 5) and (12, 12) on plane 1 (hill_foe_seen)
        m = torch.zeros(1, 2, 16, 16, dtype=torch.int8)
        m[0, 1, 5, 5] = 1
        m[0, 1, 12, 12] = 1

        with torch.no_grad():
            _, mem_out = model(b, m)

        assert mem_out[0, 1, 5, 5] == 0, "Razed enemy hill in view must be cleared to 0 (no ghost hill)"
        assert mem_out[0, 1, 12, 12] == 1, "Enemy hill in fog must be retained as 1 from memory"

    def test_where_memory_bitwise_equivalence(self):
        torch = pytest.importorskip("torch")
        spec = export.classes()["micro"]
        model = nets.build(spec, memory="where").eval()

        b = torch.randint(0, 2, (2, 7, 32, 32), dtype=torch.int8)
        m = torch.randint(0, 2, (2, 2, 32, 32), dtype=torch.int8)

        with torch.no_grad():
            _, mem_out = model(b, m)

        vis = (b[:, 6:7] != 0).numpy()
        sources = b[:, [2, 5]].numpy()
        want = np.where(vis, sources, m.numpy()).astype(np.int8)
        assert np.array_equal(mem_out.numpy(), want), "WithMemory.write() must bitwise match np.where(vis, sources, m)"


# ---- Suite 6: ONNX Export and Quantization Verification ----

class TestMicroExportAndPlatformCertification:
    def test_micro_onnx_size_under_128kb(self, exported_micro):
        onnx_path, manifest_path = exported_micro
        total_size = onnx_path.stat().st_size + manifest_path.stat().st_size
        assert total_size <= 131072, f"Total metric size {total_size} bytes exceeds 128KB (131,072 bytes)"
        assert onnx_path.stat().st_size > 16384, f"Micro model size {onnx_path.stat().st_size} bytes should exceed Nano (16KB)"

    def test_micro_onnx_operators(self, exported_micro):
        onnx = pytest.importorskip("onnx")
        onnx_path, _ = exported_micro
        model = onnx.load(str(onnx_path))
        opset_versions = [entry.version for entry in model.opset_import if entry.domain in ("", "ai.onnx")]
        assert 17 in opset_versions, f"Expected ONNX opset 17, got {opset_versions}"

        ops = set(n.op_type for n in model.graph.node)
        assert "Where" in ops, "ONNX graph must contain 'Where' operator"
        assert "Max" not in ops, "ONNX graph must NOT contain 'Max' operator"
        assert {"Conv", "Relu", "Cast", "Slice", "Concat"}.issubset(ops), f"Missing core operators: {ops}"

    def test_micro_tinybrains_check_verdict(self, exported_micro):
        if not (shutil.which("tinybrains") or os.environ.get("TINYBRAINS")):
            pytest.skip("tinybrains binary not available on PATH")
        onnx_path, manifest_path = exported_micro
        v = export.verdict(onnx_path, manifest_path)
        assert v.get("ok") is True, f"tinybrains check failed: {v.get('reason')}"
        rt = v.get("memory", {}).get("round_trip", {})
        assert rt.get("failed") == 0, f"Memory round trip failed {rt.get('failed')} times across {rt.get('checked')} cases"
        assert v.get("infer_us_max") <= 700000, f"Latency exceeds 700ms limit: {v.get('infer_us_max')} us"
        problems = export.certify("micro", v)
        assert problems == [], f"Platform certification problems reported: {problems}"


# ---- Suite 7: CLI Integration & Default Argument Verification ----

class TestMicroTrainingCLIIntegration:
    def test_bc_cli_override_micro(self):
        pytest.importorskip("torch")
        spec = export.classes()["micro"]
        args = argparse.Namespace(cls="micro", arch=None, channels=None, blocks=None, memory=True)
        overridden = bc.override(spec, args)
        assert overridden["arch"] == "trunk"
        assert overridden["channels"] == 28
        assert overridden["blocks"] == 3
        model = nets.build(overridden, memory=args.memory)
        assert nets.policy_params(model) == 24505
        assert model.trunk.channels == 28

    def test_export_cli_memory_default_where(self):
        """Verify export CLI parser correctly defaults bare --memory to 'where'."""
        parser = export.build_parser()

        # a) Bare flag: defaults to "where"
        args_bare = parser.parse_args(["--class", "micro", "--out", "out_dir", "--memory"])
        assert args_bare.memory == "where", f"Expected 'where' default for bare --memory, got {args_bare.memory}"

        # b) Omitted flag: defaults to False (preserves state dict / plain model export)
        args_omitted = parser.parse_args(["--class", "micro", "--out", "out_dir"])
        assert args_omitted.memory is False, f"Expected False when --memory is omitted, got {args_omitted.memory}"

        # c) Explicit max: overrides to "max"
        args_max = parser.parse_args(["--class", "micro", "--out", "out_dir", "--memory", "max"])
        assert args_max.memory == "max", f"Expected 'max' for explicit --memory max, got {args_max.memory}"

        # d) Explicit where: overrides to "where"
        args_where = parser.parse_args(["--class", "micro", "--out", "out_dir", "--memory", "where"])
        assert args_where.memory == "where", f"Expected 'where' for explicit --memory where, got {args_where.memory}"

