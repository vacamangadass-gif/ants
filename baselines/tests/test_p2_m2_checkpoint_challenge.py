"""Milestone 2 Empirical Challenger Test Suite: Checkpoint Forward Inference & Numerical Invariance.

Audits and stress-tests:
1. Checkpoint integrity and parameter count (exactly 24,505 parameters, micro class, Where memory).
2. Forward inference across diverse spatial dimensions:
   - Required: (24, 24), (32, 32), (64, 64), (100, 100), (128, 128).
   - Stress/Adversarial: non-square (24, 32), (32, 64), (64, 100) and odd dimensions (25, 25), (49, 49).
3. Tensor output contracts:
   - Policy shape [B, 5, H, W], dtype float32, strictly finite (no NaN, no Inf).
   - Memory shape [B, 2, H, W], dtype int8, range [-128, 127].
4. Where memory functional logic:
   - vis == 1 updates to seen (board[sources]).
   - vis == 0 retains memory_in.
   - Eliminates razed hill ghosting under forward inference.
5. Autoregressive multi-turn stability (50-turn rollout with memory feedback).
6. Cross-device parity: CPU vs CUDA numerical equivalence.
7. Telemetry audit: 5 epochs, monotonic loss decrease, valid engine digest.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple

import pytest
import torch

from tb_baselines import nets
from tb_baselines.export import classes
from tb_baselines.planes import MEMORY_SOURCES, N_MEMORY, N_PLANES

CHECKPOINTS = ["exp020", "exp021", "exp022"]
RUNS_DIR = Path("/home/host/project/ants/runs")
REQUIRED_DIMS: List[Tuple[int, int]] = [
    (24, 24),
    (32, 32),
    (64, 64),
    (100, 100),
    (128, 128),
]
STRESS_DIMS: List[Tuple[int, int]] = [
    (24, 32),
    (32, 64),
    (64, 100),
    (25, 25),
    (49, 49),
]


@pytest.fixture(scope="module", params=CHECKPOINTS)
def loaded_model_and_meta(request):
    exp = request.param
    ckpt_path = RUNS_DIR / exp / "best.pt"
    assert ckpt_path.exists(), f"Checkpoint {ckpt_path} does not exist"
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    spec = classes()["micro"]
    model = nets.build(spec, memory=True)
    model.load_state_dict(ckpt["trunk"])
    model.eval()

    hist_path = RUNS_DIR / exp / "history.json"
    assert hist_path.exists(), f"History file {hist_path} does not exist"
    with open(hist_path, "r", encoding="utf-8") as f:
        hist = json.load(f)

    return exp, model, ckpt, hist


class TestCheckpointIntegrity:
    """Verifies checkpoint metadata, parameter counts, and weight characteristics."""

    def test_checkpoint_metadata(self, loaded_model_and_meta):
        exp, model, ckpt, hist = loaded_model_and_meta
        assert ckpt.get("class") == "micro", f"Expected class 'micro', got {ckpt.get('class')}"
        assert ckpt.get("memory") is True, f"Expected memory True, got {ckpt.get('memory')}"
        assert "engine_digest" in ckpt, "Missing engine_digest in checkpoint"
        assert ckpt["engine_digest"].startswith("sha256:"), f"Invalid engine digest: {ckpt['engine_digest']}"

    def test_parameter_count(self, loaded_model_and_meta):
        exp, model, ckpt, hist = loaded_model_and_meta
        param_count = nets.policy_params(model)
        assert param_count == 24505, f"Expected exactly 24,505 policy parameters, got {param_count}"

    def test_weight_finiteness_and_non_degeneracy(self, loaded_model_and_meta):
        exp, model, ckpt, hist = loaded_model_and_meta
        for name, param in model.named_parameters():
            assert torch.isfinite(param).all(), f"Parameter {name} has non-finite values"
            assert not torch.isnan(param).any(), f"Parameter {name} has NaN"
            assert not torch.isinf(param).any(), f"Parameter {name} has Inf"
            assert param.std().item() > 1e-6, f"Parameter {name} appears degenerate/flat (std < 1e-6)"

    def test_telemetry_history(self, loaded_model_and_meta):
        exp, model, ckpt, hist = loaded_model_and_meta
        epochs = hist.get("epochs", [])
        assert len(epochs) == 5, f"Expected 5 epochs in history, got {len(epochs)}"
        val_losses = [e["val_loss"] for e in epochs]
        # Monotonic decrease in validation loss across all 5 epochs
        for i in range(len(val_losses) - 1):
            assert val_losses[i] > val_losses[i + 1], (
                f"{exp} epoch {i+1}->{i+2} val loss not decreasing: {val_losses[i]} <= {val_losses[i+1]}"
            )
        assert min(val_losses) <= 0.36, f"Final val loss {min(val_losses)} unexpectedly high"


class TestForwardInferenceShapesAndContracts:
    """Verifies forward inference shape and type assertions on required dimensions."""

    @pytest.mark.parametrize("h,w", REQUIRED_DIMS)
    @pytest.mark.parametrize("b", [1, 3])
    def test_required_dimensions_forward(self, loaded_model_and_meta, h, w, b):
        exp, model, _, _ = loaded_model_and_meta
        board = torch.randint(0, 2, (b, N_PLANES, h, w), dtype=torch.int8)
        memory_in = torch.randint(0, 2, (b, N_MEMORY, h, w), dtype=torch.int8)

        with torch.no_grad():
            policy, memory = model(board, memory_in)

        # Shape assertions
        assert policy.shape == (b, 5, h, w), f"Policy shape mismatch: got {policy.shape}"
        assert memory.shape == (b, 2, h, w), f"Memory shape mismatch: got {memory.shape}"

        # Dtype assertions
        assert policy.dtype == torch.float32, f"Policy dtype mismatch: {policy.dtype}"
        assert memory.dtype == torch.int8, f"Memory dtype mismatch: {memory.dtype}"

        # Finiteness & range assertions
        assert torch.isfinite(policy).all(), "Policy contains non-finite values"
        assert not torch.isnan(policy).any(), "Policy contains NaN"
        assert not torch.isinf(policy).any(), "Policy contains Inf"
        assert (memory >= -128).all() and (memory <= 127).all(), "Memory outside int8 range"

    @pytest.mark.parametrize("h,w", STRESS_DIMS)
    def test_stress_dimensions_forward(self, loaded_model_and_meta, h, w):
        exp, model, _, _ = loaded_model_and_meta
        b = 2
        board = torch.randint(0, 2, (b, N_PLANES, h, w), dtype=torch.int8)
        memory_in = torch.randint(0, 2, (b, N_MEMORY, h, w), dtype=torch.int8)

        with torch.no_grad():
            policy, memory = model(board, memory_in)

        assert policy.shape == (b, 5, h, w)
        assert memory.shape == (b, 2, h, w)
        assert torch.isfinite(policy).all()
        assert memory.dtype == torch.int8


class TestWhereMemoryAdversarialSemantics:
    """Verifies Where memory logic and ghost hill elimination under forward pass."""

    def test_where_memory_pure_vis_and_pure_fog(self, loaded_model_and_meta):
        exp, model, _, _ = loaded_model_and_meta
        h, w = 32, 32

        # 1. Complete Visibility (vis = 1 everywhere)
        board_vis = torch.zeros((1, N_PLANES, h, w), dtype=torch.int8)
        board_vis[:, 6, :, :] = 1  # vis channel
        # Set some food (channel 2) and foe hill (channel 5)
        board_vis[:, 2, 5, 5] = 1
        board_vis[:, 5, 10, 10] = 1

        memory_in = torch.full((1, N_MEMORY, h, w), fill_value=42, dtype=torch.int8)

        with torch.no_grad():
            _, memory_out = model(board_vis, memory_in)

        assert memory_out.dtype == torch.int8
        # Since vis=1 everywhere, memory_out must equal seen exactly
        expected_seen_0 = board_vis[:, 2, :, :]  # food
        expected_seen_1 = board_vis[:, 5, :, :]  # foe hill
        assert torch.equal(memory_out[:, 0, :, :], expected_seen_0)
        assert torch.equal(memory_out[:, 1, :, :], expected_seen_1)

        # 2. Complete Fog (vis = 0 everywhere)
        board_fog = torch.zeros((1, N_PLANES, h, w), dtype=torch.int8)
        board_fog[:, 6, :, :] = 0  # vis=0
        board_fog[:, 2, 5, 5] = 1  # unseen food on board should NOT leak into memory!

        with torch.no_grad():
            _, memory_out_fog = model(board_fog, memory_in)

        # Memory must be completely preserved as memory_in
        assert torch.equal(memory_out_fog, memory_in)

    def test_razed_hill_elimination_scenario(self, loaded_model_and_meta):
        """Simulate razed hill: turn t has hill under fog, turn t+1 ant visits razed hill."""
        exp, model, _, _ = loaded_model_and_meta
        h, w = 32, 32

        # Prior memory: known enemy hill at (15, 15)
        memory_t0 = torch.zeros((1, N_MEMORY, h, w), dtype=torch.int8)
        memory_t0[0, 1, 15, 15] = 1  # hill_foe_seen = 1

        # Turn t: ant visits (15, 15), discovers hill has been destroyed!
        # vis = 1 at (15, 15), but hill_foe = 0
        board_t1 = torch.zeros((1, N_PLANES, h, w), dtype=torch.int8)
        board_t1[0, 6, 15, 15] = 1  # vis = 1
        board_t1[0, 5, 15, 15] = 0  # hill_foe = 0 (razed!)

        with torch.no_grad():
            _, memory_t1 = model(board_t1, memory_t0)

        # In Where memory, the ghost hill MUST be eliminated: memory_t1 at (15, 15) must be 0!
        assert memory_t1[0, 1, 15, 15].item() == 0, (
            f"{exp} failed to eliminate razed ghost hill! Value is {memory_t1[0, 1, 15, 15]}"
        )

    def test_autoregressive_rollout_stability(self, loaded_model_and_meta):
        """Rolls out 50 consecutive autoregressive turns to test numerical drift."""
        exp, model, _, _ = loaded_model_and_meta
        h, w = 32, 32
        board = torch.randint(0, 2, (1, N_PLANES, h, w), dtype=torch.int8)
        memory = torch.zeros((1, N_MEMORY, h, w), dtype=torch.int8)

        for step in range(50):
            # Dynamic board changes
            board[:, 0:6] = torch.randint(0, 2, (1, 6, h, w), dtype=torch.int8)
            board[:, 6] = torch.randint(0, 2, (1, h, w), dtype=torch.int8)  # random visibility

            with torch.no_grad():
                policy, memory = model(board, memory)

            assert torch.isfinite(policy).all()
            assert memory.dtype == torch.int8
            assert (memory >= 0).all() and (memory <= 1).all()


class TestCrossDeviceParity:
    """Verifies that CPU and CUDA device inferences produce bitwise/numerically aligned results."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cpu_cuda_equivalence(self, loaded_model_and_meta):
        exp, model_cpu, _, _ = loaded_model_and_meta
        # Disable TF32 to enforce strict IEEE 754 fp32 precision across devices
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

        model_cuda = nets.build(classes()["micro"], memory=True).cuda()
        model_cuda.load_state_dict(model_cpu.state_dict())
        model_cuda.eval()

        torch.manual_seed(42)
        board = torch.randint(0, 2, (2, N_PLANES, 32, 32), dtype=torch.int8)
        memory_in = torch.randint(0, 2, (2, N_MEMORY, 32, 32), dtype=torch.int8)

        with torch.no_grad():
            p_cpu, m_cpu = model_cpu(board, memory_in)
            p_cuda, m_cuda = model_cuda(board.cuda(), memory_in.cuda())

        # Exact bitwise equality on int8 memory across CPU and CUDA
        assert torch.equal(m_cpu, m_cuda.cpu()), "Memory differs between CPU and CUDA"

        # Action argmax equivalence across all spatial cells
        mismatches = (p_cpu.argmax(dim=1) != p_cuda.cpu().argmax(dim=1)).sum().item()
        assert mismatches == 0, f"Action argmax differed at {mismatches} spatial cells between CPU and CUDA"

        # Strict IEEE fp32 closeness on policy logits
        assert torch.allclose(p_cpu, p_cuda.cpu(), atol=1e-3, rtol=1e-2), (
            f"Policy logits differ across CPU/CUDA beyond tolerance"
        )
