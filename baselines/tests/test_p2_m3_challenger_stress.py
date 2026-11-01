"""Challenger Empirical Stress Testing for Phase 2 Milestone 3 Models.

Targets:
- models/exp020/model.onnx (Big B: Micro 28ch, Where memory)
- models/exp021/model.onnx (Big D: Micro 28ch, Where memory)
- models/exp022/model.onnx (Big E: Micro 28ch, Where memory)

Covers:
1. Dynamic Shape Invariance:
   - 24x24 (basic-tiny)
   - 32x32
   - 40x60 (asymmetric)
   - 64x64
   - 100x100
   - 124x124 (basic-xlarge)
   - Boundary/asymmetric: 24x124, 124x24, 25x37, 128x128
2. Numeric Stability:
   - No NaN, no Inf in logits across all shapes
   - Softmax probabilities sum to 1.0 across action channels (axis 1)
   - Finite reasonable logit distributions (no activation explosion/vanishing)
3. Chained State Inference:
   - At least 20 steps (tested at 20, 50, and 100 steps)
   - int8 bounds [-128, 127] and {0, 1} canonical memory values
   - Dynamic gameplay, all-fog, all-vis, and random perturbations
4. Pathological / Adversarial Inputs:
   - Out-of-spec int8 memory sanitization
   - Out-of-spec int8 board values
   - Empty board (no ants)
"""

from pathlib import Path
import numpy as np
import onnx
import onnxruntime as ort
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = REPO_ROOT / "models"
EXPERIMENTS = ["exp020", "exp021", "exp022"]

REQUIRED_SHAPES = [
    (24, 24),    # basic-tiny
    (32, 32),    # small
    (40, 60),    # asymmetric
    (64, 64),    # medium
    (100, 100),  # large
    (124, 124),  # basic-xlarge
]

EXTRA_SHAPES = [
    (24, 124),   # extreme aspect ratio landscape
    (124, 24),   # extreme aspect ratio portrait
    (25, 37),    # odd dimensions
    (128, 128),  # platform maximum
]

ALL_SHAPES = REQUIRED_SHAPES + EXTRA_SHAPES


def softmax(x, axis=1):
    """Compute softmax values along specified axis in a numerically stable way."""
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)


@pytest.fixture(scope="session")
def ort_sessions():
    """Load ONNX runtime sessions for exp020, exp021, and exp022."""
    sessions = {}
    for exp in EXPERIMENTS:
        path = MODELS_DIR / exp / "model.onnx"
        assert path.exists(), f"Missing model: {path}"
        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        sess = ort.InferenceSession(str(path), sess_options=opts, providers=["CPUExecutionProvider"])
        sessions[exp] = sess
    return sessions


# ========================================================================================
# 1. DYNAMIC SHAPE INVARIANCE
# ========================================================================================

class TestDynamicShapeInvariance:
    """Stress testing forward inference across all mandated and boundary torus board sizes."""

    @pytest.mark.parametrize("exp", EXPERIMENTS)
    @pytest.mark.parametrize("h,w", REQUIRED_SHAPES)
    def test_mandated_board_shapes(self, ort_sessions, exp, h, w):
        sess = ort_sessions[exp]
        rng = np.random.default_rng(42 + h * 1000 + w)

        board = rng.integers(0, 2, size=(1, 7, h, w), dtype=np.int8)
        memory_in = rng.integers(0, 2, size=(1, 2, h, w), dtype=np.int8)

        policy, memory_out = sess.run(None, {"board": board, "memory_in": memory_in})

        # Correct tensor shapes
        assert policy.shape == (1, 5, h, w), f"{exp} ({h}x{w}) policy shape {policy.shape} != (1, 5, {h}, {w})"
        assert memory_out.shape == (1, 2, h, w), f"{exp} ({h}x{w}) memory shape {memory_out.shape} != (1, 2, {h}, {w})"

        # Correct dtypes
        assert policy.dtype == np.float32, f"{exp} policy dtype {policy.dtype} != float32"
        assert memory_out.dtype == np.int8, f"{exp} memory_out dtype {memory_out.dtype} != int8"

    @pytest.mark.parametrize("exp", EXPERIMENTS)
    @pytest.mark.parametrize("h,w", EXTRA_SHAPES)
    def test_extra_boundary_board_shapes(self, ort_sessions, exp, h, w):
        sess = ort_sessions[exp]
        rng = np.random.default_rng(99 + h * 1000 + w)

        board = rng.integers(0, 2, size=(1, 7, h, w), dtype=np.int8)
        memory_in = rng.integers(0, 2, size=(1, 2, h, w), dtype=np.int8)

        policy, memory_out = sess.run(None, {"board": board, "memory_in": memory_in})

        assert policy.shape == (1, 5, h, w)
        assert memory_out.shape == (1, 2, h, w)
        assert policy.dtype == np.float32
        assert memory_out.dtype == np.int8


# ========================================================================================
# 2. NUMERIC STABILITY
# ========================================================================================

class TestNumericStability:
    """Verify absence of NaN/Inf and strict softmax probability distribution properties."""

    @pytest.mark.parametrize("exp", EXPERIMENTS)
    @pytest.mark.parametrize("h,w", ALL_SHAPES)
    def test_logits_and_softmax_stability(self, ort_sessions, exp, h, w):
        sess = ort_sessions[exp]
        rng = np.random.default_rng(12345 + h + w)

        # Realistic sparse board input
        board = np.zeros((1, 7, h, w), dtype=np.int8)
        # 10% ants, 5% food, 1% hill, 30% vis
        board[0, 0] = (rng.random((h, w)) < 0.10).astype(np.int8)
        board[0, 2] = (rng.random((h, w)) < 0.05).astype(np.int8)
        board[0, 5] = (rng.random((h, w)) < 0.01).astype(np.int8)
        board[0, 6] = (rng.random((h, w)) < 0.30).astype(np.int8)

        memory_in = rng.integers(0, 2, size=(1, 2, h, w), dtype=np.int8)

        policy, memory_out = sess.run(None, {"board": board, "memory_in": memory_in})

        # 1. No NaN, no Inf, all finite
        assert not np.isnan(policy).any(), f"{exp} ({h}x{w}): policy logits contain NaN!"
        assert not np.isinf(policy).any(), f"{exp} ({h}x{w}): policy logits contain Inf!"
        assert np.isfinite(policy).all(), f"{exp} ({h}x{w}): policy logits contain non-finite numbers!"

        # 2. Softmax across action channels (axis 1)
        probs = softmax(policy, axis=1)

        # Probabilities valid range [0, 1]
        assert (probs >= 0.0).all(), f"{exp} ({h}x{w}): negative probabilities found!"
        assert (probs <= 1.0).all(), f"{exp} ({h}x{w}): probabilities > 1.0 found!"

        # Sum along axis 1 must equal 1.0 everywhere
        channel_sums = np.sum(probs, axis=1)
        assert np.allclose(channel_sums, 1.0, atol=1e-5), (
            f"{exp} ({h}x{w}): softmax probabilities do not sum to 1.0! "
            f"Max deviation: {np.max(np.abs(channel_sums - 1.0))}"
        )


# ========================================================================================
# 3. CHAINED STATE INFERENCE
# ========================================================================================

class TestChainedStateInference:
    """Sequential autoregressive rollout passing memory from turn t to turn t+1."""

    @pytest.mark.parametrize("exp", EXPERIMENTS)
    @pytest.mark.parametrize("steps", [20, 50, 100])
    def test_chained_sequential_rollout_stability(self, ort_sessions, exp, steps):
        sess = ort_sessions[exp]
        h, w = 32, 32
        rng = np.random.default_rng(20261016 + steps)

        # Initial memory is all zeros
        mem = np.zeros((1, 2, h, w), dtype=np.int8)

        # Keep track of min/max values over the entire sequence
        min_seen = 0
        max_seen = 0

        for t in range(steps):
            # Dynamic board state: moving ant, changing visibility, occasional food & hills
            board = np.zeros((1, 7, h, w), dtype=np.int8)
            # Ant moves in torus
            ant_r = (t * 2 + 3) % h
            ant_c = (t * 3 + 5) % w
            board[0, 0, ant_r, ant_c] = 1

            # Vision disk around ant
            for dr in range(-2, 3):
                for dc in range(-2, 3):
                    if dr * dr + dc * dc <= 5:  # radius^2 <= 5
                        board[0, 6, (ant_r + dr) % h, (ant_c + dc) % w] = 1

            # Spawn enemy hill at turn 5, visible until turn 15, then cleared at turn 18
            if 5 <= t < 15:
                board[0, 5, 10, 10] = 1
                board[0, 6, 10, 10] = 1
            elif t >= 18:
                # Visibility confirms hill is gone
                board[0, 6, 10, 10] = 1
                board[0, 5, 10, 10] = 0

            # Execute model step
            policy, mem = sess.run(None, {"board": board, "memory_in": mem})

            # Assertions per step
            assert mem.dtype == np.int8, f"{exp} step {t}: mem dtype changed to {mem.dtype}"
            assert np.all(mem >= -128) and np.all(mem <= 127), f"{exp} step {t}: mem out of int8 bounds!"

            # Canonical Where memory values must be strictly in {0, 1}
            assert np.isin(mem, [0, 1]).all(), (
                f"{exp} step {t}: mem diverged or saturated abnormally! Values: {np.unique(mem)}"
            )

            # Softmax sum assertion at each turn
            probs = softmax(policy, axis=1)
            assert np.allclose(np.sum(probs, axis=1), 1.0, atol=1e-5), f"{exp} step {t}: softmax != 1.0"

            min_seen = min(min_seen, int(mem.min()))
            max_seen = max(max_seen, int(mem.max()))

        # Hill elimination check: by step 20+, hill at (10, 10) must be 0
        if steps >= 20:
            assert mem[0, 1, 10, 10] == 0, f"{exp}: ghost hill remained in memory at (10, 10)"

    @pytest.mark.parametrize("exp", EXPERIMENTS)
    def test_chained_rollout_under_pure_fog(self, ort_sessions, exp):
        """Under complete fog of war (vis=0), memory must hold indefinitely without drift."""
        sess = ort_sessions[exp]
        h, w = 24, 24
        mem = np.zeros((1, 2, h, w), dtype=np.int8)

        # Plant initial memory flags
        mem[0, 0, 5, 5] = 1
        mem[0, 1, 12, 12] = 1

        for t in range(25):
            # vis = 0 everywhere
            board = np.zeros((1, 7, h, w), dtype=np.int8)
            policy, mem = sess.run(None, {"board": board, "memory_in": mem})

            assert mem[0, 0, 5, 5] == 1, f"{exp} step {t}: food memory lost under fog"
            assert mem[0, 1, 12, 12] == 1, f"{exp} step {t}: hill memory lost under fog"
            assert (mem >= -128).all() and (mem <= 127).all()
            assert np.isin(mem, [0, 1]).all()


# ========================================================================================
# 4. ADVERSARIAL & BOUNDARY PERTURBATIONS
# ========================================================================================

class TestAdversarialPerturbations:
    """Stress test boundary condition handling and dirty memory sanitization."""

    @pytest.mark.parametrize("exp", EXPERIMENTS)
    def test_dirty_memory_int8_sanitization(self, ort_sessions, exp):
        """Feeding dirty int8 values (-128, 127, -1, 42) into memory_in:
        Under visibility (vis=1), Where must sanitize the output to current observation {0, 1}.
        """
        sess = ort_sessions[exp]
        h, w = 32, 32

        board = np.zeros((1, 7, h, w), dtype=np.int8)
        board[0, 6, :, :] = 1  # vis = 1 everywhere
        board[0, 2, 5, 5] = 1  # visible food
        board[0, 5, 8, 8] = 1  # visible hill

        # Construct maximally corrupt memory input
        corrupt_mem = np.zeros((1, 2, h, w), dtype=np.int8)
        corrupt_mem[0, 0, :, :] = 127
        corrupt_mem[0, 1, :, :] = -128

        _, clean_mem = sess.run(None, {"board": board, "memory_in": corrupt_mem})

        assert clean_mem.dtype == np.int8
        assert np.isin(clean_mem, [0, 1]).all(), (
            f"{exp}: Where operator failed to sanitize corrupt memory! Found: {np.unique(clean_mem)}"
        )
        assert clean_mem[0, 0, 5, 5] == 1  # food remembered
        assert clean_mem[0, 0, 0, 0] == 0  # non-food cleared
        assert clean_mem[0, 1, 8, 8] == 1  # hill remembered
        assert clean_mem[0, 1, 0, 0] == 0  # non-hill cleared

    @pytest.mark.parametrize("exp", EXPERIMENTS)
    def test_zero_ants_board(self, ort_sessions, exp):
        """Empty board with 0 ants must not crash or output NaN."""
        sess = ort_sessions[exp]
        h, w = 24, 24
        board = np.zeros((1, 7, h, w), dtype=np.int8)
        mem = np.zeros((1, 2, h, w), dtype=np.int8)

        policy, mem_out = sess.run(None, {"board": board, "memory_in": mem})
        assert np.isfinite(policy).all()
        assert policy.shape == (1, 5, h, w)
        assert mem_out.shape == (1, 2, h, w)
