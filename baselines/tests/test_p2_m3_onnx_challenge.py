"""Empirical Forward Inference & Where-Memory Dynamics Stress Challenge for Phase 2 M3.

Target Models:
- models/exp020/model.onnx (Big B: Micro 28ch + Where memory, default lead=6)
- models/exp021/model.onnx (Big D: Micro 28ch + Where memory, aggressive lead=12)
- models/exp022/model.onnx (Big E: Micro 28ch + Where memory, dynamic compound teacher)

This test suite verifies:
1. Dynamic spatial grid dimensions: (24, 24), (32, 32), (64, 64), (100, 100), (128, 128), and non-square (40, 60), (60, 40), (24, 124).
2. Dynamic batch dimensions: B in [1, 2, 4].
3. Numerical stability: No NaN, no Inf, finite policy logits, valid int8 memory ranges.
4. Where memory mechanics:
   - Visible enemy hill -> memory plane 1 updates to 1.
   - Fog of war -> memory plane 1 retains 1.
   - Ant steps on hill / destroys it -> Where logic clears memory plane 1 to 0 (eliminates ghost hill).
   - Food lifecycle in memory plane 0: visible -> fog retention -> gathered -> cleared.
5. Multi-object multi-turn closed-loop autoregressive stability (50 turns).
6. Pathological inputs: all-zero, all-one, extreme int8 values.
7. Behavioral distinctness: verified different policy logits among exp020, exp021, and exp022.
"""

from pathlib import Path
import numpy as np
import onnx
import onnxruntime as ort
import pytest

MODELS_DIR = Path(__file__).resolve().parent.parent.parent / "models"
EXPERIMENTS = ("exp020", "exp021", "exp022")

GRID_DIMENSIONS = [
    (24, 24),    # Minimal allowed platform board
    (32, 32),    # Standard benchmark small board
    (64, 64),    # Standard benchmark medium board
    (100, 100),  # Standard benchmark large board
    (128, 128),  # Maximum allowed square board
    (40, 60),    # Non-square (portrait)
    (60, 40),    # Non-square (landscape)
    (24, 124),   # Extreme non-square boundary
]


@pytest.fixture(scope="session")
def onnx_sessions():
    """Load ONNX runtime sessions for all three models."""
    sessions = {}
    for exp in EXPERIMENTS:
        path = MODELS_DIR / exp / "model.onnx"
        assert path.exists(), f"Model file missing: {path}"
        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        sess = ort.InferenceSession(str(path), sess_options=opts, providers=["CPUExecutionProvider"])
        sessions[exp] = sess
    return sessions


# ========================================================================================
# 1. GRAPH STRUCTURE & OPERATOR INSPECTION
# ========================================================================================

class TestONNXGraphStructure:
    """Verify input/output metadata and operator whitelist conformance."""

    @pytest.mark.parametrize("exp", EXPERIMENTS)
    def test_graph_io_signatures(self, exp):
        path = MODELS_DIR / exp / "model.onnx"
        model = onnx.load(str(path))
        graph = model.graph

        inputs = {inp.name: inp for inp in graph.input}
        assert "board" in inputs, f"{exp}: missing 'board' input"
        assert "memory_in" in inputs, f"{exp}: missing 'memory_in' input"

        # Check shapes: board should be [B, 7, H, W] int8 (elem_type=3)
        b_shape = [d.dim_param or d.dim_value for d in inputs["board"].type.tensor_type.shape.dim]
        assert len(b_shape) == 4
        assert b_shape[1] == 7, f"{exp}: board channel dimension must be 7, got {b_shape[1]}"
        assert inputs["board"].type.tensor_type.elem_type == onnx.TensorProto.INT8

        # memory_in should be [B, 2, H, W] int8 (elem_type=3)
        m_shape = [d.dim_param or d.dim_value for d in inputs["memory_in"].type.tensor_type.shape.dim]
        assert len(m_shape) == 4
        assert m_shape[1] == 2, f"{exp}: memory_in channel dimension must be 2, got {m_shape[1]}"
        assert inputs["memory_in"].type.tensor_type.elem_type == onnx.TensorProto.INT8

        outputs = {out.name: out for out in graph.output}
        assert "policy" in outputs, f"{exp}: missing 'policy' output"
        assert "memory" in outputs, f"{exp}: missing 'memory' output"

        # policy should be [B, 5, H, W] float32 (elem_type=1)
        p_shape = [d.dim_param or d.dim_value for d in outputs["policy"].type.tensor_type.shape.dim]
        assert len(p_shape) == 4
        assert p_shape[1] == 5, f"{exp}: policy channel dimension must be 5, got {p_shape[1]}"
        assert outputs["policy"].type.tensor_type.elem_type == onnx.TensorProto.FLOAT

        # memory should be [B, 2, H, W] int8 (elem_type=3)
        mo_shape = [d.dim_param or d.dim_value for d in outputs["memory"].type.tensor_type.shape.dim]
        assert len(mo_shape) == 4
        assert mo_shape[1] == 2, f"{exp}: memory channel dimension must be 2, got {mo_shape[1]}"
        assert outputs["memory"].type.tensor_type.elem_type == onnx.TensorProto.INT8

    @pytest.mark.parametrize("exp", EXPERIMENTS)
    def test_where_operator_present(self, exp):
        path = MODELS_DIR / exp / "model.onnx"
        model = onnx.load(str(path))
        op_types = {node.op_type for node in model.graph.node}
        assert "Where" in op_types, f"{exp} must contain 'Where' operator"
        assert "Max" not in op_types, f"{exp} must NOT contain obsolete 'Max' operator for memory"


# ========================================================================================
# 2. DYNAMIC SHAPE FORWARD INFERENCE ACROSS VARIED GRID DIMENSIONS
# ========================================================================================

class TestDynamicSpatialInference:
    """Stress test forward inference across (24,24), (32,32), (64,64), (100,100), (128,128), and non-square."""

    @pytest.mark.parametrize("exp", EXPERIMENTS)
    @pytest.mark.parametrize("h,w", GRID_DIMENSIONS)
    def test_forward_dimensions_stability(self, onnx_sessions, exp, h, w):
        sess = onnx_sessions[exp]
        rng = np.random.default_rng(hash((exp, h, w)) & 0xFFFFFFFF)

        # Generate realistic random inputs
        board = rng.integers(0, 2, size=(1, 7, h, w), dtype=np.int8)
        memory_in = rng.integers(0, 2, size=(1, 2, h, w), dtype=np.int8)

        policy, memory_out = sess.run(None, {"board": board, "memory_in": memory_in})

        # Shape assertions
        assert policy.shape == (1, 5, h, w), f"{exp} policy shape mismatch for {h}x{w}: {policy.shape}"
        assert memory_out.shape == (1, 2, h, w), f"{exp} memory shape mismatch for {h}x{w}: {memory_out.shape}"

        # Numerical integrity
        assert not np.isnan(policy).any(), f"{exp} produced NaN in policy for {h}x{w}"
        assert not np.isinf(policy).any(), f"{exp} produced Inf in policy for {h}x{w}"
        assert np.isfinite(policy).all(), f"{exp} policy logits not finite for {h}x{w}"

        # Memory type and range
        assert memory_out.dtype == np.int8, f"{exp} memory_out dtype {memory_out.dtype} != int8"
        assert np.isin(memory_out, [0, 1]).all(), f"{exp} memory_out contains values outside {{0, 1}}: {np.unique(memory_out)}"

    @pytest.mark.parametrize("exp", EXPERIMENTS)
    @pytest.mark.parametrize("batch_size", [1, 2, 4])
    def test_dynamic_batch_sizes(self, onnx_sessions, exp, batch_size):
        sess = onnx_sessions[exp]
        h, w = 32, 32
        board = np.zeros((batch_size, 7, h, w), dtype=np.int8)
        memory_in = np.zeros((batch_size, 2, h, w), dtype=np.int8)

        # Distinguish batch items
        for b in range(batch_size):
            board[b, 0, b, b] = 1       # ant at (b, b)
            board[b, 6, :, :] = 1       # vis=1
            board[b, 5, 10, 10] = 1     # foe hill
            memory_in[b, 0, 5, 5] = 1   # prior food memory

        policy, memory_out = sess.run(None, {"board": board, "memory_in": memory_in})

        assert policy.shape == (batch_size, 5, h, w)
        assert memory_out.shape == (batch_size, 2, h, w)
        assert not np.isnan(policy).any()
        assert not np.isinf(policy).any()

        # Check each batch item preserves individual behavior
        for b in range(batch_size):
            assert memory_out[b, 1, 10, 10] == 1  # foe hill recorded
            assert memory_out[b, 0, 5, 5] == 0    # visible non-food cleared


# ========================================================================================
# 3. WHERE MEMORY MECHANICS & GHOST HILL ELIMINATION
# ========================================================================================

class TestWhereMemoryMechanics:
    """Stress test Where memory mechanics for both enemy hill (plane 1) and food (plane 0)."""

    @pytest.mark.parametrize("exp", EXPERIMENTS)
    def test_enemy_hill_discovery_fog_and_raze(self, onnx_sessions, exp):
        """Lifecycle of an enemy hill:
        1. Turn 1: Enemy hill discovered at (10, 15). Visible -> memory updates to 1.
        2. Turn 2: Ant moves away, cell enters fog -> memory retains 1.
        3. Turn 3: Ant returns to hill location, hill is razed (destroyed) -> memory clears to 0 (NO GHOST HILL).
        4. Turn 4: Ant leaves, cell returns to fog -> memory remains cleared (0).
        """
        sess = onnx_sessions[exp]
        h, w = 32, 32
        hill_pos = (10, 15)
        ant_pos_t1 = (10, 14)
        ant_pos_t2 = (0, 0)
        ant_pos_t3 = hill_pos  # Ant steps on hill!

        mem = np.zeros((1, 2, h, w), dtype=np.int8)

        # Step 1: Discover hill
        board_t1 = np.zeros((1, 7, h, w), dtype=np.int8)
        board_t1[0, 0, ant_pos_t1[0], ant_pos_t1[1]] = 1  # ant
        board_t1[0, 5, hill_pos[0], hill_pos[1]] = 1      # foe hill
        board_t1[0, 6, hill_pos[0], hill_pos[1]] = 1      # vis=1 at hill
        board_t1[0, 6, ant_pos_t1[0], ant_pos_t1[1]] = 1  # vis=1 at ant

        policy_t1, mem = sess.run(None, {"board": board_t1, "memory_in": mem})
        assert mem[0, 1, hill_pos[0], hill_pos[1]] == 1, f"{exp}: Turn 1 enemy hill memory not set to 1!"

        # Step 2: Cell in fog of war
        board_t2 = np.zeros((1, 7, h, w), dtype=np.int8)
        board_t2[0, 0, ant_pos_t2[0], ant_pos_t2[1]] = 1  # ant far away
        board_t2[0, 6, ant_pos_t2[0], ant_pos_t2[1]] = 1  # vision far away, hill is in fog (vis=0)

        policy_t2, mem = sess.run(None, {"board": board_t2, "memory_in": mem})
        assert mem[0, 1, hill_pos[0], hill_pos[1]] == 1, f"{exp}: Turn 2 enemy hill lost from memory in fog!"

        # Step 3: Ant steps on hill and razes it!
        # Cell is visible (vis=1), but hill is gone (hill_foe=0)
        board_t3 = np.zeros((1, 7, h, w), dtype=np.int8)
        board_t3[0, 0, ant_pos_t3[0], ant_pos_t3[1]] = 1  # ant standing on razed hill
        board_t3[0, 6, hill_pos[0], hill_pos[1]] = 1      # vis=1 at hill
        board_t3[0, 5, hill_pos[0], hill_pos[1]] = 0      # hill is destroyed!

        policy_t3, mem = sess.run(None, {"board": board_t3, "memory_in": mem})
        assert mem[0, 1, hill_pos[0], hill_pos[1]] == 0, (
            f"{exp}: Turn 3 Where logic failed to eliminate ghost hill! Value is {mem[0, 1, hill_pos[0], hill_pos[1]]}"
        )

        # Step 4: Cell enters fog again
        board_t4 = np.zeros((1, 7, h, w), dtype=np.int8)
        board_t4[0, 0, ant_pos_t2[0], ant_pos_t2[1]] = 1
        board_t4[0, 6, ant_pos_t2[0], ant_pos_t2[1]] = 1  # hill in fog again

        policy_t4, mem = sess.run(None, {"board": board_t4, "memory_in": mem})
        assert mem[0, 1, hill_pos[0], hill_pos[1]] == 0, f"{exp}: Turn 4 ghost hill resurrected in fog!"

    @pytest.mark.parametrize("exp", EXPERIMENTS)
    def test_food_seen_discovery_fog_and_collection(self, onnx_sessions, exp):
        """Lifecycle of food (memory plane 0):
        1. Discover food -> memory plane 0 updates to 1.
        2. Fog covers food -> memory plane 0 retains 1.
        3. Ant gathers food (vis=1, food=0) -> memory plane 0 clears to 0.
        4. Fog covers again -> memory plane 0 remains 0.
        """
        sess = onnx_sessions[exp]
        h, w = 32, 32
        food_pos = (5, 7)

        mem = np.zeros((1, 2, h, w), dtype=np.int8)

        # Turn 1: Discover food
        board_t1 = np.zeros((1, 7, h, w), dtype=np.int8)
        board_t1[0, 2, food_pos[0], food_pos[1]] = 1  # food
        board_t1[0, 6, food_pos[0], food_pos[1]] = 1  # vis=1
        _, mem = sess.run(None, {"board": board_t1, "memory_in": mem})
        assert mem[0, 0, food_pos[0], food_pos[1]] == 1, f"{exp}: food not remembered at Turn 1"

        # Turn 2: Fog
        board_t2 = np.zeros((1, 7, h, w), dtype=np.int8)
        _, mem = sess.run(None, {"board": board_t2, "memory_in": mem})
        assert mem[0, 0, food_pos[0], food_pos[1]] == 1, f"{exp}: food lost in fog at Turn 2"

        # Turn 3: Food eaten / gone
        board_t3 = np.zeros((1, 7, h, w), dtype=np.int8)
        board_t3[0, 6, food_pos[0], food_pos[1]] = 1  # vis=1
        board_t3[0, 2, food_pos[0], food_pos[1]] = 0  # food gone
        _, mem = sess.run(None, {"board": board_t3, "memory_in": mem})
        assert mem[0, 0, food_pos[0], food_pos[1]] == 0, f"{exp}: food ghost not eliminated at Turn 3"

        # Turn 4: Fog returns
        board_t4 = np.zeros((1, 7, h, w), dtype=np.int8)
        _, mem = sess.run(None, {"board": board_t4, "memory_in": mem})
        assert mem[0, 0, food_pos[0], food_pos[1]] == 0, f"{exp}: food ghost resurrected at Turn 4"

    @pytest.mark.parametrize("exp", EXPERIMENTS)
    def test_multi_hill_boundary_and_torus_dynamics(self, onnx_sessions, exp):
        """Adversarial multi-hill test across torus boundaries:
        - Hill A at (0, 0): Discovered, kept under fog -> must stay 1.
        - Hill B at (0, W-1): Discovered, razed -> must clear to 0.
        - Hill C at (H-1, 0): Discovered, kept under fog -> must stay 1.
        - Hill D at (H-1, W-1): Discovered, razed -> must clear to 0.
        """
        sess = onnx_sessions[exp]
        h, w = 40, 60
        corners = [(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)]

        mem = np.zeros((1, 2, h, w), dtype=np.int8)

        # Step 1: Discover all 4 corner hills
        board_t1 = np.zeros((1, 7, h, w), dtype=np.int8)
        for r, c in corners:
            board_t1[0, 5, r, c] = 1  # foe hill
            board_t1[0, 6, r, c] = 1  # visible
        _, mem = sess.run(None, {"board": board_t1, "memory_in": mem})
        for r, c in corners:
            assert mem[0, 1, r, c] == 1, f"{exp}: failed to record corner hill at ({r}, {c})"

        # Step 2: All 4 enter fog
        board_t2 = np.zeros((1, 7, h, w), dtype=np.int8)
        _, mem = sess.run(None, {"board": board_t2, "memory_in": mem})
        for r, c in corners:
            assert mem[0, 1, r, c] == 1, f"{exp}: lost corner hill in fog at ({r}, {c})"

        # Step 3: Raze Hill B (0, w-1) and Hill D (h-1, w-1), leaving Hill A and Hill C under fog
        board_t3 = np.zeros((1, 7, h, w), dtype=np.int8)
        # Vision at B and D, but no hills
        board_t3[0, 6, 0, w - 1] = 1
        board_t3[0, 6, h - 1, w - 1] = 1
        _, mem = sess.run(None, {"board": board_t3, "memory_in": mem})

        # Hill A and C MUST remain in memory (vis=0)
        assert mem[0, 1, 0, 0] == 1, f"{exp}: Hill A (0,0) erroneously cleared"
        assert mem[0, 1, h - 1, 0] == 1, f"{exp}: Hill C ({h-1},0) erroneously cleared"

        # Hill B and D MUST be cleared (vis=1, hill=0)
        assert mem[0, 1, 0, w - 1] == 0, f"{exp}: Hill B (0,{w-1}) ghost hill retained"
        assert mem[0, 1, h - 1, w - 1] == 0, f"{exp}: Hill D ({h-1},{w-1}) ghost hill retained"


# ========================================================================================
# 4. AUTOREGRESSIVE CLOSED-LOOP STABILITY (50 TURNS)
# ========================================================================================

class TestAutoregressiveRollout:
    """Simulate 50 consecutive turns feeding memory_out into memory_in."""

    @pytest.mark.parametrize("exp", EXPERIMENTS)
    def test_50_step_closed_loop(self, onnx_sessions, exp):
        sess = onnx_sessions[exp]
        h, w = 32, 32
        rng = np.random.default_rng(20261015)

        mem = np.zeros((1, 2, h, w), dtype=np.int8)

        # Place a persistent hill at (16, 16) and a temporary food at (8, 8)
        for turn in range(50):
            board = np.zeros((1, 7, h, w), dtype=np.int8)
            # Ant moves randomly
            ant_r, ant_c = (turn * 3) % h, (turn * 5) % w
            board[0, 0, ant_r, ant_c] = 1

            # Vision disk approx (3x3 around ant)
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    board[0, 6, (ant_r + dr) % h, (ant_c + dc) % w] = 1

            # At turn 5, place enemy hill at (16, 16)
            if turn >= 5 and turn < 35:
                # If visible, mark hill
                if board[0, 6, 16, 16] == 1:
                    board[0, 5, 16, 16] = 1

            # At turn 35, hill is razed when visited
            if turn >= 35:
                board[0, 6, 16, 16] = 1  # force vision
                board[0, 5, 16, 16] = 0  # razed

            policy, mem = sess.run(None, {"board": board, "memory_in": mem})

            # Check invariants at every single turn
            assert not np.isnan(policy).any(), f"{exp}: NaN at turn {turn}"
            assert not np.isinf(policy).any(), f"{exp}: Inf at turn {turn}"
            assert mem.dtype == np.int8, f"{exp}: invalid mem dtype at turn {turn}"
            assert np.isin(mem, [0, 1]).all(), f"{exp}: invalid mem values at turn {turn}: {np.unique(mem)}"

        # After turn 35, hill must be completely cleared
        assert mem[0, 1, 16, 16] == 0, f"{exp}: ghost hill remained after 50-step rollout"


# ========================================================================================
# 5. PATHOLOGICAL & BOUNDARY STRESS
# ========================================================================================

class TestPathologicalInputs:
    """Stress test boundary condition handling."""

    @pytest.mark.parametrize("exp", EXPERIMENTS)
    def test_all_zeros_and_all_ones(self, onnx_sessions, exp):
        sess = onnx_sessions[exp]
        h, w = 32, 32

        # 1. All zeros
        b_zeros = np.zeros((1, 7, h, w), dtype=np.int8)
        m_zeros = np.zeros((1, 2, h, w), dtype=np.int8)
        p_z, m_z = sess.run(None, {"board": b_zeros, "memory_in": m_zeros})
        assert np.isfinite(p_z).all()
        assert (m_z == 0).all()

        # 2. All ones
        b_ones = np.ones((1, 7, h, w), dtype=np.int8)
        m_ones = np.ones((1, 2, h, w), dtype=np.int8)
        p_o, m_o = sess.run(None, {"board": b_ones, "memory_in": m_ones})
        assert np.isfinite(p_o).all()
        assert (m_o == 1).all()

    @pytest.mark.parametrize("exp", EXPERIMENTS)
    def test_pathological_int8_memory_sanitization(self, onnx_sessions, exp):
        """If incoming memory_in contains out-of-spec int8 values (e.g. 127 or -128):
        - Any visible cell must sanitize to current observation (0 or 1).
        """
        sess = onnx_sessions[exp]
        h, w = 32, 32
        board = np.zeros((1, 7, h, w), dtype=np.int8)
        board[0, 6, :, :] = 1  # vis = 1 everywhere!

        # Dirty memory input with 127 and -128
        dirty_mem = np.full((1, 2, h, w), fill_value=127, dtype=np.int8)
        dirty_mem[:, :, :16, :] = -128

        _, clean_mem = sess.run(None, {"board": board, "memory_in": dirty_mem})
        assert clean_mem.dtype == np.int8
        assert (clean_mem == 0).all(), f"{exp}: Where failed to sanitize dirty memory under full visibility"


# ========================================================================================
# 6. BEHAVIORAL DIVERGENCE AMONG MODELS
# ========================================================================================

class TestBehavioralDivergence:
    """Verify that exp020, exp021, and exp022 produce distinct policy distributions."""

    def test_models_have_distinct_policies(self, onnx_sessions):
        h, w = 32, 32
        rng = np.random.default_rng(42)
        board = rng.integers(0, 2, size=(1, 7, h, w), dtype=np.int8)
        mem = rng.integers(0, 2, size=(1, 2, h, w), dtype=np.int8)

        policies = {}
        for exp in EXPERIMENTS:
            p, _ = onnx_sessions[exp].run(None, {"board": board, "memory_in": mem})
            policies[exp] = p

        # Check differences between pairs
        diff_020_021 = np.max(np.abs(policies["exp020"] - policies["exp021"]))
        diff_020_022 = np.max(np.abs(policies["exp020"] - policies["exp022"]))
        diff_021_022 = np.max(np.abs(policies["exp021"] - policies["exp022"]))

        assert diff_020_021 > 0.01, f"exp020 and exp021 policies are identical (max diff: {diff_020_021})"
        assert diff_020_022 > 0.01, f"exp020 and exp022 policies are identical (max diff: {diff_020_022})"
        assert diff_021_022 > 0.01, f"exp021 and exp022 policies are identical (max diff: {diff_021_022})"
