"""Empirical Stress Test Harness by Challenger 2 for Phase 2 Milestone 3.

Empirically challenges:
1. Where Memory Semantics Verification:
   - Turn 1: Ant sees enemy hill at (r, c). Memory records hill.
   - Turn 2: Ant moves away, (r, c) falls into fog of war (vis=0). Verify hill is retained in memory.
   - Turn 3: Ant returns or steps onto (r, c), hill is razed/cleared (vis=1, seen=0). Verify Where operator wipes ghost hill.
   - Turn 4: Ant leaves, cell falls back to fog (vis=0). Verify ghost hill does not resurrect.
   - Food lifecycle in memory plane 0 (visible -> fog retention -> gathered -> wiped).
   - Multi-hill differential verification (one razed, one retained under fog).
   - Dynamic board sizes & torus wrapping boundary positions.
   - Verified across all three models: exp020, exp021, exp022.

2. Model Differentiation Verification:
   - Feed identical observation tensors into exp020, exp021, and exp022.
   - Measure policy logits differences: L1 distance, L2 distance (RMSE), max difference.
   - Measure action probability distribution differences: Softmax Total Variation Distance (TVD) and ant decision agreement rates.
   - Assert that the models produce genuinely distinct probability distributions and policies trained on distinct datasets (Big B vs Big D vs Big E).
"""

from pathlib import Path
import json
import numpy as np
import onnx
from onnx import numpy_helper
import onnxruntime as ort
import pytest

MODELS_DIR = Path(__file__).resolve().parent.parent.parent / "models"
EXPERIMENTS = ("exp020", "exp021", "exp022")
MOVES = ("N", "E", "S", "W", "-")


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
        sessions[exp] = ort.InferenceSession(str(path), sess_options=opts, providers=["CPUExecutionProvider"])
    return sessions


def softmax(x: np.ndarray, axis: int = 1) -> np.ndarray:
    """Stable softmax implementation."""
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)


# ========================================================================================
# 1. WHERE MEMORY DYNAMICS VERIFICATION (3-TURN RAZE & EXTENSIONS)
# ========================================================================================

class TestWhereMemorySemantics:
    """Empirical verification of Where memory dynamics across all three models."""

    @pytest.mark.parametrize("exp", EXPERIMENTS)
    def test_three_turn_enemy_hill_raze_lifecycle(self, onnx_sessions, exp):
        """Core scenario specified by orchestrator:
        - Turn 1: Ant sees enemy hill at (r, c). Memory records hill.
        - Turn 2: Ant moves away, (r, c) falls into fog of war (vis=0). Verify hill is retained in memory.
        - Turn 3: Ant returns or steps onto (r, c), hill is razed/cleared (vis=1, seen=0). Verify Where operator wipes ghost hill.
        - Turn 4 (anti-resurrection): Ant leaves, (r, c) returns to fog (vis=0). Verify memory stays 0.
        """
        sess = onnx_sessions[exp]
        h, w = 32, 32
        hill_pos = (12, 18)
        ant_t1 = (12, 17)
        ant_t2 = (0, 0)
        ant_t3 = hill_pos  # Ant steps on razed hill

        mem = np.zeros((1, 2, h, w), dtype=np.int8)

        # --- Turn 1: Enemy hill in view ---
        b1 = np.zeros((1, 7, h, w), dtype=np.int8)
        b1[0, 0, ant_t1[0], ant_t1[1]] = 1      # ant
        b1[0, 5, hill_pos[0], hill_pos[1]] = 1  # foe hill
        b1[0, 6, ant_t1[0], ant_t1[1]] = 1      # vis
        b1[0, 6, hill_pos[0], hill_pos[1]] = 1  # vis at hill
        _, mem1 = sess.run(None, {"board": b1, "memory_in": mem})

        assert mem1.dtype == np.int8
        assert mem1[0, 1, hill_pos[0], hill_pos[1]] == 1, f"{exp}: Turn 1 enemy hill not recorded in memory plane 1"
        assert np.isin(mem1, [0, 1]).all()

        # --- Turn 2: Ant moves away into fog ---
        b2 = np.zeros((1, 7, h, w), dtype=np.int8)
        b2[0, 0, ant_t2[0], ant_t2[1]] = 1  # ant far away
        b2[0, 6, ant_t2[0], ant_t2[1]] = 1  # vision far away; hill_pos is vis=0 (fog)
        _, mem2 = sess.run(None, {"board": b2, "memory_in": mem1})

        assert mem2.dtype == np.int8
        assert mem2[0, 1, hill_pos[0], hill_pos[1]] == 1, f"{exp}: Turn 2 enemy hill lost in fog from memory plane 1"
        assert np.isin(mem2, [0, 1]).all()

        # --- Turn 3: Ant returns and steps onto hill; hill is razed ---
        b3 = np.zeros((1, 7, h, w), dtype=np.int8)
        b3[0, 0, ant_t3[0], ant_t3[1]] = 1      # ant standing on hill
        b3[0, 6, hill_pos[0], hill_pos[1]] = 1  # vis=1 at hill
        b3[0, 5, hill_pos[0], hill_pos[1]] = 0  # hill razed/cleared (seen=0)
        _, mem3 = sess.run(None, {"board": b3, "memory_in": mem2})

        assert mem3.dtype == np.int8
        assert mem3[0, 1, hill_pos[0], hill_pos[1]] == 0, (
            f"{exp}: Turn 3 Where operator failed to wipe ghost hill from memory plane 1! Got {mem3[0, 1, hill_pos[0], hill_pos[1]]}"
        )
        assert np.isin(mem3, [0, 1]).all()

        # --- Turn 4: Ant leaves, cell returns to fog ---
        b4 = np.zeros((1, 7, h, w), dtype=np.int8)
        b4[0, 0, ant_t2[0], ant_t2[1]] = 1
        b4[0, 6, ant_t2[0], ant_t2[1]] = 1  # hill in fog again (vis=0)
        _, mem4 = sess.run(None, {"board": b4, "memory_in": mem3})

        assert mem4[0, 1, hill_pos[0], hill_pos[1]] == 0, f"{exp}: Turn 4 ghost hill resurrected in fog!"

    @pytest.mark.parametrize("exp", EXPERIMENTS)
    def test_food_memory_lifecycle(self, onnx_sessions, exp):
        """Food lifecycle in memory plane 0:
        Turn 1: Food seen -> memory plane 0 updates to 1.
        Turn 2: Fog -> memory plane 0 retains 1.
        Turn 3: Food gathered / disappeared under vision (vis=1, food=0) -> memory plane 0 wipes to 0.
        Turn 4: Fog returns -> remains 0.
        """
        sess = onnx_sessions[exp]
        h, w = 32, 32
        food_pos = (7, 14)
        mem = np.zeros((1, 2, h, w), dtype=np.int8)

        # Turn 1: Food seen
        b1 = np.zeros((1, 7, h, w), dtype=np.int8)
        b1[0, 2, food_pos[0], food_pos[1]] = 1
        b1[0, 6, food_pos[0], food_pos[1]] = 1
        _, mem1 = sess.run(None, {"board": b1, "memory_in": mem})
        assert mem1[0, 0, food_pos[0], food_pos[1]] == 1, f"{exp}: Turn 1 food not recorded in plane 0"

        # Turn 2: Fog
        b2 = np.zeros((1, 7, h, w), dtype=np.int8)
        _, mem2 = sess.run(None, {"board": b2, "memory_in": mem1})
        assert mem2[0, 0, food_pos[0], food_pos[1]] == 1, f"{exp}: Turn 2 food lost in fog from plane 0"

        # Turn 3: Food gathered
        b3 = np.zeros((1, 7, h, w), dtype=np.int8)
        b3[0, 6, food_pos[0], food_pos[1]] = 1
        b3[0, 2, food_pos[0], food_pos[1]] = 0
        _, mem3 = sess.run(None, {"board": b3, "memory_in": mem2})
        assert mem3[0, 0, food_pos[0], food_pos[1]] == 0, f"{exp}: Turn 3 food ghost not wiped from plane 0"

        # Turn 4: Fog returns
        b4 = np.zeros((1, 7, h, w), dtype=np.int8)
        _, mem4 = sess.run(None, {"board": b4, "memory_in": mem3})
        assert mem4[0, 0, food_pos[0], food_pos[1]] == 0, f"{exp}: Turn 4 food ghost resurrected in fog"

    @pytest.mark.parametrize("exp", EXPERIMENTS)
    def test_differential_multi_hill_retention_and_wipe(self, onnx_sessions, exp):
        """One hill is razed while another remains in fog:
        Verify cell-level precision of Where operator.
        """
        sess = onnx_sessions[exp]
        h, w = 32, 32
        h_fog = (5, 5)     # Will remain in fog
        h_raze = (20, 20)  # Will be razed

        # Turn 1: Discover both hills
        b1 = np.zeros((1, 7, h, w), dtype=np.int8)
        b1[0, 5, h_fog[0], h_fog[1]] = 1
        b1[0, 6, h_fog[0], h_fog[1]] = 1
        b1[0, 5, h_raze[0], h_raze[1]] = 1
        b1[0, 6, h_raze[0], h_raze[1]] = 1
        mem = np.zeros((1, 2, h, w), dtype=np.int8)
        _, mem1 = sess.run(None, {"board": b1, "memory_in": mem})

        assert mem1[0, 1, h_fog[0], h_fog[1]] == 1
        assert mem1[0, 1, h_raze[0], h_raze[1]] == 1

        # Turn 2: Only h_raze is in vision (vis=1, seen=0); h_fog is in fog (vis=0)
        b2 = np.zeros((1, 7, h, w), dtype=np.int8)
        b2[0, 6, h_raze[0], h_raze[1]] = 1
        b2[0, 5, h_raze[0], h_raze[1]] = 0
        _, mem2 = sess.run(None, {"board": b2, "memory_in": mem1})

        assert mem2[0, 1, h_fog[0], h_fog[1]] == 1, f"{exp}: Fog hill erroneously cleared"
        assert mem2[0, 1, h_raze[0], h_raze[1]] == 0, f"{exp}: Razed hill ghost retained"

    @pytest.mark.parametrize("exp", EXPERIMENTS)
    @pytest.mark.parametrize("h,w", [(24, 24), (32, 32), (40, 60), (24, 124)])
    def test_torus_boundary_hills(self, onnx_sessions, exp, h, w):
        """Verify Where memory dynamics at extreme torus boundaries."""
        sess = onnx_sessions[exp]
        corners = [(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)]

        # Turn 1: See all 4 corners
        b1 = np.zeros((1, 7, h, w), dtype=np.int8)
        for r, c in corners:
            b1[0, 5, r, c] = 1
            b1[0, 6, r, c] = 1
        mem = np.zeros((1, 2, h, w), dtype=np.int8)
        _, mem1 = sess.run(None, {"board": b1, "memory_in": mem})

        for r, c in corners:
            assert mem1[0, 1, r, c] == 1

        # Turn 2: All in fog
        b2 = np.zeros((1, 7, h, w), dtype=np.int8)
        _, mem2 = sess.run(None, {"board": b2, "memory_in": mem1})
        for r, c in corners:
            assert mem2[0, 1, r, c] == 1

        # Turn 3: Raze corner (0, 0) and (h-1, w-1); keep (0, w-1) and (h-1, 0) in fog
        b3 = np.zeros((1, 7, h, w), dtype=np.int8)
        b3[0, 6, 0, 0] = 1
        b3[0, 6, h - 1, w - 1] = 1
        _, mem3 = sess.run(None, {"board": b3, "memory_in": mem2})

        assert mem3[0, 1, 0, 0] == 0, f"{exp}: (0,0) ghost not wiped"
        assert mem3[0, 1, h - 1, w - 1] == 0, f"{exp}: (h-1, w-1) ghost not wiped"
        assert mem3[0, 1, 0, w - 1] == 1, f"{exp}: (0, w-1) fog lost"
        assert mem3[0, 1, h - 1, 0] == 1, f"{exp}: (h-1, 0) fog lost"


# ========================================================================================
# 2. MODEL DIFFERENTIATION VERIFICATION (EXP-020 vs EXP-021 vs EXP-022)
# ========================================================================================

class TestModelDifferentiation:
    """Verify that exp020, exp021, and exp022 are distinct trained policies."""

    def test_weight_level_differentiation(self):
        """Verify that ONNX initializer weights differ significantly across models."""
        models = {exp: onnx.load(str(MODELS_DIR / exp / "model.onnx")) for exp in EXPERIMENTS}
        weights = {
            exp: {init.name: numpy_helper.to_array(init) for init in models[exp].graph.initializer}
            for exp in EXPERIMENTS
        }

        pairs = [("exp020", "exp021"), ("exp020", "exp022"), ("exp021", "exp022")]
        for m1, m2 in pairs:
            weight_diffs = []
            for name in weights[m1]:
                if name == "sources":
                    continue  # Buffer constant
                w1 = weights[m1][name].astype(np.float32)
                w2 = weights[m2][name].astype(np.float32)
                diff = np.max(np.abs(w1 - w2))
                weight_diffs.append(diff)
            # Assert significant weight divergence in every pair
            assert np.mean(weight_diffs) > 0.5, f"Weights of {m1} and {m2} are too similar: mean diff {np.mean(weight_diffs)}"
            assert np.max(weight_diffs) > 2.0, f"Max weight diff between {m1} and {m2} too small: {np.max(weight_diffs)}"

    def test_identical_standardized_observation_tensor_divergence(self, onnx_sessions):
        """Feed an identical tactical observation tensor into all three models and measure divergence."""
        h, w = 32, 32
        board = np.zeros((1, 7, h, w), dtype=np.int8)

        # 4 friendly ants
        ants = [(10, 10), (10, 20), (20, 10), (20, 20)]
        for r, c in ants:
            board[0, 0, r, c] = 1

        # Foes
        board[0, 1, 10, 12] = 1
        board[0, 1, 22, 20] = 1

        # Food
        board[0, 2, 8, 10] = 1
        board[0, 2, 12, 20] = 1
        board[0, 2, 19, 10] = 1

        # Water obstacles
        board[0, 3, 15, 15] = 1
        board[0, 3, 15, 16] = 1

        # Hills
        board[0, 4, 10, 8] = 1   # Own hill
        board[0, 5, 25, 25] = 1  # Enemy hill

        # Vision box around ants
        for ar, ac in ants:
            for dr in range(-2, 3):
                for dc in range(-2, 3):
                    board[0, 6, (ar + dr) % h, (ac + dc) % w] = 1

        # Memory input: 1 food in fog, 1 foe hill in fog
        mem_in = np.zeros((1, 2, h, w), dtype=np.int8)
        mem_in[0, 0, 5, 5] = 1
        mem_in[0, 1, 25, 25] = 1

        policies = {}
        probs = {}
        for exp in EXPERIMENTS:
            p, _ = onnx_sessions[exp].run(None, {"board": board, "memory_in": mem_in})
            policies[exp] = p
            probs[exp] = softmax(p, axis=1)

        pairs = [("exp020", "exp021"), ("exp020", "exp022"), ("exp021", "exp022")]
        for m1, m2 in pairs:
            p1, p2 = policies[m1], policies[m2]
            pr1, pr2 = probs[m1], probs[m2]

            # Whole-board metrics
            l1_board = np.mean(np.abs(p1 - p2))
            l2_board = np.sqrt(np.mean((p1 - p2) ** 2))
            max_diff = np.max(np.abs(p1 - p2))

            # Ant-specific metrics
            ant_l1 = np.mean([np.abs(p1[0, :, r, c] - p2[0, :, r, c]) for r, c in ants])
            ant_tvd = np.mean([0.5 * np.sum(np.abs(pr1[0, :, r, c] - pr2[0, :, r, c])) for r, c in ants])

            # Quantitative assertions of genuine distinctness
            assert l1_board > 2.0, f"{m1} vs {m2} whole-board L1 diff too small: {l1_board}"
            assert l2_board > 3.0, f"{m1} vs {m2} whole-board L2 diff too small: {l2_board}"
            assert max_diff > 15.0, f"{m1} vs {m2} max logit diff too small: {max_diff}"
            assert ant_l1 > 2.0, f"{m1} vs {m2} ant-specific L1 diff too small: {ant_l1}"
            assert ant_tvd > 0.15, f"{m1} vs {m2} ant probability TVD too small: {ant_tvd}"

    def test_full_reference_dataset_statistical_divergence(self, onnx_sessions):
        """Evaluate over reference game observations and assert distinct policy actions."""
        ref_path = Path(__file__).resolve().parent.parent.parent / "dist" / "reference" / "observations.json"
        assert ref_path.exists(), f"Reference observations file missing: {ref_path}"

        with open(ref_path) as f:
            data = json.load(f)
        obs_list = data["observations"]

        from tb_baselines import planes

        total_ants = 0
        agree_020_021 = 0
        agree_020_022 = 0
        agree_021_022 = 0
        all_three_agree = 0

        l1_diffs_020_021 = []
        l1_diffs_020_022 = []
        l1_diffs_021_022 = []

        for obs in obs_list:
            b = planes.encode(obs)
            rows, cols = obs["size"]
            m = np.zeros((1, 2, rows, cols), dtype=np.int8)

            p020, _ = onnx_sessions["exp020"].run(None, {"board": b, "memory_in": m})
            p021, _ = onnx_sessions["exp021"].run(None, {"board": b, "memory_in": m})
            p022, _ = onnx_sessions["exp022"].run(None, {"board": b, "memory_in": m})

            l1_diffs_020_021.append(np.mean(np.abs(p020 - p021)))
            l1_diffs_020_022.append(np.mean(np.abs(p020 - p022)))
            l1_diffs_021_022.append(np.mean(np.abs(p021 - p022)))

            for r, c in obs.get("mine", []):
                act020 = np.argmax(p020[0, :, r, c])
                act021 = np.argmax(p021[0, :, r, c])
                act022 = np.argmax(p022[0, :, r, c])

                total_ants += 1
                if act020 == act021:
                    agree_020_021 += 1
                if act020 == act022:
                    agree_020_022 += 1
                if act021 == act022:
                    agree_021_022 += 1
                if act020 == act021 == act022:
                    all_three_agree += 1

        assert total_ants > 1000, f"Expected > 1000 ant decisions, got {total_ants}"

        # Assert meaningful disagreement (not identical models)
        rate_020_021 = agree_020_021 / total_ants
        rate_020_022 = agree_020_022 / total_ants
        rate_021_022 = agree_021_022 / total_ants
        rate_all = all_three_agree / total_ants

        assert rate_020_021 < 0.85, f"exp020 and exp021 agree too much: {rate_020_021:.2%}"
        assert rate_020_022 < 0.80, f"exp020 and exp022 agree too much: {rate_020_022:.2%}"
        assert rate_021_022 < 0.80, f"exp021 and exp022 agree too much: {rate_021_022:.2%}"
        assert rate_all < 0.75, f"All three agree too much: {rate_all:.2%}"

        # Mean Logit L1 across dataset
        assert np.mean(l1_diffs_020_021) > 2.0
        assert np.mean(l1_diffs_020_022) > 2.0
        assert np.mean(l1_diffs_021_022) > 2.0
