"""Adversarial / Stress Challenge Suite for Milestone 1: Micro Class & Where Memory.

Independent challenge tests verifying:
1. TestMicroReceptiveFieldBoundary:
   Strict gradient backpropagation: all coordinates strictly outside [17, 47] on 64x64 grid
   must have gradient identically 0.0 across all policy actions and memory planes.
   Boundary reach verified at r=17, r=47, c=17, c=47; off-center and 100x100 grids tested.
2. TestMicroDynamicBoardShapesAndWrapping:
   Dynamic board shapes: (24, 24), (30, 40), (40, 30), (25, 37), (100, 100), (124, 124),
   (128, 128), (24, 124), (124, 24).
   Torus translation equivariance under circular shifts across square and non-square grids.
   ONNX Runtime dynamic shape inference and bitwise memory parity with PyTorch.
3. TestMicroWhereMemoryMultiTurnRaze:
   20-turn simulated game trajectory with fog and raze events:
   - Destroyed hill cleared in view and NEVER resurrects in fog (turns 11..20).
   - Intact hill retained in fog (turns 5..20).
   - Memory invariant: strictly int8 in {0, 1} at every turn.
   - 3-way bitwise equivalence: NumPy == PyTorch == ONNX Runtime.
"""

from __future__ import annotations

import warnings
import numpy as np
import pytest

from tb_baselines import export, nets, planes

# Filter non-actionable environment warnings
warnings.filterwarnings("ignore", category=UserWarning, message=".*CUDA initialization.*")
warnings.filterwarnings("ignore", category=DeprecationWarning, message=".*legacy TorchScript-based ONNX export.*")


@pytest.fixture(scope="module")
def exported_micro_stress(tmp_path_factory):
    pytest.importorskip("torch")
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")

    tmp_dir = tmp_path_factory.mktemp("micro_stress_export")
    spec = export.classes()["micro"]
    model = nets.build(spec, memory="where")
    onnx_path = tmp_dir / "model_stress.onnx"

    export.to_onnx(model, onnx_path, planes=7)
    export.halve(onnx_path)
    export.shorten(onnx_path)
    return onnx_path, model


class TestMicroReceptiveFieldBoundary:
    """Challenge 1: Strict Receptive Field Boundary Verification."""

    def test_gradient_identically_zero_outside_receptive_field(self):
        torch = pytest.importorskip("torch")
        spec = export.classes()["micro"]
        model = nets.build(spec, memory="where").eval()

        H, W = 64, 64
        torch.manual_seed(42)
        b = torch.randn(1, 7, H, W, requires_grad=True)
        m = torch.randn(1, 2, H, W, requires_grad=True)

        pol, _ = model(b, m)

        outside_mask = torch.ones((H, W), dtype=torch.bool)
        outside_mask[17:48, 17:48] = False

        for a in range(5):
            if b.grad is not None:
                b.grad.zero_()
            if m.grad is not None:
                m.grad.zero_()

            target = pol[0, a, 32, 32]
            target.backward(retain_graph=True)

            max_outside_b = b.grad[0, :, outside_mask].abs().max().item()
            max_outside_m = m.grad[0, :, outside_mask].abs().max().item()
            max_outside = max(max_outside_b, max_outside_m)

            assert max_outside == 0.0, (
                f"Action {a}: gradient outside [17, 47] must be identically zero, got max {max_outside}"
            )

            total_grad_mag = b.grad[0].abs().sum(dim=0) + m.grad[0].abs().sum(dim=0)
            active = (total_grad_mag > 1e-9).nonzero()
            min_r, max_r = active[:, 0].min().item(), active[:, 0].max().item()
            min_c, max_c = active[:, 1].min().item(), active[:, 1].max().item()

            assert min_r == 17, f"Action {a}: expected min_r == 17, got {min_r}"
            assert max_r == 47, f"Action {a}: expected max_r == 47, got {max_r}"
            assert min_c == 17, f"Action {a}: expected min_c == 17, got {min_c}"
            assert max_c == 47, f"Action {a}: expected max_c == 47, got {max_c}"

    def test_off_center_gradient_bounding(self):
        torch = pytest.importorskip("torch")
        spec = export.classes()["micro"]
        model = nets.build(spec, memory="where").eval()

        H, W = 64, 64
        torch.manual_seed(101)
        b = torch.randn(1, 7, H, W, requires_grad=True)
        m = torch.randn(1, 2, H, W, requires_grad=True)

        pol, _ = model(b, m)
        target = pol[0, :, 30, 25].sum()
        target.backward()

        total_grad = b.grad[0].abs().sum(dim=0) + m.grad[0].abs().sum(dim=0)
        outside_mask = torch.ones((H, W), dtype=torch.bool)
        outside_mask[15:46, 10:41] = False

        max_outside = total_grad[outside_mask].max().item()
        assert max_outside == 0.0, f"Off-center gradient outside [15, 45]x[10, 40] must be 0.0, got {max_outside}"

    def test_large_grid_100x100_receptive_field(self):
        torch = pytest.importorskip("torch")
        spec = export.classes()["micro"]
        model = nets.build(spec, memory="where").eval()

        H, W = 100, 100
        torch.manual_seed(202)
        b = torch.randn(1, 7, H, W, requires_grad=True)
        m = torch.randn(1, 2, H, W, requires_grad=True)

        pol, _ = model(b, m)
        target = pol[0, :, 50, 50].sum()
        target.backward()

        total_grad = b.grad[0].abs().sum(dim=0) + m.grad[0].abs().sum(dim=0)
        outside_mask = torch.ones((H, W), dtype=torch.bool)
        outside_mask[35:66, 35:66] = False

        max_outside = total_grad[outside_mask].max().item()
        assert max_outside == 0.0, f"100x100 grid gradient outside [35, 65]x[35, 65] must be 0.0, got {max_outside}"


class TestMicroDynamicBoardShapesAndWrapping:
    """Challenge 2: Dynamic Board Shapes & Torus Wrap Padding Stress Test."""

    UNUSUAL_SHAPES = [
        (24, 24),
        (30, 40),
        (40, 30),
        (25, 37),
        (100, 100),
        (124, 124),
        (128, 128),
        (24, 124),
        (124, 24),
    ]

    @pytest.mark.parametrize("H,W", UNUSUAL_SHAPES)
    def test_forward_pass_unusual_shapes(self, H: int, W: int):
        torch = pytest.importorskip("torch")
        spec = export.classes()["micro"]
        model = nets.build(spec, memory="where").eval()

        b = torch.randint(0, 2, (1, 7, H, W), dtype=torch.int8)
        m = torch.randint(0, 2, (1, 2, H, W), dtype=torch.int8)
        with torch.no_grad():
            pol, mem_out = model(b, m)

        assert pol.shape == (1, 5, H, W), f"Shape mismatch: {pol.shape}"
        assert mem_out.shape == (1, 2, H, W), f"Shape mismatch: {mem_out.shape}"
        assert mem_out.dtype == torch.int8, f"Dtype mismatch: {mem_out.dtype}"

    @pytest.mark.parametrize("H,W", [(24, 24), (30, 40), (25, 37), (100, 100), (128, 128)])
    def test_torus_translation_equivariance(self, H: int, W: int):
        torch = pytest.importorskip("torch")
        spec = export.classes()["micro"]
        model = nets.build(spec, memory="where").eval()

        torch.manual_seed(999)
        b = torch.randn(1, 7, H, W)
        m = torch.randn(1, 2, H, W)
        shifts = [(1, 0), (0, 1), (5, 7), (-3, -4), (15, 15), (H // 2, W // 2)]

        for dr, dc in shifts:
            b_rolled = torch.roll(b, (dr, dc), dims=(-2, -1))
            m_rolled = torch.roll(m, (dr, dc), dims=(-2, -1))

            with torch.no_grad():
                pol_orig, mem_orig = model(b.to(torch.int8), m.to(torch.int8))
                pol_roll, mem_roll = model(b_rolled.to(torch.int8), m_rolled.to(torch.int8))

            pol_orig_rolled = torch.roll(pol_orig, (dr, dc), dims=(-2, -1))
            mem_orig_rolled = torch.roll(mem_orig, (dr, dc), dims=(-2, -1))

            diff_pol = (pol_roll - pol_orig_rolled).abs().max().item()
            assert diff_pol < 1e-4, f"Torus equivariance failed on ({H}, {W}) shift ({dr}, {dc}): max diff {diff_pol}"

            diff_mem = (mem_roll != mem_orig_rolled).sum().item()
            assert diff_mem == 0, f"Memory equivariance failed on ({H}, {W}) shift ({dr}, {dc}): {diff_mem} diffs"

    @pytest.mark.parametrize("H,W", UNUSUAL_SHAPES)
    def test_onnxruntime_dynamic_shapes(self, exported_micro_stress, H: int, W: int):
        ort = pytest.importorskip("onnxruntime")
        torch = pytest.importorskip("torch")
        onnx_path, pt_model = exported_micro_stress

        session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        b_np = np.random.randint(0, 2, (1, 7, H, W), dtype=np.int8)
        m_np = np.random.randint(0, 2, (1, 2, H, W), dtype=np.int8)

        ort_outs = session.run(["policy", "memory"], {"board": b_np, "memory_in": m_np})
        ort_pol, ort_mem = ort_outs[0], ort_outs[1]

        assert ort_pol.shape == (1, 5, H, W)
        assert ort_mem.shape == (1, 2, H, W)
        assert ort_mem.dtype == np.int8

        with torch.no_grad():
            _, pt_mem = pt_model(torch.from_numpy(b_np), torch.from_numpy(m_np))
        np.testing.assert_array_equal(ort_mem, pt_mem.numpy())


class TestMicroWhereMemoryMultiTurnRaze:
    """Challenge 3: Where Memory Recurrence Under Multi-Turn Simulated Raze Events."""

    def test_multi_turn_raze_and_fog_invariants(self, exported_micro_stress):
        torch = pytest.importorskip("torch")
        ort = pytest.importorskip("onnxruntime")
        onnx_path, pt_model = exported_micro_stress
        ort_session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

        H, W = 32, 32
        b_obj = planes.Board(H, W)

        hill_1_pos = (10, 10)
        hill_2_pos = (25, 25)

        ant_positions = [
            (0, 0),   # T0: H1 fog, H2 fog
            (0, 0),   # T1: H1 fog, H2 fog
            (4, 4),   # T2: H1 in view (d2=72 <= 77) -> H1 discovered!
            (0, 0),   # T3: H1 fog -> H1 retained in fog!
            (0, 0),   # T4: H1 fog -> H1 retained in fog!
            (24, 24), # T5: H2 in view (d2=2 <= 77) -> H2 discovered!
            (0, 0),   # T6: H1 fog, H2 fog -> both retained!
            (4, 4),   # T7: H1 in view
            (10, 10), # T8: H1 in view (d2=0) -> Razed at turn end
            (10, 10), # T9: H1 in view, confirmed DEAD -> Memory cleared to 0!
            (10, 10), # T10: H1 in view, confirmed dead -> Memory remains 0!
            (0, 0),   # T11: H1 in fog -> MUST NOT RESURRECT!
            (0, 0),   # T12: H1 in fog
            (24, 24), # T13: H1 in fog, H2 in view -> H2 is 1, H1 is 0
            (0, 0),   # T14: H1 in fog, H2 in fog
            (0, 0),   # T15: H1 in fog, H2 in fog
            (0, 0),   # T16: H1 in fog, H2 in fog
            (0, 0),   # T17: H1 in fog, H2 in fog
            (0, 0),   # T18: H1 in fog, H2 in fog
            (0, 0),   # T19: H1 in fog, H2 in fog
            (0, 0),   # T20: H1 in fog, H2 in fog
        ]

        numpy_state = None
        pt_memory = torch.zeros((1, 2, H, W), dtype=torch.int8)
        ort_memory = np.zeros((1, 2, H, W), dtype=np.int8)

        hill_1_alive = True

        for turn, ant_pos in enumerate(ant_positions):
            if turn >= 9:
                hill_1_alive = False

            my_ants = [list(ant_pos)]
            vis_mask = b_obj.dilate(b_obj.scatter(my_ants), planes.VIEW_RADIUS2)

            hills = []
            if hill_1_alive and vis_mask[hill_1_pos] == 1:
                hills.append([hill_1_pos[0], hill_1_pos[1], 1])
            if vis_mask[hill_2_pos] == 1:
                hills.append([hill_2_pos[0], hill_2_pos[1], 1])

            p_mine = b_obj.scatter(my_ants)
            p_food = b_obj.zeros()
            p_hill_mine = b_obj.zeros()
            p_hill_foe = b_obj.zeros()
            for r, c, owner in hills:
                if owner != 0:
                    p_hill_foe[r, c] = 1
            p_water = b_obj.zeros()
            p_ants = b_obj.zeros()
            p_vis = vis_mask

            board_np = np.stack([p_mine, p_ants, p_food, p_water, p_hill_mine, p_hill_foe, p_vis], axis=0).reshape(1, 7, H, W)

            # NumPy update
            flat_vis = vis_mask.ravel()
            runs = []
            val = flat_vis[0]
            cnt = 0
            for x in flat_vis:
                if x == val:
                    cnt += 1
                else:
                    runs.extend([int(val), cnt])
                    val = x
                    cnt = 1
            runs.extend([int(val), cnt])
            obs_synthetic = {
                "size": [H, W],
                "mine": my_ants,
                "ants": [],
                "food": [],
                "water": [],
                "hills": hills,
                "vis": {"rle": runs},
            }

            numpy_state = planes.remember(numpy_state, obs_synthetic, mode="where")
            np_mem_encoded = planes.encode_memory(numpy_state, (H, W))

            # PyTorch update
            board_torch = torch.from_numpy(board_np)
            with torch.no_grad():
                _, pt_memory = pt_model(board_torch, pt_memory)

            # ONNX Runtime update
            ort_outs = ort_session.run(["policy", "memory"], {"board": board_np, "memory_in": ort_memory})
            ort_memory = ort_outs[1]

            # Invariant: dtype and values strictly int8 in {0, 1}
            assert pt_memory.dtype == torch.int8, f"Turn {turn}: PyTorch memory dtype not int8"
            assert ort_memory.dtype == np.int8, f"Turn {turn}: ORT memory dtype not int8"
            assert np_mem_encoded.dtype == np.int8, f"Turn {turn}: NumPy memory dtype not int8"

            assert set(torch.unique(pt_memory).tolist()).issubset({0, 1})
            assert set(np.unique(ort_memory).tolist()).issubset({0, 1})
            assert set(np.unique(np_mem_encoded).tolist()).issubset({0, 1})

            # 3-way bitwise equivalence
            np.testing.assert_array_equal(pt_memory.numpy(), np_mem_encoded)
            np.testing.assert_array_equal(ort_memory, np_mem_encoded)

            # Raze and non-resurrection assertion
            mem_h1 = pt_memory[0, 1, hill_1_pos[0], hill_1_pos[1]].item()
            mem_h2 = pt_memory[0, 1, hill_2_pos[0], hill_2_pos[1]].item()

            if turn < 2:
                assert mem_h1 == 0, f"Turn {turn}: Hill 1 not discovered yet"
            elif 2 <= turn <= 8:
                assert mem_h1 == 1, f"Turn {turn}: Hill 1 must be remembered"
            elif turn >= 9:
                assert mem_h1 == 0, (
                    f"Turn {turn}: destroyed hill must remain 0, never resurrect (fog={vis_mask[hill_1_pos]==0})"
                )

            if turn < 5:
                assert mem_h2 == 0, f"Turn {turn}: Hill 2 not discovered yet"
            elif turn >= 5:
                assert mem_h2 == 1, f"Turn {turn}: Hill 2 must remain remembered"
