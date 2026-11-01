"""Unit test suite for DAgger Closed-Loop Controller, Information Alignment Gate,
Student Driver, State Tracker, and Dataset Aggregator (Milestone 1).

Covers:
1. InformationAlignmentGate:
   - Rejection of privileged/unauthorized keys in obs and memory.
   - Zero-privilege local perception enforcement.
   - Correct order generation under student's Where memory.
   - Graceful handling of empty colonies (0 ants).
2. StudentDriver:
   - Checkpoint loading (real exp011 and synthetic dummy).
   - Valid action string and index generation in MOVES order.
   - Evaluation mode and no_grad verification.
   - Batched group inference across board sizes.
3. StateTracker:
   - Accurate ant-level Disagreement Rate calculation.
   - Edge cases: 0 ants, 0% disagreement, 100% disagreement.
   - State space signature hashing and novel state ratio tracking.
4. DatasetAggregator:
   - Header conformance check (engine_digest, memory_mode='where', teacher_remembers=True).
   - Rejection of mismatched or corrupted headers.
   - Episode stride offset application (stride = 100_000 * k).
   - Row schema integrity (o, k, m, a).
   - Streaming memory efficiency.
5. Integration:
   - Single-iteration simulated / smoke DAgger rollout execution.
"""

from __future__ import annotations

import gzip
import json
import os
from pathlib import Path
import shutil
import numpy as np
import pytest
import torch

from tb_baselines.dagger import (
    DatasetAggregator,
    InformationAlignmentGate,
    STRIDE,
    StateTracker,
    StudentDriver,
    aggregate_datasets,
    main,
    rollout_iteration,
)
from tb_baselines.env import Group, Seat, Step
from tb_baselines.export import classes
from tb_baselines.planes import MOVES, N_PLANES

ROOT = Path(__file__).resolve().parents[2]


# ==============================================================================
# 1. Information Alignment Gate Tests
# ==============================================================================


class TestInformationAlignmentGate:
    """Rigorous gate checks ensuring zero privileged information reaches Teacher."""

    def test_gate_accepts_valid_local_observation(self):
        gate = InformationAlignmentGate(hill_lead=6)
        obs = {
            "size": [24, 24],
            "mine": [[5, 5], [10, 10]],
            "foes": [],
            "food": [[6, 5]],
            "hills": [],
            "water": {"rle": [0, 576]},
            "vis": {"rle": [0, 576]},
            "ids": [1, 2],
        }
        mem = {"food_seen": [], "hill_foe_seen": []}
        orders = gate.query(obs, mem)
        assert isinstance(orders, np.ndarray)
        assert len(orders) == 2
        assert all(0 <= a <= 4 for a in orders)

        # Test label method
        moves, order_str = gate.label(obs, mem)
        assert len(moves) == 2
        assert len(order_str) == 2
        assert all(c in MOVES for c in order_str)

    @pytest.mark.parametrize("priv_key", [
        "score", "turn_scores", "true_hills", "global_map", "actor_critic_extra", "secret"
    ])
    def test_gate_raises_assertion_on_privileged_keys(self, priv_key):
        gate = InformationAlignmentGate(hill_lead=6)
        obs = {
            "size": [24, 24],
            "mine": [[5, 5]],
            "foes": [],
            "food": [],
            "hills": [],
            "water": {"rle": [0, 576]},
            "vis": {"rle": [0, 576]},
            "ids": [1],
            priv_key: 100,  # Unauthorized key
        }
        with pytest.raises(AssertionError, match=r"(privileged|unauthorized|keys)"):
            gate.query(obs, {})

    @pytest.mark.parametrize("bad_mem_key", ["secret_hills", "priv_target", "cheat"])
    def test_gate_raises_assertion_on_unauthorized_memory_keys(self, bad_mem_key):
        gate = InformationAlignmentGate(hill_lead=6)
        obs = {
            "size": [24, 24], "mine": [[5, 5]], "foes": [], "food": [], "hills": [],
            "water": {"rle": [0, 576]}, "vis": {"rle": [0, 576]}, "ids": [1]
        }
        mem = {"food_seen": [], "hill_foe_seen": [], bad_mem_key: [[1, 1]]}
        with pytest.raises(AssertionError, match=r"(memory|unauthorized|keys)"):
            gate.query(obs, mem)

    def test_gate_raises_assertion_on_missing_required_keys(self):
        gate = InformationAlignmentGate(hill_lead=6)
        obs = {
            "size": [24, 24], "mine": [[5, 5]], "foes": [], "food": [], "hills": [],
            "water": {"rle": [0, 576]}, "ids": [1]
            # missing "vis"
        }
        with pytest.raises(AssertionError, match=r"(privileged|unauthorized|keys)"):
            gate.query(obs, None)

    def test_gate_handles_empty_colony(self):
        gate = InformationAlignmentGate(hill_lead=6)
        obs = {
            "size": [24, 24], "mine": [], "foes": [], "food": [], "hills": [],
            "water": {"rle": [0, 576]}, "vis": {"rle": [0, 576]}, "ids": []
        }
        orders = gate.query(obs, None)
        assert isinstance(orders, np.ndarray)
        assert len(orders) == 0

        moves, order_str = gate.label(obs, None)
        assert len(moves) == 0
        assert order_str == ""

    def test_gate_evaluates_student_memory_not_teacher_memory(self):
        """Verifies teacher responds specifically to student's Where memory."""
        gate = InformationAlignmentGate(hill_lead=6)
        # On a 32x32 board, distance between (16, 16) and (16, 5) is 11.
        # 11^2 = 121 > 77 (VIEW_RADIUS2), so the hill is out of view in fog of war.
        obs = {
            "size": [32, 32], "mine": [[16, 16]], "foes": [], "food": [], "hills": [],
            "water": {"rle": [0, 1024]}, "vis": {"rle": [0, 1024]}, "ids": [1]
        }
        # With enemy hill in student memory (out of sight)
        mem_with_hill = {"food_seen": [], "hill_foe_seen": [[16, 5]]}
        orders_with_hill = gate.query(obs, mem_with_hill)

        # Without enemy hill in memory
        mem_plain = {"food_seen": [], "hill_foe_seen": []}
        orders_plain = gate.query(obs, mem_plain)

        # West is action index 3 (towards column 5 from column 16)
        assert orders_with_hill[0] == 3, "Teacher should steer towards remembered enemy hill"
        assert not np.array_equal(orders_with_hill, orders_plain)


# ==============================================================================
# 2. Student Driver Tests
# ==============================================================================


class TestStudentDriver:
    """Verifies student network inference, batched processing, and action extraction."""

    def test_driver_loads_real_exp011_checkpoint(self):
        ckpt_path = ROOT / "runs" / "exp011" / "best.pt"
        if not ckpt_path.exists():
            pytest.skip(f"Baseline checkpoint not found at {ckpt_path}")
        spec = classes()["nano"]
        driver = StudentDriver(weights_path=ckpt_path, spec=spec, device="cpu")
        assert driver.model.training is False
        assert driver.model.memory_kind == "where"

    def test_driver_loads_dummy_model(self):
        spec = classes()["nano"]
        driver = StudentDriver(weights_path=None, spec=spec, device="cpu")
        assert driver.model.training is False
        assert driver.model.memory_kind == "where"

    def test_driver_act_produces_valid_action_string_and_indices(self):
        spec = classes()["nano"]
        driver = StudentDriver(weights_path=None, spec=spec, device="cpu")
        obs = {
            "size": [24, 24], "mine": [[5, 5], [10, 10], [15, 15]],
            "foes": [], "food": [[6, 6]], "hills": [],
            "water": {"rle": [0, 576]}, "vis": {"rle": [0, 576]}, "ids": [1, 2, 3]
        }
        mem = {"food_seen": [], "hill_foe_seen": []}
        order_str, actions = driver.act(obs, mem)

        assert isinstance(order_str, str)
        assert len(order_str) == 3
        assert all(c in MOVES for c in order_str)
        assert isinstance(actions, list)
        assert len(actions) == 3
        assert all(0 <= a <= 4 for a in actions)
        assert order_str == "".join(MOVES[a] for a in actions)

    def test_driver_act_handles_empty_seat(self):
        spec = classes()["nano"]
        driver = StudentDriver(weights_path=None, spec=spec, device="cpu")
        obs = {
            "size": [24, 24], "mine": [], "foes": [], "food": [], "hills": [],
            "water": {"rle": [0, 576]}, "vis": {"rle": [0, 576]}, "ids": []
        }
        order_str, actions = driver.act(obs, None)
        assert order_str == ""
        assert actions == []

    def test_driver_eval_mode_no_grad(self):
        spec = classes()["nano"]
        driver = StudentDriver(weights_path=None, spec=spec, device="cpu")
        obs = {
            "size": [24, 24], "mine": [[5, 5]], "foes": [], "food": [], "hills": [],
            "water": {"rle": [0, 576]}, "vis": {"rle": [0, 576]}, "ids": [1]
        }
        with torch.enable_grad():
            _ = driver.act(obs, None)
            for p in driver.model.parameters():
                assert p.grad is None

    @pytest.mark.parametrize("size", [(24, 24), (32, 32), (48, 64)])
    def test_driver_spatial_generality(self, size):
        spec = classes()["nano"]
        driver = StudentDriver(weights_path=None, spec=spec, device="cpu")
        h, w = size
        obs = {
            "size": [h, w], "mine": [[1, 1], [h - 2, w - 2]], "foes": [], "food": [], "hills": [],
            "water": {"rle": [0, h * w]}, "vis": {"rle": [0, h * w]}, "ids": [1, 2]
        }
        order_str, actions = driver.act(obs, None)
        assert len(order_str) == 2

    def test_driver_predict_step_batched_group(self):
        spec = classes()["nano"]
        driver = StudentDriver(weights_path=None, spec=spec, device="cpu")
        obs1 = {
            "size": [24, 24], "mine": [[2, 2], [3, 3]], "foes": [], "food": [], "hills": [],
            "water": {"rle": [0, 576]}, "vis": {"rle": [0, 576]}, "ids": [1, 2]
        }
        obs2 = {
            "size": [24, 24], "mine": [[10, 10]], "foes": [], "food": [], "hills": [],
            "water": {"rle": [0, 576]}, "vis": {"rle": [0, 576]}, "ids": [3]
        }
        seat1 = Seat(ep=0, seat=0, wave=0, match=0, turn=1, obs=obs1)
        seat2 = Seat(ep=0, seat=1, wave=0, match=0, turn=1, obs=obs2)

        boards = np.zeros((2, N_PLANES, 24, 24), dtype=np.int8)
        group = Group(size=(24, 24), indices=[0, 1], boards=boards)
        step = Step(seats=[seat1, seat2], groups=[group], scores={0: [0, 0]}, ended=[], turn=1)

        mem = {(0, 0): {"food_seen": [], "hill_foe_seen": []}, (0, 1): {}}
        actions, moves_by_seat = driver.predict_step(step, mem)

        assert len(actions) == 2
        assert len(moves_by_seat) == 2
        assert len(actions[0]) == 2
        assert len(actions[1]) == 1
        assert len(moves_by_seat[0]) == 2
        assert len(moves_by_seat[1]) == 1


# ==============================================================================
# 3. State Tracker Tests
# ==============================================================================


class TestStateTracker:
    """Verifies ant-level disagreement metrics and state signature tracking."""

    def test_state_tracker_zero_ants(self):
        tracker = StateTracker()
        assert tracker.disagreement_rate == 0.0
        assert tracker.novel_state_ratio == 0.0

    def test_state_tracker_perfect_agreement(self):
        tracker = StateTracker()
        obs = {"size": [24, 24], "mine": [[1, 1], [2, 2]], "food": [], "foes": []}
        tracker.record_step(obs, None, student_actions=[0, 1], teacher_actions=[0, 1])
        assert tracker.total_ants == 2
        assert tracker.disagree_ants == 0
        assert tracker.disagreement_rate == 0.0

    def test_state_tracker_complete_disagreement(self):
        tracker = StateTracker()
        obs = {"size": [24, 24], "mine": [[1, 1], [2, 2]], "food": [], "foes": []}
        tracker.record_step(obs, None, student_actions=[0, 1], teacher_actions=[2, 3])
        assert tracker.total_ants == 2
        assert tracker.disagree_ants == 2
        assert tracker.disagreement_rate == 1.0

    def test_state_tracker_fractional_accumulation(self):
        tracker = StateTracker()
        obs = {"size": [24, 24], "mine": [[1, 1], [2, 2]], "food": [], "foes": []}
        # Step 1: 1 match, 1 mismatch (50%)
        tracker.record_step(obs, None, student_actions=[0, 1], teacher_actions=[0, 3])
        # Step 2: 2 matches, 0 mismatches
        tracker.record_step(obs, None, student_actions=[2, 2], teacher_actions=[2, 2])
        assert tracker.total_ants == 4
        assert tracker.disagree_ants == 1
        assert tracker.disagreement_rate == 0.25

    def test_state_tracker_signature_uniqueness_and_novelty(self):
        tracker = StateTracker()
        obs1 = {
            "size": [24, 24], "mine": [[1, 1], [2, 2]], "food": [[5, 5]], "foes": [],
            "vis": {"rle": [1, 288, 0, 288]}
        }
        mem1 = {"hill_foe_seen": []}
        is_new1 = tracker.record_step(obs1, mem1, [0, 0], [0, 0])
        assert is_new1 is True

        # Exact repeat of state
        is_new2 = tracker.record_step(obs1, mem1, [0, 0], [0, 0])
        assert is_new2 is False

        # Altered state (different ant count)
        obs2 = dict(obs1, mine=[[1, 1], [2, 2], [3, 3], [4, 4]])
        is_new3 = tracker.record_step(obs2, mem1, [0, 0, 0, 0], [0, 0, 0, 0])
        assert is_new3 is True

    def test_state_tracker_novel_state_ratio(self):
        tracker = StateTracker()
        obs = {"size": [24, 24], "mine": [[1, 1]], "food": [], "foes": []}
        # 1 novel, 3 repeated
        tracker.record_step(obs, None, [0], [0])
        tracker.record_step(obs, None, [0], [0])
        tracker.record_step(obs, None, [0], [0])
        tracker.record_step(obs, None, [0], [0])
        assert tracker.novel_state_ratio == 0.25

    def test_state_tracker_metrics_summary(self):
        tracker = StateTracker()
        obs = {"size": [24, 24], "mine": [[1, 1]], "food": [], "foes": []}
        tracker.record_step(obs, None, [0], [1])
        m = tracker.metrics()
        assert m["total_ant_decisions"] == 1
        assert m["total_ant_disagreements"] == 1
        assert m["disagreement_rate"] == 1.0
        assert m["total_seat_turn_steps"] == 1
        assert m["novel_state_ratio"] == 1.0


# ==============================================================================
# 4. Dataset Aggregator Tests
# ==============================================================================


class TestDatasetAggregator:
    """Verifies dataset merging, episode offsets, and header integrity."""

    @pytest.fixture
    def sample_parts(self, tmp_path):
        header1 = {
            "kind": "teacher-rollout",
            "engine_digest": "sha256:819166e7918117a1cbbd67dc042e7368a076a2dd45cea5122683983912d591e1",
            "evaluator": "datalogic 5.7",
            "teacher_remembers": True,
            "memory_mode": "where",
            "memory": ["food_seen", "hill_foe_seen"]
        }
        header2 = dict(header1)

        p1 = tmp_path / "part1.jsonl.gz"
        p2 = tmp_path / "part2.jsonl.gz"

        row1 = {"o": {"size": [24, 24], "mine": [[1, 1]]}, "k": [0, 0, 1], "m": {}, "a": "N"}
        row2 = {"o": {"size": [24, 24], "mine": [[2, 2]]}, "k": [1, 0, 1], "m": {}, "a": "E"}
        row3 = {"o": {"size": [24, 24], "mine": [[3, 3]]}, "k": [0, 0, 1], "m": {}, "a": "S"}

        with gzip.open(p1, "wt") as f:
            f.write(json.dumps(header1) + "\n")
            f.write(json.dumps(row1) + "\n")
            f.write(json.dumps(row2) + "\n")

        with gzip.open(p2, "wt") as f:
            f.write(json.dumps(header2) + "\n")
            f.write(json.dumps(row3) + "\n")

        return [p1, p2]

    def test_aggregator_merges_and_offsets_episodes(self, sample_parts, tmp_path):
        out = tmp_path / "merged.jsonl.gz"
        agg = DatasetAggregator(stride=100_000)
        res = agg.merge(sample_parts, out)

        assert res["rows"] == 3
        assert out.exists()

        with gzip.open(out, "rt") as f:
            merged_header = json.loads(f.readline())
            rows = [json.loads(line) for line in f]

        assert merged_header["memory_mode"] == "where"
        assert merged_header["teacher_remembers"] is True
        assert len(rows) == 3

        # Check episode offsets:
        # Part 1 rows: ep 0 and ep 1 remain 0 and 1
        assert rows[0]["k"][0] == 0
        assert rows[1]["k"][0] == 1
        # Part 2 row: ep 0 offset by 1 * 100_000
        assert rows[2]["k"][0] == 100_000

        # Check stats file written
        stats_file = out.with_suffix("").with_suffix(".stats.json")
        assert stats_file.exists()
        stats_data = json.loads(stats_file.read_text())
        assert stats_data["rows"] == 3

    def test_aggregator_rejects_mismatched_engine_digest(self, sample_parts, tmp_path):
        bad_part = tmp_path / "bad.jsonl.gz"
        bad_header = {
            "kind": "teacher-rollout",
            "engine_digest": "sha256:corrupted_hash",
            "evaluator": "datalogic 5.7",
            "teacher_remembers": True,
            "memory_mode": "where",
            "memory": ["food_seen", "hill_foe_seen"]
        }
        with gzip.open(bad_part, "wt") as f:
            f.write(json.dumps(bad_header) + "\n")
            f.write(json.dumps({"o": {"size": [24, 24], "mine": []}, "k": [0, 0, 0], "m": {}, "a": ""}) + "\n")

        agg = DatasetAggregator()
        with pytest.raises((ValueError, SystemExit), match=r"(engine_digest|digest mismatch)"):
            agg.merge([sample_parts[0], bad_part], tmp_path / "out.jsonl.gz")

    def test_aggregator_rejects_non_where_memory_mode(self, sample_parts, tmp_path):
        bad_part = tmp_path / "bad_mode.jsonl.gz"
        bad_header = {
            "kind": "teacher-rollout",
            "engine_digest": "sha256:819166e7918117a1cbbd67dc042e7368a076a2dd45cea5122683983912d591e1",
            "evaluator": "datalogic 5.7",
            "teacher_remembers": True,
            "memory_mode": "max",  # Forbidden mode
            "memory": ["food_seen", "hill_foe_seen"]
        }
        with gzip.open(bad_part, "wt") as f:
            f.write(json.dumps(bad_header) + "\n")
            f.write(json.dumps({"o": {"size": [24, 24], "mine": []}, "k": [0, 0, 0], "m": {}, "a": ""}) + "\n")

        agg = DatasetAggregator()
        with pytest.raises((ValueError, SystemExit), match=r"(memory_mode|where)"):
            agg.merge([sample_parts[0], bad_part], tmp_path / "out.jsonl.gz")

    def test_aggregator_rejects_non_remembering_teacher(self, sample_parts, tmp_path):
        bad_part = tmp_path / "bad_remembers.jsonl.gz"
        bad_header = {
            "kind": "teacher-rollout",
            "engine_digest": "sha256:819166e7918117a1cbbd67dc042e7368a076a2dd45cea5122683983912d591e1",
            "evaluator": "datalogic 5.7",
            "teacher_remembers": False,  # Forbidden
            "memory_mode": "where",
            "memory": ["food_seen", "hill_foe_seen"]
        }
        with gzip.open(bad_part, "wt") as f:
            f.write(json.dumps(bad_header) + "\n")
            f.write(json.dumps({"o": {"size": [24, 24], "mine": []}, "k": [0, 0, 0], "m": {}, "a": ""}) + "\n")

        agg = DatasetAggregator()
        with pytest.raises((ValueError, SystemExit), match=r"teacher_remembers"):
            agg.merge([sample_parts[0], bad_part], tmp_path / "out.jsonl.gz")

    def test_aggregate_datasets_function_convenience(self, sample_parts, tmp_path):
        out = tmp_path / "convenience_merged.jsonl.gz"
        res = aggregate_datasets(sample_parts, out, iteration=2)
        assert res["rows"] == 3
        assert res["iteration"] == 2
        assert out.exists()

    def test_aggregator_multi_part_exact_stride_and_metadata(self, tmp_path):
        """Verifies multi-part merge (3 parts) strictly offsets episode IDs by i * STRIDE
        and asserts that metadata declared in header['parts'] matches row values exactly:
        parts[i]['episode_offset'] == i * STRIDE == row['k'][0] - raw_ep.
        """
        part_paths = []
        raw_rows_by_part = []
        stride = 100_000

        for part_idx in range(3):
            p = tmp_path / f"part_{part_idx}.jsonl.gz"
            h = {
                "kind": "dagger-rollout" if part_idx > 0 else "teacher-rollout",
                "engine_digest": "sha256:819166e7918117a1cbbd67dc042e7368a076a2dd45cea5122683983912d591e1",
                "evaluator": "datalogic 5.7",
                "teacher_remembers": True,
                "memory_mode": "where",
                "memory": ["food_seen", "hill_foe_seen"],
                "env_seed": part_idx + 1,
            }
            rows = [
                {"o": {"size": [24, 24], "mine": [[0, 0]]}, "k": [0, 0, 1], "m": {}, "a": "N"},
                {"o": {"size": [24, 24], "mine": [[1, 1]]}, "k": [1, 1, 2], "m": {}, "a": "E"},
            ]
            with gzip.open(p, "wt") as f:
                f.write(json.dumps(h) + "\n")
                for r in rows:
                    f.write(json.dumps(r) + "\n")
            part_paths.append(p)
            raw_rows_by_part.append(rows)

        out_path = tmp_path / "merged_multi.jsonl.gz"
        agg = DatasetAggregator(stride=stride)
        stats = agg.merge(part_paths, out_path, iteration=2)

        assert stats["rows"] == 6
        assert stats["parts"] == 3

        with gzip.open(out_path, "rt") as f:
            merged_header = json.loads(f.readline())
            assert len(merged_header["parts"]) == 3
            merged_rows = [json.loads(line) for line in f]

        for i, part_meta in enumerate(merged_header["parts"]):
            assert part_meta["file"] == part_paths[i].name
            assert part_meta["episode_offset"] == i * stride

        for i in range(3):
            part_offset = merged_header["parts"][i]["episode_offset"]
            expected_offset = i * stride
            assert part_offset == expected_offset

            part_rows = merged_rows[i * 2 : (i + 1) * 2]
            for row_idx, r in enumerate(part_rows):
                raw_ep = raw_rows_by_part[i][row_idx]["k"][0]
                actual_ep = r["k"][0]

                # Core invariant: parts[i]["episode_offset"] == i * STRIDE strictly equals row["k"][0] - raw_ep
                assert part_offset == actual_ep - raw_ep
                assert actual_ep == raw_ep + expected_offset
                assert actual_ep // stride == i
                assert r["k"][1] == raw_rows_by_part[i][row_idx]["k"][1]
                assert r["k"][2] == raw_rows_by_part[i][row_idx]["k"][2]
                assert r["a"] == raw_rows_by_part[i][row_idx]["a"]



# ==============================================================================
# 5. Rollout Integration Tests
# ==============================================================================


class TestRolloutIntegration:
    """Verifies end-to-end integration of rollout_iteration with real engine."""

    @pytest.mark.skipif(
        not (os.environ.get("TINYBRAINS") or shutil.which("tinybrains")),
        reason="no `tinybrains` on PATH",
    )
    def test_rollout_iteration_smoke(self, tmp_path):
        out_file = tmp_path / "test_rollout.jsonl.gz"
        driver = StudentDriver(weights_path=None, device="cpu")
        gate = InformationAlignmentGate(hill_lead=6)

        stats = rollout_iteration(
            student_driver=driver,
            gate=gate,
            out_path=out_file,
            seat_turns=10,
            k_iter=1,
            waves=1,
            matches_per_wave=1,
            max_turns=20,
            seed=42,
            maps="basic-tiny-2p",
        )

        assert stats["rows"] == 10
        assert stats["iteration"] == 1
        assert 0.0 <= stats["disagreement_rate"] <= 1.0
        assert out_file.exists()

        stats_file = out_file.with_suffix("").with_suffix(".stats.json")
        assert stats_file.exists()

        with gzip.open(out_file, "rt") as f:
            header = json.loads(f.readline())
            assert header["kind"] == "dagger-rollout"
            assert header["memory_mode"] == "where"
            assert header["teacher_remembers"] is True
            assert header["dagger_iteration"] == 1
            assert header["episode_offset"] == 0

            rows = [json.loads(line) for line in f]
            assert len(rows) == 10
            for r in rows:
                assert set(r.keys()) == {"o", "k", "m", "a"}
                assert 0 <= r["k"][0] < 100_000
                assert len(r["a"]) == len(r["o"]["mine"])

    @pytest.mark.skipif(
        not (os.environ.get("TINYBRAINS") or shutil.which("tinybrains")),
        reason="no `tinybrains` on PATH",
    )
    def test_rollout_iteration_raw_episode_indexing_across_iterations(self, tmp_path):
        """Verifies that rollout_iteration writes un-offsetted raw episode IDs across all k."""
        driver = StudentDriver(weights_path=None, device="cpu")
        gate = InformationAlignmentGate(hill_lead=6)

        for k in (1, 2):
            out_file = tmp_path / f"raw_iter_{k}.jsonl.gz"
            stats = rollout_iteration(
                student_driver=driver,
                gate=gate,
                out_path=out_file,
                seat_turns=6,
                k_iter=k,
                waves=1,
                matches_per_wave=1,
                max_turns=10,
                seed=k * 10,
                maps="basic-tiny-2p",
            )
            assert stats["rows"] == 6
            with gzip.open(out_file, "rt") as f:
                header = json.loads(f.readline())
                assert header["dagger_iteration"] == k
                assert header["episode_offset"] == 0

                rows = [json.loads(line) for line in f]
                for r in rows:
                    ep, seat, turn = r["k"]
                    assert 0 <= ep < 100_000, f"Raw rollout iter {k} leaked offset into episode ID: {ep}"

    @pytest.mark.skipif(
        not (os.environ.get("TINYBRAINS") or shutil.which("tinybrains")),
        reason="no `tinybrains` on PATH",
    )
    def test_chained_rollout_to_aggregator_stride_consistency(self, tmp_path):
        """Chained integration test: real rollout_iteration outputs piped directly into DatasetAggregator.

        Verifies that:
        1. Raw rollout files contain raw un-offsetted episodes (episode_offset == 0).
        2. Merged dataset row episode IDs strictly match declared parts[i]['episode_offset']:
           parts[i]['episode_offset'] == i * STRIDE strictly equals row['k'][0] - raw_ep.
        3. No double-offsetting occurs (part 1 rows start at 100_000, not 200_000).
        """
        driver = StudentDriver(weights_path=None, device="cpu")
        gate = InformationAlignmentGate(hill_lead=6)

        # 1. Base expert dataset (part 0)
        base_data = tmp_path / "base_exp011.jsonl.gz"
        base_header = {
            "kind": "teacher-rollout",
            "engine_digest": "sha256:819166e7918117a1cbbd67dc042e7368a076a2dd45cea5122683983912d591e1",
            "evaluator": "datalogic 5.7",
            "teacher_remembers": True,
            "memory_mode": "where",
            "memory": ["food_seen", "hill_foe_seen"],
            "env_seed": 1,
        }
        raw_base_rows = [
            {"o": {"size": [24, 24], "mine": [[1, 1]]}, "k": [0, 0, 1], "m": {}, "a": "N"},
            {"o": {"size": [24, 24], "mine": [[2, 2]]}, "k": [1, 0, 1], "m": {}, "a": "E"},
        ]
        with gzip.open(base_data, "wt") as f:
            f.write(json.dumps(base_header) + "\n")
            for r in raw_base_rows:
                f.write(json.dumps(r) + "\n")

        # 2. Run real rollout iteration 1 (part 1)
        iter1_data = tmp_path / "delta_iter1.jsonl.gz"
        rollout_iteration(
            student_driver=driver,
            gate=gate,
            out_path=iter1_data,
            seat_turns=4,
            k_iter=1,
            waves=1,
            matches_per_wave=1,
            max_turns=10,
            seed=101,
            maps="basic-tiny-2p",
        )

        # 3. Run real rollout iteration 2 (part 2)
        iter2_data = tmp_path / "delta_iter2.jsonl.gz"
        rollout_iteration(
            student_driver=driver,
            gate=gate,
            out_path=iter2_data,
            seat_turns=4,
            k_iter=2,
            waves=1,
            matches_per_wave=1,
            max_turns=10,
            seed=201,
            maps="basic-tiny-2p",
        )

        # Verify raw delta files are un-offsetted and record their raw rows
        raw_rows_by_part = [raw_base_rows]
        for p, k in [(iter1_data, 1), (iter2_data, 2)]:
            with gzip.open(p, "rt") as f:
                h = json.loads(f.readline())
                assert h["dagger_iteration"] == k
                assert h["episode_offset"] == 0
                part_rows = [json.loads(line) for line in f]
                assert len(part_rows) == 4
                for r in part_rows:
                    assert 0 <= r["k"][0] < 100_000, f"Raw delta {p.name} had pre-offset episode {r['k'][0]}"
                raw_rows_by_part.append(part_rows)

        # 4. Aggregate parts [base, iter1, iter2]
        merged_out = tmp_path / "merged_final.jsonl.gz"
        agg = DatasetAggregator(stride=STRIDE)
        agg.merge([base_data, iter1_data, iter2_data], merged_out, iteration=2)

        # 5. Assert strict correspondence: parts[i]['episode_offset'] == i * STRIDE == row['k'][0] - raw_ep
        with gzip.open(merged_out, "rt") as f:
            merged_header = json.loads(f.readline())
            assert len(merged_header["parts"]) == 3
            assert [p["episode_offset"] for p in merged_header["parts"]] == [0, STRIDE, 2 * STRIDE]

            merged_rows = [json.loads(line) for line in f]
            assert len(merged_rows) == 10  # 2 base + 4 iter1 + 4 iter2

            row_offset = 0
            for i, part_rows in enumerate(raw_rows_by_part):
                expected_offset = i * STRIDE
                declared_offset = merged_header["parts"][i]["episode_offset"]
                assert declared_offset == expected_offset

                for row_idx, raw_r in enumerate(part_rows):
                    m_row = merged_rows[row_offset]
                    raw_ep = raw_r["k"][0]
                    actual_ep = m_row["k"][0]

                    # Chained invariant assertion
                    assert declared_offset == actual_ep - raw_ep
                    assert actual_ep - raw_ep == expected_offset
                    assert actual_ep == raw_ep + expected_offset
                    assert actual_ep // STRIDE == i
                    assert m_row["k"][1] == raw_r["k"][1]
                    assert m_row["k"][2] == raw_r["k"][2]
                    assert m_row["a"] == raw_r["a"]
                    row_offset += 1

    @pytest.mark.skipif(
        not (os.environ.get("TINYBRAINS") or shutil.which("tinybrains")),
        reason="no `tinybrains` on PATH",
    )
    def test_main_cli_pipeline_chained_stride_offsets(self, tmp_path):
        """End-to-end integration test of main() CLI loop asserting zero double-offsetting."""
        base_data = tmp_path / "exp011.jsonl.gz"
        base_header = {
            "kind": "teacher-rollout",
            "engine_digest": "sha256:819166e7918117a1cbbd67dc042e7368a076a2dd45cea5122683983912d591e1",
            "evaluator": "datalogic 5.7",
            "teacher_remembers": True,
            "memory_mode": "where",
            "memory": ["food_seen", "hill_foe_seen"],
        }
        with gzip.open(base_data, "wt") as f:
            f.write(json.dumps(base_header) + "\n")
            f.write(json.dumps({"o": {"size": [24, 24], "mine": [[1, 1]]}, "k": [0, 0, 1], "m": {}, "a": "N"}) + "\n")

        out_data = tmp_path / "exp031.jsonl.gz"
        out_run = tmp_path / "runs_exp031"
        argv = [
            "--base-weights", "runs/exp011/best.pt",
            "--base-data", str(base_data),
            "--iterations", "2",
            "--samples-per-iter", "2",
            "--seed", "1",
            "--out-data", str(out_data),
            "--out-run", str(out_run),
            "--waves", "1",
            "--matches-per-wave", "1",
            "--max-turns", "5",
            "--maps", "basic-tiny-2p",
        ]
        main(argv)

        with gzip.open(out_data, "rt") as f:
            h = json.loads(f.readline())
            declared_offsets = [p["episode_offset"] for p in h["parts"]]
            assert declared_offsets == [0, 100_000, 200_000]

            rows = [json.loads(line) for line in f]
            assert len(rows) == 5
            # Row 0 (base): k=[0, 0, 1]
            assert rows[0]["k"][0] == 0
            # Rows 1-2 (iter 1): must be in [100000, 200000), NOT 200000
            for r in rows[1:3]:
                assert 100_000 <= r["k"][0] < 200_000
                assert r["k"][0] // 100_000 == 1
            # Rows 3-4 (iter 2): must be in [200000, 300000), NOT 400000
            for r in rows[3:5]:
                assert 200_000 <= r["k"][0] < 300_000
                assert r["k"][0] // 100_000 == 2

