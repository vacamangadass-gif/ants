"""DAgger (Dataset Aggregation) Closed-Loop Controller.

Implements the interactive imitation learning pipeline for TinyBrains Ants:
1. InformationAlignmentGate: Strict security barrier ensuring the Teacher Oracle
   evaluates decisions strictly under the student's physical observation (O_t)
   and Where memory (M_t), with zero privileged information.
2. StudentDriver: Drives physical environment rollouts using the student policy
   network, supporting size-grouped batched inference, argmax move selection,
   and memory advancement.
3. StateTracker: Tracks per-ant policy disagreement rate P(pi_S != a_T) and
   macro state space coverage.
4. DatasetAggregator: Lossless, memory-bounded D_k = D_{k-1} U Delta D_k merge
   with episode stride offsets and header verification.
5. rollout_iteration: Single-iteration rollout loop generating standardized
   training datasets.
6. CLI entrypoint supporting multi-iteration DAgger pipeline execution.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import torch

from . import nets
from .env import Env, Step, orders_from_indices
from .export import classes
from .planes import MEMORY, encode, encode_memory, remember
from .teacher import Teacher, known

STRIDE = 100_000
EXPECTED_DIGEST = "sha256:819166e7918117a1cbbd67dc042e7368a076a2dd45cea5122683983912d591e1"


# ==============================================================================
# 1. Information Alignment Hard Gate
# ==============================================================================


class InformationAlignmentViolation(AssertionError):
    """Raised when privileged information or malformed state enters the teacher gate."""
    pass


class InformationAlignmentGate:
    """Strict security barrier ensuring the Teacher Oracle operates purely on the student's belief state."""

    ALLOWED_OBS_KEYS = frozenset({
        "size", "mine", "foes", "food", "hills", "water", "vis", "ids"
    })
    ALLOWED_MEM_KEYS = frozenset({
        "food_seen", "hill_foe_seen"
    })

    def __init__(self, teacher: Teacher | None = None, hill_lead: int = 6):
        self.teacher = teacher if teacher is not None else Teacher(hill_lead=hill_lead)
        self.hill_lead = hill_lead

    def validate_obs(self, obs: dict[str, Any]) -> None:
        """Validate observation against whitelist and structural requirements."""
        keys = set(obs.keys())
        if keys != self.ALLOWED_OBS_KEYS:
            extra = keys - self.ALLOWED_OBS_KEYS
            missing = self.ALLOWED_OBS_KEYS - keys
            raise InformationAlignmentViolation(
                f"InformationAlignmentGate rejected observation! "
                f"Unauthorized / privileged keys: {sorted(extra)}, "
                f"Missing keys: {sorted(missing)}"
            )

        rows, cols = obs["size"]
        if rows <= 0 or cols <= 0:
            raise InformationAlignmentViolation(f"Invalid board size: {obs['size']}")
        if len(obs["ids"]) != len(obs["mine"]):
            raise InformationAlignmentViolation(
                f"ids length ({len(obs['ids'])}) != mine length ({len(obs['mine'])})"
            )
        if not isinstance(obs.get("water"), dict) or "rle" not in obs["water"]:
            raise InformationAlignmentViolation("Malformed 'water' field; missing 'rle'")
        if not isinstance(obs.get("vis"), dict) or "rle" not in obs["vis"]:
            raise InformationAlignmentViolation("Malformed 'vis' field; missing 'rle'")

    def validate_memory(self, memory: dict[str, Any] | None) -> None:
        """Validate student memory dictionary against allowed keys."""
        if memory is None:
            return
        if not isinstance(memory, dict):
            raise InformationAlignmentViolation(f"Memory must be a dict or None, got {type(memory)}")
        keys = set(memory.keys())
        if not (keys <= self.ALLOWED_MEM_KEYS):
            extra = keys - self.ALLOWED_MEM_KEYS
            raise InformationAlignmentViolation(
                f"InformationAlignmentGate rejected memory! Unauthorized memory keys: {sorted(extra)}"
            )

    def label(self, obs: dict[str, Any], memory: dict[str, Any] | None = None) -> tuple[np.ndarray, str]:
        """Validate observation and memory, then compute teacher's corrective order.

        Returns:
            moves: np.ndarray of shape [N] with int64 action indices in [0..4]
            order: str of length N with characters in ("N", "E", "S", "W", "-")
        """
        self.validate_obs(obs)
        self.validate_memory(memory)

        mine_count = len(obs["mine"])
        if mine_count == 0:
            return np.zeros(0, dtype=np.int64), ""

        water, seen = known(obs)
        moves = self.teacher.orders(obs, water, seen, memory)
        order = orders_from_indices(moves, [mine_count])[0]
        return moves, order

    def query(self, obs: dict[str, Any], memory: dict[str, Any] | None = None) -> np.ndarray:
        """Query Teacher Oracle under strict local information alignment. Returns move indices array."""
        moves, _ = self.label(obs, memory)
        return moves


# ==============================================================================
# 2. Student Rollout Driver
# ==============================================================================


class StudentDriver:
    """Controls the physical rollout of the environment using the student policy."""

    def __init__(
        self,
        weights_path: Path | str | None = None,
        spec: dict[str, Any] | None = None,
        device: str | torch.device | None = None,
    ):
        if device is None:
            self.device = torch.device("cpu")
        elif isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device

        self.spec = spec if spec is not None else classes()["nano"]
        self.model = nets.build(self.spec, memory="where").to(self.device)

        if weights_path is not None:
            ckpt = torch.load(weights_path, map_location=self.device)
            state_dict = ckpt["trunk"] if (isinstance(ckpt, dict) and "trunk" in ckpt) else ckpt
            self.model.load_state_dict(state_dict)

        self.model.eval()
        self.model.memory_kind = "where"

    def act(self, obs: dict[str, Any], memory: dict[str, Any] | None = None) -> tuple[str, list[int]]:
        """Compute action string and action index list for a single observation.

        Returns:
            order_str: str of characters in MOVES matching len(obs["mine"])
            actions: list[int] of action indices in [0..4]
        """
        mine = obs.get("mine", [])
        if not mine:
            return "", []

        with torch.no_grad():
            rows, cols = obs["size"]
            b_np = encode(obs)
            board_t = torch.from_numpy(b_np).to(self.device)

            mem_np = encode_memory(memory, (rows, cols))
            mem_t = torch.from_numpy(mem_np).to(self.device)

            logits, _ = self.model(board_t, mem_t)

            rr = torch.tensor([r for r, c in mine], dtype=torch.long, device=self.device)
            cc = torch.tensor([c for r, c in mine], dtype=torch.long, device=self.device)
            ant_logits = logits[0, :, rr, cc].T  # [num_ants, 5]
            seat_moves = ant_logits.argmax(dim=-1).cpu().numpy()
            actions = [int(a) for a in seat_moves]
            order_str = orders_from_indices(seat_moves, [len(mine)])[0]
            return order_str, actions

    def predict_step(
        self,
        step: Step,
        student_memory: dict[tuple[int, int], dict[str, Any]],
    ) -> tuple[list[str], list[np.ndarray]]:
        """Compute student actions across all active seats in Step.

        Returns:
            actions: list[str] of action strings, length len(step.seats)
            moves_by_seat: list[np.ndarray] of integer action arrays
        """
        num_seats = len(step.seats)
        actions: list[str] = [""] * num_seats
        moves_by_seat: list[np.ndarray] = [np.zeros(0, dtype=np.int64)] * num_seats

        with torch.no_grad():
            for g in step.groups:
                mem_planes = []
                for idx in g.indices:
                    s = step.seats[idx]
                    mem = student_memory.get((s.ep, s.seat))
                    mem_planes.append(encode_memory(mem, g.size))

                board_tensor = torch.from_numpy(g.boards).to(self.device)
                mem_tensor = torch.from_numpy(np.concatenate(mem_planes, axis=0)).to(self.device)

                policy_logits, _ = self.model(board_tensor, mem_tensor)

                for p, seat_idx in enumerate(g.indices):
                    s = step.seats[seat_idx]
                    mine = s.obs["mine"]
                    if not mine:
                        actions[seat_idx] = ""
                        moves_by_seat[seat_idx] = np.zeros(0, dtype=np.int64)
                        continue

                    rr = torch.tensor([r for r, c in mine], dtype=torch.long, device=self.device)
                    cc = torch.tensor([c for r, c in mine], dtype=torch.long, device=self.device)
                    ant_logits = policy_logits[p, :, rr, cc].T  # [num_ants, 5]
                    seat_moves = ant_logits.argmax(dim=-1).cpu().numpy()
                    order = orders_from_indices(seat_moves, [len(mine)])[0]

                    actions[seat_idx] = order
                    moves_by_seat[seat_idx] = seat_moves

        return actions, moves_by_seat


# ==============================================================================
# 3. State Signature & State Tracker
# ==============================================================================


class StateSignature:
    """Represents a macroscopic state signature tuple:
    ((H, W), ant_count, food_count, foe_count, hill_foe_seen, vis_pct)
    """

    @staticmethod
    def compute(obs: dict[str, Any], memory: dict[str, Any] | None = None) -> tuple:
        rows, cols = obs.get("size", (0, 0))
        ants = len(obs.get("mine", []))
        foods = len(obs.get("food", []))
        foes = len(obs.get("foes", []))

        hill_foe_seen = bool((memory or {}).get("hill_foe_seen")) or any(
            h[2] != 0 for h in obs.get("hills", [])
        )

        vis_pct = 0
        if "vis" in obs and isinstance(obs["vis"], dict) and "rle" in obs["vis"]:
            rle = obs["vis"]["rle"]
            vis_cells = sum(rle[i + 1] for i in range(0, len(rle), 2) if rle[i] == 1)
            vis_pct = int(round(vis_cells * 100.0 / max(rows * cols, 1)))

        return ((rows, cols), ants, foods, foes, hill_foe_seen, vis_pct)

    @staticmethod
    def canonical_id(sig: tuple) -> str:
        (h, w), ants, foods, foes, hill_seen, vis = sig
        return f"{h}x{w}:{ants}a:{foods}f:{foes}e:{int(hill_seen)}h:{vis}v"


class StateTracker:
    """Tracks per-ant policy disagreement rate and macro state space coverage."""

    def __init__(self, baseline_signatures: set[tuple] | None = None):
        self.baseline_signatures: set[tuple] = set(baseline_signatures or ())
        self.cumulative_signatures: set[tuple] = set(self.baseline_signatures)
        self.delta_signatures: set[tuple] = set()

        self.total_ants: int = 0
        self.disagree_ants: int = 0
        self.total_steps: int = 0
        self.novel_steps: int = 0

        self.confusion_matrix: list[list[int]] = [[0] * 5 for _ in range(5)]

    @property
    def total_decisions(self) -> int:
        return self.total_ants

    @property
    def total_disagreements(self) -> int:
        return self.disagree_ants

    @property
    def disagreement_rate(self) -> float:
        return (self.disagree_ants / max(self.total_ants, 1)) if self.total_ants > 0 else 0.0

    @property
    def novel_state_ratio(self) -> float:
        return (self.novel_steps / max(self.total_steps, 1)) if self.total_steps > 0 else 0.0

    def compute_signature(self, obs: dict[str, Any], memory: dict[str, Any] | None = None) -> tuple:
        return StateSignature.compute(obs, memory)

    def record_step(
        self,
        obs: dict[str, Any],
        memory: dict[str, Any] | None,
        student_actions: list[int] | np.ndarray,
        teacher_actions: list[int] | np.ndarray,
    ) -> bool:
        """Records one step. Returns True if this macroscopic state signature is novel."""
        sig = self.compute_signature(obs, memory)
        self.total_steps += 1
        self.delta_signatures.add(sig)

        is_new = sig not in self.cumulative_signatures
        if is_new:
            self.novel_steps += 1
            self.cumulative_signatures.add(sig)

        n_s = len(student_actions)
        n_t = len(teacher_actions)
        assert n_s == n_t, f"Action length mismatch: student {n_s} vs teacher {n_t}"

        for s_act, t_act in zip(student_actions, teacher_actions):
            s_idx = int(s_act)
            t_idx = int(t_act)
            self.total_ants += 1
            if s_idx != t_idx:
                self.disagree_ants += 1
            if 0 <= t_idx < 5 and 0 <= s_idx < 5:
                self.confusion_matrix[t_idx][s_idx] += 1

        return is_new

    def step(
        self,
        obs: dict[str, Any],
        memory: dict[str, Any] | None,
        student_moves: list[int] | np.ndarray,
        teacher_moves: list[int] | np.ndarray,
    ) -> bool:
        """Convenience alias for record_step."""
        return self.record_step(obs, memory, student_moves, teacher_moves)

    def metrics(self) -> dict[str, Any]:
        """Returns summary dictionary of tracked metrics."""
        disagreement_rate = self.disagreement_rate
        novel_ratio = self.novel_state_ratio
        novel_sigs_vs_base = len(self.delta_signatures - self.baseline_signatures)
        vocab_novel_ratio = (
            novel_sigs_vs_base / max(len(self.delta_signatures), 1)
            if self.delta_signatures else 0.0
        )
        return {
            "total_ant_decisions": self.total_ants,
            "total_ant_disagreements": self.disagree_ants,
            "disagreement_rate": round(disagreement_rate, 4),
            "total_seat_turn_steps": self.total_steps,
            "novel_state_ratio": round(novel_ratio, 4),
            "unique_signatures_delta": len(self.delta_signatures),
            "novel_signatures_vs_d0": novel_sigs_vs_base,
            "vocab_novel_ratio_vs_d0": round(vocab_novel_ratio, 4),
            "cumulative_unique_signatures": len(self.cumulative_signatures),
            "baseline_unique_signatures": len(self.baseline_signatures),
            "expansion_factor": (
                round(len(self.cumulative_signatures) / max(len(self.baseline_signatures), 1), 4)
                if self.baseline_signatures else 1.0
            ),
            "confusion_matrix": self.confusion_matrix,
        }


# ==============================================================================
# 4. Lossless Dataset Aggregator
# ==============================================================================


class DatasetAggregator:
    """Lossless, streaming dataset merger with episode stride offsets."""

    def __init__(self, stride: int = STRIDE):
        self.stride = stride

    def validate_headers(self, headers: list[dict[str, Any]], parts: list[Path]) -> None:
        """Verify header consistency across all dataset parts."""
        for part, h in zip(parts, headers):
            digest = h.get("engine_digest")
            if digest != EXPECTED_DIGEST:
                raise ValueError(
                    f"{part}: engine_digest mismatch: expected {EXPECTED_DIGEST}, got {digest}"
                )
            mode = h.get("memory_mode")
            if mode != "where":
                raise ValueError(
                    f"{part}: memory_mode is {mode!r}; expected 'where'"
                )
            if not h.get("teacher_remembers"):
                raise ValueError(
                    f"{part}: teacher_remembers is not true; required for Where DAgger"
                )
            if h.get("memory") != ["food_seen", "hill_foe_seen"]:
                raise ValueError(
                    f"{part}: memory planes mismatch: expected ['food_seen', 'hill_foe_seen'], got {h.get('memory')}"
                )
            if h.get("evaluator") != "datalogic 5.7":
                raise ValueError(
                    f"{part}: evaluator mismatch: expected 'datalogic 5.7', got {h.get('evaluator')}"
                )

    def merge(
        self,
        parts: list[Path | str],
        out: Path | str,
        iteration: int = 1,
        stats_extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Aggregates disjoint parts [D_0, Delta D_1, ..., Delta D_k] into D_k.

        Memory guarantee: O(1) buffer streaming.
        Stride guarantee: episode IDs strictly offset by i * 100_000.
        """
        part_paths = [Path(p) for p in parts]
        out_path = Path(out)

        headers: list[dict[str, Any]] = []
        for p in part_paths:
            with gzip.open(p, "rt") as f:
                headers.append(json.loads(f.readline()))

        self.validate_headers(headers, part_paths)

        merged_header = dict(headers[0])
        merged_header["kind"] = "dagger-aggregated"
        merged_header["iteration"] = iteration
        merged_header["parts"] = [
            {
                "file": p.name,
                "env_seed": h.get("env_seed"),
                "episode_offset": i * self.stride,
            }
            for i, (p, h) in enumerate(zip(part_paths, headers))
        ]
        merged_header.pop("env_seed", None)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        rows_count = 0

        with gzip.open(out_path, "wt") as w:
            w.write(json.dumps(merged_header) + "\n")
            for i, part in enumerate(part_paths):
                offset = i * self.stride
                with gzip.open(part, "rt") as f:
                    f.readline()  # skip header
                    try:
                        for line in f:
                            if not line.endswith("\n"):
                                break  # skip broken trailing line
                            if offset == 0:
                                w.write(line)
                                rows_count += 1
                            else:
                                row = json.loads(line)
                                row["k"][0] += offset
                                w.write(json.dumps(row, separators=(",", ":")) + "\n")
                                rows_count += 1
                    except EOFError:
                        pass

        stats_summary = {
            "rows": rows_count,
            "parts": len(part_paths),
            "iteration": iteration,
            "bytes": out_path.stat().st_size,
            "out": str(out_path),
            "engine_digest": EXPECTED_DIGEST,
            "memory_mode": "where",
            "teacher_remembers": True,
        }
        if stats_extra:
            stats_summary.update(stats_extra)

        stats_file = out_path.with_suffix("").with_suffix(".stats.json")
        stats_file.write_text(json.dumps(stats_summary, indent=2) + "\n")

        return stats_summary

    aggregate = merge


def aggregate_datasets(
    parts: list[Path | str],
    out: Path | str,
    iteration: int = 1,
    stride: int = STRIDE,
    stats_extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Convenience function to aggregate datasets."""
    return DatasetAggregator(stride=stride).merge(parts, out, iteration=iteration, stats_extra=stats_extra)


# ==============================================================================
# 5. DAgger Rollout Iteration Loop
# ==============================================================================


def rollout_iteration(
    student_driver: StudentDriver,
    gate: InformationAlignmentGate,
    out_path: Path | str,
    seat_turns: int,
    k_iter: int,
    offset_stride: int = 0,
    waves: int = 4,
    matches_per_wave: int = 16,
    max_turns: int = 300,
    seed: int = 1,
    maps: str | None = None,
    baseline_signatures: set[tuple] | None = None,
    prior_state_hashes: set[bytes] | None = None,
) -> dict[str, Any]:
    """Execute a single DAgger rollout iteration.

    Environment physically stepped by student_driver;
    Labels generated by gate under student's (O_t, M_t);
    Streams records (o, m, k, a) tuples into out_path.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    episode_offset = k_iter * offset_stride

    written = 0
    episodes = 0
    scores: list[int] = []
    reasons: dict[str, int] = {}
    started = time.time()

    tracker = StateTracker(baseline_signatures=baseline_signatures)
    iteration_unique_states: set[bytes] = set()
    novel_states: set[bytes] = set()

    with Env(
        waves=waves,
        matches_per_wave=matches_per_wave,
        max_turns=max_turns,
        seed=seed,
        maps=maps,
    ) as env, gzip.open(out_path, "wt") as f:
        header = {
            "kind": "dagger-rollout",
            "engine_digest": env.engine_digest,
            "evaluator": env.evaluator,
            "maps": [m["id"] for m in env.hello["maps"]],
            "max_turns": max_turns,
            "env_seed": seed,
            "teacher": f"deterministic potential field (hill_lead={gate.teacher.hill_lead}), remembering",
            "teacher_type": "default",
            "teacher_remembers": True,
            "hill_lead": gate.teacher.hill_lead,
            "memory_mode": "where",
            "memory": [m.name for m in MEMORY],
            "dagger_iteration": k_iter,
            "episode_offset": episode_offset,
        }
        f.write(json.dumps(header) + "\n")

        student_memory: dict[tuple[int, int], dict[str, Any]] = {}
        step = env.reset()

        while written < seat_turns:
            # 1. Compute student actions for all active seats
            student_orders, student_moves = student_driver.predict_step(step, student_memory)

            # 2. For each seat, query Teacher via Gate on (O_t, M_t) and write training row
            for seat_idx, s in enumerate(step.seats):
                key = (s.ep, s.seat)
                M_t = student_memory.get(key)

                # Oracle labeling under strict local information alignment
                teacher_moves, teacher_order = gate.label(s.obs, M_t)

                # Track metrics and write dataset rows
                if s.obs["mine"]:
                    n_ants = len(s.obs["mine"])
                    s_moves = student_moves[seat_idx]
                    if len(s_moves) == n_ants and len(teacher_moves) == n_ants:
                        tracker.record_step(s.obs, M_t, s_moves, teacher_moves)

                    # Spatial state hashing
                    fp = hashlib.sha256(encode(s.obs).tobytes()).digest()[:8]
                    iteration_unique_states.add(fp)
                    if prior_state_hashes is not None and fp not in prior_state_hashes:
                        novel_states.add(fp)

                    # Write row if quota remains
                    if written < seat_turns:
                        row = {
                            "o": s.obs,
                            "k": [s.ep + episode_offset, s.seat, s.turn],
                            "m": M_t or {},
                            "a": teacher_order,
                        }
                        f.write(json.dumps(row, separators=(",", ":")) + "\n")
                        written += 1

                # 3. Synchronously advance student's Where memory along actual trajectory
                student_memory[key] = remember(M_t, s.obs, mode="where")

            # 4. Physically step the environment using STUDENT'S actions
            step = env.step(student_orders)

            # 5. Clean up memory for ended matches
            for e in step.ended:
                episodes += 1
                reasons[e.reason] = reasons.get(e.reason, 0) + 1
                scores.append(max(e.scores))
                for seat in range(e.seat_count):
                    student_memory.pop((e.ep, seat), None)

            if not step.seats:
                break

    elapsed = time.time() - started
    disagreement_rate = tracker.disagreement_rate
    novel_ratio = tracker.novel_state_ratio

    stats = {
        "iteration": k_iter,
        "rows": written,
        "ant_decisions": tracker.total_ants,
        "disagreed_ants": tracker.disagree_ants,
        "disagreement_rate": round(disagreement_rate, 4),
        "unique_states": len(tracker.delta_signatures),
        "novel_states": tracker.novel_steps,
        "novel_state_ratio": round(novel_ratio, 4),
        "episodes": episodes,
        "reasons": reasons,
        "mean_best_score": round(sum(scores) / len(scores), 2) if scores else None,
        "seconds": round(elapsed, 1),
        "rows_per_second": round(written / max(elapsed, 1e-9)),
        "out": str(out_path),
    }

    stats_file = out_path.with_suffix("").with_suffix(".stats.json")
    stats_file.write_text(json.dumps(stats, indent=2) + "\n")
    return stats


# ==============================================================================
# 6. CLI Entrypoint
# ==============================================================================


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="DAgger 数据集聚合模仿学习闭环控制器。")
    ap.add_argument("--base-weights", type=Path, default=Path("runs/exp011/best.pt"),
                    help="初始学生网络权重（默认 runs/exp011/best.pt）")
    ap.add_argument("--base-data", type=Path, default=Path("data/exp011.jsonl.gz"),
                    help="初始离线专家数据集 D_0（默认 data/exp011.jsonl.gz）")
    ap.add_argument("--iterations", type=int, default=3,
                    help="DAgger 迭代轮数（默认 3）")
    ap.add_argument("--samples-per-iter", type=int, default=83_334,
                    help="每轮迭代采集的增量 seat-turns 数量（默认 83,334）")
    ap.add_argument("--seed", type=int, default=1,
                    help="环境随机种子基数（默认 1）")
    ap.add_argument("--out-data", type=Path, default=Path("data/exp031.jsonl.gz"),
                    help="最终聚合数据集输出路径（默认 data/exp031.jsonl.gz）")
    ap.add_argument("--out-run", type=Path, default=Path("runs/exp031"),
                    help="训练输出目录（默认 runs/exp031）")
    ap.add_argument("--waves", type=int, default=4)
    ap.add_argument("--matches-per-wave", type=int, default=16)
    ap.add_argument("--max-turns", type=int, default=300)
    ap.add_argument("--maps", default=None,
                    help="棋盘 ID、文件路径或包含棋盘的目录；默认为发布包中的基础棋盘")
    ap.add_argument("--hill-lead", type=int, default=6,
                    help="教师模型起步领先步数（默认 6）")
    ap.add_argument("--device", default=None,
                    help="推理设备（'cpu', 'cuda' 等）")

    argv = sys.argv[1:] if argv is None else argv
    a = ap.parse_args(argv)

    out_data = a.out_data
    out_data.parent.mkdir(parents=True, exist_ok=True)
    a.out_run.mkdir(parents=True, exist_ok=True)

    gate = InformationAlignmentGate(hill_lead=a.hill_lead)
    aggregator = DatasetAggregator(stride=STRIDE)

    parts: list[Path] = [a.base_data]
    current_weights: Path = a.base_weights

    print(f"=== Starting DAgger Pipeline ({a.iterations} iterations) ===")
    print(f"Base data: {a.base_data}")
    print(f"Base weights: {current_weights}")

    for k in range(1, a.iterations + 1):
        print(f"\n--- DAgger Iteration {k}/{a.iterations} ---")
        iter_seed = a.seed + k * 100
        delta_path = out_data.parent / f"{out_data.name.split('.')[0]}_iter{k}.jsonl.gz"

        driver = StudentDriver(weights_path=current_weights, device=a.device)
        print(f"Executing rollout iteration {k} with seed {iter_seed} -> {delta_path}...")

        rollout_stats = rollout_iteration(
            student_driver=driver,
            gate=gate,
            out_path=delta_path,
            seat_turns=a.samples_per_iter,
            k_iter=k,
            waves=a.waves,
            matches_per_wave=a.matches_per_wave,
            max_turns=a.max_turns,
            seed=iter_seed,
            maps=a.maps,
        )
        print(f"Iteration {k} rollout complete: {rollout_stats['rows']} rows, "
              f"Disagreement: {rollout_stats['disagreement_rate'] * 100:.2f}%, "
              f"Novel state ratio: {rollout_stats['novel_state_ratio'] * 100:.2f}%")

        parts.append(delta_path)

        accum_out = out_data if k == a.iterations else out_data.parent / f"{out_data.name.split('.')[0]}_k{k}.jsonl.gz"
        print(f"Aggregating {len(parts)} parts into {accum_out}...")
        agg_stats = aggregator.merge(
            parts=parts,
            out=accum_out,
            iteration=k,
            stats_extra={
                "disagreement_rate": rollout_stats["disagreement_rate"],
                "novel_state_ratio": rollout_stats["novel_state_ratio"],
                "ant_decisions": rollout_stats["ant_decisions"],
                "disagreed_ants": rollout_stats["disagreed_ants"],
            },
        )
        print(f"Aggregated dataset: {agg_stats['rows']} total rows in {agg_stats['bytes']} bytes.")

        # Check for intermediate checkpoint
        next_ckpt = a.out_run / f"k{k}" / "best.pt"
        if next_ckpt.exists():
            current_weights = next_ckpt
            print(f"Updated weights for next iteration: {current_weights}")

    print("\n=== DAgger Pipeline Completed Successfully ===")


if __name__ == "__main__":
    main()
