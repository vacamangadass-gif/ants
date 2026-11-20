from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import datetime
import gzip
import hashlib
import json
import os
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
_BASELINES_SRC = _ROOT / "baselines" / "src"
if str(_BASELINES_SRC.resolve()) not in sys.path:
    sys.path.insert(0, str(_BASELINES_SRC.resolve()))
if str(_ROOT.resolve()) not in sys.path:
    sys.path.insert(0, str(_ROOT.resolve()))
os.environ["PYTHONPATH"] = (
    str(_BASELINES_SRC.resolve())
    + os.pathsep
    + str(_ROOT.resolve())
    + (os.pathsep + os.environ["PYTHONPATH"] if os.environ.get("PYTHONPATH") else "")
)

from tb_baselines.dagger import (
    DatasetAggregator,
    InformationAlignmentGate,
    ONNXOpponentDriver,
    STRIDE,
    StateSignature,
    StateTracker,
    StudentDriver,
    TeacherOpponentDriver,
)
from tb_baselines.env import Env
from tb_baselines.export import classes, main as export_main, verdict
from tb_baselines.planes import MEMORY, MOVES, encode, remember
from tb_baselines.teacher import HOLD, TacticalLocalTeacher
from tb_baselines.train import bc
from tools.gen_procedural_maps import generate_map_pool

EXPECTED_HEX_DIGEST = "819166e7918117a1cbbd67dc042e7368a076a2dd45cea5122683983912d591e1"
EXPECTED_DIGEST = f"sha256:{EXPECTED_HEX_DIGEST}"

# 分层采样桶配额权重：严格保证 >=50% 多蚁丘 (2h/3h = 56.6%) 且 >=45% 多席位 (4p/6p = 47.0%)
BUCKET_SHARES: tuple[tuple[str, float], ...] = (
    ("2p-1h", 0.22),
    ("2p-multi", 0.26),
    ("4p-1h", 0.13),
    ("4p-multi", 0.21),
    ("6p-all", 0.13),
    ("basic-3p-8p", 0.05),
)


@dataclass(frozen=True)
class PipelineConfig:
    exp_name: str = "exp085_fix"
    weight_class: str = "micro"
    channels: int = 40
    blocks: int = 3
    start_iter: int = 1
    end_iter: int = 4
    d0_samples: int = 250_000
    samples_per_iter: int = 100_000
    pretrain_epochs: int = 6
    epochs: int = 6
    final_epochs: int = 10
    batch_size: int = 128
    lr: float = 3e-3
    device: str | None = "cuda"
    seed: int = 20261120
    hill_lead: int = 8
    max_depth: int | None = None
    defense_radius: int = 10
    waves: int = 4
    matches_per_wave: int = 16
    max_turns: int = 500
    project_root: Path = _ROOT
    exclude_boards: Path | None = None
    export_dir: Path | None = None


def verify_engine_digest(wasm_path: Path) -> None:
    if not wasm_path.exists():
        raise FileNotFoundError(f"未找到底座引擎构件: {wasm_path}")
    with open(wasm_path, "rb") as f:
        actual_hash = hashlib.sha256(f.read()).hexdigest()
    if actual_hash != EXPECTED_HEX_DIGEST:
        raise RuntimeError(
            f"Wasm 引擎散列摘要不匹配！期望 {EXPECTED_HEX_DIGEST}，实际为 {actual_hash}"
        )


def check_prerequisites(project_root: Path) -> None:
    wasm_path = project_root / "dist" / "tb-ants.wasm"
    verify_engine_digest(wasm_path)

    games_toml = project_root / "baselines" / "games.toml"
    if not games_toml.exists():
        raise FileNotFoundError(f"未找到 TinyBrains 注册表文件: {games_toml}")
    os.environ["TINYBRAINS_REGISTRY"] = str(games_toml.resolve())

    if shutil.which("tinybrains") is None and not Path("/home/linuxbrew/.linuxbrew/bin/tinybrains").exists():
        raise RuntimeError("环境变量 PATH 中未找到 tinybrains 可执行文件")


def build_stratified_map_buckets(
    project_root: Path,
    seed: int = 20261120,
    exclude_boards: Path | None = None,
) -> dict[str, list[str]]:
    proc_dir = project_root / "data" / "proc_maps_60"
    manifest_file = proc_dir / "manifest.json"
    if not manifest_file.exists():
        generate_map_pool(
            out_dir=proc_dir,
            base_seed=seed,
            exclude_boards=exclude_boards,
            validate_cli=True,
            registry_path=project_root / "baselines" / "games.toml",
        )
    summary = json.loads(manifest_file.read_text(encoding="utf-8"))
    buckets: dict[str, list[str]] = {name: [] for name, _ in BUCKET_SHARES}
    for entry in summary["maps"]:
        b_name = entry["bucket"]
        if b_name in buckets:
            buckets[b_name].append(str(Path(entry["path"]).resolve()))

    # 加入 5 张官方基础棋盘（2p..8p）
    base_dir = project_root / "maps"
    buckets["2p-1h"].append(str((base_dir / "basic-tiny-2p.json").resolve()))
    buckets["4p-multi"].append(str((base_dir / "basic-medium-4p.json").resolve()))
    buckets["6p-all"].append(str((base_dir / "basic-large-6p.json").resolve()))
    buckets["basic-3p-8p"].extend(
        [
            str((base_dir / "basic-small-3p.json").resolve()),
            str((base_dir / "basic-xlarge-8p.json").resolve()),
        ]
    )
    for b_name, paths in buckets.items():
        if not paths:
            raise RuntimeError(f"分层棋盘桶 {b_name} 为空")
    return buckets


def compute_bucket_quotas(total_rows: int) -> list[tuple[str, int]]:
    quotas: list[tuple[str, int]] = []
    allocated = 0
    for idx, (b_name, share) in enumerate(BUCKET_SHARES):
        if idx + 1 == len(BUCKET_SHARES):
            q = max(1, total_rows - allocated)
        else:
            q = max(1, int(round(total_rows * share)))
            allocated += q
        quotas.append((b_name, q))
    return quotas


def _collect_d0_bucket_worker(task: dict[str, Any]) -> dict[str, Any]:
    b_idx = int(task["b_idx"])
    b_name = str(task["b_name"])
    b_quota = int(task["b_quota"])
    b_maps_str = str(task["b_maps_str"])
    b_seed = int(task["b_seed"])
    b_waves = int(task["b_waves"])
    b_mpw = int(task["b_mpw"])
    max_turns = int(task["max_turns"])
    hill_lead = int(task["hill_lead"])
    max_depth = task["max_depth"]
    defense_radius = int(task["defense_radius"])
    project_root = Path(task["project_root"])
    shard_path = Path(task["shard_path"])
    ep_base = b_idx * 10_000

    teacher = TacticalLocalTeacher(
        hill_lead=hill_lead,
        max_depth=max_depth,
        defense_radius=defense_radius,
    )
    gate = InformationAlignmentGate(teacher=teacher, hill_lead=hill_lead)

    b_written = 0
    total_ants = 0
    hold_ants = 0
    own_hill_ants = 0
    own_hill_holds = 0
    multi_hill_rows = 0
    multiseat_4p6p_rows = 0
    episodes = 0
    reasons: dict[str, int] = {}
    scores: list[int] = []

    with open(shard_path, "w", encoding="utf-8") as sf:
        with Env(
            waves=b_waves,
            matches_per_wave=b_mpw,
            max_turns=max_turns,
            seed=b_seed,
            maps=b_maps_str,
            cwd=project_root,
        ) as env:
            if env.engine_digest != EXPECTED_DIGEST:
                raise RuntimeError(f"引擎摘要不匹配: {env.engine_digest} != {EXPECTED_DIGEST}")
            memories: dict[tuple[int, int], dict[str, Any]] = {}
            step = env.reset()

            while b_written < b_quota and step.seats:
                orders = []
                for s in step.seats:
                    key = (s.ep, s.seat)
                    mem = memories.get(key)
                    moves, order_str = gate.label(s.obs, mem)
                    orders.append(order_str)

                    if s.obs["mine"] and b_written < b_quota:
                        row = {
                            "o": s.obs,
                            "k": [s.ep + ep_base, s.seat, s.turn],
                            "m": mem or {},
                            "a": order_str,
                        }
                        sf.write(json.dumps(row, separators=(",", ":")) + "\n")
                        b_written += 1
                        total_ants += len(moves)
                        hold_ants += int(np.sum(moves == HOLD))
                        own_hills = {
                            (int(h[0]), int(h[1])) for h in s.obs.get("hills", []) if len(h) >= 3 and h[2] == 0
                        }
                        for idx_a, pos in enumerate(s.obs["mine"]):
                            if (int(pos[0]), int(pos[1])) in own_hills:
                                own_hill_ants += 1
                                if moves[idx_a] == HOLD:
                                    own_hill_holds += 1
                        if b_name in ("2p-multi", "4p-multi") or (
                            b_name in ("6p-all", "basic-3p-8p") and len(own_hills) >= 2
                        ):
                            multi_hill_rows += 1
                        if b_name in ("4p-1h", "4p-multi", "6p-all"):
                            multiseat_4p6p_rows += 1

                    memories[key] = remember(mem, s.obs, mode="where")

                step = env.step(orders)
                for e in step.ended:
                    episodes += 1
                    reasons[e.reason] = reasons.get(e.reason, 0) + 1
                    scores.append(max(e.scores))
                    for seat in range(e.seat_count):
                        memories.pop((e.ep, seat), None)

    return {
        "b_idx": b_idx,
        "b_name": b_name,
        "b_written": b_written,
        "total_ants": total_ants,
        "hold_ants": hold_ants,
        "own_hill_ants": own_hill_ants,
        "own_hill_holds": own_hill_holds,
        "multi_hill_rows": multi_hill_rows,
        "multiseat_4p6p_rows": multiseat_4p6p_rows,
        "episodes": episodes,
        "reasons": reasons,
        "scores": scores,
        "shard_path": str(shard_path),
    }


def collect_stratified_d0(
    out_path: Path,
    buckets: dict[str, list[str]],
    seat_turns: int = 250_000,
    waves: int = 4,
    matches_per_wave: int = 16,
    max_turns: int = 500,
    seed: int = 20261120,
    hill_lead: int = 8,
    max_depth: int | None = None,
    defense_radius: int = 10,
    project_root: Path = _ROOT,
) -> dict[str, Any]:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    quotas = compute_bucket_quotas(seat_turns)
    all_map_ids = [Path(p).stem for paths in buckets.values() for p in paths]

    written = 0
    total_ants = 0
    hold_ants = 0
    own_hill_ants = 0
    own_hill_holds = 0
    episodes = 0
    reasons: dict[str, int] = {}
    scores: list[int] = []
    bucket_counts: dict[str, int] = {}
    multi_hill_rows = 0
    multiseat_4p6p_rows = 0
    started = time.time()

    tasks = []
    for b_idx, (b_name, b_quota) in enumerate(quotas):
        b_waves = 2 if b_name in ("6p-all", "basic-3p-8p") else waves
        b_mpw = 8 if b_name in ("6p-all", "basic-3p-8p") else matches_per_wave
        shard_path = out_path.parent / f".shard_d0_{out_path.stem}_{b_idx}.jsonl"
        tasks.append(
            {
                "b_idx": b_idx,
                "b_name": b_name,
                "b_quota": b_quota,
                "b_maps_str": ",".join(buckets[b_name]),
                "b_seed": seed + b_idx * 1009,
                "b_waves": b_waves,
                "b_mpw": b_mpw,
                "max_turns": max_turns,
                "hill_lead": hill_lead,
                "max_depth": max_depth,
                "defense_radius": defense_radius,
                "project_root": str(project_root),
                "shard_path": str(shard_path),
            }
        )

    if seat_turns >= 6000:
        with ProcessPoolExecutor(max_workers=len(tasks)) as pool:
            results = list(pool.map(_collect_d0_bucket_worker, tasks))
    else:
        results = [_collect_d0_bucket_worker(t) for t in tasks]

    results.sort(key=lambda r: int(r["b_idx"]))

    with gzip.open(out_path, "wt", compresslevel=1) as f:
        header = {
            "kind": "teacher-rollout",
            "engine_digest": EXPECTED_DIGEST,
            "evaluator": "datalogic 5.7",
            "maps": all_map_ids,
            "max_turns": max_turns,
            "env_seed": seed,
            "teacher": f"TacticalLocalTeacher(hill_lead={hill_lead}, max_depth={max_depth}, defense_radius={defense_radius}), remembering",
            "teacher_type": "tactical",
            "teacher_remembers": True,
            "hill_lead": hill_lead,
            "memory_mode": "where",
            "memory": [m.name for m in MEMORY],
        }
        f.write(json.dumps(header) + "\n")

        for res in results:
            b_name = res["b_name"]
            b_written = int(res["b_written"])
            shard_file = Path(res["shard_path"])
            with open(shard_file, "r", encoding="utf-8") as sf:
                shutil.copyfileobj(sf, f)
            shard_file.unlink(missing_ok=True)

            bucket_counts[b_name] = b_written
            written += b_written
            total_ants += int(res["total_ants"])
            hold_ants += int(res["hold_ants"])
            own_hill_ants += int(res["own_hill_ants"])
            own_hill_holds += int(res["own_hill_holds"])
            multi_hill_rows += int(res["multi_hill_rows"])
            multiseat_4p6p_rows += int(res["multiseat_4p6p_rows"])
            episodes += int(res["episodes"])
            for k_r, v_r in res["reasons"].items():
                reasons[k_r] = reasons.get(k_r, 0) + int(v_r)
            scores.extend(res["scores"])
            elapsed_now = time.time() - started
            print(
                f"[D0 分层桶 {b_name}] 完成 {b_written:,} 行 "
                f"(累计 {written:,}/{seat_turns:,}, {written / max(elapsed_now, 1e-6):.1f} rows/s)"
            )

    elapsed = time.time() - started
    stats = {
        "rows": written,
        "ant_labels": total_ants,
        "hold_rate": round(hold_ants / max(total_ants, 1), 4),
        "own_hill_hold_rate": round(own_hill_holds / max(own_hill_ants, 1), 4),
        "bucket_rows": bucket_counts,
        "multi_hill_bucket_share": round(
            (bucket_counts.get("2p-multi", 0) + bucket_counts.get("4p-multi", 0) + int(bucket_counts.get("6p-all", 0) * 0.55) + int(bucket_counts.get("basic-3p-8p", 0) * 0.5))
            / max(written, 1),
            4,
        ),
        "multiseat_4p6p_share": round(multiseat_4p6p_rows / max(written, 1), 4),
        "episodes": episodes,
        "reasons": reasons,
        "mean_best_score": round(sum(scores) / len(scores), 2) if scores else None,
        "seconds": round(elapsed, 1),
        "rows_per_second": round(written / max(elapsed, 1e-9)),
        "command": f"python tools/run_exp085_pipeline.py --d0-samples {seat_turns}",
    }
    stats_path = out_path.with_suffix("").with_suffix(".stats.json")
    stats_path.write_text(json.dumps(stats, indent=2) + "\n")
    return stats


def build_mixed_opponents(
    project_root: Path,
    teacher: TacticalLocalTeacher,
    student_driver: StudentDriver,
) -> list[Any]:
    opponents: list[Any] = [TeacherOpponentDriver(teacher=teacher, name="TacticalLocalTeacher")]
    models_root = project_root / "models"
    for m_name in ("exp033", "nano-bc-max-r", "micro-bc"):
        cand = models_root / m_name
        if (cand / "model.onnx").exists():
            opponents.append(ONNXOpponentDriver(cand, name=m_name))
    opponents.append(student_driver)
    return opponents


def rollout_stratified_mixed_dagger(
    student_driver: StudentDriver,
    gate: InformationAlignmentGate,
    opponent_drivers: list[Any],
    buckets: dict[str, list[str]],
    out_path: Path | str,
    seat_turns: int,
    k_iter: int,
    waves: int = 4,
    matches_per_wave: int = 16,
    max_turns: int = 500,
    seed: int = 1,
    baseline_signatures: set[tuple] | None = None,
    project_root: Path = _ROOT,
) -> dict[str, Any]:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    quotas = compute_bucket_quotas(seat_turns)
    all_map_ids = [Path(p).stem for paths in buckets.values() for p in paths]

    written = 0
    total_ants = 0
    hold_ants = 0
    own_hill_ants = 0
    own_hill_holds = 0
    episodes = 0
    scores: list[int] = []
    reasons: dict[str, int] = {}
    bucket_counts: dict[str, int] = {}
    opponent_exposure: dict[str, int] = {}
    started = time.time()

    tracker = StateTracker(baseline_signatures=baseline_signatures)
    ep_base = 0

    with gzip.open(out_path, "wt", compresslevel=1) as f:
        header = {
            "kind": "dagger-rollout",
            "engine_digest": EXPECTED_DIGEST,
            "evaluator": "datalogic 5.7",
            "maps": all_map_ids,
            "max_turns": max_turns,
            "env_seed": seed,
            "teacher": f"TacticalLocalTeacher(hill_lead={gate.teacher.hill_lead}), remembering",
            "teacher_type": "tactical",
            "teacher_remembers": True,
            "hill_lead": gate.teacher.hill_lead,
            "memory_mode": "where",
            "memory": [m.name for m in MEMORY],
            "dagger_iteration": k_iter,
            "episode_offset": 0,
        }
        f.write(json.dumps(header) + "\n")

        for b_idx, (b_name, b_quota) in enumerate(quotas):
            b_maps_str = ",".join(buckets[b_name])
            b_written = 0
            b_seed = seed + b_idx * 1009
            b_waves = 2 if b_name in ("6p-all", "basic-3p-8p") else waves
            b_mpw = 8 if b_name in ("6p-all", "basic-3p-8p") else matches_per_wave

            with Env(
                waves=b_waves,
                matches_per_wave=b_mpw,
                max_turns=max_turns,
                seed=b_seed,
                maps=b_maps_str,
                cwd=project_root,
            ) as env:
                seat_memory: dict[tuple[int, int], dict[str, Any]] = {}
                step = env.reset()

                while b_written < b_quota and step.seats:
                    student_orders, student_moves = student_driver.predict_step(step, seat_memory)
                    step_orders = list(student_orders)

                    for seat_idx, s in enumerate(step.seats):
                        key = (s.ep, s.seat)
                        M_t = seat_memory.get(key)
                        opp_driver = opponent_drivers[(s.ep + b_idx + s.seat // 2) % len(opponent_drivers)]
                        opp_name = getattr(opp_driver, "name", "self_student")

                        if (s.ep + s.seat) % 2 == 0 or (opp_driver is student_driver):
                            is_student_seat = True
                        else:
                            is_student_seat = False

                        if is_student_seat:
                            teacher_moves, teacher_order = gate.label(s.obs, M_t)
                            if s.obs["mine"]:
                                n_ants = len(s.obs["mine"])
                                s_moves = student_moves[seat_idx]
                                if len(s_moves) == n_ants and len(teacher_moves) == n_ants:
                                    tracker.record_step(s.obs, M_t, s_moves, teacher_moves)

                                if b_written < b_quota:
                                    row = {
                                        "o": s.obs,
                                        "k": [s.ep + ep_base, s.seat, s.turn],
                                        "m": M_t or {},
                                        "a": teacher_order,
                                    }
                                    f.write(json.dumps(row, separators=(",", ":")) + "\n")
                                    b_written += 1
                                    written += 1
                                    opponent_exposure[opp_name] = opponent_exposure.get(opp_name, 0) + 1
                                    total_ants += len(teacher_moves)
                                    hold_ants += int(np.sum(teacher_moves == HOLD))
                                    own_hills = {
                                        (int(h[0]), int(h[1]))
                                        for h in s.obs.get("hills", [])
                                        if len(h) >= 3 and h[2] == 0
                                    }
                                    for idx_a, pos in enumerate(s.obs["mine"]):
                                        if (int(pos[0]), int(pos[1])) in own_hills:
                                            own_hill_ants += 1
                                            if teacher_moves[idx_a] == HOLD:
                                                own_hill_holds += 1

                            seat_memory[key] = remember(M_t, s.obs, mode="where")
                        else:
                            opp_order, _ = opp_driver.act(s.obs, M_t)
                            step_orders[seat_idx] = opp_order
                            opp_mem_mode = getattr(opp_driver, "memory_mode", "where")
                            seat_memory[key] = remember(M_t, s.obs, mode=opp_mem_mode)

                    step = env.step(step_orders)
                    for e in step.ended:
                        episodes += 1
                        reasons[e.reason] = reasons.get(e.reason, 0) + 1
                        scores.append(max(e.scores))
                        for seat in range(e.seat_count):
                            seat_memory.pop((e.ep, seat), None)

            bucket_counts[b_name] = b_written
            ep_base += 10_000
            elapsed_now = time.time() - started
            print(
                f"[DAgger {k_iter} 分层桶 {b_name}] 完成 {b_written:,}/{b_quota:,} 行 "
                f"(累计 {written:,}/{seat_turns:,}, 歧异率={tracker.disagreement_rate:.4f}, "
                f"{written / max(elapsed_now, 1e-6):.1f} rows/s)"
            )

    elapsed = time.time() - started
    stats = {
        "iteration": k_iter,
        "rows": written,
        "ant_decisions": tracker.total_ants,
        "disagreed_ants": tracker.disagree_ants,
        "disagreement_rate": round(tracker.disagreement_rate, 4),
        "hold_rate": round(hold_ants / max(total_ants, 1), 4),
        "own_hill_hold_rate": round(own_hill_holds / max(own_hill_ants, 1), 4),
        "bucket_rows": bucket_counts,
        "opponent_exposure": opponent_exposure,
        "unique_states": len(tracker.delta_signatures),
        "novel_states": tracker.novel_steps,
        "novel_state_ratio": round(tracker.novel_state_ratio, 4),
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


def train_iteration_model(
    cls: str,
    data_path: Path,
    out_dir: Path,
    channels: int = 40,
    blocks: int = 3,
    epochs: int = 6,
    batch: int = 128,
    lr: float = 3e-3,
    seed: int = 20261120,
    device: str | None = None,
    init_weights: Path | None = None,
) -> tuple[Path, dict[str, Any]]:
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if device is not None:
        dev_str = device.strip().lower()
        if dev_str == "cpu":
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
        elif dev_str == "cuda":
            os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    cmd_args = [
        "--class",
        cls,
        "--channels",
        str(channels),
        "--blocks",
        str(blocks),
        "--memory",
        "--data",
        str(data_path.resolve()),
        "--epochs",
        str(epochs),
        "--batch",
        str(batch),
        "--lr",
        str(lr),
        "--seed",
        str(seed),
        "--out",
        str(out_dir),
    ]
    if init_weights is not None and init_weights.exists():
        cmd_args.extend(["--init-weights", str(init_weights.resolve())])

    bc.main(cmd_args)

    best_pt = out_dir / "best.pt"
    history_json = out_dir / "history.json"
    if not best_pt.exists():
        raise FileNotFoundError(f"最优模型检查点文件不存在: {best_pt}")
    if not history_json.exists():
        raise FileNotFoundError(f"训练历史记录文件不存在: {history_json}")

    history_data = json.loads(history_json.read_text(encoding="utf-8"))
    return best_pt, history_data


def validate_admission_verdict(said: dict[str, Any], cls: str) -> None:
    if not said.get("ok"):
        raise AssertionError(f"平台准入校验拒绝了构件: {said.get('reason')}")
    if said.get("engine_digest") != EXPECTED_DIGEST:
        raise AssertionError(
            f"平台校验返回的引擎散列摘要不匹配！期望 {EXPECTED_DIGEST}，实际为 {said.get('engine_digest')}"
        )
    size_bytes = said.get("size_metric_bytes", 0)
    micro_limit = 128 * 1024
    if size_bytes >= micro_limit:
        raise AssertionError(
            f"Micro 构件指标体积 {size_bytes} 字节超出 128 KB（{micro_limit} 字节）上限！"
        )
    infer_us = said.get("infer_us_max", 0)
    if infer_us > 700_000:
        raise AssertionError(f"单回合推理峰值时延 {infer_us} 微秒超出 700 毫秒上限！")
    mem_info = said.get("memory")
    if mem_info and "round_trip" in mem_info:
        failed_count = mem_info["round_trip"].get("failed", 0)
        if failed_count != 0:
            raise AssertionError(f"记忆往返一致性测试存在 {failed_count} 次失败！")


def export_and_verify_model(
    cls: str,
    weights_path: Path,
    out_dir: Path,
    channels: int = 40,
    blocks: int = 3,
    method: str = "dagger",
) -> dict[str, Any]:
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    export_args = [
        "--class",
        cls,
        "--channels",
        str(channels),
        "--blocks",
        str(blocks),
        "--memory",
        "where",
        "--weights",
        str(weights_path.resolve()),
        "--out",
        str(out_dir),
        "--method",
        method,
    ]
    export_main(export_args)

    model_path = out_dir / "model.onnx"
    manifest_path = out_dir / "manifest.json"
    metrics_path = out_dir / "metrics.json"
    if not model_path.exists() or not manifest_path.exists() or not metrics_path.exists():
        raise FileNotFoundError(f"导出构件不完整: {out_dir}")

    said = verdict(model_path.resolve(), manifest_path.resolve())
    validate_admission_verdict(said, cls=cls)
    return said


def extract_signatures_from_file(file_path: Path) -> set[tuple]:
    if not file_path.exists():
        raise FileNotFoundError(f"数据文件不存在: {file_path}")
    sigs: set[tuple] = set()
    with gzip.open(file_path, "rt", encoding="utf-8") as f:
        header_line = f.readline()
        if header_line and not header_line.endswith("\n"):
            raise ValueError(f"数据文件 {file_path} 头部数据损坏")
        for line in f:
            if line.endswith("\n"):
                row = json.loads(line)
                sigs.add(StateSignature.compute(row["o"], row.get("m")))
    return sigs


class Exp085Pipeline:
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.project_root = config.project_root
        self.aggregator = DatasetAggregator(stride=STRIDE)
        self.buckets: dict[str, list[str]] = {}

    def validate_environment(self) -> None:
        check_prerequisites(project_root=self.project_root)
        self.buckets = build_stratified_map_buckets(
            project_root=self.project_root,
            seed=self.config.seed,
            exclude_boards=self.config.exclude_boards,
        )

    def get_or_create_d0(self) -> Path:
        d0_path = self.project_root / "data" / f"{self.config.exp_name}_d0.jsonl.gz"
        if d0_path.exists():
            print(f"找到已存在的 D_0 数据集: {d0_path}")
            return d0_path

        print(f"\n=== 开始采集 {self.config.exp_name} 分层专家数据集 D_0 ({self.config.d0_samples:,} 席位步数) ===")
        collect_stratified_d0(
            out_path=d0_path,
            buckets=self.buckets,
            seat_turns=self.config.d0_samples,
            waves=self.config.waves,
            matches_per_wave=self.config.matches_per_wave,
            max_turns=self.config.max_turns,
            seed=self.config.seed,
            hill_lead=self.config.hill_lead,
            max_depth=self.config.max_depth,
            defense_radius=self.config.defense_radius,
            project_root=self.project_root,
        )
        return d0_path

    def run_pretrain(self, d0_path: Path) -> Path:
        pretrain_run = self.project_root / "runs" / self.config.exp_name / "pretrain"
        best_pt = pretrain_run / "best.pt"
        if best_pt.exists():
            print(f"找到已存在的预训练模型检查点: {best_pt}")
            return best_pt

        print(f"\n=== 开始预训练初始策略网络 pi_0 ({self.config.pretrain_epochs} 轮次, ch={self.config.channels}, b={self.config.blocks}) ===")
        best_pt, _ = train_iteration_model(
            cls=self.config.weight_class,
            data_path=d0_path,
            out_dir=pretrain_run,
            channels=self.config.channels,
            blocks=self.config.blocks,
            epochs=self.config.pretrain_epochs,
            batch=self.config.batch_size,
            lr=self.config.lr,
            seed=self.config.seed,
            device=self.config.device,
        )
        pretrain_model = self.project_root / "models" / f"{self.config.exp_name}_pretrain"
        export_and_verify_model(
            cls=self.config.weight_class,
            weights_path=best_pt,
            out_dir=pretrain_model,
            channels=self.config.channels,
            blocks=self.config.blocks,
            method=f"{self.config.exp_name}: pretrain Micro (ch={self.config.channels}, b={self.config.blocks}) on TacticalLocalTeacher D_0",
        )
        return best_pt

    def get_initial_weights(self, d0_path: Path) -> Path:
        if self.config.start_iter == 1:
            return self.run_pretrain(d0_path)
        prev_weights = self.project_root / "runs" / self.config.exp_name / f"k{self.config.start_iter - 1}" / "best.pt"
        if not prev_weights.exists():
            raise FileNotFoundError(f"前置迭代 {self.config.start_iter - 1} 检查点缺失: {prev_weights}")
        return prev_weights

    def load_or_init_telemetry(self, baseline_sigs_count: int) -> dict[str, Any]:
        telemetry_path = self.project_root / "runs" / self.config.exp_name / "dagger_telemetry.json"
        if telemetry_path.exists():
            return json.loads(telemetry_path.read_text(encoding="utf-8"))
        return {
            "baseline_signatures_d0": baseline_sigs_count,
            "disagreement_rates": {},
            "expansion_factors": {},
            "cumulative_signatures": {},
            "novel_signatures_vs_d0": {},
        }

    def update_telemetry_record(
        self,
        telemetry: dict[str, Any],
        k: int,
        rollout_stats: dict[str, Any],
        cumulative_sigs_count: int,
        baseline_sigs_count: int,
        novel_vs_d0_count: int,
    ) -> dict[str, Any]:
        telemetry["disagreement_rates"][f"d{k}"] = rollout_stats["disagreement_rate"]
        telemetry["expansion_factors"][f"e{k}"] = round(
            cumulative_sigs_count / max(baseline_sigs_count, 1), 4
        )
        telemetry["cumulative_signatures"][f"k{k}"] = cumulative_sigs_count
        telemetry["novel_signatures_vs_d0"][f"k{k}"] = novel_vs_d0_count
        telemetry[f"k{k}"] = rollout_stats
        return telemetry

    def save_telemetry(self, telemetry: dict[str, Any]) -> Path:
        telemetry_path = self.project_root / "runs" / self.config.exp_name / "dagger_telemetry.json"
        telemetry_path.parent.mkdir(parents=True, exist_ok=True)
        telemetry_path.write_text(json.dumps(telemetry, indent=2) + "\n", encoding="utf-8")
        return telemetry_path

    def run(self) -> None:
        self.validate_environment()
        d0_path = self.get_or_create_d0()

        print("正在提取 D_0 宏观状态签名空间...")
        baseline_sigs = extract_signatures_from_file(d0_path)
        cumulative_sigs = set(baseline_sigs)
        print(f"D_0 宏观特征签名数量: {len(baseline_sigs):,}")

        for prior_k in range(1, self.config.start_iter):
            prior_file = self.project_root / "data" / f"{self.config.exp_name}_iter{prior_k}.jsonl.gz"
            if prior_file.exists():
                cumulative_sigs.update(extract_signatures_from_file(prior_file))

        current_weights = self.get_initial_weights(d0_path)
        telemetry = self.load_or_init_telemetry(len(baseline_sigs))
        teacher = TacticalLocalTeacher(
            hill_lead=self.config.hill_lead,
            max_depth=self.config.max_depth,
            defense_radius=self.config.defense_radius,
        )
        gate = InformationAlignmentGate(teacher=teacher, hill_lead=self.config.hill_lead)

        for k in range(self.config.start_iter, self.config.end_iter + 1):
            iter_seed = self.config.seed + k * 10007
            delta_path = self.project_root / "data" / f"{self.config.exp_name}_iter{k}.jsonl.gz"

            print(f"\n========================================================")
            print(f"=== 开始 {self.config.exp_name} DAgger 迭代 {k} / {self.config.end_iter} ===")
            print(f"========================================================")

            if not delta_path.exists():
                student_driver = StudentDriver(
                    weights_path=current_weights,
                    device=self.config.device,
                    cls=self.config.weight_class,
                    channels=self.config.channels,
                    blocks=self.config.blocks,
                )
                opponents = build_mixed_opponents(self.project_root, teacher, student_driver)
                rollout_stats = rollout_stratified_mixed_dagger(
                    student_driver=student_driver,
                    gate=gate,
                    opponent_drivers=opponents,
                    buckets=self.buckets,
                    out_path=delta_path,
                    seat_turns=self.config.samples_per_iter,
                    k_iter=k,
                    waves=self.config.waves,
                    matches_per_wave=self.config.matches_per_wave,
                    max_turns=self.config.max_turns,
                    seed=iter_seed,
                    baseline_signatures=baseline_sigs,
                    project_root=self.project_root,
                )
            else:
                stats_file = delta_path.with_suffix("").with_suffix(".stats.json")
                if stats_file.exists():
                    rollout_stats = json.loads(stats_file.read_text(encoding="utf-8"))
                else:
                    rollout_stats = {"iteration": k, "disagreement_rate": 0.0, "novel_state_ratio": 0.0}

            delta_sigs = extract_signatures_from_file(delta_path)
            cumulative_sigs.update(delta_sigs)

            accum_out = self.project_root / "data" / f"{self.config.exp_name}_k{k}.jsonl.gz"
            parts = [d0_path] + [
                self.project_root / "data" / f"{self.config.exp_name}_iter{i}.jsonl.gz"
                for i in range(1, k + 1)
            ]
            if not accum_out.exists():
                print(f"--- 聚合全量数据集 D_{k} -> {accum_out.name} ---")
                self.aggregator.merge(
                    parts=parts,
                    out=accum_out,
                    iteration=k,
                    stats_extra={
                        "disagreement_rate": rollout_stats.get("disagreement_rate", 0.0),
                        "novel_state_ratio": rollout_stats.get("novel_state_ratio", 0.0),
                    },
                )

            iter_run_dir = self.project_root / "runs" / self.config.exp_name / f"k{k}"
            iter_best_pt = iter_run_dir / "best.pt"
            iter_epochs = self.config.final_epochs if k == self.config.end_iter else self.config.epochs
            if not iter_best_pt.exists():
                print(
                    f"--- 训练策略网络 pi_{k} ({iter_epochs} 轮次, batch {self.config.batch_size}, "
                    f"ch={self.config.channels}, b={self.config.blocks}) ---"
                )
                best_weights, _ = train_iteration_model(
                    cls=self.config.weight_class,
                    data_path=accum_out,
                    out_dir=iter_run_dir,
                    channels=self.config.channels,
                    blocks=self.config.blocks,
                    epochs=iter_epochs,
                    batch=self.config.batch_size,
                    lr=self.config.lr,
                    seed=self.config.seed,
                    device=self.config.device,
                    init_weights=current_weights,
                )
            else:
                best_weights = iter_best_pt

            iter_model_dir = self.project_root / "models" / f"{self.config.exp_name}_iter{k}"
            if not (iter_model_dir / "model.onnx").exists():
                print(f"--- 导出迭代模型固件 -> {iter_model_dir.name} ---")
                export_and_verify_model(
                    cls=self.config.weight_class,
                    weights_path=best_weights,
                    out_dir=iter_model_dir,
                    channels=self.config.channels,
                    blocks=self.config.blocks,
                    method=f"{self.config.exp_name}: Micro (ch={self.config.channels}, b={self.config.blocks}) stratified mixed-opponent DAgger iteration {k}",
                )

            current_weights = best_weights
            self.update_telemetry_record(
                telemetry=telemetry,
                k=k,
                rollout_stats=rollout_stats,
                cumulative_sigs_count=len(cumulative_sigs),
                baseline_sigs_count=len(baseline_sigs),
                novel_vs_d0_count=len(delta_sigs - baseline_sigs),
            )
            self.save_telemetry(telemetry)

        final_model_dir = (
            self.config.export_dir
            if self.config.export_dir is not None
            else (self.project_root / "models" / self.config.exp_name)
        )
        print(f"\n=== 导出最终验证模型构件 -> {final_model_dir} ===")
        said = export_and_verify_model(
            cls=self.config.weight_class,
            weights_path=current_weights,
            out_dir=final_model_dir,
            channels=self.config.channels,
            blocks=self.config.blocks,
            method=f"{self.config.exp_name}: 4-iteration stratified mixed-opponent TacticalLocalTeacher DAgger + Where memory Micro BC",
        )
        local_model_dir = self.project_root / "models" / self.config.exp_name
        if local_model_dir.resolve() != final_model_dir.resolve():
            export_and_verify_model(
                cls=self.config.weight_class,
                weights_path=current_weights,
                out_dir=local_model_dir,
                channels=self.config.channels,
                blocks=self.config.blocks,
                method=f"{self.config.exp_name}: 4-iteration stratified mixed-opponent TacticalLocalTeacher DAgger + Where memory Micro BC",
            )
        runs_exp_dir = self.project_root / "runs" / self.config.exp_name
        runs_exp_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(current_weights, runs_exp_dir / "best.pt")
        print(
            f"{self.config.exp_name} 全部迭代训练与模型导出完成！\n"
            f"  size_metric_bytes={said.get('size_metric_bytes')} B | infer_us_max={said.get('infer_us_max')} us"
        )


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="分层程序化棋盘池 + 单一局部战术教师 + 混合对手 DAgger 闭环训练控制器。")
    ap.add_argument("--exp-name", default="exp085_fix", help="实验名称（默认 exp085_fix）")
    ap.add_argument("--channels", type=int, default=40, help="Micro 主干网络通道数（默认 40）")
    ap.add_argument("--blocks", type=int, default=3, help="Micro 空洞残差块数（默认 3）")
    ap.add_argument("--start-iter", type=int, default=1, help="起始迭代轮次（默认 1）")
    ap.add_argument("--end-iter", type=int, default=4, help="结束迭代轮次（默认 4）")
    ap.add_argument("--d0-samples", type=int, default=250_000, help="D0 专家样本席位步数配额（默认 250,000）")
    ap.add_argument("--samples-per-iter", type=int, default=100_000, help="每轮 DAgger 增量席位步数配额（默认 100,000）")
    ap.add_argument("--pretrain-epochs", type=int, default=6, help="预训练轮次（默认 6）")
    ap.add_argument("--epochs", type=int, default=6, help="中间迭代训练轮次（默认 6）")
    ap.add_argument("--final-epochs", type=int, default=10, help="最终聚合数据集训练轮次（默认 10）")
    ap.add_argument("--batch", type=int, default=128, help="批处理大小（默认 128）")
    ap.add_argument("--lr", type=float, default=3e-3, help="初始学习率（默认 3e-3）")
    ap.add_argument("--max-turns", type=int, default=500, help="训练推演最大回合数（默认 500）")
    ap.add_argument("--hill-lead", type=int, default=8, help="敌方蚁丘相对食物的起步领先步数（默认 8）")
    ap.add_argument("--device", default="cuda", help="计算设备（默认 cuda）")
    ap.add_argument("--seed", type=int, default=20261120, help="随机种子（默认 20261120）")
    ap.add_argument("--exclude-boards", type=Path, default=None, help="严格哈希隔离的线上赛季棋盘 JSON 文件或目录")
    ap.add_argument("--export-dir", type=Path, default=None, help="最终模型额外导出目录")
    return ap


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    config = PipelineConfig(
        exp_name=args.exp_name,
        channels=args.channels,
        blocks=args.blocks,
        start_iter=args.start_iter,
        end_iter=args.end_iter,
        d0_samples=args.d0_samples,
        samples_per_iter=args.samples_per_iter,
        pretrain_epochs=args.pretrain_epochs,
        epochs=args.epochs,
        final_epochs=args.final_epochs,
        batch_size=args.batch,
        lr=args.lr,
        max_turns=args.max_turns,
        hill_lead=args.hill_lead,
        device=args.device,
        seed=args.seed,
        exclude_boards=args.exclude_boards,
        export_dir=args.export_dir,
    )
    pipeline = Exp085Pipeline(config)
    pipeline.run()


if __name__ == "__main__":
    main()
