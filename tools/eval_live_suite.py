from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import subprocess
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CLI = os.environ.get("TINYBRAINS") or shutil.which("tinybrains") or "/home/linuxbrew/.linuxbrew/bin/tinybrains"
DEFAULT_REGISTRY = os.environ.get("TINYBRAINS_REGISTRY") or str((ROOT / "baselines" / "games.toml").resolve())
EXTRACTOR = ROOT / "benchmarks" / "extract_frames.mjs"

# TinyBrains 线上 hive-minds 赛季真实席位权重分布（无 3p 与 8p）
LIVE_SEAT_WEIGHTS: dict[int, float] = {2: 0.522, 4: 0.370, 6: 0.108}
DEFAULT_OPPONENTS: list[str] = ["nano-bc-max-r", "micro-bc", "nano-bc", "exp021", "exp011", "exp083"]
ANT_CHECKPOINTS: list[int] = [25, 50, 100, 150, 200, 300, 500, 750, 1000]


@dataclass(frozen=True)
class ModelSpec:
    name: str
    weights: Path
    manifest: Path

    def seat(self, seat_index: int) -> dict[str, Any]:
        return {
            "seat": seat_index,
            "label": self.name,
            "weights": str(self.weights.resolve()),
            "manifest": str(self.manifest.resolve()),
        }


@dataclass(frozen=True)
class BoardMeta:
    map_id: str
    path: Path
    players: int
    rows: int
    cols: int
    hills_per_seat: int
    terrain: str
    shape: str

    @property
    def size_label(self) -> str:
        return f"{self.rows}x{self.cols}"


@dataclass(frozen=True)
class GateConfig:
    turn_ms: float = 500.0
    max_slice_drop_pp: float = 0.0
    max_own_hill_hold: float = 0.05
    max_hold: float = 0.08


def infer_terrain_family(map_id: str, explicit_terrain: str | None = None) -> str:
    raw = (explicit_terrain or "").strip().lower()
    if not raw:
        parts = map_id.lower().split("-")
        raw = parts[1] if len(parts) >= 4 else (parts[0] if parts else "unknown")
    for fam in ("maze", "cave", "rooms", "open", "islands", "fortress", "boulders"):
        if raw.startswith(fam):
            return fam
    return raw or "unknown"


def load_model_spec(model_dir: Path, name: str | None = None) -> ModelSpec:
    resolved = model_dir.resolve()
    weights = resolved / "model.onnx"
    manifest = resolved / "manifest.json"
    if not weights.is_file():
        raise FileNotFoundError(f"模型权重缺失: {weights}")
    if not manifest.is_file():
        raise FileNotFoundError(f"模型清单缺失: {manifest}")
    return ModelSpec(name=name or resolved.name, weights=weights, manifest=manifest)


def resolve_opponents(
    opponent_names: list[str],
    models_root: Path,
    starter_root: Path | None,
    explicit_specs: list[str] | None = None,
    exclude_names: set[str] | None = None,
) -> dict[str, ModelSpec]:
    excluded = exclude_names or set()
    resolved: dict[str, ModelSpec] = {}

    if explicit_specs:
        for item in explicit_specs:
            if "=" in item:
                alias, raw_path = item.split("=", 1)
                spec = load_model_spec(Path(raw_path), name=alias.strip())
            else:
                spec = load_model_spec(Path(item))
            if spec.name not in excluded:
                resolved[spec.name] = spec
        return resolved

    search_roots = [models_root.resolve()]
    if starter_root is not None and starter_root.exists():
        search_roots.append(starter_root.resolve())

    for name in opponent_names:
        if name in excluded:
            continue
        found: Path | None = None
        for root in search_roots:
            cand = root / name
            if (cand / "model.onnx").is_file() and (cand / "manifest.json").is_file():
                found = cand
                break
        if found is None:
            raise FileNotFoundError(
                f"未在 {[str(r) for r in search_roots]} 找到对手模型 '{name}'"
            )
        resolved[name] = load_model_spec(found, name=name)
    return resolved


def load_board_catalog(
    maps_source: Path,
    cache_dir: Path,
    max_maps_per_seat: int | None = None,
) -> list[BoardMeta]:
    src = maps_source.resolve()
    raw_entries: list[tuple[str, Path, dict[str, Any], str | None]] = []

    if src.is_file():
        bundle = json.loads(src.read_text(encoding="utf-8"))
        if "maps" not in bundle or not isinstance(bundle["maps"], list):
            raise ValueError(f"不支持的棋盘包格式 (缺少 maps 列表): {src}")
        cache_dir.mkdir(parents=True, exist_ok=True)
        for item in bundle["maps"]:
            board = dict(item["board"])
            mid = str(item.get("map_id") or board.get("id") or "")
            if not mid:
                raise ValueError(f"棋盘包条目缺少 map_id: {item.keys()}")
            board["id"] = mid
            out_file = cache_dir / f"{mid}.json"
            out_file.write_text(json.dumps(board, separators=(",", ":")), encoding="utf-8")
            raw_entries.append((mid, out_file, board, item.get("terrain")))
    elif src.is_dir():
        for p in sorted(src.glob("*.json")):
            if p.name.startswith("_"):
                continue
            board = json.loads(p.read_text(encoding="utf-8"))
            if "players" not in board or "hills" not in board:
                continue
            mid = str(board.get("id") or p.stem)
            raw_entries.append((mid, p, board, None))
    else:
        raise FileNotFoundError(f"未找到棋盘目录或棋盘包文件: {src}")

    if not raw_entries:
        raise RuntimeError(f"未加载到任何有效棋盘: {src}")

    metas: list[BoardMeta] = []
    for mid, path, board, explicit_terrain in raw_entries:
        players = int(board["players"])
        rows = int(board["rows"])
        cols = int(board["cols"])
        hills = board["hills"]
        hills_per_seat = max(1, len(hills) // max(players, 1))
        metas.append(
            BoardMeta(
                map_id=mid,
                path=path.resolve(),
                players=players,
                rows=rows,
                cols=cols,
                hills_per_seat=hills_per_seat,
                terrain=infer_terrain_family(mid, explicit_terrain),
                shape="square" if rows == cols else "rect",
            )
        )

    metas.sort(key=lambda m: (m.players, m.hills_per_seat, m.rows * m.cols, m.map_id))

    if max_maps_per_seat is not None and max_maps_per_seat > 0:
        # 按席位分桶均匀抽样，尽量同时保留不同蚁穴数 (1h/2h/3h) 以保证冒烟验证覆盖度
        by_sc: dict[int, list[BoardMeta]] = defaultdict(list)
        for m in metas:
            by_sc[m.players].append(m)
        picked: list[BoardMeta] = []
        for sc in sorted(by_sc.keys()):
            group = by_sc[sc]
            by_hills: dict[int, list[BoardMeta]] = defaultdict(list)
            for m in group:
                by_hills[m.hills_per_seat].append(m)
            selected: list[BoardMeta] = []
            hill_keys = sorted(by_hills.keys())
            if hill_keys:
                offset = sc % len(hill_keys)
                hill_keys = hill_keys[offset:] + hill_keys[:offset]
            cursor = {h: 0 for h in hill_keys}
            while len(selected) < min(max_maps_per_seat, len(group)):
                progressed = False
                for h in hill_keys:
                    if cursor[h] < len(by_hills[h]) and len(selected) < max_maps_per_seat:
                        selected.append(by_hills[h][cursor[h]])
                        cursor[h] += 1
                        progressed = True
                if not progressed:
                    break
            picked.extend(selected)
        metas = picked

    return metas


def build_paired_live_schedule(
    boards: list[BoardMeta],
    evaluated_models: list[ModelSpec],
    opponents: dict[str, ModelSpec],
    seed: int = 20261120,
    trials_2p_per_opp: int = 1,
    trials_multiseat: int = 2,
    include_h2h: bool = False,
    baseline_name: str | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    # 构造严格配对的非内战对局表：每个 trial 固定 (棋盘, 种子, 席位, 非己方对手阵容)，仅替换焦点模型
    if not opponents:
        raise ValueError("非内战配对评测至少需要 1 个外部对手模型")
    opp_names = sorted(opponents.keys())
    rng = random.Random(seed)

    rows: list[dict[str, Any]] = []
    trial_meta: list[dict[str, Any]] = []
    h2h_meta: list[dict[str, Any]] = []

    boards_2p = [b for b in boards if b.players == 2]
    boards_ms = [b for b in boards if b.players > 2]

    for idx_m, b in enumerate(boards_2p):
        for idx_o, opp_name in enumerate(opp_names):
            opp_spec = opponents[opp_name]
            for r in range(trials_2p_per_opp):
                flip = (idx_m + idx_o + r) % 2
                match_seed = seed + 10000 + idx_m * 100 + idx_o * 10 + r
                trial_key = f"2p:{b.map_id}:{opp_name}:r{r}"
                for model in evaluated_models:
                    mid = f"live-2p-{b.map_id}-{opp_name}-r{r}-{model.name}"
                    first, second = (model, opp_spec) if flip == 0 else (opp_spec, model)
                    rows.append({
                        "id": mid,
                        "seed": match_seed,
                        "map": str(b.path),
                        "seat_count": 2,
                        "seats": [first.seat(0), second.seat(1)],
                    })
                    trial_meta.append({
                        "match_id": mid,
                        "trial_key": trial_key,
                        "map_id": b.map_id,
                        "players": 2,
                        "focal_model": model.name,
                        "focal_seat": flip,
                        "opp_2p": opp_name,
                    })

        if include_h2h and baseline_name is not None:
            base_spec = next(m for m in evaluated_models if m.name == baseline_name)
            for cand_spec in evaluated_models:
                if cand_spec.name == baseline_name:
                    continue
                for flip in (0, 1):
                    first, second = (cand_spec, base_spec) if flip == 0 else (base_spec, cand_spec)
                    mid = f"h2h-2p-{b.map_id}-{cand_spec.name}-vs-{baseline_name}-f{flip}"
                    rows.append({
                        "id": mid,
                        "seed": seed + 50000 + idx_m * 10 + flip,
                        "map": str(b.path),
                        "seat_count": 2,
                        "seats": [first.seat(0), second.seat(1)],
                    })
                    h2h_meta.append({
                        "match_id": mid,
                        "map_id": b.map_id,
                        "candidate": cand_spec.name,
                        "baseline": baseline_name,
                        "candidate_seat": flip,
                    })

    for idx_m, b in enumerate(boards_ms):
        sc = b.players
        for t in range(trials_multiseat):
            match_seed = seed + 100000 + sc * 10000 + idx_m * 100 + t
            focal_seat = (idx_m + t) % sc
            bg_names = [rng.choice(opp_names) for _ in range(sc - 1)]
            trial_key = f"{sc}p:{b.map_id}:t{t}"
            for model in evaluated_models:
                lineup_specs = [opponents[nm] for nm in bg_names]
                lineup_specs.insert(focal_seat, model)
                mid = f"live-{sc}p-{b.map_id}-t{t}-{model.name}"
                rows.append({
                    "id": mid,
                    "seed": match_seed,
                    "map": str(b.path),
                    "seat_count": sc,
                    "seats": [sp.seat(i) for i, sp in enumerate(lineup_specs)],
                })
                trial_meta.append({
                    "match_id": mid,
                    "trial_key": trial_key,
                    "map_id": b.map_id,
                    "players": sc,
                    "focal_model": model.name,
                    "focal_seat": focal_seat,
                    "opp_2p": None,
                })

    return rows, trial_meta, h2h_meta


def _execute_batch(args: tuple[str, list[dict[str, Any]], Path, int, str]) -> int:
    tag, chunk_rows, replays_dir, max_turns, registry = args
    batch_file = replays_dir / f"_batch_{tag}.json"
    batch_file.write_text(
        json.dumps({"game": "ants", "vars": {"max_turns": max_turns}, "rows": chunk_rows}),
        encoding="utf-8",
    )
    env = os.environ.copy()
    env["TINYBRAINS_REGISTRY"] = registry
    res = subprocess.run(
        [CLI, str(batch_file), "--out", str(replays_dir)],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        env=env,
    )
    batch_file.unlink(missing_ok=True)
    if res.returncode != 0:
        raise RuntimeError(f"批次 {tag} 执行失败:\n{res.stderr or res.stdout}")
    return len(chunk_rows)


def execute_match_rows(
    rows: list[dict[str, Any]],
    replays_dir: Path,
    max_turns: int,
    workers: int,
    registry: str,
) -> float:
    if replays_dir.exists():
        shutil.rmtree(replays_dir)
    replays_dir.mkdir(parents=True, exist_ok=True)

    # 按席位数分桶并交错切分，避免大型 96x96 棋盘堆积在单一进程
    by_sc: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_sc[int(r["seat_count"])].append(r)

    tasks: list[tuple[str, list[dict[str, Any]], Path, int, str]] = []
    tid = 0
    for sc, sc_rows in sorted(by_sc.items()):
        n_buckets = min(max(1, workers), len(sc_rows))
        buckets: list[list[dict[str, Any]]] = [[] for _ in range(n_buckets)]
        for idx, row in enumerate(sc_rows):
            buckets[idx % n_buckets].append(row)
        for bucket in buckets:
            tasks.append((f"{sc}p_{tid}", bucket, replays_dir, max_turns, registry))
            tid += 1

    t0 = time.perf_counter()
    actual_workers = min(max(1, workers), len(tasks), max(os.cpu_count() or 1, 1))
    with ProcessPoolExecutor(max_workers=actual_workers) as pool:
        list(pool.map(_execute_batch, tasks))
    return time.perf_counter() - t0


def extract_frame_metrics(replays_dir: Path, dist_dir: Path | None = None) -> list[dict[str, Any]]:
    env = os.environ.copy()
    if dist_dir is not None:
        env["ANTS_DIST_DIR"] = str(dist_dir.resolve())
    res = subprocess.run(
        ["node", str(EXTRACTOR.resolve()), str(replays_dir.resolve())],
        cwd=str(ROOT / "benchmarks"),
        capture_output=True,
        text=True,
        env=env,
    )
    if res.returncode != 0:
        raise RuntimeError(f"extract_frames.mjs 解码失败:\n{res.stderr or res.stdout}")
    return json.loads(res.stdout)


def aggregate_seat_metrics(items: list[dict[str, Any]], turn_ms: float = 500.0) -> dict[str, Any]:
    n = len(items)
    if n == 0:
        return {}

    total_orders = sum(int(x.get("total_orders", 0)) for x in items)
    blocked_orders = sum(int(x.get("blocked_orders", 0)) for x in items)
    all_orders = sum(int(x.get("all_orders", 0)) for x in items)
    hold_orders = sum(int(x.get("hold_orders", 0)) for x in items)
    own_hill_steps = sum(int(x.get("on_own_hill_steps", 0)) for x in items)
    own_hill_holds = sum(int(x.get("on_own_hill_hold_orders", 0)) for x in items)
    idle3_steps = sum(int(x.get("idle3_steps", 0)) for x in items)
    idle3_elig = sum(int(x.get("idle3_eligible", 0)) for x in items)
    osc2_steps = sum(int(x.get("osc2_steps", 0)) for x in items)
    osc2_elig = sum(int(x.get("osc2_eligible", 0)) for x in items)
    combat_deaths = sum(int(x.get("combat_deaths", 0)) for x in items)
    combat_kills = sum(float(x.get("combat_kills", 0.0)) for x in items)
    infer_max_ms_list = [float(x.get("infer_us_max", 0)) / 1000.0 for x in items]

    ants_curve: dict[str, float | None] = {}
    for cp in ANT_CHECKPOINTS:
        vals = [
            float(x["ants_at_turns"][str(cp)])
            for x in items
            if isinstance(x.get("ants_at_turns"), dict) and str(cp) in x["ants_at_turns"]
        ]
        ants_curve[str(cp)] = round(sum(vals) / len(vals), 2) if vals else None

    return {
        "n": n,
        "rank_pct": round(100.0 * sum(float(x["rank_pct"]) for x in items) / n, 2),
        "mean_rank": round(sum(float(x["rank"]) for x in items) / n, 3),
        "first_place_rate": round(100.0 * sum(1 for x in items if int(x["rank"]) == 1) / n, 1),
        "zero_score_rate": round(100.0 * sum(1 for x in items if int(x["score"]) == 0) / n, 1),
        "mean_score": round(sum(float(x["score"]) for x in items) / n, 3),
        "mean_ants_spawned": round(sum(float(x.get("ants_spawned", 0)) for x in items) / n, 1),
        "mean_peak_ants": round(sum(float(x.get("peak_ants", 0)) for x in items) / n, 1),
        "mean_final_ants": round(sum(float(x.get("final_ants", 0)) for x in items) / n, 1),
        "ants_curve": ants_curve,
        "hills_razed": sum(int(x.get("hills_razed", 0)) for x in items),
        "hills_lost": sum(int(x.get("hills_lost", 0)) for x in items),
        "combat_deaths_mean": round(combat_deaths / n, 1),
        "combat_kills_mean": round(combat_kills / n, 1),
        "kd_ratio": round(combat_kills / max(combat_deaths, 1), 3),
        "collisions_mean": round(sum(float(x.get("collisions", 0)) for x in items) / n, 1),
        "blocked_rate": round(blocked_orders / max(total_orders, 1), 6),
        "hold_rate": round(hold_orders / max(all_orders, 1), 4),
        "on_own_hill_hold_rate": round(own_hill_holds / max(own_hill_steps, 1), 4) if own_hill_steps > 0 else 0.0,
        "own_hill_occupied_turns_mean": round(sum(float(x.get("own_hill_occupied_turns", 0)) for x in items) / n, 1),
        "idle3_rate": round(idle3_steps / max(idle3_elig, 1), 4),
        "osc2_rate": round(osc2_steps / max(osc2_elig, 1), 4),
        "mean_infer_ms": round(sum(infer_max_ms_list) / n, 2),
        "max_infer_ms": round(max(infer_max_ms_list), 2),
        "over_turn_ms_seats": sum(1 for ms in infer_max_ms_list if ms > turn_ms),
        "strikes": sum(int(x.get("strikes", 0)) for x in items),
        "forfeits": sum(1 for x in items if x.get("forfeit")),
    }


def compute_weighted_live_rank_pct(
    by_seat_metrics: dict[str, dict[str, Any]],
    seat_weights: dict[int, float] | None = None,
) -> tuple[float, dict[str, float]]:
    weights_src = seat_weights or LIVE_SEAT_WEIGHTS
    active: dict[int, float] = {}
    for sc_key, stats in by_seat_metrics.items():
        sc = int(sc_key.rstrip("p"))
        if stats and sc in weights_src:
            active[sc] = weights_src[sc]
    if not active:
        # 若测试集仅包含非标准席位（如 3p/8p），回退为等权均值并显式记录权重
        present = [int(k.rstrip("p")) for k, v in by_seat_metrics.items() if v]
        if not present:
            return 0.0, {}
        eq = 1.0 / len(present)
        norm_weights = {f"{sc}p": round(eq, 4) for sc in sorted(present)}
        val = sum(by_seat_metrics[f"{sc}p"]["rank_pct"] * eq for sc in present)
        return round(val, 2), norm_weights

    w_sum = sum(active.values())
    norm = {sc: w / w_sum for sc, w in active.items()}
    weighted = sum(by_seat_metrics[f"{sc}p"]["rank_pct"] * w for sc, w in norm.items())
    return round(weighted, 2), {f"{sc}p": round(w, 4) for sc, w in sorted(norm.items())}


def compute_paired_comparison(
    cand_by_trial: dict[str, dict[str, Any]],
    base_by_trial: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    shared = sorted(set(cand_by_trial.keys()) & set(base_by_trial.keys()))
    if not shared:
        return {"paired_trials": 0, "candidate_better": 0, "tie": 0, "baseline_better": 0, "mean_delta_pp": 0.0, "ci95_pp": [0.0, 0.0]}

    deltas: list[float] = []
    better = tie = worse = 0
    for tk in shared:
        c_rp = float(cand_by_trial[tk]["rank_pct"]) * 100.0
        b_rp = float(base_by_trial[tk]["rank_pct"]) * 100.0
        diff = c_rp - b_rp
        deltas.append(diff)
        if diff > 1e-9:
            better += 1
        elif diff < -1e-9:
            worse += 1
        else:
            tie += 1

    mean_d = sum(deltas) / len(deltas)
    if len(deltas) >= 2:
        var = sum((x - mean_d) ** 2 for x in deltas) / (len(deltas) - 1)
        se = math.sqrt(var / len(deltas))
    else:
        se = 0.0
    return {
        "paired_trials": len(shared),
        "candidate_better": better,
        "tie": tie,
        "baseline_better": worse,
        "mean_delta_pp": round(mean_d, 2),
        "ci95_pp": [round(mean_d - 1.96 * se, 2), round(mean_d + 1.96 * se, 2)],
    }


def evaluate_acceptance_gates(
    cand_report: dict[str, Any],
    base_report: dict[str, Any],
    gate_cfg: GateConfig,
) -> dict[str, Any]:
    c_all = cand_report["overall"]
    c_weighted = float(cand_report["weighted_live_rank_pct"])
    b_weighted = float(base_report["weighted_live_rank_pct"])

    seat_checks: dict[str, dict[str, Any]] = {}
    no_seat_regression = True
    for sc_key, b_stats in base_report["by_seat"].items():
        if not b_stats or sc_key not in cand_report["by_seat"]:
            continue
        c_rp = float(cand_report["by_seat"][sc_key]["rank_pct"])
        b_rp = float(b_stats["rank_pct"])
        delta = round(c_rp - b_rp, 2)
        passed = delta >= -gate_cfg.max_slice_drop_pp
        seat_checks[sc_key] = {"candidate": c_rp, "baseline": b_rp, "delta_pp": delta, "pass": passed}
        if not passed:
            no_seat_regression = False

    hill_checks: dict[str, dict[str, Any]] = {}
    multi_hill_ok = True
    for h_key in ("2h", "3h"):
        if h_key not in base_report["by_hills"] or h_key not in cand_report["by_hills"]:
            continue
        c_rp = float(cand_report["by_hills"][h_key]["rank_pct"])
        b_rp = float(base_report["by_hills"][h_key]["rank_pct"])
        delta = round(c_rp - b_rp, 2)
        passed = delta >= -gate_cfg.max_slice_drop_pp
        hill_checks[h_key] = {"candidate": c_rp, "baseline": b_rp, "delta_pp": delta, "pass": passed}
        if not passed:
            multi_hill_ok = False

    gates = {
        "zero_forfeits_and_strikes": {
            "pass": (c_all["forfeits"] == 0 and c_all["strikes"] == 0),
            "actual": {"forfeits": c_all["forfeits"], "strikes": c_all["strikes"]},
            "threshold": {"forfeits": 0, "strikes": 0},
        },
        "latency_within_turn_ms": {
            "pass": c_all["max_infer_ms"] <= gate_cfg.turn_ms,
            "actual_ms": c_all["max_infer_ms"],
            "threshold_ms": gate_cfg.turn_ms,
        },
        "weighted_rank_beats_baseline": {
            "pass": c_weighted > b_weighted,
            "candidate": c_weighted,
            "baseline": b_weighted,
            "delta_pp": round(c_weighted - b_weighted, 2),
        },
        "no_seat_regression": {
            "pass": no_seat_regression,
            "max_drop_pp": gate_cfg.max_slice_drop_pp,
            "slices": seat_checks,
        },
        "multi_hill_no_collapse": {
            "pass": multi_hill_ok,
            "max_drop_pp": gate_cfg.max_slice_drop_pp,
            "slices": hill_checks,
        },
        "own_hill_hold_healthy": {
            "pass": c_all["on_own_hill_hold_rate"] <= gate_cfg.max_own_hill_hold,
            "actual": c_all["on_own_hill_hold_rate"],
            "threshold": gate_cfg.max_own_hill_hold,
        },
        "overall_hold_healthy": {
            "pass": c_all["hold_rate"] <= gate_cfg.max_hold,
            "actual": c_all["hold_rate"],
            "threshold": gate_cfg.max_hold,
        },
    }

    failed = [name for name, g in gates.items() if not g["pass"]]
    return {
        "verdict": "PASS" if not failed else "FAIL",
        "failed_gates": failed,
        "gates": gates,
    }


def aggregate_live_evaluation(
    frame_metrics: list[dict[str, Any]],
    boards: list[BoardMeta],
    trial_meta: list[dict[str, Any]],
    baseline_name: str,
    candidate_names: list[str],
    gate_cfg: GateConfig,
    h2h_meta: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    by_match_id = {r["match_id"]: r for r in frame_metrics}
    board_by_id = {b.map_id: b for b in boards}
    all_models = [baseline_name] + [c for c in candidate_names if c != baseline_name]

    per_model_all: dict[str, list[dict[str, Any]]] = defaultdict(list)
    per_model_by_trial: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    per_model_dim: dict[str, dict[str, dict[str, list[dict[str, Any]]]]] = {
        m: defaultdict(lambda: defaultdict(list)) for m in all_models
    }

    for mt in trial_meta:
        mid = mt["match_id"]
        if mid not in by_match_id:
            raise KeyError(f"缺少回放指标记录: {mid}")
        rec = by_match_id[mid]
        bm = board_by_id[mt["map_id"]]
        focal = mt["focal_model"]
        seat_idx = int(mt["focal_seat"])
        seat_rec = next(s for s in rec["seats"] if int(s["seat"]) == seat_idx)

        per_model_all[focal].append(seat_rec)
        per_model_by_trial[focal][mt["trial_key"]] = seat_rec
        per_model_dim[focal]["by_seat"][f"{bm.players}p"].append(seat_rec)
        per_model_dim[focal]["by_hills"][f"{bm.hills_per_seat}h"].append(seat_rec)
        per_model_dim[focal]["by_seat_hills"][f"{bm.players}p-{bm.hills_per_seat}h"].append(seat_rec)
        per_model_dim[focal]["by_size"][bm.size_label].append(seat_rec)
        per_model_dim[focal]["by_shape"][bm.shape].append(seat_rec)
        per_model_dim[focal]["by_terrain"][bm.terrain].append(seat_rec)
        if mt.get("opp_2p"):
            per_model_dim[focal]["by_opponent_2p"][str(mt["opp_2p"])].append(seat_rec)

    models_report: dict[str, Any] = {}
    used_weights: dict[str, float] = {}
    for m in all_models:
        by_seat = {
            k: aggregate_seat_metrics(v, turn_ms=gate_cfg.turn_ms)
            for k, v in sorted(per_model_dim[m]["by_seat"].items())
        }
        weighted_rp, w_map = compute_weighted_live_rank_pct(by_seat)
        if not used_weights:
            used_weights = w_map
        models_report[m] = {
            "weighted_live_rank_pct": weighted_rp,
            "overall": aggregate_seat_metrics(per_model_all[m], turn_ms=gate_cfg.turn_ms),
            "by_seat": by_seat,
            "by_hills": {
                k: aggregate_seat_metrics(v, turn_ms=gate_cfg.turn_ms)
                for k, v in sorted(per_model_dim[m]["by_hills"].items())
            },
            "by_seat_hills": {
                k: aggregate_seat_metrics(v, turn_ms=gate_cfg.turn_ms)
                for k, v in sorted(per_model_dim[m]["by_seat_hills"].items())
            },
            "by_size": {
                k: aggregate_seat_metrics(v, turn_ms=gate_cfg.turn_ms)
                for k, v in sorted(per_model_dim[m]["by_size"].items())
            },
            "by_shape": {
                k: aggregate_seat_metrics(v, turn_ms=gate_cfg.turn_ms)
                for k, v in sorted(per_model_dim[m]["by_shape"].items())
            },
            "by_terrain": {
                k: aggregate_seat_metrics(v, turn_ms=gate_cfg.turn_ms)
                for k, v in sorted(per_model_dim[m]["by_terrain"].items())
            },
            "by_opponent_2p": {
                k: aggregate_seat_metrics(v, turn_ms=gate_cfg.turn_ms)
                for k, v in sorted(per_model_dim[m]["by_opponent_2p"].items())
            },
        }

    comparisons: dict[str, Any] = {}
    for cand in candidate_names:
        if cand == baseline_name:
            continue
        comparisons[cand] = {
            "vs_baseline": baseline_name,
            "paired_overall": compute_paired_comparison(
                per_model_by_trial[cand], per_model_by_trial[baseline_name]
            ),
            "acceptance": evaluate_acceptance_gates(
                models_report[cand], models_report[baseline_name], gate_cfg
            ),
        }

    direct_h2h: dict[str, Any] = {}
    if h2h_meta:
        h2h_by_cand: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for hm in h2h_meta:
            rec = by_match_id[hm["match_id"]]
            c_seat = next(s for s in rec["seats"] if int(s["seat"]) == int(hm["candidate_seat"]))
            h2h_by_cand[hm["candidate"]].append(c_seat)
        for cand, items in sorted(h2h_by_cand.items()):
            direct_h2h[cand] = aggregate_seat_metrics(items, turn_ms=gate_cfg.turn_ms)

    return {
        "baseline": baseline_name,
        "candidates": candidate_names,
        "seat_weights": used_weights,
        "boards_evaluated": len(boards),
        "models": models_report,
        "comparisons": comparisons,
        "direct_h2h_2p": direct_h2h,
    }


def print_summary(report: dict[str, Any]) -> None:
    base = report["baseline"]
    print("\n================ 线上赛季分布 OOD 基准评测汇总 (非内战配对) ================")
    print(f"棋盘数: {report['boards_evaluated']} | 天梯席位权重: {report['seat_weights']} | 基准模型: {base}")
    print("-" * 100)
    print(
        f"{'模型':<14} {'加权名次%':<10} {'2p%':<8} {'4p%':<8} {'6p%':<8} "
        f"{'1h%':<8} {'2h%':<8} {'3h%':<8} {'HOLD%':<7} {'占穴HOLD%':<10} {'阻塞%':<7} {'峰值ms':<7}"
    )
    print("-" * 100)
    for m, mrep in report["models"].items():
        ov = mrep["overall"]
        s2 = mrep["by_seat"].get("2p", {}).get("rank_pct", float("nan"))
        s4 = mrep["by_seat"].get("4p", {}).get("rank_pct", float("nan"))
        s6 = mrep["by_seat"].get("6p", {}).get("rank_pct", float("nan"))
        h1 = mrep["by_hills"].get("1h", {}).get("rank_pct", float("nan"))
        h2 = mrep["by_hills"].get("2h", {}).get("rank_pct", float("nan"))
        h3 = mrep["by_hills"].get("3h", {}).get("rank_pct", float("nan"))
        fmt = lambda v: f"{v:5.1f}" if not math.isnan(v) else "  -  "
        print(
            f"{m:<14} {mrep['weighted_live_rank_pct']:<10.2f} {fmt(s2):<8} {fmt(s4):<8} {fmt(s6):<8} "
            f"{fmt(h1):<8} {fmt(h2):<8} {fmt(h3):<8} "
            f"{ov['hold_rate'] * 100.0:<7.2f} {ov['on_own_hill_hold_rate'] * 100.0:<10.2f} "
            f"{ov['blocked_rate'] * 100.0:<7.2f} {ov['max_infer_ms']:<7.1f}"
        )

    print("\n================ 候选模型对比基线裁决 (PASS / FAIL) ================")
    for cand, comp in report["comparisons"].items():
        pr = comp["paired_overall"]
        acc = comp["acceptance"]
        print(
            f"  {cand} vs {base}: {acc['verdict']} | "
            f"配对胜平负={pr['candidate_better']}-{pr['tie']}-{pr['baseline_better']} "
            f"(均差={pr['mean_delta_pp']:+.2f} pp, 95%CI={pr['ci95_pp']}) | "
            f"未通过门禁={acc['failed_gates']}"
        )
    print("=" * 100)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="TinyBrains Ants 线上赛季分布保真 OOD 评测套件。")
    ap.add_argument("--candidates", type=Path, nargs="+", required=True, help="待测候选模型目录（一个或多个）")
    ap.add_argument("--baseline", type=Path, default=None, help="对照基准模型目录（默认 <models-root>/exp033）")
    ap.add_argument("--maps", type=Path, required=True, help="留出测试棋盘目录或 season_maps_boards.json 棋盘包")
    ap.add_argument("--models-root", type=Path, default=ROOT / "models", help="本地模型根目录")
    ap.add_argument(
        "--starter-root",
        type=Path,
        default=Path("/home/host/project/ants-starter/models"),
        help="官方 starter 基线模型目录",
    )
    ap.add_argument("--opponents", nargs="+", default=None, help="显式指定对手模型路径或 alias=path 列表")
    ap.add_argument("--max-turns", type=int, default=1000, help="单局最大回合数（默认 1000，与线上一致）")
    ap.add_argument("--turn-ms", type=float, default=500.0, help="单回合推理时限门禁毫秒数（默认 500ms，与线上一致）")
    ap.add_argument("--trials-2p-per-opp", type=int, default=1, help="2p 每图每对手配对轮次（默认 1）")
    ap.add_argument("--trials-multiseat", type=int, default=2, help="4p/6p 每图配对试验次数（默认 2）")
    ap.add_argument("--max-maps-per-seat", type=int, default=None, help="可选按席位限制评测棋盘数量（用于快速冒烟测试）")
    ap.add_argument("--include-h2h", action="store_true", help="附加运行 2p 候选与基线直接双向 H2H（独立报告）")
    ap.add_argument("--max-slice-drop-pp", type=float, default=0.0, help="各席位与多蚁穴分片允许相对基线的最大回退百分点（默认 0.0）")
    ap.add_argument("--max-own-hill-hold", type=float, default=0.05, help="己方蚁穴驻留 HOLD 率上限门禁（默认 0.05）")
    ap.add_argument("--max-hold", type=float, default=0.08, help="全图总体 HOLD 率上限门禁（默认 0.08）")
    ap.add_argument("--seed", type=int, default=20261120, help="全局确定性随机种子")
    ap.add_argument("--workers", type=int, default=12, help="并行进程数（默认 12）")
    ap.add_argument("--registry", type=str, default=DEFAULT_REGISTRY, help="TinyBrains 注册表路径")
    ap.add_argument("--dist-dir", type=Path, default=None, help="可选 dist 目录（用于 worktree 复用）")
    ap.add_argument("--replays-dir", type=Path, default=ROOT / "runs" / "eval_live_suite" / "replays", help="回放缓存目录")
    ap.add_argument("--out", type=Path, default=ROOT / "runs" / "eval_live_suite" / "report.json", help="JSON 报告输出路径")
    return ap


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = build_parser().parse_args(argv)
    baseline_dir = args.baseline or (args.models_root / "exp033")
    base_spec = load_model_spec(baseline_dir)
    cand_specs = [load_model_spec(p) for p in args.candidates]

    evaluated: list[ModelSpec] = [base_spec]
    seen_names = {base_spec.name}
    for cs in cand_specs:
        if cs.name not in seen_names:
            evaluated.append(cs)
            seen_names.add(cs.name)

    opponents = resolve_opponents(
        opponent_names=DEFAULT_OPPONENTS,
        models_root=args.models_root,
        starter_root=args.starter_root,
        explicit_specs=args.opponents,
        exclude_names=seen_names,
    )

    board_cache = args.replays_dir.parent / "_board_cache"
    boards = load_board_catalog(args.maps, cache_dir=board_cache, max_maps_per_seat=args.max_maps_per_seat)

    rows, trial_meta, h2h_meta = build_paired_live_schedule(
        boards=boards,
        evaluated_models=evaluated,
        opponents=opponents,
        seed=args.seed,
        trials_2p_per_opp=args.trials_2p_per_opp,
        trials_multiseat=args.trials_multiseat,
        include_h2h=args.include_h2h,
        baseline_name=base_spec.name,
    )

    print(
        f"开始执行线上保真配对评测: {len(boards)} 张棋盘, {len(rows)} 场真实对局 "
        f"(max_turns={args.max_turns}, turn_ms={args.turn_ms}, 对手池={sorted(opponents.keys())})..."
    )
    elapsed = execute_match_rows(
        rows=rows,
        replays_dir=args.replays_dir,
        max_turns=args.max_turns,
        workers=args.workers,
        registry=args.registry,
    )
    print(f"全部 {len(rows)} 场对局执行完成，耗时 {elapsed:.1f}s，正在调用 Wasm 引擎解算行为曲线...")

    frame_metrics = extract_frame_metrics(args.replays_dir, dist_dir=args.dist_dir)
    gate_cfg = GateConfig(
        turn_ms=args.turn_ms,
        max_slice_drop_pp=args.max_slice_drop_pp,
        max_own_hill_hold=args.max_own_hill_hold,
        max_hold=args.max_hold,
    )
    report = aggregate_live_evaluation(
        frame_metrics=frame_metrics,
        boards=boards,
        trial_meta=trial_meta,
        baseline_name=base_spec.name,
        candidate_names=[c.name for c in cand_specs],
        gate_cfg=gate_cfg,
        h2h_meta=h2h_meta,
    )
    report["metadata"] = {
        "maps_source": str(args.maps.resolve()),
        "max_turns": args.max_turns,
        "turn_ms": args.turn_ms,
        "seed": args.seed,
        "total_matches": len(rows),
        "wall_clock_seconds": round(elapsed, 2),
        "opponents": sorted(opponents.keys()),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print_summary(report)
    print(f"完整评测报告已写入: {args.out}")
    return report


if __name__ == "__main__":
    main()
