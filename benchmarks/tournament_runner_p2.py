#!/usr/bin/env python3
"""Phase 2 Parallel Round-Robin Tournament Runner for TinyBrains Ants IDEA-002 (Milestone 4).

Executes 504 reciprocal seat-swap matches across 9 benchmark maps in parallel using
ProcessPoolExecutor and tinybrains match batches.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path("/home/host/project/ants/baselines")
CLI = os.environ.get("TINYBRAINS") or shutil.which("tinybrains") or "/home/linuxbrew/.linuxbrew/bin/tinybrains"
REGISTRY = os.environ.get("TINYBRAINS_REGISTRY") or "/home/host/project/ants/baselines/games.toml"


@dataclass
class Entrant:
    name: str
    weights: Path
    manifest: Path

    def seat(self, index: int) -> dict:
        return {
            "seat": index,
            "label": self.name,
            "weights": str(self.weights.resolve()),
            "manifest": str(self.manifest.resolve()),
        }


@dataclass
class Record:
    matches: int = 0
    wins: int = 0
    draws: int = 0
    losses: int = 0
    score_for: int = 0
    score_against: int = 0
    reasons: dict[str, int] = field(default_factory=dict)

    @property
    def points(self) -> float:
        return self.wins + 0.5 * self.draws

    @property
    def win_rate(self) -> float:
        return self.points / self.matches if self.matches else 0.0


def entrant_from(path: str) -> Entrant:
    p = Path(path).resolve()
    if p.is_dir():
        w = p / "model.onnx"
        m = p / "manifest.json"
        if not w.exists() or not m.exists():
            raise SystemExit(f"Entrant directory {path} missing model.onnx or manifest.json")
        return Entrant(name=p.name, weights=w, manifest=m)
    raise SystemExit(f"{path} is not a valid exported model directory")


def resolve_boards(maps_arg: str | None, count: int) -> list[str]:
    """Resolve up to count 2-player board paths with absolute paths."""
    if maps_arg and Path(maps_arg).is_dir():
        picked = []
        for f in sorted(Path(maps_arg).glob("*.json")):
            try:
                data = json.loads(f.read_text())
                if data.get("players") == 2:
                    picked.append(str(f.resolve()))
            except Exception:
                pass
        if not picked:
            raise SystemExit(f"No two-seat boards found in directory: {maps_arg}")
        return picked[:count]

    if maps_arg:
        return [str(Path(m.strip()).resolve()) for m in maps_arg.split(",") if m.strip()][:count]

    bench_maps = Path("/home/host/project/ants/benchmarks/maps_2p")
    if bench_maps.is_dir():
        picked = []
        for f in sorted(bench_maps.glob("*.json")):
            try:
                data = json.loads(f.read_text())
                if data.get("players") == 2:
                    picked.append(str(f.resolve()))
            except Exception:
                pass
        if picked:
            return picked[:count]

    raise SystemExit("Could not resolve any 2-player maps.")


def generate_rows(entrants: list[Entrant], boards: list[str], max_turns: int, seed: int) -> tuple[list[dict], dict[str, tuple[str, str]]]:
    pairs = list(itertools.combinations(range(len(entrants)), 2))
    rows: list[dict] = []
    labels: dict[str, tuple[str, str]] = {}

    for b_idx, board in enumerate(boards):
        board_name = Path(board).stem if board.endswith(".json") else board
        for a, b in pairs:
            for flip in (0, 1):
                first, second = (a, b) if not flip else (b, a)
                mid = f"{entrants[first].name}-vs-{entrants[second].name}-{board_name}-{b_idx}-{flip}"
                rows.append({
                    "id": mid,
                    "seed": seed + len(rows),
                    "map": board,
                    "seat_count": 2,
                    "seats": [entrants[first].seat(0), entrants[second].seat(1)],
                })
                labels[mid] = (entrants[first].name, entrants[second].name)
    return rows, labels


def _execute_worker_batch(args: tuple[int, list[dict], Path, int]) -> tuple[int, float, list[str]]:
    worker_id, chunk_rows, out_dir, max_turns = args
    batch_file = out_dir / f"_batch_{worker_id}.json"
    batch_file.write_text(json.dumps({
        "game": "ants",
        "vars": {"max_turns": max_turns},
        "rows": chunk_rows,
    }))

    env = os.environ.copy()
    env["TINYBRAINS_REGISTRY"] = REGISTRY

    t0 = time.perf_counter()
    res = subprocess.run(
        [CLI, str(batch_file), "--out", str(out_dir)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        env=env,
    )
    duration = time.perf_counter() - t0

    if res.returncode != 0:
        raise RuntimeError(f"Worker {worker_id} failed:\n{res.stderr or res.stdout}")

    try:
        batch_file.unlink(missing_ok=True)
    except Exception:
        pass

    return len(chunk_rows), duration, [r["id"] for r in chunk_rows]


def run_tournament(
    entrants: list[Entrant],
    boards: list[str],
    out_dir: Path,
    max_turns: int = 300,
    seed: int = 90000,
    workers: int = 16,
) -> tuple[dict[str, Record], float, list[dict]]:
    rows, labels = generate_rows(entrants, boards, max_turns, seed)
    total_matches = len(rows)

    out_dir.mkdir(parents=True, exist_ok=True)
    num_workers = min(workers, total_matches)
    chunk_size = (total_matches + num_workers - 1) // num_workers
    chunks = [rows[i:i + chunk_size] for i in range(0, total_matches, chunk_size)]
    tasks = [(i, chunk, out_dir, max_turns) for i, chunk in enumerate(chunks)]

    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        worker_results = list(executor.map(_execute_worker_batch, tasks))
    wall_clock = time.perf_counter() - t0

    records = {e.name: Record() for e in entrants}
    for row in rows:
        replay_path = out_dir / f"{row['id']}.json"
        if not replay_path.exists():
            raise FileNotFoundError(f"Missing expected replay: {replay_path}")
        replay = json.loads(replay_path.read_text())
        left, right = labels[row["id"]]
        ranks, scores = replay["engine_ranks"], replay["scores"]
        reason = replay.get("reason", "unknown")

        for who, mine, theirs in ((left, 0, 1), (right, 1, 0)):
            rec = records[who]
            rec.matches += 1
            rec.score_for += scores[mine]
            rec.score_against += scores[theirs]
            rec.reasons[reason] = rec.reasons.get(reason, 0) + 1
            if ranks[mine] < ranks[theirs]:
                rec.wins += 1
            elif ranks[mine] > ranks[theirs]:
                rec.losses += 1
            else:
                rec.draws += 1

    return records, wall_clock, rows


def format_table(records: dict[str, Record]) -> str:
    lines = [
        f"{'entrant':22s} {'played':>7s} {'W':>4s} {'D':>4s} {'L':>4s} {'rate':>6s} {'score':>7s} {'against':>8s}"
    ]
    for name, r in sorted(records.items(), key=lambda kv: (-kv[1].win_rate, -kv[1].score_for)):
        lines.append(
            f"{name:22s} {r.matches:>7d} {r.wins:>4d} {r.draws:>4d} {r.losses:>4d} "
            f"{r.win_rate:>5.0%} {r.score_for / max(r.matches, 1):>7.2f} "
            f"{r.score_against / max(r.matches, 1):>8.2f}"
        )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("entrants", nargs="+", help="Model directories to evaluate")
    parser.add_argument("--maps", default="benchmarks/maps_2p", help="Map directory or comma-separated maps")
    parser.add_argument("--boards", type=int, default=9, help="Number of boards to select (default: 9)")
    parser.add_argument("--max-turns", type=int, default=300, help="Max turns per match (default: 300)")
    parser.add_argument("--seed", type=int, default=90000, help="Base random seed (default: 90000)")
    parser.add_argument("--workers", type=int, default=16, help="Parallel worker processes (default: 16)")
    parser.add_argument("--out", type=Path, default=Path("replays/tournament_p2_9map"), help="Output replay directory")
    parser.add_argument("--json", dest="as_json", action="store_true", help="Output summary as JSON")
    args = parser.parse_args()

    entrants = [entrant_from(e) for e in args.entrants]
    boards = resolve_boards(args.maps, args.boards)

    print(f"Loaded {len(entrants)} entrants and {len(boards)} boards.")
    print(f"Starting tournament: {len(entrants)*(len(entrants)-1)*len(boards)} matches across {args.workers} workers...")

    records, wall_time, rows = run_tournament(
        entrants=entrants,
        boards=boards,
        out_dir=args.out.resolve(),
        max_turns=args.max_turns,
        seed=args.seed,
        workers=args.workers,
    )

    total_matches = len(rows)
    throughput = total_matches / wall_time if wall_time > 0 else 0

    if args.as_json:
        out_data = {
            "summary": {
                "total_matches": total_matches,
                "wall_time_seconds": round(wall_time, 2),
                "throughput_matches_per_sec": round(throughput, 2),
                "workers": args.workers,
            },
            "records": {n: vars(r) for n, r in records.items()},
        }
        print(json.dumps(out_data, indent=2))
    else:
        print(f"=== Ants Phase 2 Tournament Execution ({total_matches} matches across {len(boards)} boards, {args.workers} workers) ===")
        print(f"Total Wall Clock Time: {wall_time:.2f} s | Throughput: {throughput:.2f} matches/s\n")
        print(format_table(records))


if __name__ == "__main__":
    main()
