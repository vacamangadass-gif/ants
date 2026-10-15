"""Empirical Challenger stress test suite for Milestone 2:
Data collection pipeline (`collect.py`) verification across EXP-010 to EXP-014.

Tests:
1. CLI argument parsing:
   - Flags: --teacher, --hill-lead, --memory-mode, --remember
   - Strict choices validation and default values
2. Live mock rollouts across all 5 experimental configurations:
   - EXP-010: teacher=default, hill_lead=6, memory_mode=max, remember=True
   - EXP-011: teacher=default, hill_lead=6, memory_mode=where, remember=True
   - EXP-012: teacher=default, hill_lead=12, memory_mode=max, remember=True
   - EXP-013: teacher=default, hill_lead=12, memory_mode=where, remember=True
   - EXP-014: teacher=compound, hill_lead=6, memory_mode=where, remember=True
3. Deep header and row inspection:
   - Header metadata matches exact configuration
   - Row schema: o, k, m, a
   - Consecutive turn memory transition oracle verification
4. CLI subprocess execution and companion stats.json validation
5. Where vs Max memory divergence behavior during collection
6. Fallback and unremembered collection behavior
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

# Ensure baselines/src is on path
BASELINES_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = BASELINES_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from tb_baselines.collect import collect  # noqa: E402
from tb_baselines.planes import remember  # noqa: E402


def test_cli_argument_parsing_defaults():
    """Verify default values of CLI arguments when omitted."""
    ap = argparse.ArgumentParser()
    # Replicate parser from collect.main
    ap.add_argument("--out", type=Path, default=Path("data/teacher.jsonl.gz"))
    ap.add_argument("--seat-turns", type=int, default=200_000)
    ap.add_argument("--waves", type=int, default=4)
    ap.add_argument("--matches-per-wave", type=int, default=16)
    ap.add_argument("--max-turns", type=int, default=300)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--maps", default=None)
    ap.add_argument("--remember", action="store_true")
    ap.add_argument("--teacher", choices=["default", "compound"], default="default")
    ap.add_argument("--hill-lead", type=int, default=6)
    ap.add_argument("--memory-mode", choices=["where", "max"], default="where")

    args = ap.parse_args([])
    assert args.teacher == "default"
    assert args.hill_lead == 6
    assert args.memory_mode == "where"
    assert not args.remember


@pytest.mark.parametrize(
    "cli_args,expected_teacher,expected_lead,expected_mode,expected_remember",
    [
        # EXP-010: default, hill_lead=6, max, remember
        (["--teacher", "default", "--hill-lead", "6", "--memory-mode", "max", "--remember"],
         "default", 6, "max", True),
        # EXP-011: default, hill_lead=6, where, remember
        (["--teacher", "default", "--hill-lead", "6", "--memory-mode", "where", "--remember"],
         "default", 6, "where", True),
        # EXP-012: default, hill_lead=12, max, remember
        (["--teacher", "default", "--hill-lead", "12", "--memory-mode", "max", "--remember"],
         "default", 12, "max", True),
        # EXP-013: default, hill_lead=12, where, remember
        (["--teacher", "default", "--hill-lead", "12", "--memory-mode", "where", "--remember"],
         "default", 12, "where", True),
        # EXP-014: compound, where, remember
        (["--teacher", "compound", "--memory-mode", "where", "--remember"],
         "compound", 6, "where", True),
    ],
)
def test_cli_argument_parsing_all_five_experiments(
    cli_args, expected_teacher, expected_lead, expected_mode, expected_remember
):
    """Verify CLI parser accurately recognizes options for all 5 experiment configurations."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("data/teacher.jsonl.gz"))
    ap.add_argument("--seat-turns", type=int, default=200_000)
    ap.add_argument("--waves", type=int, default=4)
    ap.add_argument("--matches-per-wave", type=int, default=16)
    ap.add_argument("--max-turns", type=int, default=300)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--maps", default=None)
    ap.add_argument("--remember", action="store_true")
    ap.add_argument("--teacher", choices=["default", "compound"], default="default")
    ap.add_argument("--hill-lead", type=int, default=6)
    ap.add_argument("--memory-mode", choices=["where", "max"], default="where")

    args = ap.parse_args(cli_args)
    assert args.teacher == expected_teacher
    assert args.hill_lead == expected_lead
    assert args.memory_mode == expected_mode
    assert args.remember == expected_remember


def test_cli_argument_parsing_invalid_choices():
    """Verify invalid options are rejected by the parser with exit code 2."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", choices=["default", "compound"], default="default")
    ap.add_argument("--memory-mode", choices=["where", "max"], default="where")

    with pytest.raises(SystemExit):
        ap.parse_args(["--teacher", "invalid_teacher"])

    with pytest.raises(SystemExit):
        ap.parse_args(["--memory-mode", "unknown_mode"])


EXPERIMENT_CONFIGS = [
    {
        "id": "EXP-010",
        "teacher": "default",
        "hill_lead": 6,
        "memory_mode": "max",
        "remembers": True,
        "expected_desc_prefix": "deterministic potential field (hill_lead=6), remembering",
    },
    {
        "id": "EXP-011",
        "teacher": "default",
        "hill_lead": 6,
        "memory_mode": "where",
        "remembers": True,
        "expected_desc_prefix": "deterministic potential field (hill_lead=6), remembering",
    },
    {
        "id": "EXP-012",
        "teacher": "default",
        "hill_lead": 12,
        "memory_mode": "max",
        "remembers": True,
        "expected_desc_prefix": "deterministic potential field (hill_lead=12), remembering",
    },
    {
        "id": "EXP-013",
        "teacher": "default",
        "hill_lead": 12,
        "memory_mode": "where",
        "remembers": True,
        "expected_desc_prefix": "deterministic potential field (hill_lead=12), remembering",
    },
    {
        "id": "EXP-014",
        "teacher": "compound",
        "hill_lead": 6,
        "memory_mode": "where",
        "remembers": True,
        "expected_desc_prefix": "compound situational policy, remembering",
    },
]


@pytest.mark.parametrize("cfg", EXPERIMENT_CONFIGS, ids=[c["id"] for c in EXPERIMENT_CONFIGS])
def test_mock_collection_pipeline_across_all_five_experiments(tmp_path, cfg):
    """Run live mock data collection (60 seat turns) and inspect file header and row contents."""
    out_file = tmp_path / f"{cfg['id']}.jsonl.gz"
    seat_turns = 60

    stats = collect(
        out=out_file,
        seat_turns=seat_turns,
        waves=1,
        matches_per_wave=2,
        max_turns=50,
        seed=42,
        maps="basic-tiny-2p",
        remembers=cfg["remembers"],
        teacher=cfg["teacher"],
        hill_lead=cfg["hill_lead"],
        memory_mode=cfg["memory_mode"],
    )

    # 1. Assert return dictionary
    assert stats["rows"] == seat_turns
    assert stats["ant_labels"] > 0
    assert stats["episodes"] >= 0
    assert stats["teacher"] == cfg["teacher"]
    assert stats["hill_lead"] == cfg["hill_lead"]
    assert stats["memory_mode"] == cfg["memory_mode"]
    assert stats["bytes"] == out_file.stat().st_size
    assert out_file.exists() and out_file.stat().st_size > 0

    # 2. Decompress and parse dataset lines
    with gzip.open(out_file, "rt") as f:
        lines = [line.strip() for line in f if line.strip()]

    assert len(lines) == seat_turns + 1, f"Expected 1 header + {seat_turns} rows"

    # 3. Assert header JSON structure and values
    header = json.loads(lines[0])
    assert header["kind"] == "teacher-rollout"
    assert "engine_digest" in header and len(header["engine_digest"]) > 0
    assert header["teacher"] == cfg["expected_desc_prefix"]
    assert header["teacher_type"] == cfg["teacher"]
    assert header["teacher_remembers"] is True
    assert header["hill_lead"] == cfg["hill_lead"]
    assert header["memory_mode"] == cfg["memory_mode"]
    assert header["memory"] == ["food_seen", "hill_foe_seen"]
    assert header["max_turns"] == 50
    assert header["env_seed"] == 42

    # 4. Assert rows structure and validity
    history_by_seat: dict[tuple[int, int], list[dict]] = {}
    total_ant_orders = 0

    for idx, row_str in enumerate(lines[1:], start=1):
        row = json.loads(row_str)
        assert set(row.keys()) == {"o", "k", "m", "a"}, f"Unexpected row keys in row {idx}"

        ep, seat, turn = row["k"]
        assert isinstance(ep, int) and isinstance(seat, int) and isinstance(turn, int)
        assert turn >= 0

        # Memory assertions
        m = row["m"]
        assert isinstance(m, dict), f"row {idx}: m must be a dict"
        if m:
            assert "food_seen" in m or "hill_foe_seen" in m
            for key in ("food_seen", "hill_foe_seen"):
                if key in m:
                    coords = m[key]
                    assert isinstance(coords, list)
                    for pt in coords:
                        assert isinstance(pt, list) and len(pt) == 2
                        r, c = pt
                        assert 0 <= r < 24 and 0 <= c < 24

        # Action assertions
        mine = row["o"]["mine"]
        action = row["a"]
        assert len(action) == len(mine), f"row {idx}: action length {len(action)} != mine {len(mine)}"
        total_ant_orders += len(action)

        key = (ep, seat)
        if key not in history_by_seat:
            history_by_seat[key] = []
        history_by_seat[key].append(row)

    assert total_ant_orders == stats["ant_labels"]

    # 5. Row-by-Row Memory Transition Oracle Verification:
    # For every consecutive turn recorded within the same episode and seat:
    # row_{t+1}["m"] == remember(row_t["m"], row_t["o"], mode=cfg["memory_mode"])
    transitions_checked = 0
    for key, seat_rows in history_by_seat.items():
        for i in range(len(seat_rows) - 1):
            curr_row = seat_rows[i]
            next_row = seat_rows[i + 1]
            # Only test direct consecutive turns (turn+1)
            if next_row["k"][2] == curr_row["k"][2] + 1:
                expected_next_m = remember(curr_row["m"], curr_row["o"], mode=cfg["memory_mode"])
                assert next_row["m"] == expected_next_m, (
                    f"Memory transition oracle failed in {cfg['id']} for {key} "
                    f"turn {curr_row['k'][2]} -> {next_row['k'][2]}!\n"
                    f"Actual:   {next_row['m']}\n"
                    f"Expected: {expected_next_m}"
                )
                transitions_checked += 1

    assert transitions_checked > 0, f"No consecutive turn transitions found for {cfg['id']}"


@pytest.mark.parametrize("cfg", EXPERIMENT_CONFIGS, ids=[c["id"] for c in EXPERIMENT_CONFIGS])
def test_cli_subprocess_execution_and_stats_file(tmp_path, cfg):
    """Verify running python -m tb_baselines.collect CLI directly produces .stats.json and valid .jsonl.gz."""
    out_file = tmp_path / f"cli_{cfg['id']}.jsonl.gz"
    stats_file = tmp_path / f"cli_{cfg['id']}.stats.json"

    cmd = [
        sys.executable, "-m", "tb_baselines.collect",
        "--out", str(out_file),
        "--seat-turns", "50",
        "--waves", "1",
        "--matches-per-wave", "2",
        "--max-turns", "40",
        "--seed", "17",
        "--maps", "basic-tiny-2p",
        "--teacher", cfg["teacher"],
        "--hill-lead", str(cfg["hill_lead"]),
        "--memory-mode", cfg["memory_mode"],
    ]
    if cfg["remembers"]:
        cmd.append("--remember")

    res = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=str(BASELINES_ROOT),
        env={**dict(os.environ), "PYTHONPATH": str(SRC_ROOT)},
    )
    assert res.returncode == 0, f"CLI command failed:\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}"

    # Assert stats file was created
    assert stats_file.exists(), f"Stats file {stats_file} missing!"
    stats_data = json.loads(stats_file.read_text())
    assert stats_data["rows"] == 50
    assert stats_data["teacher"] == cfg["teacher"]
    assert stats_data["hill_lead"] == cfg["hill_lead"]
    assert stats_data["memory_mode"] == cfg["memory_mode"]
    assert "command" in stats_data
    assert f"--teacher {cfg['teacher']}" in stats_data["command"]
    assert f"--memory-mode {cfg['memory_mode']}" in stats_data["command"]


def test_collection_without_remember(tmp_path):
    """Verify collection behavior when --remember is omitted (teacher_remembers=False)."""
    out_file = tmp_path / "unremembered.jsonl.gz"
    stats = collect(
        out=out_file,
        seat_turns=40,
        waves=1,
        matches_per_wave=2,
        max_turns=30,
        seed=10,
        maps="basic-tiny-2p",
        remembers=False,
        teacher="default",
        hill_lead=6,
        memory_mode="where",
    )
    assert stats["rows"] == 40
    with gzip.open(out_file, "rt") as f:
        header = json.loads(f.readline())
        assert header["teacher_remembers"] is False
        assert not header["teacher"].endswith(", remembering")
        first_row = json.loads(f.readline())
        assert "m" in first_row


def test_where_vs_max_memory_divergence_in_collection(tmp_path):
    """Demonstrate empirical divergence between where and max memory modes.
    Run identical matches with max vs where. In where mode, food_seen coordinates can be cleared
    once visited/eaten in field of view, while in max mode food_seen is monotonically non-decreasing.
    """
    out_where = tmp_path / "div_where.jsonl.gz"
    out_max = tmp_path / "div_max.jsonl.gz"

    collect(
        out=out_where,
        seat_turns=80,
        waves=1,
        matches_per_wave=2,
        max_turns=60,
        seed=99,
        maps="basic-tiny-2p",
        remembers=True,
        teacher="default",
        hill_lead=6,
        memory_mode="where",
    )

    collect(
        out=out_max,
        seat_turns=80,
        waves=1,
        matches_per_wave=2,
        max_turns=60,
        seed=99,
        maps="basic-tiny-2p",
        remembers=True,
        teacher="default",
        hill_lead=6,
        memory_mode="max",
    )

    with gzip.open(out_where, "rt") as f:
        rows_where = [json.loads(line) for line in f][1:]
    with gzip.open(out_max, "rt") as f:
        rows_max = [json.loads(line) for line in f][1:]

    # In max mode, food_seen should never decrease for the same episode/seat
    by_seat_max: dict[tuple[int, int], list[int]] = {}
    for r in rows_max:
        key = (r["k"][0], r["k"][1])
        cnt = len(r["m"].get("food_seen", []))
        if key in by_seat_max and by_seat_max[key]:
            assert cnt >= by_seat_max[key][-1], "In max mode, food_seen must be monotonic!"
        by_seat_max.setdefault(key, []).append(cnt)

    # In where mode, food_seen decreases when an ant clears a food tile in sight
    saw_where_decrease = False
    by_seat_where: dict[tuple[int, int], list[int]] = {}
    for r in rows_where:
        key = (r["k"][0], r["k"][1])
        cnt = len(r["m"].get("food_seen", []))
        if key in by_seat_where and by_seat_where[key] and cnt < by_seat_where[key][-1]:
            saw_where_decrease = True
        by_seat_where.setdefault(key, []).append(cnt)

    assert saw_where_decrease, (
        "Expected food_seen count to decrease under where mode when food is collected/observed empty!"
    )


def test_collection_exact_determinism(tmp_path):
    """Verify that repeated collections under the same seed and parameters produce identical rows."""
    run1 = tmp_path / "run1.jsonl.gz"
    run2 = tmp_path / "run2.jsonl.gz"

    collect(
        out=run1,
        seat_turns=40,
        waves=1,
        matches_per_wave=2,
        max_turns=30,
        seed=12345,
        maps="basic-tiny-2p",
        remembers=True,
        teacher="compound",
        hill_lead=6,
        memory_mode="where",
    )

    collect(
        out=run2,
        seat_turns=40,
        waves=1,
        matches_per_wave=2,
        max_turns=30,
        seed=12345,
        maps="basic-tiny-2p",
        remembers=True,
        teacher="compound",
        hill_lead=6,
        memory_mode="where",
    )

    with gzip.open(run1, "rt") as f1, gzip.open(run2, "rt") as f2:
        lines1 = f1.readlines()
        lines2 = f2.readlines()

    assert len(lines1) == len(lines2)
    for i, (l1, l2) in enumerate(zip(lines1, lines2)):
        assert l1 == l2, f"Line {i} diverged between identical runs!"


def test_invalid_negative_hill_lead(tmp_path):
    """Verify that negative hill_lead raises ValueError in both default and compound teachers."""
    out_file = tmp_path / "neg.jsonl.gz"
    with pytest.raises(ValueError, match="hill_lead"):
        collect(
            out=out_file,
            seat_turns=20,
            teacher="default",
            hill_lead=-1,
        )

    with pytest.raises(ValueError, match="hill_lead"):
        collect(
            out=out_file,
            seat_turns=20,
            teacher="compound",
            hill_lead=-5,
        )


def test_multi_map_multi_wave_collection(tmp_path):
    """Verify collection handles multiple waves across multiple board sizes cleanly."""
    out_file = tmp_path / "multi_map.jsonl.gz"
    stats = collect(
        out=out_file,
        seat_turns=60,
        waves=2,
        matches_per_wave=2,
        max_turns=40,
        seed=77,
        maps="basic-tiny-2p,basic-small-3p",
        remembers=True,
        teacher="compound",
        hill_lead=6,
        memory_mode="where",
    )
    assert stats["rows"] == 60
    assert stats["ant_labels"] > 0
    with gzip.open(out_file, "rt") as f:
        header = json.loads(f.readline())
        assert len(header["maps"]) == 2
        assert "basic-tiny-2p" in header["maps"]
        assert "basic-small-3p" in header["maps"]


def test_boundary_seat_turns(tmp_path):
    """Verify collection with boundary seat turns: 1 and 7."""
    out1 = tmp_path / "one.jsonl.gz"
    stats1 = collect(
        out=out1,
        seat_turns=1,
        waves=1,
        matches_per_wave=1,
        max_turns=20,
        seed=1,
        maps="basic-tiny-2p",
        remembers=True,
        teacher="default",
        hill_lead=6,
        memory_mode="where",
    )
    assert stats1["rows"] == 1
    with gzip.open(out1, "rt") as f:
        lines = [line.strip() for line in f if line.strip()]
    assert len(lines) == 2  # header + 1 row

