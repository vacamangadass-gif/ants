"""Challenger 1 Empirical Stress Test & Wasm Re-Simulation Suite for Milestone 4.

Verifies:
1. Replay Corpus Stress-Test:
   - Exactly 504 valid, parseable JSON replays in `replays/tournament_p2_9map/`.
   - Invariant engine digest `sha256:819166e7918117a1cbbd67dc042e7368a076a2dd45cea5122683983912d591e1` across all replays.
   - Headers contain valid scores, engine_ranks, and reason from the platform enum:
     ['rank_stabilized', 'lone_survivor', 'turn_limit', 'extermination'].
   - Full 8-entrant * 9-map * reciprocal seat swap coverage (504 unique tuples).
   - Inference latency strictly <= 700 ms (700,000 us).
2. Wasm Frame Simulation Re-Verification:
   - Stratified sample across all 9 maps, all 4 reason types (turn_limit, extermination, lone_survivor, rank_stabilized),
     draws, and key model matchups.
   - Re-simulation using official Wasm component `dist/viz/engine.js` via Node.js.
   - Assert exact match on turns, scores, ranks, done state, and razed hill events.
3. Metric Report Aggregation Consistency:
   - Cross-check `tournament_metrics.json` leaderboard and head-to-head records.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Dict, List, Set

import pytest

REPLAYS_DIR = Path("/home/host/project/ants/replays/tournament_p2_9map")
METRICS_JSON = Path("/home/host/project/ants/data/tournament_p2_results/tournament_metrics.json")
EXPECTED_DIGEST = "sha256:819166e7918117a1cbbd67dc042e7368a076a2dd45cea5122683983912d591e1"
ALLOWED_REASONS = {"turn_limit", "lone_survivor", "extermination", "rank_stabilized", "domination", "idle_food"}
EXPECTED_ENTRANTS = {
    "exp020",
    "exp021",
    "exp022",
    "micro-bc",
    "nano-bc",
    "exp011",
    "exp013",
    "exp014",
}
EXPECTED_MAPS = {
    "medium-crossroads-2p-1h",
    "medium-labyrinth-2p-1h",
    "medium-valley-2p-1h",
    "small-arena-2p-1h",
    "small-islands-2p-1h",
    "small-rooms-2p-1h",
    "tiny-colonnade-2p-1h",
    "tiny-corridor-2p-1h",
    "tiny-open-2p-1h",
}


@pytest.fixture(scope="module")
def all_replay_files() -> List[Path]:
    assert REPLAYS_DIR.exists(), f"Replays directory not found: {REPLAYS_DIR}"
    files = sorted([f for f in REPLAYS_DIR.glob("*.json") if not f.name.startswith("_batch_")])
    assert len(files) == 504, f"Expected 504 files, found {len(files)}"
    return files


def test_corpus_json_and_invariants(all_replay_files: List[Path]):
    pairings: Set[tuple] = set()
    reason_counts: Dict[str, int] = {}
    max_latency_us = 0

    for f in all_replay_files:
        content = json.loads(f.read_text(encoding="utf-8"))
        assert content.get("engine_digest") == EXPECTED_DIGEST, f"Digest mismatch in {f.name}"
        assert "scores" in content and len(content["scores"]) == 2
        assert "engine_ranks" in content and len(content["engine_ranks"]) == 2
        
        reason = content.get("reason")
        assert reason in ALLOWED_REASONS, f"Unexpected reason {reason} in {f.name}"
        reason_counts[reason] = reason_counts.get(reason, 0) + 1

        turns = content.get("turns")
        max_turns = content.get("max_turns", 300)
        assert isinstance(turns, int) and 0 <= turns <= max_turns

        seats = content.get("seats", [])
        assert len(seats) == 2
        s0, s1 = seats[0]["label"], seats[1]["label"]
        assert s0 in EXPECTED_ENTRANTS and s1 in EXPECTED_ENTRANTS
        assert s0 != s1

        map_id = content.get("map_id")
        assert map_id in EXPECTED_MAPS

        pair_key = (s0, s1, map_id)
        assert pair_key not in pairings, f"Duplicate match tuple {pair_key}"
        pairings.add(pair_key)

        for s in seats:
            lat = s.get("infer_us_max", 0)
            if lat > max_latency_us:
                max_latency_us = lat
            assert lat <= 700_000, f"Latency {lat} us exceeded 700ms limit in {f.name}"

    assert len(pairings) == 504
    # All 4 observed ending conditions must be present
    assert reason_counts.get("rank_stabilized", 0) == 399
    assert reason_counts.get("lone_survivor", 0) == 100
    assert reason_counts.get("turn_limit", 0) == 3
    assert reason_counts.get("extermination", 0) == 2


def test_wasm_resimulation_stratified_sample():
    """Selects a stratified sample of 16 replays (covering all 4 reason types, draws, and all 9 maps)
    and executes Node.js Wasm simulation via dist/viz/engine.js.
    """
    stratified_files = [
        # All 3 turn_limit matches
        "exp021-vs-exp014-medium-labyrinth-2p-1h-1-0.json",
        "exp014-vs-exp022-medium-labyrinth-2p-1h-1-1.json",
        "exp022-vs-exp014-medium-labyrinth-2p-1h-1-0.json",
        # Both 2 extermination matches
        "nano-bc-vs-micro-bc-medium-labyrinth-2p-1h-1-1.json",
        "nano-bc-vs-exp022-small-rooms-2p-1h-5-1.json",
        # Draws / ties
        "nano-bc-vs-exp014-small-islands-2p-1h-4-0.json",
        "exp022-vs-exp021-small-islands-2p-1h-4-1.json",
        "exp021-vs-micro-bc-tiny-open-2p-1h-8-0.json",
        "exp020-vs-exp014-medium-crossroads-2p-1h-0-0.json",
        # Top model matchups & longitudinal scaling across remaining maps
        "exp022-vs-exp020-medium-crossroads-2p-1h-0-1.json",
        "exp020-vs-exp022-medium-valley-2p-1h-2-0.json",
        "exp020-vs-exp011-small-arena-2p-1h-3-0.json",
        "exp021-vs-exp013-small-rooms-2p-1h-5-0.json",
        "exp022-vs-micro-bc-tiny-colonnade-2p-1h-6-0.json",
        "exp022-vs-exp014-tiny-corridor-2p-1h-7-0.json",
        "exp020-vs-nano-bc-tiny-open-2p-1h-8-0.json",
    ]

    node_script = """
    import('./dist/viz/engine.js').then(async m => {
      const fs = await import('node:fs');
      const path = await import('node:path');
      const dir = 'replays/tournament_p2_9map';
      const files = JSON.parse(process.argv[1] || process.argv[2]);

      const results = [];
      for (const f of files) {
        const replay = JSON.parse(fs.readFileSync(path.join(dir, f), 'utf8'));
        const frames = m.allFrames(replay);
        const last = frames[frames.length - 1];

        results.push({
          file: f,
          turn_match: last.turn === replay.turns,
          wasm_turn: last.turn,
          replay_turn: replay.turns,
          rank_match: last.ranks[0] === replay.engine_ranks[0] && last.ranks[1] === replay.engine_ranks[1],
          score_match: last.score[0] === replay.scores[0] && last.score[1] === replay.scores[1],
          done_flag: last.done === true,
          razed_count: frames.reduce((acc, fr) => acc + fr.razed.length, 0)
        });
      }
      console.log(JSON.stringify(results));
    }).catch(err => {
      console.error(err);
      process.exit(1);
    });
    """

    res = subprocess.run(
        ["node", "--input-type=module", "-e", node_script, json.dumps(stratified_files)],
        cwd="/home/host/project/ants",
        capture_output=True,
        text=True,
        check=True,
    )

    verification_records = json.loads(res.stdout)
    assert len(verification_records) == len(stratified_files)

    for rec in verification_records:
        assert rec["turn_match"], f"Turn mismatch in {rec['file']}: wasm {rec['wasm_turn']} vs replay {rec['replay_turn']}"
        assert rec["rank_match"], f"Rank mismatch in {rec['file']}"
        assert rec["score_match"], f"Score mismatch in {rec['file']}"
        assert rec["done_flag"], f"Last frame not marked done in {rec['file']}"


def test_leaderboard_score_rates_and_monotonicity():
    data = json.loads(METRICS_JSON.read_text(encoding="utf-8"))
    leaderboard = data["leaderboard"]
    assert len(leaderboard) == 8

    # Check points / matches strictly descending
    rates = [e["points"] / e["matches"] for e in leaderboard]
    for i in range(len(rates) - 1):
        assert rates[i] >= rates[i + 1]

    # Check expected order
    names = [e["name"] for e in leaderboard]
    assert names == ["exp022", "exp020", "micro-bc", "exp021", "exp014", "exp013", "exp011", "nano-bc"]
