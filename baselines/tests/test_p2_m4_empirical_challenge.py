"""Phase 2 Milestone 4 Empirical Challenge & Verification Test Suite.

Audits and stress-tests:
1. Tournament Execution:
   - Exactly 504 replay files in replays/tournament_p2_9map/
   - 8 entrants across 9 benchmark maps with reciprocal seat swap (18 matches per pair, 126 per entrant).
2. Platform Envelope & Invariants:
   - Invariant engine digest sha256:819166e7918117a1cbbd67dc042e7368a076a2dd45cea5122683983912d591e1 across all 504 replays.
   - Max inference latency strictly <= 700 ms (700,000 us).
3. Tournament Artifacts:
   - data/tournament_p2_results/tournament_metrics.json exists and is valid JSON.
   - data/tournament_p2_results/tournament_summary.md exists and contains executive summary, leaderboard, and matrices.
4. Leaderboard Ranking & Bugfix Verification:
   - Rankings are sorted strictly by Score Rate descending (points / total matches), NOT by alphabetical name.
   - Verify rank 1 is exp022 (Score Rate 61.1%), rank 2 is exp020 (60.7%), rank 3 is micro-bc (59.1%), rank 4 is exp021 (58.7%).
5. Reciprocal Mathematical Symmetry:
   - Head-to-head records satisfy: W_AB == L_BA, D_AB == D_BA, Pts_AB + Pts_BA == 18.0.
6. Capacity Scaling Affirmation:
   - All 3 scaled models (exp020, exp021, exp022) achieve positive capacity scaling gain (> +10%) over corresponding nano baselines.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import pytest

REPLAYS_DIR = Path("/home/host/project/ants/replays/tournament_p2_9map")
METRICS_JSON = Path("/home/host/project/ants/data/tournament_p2_results/tournament_metrics.json")
SUMMARY_MD = Path("/home/host/project/ants/data/tournament_p2_results/tournament_summary.md")
EXPECTED_DIGEST = "sha256:819166e7918117a1cbbd67dc042e7368a076a2dd45cea5122683983912d591e1"
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


@pytest.fixture(scope="module")
def replay_files() -> List[Path]:
    assert REPLAYS_DIR.exists(), f"Replays directory not found: {REPLAYS_DIR}"
    files = sorted(REPLAYS_DIR.glob("*.json"))
    return [f for f in files if not f.name.startswith("_batch_")]


@pytest.fixture(scope="module")
def metrics() -> dict:
    assert METRICS_JSON.exists(), f"Metrics JSON missing: {METRICS_JSON}"
    return json.loads(METRICS_JSON.read_text())


@pytest.fixture(scope="module")
def summary_md() -> str:
    assert SUMMARY_MD.exists(), f"Summary MD missing: {SUMMARY_MD}"
    return SUMMARY_MD.read_text()


class TestPhase2TournamentExecution:
    """Verifies all 504 tournament matches and replay files exist and adhere to platform envelope."""

    def test_total_matches_is_504(self, replay_files: List[Path]):
        assert len(replay_files) == 504, f"Expected exactly 504 replay files, found {len(replay_files)}"

    def test_engine_digest_invariant_across_replays(self, replay_files: List[Path]):
        # Spot check all replays for invariant engine digest
        for f in replay_files:
            data = json.loads(f.read_text())
            assert data.get("engine_digest") == EXPECTED_DIGEST, (
                f"Replay {f.name} divergent engine digest: {data.get('engine_digest')}"
            )

    def test_latency_under_platform_deadline(self, replay_files: List[Path]):
        for f in replay_files:
            data = json.loads(f.read_text())
            for seat in data.get("seats", []):
                infer_max = seat.get("infer_us_max", 0)
                assert infer_max <= 700_000, (
                    f"Match {f.name} seat {seat.get('label')} exceeded 700ms deadline: {infer_max} us"
                )

    def test_exact_reciprocal_seat_and_map_symmetry(self, replay_files: List[Path]):
        """Empirically validates exact 63/63 seat balance per entrant and 56 matches per map."""
        from collections import Counter, defaultdict

        seat0_counts = Counter()
        seat1_counts = Counter()
        pairs = Counter()
        map_counts = defaultdict(lambda: Counter())

        for f in replay_files:
            data = json.loads(f.read_text())
            s0 = data["seats"][0]["label"]
            s1 = data["seats"][1]["label"]
            m_id = data.get("map_id")

            seat0_counts[s0] += 1
            seat1_counts[s1] += 1
            pairs[(s0, s1)] += 1
            map_counts[m_id][(s0, s1)] += 1

        for entrant in EXPECTED_ENTRANTS:
            assert seat0_counts[entrant] == 63, f"Entrant {entrant} seat 0 count != 63: {seat0_counts[entrant]}"
            assert seat1_counts[entrant] == 63, f"Entrant {entrant} seat 1 count != 63: {seat1_counts[entrant]}"

        # Exactly 9 benchmark maps, each hosting 56 matches (all 56 unique ordered pairs)
        assert len(map_counts) == 9, f"Expected 9 maps, got {len(map_counts)}"
        for m_id, pdict in map_counts.items():
            assert len(pdict) == 56, f"Map {m_id} did not execute all 56 ordered pairs"
            for pair, count in pdict.items():
                assert count == 1, f"Map {m_id} pair {pair} executed {count} times instead of 1"

        # Every ordered pair must have exactly 9 matches (1 per map)
        for pair, count in pairs.items():
            assert count == 9, f"Pair {pair} executed {count} times instead of 9"


class TestPhase2TournamentMetrics:
    """Verifies tournament metrics JSON and Markdown summary structure and values."""

    def test_entrants_and_match_totals(self, metrics: dict):
        entrants = set(metrics.get("entrants", []))
        assert entrants == EXPECTED_ENTRANTS, f"Entrants mismatch: {entrants}"
        assert metrics.get("total_matches") == 504
        assert metrics.get("engine_digest") == EXPECTED_DIGEST

    def test_leaderboard_sorted_by_score_rate_not_alphabetical(self, metrics: dict):
        leaderboard = metrics.get("leaderboard", [])
        assert len(leaderboard) == 8

        # Verify strictly descending by score_rate
        score_rates = [e["points"] / e["matches"] for e in leaderboard]
        for i in range(len(score_rates) - 1):
            assert score_rates[i] >= score_rates[i + 1], (
                f"Leaderboard not sorted descending at index {i}: {score_rates[i]} < {score_rates[i+1]}"
            )

        # Bugfix assertion: verify rankings are NOT alphabetical
        names = [e["name"] for e in leaderboard]
        alphabetical_names = sorted(names)
        assert names != alphabetical_names, "Leaderboard appears alphabetically sorted instead of by score rate!"

        # Specific expected top rankings
        assert names[0] == "exp022", f"Expected exp022 as Rank 1, got {names[0]}"
        assert names[1] == "exp020", f"Expected exp020 as Rank 2, got {names[1]}"
        assert names[2] == "micro-bc", f"Expected micro-bc as Rank 3, got {names[2]}"
        assert names[3] == "exp021", f"Expected exp021 as Rank 4, got {names[3]}"

    def test_pairwise_reciprocal_symmetry(self, metrics: dict):
        h2h = metrics.get("head_to_head", {})
        for p1 in EXPECTED_ENTRANTS:
            for p2 in EXPECTED_ENTRANTS:
                if p1 == p2:
                    continue
                k1 = f"{p1}_vs_{p2}"
                k2 = f"{p2}_vs_{p1}"
                assert k1 in h2h, f"Missing pair {k1}"
                assert k2 in h2h, f"Missing pair {k2}"

                rec1 = h2h[k1]
                rec2 = h2h[k2]
                assert rec1["matches"] == 18, f"Expected 18 matches for {k1}, got {rec1['matches']}"
                assert rec1["p1_wins"] == rec2["p2_wins"], f"Wins asymmetry in {k1} vs {k2}"
                assert rec1["draws"] == rec2["draws"], f"Draws asymmetry in {k1} vs {k2}"
                total_pts = rec1["p1_points"] + rec2["p1_points"]
                assert abs(total_pts - 18.0) < 1e-5, f"Points sum != 18 for {k1}: {total_pts}"

    def test_pairwise_score_rate_matrix_exact_reciprocity(self, metrics: dict):
        """Asserts 8x8 pairwise score rate matrix satisfies M[A][B] + M[B][A] == 1.0."""
        h2h = metrics.get("head_to_head", {})
        leaderboard = metrics.get("leaderboard", [])

        # Total points in tournament must equal 504.0
        total_leaderboard_points = sum(e["points"] for e in leaderboard)
        assert abs(total_leaderboard_points - 504.0) < 1e-5, (
            f"Total points sum across leaderboard {total_leaderboard_points} != 504.0"
        )

        entrants = sorted(list(EXPECTED_ENTRANTS))
        for i, a in enumerate(entrants):
            for b in entrants[i + 1:]:
                k_ab = f"{a}_vs_{b}"
                k_ba = f"{b}_vs_{a}"
                assert k_ab in h2h and k_ba in h2h

                pts_ab = h2h[k_ab]["p1_points"]
                pts_ba = h2h[k_ba]["p1_points"]
                assert abs((pts_ab + pts_ba) - 18.0) < 1e-5, f"Points sum != 18.0 for {k_ab}"

                rate_ab = pts_ab / 18.0
                rate_ba = pts_ba / 18.0
                assert abs((rate_ab + rate_ba) - 1.0) < 1e-6, (
                    f"Reciprocal normalized score rate sum != 1.0 for {a} vs {b}: {rate_ab} + {rate_ba} = {rate_ab + rate_ba}"
                )

    def test_mean_hill_raze_turns_bounded_and_positive(self, metrics: dict):
        """Asserts mean hill raze turn across all winning games is positive and <= 300."""
        leaderboard = metrics.get("leaderboard", [])
        for e in leaderboard:
            raze_turns = e.get("raze_turns", [])
            assert len(raze_turns) > 0, f"Entrant {e['name']} has 0 raze turns"
            mean_turn = sum(raze_turns) / len(raze_turns)
            assert 0.0 < mean_turn <= 300.0, f"Entrant {e['name']} mean raze turn out of bounds: {mean_turn}"
            for t in raze_turns:
                assert 0 < t <= 300, f"Entrant {e['name']} individual raze turn out of bounds: {t}"

        # Also verify head to head raze turns
        h2h = metrics.get("head_to_head", {})
        for pair_key, record in h2h.items():
            for turn_key in ["p1_raze_turns", "p2_raze_turns"]:
                turns = record.get(turn_key, [])
                if turns:
                    mean_pair_turn = sum(turns) / len(turns)
                    assert 0.0 < mean_pair_turn <= 300.0, f"Pair {pair_key} {turn_key} mean turn out of bounds: {mean_pair_turn}"
                    for t in turns:
                        assert 0 < t <= 300, f"Pair {pair_key} {turn_key} turn out of bounds: {t}"

    def test_collision_rates_and_blocked_moves_physically_bounded(self, metrics: dict):
        """Asserts collision rates and blocked move percentages are bounded and physically plausible."""
        leaderboard = metrics.get("leaderboard", [])
        for e in leaderboard:
            matches = e.get("matches", 126)
            total_orders = e.get("total_orders", 0)
            blocked_orders = e.get("blocked_orders", 0)
            collisions = e.get("collisions", 0)
            total_turns = e.get("total_turns", 0)

            assert matches == 126
            assert total_orders > 0
            assert total_turns > 0

            # Blocked orders percentage bounded (e.g. 0.5% - 10.0%)
            blocked_pct = (blocked_orders / total_orders) * 100.0
            assert 0.5 <= blocked_pct <= 10.0, f"Blocked move % out of plausible bounds for {e['name']}: {blocked_pct}%"

            # Collisions per match bounded (e.g. 1.0 - 30.0)
            coll_per_match = collisions / matches
            assert 1.0 <= coll_per_match <= 30.0, f"Collisions per match out of bounds for {e['name']}: {coll_per_match}"

            # Collision rate relative to orders (e.g. 0.5% - 10.0%)
            coll_rate = (collisions / total_orders) * 100.0
            assert 0.5 <= coll_rate <= 10.0, f"Collision rate out of bounds for {e['name']}: {coll_rate}%"

    def test_capacity_scaling_positive_gains(self, metrics: dict):
        cap = metrics.get("capacity_scaling", {})
        assert "exp020_vs_exp011" in cap
        assert "exp021_vs_exp013" in cap
        assert "exp022_vs_exp014" in cap

        assert cap["exp020_vs_exp011"]["delta_score_rate"] > 0.10, "EXP-020 capacity gain must exceed +10%"
        assert cap["exp021_vs_exp013"]["delta_score_rate"] > 0.10, "EXP-021 capacity gain must exceed +10%"
        assert cap["exp022_vs_exp014"]["delta_score_rate"] > 0.10, "EXP-022 capacity gain must exceed +10%"

    def test_summary_markdown_contains_required_sections(self, summary_md: str):
        assert "# TinyBrains Ants IDEA-002 Phase 2: 9-Map Round-Robin Tournament" in summary_md
        assert "## 2. Tournament Overall Leaderboard" in summary_md
        assert "## 3. Pairwise Score Rate Cross-Matrix (8×8)" in summary_md
        assert "## 4. Pairwise Mean Hill Destroy Turn Matrix (8×8)" in summary_md
        assert "## 5. Economic & Navigation Operational Quality" in summary_md
        assert "## 6. Capacity Scaling & Benchmark Comparative Net Gain Analysis" in summary_md
        assert "## 7. Scientific Hypothesis Arbitration & Empirical Verdict" in summary_md
