"""Milestone 4 Adversarial Cross-Matrix and Empirical Challenge Suite.

Author: teamwork_preview_challenger_p2_m4_v2
Archetype: EMPIRICAL CHALLENGER (critic, specialist)

Strictly validates:
1. Mathematical Reciprocity of 8x8 Pairwise Score Rate Matrix:
   - M[A][B] + M[B][A] == 1.0 (exact normalized score rate reciprocity)
   - Points(A, B) + Points(B, A) == 18.0 (exact match points reciprocity)
   - Row score rate identity: ScoreRate(A) == (1 / 7) * sum_{B != A} M[A][B]
   - Total points sum across tournament == 504.0
2. Hill Raze Dynamics:
   - Mean hill raze turn across all winning games is positive and <= 300.
   - Individual raze turns satisfy 0 < turn <= 300.
   - Censored mean raze turn >= success mean raze turn for every entrant.
3. Physics and Navigation Boundedness:
   - Collision rates and blocked move percentages are bounded and physically plausible.
   - Collisions per match in [1.0, 30.0].
   - Blocked moves percentage in [0.5%, 10.0%].
4. Independent Raw Replay Oracle:
   - Re-extracts tournament results from raw JSON files in replays/tournament_p2_9map/*.json
   - Asserts exact 1:1 match against data/tournament_p2_results/tournament_metrics.json
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import pytest

REPLAYS_DIR = Path("/home/host/project/ants/replays/tournament_p2_9map")
METRICS_JSON = Path("/home/host/project/ants/data/tournament_p2_results/tournament_metrics.json")
SUMMARY_MD = Path("/home/host/project/ants/data/tournament_p2_results/tournament_summary.md")
EXPECTED_DIGEST = "sha256:819166e7918117a1cbbd67dc042e7368a076a2dd45cea5122683983912d591e1"
EXPECTED_ENTRANTS = [
    "exp020",
    "exp021",
    "exp022",
    "micro-bc",
    "nano-bc",
    "exp011",
    "exp013",
    "exp014",
]


@pytest.fixture(scope="module")
def metrics() -> dict:
    assert METRICS_JSON.exists(), f"Metrics file not found: {METRICS_JSON}"
    return json.loads(METRICS_JSON.read_text())


@pytest.fixture(scope="module")
def raw_replays() -> List[dict]:
    assert REPLAYS_DIR.exists(), f"Replays directory not found: {REPLAYS_DIR}"
    files = sorted(REPLAYS_DIR.glob("*.json"))
    valid_files = [f for f in files if not f.name.startswith("_batch_")]
    assert len(valid_files) == 504, f"Expected 504 files, found {len(valid_files)}"
    return [json.loads(f.read_text()) for f in valid_files]


class TestAdversarialMatrixReciprocity:
    """Stress-tests mathematical axioms of the 8x8 pairwise score rate matrix."""

    def test_pairwise_score_rate_and_point_reciprocity(self, metrics: dict):
        h2h = metrics["head_to_head"]
        for i, a in enumerate(EXPECTED_ENTRANTS):
            for b in EXPECTED_ENTRANTS[i + 1:]:
                k_ab = f"{a}_vs_{b}"
                k_ba = f"{b}_vs_{a}"
                rec_ab = h2h[k_ab]
                rec_ba = h2h[k_ba]

                assert rec_ab["matches"] == 18
                assert rec_ba["matches"] == 18

                pts_ab = rec_ab["p1_points"]
                pts_ba = rec_ba["p1_points"]

                # Pairwise points must sum to exactly 18.0
                assert abs((pts_ab + pts_ba) - 18.0) < 1e-6, (
                    f"Points sum != 18.0 for {a} vs {b}: {pts_ab} + {pts_ba} = {pts_ab + pts_ba}"
                )

                # Normalized score rate matrix: M[A][B] + M[B][A] == 1.0
                m_ab = pts_ab / 18.0
                m_ba = pts_ba / 18.0
                assert abs((m_ab + m_ba) - 1.0) < 1e-6, (
                    f"Normalized score rate reciprocity failed for {a} vs {b}: {m_ab} + {m_ba} = {m_ab + m_ba}"
                )

                # Inversion of win/loss counts
                assert rec_ab["p1_wins"] == rec_ba["p2_wins"]
                assert rec_ab["p2_wins"] == rec_ba["p1_wins"]
                assert rec_ab["draws"] == rec_ba["draws"]

    def test_row_score_rate_identity(self, metrics: dict):
        """Verifies overall score rate equals exactly the mean of pairwise score rates."""
        h2h = metrics["head_to_head"]
        leaderboard = {e["name"]: e for e in metrics["leaderboard"]}

        for a in EXPECTED_ENTRANTS:
            e = leaderboard[a]
            assert e["matches"] == 126

            pairwise_rates = []
            pairwise_points = []
            for b in EXPECTED_ENTRANTS:
                if a == b:
                    continue
                k_ab = f"{a}_vs_{b}"
                pts = h2h[k_ab]["p1_points"]
                pairwise_points.append(pts)
                pairwise_rates.append(pts / 18.0)

            # Sum of pairwise points must equal total entrant points
            sum_pts = sum(pairwise_points)
            assert abs(sum_pts - e["points"]) < 1e-6, (
                f"Entrant {a} point sum mismatch: {sum_pts} vs {e['points']}"
            )

            # Mean of 7 pairwise rates must equal overall score rate
            mean_rate = sum(pairwise_rates) / 7.0
            actual_rate = e["points"] / 126.0
            assert abs(mean_rate - actual_rate) < 1e-6, (
                f"Entrant {a} rate mismatch: mean({mean_rate}) != actual({actual_rate})"
            )

    def test_tournament_total_points_conservation(self, metrics: dict):
        """The total points in the entire tournament must be conserved at exactly 504.0."""
        total_points = sum(e["points"] for e in metrics["leaderboard"])
        assert abs(total_points - 504.0) < 1e-6, f"Conserved points != 504.0: {total_points}"


class TestAdversarialHillRazeDynamics:
    """Stress-tests hill raze turn limits and censoring bounds."""

    def test_mean_hill_raze_turns_positive_and_bounded(self, metrics: dict):
        for e in metrics["leaderboard"]:
            name = e["name"]
            raze_turns = e["raze_turns"]
            assert len(raze_turns) > 0, f"Entrant {name} has zero raze turns"

            mean_succ = sum(raze_turns) / len(raze_turns)
            assert 0.0 < mean_succ <= 300.0, f"Entrant {name} mean raze turn out of bounds: {mean_succ}"

            # Every individual raze turn must be physically within game turn limits
            for turn in raze_turns:
                assert isinstance(turn, int)
                assert 0 < turn <= 300, f"Entrant {name} individual turn out of bounds: {turn}"

    def test_censored_mean_destroy_turn_relationship(self, metrics: dict):
        for e in metrics["leaderboard"]:
            name = e["name"]
            raze_turns = e["raze_turns"]
            matches = e["matches"]
            assert matches == 126

            mean_succ = sum(raze_turns) / len(raze_turns)
            censored_total = sum(raze_turns) + (matches - len(raze_turns)) * 300
            censored_mean = censored_total / matches

            # Censored mean must be strictly >= success mean since non-razed games are penalized at 300 turns
            assert censored_mean >= mean_succ, (
                f"Entrant {name} censored mean {censored_mean} < success mean {mean_succ}"
            )


class TestAdversarialPhysicsAndCollisions:
    """Stress-tests physical bounds of ant movement, collisions, and pathing."""

    def test_collision_rates_and_blocked_percentages_plausibility(self, metrics: dict):
        for e in metrics["leaderboard"]:
            name = e["name"]
            matches = e["matches"]
            total_orders = e["total_orders"]
            blocked_orders = e["blocked_orders"]
            collisions = e["collisions"]
            total_turns = e["total_turns"]

            assert total_orders > 0
            assert total_turns > 0
            assert blocked_orders >= 0
            assert collisions >= 0

            # Blocked orders percentage bounded
            blocked_pct = (blocked_orders / total_orders) * 100.0
            assert 0.5 <= blocked_pct <= 10.0, f"Blocked orders % anomalous for {name}: {blocked_pct}%"

            # Collisions per match bounded
            coll_per_match = collisions / matches
            assert 1.0 <= coll_per_match <= 30.0, f"Collisions per match anomalous for {name}: {coll_per_match}"

            # Collision rate relative to orders bounded
            coll_rate = (collisions / total_orders) * 100.0
            assert 0.5 <= coll_rate <= 10.0, f"Collision rate anomalous for {name}: {coll_rate}%"

            # Food gathered must be positive and bounded per turn
            food = e["food_gathered"]
            assert food > 0
            food_per_100t = (food / total_turns) * 100.0
            assert 10.0 <= food_per_100t <= 100.0, f"Food gathering rate anomalous for {name}: {food_per_100t}"


class TestIndependentReplayOracle:
    """Independently extracts metrics directly from raw replays and compares against tournament_metrics.json."""

    def test_oracle_rederivation_matches_metrics_json(self, raw_replays: List[dict], metrics: dict):
        assert len(raw_replays) == 504

        rederived_stats = defaultdict(lambda: {
            "matches": 0, "wins": 0, "draws": 0, "losses": 0, "points": 0.0,
            "score_for": 0, "score_against": 0
        })

        for replay in raw_replays:
            assert replay.get("engine_digest") == EXPECTED_DIGEST
            s0 = replay["seats"][0]["label"]
            s1 = replay["seats"][1]["label"]
            score0 = replay["scores"][0]
            score1 = replay["scores"][1]
            rank0 = replay["engine_ranks"][0]
            rank1 = replay["engine_ranks"][1]

            rederived_stats[s0]["matches"] += 1
            rederived_stats[s1]["matches"] += 1
            rederived_stats[s0]["score_for"] += score0
            rederived_stats[s0]["score_against"] += score1
            rederived_stats[s1]["score_for"] += score1
            rederived_stats[s1]["score_against"] += score0

            if rank0 < rank1:
                rederived_stats[s0]["wins"] += 1
                rederived_stats[s0]["points"] += 1.0
                rederived_stats[s1]["losses"] += 1
            elif rank1 < rank0:
                rederived_stats[s1]["wins"] += 1
                rederived_stats[s1]["points"] += 1.0
                rederived_stats[s0]["losses"] += 1
            else:
                rederived_stats[s0]["draws"] += 1
                rederived_stats[s0]["points"] += 0.5
                rederived_stats[s1]["draws"] += 1
                rederived_stats[s1]["points"] += 0.5

        for entrant_data in metrics["leaderboard"]:
            name = entrant_data["name"]
            oracle = rederived_stats[name]
            assert entrant_data["matches"] == oracle["matches"] == 126
            assert entrant_data["wins"] == oracle["wins"]
            assert entrant_data["draws"] == oracle["draws"]
            assert entrant_data["losses"] == oracle["losses"]
            assert abs(entrant_data["points"] - oracle["points"]) < 1e-6
            assert entrant_data["score_for"] == oracle["score_for"]
            assert entrant_data["score_against"] == oracle["score_against"]
