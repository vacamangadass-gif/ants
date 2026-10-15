"""Milestone 4 Empirical Challenge & Adversarial Stress Test Suite.

Adversarially scrutinizes:
1. Platform envelope: <= 700 ms CPU turn latency across all 378 matches and 25,000+ turns.
2. Engine digest invariance across all matches.
3. Hypothesis H1 statistical significance (binomial p-value, Wilson CI) and 9-map breakdown.
4. Hypothesis H2 economic starvation verification for EXP-012.
5. Hypothesis H3 dual synergy restoration (+52.3% win rate gain) in EXP-013 vs EXP-012.
6. Hypothesis H4 dynamic division, collision rate, and leaderboard sorting audit in EXP-014.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import pytest

REPLAYS_DIR = Path("/home/host/project/ants/replays/tournament_9map")
METRICS_JSON = Path("/home/host/project/ants/data/tournament_results/tournament_metrics.json")
SUMMARY_MD = Path("/home/host/project/ants/data/tournament_results/tournament_summary.md")


def binom_test_two_tailed(k: int, n: int, p: float = 0.5) -> float:
    """Exact two-tailed binomial test."""
    p_k = math.comb(n, k) * (p**k) * ((1 - p) ** (n - k))
    p_val = sum(
        math.comb(n, i) * (p**i) * ((1 - p) ** (n - i))
        for i in range(n + 1)
        if math.comb(n, i) * (p**i) * ((1 - p) ** (n - i)) <= p_k + 1e-12
    )
    return min(1.0, p_val)


def wilson_interval(k: float, n: int, z: float = 1.96) -> Tuple[float, float]:
    """Wilson score confidence interval for a proportion."""
    p_hat = k / n
    denom = 1 + z**2 / n
    center = (p_hat + z**2 / (2 * n)) / denom
    margin = (z * math.sqrt(p_hat * (1 - p_hat) / n + z**2 / (4 * n**2))) / denom
    return max(0.0, center - margin), min(1.0, center + margin)


class TestPlatformEnvelope:
    """Rigorous audit of platform envelope constraints across all tournament replays."""

    @pytest.fixture(scope="class")
    def all_replays(self) -> List[dict]:
        assert REPLAYS_DIR.exists(), f"Replays directory not found: {REPLAYS_DIR}"
        files = sorted(REPLAYS_DIR.glob("*.json"))
        assert len(files) == 378, f"Expected 378 replays, found {len(files)}"
        return [json.loads(f.read_text()) for f in files]

    def test_total_matches_and_turns(self, all_replays: List[dict]):
        total_turns = sum(r.get("turns", 0) for r in all_replays)
        assert len(all_replays) == 378, "Must execute exactly 378 matches"
        assert total_turns >= 25000, f"Total turns {total_turns} must exceed 25,000"
        # Actual turns observed: 26,334
        assert total_turns == 26334

    def test_engine_digest_strict_invariance(self, all_replays: List[dict]):
        expected_digest = "sha256:819166e7918117a1cbbd67dc042e7368a076a2dd45cea5122683983912d591e1"
        for r in all_replays:
            assert r.get("engine_digest") == expected_digest, (
                f"Match {r.get('match_id')} divergent digest: {r.get('engine_digest')}"
            )

    def test_single_turn_cpu_latency_limit(self, all_replays: List[dict]):
        max_allowed_us = 700_000  # 700 ms platform turn budget
        observed_max_by_model: Dict[str, int] = {}

        for r in all_replays:
            for s in r.get("seats", []):
                model_name = s["label"]
                max_us = s.get("infer_us_max", 0)
                assert max_us <= max_allowed_us, (
                    f"Model {model_name} in match {r.get('match_id')} violated 700ms: {max_us} us"
                )
                observed_max_by_model[model_name] = max(
                    observed_max_by_model.get(model_name, 0), max_us
                )

        # Confirm all 7 models were tested and latency <= 6000 us (6 ms, 116x below ceiling)
        expected_models = {"exp010", "exp011", "exp012", "exp013", "exp014", "nano-bc", "nano-bc-max-r"}
        assert set(observed_max_by_model.keys()) == expected_models
        for model, peak_us in observed_max_by_model.items():
            assert peak_us < 6000, f"Model {model} peak latency {peak_us} us unexpectedly high"


class TestHypothesisChallenge:
    """Statistical and adversarial challenge of hypotheses H1 to H4."""

    @pytest.fixture(scope="class")
    def metrics(self) -> dict:
        assert METRICS_JSON.exists(), f"Missing metrics file: {METRICS_JSON}"
        return json.loads(METRICS_JSON.read_text())

    def test_h1_statistical_significance_and_map_breakdown(self, metrics: dict):
        """H1 challenge: Scrutinize whether EXP-011's 63.9% win rate over EXP-010

        is statistically significant and verified across all 9 maps.
        """
        h2h = metrics["head_to_head"]
        pair_11_10 = h2h.get("exp011_vs_exp010")
        assert pair_11_10 is not None

        matches = pair_11_10["matches"]
        wins = pair_11_10["p1_wins"]
        draws = pair_11_10["draws"]
        losses = pair_11_10["p2_wins"]
        points = pair_11_10["p1_points"]

        assert matches == 18
        assert wins == 11
        assert draws == 1
        assert losses == 6
        assert abs(points - 11.5) < 1e-5

        # 1. Two-tailed binomial significance on decisive games (n=17, k=11)
        p_decisive = binom_test_two_tailed(wins, wins + losses, 0.5)
        # 11/17 yields p = 0.3323, which is >> 0.05
        assert p_decisive > 0.05, f"Expected p > 0.05 for n=17, k=11, got {p_decisive}"
        assert abs(p_decisive - 0.3323) < 0.01

        # 2. Wilson Score Interval for 11/18
        ci_low, ci_high = wilson_interval(wins, matches)
        # CI [0.386, 0.797] includes 0.50 (parity)
        assert ci_low < 0.50 < ci_high, f"Wilson CI [{ci_low}, {ci_high}] does not span 0.50"

        # 3. Overall tournament performance comparison (108 matches each)
        leaderboard = {s["name"]: s for s in metrics["leaderboard"]}
        s11 = leaderboard["exp011"]
        s10 = leaderboard["exp010"]
        # In overall win rate: 66.67% vs 65.28% (diff = 1.39%)
        diff_overall = (s11["points"] - s10["points"]) / 108
        assert abs(diff_overall - 0.0139) < 0.005

        # 4. Map-by-map audit: Verify it is NOT won across all 9 maps
        replays = sorted(REPLAYS_DIR.glob("*.json"))
        map_outcomes: Dict[str, Dict[str, int]] = {}
        for r_path in replays:
            d = json.loads(r_path.read_text())
            labels = [s["label"] for s in d["seats"]]
            if "exp010" in labels and "exp011" in labels:
                map_id = d["map_id"]
                if map_id not in map_outcomes:
                    map_outcomes[map_id] = {"exp011": 0, "exp010": 0, "draw": 0}
                ranks = d["engine_ranks"]
                s_labels = [s["label"] for s in d["seats"]]
                if ranks[0] < ranks[1]:
                    map_outcomes[map_id][s_labels[0]] += 1
                elif ranks[1] < ranks[0]:
                    map_outcomes[map_id][s_labels[1]] += 1
                else:
                    map_outcomes[map_id]["draw"] += 1

        assert len(map_outcomes) == 9
        # Empirical finding: EXP-011 won on only 3 maps, tied on 5 maps, and lost on 1 map (labyrinth)
        maps_won_by_11 = [m for m, res in map_outcomes.items() if res["exp011"] > res["exp010"]]
        maps_tied = [m for m, res in map_outcomes.items() if res["exp011"] == res["exp010"]]
        maps_won_by_10 = [m for m, res in map_outcomes.items() if res["exp010"] > res["exp011"]]

        assert len(maps_won_by_11) == 3, f"Maps won by exp011: {maps_won_by_11}"
        assert len(maps_tied) == 5, f"Maps tied: {maps_tied}"
        assert len(maps_won_by_10) == 1, f"Maps won by exp010: {maps_won_by_10}"
        assert "medium-labyrinth-2p-1h" in maps_won_by_10

    def test_h2_economic_starvation_verification(self, metrics: dict):
        """H2 challenge: Scrutinize whether EXP-012's low win rate (7.4%) is indeed

        caused by economic starvation (food gathered 11.3 vs 33.4).
        """
        leaderboard = {s["name"]: s for s in metrics["leaderboard"]}
        s12 = leaderboard["exp012"]
        s10 = leaderboard["exp010"]

        win_rate_12 = s12["points"] / s12["matches"]
        assert abs(win_rate_12 - 0.0741) < 0.005  # 7.4%
        assert s12["wins"] == 8
        assert s12["losses"] == 100

        # Food gathered per match: 11.3 vs 33.4
        food_per_match_12 = s12["food_gathered"] / s12["matches"]
        food_per_match_10 = s10["food_gathered"] / s10["matches"]
        assert abs(food_per_match_12 - 11.306) < 0.1
        assert abs(food_per_match_10 - 33.370) < 0.1
        assert food_per_match_10 > 2.9 * food_per_match_12

        # Fast hill destruction on the few successful matches: 27.4t vs 70.0t
        mean_raze_12 = sum(s12["raze_turns"]) / len(s12["raze_turns"])
        mean_raze_10 = sum(s10["raze_turns"]) / len(s10["raze_turns"])
        assert abs(mean_raze_12 - 27.375) < 0.1
        assert abs(mean_raze_10 - 70.028) < 0.1

        # Pathological blocked orders due to Max memory ghost hill chasing:
        blocked_rate_12 = s12["blocked_orders"] / s12["total_orders"]
        blocked_rate_10 = s10["blocked_orders"] / s10["total_orders"]
        assert blocked_rate_12 > 0.11  # 11.62%
        assert blocked_rate_10 < 0.04  # 3.29%
        assert blocked_rate_12 > 3.5 * blocked_rate_10

    def test_h3_dual_synergy_restoration(self, metrics: dict):
        """H3 challenge: Scrutinize whether EXP-013 resolves EXP-012's failure mode

        (win rate 59.7% vs 7.4%, +52.3% gain).
        """
        leaderboard = {s["name"]: s for s in metrics["leaderboard"]}
        s13 = leaderboard["exp013"]
        s12 = leaderboard["exp012"]

        wr13 = s13["points"] / s13["matches"]
        wr12 = s12["points"] / s12["matches"]
        gain = wr13 - wr12

        assert abs(wr13 - 0.5972) < 0.005  # 59.7%
        assert abs(gain - 0.5231) < 0.005  # +52.3% gain

        # Two-proportion z-score test for win rate difference
        p_pool = (s13["points"] + s12["points"]) / (s13["matches"] + s12["matches"])
        se = math.sqrt(p_pool * (1 - p_pool) * (2 / 108))
        z = gain / se
        assert z > 8.0  # Extremely significant, z > 8.0, p < 1e-15

        # Economic recovery: food gathered jumps from 11.3 to 25.7 (+127%)
        food13 = s13["food_gathered"] / s13["matches"]
        food12 = s12["food_gathered"] / s12["matches"]
        assert abs(food13 - 25.7) < 0.5
        assert food13 > 2.2 * food12

        # Blocked move rate drops from 11.62% to 5.14% (55.8% drop)
        blk13 = s13["blocked_orders"] / s13["total_orders"]
        blk12 = s12["blocked_orders"] / s12["total_orders"]
        assert blk13 < 0.055
        assert blk12 > 0.115

    def test_h4_compound_teacher_operational_quality_and_sorting_defect(self, metrics: dict):
        """H4 challenge: Scrutinize EXP-014's dynamic division, collision rate (11.2),

        and uncover the Table 6 sorting defect in tournament_summary.md.
        """
        leaderboard = {s["name"]: s for s in metrics["leaderboard"]}
        s14 = leaderboard["exp014"]
        s11 = leaderboard["exp011"]
        s10 = leaderboard["exp010"]

        # 1. Collision rate audit: exactly 11.2 collisions/match
        col_per_match_14 = s14["collisions"] / s14["matches"]
        col_per_match_10 = s10["collisions"] / s10["matches"]
        col_per_match_11 = s11["collisions"] / s11["matches"]

        assert abs(col_per_match_14 - 11.222) < 0.1
        assert abs(col_per_match_10 - 13.953) < 0.1
        assert abs(col_per_match_11 - 14.305) < 0.1
        # EXP-014 has ~20% fewer collisions than exp010/exp011
        assert col_per_match_14 < col_per_match_10
        assert col_per_match_14 < col_per_match_11

        # 2. Leaderboard rank: EXP-014 is Rank 3 overall by win rate (61.57%)
        sorted_leaderboard = sorted(metrics["leaderboard"], key=lambda x: -x["points"])
        rank_14 = [s["name"] for s in sorted_leaderboard].index("exp014") + 1
        assert rank_14 == 3, f"Expected EXP-014 to be Rank 3, got Rank {rank_14}"

        # 3. Head-to-Head against EXP-011: EXP-014 wins 11-0-7 (61.1%) against the tournament #1
        h2h_14_11 = metrics["head_to_head"]["exp014_vs_exp011"]
        assert h2h_14_11["p1_wins"] == 11
        assert h2h_14_11["p2_wins"] == 7
        assert h2h_14_11["p1_points"] / h2h_14_11["matches"] > 0.61

        # 4. Scrutinize the Table 6 defect in tournament_summary.md:
        md_text = SUMMARY_MD.read_text()
        assert "Rank #5" in md_text, "Expected Table 6 to contain the known 'Rank #5' artifact"
        # Explain why: process_tournament_metrics.py sorted names alphabetically:
        alpha_names = sorted([s["name"] for s in sorted_leaderboard])
        assert alpha_names.index("exp014") + 1 == 5, (
            "Proof that 'Rank #5' was caused by alphabetical sorting of model names"
        )
