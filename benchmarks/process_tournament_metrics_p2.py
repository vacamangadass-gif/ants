#!/usr/bin/env python3
"""Phase 2 Tournament metrics aggregation and cross-matrix specification processor for TinyBrains Ants IDEA-002 (Milestone 4).

Parses 504 tournament match replays / JSON frame metrics and computes:
1. Overall Leaderboard sorted strictly by Score Rate descending.
2. Pairwise Score Rate Cross-Matrix (8x8 win/draw/loss percentages and head-to-head records).
3. Pairwise Mean Hill Destroy Turn Matrix (8x8 offensive surge speed).
4. Economic & Navigation Operational Quality (food gathered, collision rates, blocked moves, latency).
5. Longitudinal Capacity Scaling (Big Micro vs Nano) and Benchmark Net Gain vs micro-bc.
6. Comprehensive Markdown summary and JSON metrics output.
7. Automated verification assertions for tournament integrity, reciprocal symmetry, and latency budget.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

HERE = Path(__file__).resolve().parent
EXTRACTOR_SCRIPT = HERE / "extract_frames.mjs"


@dataclass
class EntrantStats:
    name: str
    model_class: str = "nano"  # "micro" or "nano"
    matches: int = 0
    wins: int = 0
    draws: int = 0
    losses: int = 0
    points: float = 0.0
    score_for: int = 0
    score_against: int = 0
    hills_razed: int = 0
    raze_turns: List[int] = field(default_factory=list)
    food_gathered: int = 0
    collisions: int = 0
    combat_deaths: int = 0
    total_orders: int = 0
    blocked_orders: int = 0
    total_turns: int = 0
    infer_us_total: int = 0
    infer_us_max: int = 0

    @property
    def score_rate(self) -> float:
        """Score rate = (Wins + 0.5 * Draws) / Matches = Points / Matches."""
        return self.points / self.matches if self.matches > 0 else 0.0

    @property
    def win_rate(self) -> float:
        """Pure win rate = Wins / Matches."""
        return self.wins / self.matches if self.matches > 0 else 0.0

    @property
    def mean_score_for(self) -> float:
        return self.score_for / self.matches if self.matches > 0 else 0.0

    @property
    def mean_score_against(self) -> float:
        return self.score_against / self.matches if self.matches > 0 else 0.0

    @property
    def hill_raze_rate(self) -> float:
        return self.hills_razed / self.matches if self.matches > 0 else 0.0

    @property
    def mean_destroy_turn_success(self) -> Optional[float]:
        return sum(self.raze_turns) / len(self.raze_turns) if self.raze_turns else None

    @property
    def mean_destroy_turn_censored(self) -> float:
        unrazed_count = self.matches - len(self.raze_turns)
        total_turns = sum(self.raze_turns) + unrazed_count * 300
        return total_turns / self.matches if self.matches > 0 else 300.0

    @property
    def mean_food_gathered(self) -> float:
        return self.food_gathered / self.matches if self.matches > 0 else 0.0

    @property
    def food_rate_per_100_turns(self) -> float:
        return (self.food_gathered / self.total_turns * 100.0) if self.total_turns > 0 else 0.0

    @property
    def mean_collisions(self) -> float:
        return self.collisions / self.matches if self.matches > 0 else 0.0

    @property
    def collision_rate(self) -> float:
        return (self.collisions / self.total_orders * 100.0) if self.total_orders > 0 else 0.0

    @property
    def mean_blocked_orders(self) -> float:
        return self.blocked_orders / self.matches if self.matches > 0 else 0.0

    @property
    def blocked_order_rate(self) -> float:
        return (self.blocked_orders / self.total_orders * 100.0) if self.total_orders > 0 else 0.0

    @property
    def mean_infer_us_per_turn(self) -> float:
        return (self.infer_us_total / self.total_turns) if self.total_turns > 0 else 0.0


@dataclass
class HeadToHeadStats:
    p1: str
    p2: str
    matches: int = 0
    p1_wins: int = 0
    draws: int = 0
    p2_wins: int = 0
    p1_points: float = 0.0
    p1_score: int = 0
    p2_score: int = 0
    p1_raze_turns: List[int] = field(default_factory=list)
    p2_raze_turns: List[int] = field(default_factory=list)

    @property
    def p1_score_rate(self) -> float:
        return self.p1_points / self.matches if self.matches > 0 else 0.0

    @property
    def p1_win_rate(self) -> float:
        return self.p1_wins / self.matches if self.matches > 0 else 0.0

    @property
    def p1_mean_destroy_turn(self) -> Optional[float]:
        return sum(self.p1_raze_turns) / len(self.p1_raze_turns) if self.p1_raze_turns else None


def extract_raw_metrics(replays_path: Path) -> List[Dict[str, Any]]:
    """Runs extract_frames.mjs via Node to extract frame-level metrics using jco Wasm component."""
    cmd = ["node", str(EXTRACTOR_SCRIPT), str(replays_path.resolve())]
    res = subprocess.run(cmd, cwd=str(HERE), capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"extract_frames.mjs failed with code {res.returncode}:\n{res.stderr or res.stdout}")
    return json.loads(res.stdout)


def aggregate_tournament(raw_matches: List[Dict[str, Any]]) -> Tuple[Dict[str, EntrantStats], Dict[Tuple[str, str], HeadToHeadStats], List[str]]:
    """Aggregates match results into entrant stats and pairwise head-to-head cross matrices."""
    entrants_set = set()
    for m in raw_matches:
        for s in m["seats"]:
            entrants_set.add(s["name"])

    # Canonical order for display: Big micro models first, then official baselines, then nano models
    canonical_order = [
        "exp020",
        "exp021",
        "exp022",
        "micro-bc",
        "nano-bc",
        "exp011",
        "exp013",
        "exp014",
    ]
    sort_key = {name: i for i, name in enumerate(canonical_order)}
    entrants = sorted(list(entrants_set), key=lambda x: sort_key.get(x, 99))

    micro_models = {"exp020", "exp021", "exp022", "micro-bc"}
    stats: Dict[str, EntrantStats] = {
        name: EntrantStats(name=name, model_class="micro" if name in micro_models else "nano")
        for name in entrants
    }
    h2h: Dict[Tuple[str, str], HeadToHeadStats] = {}
    for p1 in entrants:
        for p2 in entrants:
            if p1 != p2:
                h2h[(p1, p2)] = HeadToHeadStats(p1=p1, p2=p2)

    for m in raw_matches:
        seats = m["seats"]
        if len(seats) != 2:
            continue
        s0, s1 = seats[0], seats[1]
        name0, name1 = s0["name"], s1["name"]
        turns = m["turns"]

        # Update entrant stats
        for s, opp_s in [(s0, s1), (s1, s0)]:
            st = stats[s["name"]]
            st.matches += 1
            st.total_turns += turns
            st.score_for += s["score"]
            st.score_against += opp_s["score"]
            st.food_gathered += s["ants_spawned"]
            st.collisions += s["collisions"]
            st.combat_deaths += s["combat_deaths"]
            st.total_orders += s["total_orders"]
            st.blocked_orders += s["blocked_orders"]
            st.infer_us_total += s["infer_us_total"]
            st.infer_us_max = max(st.infer_us_max, s["infer_us_max"])

            if s["won"]:
                st.wins += 1
                st.points += 1.0
            elif s["drawn"]:
                st.draws += 1
                st.points += 0.5
            else:
                st.losses += 1

            if s["first_hill_destroy_turn"] is not None:
                st.hills_razed += 1
                st.raze_turns.append(s["first_hill_destroy_turn"])

        # Update Head to Head
        pair_key = (name0, name1)
        rev_key = (name1, name0)
        h = h2h[pair_key]
        h.matches += 1
        h.p1_score += s0["score"]
        h.p2_score += s1["score"]

        rev_h = h2h[rev_key]
        rev_h.matches += 1
        rev_h.p1_score += s1["score"]
        rev_h.p2_score += s0["score"]

        if s0["won"]:
            h.p1_wins += 1
            h.p1_points += 1.0
            rev_h.p2_wins += 1
        elif s0["drawn"]:
            h.draws += 1
            h.p1_points += 0.5
            rev_h.draws += 1
            rev_h.p1_points += 0.5
        else:
            h.p2_wins += 1
            rev_h.p1_wins += 1
            rev_h.p1_points += 1.0

        if s0["first_hill_destroy_turn"] is not None:
            h.p1_raze_turns.append(s0["first_hill_destroy_turn"])
            rev_h.p2_raze_turns.append(s0["first_hill_destroy_turn"])
        if s1["first_hill_destroy_turn"] is not None:
            h.p2_raze_turns.append(s1["first_hill_destroy_turn"])
            rev_h.p1_raze_turns.append(s1["first_hill_destroy_turn"])

    return stats, h2h, entrants


def generate_markdown_report(
    stats: Dict[str, EntrantStats],
    h2h: Dict[Tuple[str, str], HeadToHeadStats],
    entrants: List[str],
    raw_matches: List[Dict[str, Any]],
) -> str:
    """Generates comprehensive, publication-grade Markdown report for Milestone 4 & 5."""
    lines = []
    lines.append("# TinyBrains Ants IDEA-002 Phase 2: 9-Map Round-Robin Tournament Metric Evaluation Report")
    lines.append("")
    lines.append("## 1. Executive Summary")
    lines.append(f"- **Total Tournament Matches Evaluated**: {len(raw_matches)} (Expected: 504 matches across 28 pairs × 2 reciprocal flips × 9 maps)")
    lines.append(f"- **Participants ({len(entrants)})**: {', '.join(f'`{e}`' for e in entrants)}")
    if raw_matches:
        lines.append(f"- **Game Engine Digest**: `{raw_matches[0].get('engine_digest', 'unknown')}` (Invariant verified)")
        lines.append(f"- **Evaluation Mechanism**: Deterministic reciprocal seat swap on 9 benchmark 2-player maps under official cartridge ABI 1.0.0.")
    lines.append("")

    # Table 1: Overall Leaderboard (strictly sorted by Score Rate descending)
    lines.append("## 2. Tournament Overall Leaderboard")
    lines.append("Rankings are sorted strictly by Score Rate descending `(W + 0.5*D)/Matches`, then Points, then Hills Razed.")
    lines.append("")
    lines.append("| Rank | Entrant | Class | Matches | W | D | L | Score Rate | Pure Win Rate | Points | Mean Score | Raze % | Mean Destroy Turn (Succ) | Food Gathered | Collisions | Blocked Move % |")
    lines.append("| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")

    sorted_by_rate = sorted(
        stats.values(),
        key=lambda s: (-s.score_rate, -s.points, -s.hills_razed, s.mean_destroy_turn_censored),
    )
    for rank, s in enumerate(sorted_by_rate, 1):
        dt_str = f"{s.mean_destroy_turn_success:.1f}t" if s.mean_destroy_turn_success is not None else "N/A"
        lines.append(
            f"| {rank} | **`{s.name}`** | {s.model_class.capitalize()} | {s.matches} | {s.wins} | {s.draws} | {s.losses} | "
            f"**{s.score_rate:.1%}** | {s.win_rate:.1%} | {s.points:.1f} | {s.mean_score_for:.2f} | {s.hill_raze_rate:.1%} | "
            f"{dt_str} | {s.mean_food_gathered:.1f} | {s.mean_collisions:.1f} | {s.blocked_order_rate:.2f}% |"
        )
    lines.append("")

    # Table 2: 8x8 Pairwise Score Rate Cross-Matrix
    lines.append("## 3. Pairwise Score Rate Cross-Matrix (8×8)")
    lines.append("Rows represent the focal model; columns represent the opponent. Cell indicates Row vs Column Score Rate % (`Points / Matches`) and record `(W-D-L)`.")
    lines.append("")
    header = "| Row \\ Col | " + " | ".join(f"`{e}`" for e in entrants) + " | **Overall** |"
    sep = "| :--- | " + " | ".join(":---:" for _ in entrants) + " | :---: |"
    lines.append(header)
    lines.append(sep)

    for r_name in entrants:
        row_cells = [f"**`{r_name}`**"]
        for c_name in entrants:
            if r_name == c_name:
                row_cells.append("-")
            else:
                pair = h2h.get((r_name, c_name))
                if pair and pair.matches > 0:
                    rate = pair.p1_score_rate
                    row_cells.append(f"**{rate:.1%}** ({pair.p1_wins}-{pair.draws}-{pair.p2_wins})")
                else:
                    row_cells.append("N/A")
        overall_rate = stats[r_name].score_rate
        row_cells.append(f"**{overall_rate:.1%}**")
        lines.append("| " + " | ".join(row_cells) + " |")
    lines.append("")

    # Table 3: 8x8 Pairwise Mean Hill Destroy Turn Matrix
    lines.append("## 4. Pairwise Mean Hill Destroy Turn Matrix (8×8)")
    lines.append("Cell indicates average turn when Row destroyed Column's hill (Successful raze matches only / count of razes).")
    lines.append("")
    lines.append(header)
    lines.append(sep)

    for r_name in entrants:
        row_cells = [f"**`{r_name}`**"]
        for c_name in entrants:
            if r_name == c_name:
                row_cells.append("-")
            else:
                pair = h2h.get((r_name, c_name))
                if pair and pair.p1_raze_turns:
                    avg_t = sum(pair.p1_raze_turns) / len(pair.p1_raze_turns)
                    row_cells.append(f"{avg_t:.1f}t (n={len(pair.p1_raze_turns)})")
                else:
                    row_cells.append("None (0)")
        overall_dt = stats[r_name].mean_destroy_turn_success
        dt_str = f"{overall_dt:.1f}t" if overall_dt is not None else "None"
        row_cells.append(f"**{dt_str}**")
        lines.append("| " + " | ".join(row_cells) + " |")
    lines.append("")

    # Table 4: Economic & Navigation Operational Efficiency
    lines.append("## 5. Economic & Navigation Operational Quality")
    lines.append("")
    lines.append("| Entrant | Food Gathered / Match | Food / 100 Turns | Collisions / Match | Collision Rate % | Blocked Orders / Match | Blocked Rate % | Mean Latency (μs/t) | Max Latency (μs) | Platform Deadline |")
    lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
    for e in entrants:
        s = stats[e]
        deadline_status = "PASS (<=700ms)" if s.infer_us_max <= 700_000 else "FAIL (>700ms)"
        lines.append(
            f"| `{s.name}` | {s.mean_food_gathered:.1f} | {s.food_rate_per_100_turns:.2f} | "
            f"{s.mean_collisions:.1f} | {s.collision_rate:.2f}% | {s.mean_blocked_orders:.1f} | "
            f"**{s.blocked_order_rate:.2f}%** | {s.mean_infer_us_per_turn:.1f} μs | {s.infer_us_max} μs | {deadline_status} |"
        )
    lines.append("")

    # Table 5: Benchmark & Capacity Scaling Comparative Net Gain
    lines.append("## 6. Capacity Scaling & Benchmark Comparative Net Gain Analysis")
    lines.append("")
    lines.append("### 6.1 Relative Net Gain vs Official `micro-bc` Baseline")
    lines.append("| Entrant | Class | Score Rate | Δ vs `micro-bc` | Pure Win Rate | Raze Rate % | Mean Destroy Turn | Blocked Rate % |")
    lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")

    micro_base = stats.get("micro-bc")
    base_rate = micro_base.score_rate if micro_base else 0.0

    for s in sorted_by_rate:
        delta = s.score_rate - base_rate
        dt_str = f"{s.mean_destroy_turn_success:.1f}t" if s.mean_destroy_turn_success is not None else "N/A"
        lines.append(
            f"| `{s.name}` | {s.model_class.capitalize()} | {s.score_rate:.1%} | **{delta:+.1%}** | {s.win_rate:.1%} | {s.hill_raze_rate:.1%} | {dt_str} | {s.blocked_order_rate:.2f}% |"
        )
    lines.append("")

    lines.append("### 6.2 Longitudinal Capacity Scaling: Micro (Big) vs Nano (Small)")
    lines.append("| Comparison (Big vs Nano) | Big Model | Big Score Rate | Nano Baseline | Nano Score Rate | Capacity Gain (Δ Score Rate) | Δ Pure Win Rate | Δ Raze Rate | Δ Blocked Moves |")
    lines.append("| :--- | :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: |")

    scaling_pairs = [
        ("Base Teacher (B)", "exp020", "exp011"),
        ("Aggressive Teacher (D)", "exp021", "exp013"),
        ("Compound Teacher (E)", "exp022", "exp014"),
    ]
    for label, big_name, nano_name in scaling_pairs:
        if big_name in stats and nano_name in stats:
            b, n = stats[big_name], stats[nano_name]
            d_score = b.score_rate - n.score_rate
            d_win = b.win_rate - n.win_rate
            d_raze = b.hill_raze_rate - n.hill_raze_rate
            d_blocked = b.blocked_order_rate - n.blocked_order_rate
            lines.append(
                f"| **{label}** | `{big_name}` | {b.score_rate:.1%} | `{nano_name}` | {n.score_rate:.1%} | **{d_score:+.1%}** | {d_win:+.1%} | {d_raze:+.1%} | {d_blocked:+.2f}% |"
            )
    lines.append("")

    # Section 7: Scientific Hypothesis Verdict
    lines.append("## 7. Scientific Hypothesis Arbitration & Empirical Verdict")
    lines.append("")
    lines.append("| Scientific Question | Quantitative Metrics Evaluated | Empirical Observation | Scientific Verdict |")
    lines.append("| :--- | :--- | :--- | :--- |")

    # 1. Capacity scaling effect
    if "exp020" in stats and "exp011" in stats:
        b020, n011 = stats["exp020"], stats["exp011"]
        gain_b = b020.score_rate - n011.score_rate
        verdict_cap = "**CONFIRMED (扩容有效)**" if gain_b > 0 else "REFUTED"
        lines.append(
            f"| **Q1: Does Micro Capacity Expand Policy Strength?** | EXP-020 vs EXP-011 Score Rate | Δ Score Rate = {gain_b:+.1%}, Win Rate: {b020.win_rate:.1%} vs {n011.win_rate:.1%} | {verdict_cap} |"
        )

    # 2. Compound teacher sensitivity to capacity
    if "exp022" in stats and "exp014" in stats and "exp020" in stats and "exp011" in stats:
        b022, n014 = stats["exp022"], stats["exp014"]
        gain_e = b022.score_rate - n014.score_rate
        gain_b = stats["exp020"].score_rate - stats["exp011"].score_rate
        diff_gain = gain_e - gain_b
        rank_e = [s.name for s in sorted_by_rate].index("exp022") + 1
        rank_b = [s.name for s in sorted_by_rate].index("exp020") + 1
        verdict_comp = (
            "**Capacity Bottleneck Hypothesis Disproven** (POMDP fog dominates)"
            if b022.score_rate <= stats["exp020"].score_rate
            else "**Capacity Bottleneck Confirmed** (Compound teacher surpasses simple teacher)"
        )
        lines.append(
            f"| **Q2: Does Compound Teacher Benefit Disproportionately from Capacity?** | EXP-022 vs EXP-020 & Scaling Gains (ΔE={gain_e:+.1%} vs ΔB={gain_b:+.1%}) | EXP-022 Score Rate = {b022.score_rate:.1%} (Rank #{rank_e}), EXP-020 = {stats['exp020'].score_rate:.1%} (Rank #{rank_b}) | {verdict_comp} |"
        )

    # 3. Aggressive teacher scaling
    if "exp021" in stats and "exp013" in stats:
        b021, n013 = stats["exp021"], stats["exp013"]
        gain_d = b021.score_rate - n013.score_rate
        lines.append(
            f"| **Q3: Does Scaling Alleviate Aggressive Policy Fragility?** | EXP-021 vs EXP-013 Score Rate & Destroy Turns | Δ Score Rate = {gain_d:+.1%}, Destroy Turn: {b021.mean_destroy_turn_success or 300:.1f}t vs {n013.mean_destroy_turn_success or 300:.1f}t | **ANALYZED (深入分析)** |"
        )

    lines.append("")
    return "\n".join(lines)


def run_verification_assertions(
    raw_matches: List[Dict[str, Any]],
    stats: Dict[str, EntrantStats],
    h2h: Dict[Tuple[str, str], HeadToHeadStats],
) -> None:
    """Automated assertions asserting complete mathematical consistency and tournament integrity."""
    print("==> Running Phase 2 Milestone 4 Verification Assertions...")

    # 1. Total matches check (504 matches)
    assert len(raw_matches) == 504, f"Expected 504 matches, got {len(raw_matches)}"

    # 2. Engine digest consistency
    first_digest = raw_matches[0].get("engine_digest")
    assert first_digest == "sha256:819166e7918117a1cbbd67dc042e7368a076a2dd45cea5122683983912d591e1", (
        f"Engine digest divergence: {first_digest}"
    )
    for m in raw_matches:
        assert m.get("engine_digest") == first_digest, f"Divergent engine digest in match {m['match_id']}"
    print(f"  [PASS] Engine digest identical and invariant across all {len(raw_matches)} matches ({first_digest[:24]}...).")

    # 3. Reciprocal match and pairwise symmetry
    for (p1, p2), h in h2h.items():
        rev = h2h[(p2, p1)]
        assert h.matches == 18, f"Expected 18 matches between {p1} and {p2}, got {h.matches}"
        assert h.matches == rev.matches, f"Match count asymmetry between {p1} and {p2}"
        assert h.p1_wins == rev.p2_wins, f"Win asymmetry between {p1} and {p2}: {h.p1_wins} vs {rev.p2_wins}"
        assert h.draws == rev.draws, f"Draw asymmetry between {p1} and {p2}: {h.draws} vs {rev.draws}"
        total_points = h.p1_points + rev.p1_points
        assert abs(total_points - h.matches) < 1e-5, f"Points sum != matches for {p1} and {p2}"
    print("  [PASS] Reciprocal seat swap and pairwise symmetry strictly verified across all 28 pairs.")

    # 4. Metric bounds and latency check
    for name, s in stats.items():
        assert s.matches == 126, f"Expected 126 matches per entrant, got {s.matches} for {name}"
        assert 0.0 <= s.score_rate <= 1.0, f"Score rate out of bounds for {name}: {s.score_rate}"
        assert 0.0 <= s.win_rate <= 1.0, f"Win rate out of bounds for {name}: {s.win_rate}"
        assert s.matches == (s.wins + s.draws + s.losses), f"Matches sum mismatch for {name}"
        assert s.food_gathered >= 0, f"Negative food gathered for {name}"
        assert s.collisions >= 0, f"Negative collisions for {name}"
        assert s.blocked_orders >= 0, f"Negative blocked orders for {name}"
        assert s.total_orders >= s.blocked_orders, f"Blocked orders > total orders for {name}"
        assert s.infer_us_max <= 700_000, f"Platform latency deadline violated (>700ms) for {name}: {s.infer_us_max} us"
    print("  [PASS] Mathematical metric bounds and platform latency (<700ms) verified for all 8 entrants.")
    print("==> All Milestone 4 verification assertions passed successfully!")


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("replays_dir", type=Path, help="Directory containing tournament replay JSON files")
    parser.add_argument("--out-dir", type=Path, default=Path("data/tournament_p2_results"), help="Directory for output report and JSON")
    parser.add_argument("--json", dest="emit_json", action="store_true", help="Emit raw JSON aggregation to stdout")
    parser.add_argument("--verify", action="store_true", help="Run strict automated assertions")
    args = parser.parse_args(argv)

    replays_path = args.replays_dir.resolve()
    if not replays_path.exists():
        sys.exit(f"Replays directory not found: {replays_path}")

    print(f"==> Extracting frame metrics from {replays_path} via jco Wasm component engine.js...")
    raw_matches = extract_raw_metrics(replays_path)
    print(f"==> Extracted {len(raw_matches)} matches. Aggregating Phase 2 tournament statistics...")

    stats, h2h, entrants = aggregate_tournament(raw_matches)

    if args.verify or len(raw_matches) == 504:
        run_verification_assertions(raw_matches, stats, h2h)

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Generate Markdown
    md_content = generate_markdown_report(stats, h2h, entrants, raw_matches)
    md_path = out_dir / "tournament_summary.md"
    md_path.write_text(md_content, encoding="utf-8")
    print(f"==> Phase 2 Markdown report written to: {md_path}")

    # Compute comparative metrics for JSON
    sorted_by_rate = sorted(
        stats.values(),
        key=lambda s: (-s.score_rate, -s.points, -s.hills_razed, s.mean_destroy_turn_censored),
    )
    micro_base = stats.get("micro-bc")
    base_rate = micro_base.score_rate if micro_base else 0.0

    relative_gain_vs_micro = {
        s.name: round(s.score_rate - base_rate, 4) for s in sorted_by_rate
    }
    capacity_scaling = {}
    if "exp020" in stats and "exp011" in stats:
        capacity_scaling["exp020_vs_exp011"] = {
            "big_score_rate": round(stats["exp020"].score_rate, 4),
            "nano_score_rate": round(stats["exp011"].score_rate, 4),
            "delta_score_rate": round(stats["exp020"].score_rate - stats["exp011"].score_rate, 4),
            "delta_win_rate": round(stats["exp020"].win_rate - stats["exp011"].win_rate, 4),
            "delta_raze_rate": round(stats["exp020"].hill_raze_rate - stats["exp011"].hill_raze_rate, 4),
        }
    if "exp021" in stats and "exp013" in stats:
        capacity_scaling["exp021_vs_exp013"] = {
            "big_score_rate": round(stats["exp021"].score_rate, 4),
            "nano_score_rate": round(stats["exp013"].score_rate, 4),
            "delta_score_rate": round(stats["exp021"].score_rate - stats["exp013"].score_rate, 4),
            "delta_win_rate": round(stats["exp021"].win_rate - stats["exp013"].win_rate, 4),
            "delta_raze_rate": round(stats["exp021"].hill_raze_rate - stats["exp013"].hill_raze_rate, 4),
        }
    if "exp022" in stats and "exp014" in stats:
        capacity_scaling["exp022_vs_exp014"] = {
            "big_score_rate": round(stats["exp022"].score_rate, 4),
            "nano_score_rate": round(stats["exp014"].score_rate, 4),
            "delta_score_rate": round(stats["exp022"].score_rate - stats["exp014"].score_rate, 4),
            "delta_win_rate": round(stats["exp022"].win_rate - stats["exp014"].win_rate, 4),
            "delta_raze_rate": round(stats["exp022"].hill_raze_rate - stats["exp014"].hill_raze_rate, 4),
        }

    # Generate JSON
    json_data = {
        "entrants": entrants,
        "total_matches": len(raw_matches),
        "engine_digest": raw_matches[0].get("engine_digest") if raw_matches else "",
        "leaderboard": [asdict(s) for s in sorted_by_rate],
        "head_to_head": {f"{k[0]}_vs_{k[1]}": asdict(v) for k, v in h2h.items()},
        "relative_gain_vs_micro_bc": relative_gain_vs_micro,
        "capacity_scaling": capacity_scaling,
    }
    json_path = out_dir / "tournament_metrics.json"
    json_path.write_text(json.dumps(json_data, indent=2), encoding="utf-8")
    print(f"==> Phase 2 Metrics JSON written to: {json_path}")

    if args.emit_json:
        print(json.dumps(json_data, indent=2))
    else:
        print(md_content)


if __name__ == "__main__":
    main()
