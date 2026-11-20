from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
TOOLS_DIR = REPO_ROOT / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import eval_live_suite
import run_exp083_tournament
import run_multiseat_tournament

CLI = os.environ.get("TINYBRAINS") or shutil.which("tinybrains") or "/home/linuxbrew/.linuxbrew/bin/tinybrains"
EXTRACTOR = REPO_ROOT / "benchmarks" / "extract_frames.mjs"


def _resolve_dist_dir() -> Path:
    local_dist = REPO_ROOT / "dist"
    if (local_dist / "viz" / "engine.js").is_file() and (local_dist / "tb-ants.wasm").is_file():
        return local_dist
    fallback = Path("/home/host/project/ants/dist")
    if (fallback / "viz" / "engine.js").is_file() and (fallback / "tb-ants.wasm").is_file():
        return fallback
    raise FileNotFoundError("未找到已构建的 dist/tb-ants.wasm 与 dist/viz/engine.js")


def _resolve_registry() -> str:
    if (REPO_ROOT / "dist" / "tb-ants.wasm").is_file():
        return str((REPO_ROOT / "baselines" / "games.toml").resolve())
    fallback = Path("/home/host/project/ants/baselines/games.toml")
    if fallback.is_file():
        return str(fallback.resolve())
    raise FileNotFoundError("未找到 baselines/games.toml")


def _resolve_models_root() -> Path:
    local_models = REPO_ROOT / "models"
    if (local_models / "exp033" / "model.onnx").is_file():
        return local_models
    fallback = Path("/home/host/project/ants/models")
    if (fallback / "exp033" / "model.onnx").is_file():
        return fallback
    raise FileNotFoundError("未找到 models/exp033/model.onnx")


def _seat(models_root: Path, idx: int, name: str) -> dict:
    d = (models_root / name).resolve()
    return {
        "seat": idx,
        "label": name,
        "weights": str(d / "model.onnx"),
        "manifest": str(d / "manifest.json"),
    }


def _run_real_matches(rows: list[dict], out_dir: Path, max_turns: int) -> list[dict]:
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    by_sc: dict[int, list[dict]] = {}
    for r in rows:
        by_sc.setdefault(int(r["seat_count"]), []).append(r)

    env = os.environ.copy()
    env["TINYBRAINS_REGISTRY"] = _resolve_registry()
    for sc, sc_rows in sorted(by_sc.items()):
        bf = out_dir / f"_batch_{sc}p.json"
        bf.write_text(json.dumps({"game": "ants", "vars": {"max_turns": max_turns}, "rows": sc_rows}), encoding="utf-8")
        res = subprocess.run([CLI, str(bf), "--out", str(out_dir)], cwd=str(REPO_ROOT), capture_output=True, text=True, env=env)
        bf.unlink(missing_ok=True)
        assert res.returncode == 0, f"tinybrains 执行失败: {res.stderr or res.stdout}"

    node_env = os.environ.copy()
    node_env["ANTS_DIST_DIR"] = str(_resolve_dist_dir())
    ext = subprocess.run(
        ["node", str(EXTRACTOR), str(out_dir)],
        cwd=str(REPO_ROOT / "benchmarks"),
        capture_output=True,
        text=True,
        env=node_env,
    )
    assert ext.returncode == 0, f"extract_frames.mjs 失败: {ext.stderr or ext.stdout}"
    return json.loads(ext.stdout)


@pytest.fixture(scope="module")
def real_multiseat_records() -> list[dict]:
    models_root = _resolve_models_root()
    scratch_dir = REPO_ROOT / "runs" / "test_eval_harness_fixes" / "real_replays"
    rows = [
        {
            "id": "t-2p-1h",
            "seed": 77101,
            "map": str((REPO_ROOT / "maps" / "basic-tiny-2p.json").resolve()),
            "seat_count": 2,
            "seats": [_seat(models_root, 0, "exp085"), _seat(models_root, 1, "exp033")],
        },
        {
            "id": "t-4p-2h",
            "seed": 77201,
            "map": str((REPO_ROOT / "maps" / "basic-medium-4p.json").resolve()),
            "seat_count": 4,
            "seats": [
                _seat(models_root, 0, "exp085"),
                _seat(models_root, 1, "exp033"),
                _seat(models_root, 2, "exp021"),
                _seat(models_root, 3, "exp011"),
            ],
        },
        {
            "id": "t-6p-1h",
            "seed": 77301,
            "map": str((REPO_ROOT / "maps" / "basic-large-6p.json").resolve()),
            "seat_count": 6,
            "seats": [
                _seat(models_root, 0, "exp085"),
                _seat(models_root, 1, "exp033"),
                _seat(models_root, 2, "exp021"),
                _seat(models_root, 3, "exp011"),
                _seat(models_root, 4, "exp083"),
                _seat(models_root, 5, "exp033"),
            ],
        },
    ]
    records = _run_real_matches(rows, scratch_dir, max_turns=90)
    yield records
    if scratch_dir.parent.exists():
        shutil.rmtree(scratch_dir.parent, ignore_errors=True)


class TestExtractFramesMultiSeatAndMultiHill:
    # 验证 extract_frames.mjs 彻底消除 opp = 1 - s 与 initialAnts = 1 硬编码缺陷
    def test_2p_4p_6p_rank_pct_and_initial_ants_accuracy(self, real_multiseat_records: list[dict]) -> None:
        by_id = {r["match_id"]: r for r in real_multiseat_records}
        assert set(by_id.keys()) == {"t-2p-1h", "t-4p-2h", "t-6p-1h"}

        # 2p-1h: 初始每席 1 蚁穴 1 蚂蚁，rank_pct 严格退化为 1.0/0.5/0.0
        r2 = by_id["t-2p-1h"]
        assert r2["players"] == 2
        assert r2["hills_per_seat"] == 1
        assert abs(sum(s["rank_pct"] for s in r2["seats"]) - 1.0) < 1e-9
        for s in r2["seats"]:
            assert s["initial_ants"] == 1
            assert s["initial_hills"] == 1
            assert s["rank_pct"] in (0.0, 0.5, 1.0)
            assert s["points"] == s["rank_pct"]

        # 4p-2h: 初始每席 2 蚁穴 2 蚂蚁，绝不再错用 initialAnts=1 虚增 ants_spawned
        r4 = by_id["t-4p-2h"]
        assert r4["players"] == 4
        assert r4["hills_per_seat"] == 2
        assert abs(sum(s["rank_pct"] for s in r4["seats"]) - 2.0) < 1e-9
        for s in r4["seats"]:
            assert s["initial_ants"] == 2
            assert s["initial_hills"] == 2
            assert isinstance(s["won"], bool) and isinstance(s["drawn"], bool) and isinstance(s["lost"], bool)
            assert int(s["won"]) + int(s["drawn"]) + int(s["lost"]) == 1
            assert 0.0 <= s["rank_pct"] <= 1.0

        # 6p-1h: 第 2..5 号席位同样满足精确多席位名次与归一化总和 N/2
        r6 = by_id["t-6p-1h"]
        assert r6["players"] == 6
        assert r6["hills_per_seat"] == 1
        assert abs(sum(s["rank_pct"] for s in r6["seats"]) - 3.0) < 1e-9
        for s in r6["seats"]:
            assert s["initial_ants"] == 1
            assert int(s["won"]) + int(s["drawn"]) + int(s["lost"]) == 1

    # 验证行为遥测计数守恒与 exp085 在 4p-2h 上的己方蚁穴驻留 HOLD 信号捕获
    def test_behavioral_counters_and_own_hill_hold_detection(self, real_multiseat_records: list[dict]) -> None:
        for rec in real_multiseat_records:
            for s in rec["seats"]:
                assert 0 <= s["blocked_orders"] <= s["total_orders"]
                assert s["all_orders"] == s["total_orders"] + s["hold_orders"]
                assert 0 <= s["on_own_hill_hold_orders"] <= s["hold_orders"]
                assert 0 <= s["on_own_hill_hold_orders"] <= s["on_own_hill_steps"]
                assert 0.0 <= s["on_own_hill_hold_rate"] <= 1.0
                assert 0.0 <= s["hold_rate"] <= 1.0
                assert 0.0 <= s["idle3_rate"] <= 1.0
                assert 0.0 <= s["osc2_rate"] <= 1.0
                assert s["kd_ratio"] >= 0.0
                assert s["strikes"] == 0
                assert s["forfeit"] is False

        r4 = next(r for r in real_multiseat_records if r["match_id"] == "t-4p-2h")
        s85 = next(s for s in r4["seats"] if s["name"] == "exp085")
        s33 = next(s for s in r4["seats"] if s["name"] == "exp033")
        assert s85["on_own_hill_hold_rate"] > s33["on_own_hill_hold_rate"]
        assert s85["hold_rate"] > s33["hold_rate"]

    # 验证更新后的 extract_frames.mjs 对历史 2p 回放语料库保持 100% 向后兼容
    def test_backward_compatibility_with_historical_2p_corpus(self) -> None:
        corpus_dir = Path("/home/host/project/ants/replays/tournament_p3_9map")
        long_match = corpus_dir / "exp031_iter3-vs-exp033_iter1-medium-valley-2p-1h-2-0.json"
        fast_raze = corpus_dir / "exp033_iter1-vs-exp031_iter2-tiny-colonnade-2p-1h-6-1.json"
        if not (long_match.is_file() and fast_raze.is_file()):
            return
        node_env = os.environ.copy()
        node_env["ANTS_DIST_DIR"] = str(_resolve_dist_dir())
        proc = subprocess.run(
            ["node", str(EXTRACTOR.resolve()), str(long_match), str(fast_raze)],
            capture_output=True,
            text=True,
            env=node_env,
        )
        assert proc.returncode == 0, f"extract_frames.mjs 解析历史回放失败: {proc.stderr}"
        results = {Path(r["file"]).name: r for r in json.loads(proc.stdout)}
        r_fast = results[fast_raze.name]
        assert r_fast["turns"] == 10
        assert r_fast["seats"][0]["won"] is False and r_fast["seats"][0]["lost"] is True
        assert r_fast["seats"][1]["won"] is True and r_fast["seats"][1]["lost"] is False
        assert r_fast["seats"][1]["points"] == 1.0 and r_fast["seats"][0]["points"] == 0.0
        assert r_fast["seats"][1]["hills_razed"] == 1 and r_fast["seats"][0]["hills_lost"] == 1


class TestRunMultiseatTournamentFixes:
    # 验证修复 s.get("orders") 键名缺陷后，真实阻塞率精确等于 blocked_orders / total_orders
    def test_blocked_order_rate_reads_total_orders(self, real_multiseat_records: list[dict]) -> None:
        # 确认原始提取记录中不含错误的 "orders" 键
        for rec in real_multiseat_records:
            for s in rec["seats"]:
                assert "orders" not in s
                assert "total_orders" in s

        report = run_multiseat_tournament.analyze_match_records(real_multiseat_records)
        # 至少有一个模型在真实对局中出现受阻移动，阻塞率绝不会全员虚假归零
        any_blocked = False
        for m, m_stats in report["models"].items():
            expected_blk = m_stats["blocked_orders"] / max(m_stats["total_orders"], 1)
            assert abs(m_stats["blocked_order_rate"] - round(expected_blk, 6)) < 1e-6
            if m_stats["blocked_orders"] > 0:
                assert m_stats["blocked_order_rate"] > 0.0
                any_blocked = True
            assert "rank_pct" in m_stats
            assert "hold_rate" in m_stats
            assert "on_own_hill_hold_rate" in m_stats
            assert "idle3_rate" in m_stats
            assert "kd_ratio" in m_stats
            assert "zero_score_rate" in m_stats
        assert any_blocked, "真实 2p/4p/6p 对局应观测到非零受阻移动以验证修复"

    # 验证邻接均衡排列消除 basic-xlarge-8p 的固定循环移位邻接偏差
    def test_neighbor_balanced_lineups_eliminate_8p_bias(self) -> None:
        models_root = _resolve_models_root()
        map_8p = REPO_ROOT / "maps" / "basic-xlarge-8p.json"
        offsets = run_multiseat_tournament.compute_board_neighbor_offsets(map_8p)
        assert offsets == [1, 3, 5, 7], f"basic-xlarge-8p 最近邻偏移应为 [1, 3, 5, 7]，实际为 {offsets}"

        rows = run_multiseat_tournament.build_multiseat_tournament_rows(
            rounds=4, base_seed=85000, models_root=models_root
        )
        rows_8p = [r for r in rows if r["seat_count"] == 8]
        assert len(rows_8p) == 32

        # 验证每个模型在 8 个几何席位上出现次数完全一致（消除几何位置偏差）
        seat_visits: dict[str, Counter[int]] = {}
        weak_neighbors: dict[str, list[int]] = {"exp085": [0, 0], "exp033": [0, 0]}
        for r in rows_8p:
            labels = [s["label"] for s in r["seats"]]
            for idx, nm in enumerate(labels):
                seat_visits.setdefault(nm, Counter())[idx] += 1
                if nm in weak_neighbors:
                    for off in offsets:
                        nb = labels[(idx + off) % 8]
                        weak_neighbors[nm][1] += 1
                        if nb in ("exp011", "exp083"):
                            weak_neighbors[nm][0] += 1

        for nm, cnts in seat_visits.items():
            assert len(cnts) == 8
            assert len(set(cnts.values())) == 1, f"模型 {nm} 几何席位分布不均匀: {cnts}"

        share_085 = weak_neighbors["exp085"][0] / weak_neighbors["exp085"][1]
        share_033 = weak_neighbors["exp033"][0] / weak_neighbors["exp033"][1]
        # 旧固定循环移位下 share_033 = 0.00, share_085 = 0.50 (差值 0.50)；修复后两者均应接近理想值 2/7 ≈ 0.286
        assert share_033 > 0.18, f"exp033 弱对手近邻暴露过低: {share_033:.3f}"
        assert share_085 > 0.18, f"exp085 弱对手近邻暴露过低: {share_085:.3f}"
        assert abs(share_085 - share_033) < 0.08, (
            f"exp085 与 exp033 弱对手近邻占比失衡: 085={share_085:.3f} vs 033={share_033:.3f}"
        )

    # 验证同时汇报等权棋盘均值与席位加权均值，阻止单图加权掩盖多图失利
    def test_unweighted_map_summary_detects_simpsons_paradox(self, real_multiseat_records: list[dict]) -> None:
        # 用 1 份 2p + 1 份 4p + 4 份 6p 构造非均匀席位权重输入（全部基于真实回放记录）
        r2 = next(r for r in real_multiseat_records if r["match_id"] == "t-2p-1h")
        r4 = next(r for r in real_multiseat_records if r["match_id"] == "t-4p-2h")
        r6 = next(r for r in real_multiseat_records if r["match_id"] == "t-6p-1h")
        skewed = [r2, r4, r6, r6, r6, r6]

        report = run_multiseat_tournament.analyze_match_records(skewed)
        assert "unweighted_map_summary" in report
        assert "aggregation_audit" in report
        u85 = report["unweighted_map_summary"]["exp085"]
        assert u85["boards_played"] == 3
        expected_unweighted_rp = round(
            (
                report["per_board"]["2p"]["exp085"]["rank_pct"]
                + report["per_board"]["4p"]["exp085"]["rank_pct"]
                + report["per_board"]["6p"]["exp085"]["rank_pct"]
            ) / 3.0,
            2,
        )
        assert abs(u85["unweighted_rank_pct"] - expected_unweighted_rp) < 1e-6
        assert "exp085_vs_exp033" in report["aggregation_audit"]["per_board_rank_comparison"]


class TestRunExp083TournamentFixes:
    # 验证 food_gathered 统计真实繁衍蚂蚁数 ants_spawned，彻底与蚁穴得分 score_for 解耦
    def test_food_gathered_decoupled_from_hill_score(self, real_multiseat_records: list[dict]) -> None:
        r2 = next(r for r in real_multiseat_records if r["match_id"] == "t-2p-1h")
        s85 = next(s for s in r2["seats"] if s["name"] == "exp085")
        report = run_exp083_tournament.aggregate_exp083_metrics([r2], target_name="exp085", wall_clock=1.5)

        ov = report["overall"]
        assert ov["food_gathered"] == s85["ants_spawned"]
        assert ov["ants_spawned"] == s85["ants_spawned"]
        assert ov["score_for"] == s85["score"]
        assert ov["hill_score_for"] == s85["score"]
        assert ov["food_gathered"] != ov["score_for"], "真实对局中繁衍蚂蚁数应与蚁穴得分明显区分"
        expected_blk_pct = round(s85["blocked_orders"] / max(s85["total_orders"], 1) * 100.0, 4)
        assert abs(ov["blocked_order_rate_pct"] - expected_blk_pct) < 1e-6
        assert abs(report["head_to_head"]["exp033"]["blocked_order_rate_pct"] - expected_blk_pct) < 1e-6


class TestEvalLiveSuite:
    # 验证非内战配对调度：零同模型内战、同 trial_key 严格一致的棋盘/种子/席位/背景对手
    def test_paired_schedule_non_self_and_strict_alignment(self) -> None:
        models_root = _resolve_models_root()
        base = eval_live_suite.load_model_spec(models_root / "exp033")
        cand = eval_live_suite.load_model_spec(models_root / "exp085")
        opps = eval_live_suite.resolve_opponents(
            opponent_names=["exp011", "exp021", "exp083"],
            models_root=models_root,
            starter_root=None,
            exclude_names={base.name, cand.name},
        )
        boards = eval_live_suite.load_board_catalog(REPO_ROOT / "maps", cache_dir=REPO_ROOT / "runs" / "_unused")
        boards_246 = [b for b in boards if b.players in (2, 4, 6)]
        rows, trial_meta, _ = eval_live_suite.build_paired_live_schedule(
            boards=boards_246,
            evaluated_models=[base, cand],
            opponents=opps,
            seed=2026,
            trials_2p_per_opp=1,
            trials_multiseat=2,
        )

        rows_by_id = {r["id"]: r for r in rows}
        by_trial: dict[str, dict[str, dict]] = {}
        for mt in trial_meta:
            row = rows_by_id[mt["match_id"]]
            labels = [s["label"] for s in row["seats"]]
            # 确认同一场对局中绝不出现被测模型之间的内战或自博弈
            assert labels.count(mt["focal_model"]) == 1
            other_eval = cand.name if mt["focal_model"] == base.name else base.name
            assert other_eval not in labels
            by_trial.setdefault(mt["trial_key"], {})[mt["focal_model"]] = (mt, row)

        for tk, pair in by_trial.items():
            assert set(pair.keys()) == {base.name, cand.name}
            (mt_b, row_b), (mt_c, row_c) = pair[base.name], pair[cand.name]
            assert row_b["map"] == row_c["map"]
            assert row_b["seed"] == row_c["seed"]
            assert mt_b["focal_seat"] == mt_c["focal_seat"]
            fs = mt_b["focal_seat"]
            bg_b = [s["label"] for i, s in enumerate(row_b["seats"]) if i != fs]
            bg_c = [s["label"] for i, s in enumerate(row_c["seats"]) if i != fs]
            assert bg_b == bg_c, f"trial {tk} 背景对手阵容不一致"

    # 验证线上 hive-minds 席位加权公式与 PASS/FAIL 验收门禁裁决逻辑
    def test_weighted_rank_and_acceptance_gates(self) -> None:
        by_seat = {
            "2p": {"rank_pct": 80.0},
            "4p": {"rank_pct": 70.0},
            "6p": {"rank_pct": 60.0},
        }
        weighted, weights = eval_live_suite.compute_weighted_live_rank_pct(by_seat)
        expected = round(0.522 * 80.0 + 0.370 * 70.0 + 0.108 * 60.0, 2)
        assert weighted == expected
        assert weights == {"2p": 0.522, "4p": 0.37, "6p": 0.108}

        base_rep = {
            "weighted_live_rank_pct": 80.0,
            "overall": {"forfeits": 0, "strikes": 0, "max_infer_ms": 180.0, "on_own_hill_hold_rate": 0.01, "hold_rate": 0.03},
            "by_seat": {"2p": {"rank_pct": 82.0}, "4p": {"rank_pct": 79.0}, "6p": {"rank_pct": 74.0}},
            "by_hills": {"1h": {"rank_pct": 80.0}, "2h": {"rank_pct": 82.0}, "3h": {"rank_pct": 75.0}},
        }
        good_cand = {
            "weighted_live_rank_pct": 86.0,
            "overall": {"forfeits": 0, "strikes": 0, "max_infer_ms": 220.0, "on_own_hill_hold_rate": 0.005, "hold_rate": 0.02},
            "by_seat": {"2p": {"rank_pct": 87.0}, "4p": {"rank_pct": 86.0}, "6p": {"rank_pct": 81.0}},
            "by_hills": {"1h": {"rank_pct": 86.0}, "2h": {"rank_pct": 87.0}, "3h": {"rank_pct": 82.0}},
        }
        cfg = eval_live_suite.GateConfig(turn_ms=500.0)
        res_pass = eval_live_suite.evaluate_acceptance_gates(good_cand, base_rep, cfg)
        assert res_pass["verdict"] == "PASS"
        assert res_pass["failed_gates"] == []

        bad_cand = {
            "weighted_live_rank_pct": 75.0,
            "overall": {"forfeits": 1, "strikes": 2, "max_infer_ms": 540.0, "on_own_hill_hold_rate": 0.65, "hold_rate": 0.14},
            "by_seat": {"2p": {"rank_pct": 78.0}, "4p": {"rank_pct": 74.0}, "6p": {"rank_pct": 60.0}},
            "by_hills": {"1h": {"rank_pct": 84.0}, "2h": {"rank_pct": 60.0}, "3h": {"rank_pct": 65.0}},
        }
        res_fail = eval_live_suite.evaluate_acceptance_gates(bad_cand, base_rep, cfg)
        assert res_fail["verdict"] == "FAIL"
        assert set(res_fail["failed_gates"]) == {
            "zero_forfeits_and_strikes",
            "latency_within_turn_ms",
            "weighted_rank_beats_baseline",
            "no_seat_regression",
            "multi_hill_no_collapse",
            "own_hill_hold_healthy",
            "overall_hold_healthy",
        }

    # 真实端到端冒烟验证：运行 eval_live_suite.py 对比真实 exp085 与 exp033，确认完整报告与门禁检出退化
    def test_end_to_end_real_smoke_run(self) -> None:
        models_root = _resolve_models_root()
        smoke_root = REPO_ROOT / "runs" / "test_eval_harness_fixes" / "live_suite_smoke"
        maps_dir = smoke_root / "maps"
        maps_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO_ROOT / "maps" / "basic-tiny-2p.json", maps_dir / "basic-tiny-2p.json")
        shutil.copy2(REPO_ROOT / "maps" / "basic-medium-4p.json", maps_dir / "basic-medium-4p.json")

        out_json = smoke_root / "report.json"
        replays_dir = smoke_root / "replays"
        try:
            report = eval_live_suite.main([
                "--candidates", str(models_root / "exp085"),
                "--baseline", str(models_root / "exp033"),
                "--maps", str(maps_dir),
                "--models-root", str(models_root),
                "--opponents", f"exp011={models_root / 'exp011'}", f"exp021={models_root / 'exp021'}",
                "--max-turns", "60",
                "--turn-ms", "500",
                "--trials-2p-per-opp", "1",
                "--trials-multiseat", "1",
                "--include-h2h",
                "--workers", "4",
                "--registry", _resolve_registry(),
                "--dist-dir", str(_resolve_dist_dir()),
                "--replays-dir", str(replays_dir),
                "--out", str(out_json),
            ])
            assert out_json.is_file()
            assert report["boards_evaluated"] == 2
            assert set(report["models"].keys()) == {"exp033", "exp085"}
            assert set(report["models"]["exp085"]["by_seat"].keys()) == {"2p", "4p"}
            assert set(report["models"]["exp085"]["by_hills"].keys()) == {"1h", "2h"}
            assert report["models"]["exp085"]["overall"]["forfeits"] == 0
            assert report["models"]["exp085"]["overall"]["strikes"] == 0
            assert "exp085" in report["direct_h2h_2p"]
            # 真实 exp085 在 4p-2h 上存在己方蚁穴驻留 HOLD 缺陷，应被门禁检出 FAIL
            acc = report["comparisons"]["exp085"]["acceptance"]
            assert acc["verdict"] == "FAIL"
            assert "own_hill_hold_healthy" in acc["failed_gates"]
        finally:
            if smoke_root.exists():
                shutil.rmtree(smoke_root, ignore_errors=True)
