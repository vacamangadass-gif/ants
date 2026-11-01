from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path("/home/host/project/ants")
MODELS_DIR = ROOT / "models"
MAP_PATH = ROOT / "benchmarks" / "maps_2p" / "medium-crossroads-2p-1h.json"
REPLAYS_DIR = ROOT / "replays" / "m2_stress_challenge"
CORPUS_DIR = ROOT / "replays" / "tournament_p3_9map"
CLI = os.environ.get("TINYBRAINS") or shutil.which("tinybrains") or "/home/linuxbrew/.linuxbrew/bin/tinybrains"
REGISTRY = ROOT / "baselines" / "games.toml"
EXTRACT_SCRIPT = ROOT / "benchmarks" / "extract_frames.mjs"
ENGINE_DIGEST = "sha256:819166e7918117a1cbbd67dc042e7368a076a2dd45cea5122683983912d591e1"
MAX_ALLOWED_US = 700_000
ALLOWED_REASONS = {"turn_limit", "lone_survivor", "extermination", "rank_stabilized", "domination", "idle_food"}

ALL_16_MODELS = [f"exp031_iter{i}" for i in range(1, 9)] + [f"exp033_iter{i}" for i in range(1, 9)]


@pytest.fixture(scope="module")
def challenge_replays_dir() -> Path:
    REPLAYS_DIR.mkdir(parents=True, exist_ok=True)
    return REPLAYS_DIR


class TestModelIntegrityAndUniqueness:
    # 验证 16 个模型构件完整性与字节体积门禁限制
    @pytest.mark.parametrize("model_name", ALL_16_MODELS)
    def test_model_directory_artifacts(self, model_name: str):
        model_dir = MODELS_DIR / model_name
        assert model_dir.is_dir(), f"模型目录不存在: {model_dir}"
        for filename in ("model.onnx", "manifest.json", "metrics.json", "card.md"):
            artifact = model_dir / filename
            assert artifact.is_file(), f"构件文件缺失: {artifact}"
            assert artifact.stat().st_size > 0, f"构件文件为空: {artifact}"

        onnx_size = (model_dir / "model.onnx").stat().st_size
        if model_name.startswith("exp031"):
            assert onnx_size < 16_384, f"Nano 模型体积超限: {onnx_size} 字节"
        else:
            assert onnx_size < 131_072, f"Micro 模型体积超限: {onnx_size} 字节"

    # 验证 16 个模型权重哈希值彼此独立唯一无克隆
    def test_all_16_models_sha256_uniqueness(self):
        hashes = {}
        for m in ALL_16_MODELS:
            onnx_path = MODELS_DIR / m / "model.onnx"
            content = onnx_path.read_bytes()
            sha = hashlib.sha256(content).hexdigest()
            assert sha not in hashes, f"发现重复模型权重: {m} 与 {hashes[sha]} 哈希相同 ({sha})"
            hashes[sha] = m
        assert len(hashes) == 16, f"预期 16 个独立哈希值，实测 {len(hashes)}"


class TestCrossroadsLiveCombatLatency:
    # 在代表性复杂地图 medium-crossroads-2p-1h 上组织全部 16 个模型实盘对战并校验时延门禁
    def test_live_play_crossroads_all_16_models(self, challenge_replays_dir: Path):
        assert MAP_PATH.is_file(), f"地图文件不存在: {MAP_PATH}"

        rows = []
        # 构建 8 组迭代对照双向对局，覆盖 16 个模型在 Seat 0 与 Seat 1 的双向实盘对弈
        for i in range(1, 9):
            m_a = f"exp031_iter{i}"
            m_b = f"exp033_iter{i}"
            dir_a = MODELS_DIR / m_a
            dir_b = MODELS_DIR / m_b

            # 正向对局
            mid_fwd = f"stress-{m_a}-vs-{m_b}-fwd"
            rows.append({
                "id": mid_fwd,
                "seed": 95000 + i * 2,
                "map": str(MAP_PATH.resolve()),
                "seat_count": 2,
                "seats": [
                    {"seat": 0, "label": m_a, "weights": str((dir_a / "model.onnx").resolve()), "manifest": str((dir_a / "manifest.json").resolve())},
                    {"seat": 1, "label": m_b, "weights": str((dir_b / "model.onnx").resolve()), "manifest": str((dir_b / "manifest.json").resolve())},
                ]
            })

            # 反向对局
            mid_rev = f"stress-{m_b}-vs-{m_a}-rev"
            rows.append({
                "id": mid_rev,
                "seed": 95000 + i * 2 + 1,
                "map": str(MAP_PATH.resolve()),
                "seat_count": 2,
                "seats": [
                    {"seat": 0, "label": m_b, "weights": str((dir_b / "model.onnx").resolve()), "manifest": str((dir_b / "manifest.json").resolve())},
                    {"seat": 1, "label": m_a, "weights": str((dir_a / "model.onnx").resolve()), "manifest": str((dir_a / "manifest.json").resolve())},
                ]
            })

        batch_file = challenge_replays_dir / "_batch_stress.json"
        batch_file.write_text(json.dumps({
            "game": "ants",
            "vars": {"max_turns": 300},
            "rows": rows
        }))

        env = os.environ.copy()
        env["TINYBRAINS_REGISTRY"] = str(REGISTRY.resolve())

        proc = subprocess.run(
            [CLI, str(batch_file.resolve()), "--out", str(challenge_replays_dir.resolve())],
            capture_output=True,
            text=True,
            env=env
        )
        assert proc.returncode == 0, f"tinybrains 实盘对弈执行失败: {proc.stderr or proc.stdout}"
        batch_file.unlink(missing_ok=True)

        model_peak_latencies = {m: 0 for m in ALL_16_MODELS}
        model_play_counts = {m: 0 for m in ALL_16_MODELS}

        for row in rows:
            mid = row["id"]
            replay_file = challenge_replays_dir / f"{mid}.json"
            assert replay_file.is_file(), f"实盘对弈回放文件未生成: {replay_file}"

            rep = json.loads(replay_file.read_text())
            assert rep.get("engine_digest") == ENGINE_DIGEST, f"引擎哈希值不一致: {rep.get('engine_digest')}"
            assert rep.get("reason") in ALLOWED_REASONS, f"未知结束原因: {rep.get('reason')}"

            seats = rep.get("seats", [])
            assert len(seats) == 2, f"席位数量异常: {len(seats)}"

            for s in seats:
                label = s["label"]
                lat = s.get("infer_us_max", 0)
                model_play_counts[label] += 1
                if lat > model_peak_latencies[label]:
                    model_peak_latencies[label] = lat

                assert lat <= MAX_ALLOWED_US, (
                    f"模型 {label} 实盘推理峰值时延超限 ({lat} 微秒 > {MAX_ALLOWED_US} 微秒)"
                )

        # 验证每个模型均参与了实盘对战并记录推理时延数值
        for m in ALL_16_MODELS:
            assert model_play_counts[m] >= 2, f"模型 {m} 参与对战次数不足: {model_play_counts[m]}"
            peak_us = model_peak_latencies[m]
            print(f"[实盘对战时延检验] 模型: {m:15} 峰值时延: {peak_us:6} 微秒 ({peak_us / 1000.0:6.2f} 毫秒) 门禁通过 (<700毫秒)")


class TestExtractFramesExtremeReplaysRobustness:
    # 检验极端 300 回合满轮对局与 10 回合极速拆穴对局逐帧指标抽取脚本鲁棒性
    def test_extract_frames_extreme_cases(self):
        long_match = CORPUS_DIR / "exp031_iter3-vs-exp033_iter1-medium-valley-2p-1h-2-0.json"
        fast_raze_match = CORPUS_DIR / "exp033_iter1-vs-exp031_iter2-tiny-colonnade-2p-1h-6-1.json"
        fast_combat_match = CORPUS_DIR / "exp031_iter2-vs-exp033_iter5-medium-labyrinth-2p-1h-1-0.json"

        assert long_match.is_file(), f"300回合满轮样本缺失: {long_match}"
        assert fast_raze_match.is_file(), f"10回合极速拆穴样本缺失: {fast_raze_match}"
        assert fast_combat_match.is_file(), f"4回合极速阵亡样本缺失: {fast_combat_match}"

        cmd = [
            "node",
            str(EXTRACT_SCRIPT.resolve()),
            str(long_match.resolve()),
            str(fast_raze_match.resolve()),
            str(fast_combat_match.resolve())
        ]

        proc = subprocess.run(cmd, capture_output=True, text=True)
        assert proc.returncode == 0, f"extract_frames.mjs 执行崩溃: {proc.stderr}"

        results = json.loads(proc.stdout)
        assert len(results) == 3, f"预期 3 组抽取结果，实测 {len(results)}"
        res_by_path = {Path(r["file"]).resolve(): r for r in results}

        # 1. 验证 300 回合极限满轮对局
        r_long = res_by_path[long_match.resolve()]
        assert r_long["turns"] == 300
        assert r_long["reason"] == "turn_limit"
        assert len(r_long["seats"]) == 2
        for s in r_long["seats"]:
            assert s["total_orders"] > 1000, f"300 回合指令总数过低: {s['total_orders']}"
            assert s["blocked_orders"] <= s["total_orders"]
            assert 0.0 <= s["blocked_rate"] <= 1.0
            assert s["collisions"] >= 0
            assert s["combat_deaths"] >= 0
            assert s["ants_spawned"] > 50
            assert s["infer_us_max"] <= MAX_ALLOWED_US

        # 2. 验证 10 回合极速拆穴对局
        r_fast = res_by_path[fast_raze_match.resolve()]
        assert r_fast["turns"] == 10
        assert r_fast["reason"] == "lone_survivor"
        assert len(r_fast["razed_hills"]) == 1
        raze_event = r_fast["razed_hills"][0]
        assert raze_event["turn"] == 10
        assert raze_event["owner"] == 0
        assert raze_event["razer"] == 1

        s0 = r_fast["seats"][0]
        s1 = r_fast["seats"][1]
        assert s0["hills_razed"] == 0
        assert s0["first_hill_destroy_turn"] is None
        assert s1["hills_razed"] == 1
        assert s1["first_hill_destroy_turn"] == 10
        assert s1["won"] is True
        assert s0["lost"] is True
        assert s0["total_orders"] > 0
        assert s1["total_orders"] > 0

        # 3. 验证 4 回合极速阵亡对局
        r_combat = res_by_path[fast_combat_match.resolve()]
        assert r_combat["turns"] == 4
        assert len(r_combat["seats"]) == 2
        for s in r_combat["seats"]:
            assert s["blocked_orders"] <= s["total_orders"]
            assert 0.0 <= s["blocked_rate"] <= 1.0
            assert s["infer_us_max"] <= MAX_ALLOWED_US

    # 检验全部 5 种对局终止类型分层样本在 extract_frames.mjs 中的解析稳定性
    def test_extract_frames_stratified_reasons(self):
        samples = [
            CORPUS_DIR / "exp033_iter3-vs-exp033_iter4-small-islands-2p-1h-4-0.json",   # rank_stabilized
            CORPUS_DIR / "exp033_iter4-vs-exp033_iter1-medium-labyrinth-2p-1h-1-1.json", # lone_survivor
            CORPUS_DIR / "exp031_iter3-vs-exp033_iter1-medium-valley-2p-1h-2-0.json",   # turn_limit
            CORPUS_DIR / "exp031_iter1-vs-exp033_iter4-medium-labyrinth-2p-1h-1-0.json", # idle_food
            CORPUS_DIR / "exp031_iter3-vs-exp031_iter4-medium-labyrinth-2p-1h-1-0.json", # extermination
        ]
        for p in samples:
            assert p.is_file(), f"样本缺失: {p}"

        cmd = ["node", str(EXTRACT_SCRIPT.resolve())] + [str(p.resolve()) for p in samples]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        assert proc.returncode == 0, f"extract_frames.mjs 执行失败: {proc.stderr}"

        results = json.loads(proc.stdout)
        assert len(results) == len(samples), f"预期 {len(samples)} 组结果，实测 {len(results)}"
        observed_reasons = {r["reason"] for r in results}
        assert observed_reasons == {"rank_stabilized", "lone_survivor", "turn_limit", "idle_food", "extermination"}

        for r in results:
            assert len(r["seats"]) == 2
            for s in r["seats"]:
                assert s["blocked_orders"] <= s["total_orders"]
                assert 0.0 <= s["blocked_rate"] <= 1.0
                assert s["infer_us_max"] <= MAX_ALLOWED_US

    # 检验 extract_frames.mjs 直接读取目录并解析新生成的 16 场实盘 Crossroads 回放
    def test_extract_frames_live_crossroads_dir(self, challenge_replays_dir: Path):
        cmd = ["node", str(EXTRACT_SCRIPT.resolve()), str(challenge_replays_dir.resolve())]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        assert proc.returncode == 0, f"抽取目录回放执行失败: {proc.stderr}"

        results = json.loads(proc.stdout)
        # 16 场实盘对弈回放
        assert len(results) == 16, f"预期解析 16 场对局，实测 {len(results)}"
        for r in results:
            assert r["map_id"] == "medium-crossroads-2p-1h"
            assert r["engine_digest"] == ENGINE_DIGEST
            assert len(r["seats"]) == 2
            for s in r["seats"]:
                assert s["blocked_orders"] <= s["total_orders"]
                assert 0.0 <= s["blocked_rate"] <= 1.0
                assert s["infer_us_max"] <= MAX_ALLOWED_US


class TestTournamentCorpusLatencyStress:
    # 针对全量 2,160 场循环赛对局进行时延门禁全局扫描
    def test_full_corpus_latency_and_invariants(self):
        assert CORPUS_DIR.is_dir(), f"循环赛回放目录不存在: {CORPUS_DIR}"
        files = sorted([f for f in CORPUS_DIR.glob("*.json") if not f.name.startswith("_batch_")])
        assert len(files) == 2160, f"预期 2160 场回放，实测 {len(files)}"

        max_latency_per_entrant = {m: 0 for m in ALL_16_MODELS}
        global_peak_latency = 0

        for f in files:
            rep = json.loads(f.read_text())
            assert rep.get("engine_digest") == ENGINE_DIGEST, f"引擎散列值异常: {f.name}"
            for s in rep.get("seats", []):
                label = s["label"]
                lat = s.get("infer_us_max", 0)
                if lat > global_peak_latency:
                    global_peak_latency = lat
                if label in max_latency_per_entrant:
                    if lat > max_latency_per_entrant[label]:
                        max_latency_per_entrant[label] = lat
                assert lat <= MAX_ALLOWED_US, (
                    f"回放 {f.name} 中模型 {label} 时延超限: {lat} 微秒"
                )

        assert global_peak_latency <= MAX_ALLOWED_US, f"全局峰值时延超限: {global_peak_latency} 微秒"
        assert all(max_latency_per_entrant[m] > 0 for m in ALL_16_MODELS), "存在未统计到时延的模型"

        print(f"\n[全量对战回放时延扫描完毕] 扫描场次: {len(files)}, 全局峰值时延: {global_peak_latency} 微秒 ({global_peak_latency / 1000.0:.2f} 毫秒)")
