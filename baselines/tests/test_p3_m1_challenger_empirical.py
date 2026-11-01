from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tracemalloc
from pathlib import Path

import numpy as np
import onnx
import pytest
import torch

from tb_baselines import export, nets
from tb_baselines.planes import MEMORY_SOURCES, N_MEMORY, N_PLANES

ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = ROOT / "models"
RUNS_DIR = ROOT / "runs" / "exp033"
BENCH_MAPS = ROOT / "benchmarks" / "maps_2p"
CHALLENGE_DIR = ROOT / "replays" / "m1_empirical_challenge"
REGISTRY = ROOT / "baselines" / "games.toml"
ENGINE_DIGEST = "sha256:819166e7918117a1cbbd67dc042e7368a076a2dd45cea5122683983912d591e1"
MAX_ALLOWED_US = 700_000
LEGAL_ACTIONS = {"N", "E", "S", "W", "-"}

CLI = os.environ.get("TINYBRAINS") or shutil.which("tinybrains") or "/home/linuxbrew/.linuxbrew/bin/tinybrains"


@pytest.fixture(scope="module")
def challenge_output_dir():
    CHALLENGE_DIR.mkdir(parents=True, exist_ok=True)
    return CHALLENGE_DIR


class TestPlatformAdmissionAndIntegrity:
    # 检验交付模型目录结构与基础文件完整性
    def test_model_directory_artifacts(self):
        for exp_name in ("exp033_iter7", "exp033_iter8", "exp033"):
            model_dir = MODELS_DIR / exp_name
            assert model_dir.is_dir(), f"模型目录不存在: {model_dir}"
            for filename in ("model.onnx", "manifest.json", "metrics.json", "card.md"):
                p = model_dir / filename
                assert p.is_file(), f"构件文件缺失: {p}"
                assert p.stat().st_size > 0, f"构件文件为空: {p}"

    # 使用 tinybrains check 执行平台准入门禁与内存往返检验
    @pytest.mark.parametrize("exp_name", ["exp033_iter7", "exp033_iter8", "exp033"])
    def test_tinybrains_check_platform_admission(self, exp_name):
        model_dir = MODELS_DIR / exp_name
        weights = model_dir / "model.onnx"
        manifest = model_dir / "manifest.json"

        cmd = [CLI, "check", str(weights), str(manifest), "--json"]
        env = os.environ.copy()
        env["TINYBRAINS_REGISTRY"] = str(REGISTRY)

        res = subprocess.run(cmd, capture_output=True, text=True, env=env)
        assert res.returncode == 0, f"tinybrains check 异常退出: {res.stderr}"

        data = json.loads(res.stdout)
        assert data.get("ok") is True, f"平台准入判定未通过: {data}"
        assert data.get("engine_digest") == ENGINE_DIGEST, f"引擎摘要散列不一致: {data.get('engine_digest')}"
        assert data.get("size_metric_bytes") <= 131072, f"模型体积超出 128KB 门禁: {data.get('size_metric_bytes')}"
        assert data.get("infer_us_max") <= MAX_ALLOWED_US, f"推理峰值时延超出 700ms 门禁: {data.get('infer_us_max')}"

        mem = data.get("memory", {})
        round_trip = mem.get("round_trip", {})
        assert round_trip.get("failed") == 0, f"记忆往返测试存在失败项: {round_trip}"
        assert round_trip.get("checked", 0) >= 200, f"往返测试用例数量不足: {round_trip}"

        surface = data.get("surface", {})
        assert surface.get("ok") is True, f"算子界面校验未通过: {surface}"
        assert surface.get("op_not_allowed") == [], f"存在非白名单算子: {surface.get('op_not_allowed')}"

        graph_ops = set(data.get("graph", {}).get("operators", []))
        assert "Where" in graph_ops, f"模型计算图中缺少 Where 算子: {graph_ops}"
        assert "Max" not in graph_ops, f"模型计算图中残留 Max 算子: {graph_ops}"

    # 检验 ONNX 图结构与规范版本
    @pytest.mark.parametrize("exp_name", ["exp033_iter7", "exp033_iter8"])
    def test_onnx_model_spec(self, exp_name):
        model_path = MODELS_DIR / exp_name / "model.onnx"
        model = onnx.load(str(model_path))
        onnx.checker.check_model(model)

        opset_imports = {entry.domain: entry.version for entry in model.opset_import}
        default_opset = opset_imports.get("", opset_imports.get("ai.onnx"))
        assert default_opset == 17, f"ONNX opset 必须为 17，实际为: {default_opset}"


@pytest.fixture(scope="module")
def executed_matches(challenge_output_dir):
    # 构建 27 场实战对局（覆盖 tiny, small, medium 尺寸地图与席位互换对弈）
    maps_tiny = ["tiny-open-2p-1h.json", "tiny-corridor-2p-1h.json", "tiny-colonnade-2p-1h.json"]
    maps_small = ["small-arena-2p-1h.json", "small-islands-2p-1h.json", "small-rooms-2p-1h.json"]
    maps_medium = ["medium-crossroads-2p-1h.json", "medium-labyrinth-2p-1h.json", "medium-valley-2p-1h.json"]
    all_maps = maps_tiny + maps_small + maps_medium

    m7 = {
        "label": "exp033_iter7",
        "weights": str((MODELS_DIR / "exp033_iter7" / "model.onnx").resolve()),
        "manifest": str((MODELS_DIR / "exp033_iter7" / "manifest.json").resolve()),
    }
    m8 = {
        "label": "exp033_iter8",
        "weights": str((MODELS_DIR / "exp033_iter8" / "model.onnx").resolve()),
        "manifest": str((MODELS_DIR / "exp033_iter8" / "manifest.json").resolve()),
    }

    rows = []
    match_ids = []
    seed_base = 20000

    for idx, map_file in enumerate(all_maps):
        map_path = str((BENCH_MAPS / map_file).resolve())
        stem = Path(map_file).stem

        # 对弈 1: iter7 (seat 0) vs iter8 (seat 1)
        mid1 = f"challenge-iter7-vs-iter8-{stem}"
        rows.append({
            "id": mid1,
            "seed": seed_base + idx * 3,
            "map": map_path,
            "seat_count": 2,
            "seats": [
                {"seat": 0, "label": m7["label"], "weights": m7["weights"], "manifest": m7["manifest"]},
                {"seat": 1, "label": m8["label"], "weights": m8["weights"], "manifest": m8["manifest"]},
            ]
        })
        match_ids.append(mid1)

        # 对弈 2: iter8 (seat 0) vs iter7 (seat 1)
        mid2 = f"challenge-iter8-vs-iter7-{stem}"
        rows.append({
            "id": mid2,
            "seed": seed_base + idx * 3 + 1,
            "map": map_path,
            "seat_count": 2,
            "seats": [
                {"seat": 0, "label": m8["label"], "weights": m8["weights"], "manifest": m8["manifest"]},
                {"seat": 1, "label": m7["label"], "weights": m7["weights"], "manifest": m7["manifest"]},
            ]
        })
        match_ids.append(mid2)

        # 对弈 3: iter8 (seat 0) vs iter8 (seat 1) 自对弈
        mid3 = f"challenge-iter8-vs-iter8-{stem}"
        rows.append({
            "id": mid3,
            "seed": seed_base + idx * 3 + 2,
            "map": map_path,
            "seat_count": 2,
            "seats": [
                {"seat": 0, "label": m8["label"], "weights": m8["weights"], "manifest": m8["manifest"]},
                {"seat": 1, "label": m8["label"], "weights": m8["weights"], "manifest": m8["manifest"]},
            ]
        })
        match_ids.append(mid3)

    missing_rows = [r for r in rows if not (challenge_output_dir / f"{r['id']}.json").is_file()]
    if missing_rows:
        batch_file = challenge_output_dir / "challenge_batch_27m.json"
        batch_file.write_text(json.dumps({
            "game": "ants",
            "vars": {"max_turns": 200},
            "rows": missing_rows,
        }))

        env = os.environ.copy()
        env["TINYBRAINS_REGISTRY"] = str(REGISTRY)

        res = subprocess.run([CLI, str(batch_file), "--out", str(challenge_output_dir)], capture_output=True, text=True, env=env)
        assert res.returncode == 0, f"实战批次执行失败:\n{res.stderr or res.stdout}"

    # 读取全部重放数据
    replays = []
    for mid in match_ids:
        rp = challenge_output_dir / f"{mid}.json"
        assert rp.is_file(), f"缺少对局重放文件: {rp}"
        replays.append(json.loads(rp.read_text()))

    return replays


class TestDeterministicMatchExecution:
    # 检验 200 回合实战对局稳定性与合法退出原因
    def test_all_matches_completed_cleanly(self, executed_matches):
        assert len(executed_matches) == 27, f"预期完成 27 场对局，实际为: {len(executed_matches)}"
        valid_reasons = {
            "turn_limit",
            "turns_exhausted",
            "lone_survivor",
            "extermination",
            "rank_stabilized",
            "domination",
            "idle_food",
        }

        for replay in executed_matches:
            mid = replay.get("match_id")
            digest = replay.get("engine_digest")
            assert digest == ENGINE_DIGEST, f"对局 {mid} 引擎散列发生偏移: {digest}"

            reason = replay.get("reason")
            assert reason in valid_reasons, f"对局 {mid} 异常终止原因: {reason}"

            turns = replay.get("turns", 0)
            assert 1 <= turns <= 200, f"对局 {mid} 回合数异常: {turns}"

            seats = replay.get("seats", [])
            assert len(seats) == 2, f"对局 {mid} 席位数量异常: {len(seats)}"


class TestLatencyDistribution:
    # 统计单步推理耗时分布并确认严格低于 700 毫秒上限
    def test_single_turn_inference_latency(self, executed_matches):
        max_latencies = []
        all_seat_max_latencies = []

        for replay in executed_matches:
            mid = replay.get("match_id")
            for seat in replay.get("seats", []):
                infer_us_max = seat.get("infer_us_max", 0)
                all_seat_max_latencies.append(infer_us_max)
                assert infer_us_max <= MAX_ALLOWED_US, (
                    f"模型 {seat.get('label')} 在对局 {mid} 中单步推理耗时 {infer_us_max} 微秒超出 700 毫秒上限"
                )

        arr = np.array(all_seat_max_latencies, dtype=np.float64)
        peak_us = float(np.max(arr))
        p99_us = float(np.percentile(arr, 99))
        p95_us = float(np.percentile(arr, 95))
        p50_us = float(np.median(arr))
        mean_us = float(np.mean(arr))

        # 峰值耗时应小于 100 毫秒（100,000 微秒），远低于 700 毫秒硬性门禁
        assert peak_us < 100_000, f"峰值推理时延过高: {peak_us} 微秒"
        assert p99_us < 80_000, f"P99 推理时延过高: {p99_us} 微秒"


class TestActionSequenceLegality:
    # 检验模型输出动作序列全部合法无非法字符
    def test_action_characters_valid(self, executed_matches):
        total_actions_checked = 0

        for replay in executed_matches:
            mid = replay.get("match_id")
            deltas = replay.get("deltas", [])
            for d in deltas:
                action_list = d.get("a", [])
                for seat_idx, act_str in enumerate(action_list):
                    assert isinstance(act_str, str), f"动作格式非字符串: {type(act_str)}"
                    for ch in act_str:
                        assert ch in LEGAL_ACTIONS, f"对局 {mid} 回合 {d.get('t')} 出现非法动作字符 '{ch}'"
                        total_actions_checked += 1

        assert total_actions_checked > 1000, f"检验动作样本数量不足: {total_actions_checked}"


# 构建 Micro 模型并加载权重检查点
@pytest.fixture(scope="module")
def loaded_micro_models():
    spec = export.classes()["micro"]

    ckpt7 = torch.load(RUNS_DIR / "k7" / "best.pt", map_location="cpu")
    net7 = nets.build(spec, memory="where")
    net7.load_state_dict(ckpt7["trunk"])
    net7.eval()

    ckpt8 = torch.load(RUNS_DIR / "k8" / "best.pt", map_location="cpu")
    net8 = nets.build(spec, memory="where")
    net8.load_state_dict(ckpt8["trunk"])
    net8.eval()

    return {"iter7": net7, "iter8": net8}


class TestMemoryTensorEvolutionAndLeakage:
    # 连续 200 回合多尺寸地图记忆自回归递推演化检验
    @pytest.mark.parametrize("dim", [(24, 24), (32, 32), (40, 40), (64, 64), (128, 128)])
    @pytest.mark.parametrize("model_key", ["iter7", "iter8"])
    def test_200_turn_autoregressive_rollout(self, loaded_micro_models, dim, model_key):
        net = loaded_micro_models[model_key]
        h, w = dim
        mem = torch.zeros((1, 2, h, w), dtype=torch.int8)

        # 随机发生器生成稳定伪对局观测输入
        rng = np.random.default_rng(42)

        with torch.no_grad():
            for turn in range(200):
                # 构造符合规则的 7 通道观测输入
                board_np = np.zeros((1, 7, h, w), dtype=np.int8)

                # 随机生成少量存活蚂蚁与视野
                ant_r = int(rng.integers(0, h))
                ant_c = int(rng.integers(0, w))
                board_np[0, 0, ant_r, ant_c] = 1

                # 环面视野覆盖中心
                r_indices = np.arange(ant_r - 4, ant_r + 5) % h
                c_indices = np.arange(ant_c - 4, ant_c + 5) % w
                board_np[0, 6, :, :] = 0
                for r_idx in r_indices:
                    for c_idx in c_indices:
                        board_np[0, 6, r_idx, c_idx] = 1

                # 视野内随机出现食物与敌穴
                if rng.random() < 0.4:
                    board_np[0, 2, ant_r, (ant_c + 1) % w] = 1
                if rng.random() < 0.2:
                    board_np[0, 5, (ant_r + 1) % h, ant_c] = 1

                board = torch.from_numpy(board_np)
                policy, mem_next = net(board, mem)

                # 形状断言
                assert policy.shape == (1, 5, h, w), f"策略形状异常: {policy.shape}"
                assert mem_next.shape == (1, 2, h, w), f"记忆形状错位: {mem_next.shape}"
                assert mem_next.dtype == torch.int8, f"记忆类型异常: {mem_next.dtype}"

                # 数值断言
                unique_vals = torch.unique(mem_next).tolist()
                for v in unique_vals:
                    assert v in (0, 1), f"第 {turn} 回合记忆出现非二值数值: {v}"

                assert not torch.isnan(policy).any(), f"第 {turn} 回合策略输出存在 NaN"
                assert not torch.isinf(policy).any(), f"第 {turn} 回合策略输出存在 Inf"

                mem = mem_next

    # Where 记忆语义状态机四阶段周期检验（食物与敌穴发现、迷雾保持、清除与防复活）
    @pytest.mark.parametrize("model_key", ["iter7", "iter8"])
    def test_where_memory_semantics_lifecycle(self, loaded_micro_models, model_key):
        net = loaded_micro_models[model_key]
        h, w = 32, 32

        # 测试单元格：food_pos (10, 10), hill_pos (20, 20)
        food_pos = (10, 10)
        hill_pos = (20, 20)

        mem = torch.zeros((1, 2, h, w), dtype=torch.int8)

        with torch.no_grad():
            # 阶段 1: 发现目标（视野可见且存在目标）
            b1 = torch.zeros((1, 7, h, w), dtype=torch.int8)
            b1[0, 6, food_pos[0], food_pos[1]] = 1
            b1[0, 2, food_pos[0], food_pos[1]] = 1
            b1[0, 6, hill_pos[0], hill_pos[1]] = 1
            b1[0, 5, hill_pos[0], hill_pos[1]] = 1

            _, m1 = net(b1, mem)
            assert m1[0, 0, food_pos[0], food_pos[1]] == 1, "食物未成功写入记忆平面 0"
            assert m1[0, 1, hill_pos[0], hill_pos[1]] == 1, "敌穴未成功写入记忆平面 1"

            # 阶段 2: 蚂蚁远离进入战争迷雾（视野为 0，目标依然在迷雾中）
            b2 = torch.zeros((1, 7, h, w), dtype=torch.int8)
            # 视野在别处
            b2[0, 6, 0, 0] = 1

            _, m2 = net(b2, m1)
            assert m2[0, 0, food_pos[0], food_pos[1]] == 1, "迷雾中食物记忆遗失"
            assert m2[0, 1, hill_pos[0], hill_pos[1]] == 1, "迷雾中敌穴记忆遗失"

            # 阶段 3: 蚂蚁返回，食物被采集、敌穴被拆除（视野为 1，但平面为 0）
            b3 = torch.zeros((1, 7, h, w), dtype=torch.int8)
            b3[0, 6, food_pos[0], food_pos[1]] = 1
            b3[0, 2, food_pos[0], food_pos[1]] = 0
            b3[0, 6, hill_pos[0], hill_pos[1]] = 1
            b3[0, 5, hill_pos[0], hill_pos[1]] = 0

            _, m3 = net(b3, m2)
            assert m3[0, 0, food_pos[0], food_pos[1]] == 0, "已被采集食物幽灵记忆未被清除"
            assert m3[0, 1, hill_pos[0], hill_pos[1]] == 0, "已被拆毁敌穴幽灵记忆未被清除"

            # 阶段 4: 蚂蚁再次离开（视野为 0），验证记忆不发生复活
            b4 = torch.zeros((1, 7, h, w), dtype=torch.int8)
            b4[0, 6, 0, 0] = 1

            _, m4 = net(b4, m3)
            assert m4[0, 0, food_pos[0], food_pos[1]] == 0, "已清除食物记忆发生异常复活"
            assert m4[0, 1, hill_pos[0], hill_pos[1]] == 0, "已清除敌穴记忆发生异常复活"

    # 环面跨边界拓扑连通记忆更新检验
    @pytest.mark.parametrize("model_key", ["iter7", "iter8"])
    def test_torus_boundary_memory_update(self, loaded_micro_models, model_key):
        net = loaded_micro_models[model_key]
        h, w = 24, 24

        corner_food = (0, 0)
        corner_hill = (h - 1, w - 1)

        mem = torch.zeros((1, 2, h, w), dtype=torch.int8)

        with torch.no_grad():
            b = torch.zeros((1, 7, h, w), dtype=torch.int8)
            b[0, 6, corner_food[0], corner_food[1]] = 1
            b[0, 2, corner_food[0], corner_food[1]] = 1
            b[0, 6, corner_hill[0], corner_hill[1]] = 1
            b[0, 5, corner_hill[0], corner_hill[1]] = 1

            _, m1 = net(b, mem)
            assert m1[0, 0, corner_food[0], corner_food[1]] == 1, "环面边界食物记忆写入失败"
            assert m1[0, 1, corner_hill[0], corner_hill[1]] == 1, "环面边界敌穴记忆写入失败"

    # 内存泄露压力测试：500 次推演监控堆内存分配增量
    def test_memory_leak_free_across_500_steps(self, loaded_micro_models):
        net = loaded_micro_models["iter8"]
        h, w = 40, 40
        board = torch.zeros((1, 7, h, w), dtype=torch.int8)
        mem = torch.zeros((1, 2, h, w), dtype=torch.int8)

        # 启动跟踪
        tracemalloc.start()

        with torch.no_grad():
            # 预热 50 步建立稳态分配
            for _ in range(50):
                _, mem = net(board, mem)

            snapshot_warmup = tracemalloc.take_snapshot()

            # 执行 500 步高压推演循环
            for _ in range(500):
                _, mem = net(board, mem)

            snapshot_final = tracemalloc.take_snapshot()

        tracemalloc.stop()

        top_stats = snapshot_final.compare_to(snapshot_warmup, "lineno")
        total_growth_bytes = sum(stat.size_diff for stat in top_stats if stat.size_diff > 0)

        # 500 步连续推演内存增量必须严格小于 1MB（实测为微量缓存）
        assert total_growth_bytes < 1_048_576, f"连续推演存在内存泄露，增长字节数: {total_growth_bytes}"
