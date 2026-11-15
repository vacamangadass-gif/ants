from __future__ import annotations

import hashlib
import json
import subprocess
import time
from pathlib import Path
import numpy as np
import onnxruntime as ort
import pytest

ROOT = Path(__file__).resolve().parents[2]
EXP083_DIR = ROOT / "models" / "exp083"
ONNX_PATH = EXP083_DIR / "model.onnx"
MANIFEST_PATH = EXP083_DIR / "manifest.json"
EVAL_PATH = ROOT / "runs" / "exp083" / "tournament_evaluation.json"
REPLAYS_DIR = ROOT / "replays" / "tournament_exp083"

EXPECTED_HEX_DIGEST = "819166e7918117a1cbbd67dc042e7368a076a2dd45cea5122683983912d591e1"


def get_inference_session() -> ort.InferenceSession:
    assert ONNX_PATH.is_file(), f"ONNX 模型文件不存在: {ONNX_PATH}"
    session_options = ort.SessionOptions()
    session_options.intra_op_num_threads = 1
    session_options.inter_op_num_threads = 1
    return ort.InferenceSession(
        str(ONNX_PATH),
        sess_options=session_options,
        providers=["CPUExecutionProvider"],
    )


def test_onnx_model_metadata_and_file_size():
    # 验证模型文件存在与物理体积上限（128 KB 即 131,072 字节）
    assert ONNX_PATH.is_file(), f"未找到模型固件文件: {ONNX_PATH}"
    byte_size = ONNX_PATH.stat().st_size
    assert byte_size <= 131072, f"模型固件体积超出上限: {byte_size} 字节 > 131072 字节"

    # 验证清单元数据文件格式与输入输出声明
    assert MANIFEST_PATH.is_file(), f"未找到清单文件: {MANIFEST_PATH}"
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    assert manifest.get("format") == "onnx"
    assert manifest.get("abi") == "orion:model@1.0.0"

    inputs = {item["name"]: item for item in manifest["inputs"]}
    outputs = {item["name"]: item for item in manifest["outputs"]}
    assert "board" in inputs, "清单输入缺少 board"
    assert "memory_in" in inputs, "清单输入缺少 memory_in"
    assert "policy" in outputs, "清单输出缺少 policy"
    assert "memory" in outputs, "清单输出缺少 memory"

    assert inputs["board"]["dtype"] == "i8"
    assert inputs["memory_in"]["dtype"] == "i8"
    assert outputs["policy"]["dtype"] == "f32"
    assert outputs["memory"]["dtype"] == "i8"


def test_onnx_extreme_board_adversarial_inference():
    # 对极端棋盘状态执行对抗推理检验
    sess = get_inference_session()
    h, w = 32, 32

    adversarial_cases = []

    # 极端案例 1：全零空棋盘与全零记忆
    b_zeros = np.zeros((1, 7, h, w), dtype=np.int8)
    m_zeros = np.zeros((1, 2, h, w), dtype=np.int8)
    adversarial_cases.append(("all_zeros", b_zeros, m_zeros))

    # 极端案例 2：全图布满己方蚂蚁（通道 0 全部置 1）
    b_full_ants = np.zeros((1, 7, h, w), dtype=np.int8)
    b_full_ants[0, 0, :, :] = 1
    adversarial_cases.append(("all_ants", b_full_ants, m_zeros))

    # 极端案例 3：全图布满食物（通道 2 全部置 1）
    b_full_food = np.zeros((1, 7, h, w), dtype=np.int8)
    b_full_food[0, 2, :, :] = 1
    adversarial_cases.append(("all_food", b_full_food, m_zeros))

    # 极端案例 4：全通道数值饱和（所有 7 个输入通道全部置 1）
    b_saturated = np.ones((1, 7, h, w), dtype=np.int8)
    m_ones = np.ones((1, 2, h, w), dtype=np.int8)
    adversarial_cases.append(("all_saturated", b_saturated, m_ones))

    # 极端案例 5：高密度迷宫障碍水域（棋盘奇数行布满水域）
    b_maze = np.zeros((1, 7, h, w), dtype=np.int8)
    b_maze[0, 3, 1::2, :] = 1
    b_maze[0, 0, 0::2, 0::2] = 1
    adversarial_cases.append(("dense_maze", b_maze, m_zeros))

    # 极端案例 6：越界对抗数值（极小值 -128 与极大值 127）
    b_extreme_int8 = np.full((1, 7, h, w), 127, dtype=np.int8)
    m_extreme_int8 = np.full((1, 2, h, w), -128, dtype=np.int8)
    adversarial_cases.append(("extreme_int8", b_extreme_int8, m_extreme_int8))

    for name, board, memory_in in adversarial_cases:
        outputs = sess.run(None, {"board": board, "memory_in": memory_in})
        policy, memory_out = outputs[0], outputs[1]

        # 检验策略输出形状与数据类型
        assert policy.shape == (1, 5, h, w), f"[{name}] 策略输出形状不正确: {policy.shape}"
        assert policy.dtype == np.float32, f"[{name}] 策略输出类型不正确: {policy.dtype}"

        # 检验数值有效性（禁止存在 NaN 或 Inf）
        assert np.all(np.isfinite(policy)), f"[{name}] 策略输出包含非数值（NaN 或 Inf）"

        # 检验动作决策空间严格位于 [0, 4] 范围之内
        actions = np.argmax(policy, axis=1)
        assert np.all((actions >= 0) & (actions <= 4)), f"[{name}] 动作输出超出 [0, 4] 范围"

        # 检验记忆输出形状与数据类型
        assert memory_out.shape == (1, 2, h, w), f"[{name}] 记忆输出形状不正确: {memory_out.shape}"
        assert memory_out.dtype == np.int8, f"[{name}] 记忆输出类型不正确: {memory_out.dtype}"


def test_onnx_dynamic_shapes_and_boundary_resolutions():
    # 检验全卷积动态尺寸支持（覆盖平台包络边长 24 至 124）
    sess = get_inference_session()

    test_resolutions = [
        (24, 24),
        (32, 32),
        (40, 40),
        (64, 64),
        (124, 124),
        (128, 128),
        (24, 40),
        (40, 24),
        (25, 37),
        (124, 64),
        (64, 124),
    ]

    for h, w in test_resolutions:
        board = np.zeros((1, 7, h, w), dtype=np.int8)
        memory_in = np.zeros((1, 2, h, w), dtype=np.int8)
        policy, memory_out = sess.run(None, {"board": board, "memory_in": memory_in})
        assert policy.shape == (1, 5, h, w)
        assert memory_out.shape == (1, 2, h, w)
        assert np.all(np.isfinite(policy))

    # 检验批处理动态轴（B > 1）支持
    for b in [2, 4]:
        board = np.zeros((b, 7, 32, 32), dtype=np.int8)
        memory_in = np.zeros((b, 2, 32, 32), dtype=np.int8)
        policy, memory_out = sess.run(None, {"board": board, "memory_in": memory_in})
        assert policy.shape == (b, 5, 32, 32)
        assert memory_out.shape == (b, 2, 32, 32)


def test_onnx_where_memory_preservation_semantics():
    # 检验 Where 条件记忆算子跨回合更新逻辑与遗忘保留语义
    sess = get_inference_session()
    h, w = 24, 24

    # 回合 0：全零初始化，初始记忆全零
    b0 = np.zeros((1, 7, h, w), dtype=np.int8)
    m0 = np.zeros((1, 2, h, w), dtype=np.int8)
    _, m_out0 = sess.run(None, {"board": b0, "memory_in": m0})
    assert np.all(m_out0 == 0)

    # 回合 1：单元格 (5, 5) 发现食物，单元格 (10, 10) 发现敌方蚁穴，可见性 vis 置 1
    b1 = np.zeros((1, 7, h, w), dtype=np.int8)
    b1[0, 2, 5, 5] = 1  # 食物平面通道 2
    b1[0, 6, 5, 5] = 1  # 可见性平面通道 6
    b1[0, 5, 10, 10] = 1  # 敌方蚁穴平面通道 5
    b1[0, 6, 10, 10] = 1  # 可见性平面通道 6

    _, m_out1 = sess.run(None, {"board": b1, "memory_in": m_out0})
    assert m_out1[0, 0, 5, 5] == 1, "食物记忆未写入通道 0"
    assert m_out1[0, 1, 10, 10] == 1, "敌穴记忆未写入通道 1"

    # 回合 2：两处单元格均离开视野进入迷雾（vis 为 0），当前棋盘无食物与敌穴观测
    b2 = np.zeros((1, 7, h, w), dtype=np.int8)
    # vis 保持为 0
    _, m_out2 = sess.run(None, {"board": b2, "memory_in": m_out1})
    assert m_out2[0, 0, 5, 5] == 1, "迷雾中食物历史记忆被错误清零"
    assert m_out2[0, 1, 10, 10] == 1, "迷雾中敌穴历史记忆被错误清零"

    # 回合 3：单元格 (5, 5) 重新可见（vis 为 1），但食物已被采食（通道 2 为 0）
    b3 = np.zeros((1, 7, h, w), dtype=np.int8)
    b3[0, 6, 5, 5] = 1  # 可见性恢复
    b3[0, 2, 5, 5] = 0  # 食物已消失
    _, m_out3 = sess.run(None, {"board": b3, "memory_in": m_out2})
    assert m_out3[0, 0, 5, 5] == 0, "食物消失后记忆未完成条件清零更新"
    assert m_out3[0, 1, 10, 10] == 1, "未在视野内的敌穴记忆被异常修改"


def test_onnx_inference_latency_stress():
    # 连续测量 100 步推理耗时，检验延迟指标严格满足时限约束
    sess = get_inference_session()

    board_sizes = [24, 40, 64, 128]
    for size in board_sizes:
        board = np.zeros((1, 7, size, size), dtype=np.int8)
        memory_in = np.zeros((1, 2, size, size), dtype=np.int8)

        # 预热执行 5 步
        for _ in range(5):
            sess.run(None, {"board": board, "memory_in": memory_in})

        # 连续测量 100 步耗时
        latencies_ms = []
        for _ in range(100):
            t0 = time.perf_counter()
            sess.run(None, {"board": board, "memory_in": memory_in})
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            latencies_ms.append(elapsed_ms)

        median_ms = float(np.median(latencies_ms))
        p95_ms = float(np.percentile(latencies_ms, 95))
        max_ms = float(np.max(latencies_ms))

        # 验证单步耗时严格低于平台 1000 毫秒上限
        assert max_ms <= 1000.0, f"尺寸 {size}x{size} 最大耗时超标: {max_ms:.2f} ms > 1000 ms"

        # 在探针基准测试尺寸 128x128 下验证中位数耗时低于 30 毫秒门禁
        if size == 128:
            assert median_ms <= 30.0, f"探针尺寸 128x128 中位数耗时超标: {median_ms:.2f} ms > 30 ms"


def test_tournament_metric_independent_audit():
    # 独立复算 Gate 3-2 锦标赛评测聚合指标
    assert EVAL_PATH.is_file(), f"未找到评测结果文件: {EVAL_PATH}"
    eval_data = json.loads(EVAL_PATH.read_text(encoding="utf-8"))

    overall = eval_data["overall"]
    matches = overall["matches"]
    wins = overall["wins"]
    draws = overall["draws"]
    losses = overall["losses"]
    points = overall["points"]
    total_orders = overall["total_orders"]
    blocked_orders = overall["blocked_orders"]

    # 1. 验证场次胜平负与积分计算完全自洽
    assert matches == 180, f"锦标赛总场次不为 180: {matches}"
    assert wins + draws + losses == matches, "胜平负场次总和与总场次数不相等"
    expected_points = wins * 1.0 + draws * 0.5
    assert abs(points - expected_points) < 1e-6, f"积分计算错误: 实际 {points} != 预期 {expected_points}"

    # 2. 独立复算得分率并验证 Gate 3-2 得分率要求（>= 65.74%）
    calculated_score_rate = points / matches
    assert abs(overall["score_rate"] - calculated_score_rate) < 1e-4
    assert calculated_score_rate >= 0.6574, f"得分率未达标: {calculated_score_rate:.4%} < 65.74%"

    # 3. 独立复算通道阻挡率并验证 Gate 3-2 阻挡率要求（<= 0.80%）
    calculated_blocked_rate = blocked_orders / total_orders
    assert calculated_blocked_rate <= 0.0080, f"通道阻挡率超标: {calculated_blocked_rate:.4%} > 0.80%"

    # 4. 验证十位竞争对手对战数据总和与整体指标完全一致
    h2h = eval_data["head_to_head"]
    assert len(h2h) == 10, f"对手模型数量不为 10: {len(h2h)}"
    h2h_matches = sum(item["matches"] for item in h2h.values())
    h2h_wins = sum(item["wins"] for item in h2h.values())
    h2h_draws = sum(item["draws"] for item in h2h.values())
    h2h_losses = sum(item["losses"] for item in h2h.values())
    h2h_points = sum(item["points"] for item in h2h.values())

    assert h2h_matches == matches
    assert h2h_wins == wins
    assert h2h_draws == draws
    assert h2h_losses == losses
    assert abs(h2h_points - points) < 1e-6

    # 5. 验证九张地图场次总和与整体指标完全一致
    maps = eval_data["map_breakdown"]
    assert len(maps) == 9, f"基准地图数量不为 9: {len(maps)}"
    map_matches = sum(item["matches"] for item in maps.values())
    map_wins = sum(item["wins"] for item in maps.values())
    map_draws = sum(item["draws"] for item in maps.values())
    map_losses = sum(item["losses"] for item in maps.values())
    map_points = sum(item["points"] for item in maps.values())

    assert map_matches == matches
    assert map_wins == wins
    assert map_draws == draws
    assert map_losses == losses
    assert abs(map_points - points) < 1e-6

    assert eval_data.get("gate3_2_pass") is True, "Gate 3-2 门禁状态非通过"


def test_replays_raw_data_ground_truth_consistency():
    # 遍历真实回放文件，以回放底层事实核验指标真实性
    assert REPLAYS_DIR.is_dir(), f"回放目录不存在: {REPLAYS_DIR}"
    exp083_replays = list(REPLAYS_DIR.glob("*exp083*.json"))
    assert len(exp083_replays) == 180, f"exp083 回放文件数量异常: {len(exp083_replays)} != 180"

    raw_wins = 0
    raw_draws = 0
    raw_losses = 0
    raw_points = 0.0
    raw_score_for = 0
    raw_score_against = 0
    raw_turns = 0

    for r_path in exp083_replays:
        content = json.loads(r_path.read_text(encoding="utf-8"))
        assert content.get("engine_digest") == f"sha256:{EXPECTED_HEX_DIGEST}"

        seats = content["seats"]
        my_seat = 0 if seats[0]["label"] == "exp083" else 1
        opp_seat = 1 - my_seat

        ranks = content["engine_ranks"]
        scores = content["scores"]
        turns = content["turns"]

        raw_turns += turns
        raw_score_for += scores[my_seat]
        raw_score_against += scores[opp_seat]

        if ranks[my_seat] < ranks[opp_seat]:
            raw_wins += 1
            raw_points += 1.0
        elif ranks[my_seat] == ranks[opp_seat]:
            raw_draws += 1
            raw_points += 0.5
        else:
            raw_losses += 1

    assert raw_wins == 154, f"真实胜场不匹配: {raw_wins} != 154"
    assert raw_draws == 3, f"真实平局不匹配: {raw_draws} != 3"
    assert raw_losses == 23, f"真实负场不匹配: {raw_losses} != 23"
    assert abs(raw_points - 155.5) < 1e-6, f"真实积分不匹配: {raw_points} != 155.5"
    assert raw_score_for == 467, f"真实总进球/得分不匹配: {raw_score_for} != 467"
    assert raw_score_against == 74, f"真实失球/失分不匹配: {raw_score_against} != 74"
    assert raw_turns == 11156, f"真实总回合数不匹配: {raw_turns} != 11156"


def test_determinism_deny_sh_and_engine_untouched():
    # 验证底座确定性检测与引擎目录零变更
    deny_res = subprocess.run(["./tools/deny.sh"], cwd=str(ROOT), capture_output=True, text=True)
    assert deny_res.returncode == 0, f"tools/deny.sh 执行失败:\n{deny_res.stderr or deny_res.stdout}"
    assert "determinism: no floating point in game logic" in deny_res.stdout

    git_res = subprocess.run(["git", "status", "--porcelain", "engine/"], cwd=str(ROOT), capture_output=True, text=True)
    assert git_res.returncode == 0
    assert git_res.stdout.strip() == "", f"engine/ 目录存在未预期改动:\n{git_res.stdout}"
