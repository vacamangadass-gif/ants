from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
METRICS_PATH = ROOT / "data" / "tournament_p3_results" / "tournament_metrics.json"
TELEMETRY_B_PATH = ROOT / "runs" / "exp033" / "dagger_telemetry.json"
TELEMETRY_A_PATH = ROOT / "runs" / "exp031" / "dagger_telemetry.json"


def independent_kneedle(x_vals: np.ndarray, y_vals: np.ndarray, invert: bool = False) -> tuple[np.ndarray, int]:
    x = np.asarray(x_vals, dtype=float)
    y = np.asarray(y_vals, dtype=float)
    x_span = np.max(x) - np.min(x)
    y_span = np.max(y) - np.min(y)
    assert x_span > 0.0, "x 轴跨度必须严格大于零"
    assert y_span > 0.0, "y 轴跨度必须严格大于零"

    x_norm = (x - np.min(x)) / x_span
    y_norm = (y - np.min(y)) / y_span

    if invert:
        d = (1.0 - x_norm) - y_norm
    else:
        d = y_norm - x_norm

    knee_idx = int(np.argmax(d))
    return d, knee_idx


def independent_marginal_derivatives(x_vals: np.ndarray, y_vals: np.ndarray, scale: float = 100000.0) -> np.ndarray:
    x = np.asarray(x_vals, dtype=float)
    y = np.asarray(y_vals, dtype=float)
    dx = np.diff(x)
    dy = np.diff(y)
    assert np.all(dx > 0.0), "x 步长必须严格单调递增"
    return (dy / dx) * scale


class TestTournamentMetricsExtraction:
    # 提取比赛指标并检验数据完整性
    def test_extract_raw_metrics(self):
        assert METRICS_PATH.is_file(), f"评测指标文件不存在: {METRICS_PATH}"
        with open(METRICS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert "leaderboard" in data, "缺少排行榜数据结构"
        leaderboard = {entry["name"]: entry for entry in data["leaderboard"]}

        iters = list(range(1, 9))
        for k in iters:
            assert f"exp031_iter{k}" in leaderboard, f"缺少路线 A 迭代 {k} 数据"
            assert f"exp033_iter{k}" in leaderboard, f"缺少路线 B 迭代 {k} 数据"

        b_scores = np.array([leaderboard[f"exp033_iter{k}"]["score_rate"] * 100.0 for k in iters])
        a_scores = np.array([leaderboard[f"exp031_iter{k}"]["score_rate"] * 100.0 for k in iters])
        b_blocked = np.array([leaderboard[f"exp033_iter{k}"]["blocked_move_pct"] for k in iters])
        a_blocked = np.array([leaderboard[f"exp031_iter{k}"]["blocked_move_pct"] for k in iters])

        assert len(b_scores) == 8, "路线 B 样本序列长度必须为 8"
        assert len(a_scores) == 8, "路线 A 样本序列长度必须为 8"
        assert len(b_blocked) == 8, "路线 B 受阻率序列长度必须为 8"
        assert len(a_blocked) == 8, "路线 A 受阻率序列长度必须为 8"


class TestKneedleSolverValidation:
    # 闭式解 Kneedle 算法全局与次级局部极大值检验
    def test_kneedle_route_b_extrema(self):
        with open(METRICS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        leaderboard = {entry["name"]: entry for entry in data["leaderboard"]}

        iters = np.arange(1, 9)
        samples = np.array([250000 + 83333 * k for k in iters], dtype=float)
        b_scores = np.array([leaderboard[f"exp033_iter{k}"]["score_rate"] * 100.0 for k in iters], dtype=float)

        d_k, knee_idx = independent_kneedle(samples, b_scores)

        # 验证全局极大值索引严格为迭代 3 (索引为 2)
        assert knee_idx == 2, f"Kneedle 全局极大值索引应为 2 (第 3 轮)，实际为: {knee_idx}"
        assert iters[knee_idx] == 3, "Kneedle 全局拐点轮次必须严格为第 3 轮"

        # 验证 D_3 与 D_5 精度数值
        d3 = float(d_k[2])
        d5 = float(d_k[4])
        assert abs(d3 - 0.4539) < 1e-4, f"D_3 数值异常: 预期 0.4539，实际为 {d3:.4f}"
        assert abs(d5 - 0.3598) < 1e-4, f"D_5 数值异常: 预期 0.3598，实际为 {d5:.4f}"

        # 验证 D_k 序列边界值 D_1 与 D_8 严格归零
        assert abs(d_k[0] - 0.0) < 1e-6, f"D_1 必须严格归零，实际为 {d_k[0]}"
        assert abs(d_k[7] - 0.0) < 1e-6, f"D_8 必须严格归零，实际为 {d_k[7]}"

        # 验证第 5 轮严格构成次级局部极大值 (D_5 > D_4 且 D_5 > D_6)
        d4 = float(d_k[3])
        d6 = float(d_k[5])
        assert d5 > d4, f"第 5 轮非局部极大值: D_5 ({d5:.4f}) <= D_4 ({d4:.4f})"
        assert d5 > d6, f"第 5 轮非局部极大值: D_5 ({d5:.4f}) <= D_6 ({d6:.4f})"

        # 验证全局极大值 D_3 严格大于次级极大值 D_5
        assert d3 > d5, f"全局极大值 D_3 ({d3:.4f}) 必须严格大于次级极大值 D_5 ({d5:.4f})"

        # 完整校验全部 8 个点位的 D_k 数值精度
        expected_d = [0.0000, 0.4466, 0.4539, 0.0648, 0.3598, -0.6596, -0.3091, 0.0000]
        for idx, (actual, exp) in enumerate(zip(d_k, expected_d)):
            assert abs(actual - exp) < 1e-4, f"D_{idx + 1} 精度不符: 实际 {actual:.4f} vs 预期 {exp:.4f}"


class TestMarginalDerivativesValidation:
    # 离散边际导数均值与收益衰减倍率检验
    def test_marginal_derivatives_route_b(self):
        with open(METRICS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        leaderboard = {entry["name"]: entry for entry in data["leaderboard"]}

        iters = np.arange(1, 9)
        samples = np.array([250000 + 83333 * k for k in iters], dtype=float)
        b_scores = np.array([leaderboard[f"exp033_iter{k}"]["score_rate"] * 100.0 for k in iters], dtype=float)

        derivatives = independent_marginal_derivatives(samples, b_scores, scale=100000.0)
        assert len(derivatives) == 7, "7 个步进区间必须生成 7 个导数值"

        # g_2: 迭代 1 到 2 导数
        g2 = float(derivatives[0])
        # g_3: 迭代 2 到 3 导数
        g3 = float(derivatives[1])

        assert abs(g2 - 9.564) < 0.01, f"g_2 数值异常: 实际 {g2:.4f}"
        assert abs(g3 - 2.436) < 0.01, f"g_3 数值异常: 实际 {g3:.4f}"

        # 前 3 轮平均导数 (区间 1->2 与 2->3)
        first_3_mean = float(np.mean(derivatives[:2]))
        # 后 5 轮平均导数 (区间 3->4, 4->5, 5->6, 6->7, 7->8)
        next_5_mean = float(np.mean(derivatives[2:]))

        assert abs(first_3_mean - 6.00) < 0.01, f"前 3 轮平均导数异常: 实际 {first_3_mean:.4f}%"
        assert abs(next_5_mean - 0.84) < 0.01, f"后 5 轮平均导数异常: 实际 {next_5_mean:.4f}%"

        # 闭合关系检验: 弦斜率与导数均值在等步长下的代数等价性
        direct_first_3_slope = ((b_scores[2] - b_scores[0]) / (samples[2] - samples[0])) * 100000.0
        direct_next_5_slope = ((b_scores[7] - b_scores[2]) / (samples[7] - samples[2])) * 100000.0
        assert abs(first_3_mean - direct_first_3_slope) < 1e-6, "前 3 轮导数均值与割线斜率必须代数等价"
        assert abs(next_5_mean - direct_next_5_slope) < 1e-6, "后 5 轮导数均值与割线斜率必须代数等价"

        # 边际效率衰减倍率检验
        decay_ratio = first_3_mean / next_5_mean
        assert abs(decay_ratio - 7.1) < 0.1, f"边际效率衰减倍率异常: 实际 {decay_ratio:.2f}"


class TestBlockedMoveRateDropValidation:
    # 移动受阻率在迭代 2 的最大相对衰减幅度检验
    def test_blocked_rate_max_drop(self):
        with open(METRICS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        leaderboard = {entry["name"]: entry for entry in data["leaderboard"]}

        iters = np.arange(1, 9)
        b_blocked = np.array([leaderboard[f"exp033_iter{k}"]["blocked_move_pct"] for k in iters], dtype=float)
        a_blocked = np.array([leaderboard[f"exp031_iter{k}"]["blocked_move_pct"] for k in iters], dtype=float)

        # 相对变化率: (blocked[k] - blocked[k-1]) / blocked[k-1] * 100%
        b_rel_drops = ((b_blocked[1:] - b_blocked[:-1]) / b_blocked[:-1]) * 100.0
        a_rel_drops = ((a_blocked[1:] - a_blocked[:-1]) / a_blocked[:-1]) * 100.0

        # 验证路线 B 全周期最大相对降幅严格发生于第 2 轮 (即索引 0 对应的 1->2 区间)
        b_max_drop_idx = int(np.argmin(b_rel_drops))
        assert b_max_drop_idx == 0, f"路线 B 最大降幅轮次异常: 索引 {b_max_drop_idx}"

        # 验证相对降幅数值 (路线 B 降幅 -43.7%，路线 A 降幅 -30.6%)
        b_drop_iter2 = float(b_rel_drops[0])
        a_drop_iter2 = float(a_rel_drops[0])

        assert abs(b_drop_iter2 - (-43.7)) < 0.2, f"路线 B 迭代 2 降幅异常: 实际 {b_drop_iter2:.2f}%"
        assert abs(a_drop_iter2 - (-30.6)) < 0.2, f"路线 A 迭代 2 降幅异常: 实际 {a_drop_iter2:.2f}%"

        # 验证路线 A 在初期推演中，第 2 轮为最显著降幅区间
        assert a_rel_drops[0] < 0.0, "路线 A 迭代 2 必须呈现负向降幅"
        assert a_rel_drops[0] < a_rel_drops[1], "路线 A 迭代 2 降幅必须明显优于迭代 3 反弹"


class TestRouteAKneedleCapacityValidation:
    # 路线 A (Nano) 容量瓶颈拐点数学检验
    def test_route_a_capacity_knee_at_iter_2(self):
        with open(METRICS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        leaderboard = {entry["name"]: entry for entry in data["leaderboard"]}

        iters = np.arange(1, 9)
        samples = np.array([250000 + 83333 * k for k in iters], dtype=float)
        a_scores = np.array([leaderboard[f"exp031_iter{k}"]["score_rate"] * 100.0 for k in iters], dtype=float)

        d_k_a, knee_idx_a = independent_kneedle(samples, a_scores)

        # 验证路线 A 容量瓶颈拐点严格位于第 2 轮 (索引为 1)
        assert knee_idx_a == 1, f"路线 A 拐点索引应为 1 (第 2 轮)，实际为: {knee_idx_a}"
        assert iters[knee_idx_a] == 2, "路线 A 拐点轮次必须严格为第 2 轮"

        # 验证 D_2_A 达到峰值 0.7214
        d2_a = float(d_k_a[1])
        assert abs(d2_a - 0.7214) < 1e-4, f"路线 A D_2 数值异常: 实际 {d2_a:.4f}"

        # 验证第 3 轮出现严重断崖式下跌 (D_3_A 为负值 -0.2857)
        d3_a = float(d_k_a[2])
        assert d3_a < 0.0, f"路线 A 第 3 轮应严重跌出弦线下方: 实际 {d3_a:.4f}"
        assert abs(d3_a - (-0.2857)) < 1e-4, f"路线 A D_3 数值异常: 实际 {d3_a:.4f}"


class TestKneedleAdversarialEdgeCases:
    # 对抗性极值与异常边界条件检验
    def test_zero_span_exceptions(self):
        # 常数序列跨度为零必须直接触发断言
        constant_x = np.array([100.0, 100.0, 100.0])
        constant_y = np.array([10.0, 20.0, 30.0])
        with pytest.raises(AssertionError):
            independent_kneedle(constant_x, constant_y)

        linear_x = np.array([1.0, 2.0, 3.0])
        constant_y_flat = np.array([5.0, 5.0, 5.0])
        with pytest.raises(AssertionError):
            independent_kneedle(linear_x, constant_y_flat)

    def test_colinear_points_yield_zero_distances(self):
        # 严格共线点序列在归一化后距离差值必须处处为零
        x = np.linspace(0.0, 10.0, 11)
        y = np.linspace(0.0, 100.0, 11)
        d_k, knee_idx = independent_kneedle(x, y)
        assert np.allclose(d_k, 0.0, atol=1e-12), "线性共线序列弦距离必须严格为零"

    def test_strictly_convex_diminishing_returns(self):
        # 构造纯对数饱和曲线，验证 Kneedle 必能准确定位对数拐点
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        y = np.log(x)
        d_k, knee_idx = independent_kneedle(x, y)
        assert knee_idx > 0 and knee_idx < len(x) - 1, "饱和曲线拐点必须位于内部节点"
        assert d_k[knee_idx] > 0.0, "凸函数拐点垂直距离必须严格大于零"


class TestTelemetryDataConsistency:
    # 遥测数据与推演全周期一致性检验
    def test_telemetry_consistency(self):
        assert TELEMETRY_B_PATH.is_file(), f"路线 B 遥测文件不存在: {TELEMETRY_B_PATH}"
        with open(TELEMETRY_B_PATH, "r", encoding="utf-8") as f:
            dag_b = json.load(f)

        disagree = dag_b["disagreement_rates"]
        d1 = disagree["d1"]
        d2 = disagree["d2"]
        d3 = disagree["d3"]

        # 前三轮动作分歧率严格单调下降
        assert d1 > d2 > d3, f"前三轮分歧率未单调下降: {d1} -> {d2} -> {d3}"

        # 最终累计样本规模严格达到 916,664 席位步数
        final_samples = 250000 + 83333 * 8
        assert final_samples == 916664, f"最终累计样本计算异常: {final_samples}"
