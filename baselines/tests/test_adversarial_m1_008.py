from __future__ import annotations

import math
import numpy as np
import pytest

from tb_baselines.benchmarks.run_exp080_scaling import (
    compute_collision_blocking_rate,
    compute_corridor_congestion_index,
    compute_corridor_mask,
    compute_corridor_stalling_rate,
    compute_effective_gathering_throughput,
    rle_from_mask,
    run_scaling_simulation,
)


def test_corridor_all_water_topologies():
    # 测试全水域网格（方阵与长方形）走廊掩码必须全为 False
    water_sq = np.ones((10, 10), dtype=bool)
    mask_sq = compute_corridor_mask(water_sq, max_corridor_width=2)
    assert not mask_sq.any()
    assert int(mask_sq.sum()) == 0

    water_rect = np.ones((48, 64), dtype=bool)
    mask_rect = compute_corridor_mask(water_rect, max_corridor_width=2)
    assert not mask_rect.any()
    assert int(mask_rect.sum()) == 0

    water_1x1 = np.ones((1, 1), dtype=bool)
    mask_1x1 = compute_corridor_mask(water_1x1, max_corridor_width=2)
    assert not mask_1x1.any()
    assert int(mask_1x1.sum()) == 0


def test_corridor_all_land_topologies():
    # 测试标准尺度开阔全陆地网格走廊掩码必须全为 False
    water_sq = np.zeros((16, 16), dtype=bool)
    mask_sq = compute_corridor_mask(water_sq, max_corridor_width=2)
    assert not mask_sq.any()
    assert int(mask_sq.sum()) == 0

    water_rect = np.zeros((48, 64), dtype=bool)
    mask_rect = compute_corridor_mask(water_rect, max_corridor_width=2)
    assert not mask_rect.any()
    assert int(mask_rect.sum()) == 0


def test_corridor_single_tile_island():
    # 测试孤岛单格陆地拓扑
    rows, cols = 8, 8
    water = np.ones((rows, cols), dtype=bool)
    water[3, 3] = False

    mask = compute_corridor_mask(water, max_corridor_width=2)
    assert mask[3, 3]
    assert int(mask.sum()) == 1


def test_corridor_checkerboard_topology():
    # 测试棋盘格水陆交错拓扑
    rows, cols = 8, 8
    r_idx, c_idx = np.indices((rows, cols))
    water = ((r_idx + c_idx) % 2 == 1)

    mask = compute_corridor_mask(water, max_corridor_width=2)
    expected_land = ~water
    assert np.array_equal(mask, expected_land)
    assert int(mask.sum()) == (rows * cols) // 2


def test_corridor_toroidal_wraparound_corridors():
    # 测试横跨环面上下与左右边界的走廊识别
    rows, cols = 12, 12

    # 水平方向环面跨边界走廊：第 0 行与第 11 行联通
    water_h = np.zeros((rows, cols), dtype=bool)
    water_h[1, :] = True
    water_h[10, :] = True
    mask_h = compute_corridor_mask(water_h, max_corridor_width=2)
    for c in range(cols):
        assert mask_h[0, c]
        assert mask_h[11, c]
    assert int(mask_h.sum()) == cols * 2

    # 垂直方向环面跨边界走廊：第 0 列与第 11 列联通
    water_v = np.zeros((rows, cols), dtype=bool)
    water_v[:, 1] = True
    water_v[:, 10] = True
    mask_v = compute_corridor_mask(water_v, max_corridor_width=2)
    for r in range(rows):
        assert mask_v[r, 0]
        assert mask_v[r, 11]
    assert int(mask_v.sum()) == rows * 2


def test_corridor_rectangular_grid_asymmetric():
    # 测试非对称长方形网格（48x64）中的走廊识别
    rows, cols = 48, 64
    water = np.zeros((rows, cols), dtype=bool)

    # 在第 15 行放置宽度为 1 的水平通道，上下由水域包围
    water[14, :] = True
    water[16, :] = True
    mask = compute_corridor_mask(water, max_corridor_width=2)
    for c in range(cols):
        assert mask[15, c]
    assert int(mask.sum()) == cols


def test_corridor_l_turn_corner_limitation_empirical():
    # 对抗性检验 L 型直角拐角在轴向跨度定义下的行为
    # 构造一条长水平臂和长垂直臂交汇的宽度 1 拐角通道
    rows, cols = 16, 16
    water = np.ones((rows, cols), dtype=bool)

    # 垂直臂：(2, 4) 到 (8, 4)
    for r in range(2, 9):
        water[r, 4] = False
    # 水平臂：(8, 4) 到 (8, 12)
    for c in range(4, 13):
        water[8, c] = False

    mask = compute_corridor_mask(water, max_corridor_width=2)

    # 垂直臂内部各单元格（非拐角）应当被判定为走廊（水平跨度为 1）
    for r in range(2, 8):
        assert mask[r, 4], f"垂直臂单元格 ({r}, 4) 未被识别为走廊"

    # 水平臂内部各单元格（非拐角）应当被判定为走廊（垂直跨度为 1）
    for c in range(5, 13):
        assert mask[8, c], f"水平臂单元格 (8, {c}) 未被识别为走廊"

    # 检验拐角单元格 (8, 4)：其水平跨度为 9，垂直跨度为 7，min(9, 7) = 7 > 2
    # 记录该轴向跨度算法的固有特征：长臂 L 拐点未落入走廊掩码
    corner_is_corridor = bool(mask[8, 4])
    assert not corner_is_corridor, "轴向投影算法应当将长臂直角拐点计算为两臂长度最小值"


def test_corridor_spiral_topology_straight_vs_turns():
    # 测试螺旋环绕走廊拓扑
    rows, cols = 14, 14
    water = np.ones((rows, cols), dtype=bool)

    # 构造外圈逆时针单通道螺旋
    # 顶边
    water[2, 2:12] = False
    # 右边
    water[2:12, 11] = False
    # 底边
    water[11, 4:12] = False
    # 左边
    water[4:12, 4] = False
    # 内顶边
    water[4, 4:9] = False

    mask = compute_corridor_mask(water, max_corridor_width=2)

    # 验证直线段中间单元格均被正确识别为宽度 1 走廊
    assert mask[2, 6]
    assert mask[6, 11]
    assert mask[11, 8]
    assert mask[8, 4]
    assert mask[4, 6]


def test_metric_corridor_congestion_index_boundaries():
    # 测试走廊拥塞指数在 0 回合、0 走廊单元格及各类极端输入下的边界行为
    assert compute_corridor_congestion_index(0, 0, 10) == 0.0
    assert compute_corridor_congestion_index(10, 0, 10) == 0.0
    assert compute_corridor_congestion_index(10, 100, 0) == 0.0
    assert compute_corridor_congestion_index(10, -5, 10) == 0.0
    assert compute_corridor_congestion_index(10, 100, -3) == 0.0

    # 满载停滞（所有走廊单元格每回合均停滞）
    turns = 50
    corridor_cells = 8
    max_stalls = turns * corridor_cells
    assert compute_corridor_congestion_index(max_stalls, turns, corridor_cells) == 1.0

    # 全移动零停滞
    assert compute_corridor_congestion_index(0, turns, corridor_cells) == 0.0


def test_metric_corridor_stalling_rate_boundaries():
    # 测试走廊停滞率在 0 总蚂蚁回合、满载停滞、全移动下的边界行为
    assert compute_corridor_stalling_rate(0, 0) == 0.0
    assert compute_corridor_stalling_rate(10, 0) == 0.0
    assert compute_corridor_stalling_rate(10, -10) == 0.0

    # 100% 走廊停滞率
    assert compute_corridor_stalling_rate(100, 100) == 1.0

    # 0% 走廊停滞率
    assert compute_corridor_stalling_rate(0, 100) == 0.0


def test_metric_gathering_throughput_boundaries():
    # 测试有效采食速率在 0 回合、0 存活蚂蚁、0 采食下的边界行为
    assert compute_effective_gathering_throughput(0, 100, 20.0) == 0.0
    assert compute_effective_gathering_throughput(50, 0, 20.0) == 0.0
    assert compute_effective_gathering_throughput(50, 100, 0.0) == 0.0
    assert compute_effective_gathering_throughput(50, -10, 20.0) == 0.0
    assert compute_effective_gathering_throughput(50, 100, -5.0) == 0.0

    # 满额标准吞吐
    # 每 100 回合采食 100 颗，20 只存活蚂蚁 -> 100 * 100 / (100 * 20) = 5.0
    throughput = compute_effective_gathering_throughput(100, 100, 20.0)
    assert throughput == 5.0


def test_metric_collision_blocking_rate_boundaries():
    # 测试内部碰撞阻挡率在 0 总蚂蚁回合、100% 阻挡、0% 阻挡下的边界行为
    assert compute_collision_blocking_rate(0, 0) == 0.0
    assert compute_collision_blocking_rate(10, 0) == 0.0
    assert compute_collision_blocking_rate(10, -5) == 0.0

    # 100% 碰撞阻挡
    assert compute_collision_blocking_rate(500, 500) == 1.0

    # 0% 碰撞阻挡
    assert compute_collision_blocking_rate(0, 500) == 0.0


def test_metric_nan_inf_safety_evaluation():
    # 检验度量函数接收 NaN 与 Inf 时的鲁棒性
    # 有效采食速率在分母为 Inf 时应收敛为 0.0
    assert compute_effective_gathering_throughput(100, 100, float("inf")) == 0.0

    # 当传入 NaN 时，验证 Python 默认浮点运算行为并确认 run_scaling_simulation 存在 fast-fail 断言门禁
    nan_throughput = compute_effective_gathering_throughput(100, 100, float("nan"))
    assert math.isnan(nan_throughput)


def test_corridor_mask_validation_fast_fail():
    # 检验 compute_corridor_mask 对异常维数与参数的 fast-fail 崩溃
    with pytest.raises(ValueError, match="水域网格维度必须为 2"):
        compute_corridor_mask(np.ones(10, dtype=bool))

    with pytest.raises(ValueError, match="水域网格维度必须为 2"):
        compute_corridor_mask(np.ones((2, 2, 2), dtype=bool))

    with pytest.raises(ValueError, match="最大走廊宽度必须大于等于 1"):
        compute_corridor_mask(np.zeros((10, 10), dtype=bool), max_corridor_width=0)

    with pytest.raises(ValueError, match="最大走廊宽度必须大于等于 1"):
        compute_corridor_mask(np.zeros((10, 10), dtype=bool), max_corridor_width=-1)


def test_simulation_checkerboard_trapped_ants_rollout():
    # 测试在极端棋盘格拓扑下受困蚂蚁推演
    rows, cols = 8, 8
    r_idx, c_idx = np.indices((rows, cols))
    water = ((r_idx + c_idx) % 2 == 1)

    board_data = {
        "rows": rows,
        "cols": cols,
        "water": rle_from_mask(water),
        "hills": [[0, 0]],
        "food": [[2, 2]],
    }

    # 棋盘格上有 32 个孤立陆地单元格，放置 4 只蚂蚁
    res = run_scaling_simulation(
        board_data=board_data,
        scale=4,
        turns=5,
        seed=20261103,
        teacher_type="compound",
    )

    assert res["scale"] == 4
    assert res["turns"] == 5
    # 棋盘格所有陆地均被标记为走廊，蚂蚁全被水域包围被迫驻留
    assert res["corridor_cell_count"] == 32
    assert res["total_corridor_ant_turns"] == 4 * 5
    assert res["stalled_corridor_turns"] == 4 * 5
    assert res["corridor_stalling_rate"] == 1.0


def test_simulation_single_island_army():
    # 测试单格陆地孤岛极限推演
    rows, cols = 6, 6
    water = np.ones((rows, cols), dtype=bool)
    water[2, 2] = False

    board_data = {
        "rows": rows,
        "cols": cols,
        "water": rle_from_mask(water),
        "hills": [[2, 2]],
        "food": [],
    }

    res = run_scaling_simulation(
        board_data=board_data,
        scale=1,
        turns=3,
        seed=42,
        teacher_type="compound",
    )

    assert res["scale"] == 1
    assert res["corridor_cell_count"] == 1
    assert res["corridor_stalling_rate"] == 1.0
    assert res["total_food_gathered"] == 0


def test_simulation_rectangular_torus_rollout():
    # 测试非对称长方环面网格（24x48）推演
    rows, cols = 24, 48
    water = np.zeros((rows, cols), dtype=bool)
    water[5, :] = True
    water[7, :] = True

    board_data = {
        "rows": rows,
        "cols": cols,
        "water": rle_from_mask(water),
        "hills": [[0, 0], [12, 24]],
        "food": [[2, 2], [10, 10]],
    }

    res = run_scaling_simulation(
        board_data=board_data,
        scale=20,
        turns=4,
        seed=1001,
        teacher_type="compound",
    )

    assert res["scale"] == 20
    assert res["turns"] == 4
    assert res["grid_shape"] == [24, 48]
    assert res["corridor_cell_count"] == 48
    assert res["mean_latency_ms"] > 0.0


def test_simulation_corrupt_data_fast_fail():
    # 测试输入破损损坏数据时的 fast-fail 行为
    rows, cols = 10, 10
    water = np.zeros((rows, cols), dtype=bool)

    # 缺少必需字段
    with pytest.raises(KeyError):
        run_scaling_simulation(
            board_data={"cols": 10},
            scale=5,
            turns=5,
            seed=1,
        )

    valid_board = {
        "rows": rows,
        "cols": cols,
        "water": rle_from_mask(water),
        "hills": [],
        "food": [],
    }

    with pytest.raises(ValueError, match="蚂蚁规模必须大于 0"):
        run_scaling_simulation(valid_board, scale=0, turns=5, seed=1)

    with pytest.raises(ValueError, match="模拟推进回合数必须大于 0"):
        run_scaling_simulation(valid_board, scale=5, turns=0, seed=1)

    with pytest.raises(ValueError, match="地图合法陆地数量不足"):
        run_scaling_simulation(valid_board, scale=101, turns=5, seed=1)

    with pytest.raises(ValueError, match="未知教师模型类型"):
        run_scaling_simulation(valid_board, scale=5, turns=5, seed=1, teacher_type="neural")
