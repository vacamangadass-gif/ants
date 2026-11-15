import json
import math
from pathlib import Path
import numpy as np
import pytest

from tb_baselines.coordination import (
    DELTAS,
    FAR,
    HOLD,
    RF_DIAMETER,
    RF_RADIUS,
    GateStatus,
    JointCondition,
    canonicalize_receptive_field_and_action,
    compute_discrete_feature_hash,
    compute_empirical_conditional_entropy,
    compute_single_patch_hash,
    evaluate_gate3_1,
    extract_batch_receptive_field_slices,
    extract_receptive_field_slice,
    generate_feature_projection_pheromone_plane,
    make_joint_condition,
    trace_geodesic_allocation_path,
)


def test_torus_boundary_wrap_four_corners_seamless():
    # 构造环面四角边界与内部对照场景，验证坐标模运算在 (0, 0)、(0, W-1)、(H-1, 0)、(H-1, W-1) 处的无缝性
    # 环面拓扑下四角互为相邻单元格，需在独立棋盘上布置相对特征以防多蚂蚁视野相互叠合干扰
    rows, cols = 32, 32

    # 包含中心参考点与四个极端环面边界角落
    test_positions = [
        (rows // 2, cols // 2),  # 中心基准
        (0, 0),                  # 左上角
        (0, cols - 1),           # 右上角
        (rows - 1, 0),           # 左下角
        (rows - 1, cols - 1),    # 右下角
    ]

    slices = []
    batch_slices_list = []

    for ar, ac in test_positions:
        board = np.zeros((7, rows, cols), dtype=np.int8)
        # 在该测试点周围相对位移处布置相同结构特征：
        # North (-1, 0): 食物 (通道 2)
        # East (0, 1): 水域 (通道 3)
        # South (1, 0): 敌蚁 (通道 1)
        # West (0, -1): 己蚁 (通道 0)
        # North-East (-2, 2): 己方蚁巢 (通道 4)
        board[2, (ar - 1) % rows, ac] = 1
        board[3, ar, (ac + 1) % cols] = 1
        board[1, (ar + 1) % rows, ac] = 1
        board[0, ar, (ac - 1) % cols] = 1
        board[4, (ar - 2) % rows, (ac + 2) % cols] = 1

        rf_slice = extract_receptive_field_slice(board, (ar, ac), radius=RF_RADIUS)
        batch_slice = extract_batch_receptive_field_slices(board, [(ar, ac)], radius=RF_RADIUS)[0]

        np.testing.assert_array_equal(rf_slice, batch_slice)
        slices.append(rf_slice)
        batch_slices_list.append(batch_slice)

    ref_slice = slices[0]
    # 验证切片几何形状与中心对应
    assert ref_slice.shape == (RF_DIAMETER, RF_DIAMETER, 7)
    assert ref_slice[RF_RADIUS - 1, RF_RADIUS, 2] == 1
    assert ref_slice[RF_RADIUS, RF_RADIUS + 1, 3] == 1
    assert ref_slice[RF_RADIUS + 1, RF_RADIUS, 1] == 1
    assert ref_slice[RF_RADIUS, RF_RADIUS - 1, 0] == 1
    assert ref_slice[RF_RADIUS - 2, RF_RADIUS + 2, 4] == 1

    # 验证四个边界角点的提取结果与中心基准完全恒等
    for i in range(1, len(test_positions)):
        np.testing.assert_array_equal(slices[i], ref_slice)

    # 四个边界位置与中心基准的单切片哈希必须完全相同
    hashes = [compute_single_patch_hash(s) for s in slices]
    for h in hashes[1:]:
        assert h == hashes[0]

    # D4 规范哈希与规范动作完全恒等
    canon_results = [
        canonicalize_receptive_field_and_action(s, action=0) for s in slices
    ]
    for c in canon_results[1:]:
        assert c == canon_results[0]

    # BLAKE2b 离散特征哈希完全恒等
    blake_hashes = [
        compute_discrete_feature_hash(s, canonicalize=True, use_blake2b=True)
        for s in slices
    ]
    for bh in blake_hashes[1:]:
        assert bh == blake_hashes[0]


def test_torus_global_translation_invariance():
    # 验证整个棋盘在环面上任意平移后，切片哈希与动作映射保持严格不变
    rows, cols = 28, 28
    base_board = np.zeros((7, rows, cols), dtype=np.int8)

    # 随机布置若干特征
    base_board[2, 5, 8] = 1   # 食物
    base_board[3, 6, 8] = 1   # 水域
    base_board[1, 5, 7] = 1   # 敌蚁
    ant_base = (5, 8)

    slice_base = extract_receptive_field_slice(base_board, ant_base, radius=RF_RADIUS)
    hash_base, act_base = canonicalize_receptive_field_and_action(slice_base, action=1)

    # 测试跨越不同行列边界的大平移向量
    translations = [
        (0, 0),
        (0, cols - 1),
        (rows - 1, 0),
        (rows - 1, cols - 1),
        (13, 17),
        (rows - 5, cols - 8),
    ]

    for tr, tc in translations:
        shifted_board = np.roll(np.roll(base_board, tr, axis=1), tc, axis=2)
        ant_shifted = ((ant_base[0] + tr) % rows, (ant_base[1] + tc) % cols)

        slice_shifted = extract_receptive_field_slice(
            shifted_board, ant_shifted, radius=RF_RADIUS
        )
        hash_shifted, act_shifted = canonicalize_receptive_field_and_action(
            slice_shifted, action=1
        )

        np.testing.assert_array_equal(slice_base, slice_shifted)
        assert hash_base == hash_shifted
        assert act_base == act_shifted


def test_empirical_conditional_entropy_non_negativity_and_upper_bound():
    # 严格检验 H(A|C) 的非负性与理论上界 log2(5)
    max_theoretical_entropy = math.log2(5.0)

    # 1. 确定性分布：熵严格为 0.0
    actions_det = [2] * 50
    conds_det = [make_joint_condition(12345, 1, 0)] * 50
    h_det, _ = compute_empirical_conditional_entropy(actions_det, conds_det, min_category_samples=10)
    assert h_det >= 0.0
    assert math.isclose(h_det, 0.0, abs_tol=1e-12)

    # 2. 完全均匀 5 动作分布：达到上界 log2(5)
    actions_uniform = [0, 1, 2, 3, 4] * 20
    conds_uniform = [make_joint_condition(99999, 0, 0)] * 100
    h_max, _ = compute_empirical_conditional_entropy(actions_uniform, conds_uniform, min_category_samples=10)
    assert h_max >= 0.0
    assert math.isclose(h_max, max_theoretical_entropy, rel_tol=1e-9)
    assert h_max <= max_theoretical_entropy + 1e-12

    # 3. 随机多类别、多分布的应力测试：验证每一个分布均满足 0 <= H <= log2(5)
    rng = np.random.default_rng(20261104)
    num_samples = 5000
    num_categories = 25

    category_hashes = [int(h) for h in rng.integers(1000, 999999, size=num_categories)]
    sampled_cond_indices = rng.integers(0, num_categories, size=num_samples)
    sampled_conds = [make_joint_condition(category_hashes[idx], 1, 0) for idx in sampled_cond_indices]
    sampled_actions = [int(a) for a in rng.integers(0, 5, size=num_samples)]

    h_random, meta_random = compute_empirical_conditional_entropy(
        sampled_actions, sampled_conds, min_category_samples=15
    )
    assert 0.0 <= h_random <= max_theoretical_entropy + 1e-12
    assert meta_random["valid_samples"] <= num_samples
    assert meta_random["coverage_ratio"] >= 0.80

    # 4. 极度偏斜分布（数值下溢边界测试）
    actions_skewed = [0] * 999 + [1] * 1
    conds_skewed = [make_joint_condition(777, 1, 0)] * 1000
    h_skewed, _ = compute_empirical_conditional_entropy(actions_skewed, conds_skewed, min_category_samples=10)
    assert h_skewed >= 0.0
    p = 1.0 / 1000.0
    expected_skewed = -(0.999 * math.log2(0.999) + 0.001 * math.log2(0.001))
    assert math.isclose(h_skewed, expected_skewed, rel_tol=1e-6)


def test_empirical_conditional_entropy_data_processing_inequality():
    # 验证信息论单调性：对条件细化（增加特征）不能使期望条件熵增加 H(A|C1, C2) <= H(A|C1)
    rng = np.random.default_rng(888)
    n = 2000
    base_hashes = [101, 102, 103, 104]

    coarse_conds = []
    fine_conds = []
    actions = []

    for _ in range(n):
        bh = int(rng.choice(base_hashes))
        sub_feature = int(rng.choice([1, 2]))
        # 动作与细粒度特征相关
        if sub_feature == 1:
            act = int(rng.choice([0, 1]))
        else:
            act = int(rng.choice([2, 3]))

        coarse_conds.append(make_joint_condition(bh, 0, 0))
        fine_conds.append(make_joint_condition((bh << 2) | sub_feature, 0, 0))
        actions.append(act)

    h_coarse, _ = compute_empirical_conditional_entropy(actions, coarse_conds, min_category_samples=20)
    h_fine, _ = compute_empirical_conditional_entropy(actions, fine_conds, min_category_samples=20)

    # 细化条件后的熵必须严格小于粗粒度条件熵
    assert h_fine <= h_coarse + 1e-12
    assert h_fine < h_coarse


def test_symmetrical_conflict_resolution_across_torus_boundary():
    # 构造跨越环面水平边界 (0, 0) <-> (0, W-1) 的对称冲突场景
    rows, cols = 36, 36
    passable = np.ones((rows, cols), dtype=bool)

    # 挖出水平环面走廊（在 row 0 贯穿全图，其余行设为水域以形成纯净一维拓扑）
    passable[1:, :] = False
    passable[0, :] = True

    # 起始点位于环面原点 (0, 0)
    start_pos = (0, 0)

    # 目标 1: 向东走到 (0, 10)，距离 10 步
    target_east = (0, 10)
    # 目标 2: 向西跨越环面缝隙走到 (0, 26)，距离 10 步 ( (0 - 10) % 36 = 26 )
    target_west = (0, 26)

    # 构建距离场
    field_east = np.full((rows, cols), FAR, dtype=np.int32)
    field_west = np.full((rows, cols), FAR, dtype=np.int32)

    for c in range(cols):
        # 在 row 0 上的环面距离
        dist_e = min((c - target_east[1]) % cols, (target_east[1] - c) % cols)
        field_east[0, c] = dist_e
        dist_w = min((c - target_west[1]) % cols, (target_west[1] - c) % cols)
        field_west[0, c] = dist_w

    # 验证起点到两目标的环面距离完全对称且均为 10
    assert field_east[0, 0] == 10
    assert field_west[0, 0] == 10

    # 验证测地路径回溯在环面上正确解算：
    path_east = trace_geodesic_allocation_path(start_pos, target_east, field_east, passable, rows, cols)
    path_west = trace_geodesic_allocation_path(start_pos, target_west, field_west, passable, rows, cols)

    assert path_east[0] == start_pos
    assert path_east[-1] == target_east
    assert len(path_east) == 11
    # 东向路径第一步走向 (0, 1) -> 动作 East (1)
    assert path_east[1] == (0, 1)

    assert path_west[0] == start_pos
    assert path_west[-1] == target_west
    assert len(path_west) == 11
    # 西向路径第一步跨越环面缝隙走向 (0, 35) -> 动作 West (3)
    assert path_west[1] == (0, cols - 1)

    # 构造无补偿的对称冲突样本：40 只蚂蚁位于起点 (0, 0)
    # 20 只要去 target_east (动作 1: East)
    # 20 只要去 target_west (动作 3: West)
    blind_patch_hash = 654321
    base_conditions = [make_joint_condition(blind_patch_hash, 1, 0)] * 40
    actions = [1] * 20 + [3] * 20

    raw_entropy, _ = compute_empirical_conditional_entropy(
        actions, base_conditions, min_category_samples=10
    )
    # 局部无认知时，由于二元等概率冲突，条件熵精确为 1.0 bit
    assert math.isclose(raw_entropy, 1.0, rel_tol=1e-9)
    status_raw, report_raw = evaluate_gate3_1(raw_entropy, threshold=0.50)
    assert status_raw == GateStatus.FAIL
    assert report_raw["projection_compensation_triggered"] is True

    # 启动特征投影机制：分别对两批蚂蚁生成特征投影信息素平面与梯度
    allocs_east = [(0, start_pos, target_east)]
    dict_east = {target_east: field_east}
    _, grad_east = generate_feature_projection_pheromone_plane(
        allocs_east, dict_east, passable, rows, cols, decay_tau=8.0, spread_radius=1
    )

    allocs_west = [(1, start_pos, target_west)]
    dict_west = {target_west: field_west}
    _, grad_west = generate_feature_projection_pheromone_plane(
        allocs_west, dict_west, passable, rows, cols, decay_tau=8.0, spread_radius=1
    )

    # 检查起始点处的局部信息素梯度方向
    dir_east = grad_east[start_pos[0], start_pos[1]]
    dir_west = grad_west[start_pos[0], start_pos[1]]
    assert dir_east == 1  # East
    assert dir_west == 3  # West
    assert dir_east != dir_west

    # 将局部投影梯度编码注入联合条件
    cond_comp_east = make_joint_condition((blind_patch_hash << 3) | dir_east, 1, 0)
    cond_comp_west = make_joint_condition((blind_patch_hash << 3) | dir_west, 1, 0)
    comp_conditions = [cond_comp_east] * 20 + [cond_comp_west] * 20

    comp_entropy, _ = compute_empirical_conditional_entropy(
        actions, comp_conditions, min_category_samples=10
    )
    # 注入梯度后，条件熵塌陷为 0.0 bit，完全消除策略分歧歧义
    assert math.isclose(comp_entropy, 0.0, abs_tol=1e-12)
    status_comp, report_comp = evaluate_gate3_1(comp_entropy, threshold=0.50)
    assert status_comp == GateStatus.PASS
    assert report_comp["projection_compensation_triggered"] is False


def test_vertical_torus_boundary_symmetrical_conflict_resolution():
    # 构造跨越环面垂直边界 (0, 0) <-> (H-1, 0) 的对称冲突场景
    rows, cols = 30, 30
    passable = np.ones((rows, cols), dtype=bool)
    passable[:, 1:] = False
    passable[:, 0] = True  # 仅 col 0 垂直连通

    start_pos = (0, 0)
    target_south = (8, 0)
    target_north = (rows - 8, 0)

    field_south = np.full((rows, cols), FAR, dtype=np.int32)
    field_north = np.full((rows, cols), FAR, dtype=np.int32)

    for r in range(rows):
        field_south[r, 0] = min((r - target_south[0]) % rows, (target_south[0] - r) % rows)
        field_north[r, 0] = min((r - target_north[0]) % rows, (target_north[0] - r) % rows)

    path_south = trace_geodesic_allocation_path(start_pos, target_south, field_south, passable, rows, cols)
    path_north = trace_geodesic_allocation_path(start_pos, target_north, field_north, passable, rows, cols)

    # 南向第一步为 (1, 0) -> 动作 South (2)
    assert path_south[1] == (1, 0)
    # 北向第一步跨越环面缝隙走向 (rows-1, 0) -> 动作 North (0)
    assert path_north[1] == (rows - 1, 0)

    blind_hash = 777123
    base_conditions = [make_joint_condition(blind_hash, 0, 0)] * 30
    # 15 只向 North (0)，15 只向 South (2)
    actions = [0] * 15 + [2] * 15

    raw_entropy, _ = compute_empirical_conditional_entropy(actions, base_conditions, min_category_samples=10)
    assert math.isclose(raw_entropy, 1.0, rel_tol=1e-9)
    status_raw, _ = evaluate_gate3_1(raw_entropy, threshold=0.50)
    assert status_raw == GateStatus.FAIL

    _, grad_south = generate_feature_projection_pheromone_plane(
        [(0, start_pos, target_south)], {target_south: field_south}, passable, rows, cols
    )
    _, grad_north = generate_feature_projection_pheromone_plane(
        [(1, start_pos, target_north)], {target_north: field_north}, passable, rows, cols
    )

    assert grad_south[start_pos[0], start_pos[1]] == 2  # South
    assert grad_north[start_pos[0], start_pos[1]] == 0  # North

    cond_s = make_joint_condition((blind_hash << 3) | 2, 0, 0)
    cond_n = make_joint_condition((blind_hash << 3) | 0, 0, 0)
    comp_conditions = [cond_n] * 15 + [cond_s] * 15

    comp_entropy, _ = compute_empirical_conditional_entropy(actions, comp_conditions, min_category_samples=10)
    assert math.isclose(comp_entropy, 0.0, abs_tol=1e-12)
    status_comp, _ = evaluate_gate3_1(comp_entropy, threshold=0.50)
    assert status_comp == GateStatus.PASS


def test_telemetry_summary_integrity_and_mathematical_consistency():
    summary_path = Path("runs/exp082_gate3_1/telemetry_summary.json")
    jsonl_path = Path("runs/exp082_gate3_1/telemetry.jsonl")

    assert summary_path.exists(), f"{summary_path} 不存在"
    assert jsonl_path.exists(), f"{jsonl_path} 不存在"

    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)

    # 验证基本门禁规范配置
    assert summary["gate_id"] == "Gate 3-1"
    assert summary["threshold"] == 0.50
    assert summary["min_category_samples"] == 10

    # 验证聚合样本总数
    total_agg = summary["total_aggregated_samples"]
    assert total_agg == 810000

    # 验证单用例记录与 jsonl 逐行严格对应
    case_records = summary["case_records"]
    assert len(case_records) > 0

    jsonl_records = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                jsonl_records.append(json.loads(line.strip()))

    assert len(case_records) == len(jsonl_records)
    sum_samples = sum(rec["total_samples"] for rec in case_records)
    assert sum_samples == total_agg

    for i in range(len(case_records)):
        assert case_records[i]["map"] == jsonl_records[i]["map"]
        assert case_records[i]["scale"] == jsonl_records[i]["scale"]
        assert case_records[i]["seed"] == jsonl_records[i]["seed"]
        assert case_records[i]["total_samples"] == jsonl_records[i]["total_samples"]
        assert math.isclose(case_records[i]["empirical_entropy"], jsonl_records[i]["empirical_entropy"], abs_tol=1e-9)

    # 验证原始评估指标自洽性
    raw_eval = summary["raw_evaluation"]
    assert raw_eval["status"] == "PASS"
    raw_h = raw_eval["empirical_conditional_entropy"]
    assert 0.0 <= raw_h <= 0.50
    assert raw_eval["coverage_ratio"] >= 0.70
    assert raw_eval["valid_categories"] <= raw_eval["total_categories"]

    # 验证补偿评估指标自洽性
    comp_eval = summary["compensated_evaluation"]
    assert comp_eval["status"] == "PASS"
    comp_h = comp_eval["empirical_conditional_entropy"]
    assert 0.0 <= comp_h <= 0.50
    assert comp_eval["coverage_ratio"] >= 0.70
    assert comp_eval["valid_categories"] <= comp_eval["total_categories"]

    # 验证条件熵消减量的精确代数一致性
    reduction = summary["entropy_reduction"]
    expected_reduction = raw_h - comp_h
    assert math.isclose(reduction, expected_reduction, rel_tol=1e-9)
    assert reduction > 0.0  # 特征投影确实促成了条件熵的进一步消减
