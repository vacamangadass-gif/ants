import hashlib
import math
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
    generate_flow_projection_plane,
    make_joint_condition,
    trace_geodesic_allocation_path,
)


def test_receptive_field_extreme_boundary_and_sub31_torus_wraparound():
    # 检验极端坐标与小于 31x31 棋盘上的环面自包裹切片提取一致性与形状不变性
    board_dimensions = [
        (10, 10),  # 允许提取的最小极限棋盘尺寸 (10x10)
        (16, 16),  # 远小于感受野直径 31 的中间尺寸
        (24, 24),  # 官方 basic-tiny-2p 规格尺寸
        (35, 45),  # 宽高不等的非方阵棋盘
    ]

    for rows, cols in board_dimensions:
        # 构造三维特征平面 (7, rows, cols)
        board = np.zeros((7, rows, cols), dtype=np.int8)

        # 随机在棋盘上散布不同通道的特征标记
        rng = np.random.default_rng(12345 + rows * 100 + cols)
        for ch in range(7):
            num_points = max(2, (rows * cols) // 30)
            for _ in range(num_points):
                pr = int(rng.integers(0, rows))
                pc = int(rng.integers(0, cols))
                board[ch, pr, pc] = int(rng.integers(1, 10))

        # 测试极端中心点坐标
        test_centers = [
            (0, 0),                  # 左上角
            (0, cols - 1),           # 右上角
            (rows - 1, 0),           # 左下角
            (rows - 1, cols - 1),    # 右下角
            (rows // 2, cols // 2),  # 内部中心点
        ]

        batch_slices = extract_batch_receptive_field_slices(board, test_centers, radius=RF_RADIUS)
        assert batch_slices.shape == (len(test_centers), RF_DIAMETER, RF_DIAMETER, 7)

        for idx, (ar, ac) in enumerate(test_centers):
            single_slice = extract_receptive_field_slice(board, (ar, ac), radius=RF_RADIUS)

            # 验证切片几何形状严格为 (31, 31, 7)
            assert single_slice.shape == (RF_DIAMETER, RF_DIAMETER, 7)

            # 验证单切片与批次切片提取逐元素严格恒等
            np.testing.assert_array_equal(single_slice, batch_slices[idx])

            # 严格依据环面坐标变换公式检验切片中的每一个像素映射
            for dr in range(-RF_RADIUS, RF_RADIUS + 1):
                for dc in range(-RF_RADIUS, RF_RADIUS + 1):
                    expected_r = (ar + dr) % rows
                    expected_c = (ac + dc) % cols
                    slice_r = dr + RF_RADIUS
                    slice_c = dc + RF_RADIUS

                    # 逐通道核对像素数值
                    np.testing.assert_array_equal(
                        single_slice[slice_r, slice_c, :],
                        board[:, expected_r, expected_c],
                    )

        # 检验 (H, W, C) 输入通道排列格式下的等价提取能力
        board_hwc = np.transpose(board, (1, 2, 0))
        for ar, ac in [(0, 0), (rows - 1, cols - 1)]:
            slice_from_hwc = extract_receptive_field_slice(board_hwc, (ar, ac), radius=RF_RADIUS)
            slice_from_chw = extract_receptive_field_slice(board, (ar, ac), radius=RF_RADIUS)
            np.testing.assert_array_equal(slice_from_hwc, slice_from_chw)


def test_receptive_field_extraction_contract_exceptions():
    # 检验切片提取在非法输入下的就地快速崩溃机制
    valid_board = np.zeros((7, 30, 30), dtype=np.int8)

    # 1. 浮点类型输入禁止通过
    with pytest.raises(TypeError):
        extract_receptive_field_slice(valid_board.astype(np.float32), (5, 5))

    # 2. 半径不等于 15 时禁止通过
    with pytest.raises(ValueError):
        extract_receptive_field_slice(valid_board, (5, 5), radius=10)

    # 3. 蚂蚁坐标越界禁止通过
    with pytest.raises(ValueError):
        extract_receptive_field_slice(valid_board, (30, 0))
    with pytest.raises(ValueError):
        extract_receptive_field_slice(valid_board, (0, 30))
    with pytest.raises(ValueError):
        extract_receptive_field_slice(valid_board, (-1, 5))

    # 4. 棋盘尺寸低于 10x10 时禁止通过
    too_small_board = np.zeros((7, 8, 8), dtype=np.int8)
    with pytest.raises(ValueError):
        extract_receptive_field_slice(too_small_board, (0, 0))

    # 5. 批次提取空蚂蚁列表应返回空数组
    empty_batch = extract_batch_receptive_field_slices(valid_board, [], radius=RF_RADIUS)
    assert empty_batch.shape == (0, RF_DIAMETER, RF_DIAMETER, 7)


def test_blake2b_feature_hash_determinism_and_d4_invariance():
    # 检验 BLAKE2b 特征哈希在 D4 群 8 种对称变换下的严格不变性与跨调用确定性
    rng = np.random.default_rng(998877)
    patch = np.zeros((RF_DIAMETER, RF_DIAMETER, 7), dtype=np.int8)

    # 构造非对称复杂局部视野：布置非均匀食物、水域、敌友蚂蚁
    # 正北 1 格食物
    patch[RF_RADIUS - 1, RF_RADIUS, 2] = 1
    # 东北象限布置敌军
    patch[RF_RADIUS - 4, RF_RADIUS + 3, 1] = 1
    # 西南远端布置水域
    patch[RF_RADIUS + 8, RF_RADIUS - 7, 3] = 1
    # 西南近邻布置友方蚂蚁
    patch[RF_RADIUS + 2, RF_RADIUS - 1, 0] = 1

    # 计算基准哈希
    h_blake_canonical = compute_discrete_feature_hash(patch, canonicalize=True, use_blake2b=True)
    h_int_canonical = compute_discrete_feature_hash(patch, canonicalize=True, use_blake2b=False)

    # 验证哈希输出范围符合 64 位与 42 位无符号整数规范
    assert isinstance(h_blake_canonical, int)
    assert 0 <= h_blake_canonical < (1 << 64)
    assert isinstance(h_int_canonical, int)
    assert 0 <= h_int_canonical < (1 << 42)

    # 验证 1000 次重复调用下的绝对数值确定性
    for _ in range(1000):
        assert compute_discrete_feature_hash(patch, canonicalize=True, use_blake2b=True) == h_blake_canonical
        assert compute_discrete_feature_hash(patch, canonicalize=True, use_blake2b=False) == h_int_canonical

    # 遍历 D4 对称群全部 8 种空间变换
    # 变换包括：水平翻转（flip=0, 1）复合 4 种正交旋转（rot=0, 1, 2, 3）
    for flip in [False, True]:
        base_t = np.fliplr(patch) if flip else patch
        for k in range(4):
            trans_patch = np.rot90(base_t, k=k, axes=(0, 1))

            # 规范化 BLAKE2b 哈希必须对所有 8 种变换恒等
            h_trans_blake = compute_discrete_feature_hash(trans_patch, canonicalize=True, use_blake2b=True)
            assert h_trans_blake == h_blake_canonical

            # 规范化整型哈希必须对所有 8 种变换恒等
            h_trans_int = compute_discrete_feature_hash(trans_patch, canonicalize=True, use_blake2b=False)
            assert h_trans_int == h_int_canonical

    # 验证非规范化模式下不同旋转切片的哈希敏感性
    # 旋转 90 度切片在未规范化时不应相同
    patch_rot90 = np.rot90(patch, k=1, axes=(0, 1))
    raw_h_orig = compute_discrete_feature_hash(patch, canonicalize=False, use_blake2b=True)
    raw_h_rot = compute_discrete_feature_hash(patch_rot90, canonicalize=False, use_blake2b=True)
    assert raw_h_orig != raw_h_rot


def test_discrete_feature_hash_collision_rate_stress():
    # 对离散空间哈希进行抗碰撞能力实测
    rng = np.random.default_rng(556677)
    num_distinct_patches = 200
    hashes_int = set()
    hashes_blake = set()

    for i in range(num_distinct_patches):
        p = np.zeros((RF_DIAMETER, RF_DIAMETER, 7), dtype=np.int8)
        # 随机布置四个正交方向的障碍物组合
        cardinal_dirs = [(-1, 0), (0, 1), (1, 0), (0, -1)]
        for dr, dc in cardinal_dirs:
            ch_choice = int(rng.integers(0, 5))
            if ch_choice > 0:
                p[RF_RADIUS + dr, RF_RADIUS + dc, ch_choice - 1] = 1

        # 随机布置近邻与远端特征
        for _ in range(5):
            r_offset = int(rng.integers(-10, 11))
            c_offset = int(rng.integers(-10, 11))
            ch = int(rng.integers(0, 4))
            p[RF_RADIUS + r_offset, RF_RADIUS + c_offset, ch] = 1

        h_int = compute_discrete_feature_hash(p, canonicalize=True, use_blake2b=False)
        h_blake = compute_discrete_feature_hash(p, canonicalize=True, use_blake2b=True)

        hashes_int.add(h_int)
        hashes_blake.add(h_blake)

    # 离散哈希应保持较高的特征空间分辨率，非同构状态间哈希碰撞率应受到严格压制
    unique_ratio_int = len(hashes_int) / float(num_distinct_patches)
    unique_ratio_blake = len(hashes_blake) / float(num_distinct_patches)
    assert unique_ratio_int > 0.85
    assert unique_ratio_blake > 0.85


def test_symmetric_ambiguity_triggers_gate_fail_and_projection_restores_pass():
    # 构造局部对称歧义场景，实证驱动无补偿经验条件熵 H(A|C) > 0.50 bit (Gate FAIL)
    # 并注入特征投影信息素梯度 g_Phi 实证条件熵压降至 0.50 bit 以下 (Gate PASS)
    rows, cols = 48, 48
    passable = np.ones((rows, cols), dtype=bool)

    # 场景设计：环面中心走廊上有 60 只蚂蚁聚集在 (24, 24)
    # 存在两个完全对称的战略目标：
    # 目标 1: 北方战略食物点 (4, 24)，测地距离 20 步
    # 目标 2: 南方战略食物点 (44, 24)，测地距离 20 步 ((24 + 20) % 48 = 44)
    ant_start = (24, 24)
    target_north = (4, 24)
    target_south = (44, 24)

    # 计算环面测地距离场
    field_north = np.full((rows, cols), FAR, dtype=np.int32)
    field_south = np.full((rows, cols), FAR, dtype=np.int32)

    for r in range(rows):
        for c in range(cols):
            # 环面曼哈顿距离
            dr_n = min(abs(r - target_north[0]), rows - abs(r - target_north[0]))
            dc_n = min(abs(c - target_north[1]), cols - abs(c - target_north[1]))
            field_north[r, c] = dr_n + dc_n

            dr_s = min(abs(r - target_south[0]), rows - abs(r - target_south[0]))
            dc_s = min(abs(c - target_south[1]), cols - abs(c - target_south[1]))
            field_south[r, c] = dr_s + dc_s

    # 验证起点到南北两目标的测地距离完全对称且相等
    assert field_north[ant_start[0], ant_start[1]] == 20
    assert field_south[ant_start[0], ant_start[1]] == 20

    # 回溯最优测地路径
    path_n = trace_geodesic_allocation_path(ant_start, target_north, field_north, passable, rows, cols)
    path_s = trace_geodesic_allocation_path(ant_start, target_south, field_south, passable, rows, cols)

    # 北向第一步向北走向 (23, 24) -> 动作 North (编号 0)
    assert path_n[1] == ((ant_start[0] - 1) % rows, ant_start[1])
    # 南向第一步向南走向 (25, 24) -> 动作 South (编号 2)
    assert path_s[1] == ((ant_start[0] + 1) % rows, ant_start[1])

    # 阶段 1: 无补偿对称歧义
    # 60 只蚂蚁具有完全相同的局部切片哈希与角色记忆
    # 30 只执行北向动作 (0)，30 只执行南向动作 (2)
    shared_feature_hash = 987654321
    base_conditions = [make_joint_condition(shared_feature_hash, 1, 0)] * 60
    actions = [0] * 30 + [2] * 30

    raw_entropy, raw_meta = compute_empirical_conditional_entropy(
        actions, base_conditions, min_category_samples=10
    )

    # 此时条件分布为均匀二值分布，经验条件熵严格等于 1.0 bit
    assert math.isclose(raw_entropy, 1.0, rel_tol=1e-9)
    assert raw_entropy > 0.50

    # 执行 Gate 3-1 门禁裁决：必须触发 FAIL
    status_raw, report_raw = evaluate_gate3_1(raw_entropy, threshold=0.50, metadata=raw_meta)
    assert status_raw == GateStatus.FAIL
    assert report_raw["status"] == "FAIL"
    assert report_raw["projection_compensation_triggered"] is True

    # 阶段 2: 注入特征投影信息素梯度 g_Phi 补偿机制
    # 生成北向分配流的信息素平面与梯度
    allocs_north = [(i, ant_start, target_north) for i in range(30)]
    allocs_south = [(i + 30, ant_start, target_south) for i in range(30)]

    _, grad_north = generate_feature_projection_pheromone_plane(
        allocs_north, {target_north: field_north}, passable, rows, cols, decay_tau=8.0, spread_radius=2
    )
    _, grad_south = generate_feature_projection_pheromone_plane(
        allocs_south, {target_south: field_south}, passable, rows, cols, decay_tau=8.0, spread_radius=2
    )

    # 检查起始点处提取出的信息素梯度方向
    g_phi_north = grad_north[ant_start[0], ant_start[1]]
    g_phi_south = grad_south[ant_start[0], ant_start[1]]

    # 梯度方向必须准确指示各自的前进动作方向
    assert g_phi_north == 0  # North
    assert g_phi_south == 2  # South
    assert g_phi_north != g_phi_south

    # 将信息素梯度转换为原生 Python int 以免 numpy.int32 对 64 位大哈希触发溢出
    g_phi_north_int = int(g_phi_north)
    g_phi_south_int = int(g_phi_south)
    cond_comp_north = make_joint_condition((shared_feature_hash << 4) | g_phi_north_int, 1, 0)
    cond_comp_south = make_joint_condition((shared_feature_hash << 4) | g_phi_south_int, 1, 0)
    compensated_conditions = [cond_comp_north] * 30 + [cond_comp_south] * 30

    comp_entropy, comp_meta = compute_empirical_conditional_entropy(
        actions, compensated_conditions, min_category_samples=10
    )

    # 注入补偿特征后，条件熵由 1.0 bit 塌陷至 0.0 bit
    assert math.isclose(comp_entropy, 0.0, abs_tol=1e-12)
    assert comp_entropy < 0.50

    # 执行 Gate 3-1 门禁裁决：转换为严格 PASS
    status_comp, report_comp = evaluate_gate3_1(comp_entropy, threshold=0.50, metadata=comp_meta)
    assert status_comp == GateStatus.PASS
    assert report_comp["status"] == "PASS"
    assert report_comp["projection_compensation_triggered"] is False


def test_four_way_crossroad_symmetric_ambiguity_stress():
    # 构造四向十字路口极端对称歧义场景：验证更高熵状态 (2.0 bit) 下投影补偿有效性
    rows, cols = 40, 40
    ant_center = (20, 20)
    passable = np.ones((rows, cols), dtype=bool)

    targets = {
        0: (10, 20),  # North
        1: (20, 30),  # East
        2: (30, 20),  # South
        3: (20, 10),  # West
    }

    # 100 只蚂蚁均匀分配到四个正交方向（每个方向 25 只）
    actions = [0] * 25 + [1] * 25 + [2] * 25 + [3] * 25
    center_hash = 11223344
    blind_conditions = [make_joint_condition(center_hash, 1, 0)] * 100

    raw_entropy, raw_meta = compute_empirical_conditional_entropy(actions, blind_conditions, min_category_samples=10)

    # 四元均匀分布，理论条件熵严格等于 log2(4) = 2.0 bit
    assert math.isclose(raw_entropy, 2.0, rel_tol=1e-9)
    status_raw, _ = evaluate_gate3_1(raw_entropy, threshold=0.50, metadata=raw_meta)
    assert status_raw == GateStatus.FAIL

    # 对各方向生成特征投影梯度
    comp_conditions = []
    for dir_idx in range(4):
        target_pos = targets[dir_idx]
        field = np.full((rows, cols), FAR, dtype=np.int32)
        for r in range(rows):
            for c in range(cols):
                dr = min(abs(r - target_pos[0]), rows - abs(r - target_pos[0]))
                dc = min(abs(c - target_pos[1]), cols - abs(c - target_pos[1]))
                field[r, c] = dr + dc

        allocs = [(i, ant_center, target_pos) for i in range(25)]
        _, grad_plane = generate_feature_projection_pheromone_plane(
            allocs, {target_pos: field}, passable, rows, cols, decay_tau=8.0, spread_radius=2
        )
        dir_grad = grad_plane[ant_center[0], ant_center[1]]
        assert dir_grad == dir_idx

        comp_cond = make_joint_condition((center_hash << 4) | dir_grad, 1, 0)
        comp_conditions.extend([comp_cond] * 25)

    comp_entropy, comp_meta = compute_empirical_conditional_entropy(actions, comp_conditions, min_category_samples=10)
    assert math.isclose(comp_entropy, 0.0, abs_tol=1e-12)
    status_comp, _ = evaluate_gate3_1(comp_entropy, threshold=0.50, metadata=comp_meta)
    assert status_comp == GateStatus.PASS


def test_long_tail_noise_filter_nmin_10_robustness():
    # 严格检验 N_min = 10 长尾过滤阈值的边界鲁棒性与噪声隔离能力
    cond_clean_a = make_joint_condition(1001, 1, 0)
    cond_clean_b = make_joint_condition(1002, 1, 0)

    # 1. 纯净基准数据集：类别 A (30 个样本，全部动作 0) 与类别 B (20 个样本，全部动作 1)
    base_actions = [0] * 30 + [1] * 20
    base_conds = [cond_clean_a] * 30 + [cond_clean_b] * 20

    h_base, meta_base = compute_empirical_conditional_entropy(base_actions, base_conds, min_category_samples=10)
    assert math.isclose(h_base, 0.0, abs_tol=1e-12)
    assert meta_base["total_samples"] == 50
    assert meta_base["valid_samples"] == 50
    assert meta_base["coverage_ratio"] == 1.0

    # 2. 注入 9 个孤立小样本类别（样本数分别从 1 到 9），包含完全随机构造的冲突动作
    noisy_actions = list(base_actions)
    noisy_conds = list(base_conds)
    rng = np.random.default_rng(778899)

    for k in range(1, 10):
        rare_cond = make_joint_condition(2000 + k, 1, 0)
        rare_acts = [int(rng.integers(0, 5)) for _ in range(k)]
        noisy_actions.extend(rare_acts)
        noisy_conds.extend([rare_cond] * k)

    total_noise_samples = sum(range(1, 10))  # 1+2+...+9 = 45 个噪声样本
    assert len(noisy_actions) == 50 + total_noise_samples

    # 计算注入长尾噪声后的经验条件熵
    h_noisy, meta_noisy = compute_empirical_conditional_entropy(
        noisy_actions, noisy_conds, min_category_samples=10
    )

    # 验证长尾过滤机制：
    # 所有 9 个小样本类别均因样本数 < 10 被严格滤除，有效样本数仍为 50
    assert meta_noisy["valid_samples"] == 50
    assert meta_noisy["valid_categories"] == 2
    assert meta_noisy["total_categories"] == 2 + 9
    assert math.isclose(meta_noisy["coverage_ratio"], 50.0 / float(50 + total_noise_samples), rel_tol=1e-9)

    # 验证经验条件熵未被任何稀疏小样本噪声抬高，依然严格等于 0.0 bit
    assert math.isclose(h_noisy, 0.0, abs_tol=1e-12)

    # 3. 临界跃迁测试：将其中一个噪声类别样本数补足至恰好 10
    # 该类别具有 5 个动作 0 与 5 个动作 1（条件熵 1.0 bit）
    cond_trans = make_joint_condition(9999, 1, 0)
    trans_actions = list(base_actions) + [0] * 5 + [1] * 5
    trans_conds = list(base_conds) + [cond_trans] * 10

    h_trans, meta_trans = compute_empirical_conditional_entropy(
        trans_actions, trans_conds, min_category_samples=10
    )

    # 达到 10 个样本后，该类别立即被纳入有效类别集合
    assert meta_trans["valid_samples"] == 60
    assert meta_trans["valid_categories"] == 3
    # 期望熵为 2 个纯净类 (权重 50/60，熵 0.0) + 1 个二值类 (权重 10/60，熵 1.0)
    expected_entropy = (10.0 / 60.0) * 1.0
    assert math.isclose(h_trans, expected_entropy, rel_tol=1e-9)

    # 4. 样本覆盖率安全红线测试：验证当稀疏长尾噪声占据主导导致覆盖率低于 0.70 时触发 FAIL
    # 即使保留的有效类别熵为 0.0，低覆盖率必须触发门禁拦截
    low_cov_actions = list(base_actions)
    low_cov_conds = list(base_conds)
    # 注入 30 个只有 1 个样本的孤立类别，使有效样本 50，总样本 80 (覆盖率 50/80 = 0.625 < 0.70)
    for idx in range(30):
        low_cov_conds.append(make_joint_condition(5000 + idx, 0, 0))
        low_cov_actions.append(0)

    h_low_cov, meta_low_cov = compute_empirical_conditional_entropy(
        low_cov_actions, low_cov_conds, min_category_samples=10
    )
    assert math.isclose(h_low_cov, 0.0, abs_tol=1e-12)
    assert meta_low_cov["coverage_ratio"] == 50.0 / 80.0
    assert meta_low_cov["coverage_ratio"] < 0.70

    # 判定结果必须因覆盖率不合格而拒绝通过 (FAIL)
    status_low_cov, report_low_cov = evaluate_gate3_1(
        h_low_cov, threshold=0.50, min_coverage_ratio=0.70, metadata=meta_low_cov
    )
    assert status_low_cov == GateStatus.FAIL
    assert report_low_cov["status"] == "FAIL"
    assert report_low_cov["projection_compensation_triggered"] is True

    # 5. 全孤立稀疏类别快速崩溃保护
    all_rare_conds = [make_joint_condition(8000 + i, 0, 0) for i in range(8)]
    with pytest.raises(ValueError):
        compute_empirical_conditional_entropy([0] * 8, all_rare_conds, min_category_samples=10)


def test_d4_action_canonicalization_all_eight_transforms():
    # 检验 D4 对称群全部 8 种空间变换下，局部切片与对应动作经规范化映射后的绝对一致性
    patch = np.zeros((RF_DIAMETER, RF_DIAMETER, 7), dtype=np.int8)
    # 在非对称位置设置特征：食物在 (12, 15)，即正北 3 格
    patch[RF_RADIUS - 3, RF_RADIUS, 2] = 1
    # 敌方在 (15, 18)，即正东 3 格
    patch[RF_RADIUS, RF_RADIUS + 3, 1] = 1

    # 动作 0 为向北（North）移动
    orig_action = 0
    ref_h, ref_act = canonicalize_receptive_field_and_action(patch, orig_action)

    # 检验 8 种 D4 变换：
    # 旋转与镜像不仅变换切片像素，同时将动作映射到对应变换后的空间朝向
    # 动作编号映射：0: North (-1, 0), 1: East (0, 1), 2: South (1, 0), 3: West (0, -1), 4: Hold
    def transform_action(action: int, flip: bool, rot_k: int) -> int:
        if action == HOLD:
            return HOLD
        dr, dc = DELTAS[action]
        # 先执行水平镜像 fliplr: (dr, dc) -> (dr, -dc)
        if flip:
            dc = -dc
        # 再执行 rot90(..., k=rot_k):
        # np.rot90 逆时针旋转 90 度 k 次：
        # k=1: (r, c) -> (-c, r) 即 (dr, dc) -> (-dc, dr)
        for _ in range(rot_k % 4):
            dr, dc = -dc, dr
        # 匹配回动作编号
        for act_idx, (adr, adc) in enumerate(DELTAS):
            if (adr, adc) == (dr, dc):
                return act_idx
        return HOLD

    for flip in [False, True]:
        base_patch = np.fliplr(patch) if flip else patch
        for k in range(4):
            t_patch = np.rot90(base_patch, k=k, axes=(0, 1))
            t_action = transform_action(orig_action, flip, k)

            h_trans, act_trans = canonicalize_receptive_field_and_action(t_patch, t_action)
            assert h_trans == ref_h
            assert act_trans == ref_act


def test_gate3_1_entropy_theoretical_boundary_curve():
    # 检验二值策略分歧在经验条件熵阈值 0.50 bit 处的分界判定准确性
    # 二值伯努利熵函数 H(p) = -p log2(p) - (1-p) log2(1-p)
    # 当 p = 0.20 时，H(0.20) ≈ 0.7219 bit > 0.50 bit -> 触发 FAIL
    # 当 p = 0.10 时，H(0.10) ≈ 0.4690 bit < 0.50 bit -> 触发 PASS
    cond = make_joint_condition(13579, 1, 0)

    # 用例 1: p = 0.20 (20 个动作 1，80 个动作 0)
    acts_p20 = [1] * 20 + [0] * 80
    conds_p20 = [cond] * 100
    h_p20, meta_p20 = compute_empirical_conditional_entropy(acts_p20, conds_p20, min_category_samples=10)
    expected_h_p20 = -(0.20 * math.log2(0.20) + 0.80 * math.log2(0.80))
    assert math.isclose(h_p20, expected_h_p20, rel_tol=1e-6)
    assert h_p20 > 0.50

    status_p20, rep_p20 = evaluate_gate3_1(h_p20, threshold=0.50, metadata=meta_p20)
    assert status_p20 == GateStatus.FAIL
    assert rep_p20["projection_compensation_triggered"] is True

    # 用例 2: p = 0.10 (10 个动作 1，90 个动作 0)
    acts_p10 = [1] * 10 + [0] * 90
    conds_p10 = [cond] * 100
    h_p10, meta_p10 = compute_empirical_conditional_entropy(acts_p10, conds_p10, min_category_samples=10)
    expected_h_p10 = -(0.10 * math.log2(0.10) + 0.90 * math.log2(0.90))
    assert math.isclose(h_p10, expected_h_p10, rel_tol=1e-6)
    assert h_p10 < 0.50

    status_p10, rep_p10 = evaluate_gate3_1(h_p10, threshold=0.50, metadata=meta_p10)
    assert status_p10 == GateStatus.PASS
    assert rep_p10["projection_compensation_triggered"] is False


def test_adversarial_noise_fragmentation_coverage_defense():
    # 对抗性压力测试：攻击者试图通过将冲突动作切分为大量伪小类别绕过门禁
    # 验证门禁覆盖率门限 (min_coverage_ratio = 0.70) 能否有效拦截该作弊模式
    cond_main = make_joint_condition(666, 1, 0)
    actions = [0] * 50
    conditions = [cond_main] * 50

    # 注入 50 个每个仅有 1 个样本的伪类别，动作全为冲突动作
    for fake_idx in range(50):
        fake_cond = make_joint_condition(70000 + fake_idx, 1, 0)
        conditions.append(fake_cond)
        actions.append(1)

    h_frag, meta_frag = compute_empirical_conditional_entropy(actions, conditions, min_category_samples=10)
    # 局部留存类别的条件熵为 0.0 bit
    assert math.isclose(h_frag, 0.0, abs_tol=1e-12)
    # 但覆盖率仅为 50 / 100 = 0.50 < 0.70
    assert meta_frag["coverage_ratio"] == 0.50

    # 双重安全验证：尽管熵为 0.0，但因样本覆盖率不达标，必须判定为 FAIL
    status_frag, rep_frag = evaluate_gate3_1(h_frag, threshold=0.50, min_coverage_ratio=0.70, metadata=meta_frag)
    assert status_frag == GateStatus.FAIL
    assert rep_frag["status"] == "FAIL"
    assert rep_frag["projection_compensation_triggered"] is True

