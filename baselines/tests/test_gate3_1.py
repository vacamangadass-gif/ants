import numpy as np
import pytest

from tb_baselines.coordination import (
    FAR,
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


def test_joint_condition_dataclass_contract():
    cond = make_joint_condition(123456789, 1, 0)
    assert isinstance(cond, JointCondition)
    assert cond.feature_hash == 123456789
    assert cond.role == 1
    assert cond.memory_state == 0

    # 异常输入快速就地崩溃
    with pytest.raises(TypeError):
        make_joint_condition("not_int", 1, 0)  # type: ignore
    with pytest.raises(ValueError):
        make_joint_condition(123, -1, 0)
    with pytest.raises(ValueError):
        make_joint_condition(123, 1, -1)


def test_receptive_field_slice_extraction_torus():
    # 构造 24x24 小型棋盘（边界尺寸小于 31x31 感受野）
    rows, cols = 24, 24
    board = np.zeros((7, rows, cols), dtype=np.int8)
    # (0, 0) 放置食物 (通道 2)
    board[2, 0, 0] = 1
    # (23, 23) 放置水域 (通道 3)
    board[3, 23, 23] = 1

    # 在 (0, 0) 提取 31x31 感受野切片
    rf_slice = extract_receptive_field_slice(board, (0, 0), radius=RF_RADIUS)
    assert rf_slice.shape == (RF_DIAMETER, RF_DIAMETER, 7)
    # 中心位置 (15, 15) 应对应原棋盘 (0, 0)
    assert rf_slice[RF_RADIUS, RF_RADIUS, 2] == 1
    # 原棋盘 (23, 23) 相对于 (0, 0) 偏移为 (-1, -1)，对应切片 (14, 14)
    assert rf_slice[RF_RADIUS - 1, RF_RADIUS - 1, 3] == 1

    # 批次提取测试
    batch_slices = extract_batch_receptive_field_slices(
        board, [(0, 0), (23, 23)], radius=RF_RADIUS
    )
    assert batch_slices.shape == (2, RF_DIAMETER, RF_DIAMETER, 7)
    assert batch_slices[0, RF_RADIUS, RF_RADIUS, 2] == 1
    assert batch_slices[1, RF_RADIUS, RF_RADIUS, 3] == 1

    # 异常输入快速就地崩溃
    with pytest.raises(TypeError):
        extract_receptive_field_slice(board.astype(np.float32), (0, 0))
    with pytest.raises(ValueError):
        extract_receptive_field_slice(board, (0, 0), radius=10)
    with pytest.raises(ValueError):
        extract_receptive_field_slice(board, (30, 0))


def test_discrete_feature_hash_and_d4_canonicalization():
    # 构造向北移动的切片
    patch_north = np.zeros((RF_DIAMETER, RF_DIAMETER, 7), dtype=np.int8)
    patch_north[RF_RADIUS - 1, RF_RADIUS, 2] = 1  # 食物在正北 1 格
    h_n, a_n = canonicalize_receptive_field_and_action(patch_north, action=0)  # 动作 N

    # 构造向东移动的切片 (逆时针旋转 90 度得到的等价场景)
    patch_east = np.rot90(patch_north, k=3, axes=(0, 1))
    h_e, a_e = canonicalize_receptive_field_and_action(patch_east, action=1)  # 动作 E

    # 构造向南移动的切片 (旋转 180 度得到的等价场景)
    patch_south = np.rot90(patch_north, k=2, axes=(0, 1))
    h_s, a_s = canonicalize_receptive_field_and_action(patch_south, action=2)  # 动作 S

    # 断言对称同构切片的规范哈希与规范动作完全恒等
    assert h_n == h_e == h_s
    assert a_n == a_e == a_s

    # 验证单切片哈希为非负 42 比特无符号整数范围
    raw_hash = compute_single_patch_hash(patch_north)
    assert isinstance(raw_hash, int)
    assert 0 <= raw_hash < (1 << 42)


def test_empirical_entropy_analytical_distributions():
    # 场景 1: 完全确定性动作 (P = 1.0)，熵严格为 0.0 bit
    cond_a = make_joint_condition(1001, 1, 0)
    actions_det = [0] * 20
    conditions_det = [cond_a] * 20
    entropy_det, meta_det = compute_empirical_conditional_entropy(
        actions_det, conditions_det, min_category_samples=10
    )
    assert np.isclose(entropy_det, 0.0)
    assert meta_det["valid_samples"] == 20
    assert meta_det["coverage_ratio"] == 1.0

    # 场景 2: 二值均匀分布 (P = 0.5, 0.5)，熵严格为 1.0 bit
    actions_bin = [0] * 10 + [1] * 10
    conditions_bin = [cond_a] * 20
    entropy_bin, meta_bin = compute_empirical_conditional_entropy(
        actions_bin, conditions_bin, min_category_samples=10
    )
    assert np.isclose(entropy_bin, 1.0)

    # 场景 3: 五元均匀分布 (P = 0.2)，熵严格为 log2(5) ~= 2.321928 bit
    actions_5 = [0] * 10 + [1] * 10 + [2] * 10 + [3] * 10 + [4] * 10
    conditions_5 = [cond_a] * 50
    entropy_5, meta_5 = compute_empirical_conditional_entropy(
        actions_5, conditions_5, min_category_samples=10
    )
    assert np.isclose(entropy_5, np.log2(5.0))


def test_finite_sample_long_tail_filtering():
    cond_main = make_joint_condition(100, 1, 0)
    cond_rare1 = make_joint_condition(201, 1, 0)
    cond_rare2 = make_joint_condition(202, 1, 0)

    # 主类别样本数为 20 (>= 10)，罕见类别样本数各为 1 (< 10)
    actions = [0] * 20 + [1] + [2]
    conditions = [cond_main] * 20 + [cond_rare1, cond_rare2]

    entropy, meta = compute_empirical_conditional_entropy(
        actions, conditions, min_category_samples=10
    )
    # 孤立单样本被正确剔除，未稀释计算
    assert meta["total_samples"] == 22
    assert meta["valid_samples"] == 20
    assert meta["total_categories"] == 3
    assert meta["valid_categories"] == 1
    assert np.isclose(meta["coverage_ratio"], 20.0 / 22.0)
    assert np.isclose(entropy, 0.0)

    # 全部为孤立单样本时快速就地崩溃
    with pytest.raises(ValueError):
        compute_empirical_conditional_entropy([0, 1], [cond_rare1, cond_rare2], min_category_samples=10)


def test_symmetry_equivalent_actions_normalization():
    cond = make_joint_condition(300, 1, 0)
    # 对角线对称移动意向：North (0) 与 East (1) 等价
    actions = [0] * 10 + [1] * 10
    conditions = [cond] * 20

    # 未提供等价规范化：判定为策略分歧冲突，熵为 1.0 bit
    entropy_raw, _ = compute_empirical_conditional_entropy(actions, conditions, min_category_samples=10)
    assert np.isclose(entropy_raw, 1.0)

    # 施加等价类规范化映射集合
    equiv_actions = [{0, 1} for _ in range(20)]
    entropy_norm, _ = compute_empirical_conditional_entropy(
        actions, conditions, min_category_samples=10, optimal_equivalent_actions=equiv_actions
    )
    # 等价动作规范化后熵消减为 0.0 bit
    assert np.isclose(entropy_norm, 0.0)


def test_gate_3_1_passgate_decision():
    status, report = evaluate_gate3_1(
        conditional_entropy=0.35, threshold=0.50, min_coverage_ratio=0.70, metadata={"coverage_ratio": 0.85}
    )
    assert status == GateStatus.PASS
    assert report["status"] == "PASS"
    assert report["projection_compensation_triggered"] is False


def test_gate_3_1_failgate_decision():
    # 场景 1: 条件熵超标 (0.65 > 0.50)
    status_1, report_1 = evaluate_gate3_1(
        conditional_entropy=0.65, threshold=0.50, min_coverage_ratio=0.70, metadata={"coverage_ratio": 0.85}
    )
    assert status_1 == GateStatus.FAIL
    assert report_1["status"] == "FAIL"
    assert report_1["projection_compensation_triggered"] is True

    # 场景 2: 覆盖率不足 (0.50 < 0.70)
    status_2, report_2 = evaluate_gate3_1(
        conditional_entropy=0.20, threshold=0.50, min_coverage_ratio=0.70, metadata={"coverage_ratio": 0.50}
    )
    assert status_2 == GateStatus.FAIL
    assert report_2["projection_compensation_triggered"] is True


def test_feature_projection_compensation_entropy_reduction():
    rows, cols = 40, 40
    # 两个相反方向的目标：目标 1 位于远东 (10, 35)，目标 2 位于远西 (10, 5)
    target_east = (10, 35)
    target_west = (10, 5)

    field_east = np.full((rows, cols), FAR, dtype=np.int32)
    field_west = np.full((rows, cols), FAR, dtype=np.int32)

    for r in range(rows):
        for c in range(cols):
            dr_e = min(abs(r - target_east[0]), rows - abs(r - target_east[0]))
            dc_e = min(abs(c - target_east[1]), cols - abs(c - target_east[1]))
            field_east[r, c] = dr_e + dc_e

            dr_w = min(abs(r - target_west[0]), rows - abs(r - target_west[0]))
            dc_w = min(abs(c - target_west[1]), cols - abs(c - target_west[1]))
            field_west[r, c] = dr_w + dc_w

    # 在缺少全局信息时，位于中央 (10, 20) 的蚂蚁由于局部切片完全同构，动作各半冲突
    cond_blind = make_joint_condition(500, 1, 0)
    actions_blind = [1] * 10 + [3] * 10  # 10 只向 East (1)，10 只向 West (3)
    conditions_blind = [cond_blind] * 20

    entropy_blind, _ = compute_empirical_conditional_entropy(
        actions_blind, conditions_blind, min_category_samples=10
    )
    # 认知遮蔽导致条件熵高达 1.0 bit，触发 FailGate
    assert np.isclose(entropy_blind, 1.0)
    status_blind, _ = evaluate_gate3_1(entropy_blind, threshold=0.50)
    assert status_blind == GateStatus.FAIL

    # 启用特征投影补偿机制：构建连续势能特征平面
    proj_plane_east = generate_flow_projection_plane([target_east], [field_east], rows, cols)
    proj_plane_west = generate_flow_projection_plane([target_west], [field_west], rows, cols)

    # 势能平面在局部单元格呈现反向梯度（East 方位导数正负相反）
    grad_e = proj_plane_east[10, 21] - proj_plane_east[10, 19]
    grad_w = proj_plane_west[10, 21] - proj_plane_west[10, 19]
    assert grad_e * grad_w < 0.0  # 梯度方向完全相反，提供了判别信息

    # 将投影梯度的离散符号编码注入联合条件中
    hash_east = (500 << 2) | (1 if grad_e > 0 else 2)
    hash_west = (500 << 2) | (1 if grad_w > 0 else 2)
    cond_comp_east = make_joint_condition(hash_east, 1, 0)
    cond_comp_west = make_joint_condition(hash_west, 1, 0)

    conditions_comp = [cond_comp_east] * 10 + [cond_comp_west] * 10
    entropy_comp, _ = compute_empirical_conditional_entropy(
        actions_blind, conditions_comp, min_category_samples=10
    )

    # 引入特征投影补偿后，条件熵由 1.0 bit 塌陷至 0.0 bit，成功通过门禁裁决
    assert np.isclose(entropy_comp, 0.0)
    status_comp, _ = evaluate_gate3_1(entropy_comp, threshold=0.50)
    assert status_comp == GateStatus.PASS


def test_geodesic_path_tracing_and_pheromone_projection():
    rows, cols = 32, 32
    passable = np.ones((rows, cols), dtype=bool)

    # 简单 BFS 生成测地场
    dist_field = np.full((rows, cols), FAR, dtype=np.int32)
    target = (10, 25)
    dist_field[target[0], target[1]] = 0
    q = [target]
    while q:
        cr, cc = q.pop(0)
        cd = dist_field[cr, cc]
        for dr, dc in ((-1, 0), (0, 1), (1, 0), (0, -1)):
            nr = (cr + dr) % rows
            nc = (cc + dc) % cols
            if passable[nr, nc] and dist_field[nr, nc] == FAR:
                dist_field[nr, nc] = cd + 1
                q.append((nr, nc))

    start = (10, 10)
    path = trace_geodesic_allocation_path(start, target, dist_field, passable, rows, cols)
    assert path[0] == start
    assert path[-1] == target
    assert len(path) == dist_field[start[0], start[1]] + 1

    # 验证沿路径距离严格单调递减
    for i in range(len(path) - 1):
        assert dist_field[path[i + 1][0], path[i + 1][1]] < dist_field[path[i][0], path[i][1]]

    # 验证特征投影信息素平面与局部梯度
    winning_allocs = [(0, start, target)]
    fields_dict = {target: dist_field}
    ph_plane, grad_map = generate_feature_projection_pheromone_plane(
        winning_allocs, fields_dict, passable, rows, cols, decay_tau=8.0, spread_radius=2
    )

    assert ph_plane.shape == (rows, cols)
    assert grad_map.shape == (rows, cols)
    assert 0.0 <= ph_plane.max() <= 1.0
    # 起始点处局部梯度应明确指向 East (动作编号 1)
    assert grad_map[start[0], start[1]] == 1


def test_gate3_1_entropy_reduction_protocol_end_to_end():
    # 模拟 40 只蚂蚁在同构局部走廊遭遇宏观任务分歧
    actions = [1] * 20 + [3] * 20  # 20 只向 East (1)，20 只向 West (3)
    base_conditions = [make_joint_condition(8888, 0, 0)] * 40

    # 1. 基础经验条件熵测量：未获补偿时条件熵高达 1.0 bit
    base_entropy, meta_base = compute_empirical_conditional_entropy(
        actions, base_conditions, min_category_samples=10
    )
    assert np.isclose(base_entropy, 1.0)
    assert meta_base["coverage_ratio"] == 1.0

    # 2. Gate 3-1 门禁裁决：超标触发 FAIL 并指令启动特征投影补偿
    status_base, report_base = evaluate_gate3_1(base_entropy, threshold=0.50)
    assert status_base == "FAIL"
    assert status_base == GateStatus.FAIL
    assert report_base["projection_compensation_triggered"] is True

    # 3. 注入特征投影梯度 directions: 1 (East) 与 3 (West)
    cond_east = make_joint_condition(8888 * 10 + 1, 0, 0)
    cond_west = make_joint_condition(8888 * 10 + 3, 0, 0)
    comp_conditions = [cond_east] * 20 + [cond_west] * 20

    # 4. 复测经验条件熵：注入投影特征后条件熵塌陷至 0.0 bit
    comp_entropy, meta_comp = compute_empirical_conditional_entropy(
        actions, comp_conditions, min_category_samples=10
    )
    assert np.isclose(comp_entropy, 0.0)

    # 5. Gate 3-1 门禁复审：通过门禁
    status_comp, report_comp = evaluate_gate3_1(comp_entropy, threshold=0.50)
    assert status_comp == "PASS"
    assert status_comp == GateStatus.PASS
    assert report_comp["projection_compensation_triggered"] is False


def test_discrete_feature_hash_blake2b_support():
    patch = np.zeros((RF_DIAMETER, RF_DIAMETER, 7), dtype=np.int8)
    patch[RF_RADIUS, RF_RADIUS + 1, 2] = 1  # 食物在正东 1 格

    h_blake = compute_discrete_feature_hash(patch, canonicalize=True, use_blake2b=True)
    assert isinstance(h_blake, int)
    assert h_blake >= 0

    # 多次调用满足严格逐位确定性
    assert h_blake == compute_discrete_feature_hash(patch, canonicalize=True, use_blake2b=True)

    # 旋转 90 度同构切片满足 D4 规范哈希恒等性
    patch_rot = np.rot90(patch, k=1, axes=(0, 1))
    h_rot = compute_discrete_feature_hash(patch_rot, canonicalize=True, use_blake2b=True)
    assert h_blake == h_rot

