from __future__ import annotations

import numpy as np

from tb_baselines.planes import remember
from tb_baselines.teacher import (
    DELTAS,
    FAR,
    HOLD,
    CompoundTeacher,
    Teacher,
    extract_food_gathering_points,
    extract_memory_food_gathering_points,
    known,
)
from tb_baselines.telemetry import HoldReason, RejectionCode, SourceType


def make_board(
    size: int = 24,
    ants: tuple[tuple[int, int], ...] = ((12, 12),),
    hills: tuple[tuple[int, int, int], ...] = (),
    food: tuple[tuple[int, int], ...] = (),
    water_coords: tuple[tuple[int, int], ...] = (),
    visible_coords: tuple[tuple[int, int], ...] | None = None,
) -> dict:
    """构造确定性测试棋盘观测字典。"""
    rows, cols = size, size
    water_mask = np.zeros((rows, cols), dtype=bool)
    for r, c in water_coords:
        if 0 <= r < rows and 0 <= c < cols:
            water_mask[r, c] = True

    seen_mask = np.ones((rows, cols), dtype=bool)
    if visible_coords is not None:
        seen_mask.fill(False)
        for r, c in visible_coords:
            if 0 <= r < rows and 0 <= c < cols:
                seen_mask[r, c] = True

    def to_rle(mask: np.ndarray) -> dict:
        flat = mask.flatten()
        diffs = np.diff(flat.astype(np.int8))
        change_indices = np.where(diffs != 0)[0] + 1
        starts = np.insert(change_indices, 0, 0)
        ends = np.append(change_indices, len(flat))
        runs = []
        for s, e in zip(starts, ends):
            runs.extend([int(flat[s]), int(e - s)])
        return {"rle": runs}

    return {
        "size": [rows, cols],
        "mine": [list(a) for a in ants],
        "ids": list(range(len(ants))),
        "foes": [],
        "food": [list(f) for f in food],
        "hills": [list(h) for h in hills],
        "water": to_rle(water_mask),
        "vis": to_rle(seen_mask),
    }


# ==============================================================================
# Suite 1: G_food 提取与环面边界正交相邻计算
# ==============================================================================


def test_extract_g_food_open_interior():
    """验证开阔地形内部单个食物的正交相邻可通行陆地采集位提取。"""
    rows, cols = 20, 20
    water = np.zeros((rows, cols), dtype=bool)
    food = [(10, 10)]

    g_food = extract_food_gathering_points(food, water, rows, cols)

    assert isinstance(g_food, np.ndarray)
    assert g_food.shape == (rows, cols)
    assert g_food.dtype == bool

    expected_neighbors = {(9, 10), (11, 10), (10, 9), (10, 11)}
    for r, c in expected_neighbors:
        assert bool(g_food[r, c]) is True

    assert not g_food[10, 10]
    assert np.count_nonzero(g_food) == 4


def test_extract_g_food_torus_top_boundary_wrap():
    """验证顶边界食物跨越环面到达底边界的向量化提取。"""
    rows, cols = 20, 20
    water = np.zeros((rows, cols), dtype=bool)
    food = [(0, 10)]

    g_food = extract_food_gathering_points(food, water, rows, cols)

    assert g_food[19, 10]
    assert g_food[1, 10]
    assert g_food[0, 9]
    assert g_food[0, 11]
    assert not g_food[0, 10]
    assert np.count_nonzero(g_food) == 4


def test_extract_g_food_torus_bottom_boundary_wrap():
    """验证底边界食物跨越环面到达顶边界的向量化提取。"""
    rows, cols = 20, 20
    water = np.zeros((rows, cols), dtype=bool)
    food = [(19, 10)]

    g_food = extract_food_gathering_points(food, water, rows, cols)

    assert g_food[0, 10]
    assert g_food[18, 10]
    assert g_food[19, 9]
    assert g_food[19, 11]
    assert not g_food[19, 10]
    assert np.count_nonzero(g_food) == 4


def test_extract_g_food_torus_left_boundary_wrap():
    """验证左边界食物跨越环面到达右边界的向量化提取。"""
    rows, cols = 20, 20
    water = np.zeros((rows, cols), dtype=bool)
    food = [(10, 0)]

    g_food = extract_food_gathering_points(food, water, rows, cols)

    assert g_food[10, 19]
    assert g_food[10, 1]
    assert g_food[9, 0]
    assert g_food[11, 0]
    assert not g_food[10, 0]
    assert np.count_nonzero(g_food) == 4


def test_extract_g_food_torus_right_boundary_wrap():
    """验证右边界食物跨越环面到达左边界的向量化提取。"""
    rows, cols = 20, 20
    water = np.zeros((rows, cols), dtype=bool)
    food = [(10, 19)]

    g_food = extract_food_gathering_points(food, water, rows, cols)

    assert g_food[10, 0]
    assert g_food[10, 18]
    assert g_food[9, 19]
    assert g_food[11, 19]
    assert not g_food[10, 19]
    assert np.count_nonzero(g_food) == 4


def test_extract_g_food_torus_corner_bidirectional_wrap():
    """验证左上角食物双向跨越环面边界的提取完整性。"""
    rows, cols = 20, 20
    water = np.zeros((rows, cols), dtype=bool)
    food = [(0, 0)]

    g_food = extract_food_gathering_points(food, water, rows, cols)

    assert g_food[19, 0]
    assert g_food[0, 19]
    assert g_food[1, 0]
    assert g_food[0, 1]
    assert not g_food[0, 0]
    assert np.count_nonzero(g_food) == 4


def test_extract_g_food_water_filtering():
    """验证邻接水域时水域单元格被严格剔除。"""
    rows, cols = 20, 20
    water = np.zeros((rows, cols), dtype=bool)
    water[9, 10] = True
    water[10, 11] = True
    food = [(10, 10)]

    g_food = extract_food_gathering_points(food, water, rows, cols)

    assert not g_food[9, 10]
    assert not g_food[10, 11]
    assert g_food[11, 10]
    assert g_food[10, 9]
    assert np.count_nonzero(g_food) == 2


def test_extract_g_food_fully_surrounded_by_water():
    """验证四面环水的食物产生空采集位集合。"""
    rows, cols = 20, 20
    water = np.zeros((rows, cols), dtype=bool)
    water[9, 10] = True
    water[11, 10] = True
    water[10, 9] = True
    water[10, 11] = True
    food = [(10, 10)]

    g_food = extract_food_gathering_points(food, water, rows, cols)
    assert np.count_nonzero(g_food) == 0


# ==============================================================================
# Suite 2: 多食物实体采集位去重与相邻实体过滤
# ==============================================================================


def test_deduplication_collinear_pair_shared_cell():
    """验证同一直线上两食物共享单个相邻陆地格时的唯一性去重。"""
    rows, cols = 20, 20
    water = np.zeros((rows, cols), dtype=bool)
    food = [(5, 5), (5, 7)]

    g_food = extract_food_gathering_points(food, water, rows, cols)

    assert g_food[5, 6]
    assert np.count_nonzero(g_food) == 7


def test_deduplication_diagonal_pair_shared_two_cells():
    """验证对角分布的两食物共享两个相邻陆地格时的唯一性去重。"""
    rows, cols = 20, 20
    water = np.zeros((rows, cols), dtype=bool)
    food = [(5, 5), (6, 6)]

    g_food = extract_food_gathering_points(food, water, rows, cols)

    assert g_food[5, 6]
    assert g_food[6, 5]
    assert np.count_nonzero(g_food) == 6


def test_deduplication_triplet_t_junction_shared_central_cell():
    """验证呈 T 字形分布的三食物共享中心格时的唯一性去重。"""
    rows, cols = 20, 20
    water = np.zeros((rows, cols), dtype=bool)
    food = [(4, 5), (6, 5), (5, 4)]

    g_food = extract_food_gathering_points(food, water, rows, cols)

    assert g_food[5, 5]
    expected = {(3, 5), (4, 4), (4, 6), (5, 3), (5, 5), (6, 4), (6, 6), (7, 5)}
    assert np.count_nonzero(g_food) == len(expected)
    for r, c in expected:
        assert g_food[r, c]


def test_deduplication_quadruplet_cross_shared_central_cell():
    """验证呈十字环绕的四食物共享中心格时的唯一性去重。"""
    rows, cols = 20, 20
    water = np.zeros((rows, cols), dtype=bool)
    food = [(4, 5), (6, 5), (5, 4), (5, 6)]

    g_food = extract_food_gathering_points(food, water, rows, cols)

    assert g_food[5, 5]
    for r, c in food:
        assert not g_food[r, c]


def test_adjacent_food_entities_mutually_excluded():
    """验证相邻食物实体格相互排除，绝不作为采集位。"""
    rows, cols = 20, 20
    water = np.zeros((rows, cols), dtype=bool)
    food = [(5, 5), (5, 6)]

    g_food = extract_food_gathering_points(food, water, rows, cols)

    assert not g_food[5, 5]
    assert not g_food[5, 6]

    expected = {(4, 5), (6, 5), (5, 4), (4, 6), (6, 6), (5, 7)}
    assert np.count_nonzero(g_food) == len(expected)
    for r, c in expected:
        assert g_food[r, c]


# ==============================================================================
# Suite 3: 禁止蚂蚁直接步入食物实体格
# ==============================================================================


def test_ant_adjacent_to_food_chooses_hold():
    """验证当蚂蚁位于合法采集位（正交相邻食物）时决策保持 HOLD 采食。"""
    obs = make_board(size=24, ants=((10, 9),), food=((10, 10),))
    water, seen = known(obs)
    teacher = Teacher(hill_lead=6)

    orders = teacher.orders(obs, water, seen)
    assert orders[0] == HOLD


def test_candidate_rejections_records_food_for_blocked_move():
    """验证朝向食物实体移动的候选动作在遥测剔除字典中记录为 FOOD。"""
    obs = make_board(size=24, ants=((10, 9),), food=((10, 10),))
    water, seen = known(obs)
    teacher = Teacher(hill_lead=6)

    orders, telemetry = teacher.orders_with_telemetry(obs, water, seen)
    assert orders[0] == HOLD
    assert len(telemetry) == 1

    rec = telemetry[0]
    assert 1 in rec.candidate_rejections
    assert rec.candidate_rejections[1] == RejectionCode.FOOD.value


def test_ant_surrounded_by_food_three_sides_never_steps_into_food():
    """验证蚂蚁三面为食物时绝不选择步入食物实体。"""
    food = ((9, 10), (10, 11), (11, 10))
    obs = make_board(size=24, ants=((10, 10),), food=food)
    water, seen = known(obs)
    teacher = Teacher(hill_lead=6)

    orders = teacher.orders(obs, water, seen)
    action = orders[0]

    assert action == HOLD
    dr, dc = DELTAS[action] if action != HOLD else (0, 0)
    dest = (10 + dr, 10 + dc)
    assert dest not in food


def test_multi_ant_dense_cluster_no_destination_in_food():
    """验证密集多兵场景下所有蚂蚁的目标方格均不落在食物实体上。"""
    ants = (
        (10, 8), (10, 9), (11, 8), (11, 9),
        (12, 8), (12, 9), (9, 8), (9, 9),
    )
    food = ((10, 10), (11, 10), (12, 10))
    obs = make_board(size=24, ants=ants, food=food)
    water, seen = known(obs)
    ct = CompoundTeacher(min_raid_ants=4, raid_ratio=0.5)

    orders = ct.orders(obs, water, seen)
    rows, cols = obs["size"]
    food_set = set(food)

    for i, a in enumerate(orders):
        ar, ac = ants[i]
        if a == HOLD:
            dest = (ar, ac)
        else:
            dr, dc = DELTAS[a]
            dest = ((ar + dr) % rows, (ac + dc) % cols)
        assert dest not in food_set


def test_potential_field_food_entity_is_far():
    """验证势能场中食物实体单元格自身为 FAR，合法采集位具有最低势能。"""
    obs = make_board(size=24, ants=((12, 12),), food=((10, 10),))
    water, seen = known(obs)
    teacher = Teacher(hill_lead=6)
    dist = teacher.field(obs, water, seen)

    assert dist[10, 10] == FAR
    expected_gathering = {(9, 10), (11, 10), (10, 9), (10, 11)}
    for r, c in expected_gathering:
        assert dist[r, c] == teacher.hill_lead


# ==============================================================================
# Suite 4: 记忆食物采集位提取与集合隔离
# ==============================================================================


def test_extract_memory_food_gathering_points_basic():
    """验证迷雾中记忆食物的正交相邻采集位提取。"""
    rows, cols = 20, 20
    water = np.zeros((rows, cols), dtype=bool)
    visible_food = np.zeros((rows, cols), dtype=bool)
    g_food = np.zeros((rows, cols), dtype=bool)
    memory_food = [(10, 10)]

    g_mem = extract_memory_food_gathering_points(
        memory_food, water, visible_food, g_food, rows, cols
    )

    expected = {(9, 10), (11, 10), (10, 9), (10, 11)}
    assert np.count_nonzero(g_mem) == 4
    for r, c in expected:
        assert g_mem[r, c]


def test_extract_memory_food_gathering_points_excludes_visible_g_food():
    """验证当记忆食物采集位与可见食物采集位重叠时，重叠格完全由可见采集位占有。"""
    rows, cols = 20, 20
    water = np.zeros((rows, cols), dtype=bool)
    visible_food_coords = [(5, 5)]
    visible_food = np.zeros((rows, cols), dtype=bool)
    visible_food[5, 5] = True
    g_food = extract_food_gathering_points(visible_food_coords, water, rows, cols)

    memory_food = [(5, 7)]
    g_mem = extract_memory_food_gathering_points(
        memory_food, water, visible_food, g_food, rows, cols
    )

    assert g_food[5, 6]
    assert not g_mem[5, 6]
    assert not (g_food & g_mem).any()


def test_extract_memory_food_gathering_points_water_and_visible_food_filtering():
    """验证记忆食物相邻的水域和可见食物实体被严格过滤。"""
    rows, cols = 20, 20
    water = np.zeros((rows, cols), dtype=bool)
    water[9, 10] = True

    visible_food = np.zeros((rows, cols), dtype=bool)
    visible_food[10, 11] = True

    g_food = extract_food_gathering_points([(10, 11)], water, rows, cols)
    memory_food = [(10, 10)]

    g_mem = extract_memory_food_gathering_points(
        memory_food, water, visible_food, g_food, rows, cols
    )

    assert not g_mem[9, 10]
    assert not g_mem[10, 11]
    assert g_mem[11, 10]
    assert g_mem[10, 9]


def test_extract_memory_food_empty_returns_zeros():
    """验证无记忆食物时返回全 False 数组。"""
    rows, cols = 16, 16
    water = np.zeros((rows, cols), dtype=bool)
    visible_food = np.zeros((rows, cols), dtype=bool)
    g_food = np.zeros((rows, cols), dtype=bool)

    g_mem = extract_memory_food_gathering_points(
        [], water, visible_food, g_food, rows, cols
    )
    assert g_mem.shape == (rows, cols)
    assert np.count_nonzero(g_mem) == 0


# ==============================================================================
# Suite 5: 物理阻挡与迷雾记忆严格隔离
# ==============================================================================


def test_fog_memory_food_is_passable_for_pathfinding():
    """验证迷雾中的记忆食物不构成通行阻挡，波前扩散可自然穿透。"""
    size = 24
    water_coords = []
    for c in range(5, 16):
        water_coords.append((9, c))
        water_coords.append((11, c))

    visible_coords = [(10, c) for c in range(4, 8)]
    obs = make_board(
        size=size,
        ants=((10, 5),),
        hills=((10, 15, 1),),
        food=(),
        water_coords=tuple(water_coords),
        visible_coords=tuple(visible_coords),
    )
    water, seen = known(obs)

    ct = CompoundTeacher(min_raid_ants=1, raid_ratio=1.0)
    raid_dist = ct.field_raid(obs, water, {(10, 15)})

    assert raid_dist[10, 5] == 10
    assert raid_dist[10, 10] == 5


def test_visible_food_blocks_pathfinding_in_corridor():
    """验证可见食物构成刚性通行阻挡，波前扩散无法穿透。"""
    size = 24
    water_coords = []
    for c in range(5, 16):
        water_coords.append((9, c))
        water_coords.append((11, c))

    obs = make_board(
        size=size,
        ants=((10, 5),),
        hills=((10, 15, 1),),
        food=((10, 10),),
        water_coords=tuple(water_coords),
    )
    water, seen = known(obs)

    teacher = Teacher(hill_lead=15)
    dist = teacher.field(obs, water, seen)
    assert dist[10, 10] == FAR
    orders = teacher.orders(obs, water, seen)
    assert orders[0] != 1


def test_orders_permit_stepping_onto_fog_memory_food_coordinate():
    """验证迷雾中的历史记忆食物方格允许蚂蚁正常步入，不被错误判定为 FOOD 剔除。"""
    visible_coords = [(10, 9), (9, 9), (11, 9), (10, 8)]
    obs = make_board(
        size=24,
        ants=((10, 9),),
        food=(),
        visible_coords=tuple(visible_coords),
    )
    water, seen = known(obs)
    memory = {"hill_foe_seen": [], "food_seen": [[10, 10]]}

    teacher = Teacher(hill_lead=6)
    orders, telemetry = teacher.orders_with_telemetry(obs, water, seen, memory)

    rec = telemetry[0]
    rejection = rec.candidate_rejections.get(1)
    assert rejection != RejectionCode.FOOD.value


# ==============================================================================
# Suite 6: 优先级分级机制 (b_visible = 0 vs b_memory = 1)
# ==============================================================================


def test_equidistant_ant_prefers_visible_food_over_memory_food():
    """验证等距情况下蚂蚁确定性优先采食可见食物。"""
    visible_coords = [(r, c) for r in range(8, 17) for c in range(11, 18)]
    obs = make_board(
        size=24,
        ants=((12, 12),),
        food=((12, 15),),
        visible_coords=tuple(visible_coords),
    )
    water, seen = known(obs)
    memory = {"hill_foe_seen": [], "food_seen": [[12, 9]]}

    teacher = Teacher(hill_lead=6)
    orders = teacher.orders(obs, water, seen, memory)

    assert orders[0] == 1


def test_closer_memory_food_preferred_when_distance_overcomes_bias():
    """验证当记忆食物距离更近且足以弥补优先级偏置时蚂蚁有效利用记忆。"""
    visible_coords = [(r, c) for r in range(9, 16) for c in range(11, 21)]
    obs = make_board(
        size=24,
        ants=((12, 12),),
        food=((12, 19),),
        visible_coords=tuple(visible_coords),
    )
    water, seen = known(obs)
    memory = {"hill_foe_seen": [], "food_seen": [[12, 10]]}

    teacher = Teacher(hill_lead=6)
    orders, telemetry = teacher.orders_with_telemetry(obs, water, seen, memory)

    assert orders[0] == 3
    assert len(telemetry) == 1
    assert telemetry[0].nearest_source_type == SourceType.FOOD_MEMORY.value


# ==============================================================================
# Suite 7: 视野点亮动态清除与失效记忆即时失效
# ==============================================================================


def test_revealing_empty_food_location_clears_memory_and_target():
    """验证点亮曾经记忆有食物但实际为空的单元格时，where 机制即时清除该记忆。"""
    rows, cols = 24, 24
    initial_memory = {"food_seen": [[10, 10]], "hill_foe_seen": []}

    obs = make_board(size=rows, ants=((10, 9),), food=())
    water, seen = known(obs)

    new_memory = remember(initial_memory, obs, mode="where")

    assert [10, 10] not in new_memory.get("food_seen", [])

    visible_food = np.zeros((rows, cols), dtype=bool)
    g_food = np.zeros((rows, cols), dtype=bool)
    g_mem = extract_memory_food_gathering_points(
        new_memory.get("food_seen", []), water, visible_food, g_food, rows, cols
    )
    assert np.count_nonzero(g_mem) == 0


def test_revealing_existing_food_location_promotes_to_visible_tier():
    """验证点亮确认仍有食物的单元格时自动升级为可见食物采集位。"""
    rows, cols = 24, 24
    initial_memory = {"food_seen": [[10, 10]], "hill_foe_seen": []}

    obs = make_board(size=rows, ants=((10, 9),), food=((10, 10),))
    water, seen = known(obs)

    new_memory = remember(initial_memory, obs, mode="where")
    assert [10, 10] in new_memory.get("food_seen", [])

    visible_food = np.zeros((rows, cols), dtype=bool)
    visible_food[10, 10] = True
    g_food = extract_food_gathering_points([(10, 10)], water, rows, cols)
    g_mem = extract_memory_food_gathering_points(
        new_memory.get("food_seen", []), water, visible_food, g_food, rows, cols
    )

    assert np.count_nonzero(g_food) == 4
    assert np.count_nonzero(g_mem) == 0


# ==============================================================================
# Suite 8: 全量回归安全校验
# ==============================================================================


def test_regression_telemetry_harvest_wait_compatible():
    """验证位于采集位上的蚂蚁与遥测 HARVEST_WAIT 状态判定完全相容。"""
    obs = make_board(size=24, ants=((10, 9),), food=((10, 10),))
    water, seen = known(obs)
    teacher = Teacher(hill_lead=6)

    orders, telemetry = teacher.orders_with_telemetry(obs, water, seen)
    assert orders[0] == HOLD
    assert len(telemetry) == 1

    rec = telemetry[0]
    assert rec.hold_reason == HoldReason.HARVEST_WAIT.value
    assert rec.is_intentional_hold is True
    assert rec.nearest_source_type == SourceType.FOOD_VISIBLE.value


def test_regression_plain_teacher_and_compound_teacher_compatibility():
    """验证基础教师与复合教师的既有调用接口与返回类型保持一致。"""
    obs = make_board(size=24, ants=((12, 12), (12, 13)), food=((10, 10),))
    water, seen = known(obs)

    t = Teacher()
    ct = CompoundTeacher()

    orders_t = t.orders(obs, water, seen)
    orders_ct = ct.orders(obs, water, seen)

    assert isinstance(orders_t, np.ndarray)
    assert isinstance(orders_ct, np.ndarray)
    assert orders_t.shape == (2,)
    assert orders_ct.shape == (2,)
