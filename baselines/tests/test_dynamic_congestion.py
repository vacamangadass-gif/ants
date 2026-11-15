import time
import numpy as np
import pytest

from tb_baselines.coordination import (
    FAR,
    compute_batch_torus_geodesics_multibucket,
    compute_corridor_width,
    compute_cost_matrix_bitmask_multibucket,
    compute_dynamic_congestion_penalty,
    compute_local_ant_density,
)
from tb_baselines.mcmf_teacher import MCMFTeacher


def test_corridor_width_computation():
    rows, cols = 10, 10
    water = np.ones((rows, cols), dtype=bool)
    # 宽度为 1 的走廊 (行 3, 列 1 至 8)
    water[3, 1:9] = False
    # 宽度为 3 的走廊 (行 6, 7, 8, 列 1 至 8)
    water[6:9, 1:9] = False

    width_map = compute_corridor_width(water, rows, cols, max_search_width=8)
    assert width_map.shape == (rows, cols)
    # 行 3 走廊宽度度量为 1
    assert width_map[3, 5] == 1
    # 行 6, 7, 8 走廊宽度度量为 3
    assert width_map[6, 5] == 3
    assert width_map[7, 5] == 3
    assert width_map[8, 5] == 3
    # 水域方格宽度度量为 0
    assert width_map[0, 0] == 0

    # 开阔平原测试
    open_water = np.zeros((20, 20), dtype=bool)
    open_width = compute_corridor_width(open_water, 20, 20, max_search_width=8)
    assert (open_width == 8).all()

    # 类型校验与快速就地崩溃
    with pytest.raises(TypeError):
        compute_corridor_width(water.astype(np.float32), rows, cols)
    with pytest.raises(ValueError):
        compute_corridor_width(water, rows, cols, max_search_width=0)


def test_local_ant_density_computation():
    rows, cols = 20, 20
    passable = np.ones((rows, cols), dtype=bool)
    # 构造单行走廊 (行 10)
    corridor_passable = np.zeros((rows, cols), dtype=bool)
    corridor_passable[10, :] = True

    # 空载走廊密度为 0.0
    dens_empty = compute_local_ant_density([], corridor_passable, rows, cols, radius=2)
    assert dens_empty[10, 5] == 0.0

    # 走廊内 1 只蚂蚁
    dens_single = compute_local_ant_density([(10, 5)], corridor_passable, rows, cols, radius=2)
    # 局部 5x5 窗口内可通行方格数为 5 (仅行 10 有 5 格)
    assert np.isclose(dens_single[10, 5], 1.0 / 5.0)

    # 走廊内 4 只蚂蚁聚集
    ants_4 = [(10, 4), (10, 5), (10, 6), (10, 7)]
    dens_cluster = compute_local_ant_density(ants_4, corridor_passable, rows, cols, radius=2)
    # 局部窗口内蚂蚁数上限为 4，分母为 5
    assert np.isclose(dens_cluster[10, 5], 4.0 / 5.0)

    # 开阔平原 4 只蚂蚁
    dens_open = compute_local_ant_density(ants_4, passable, rows, cols, radius=2)
    # 开阔平原分母为 25
    assert np.isclose(dens_open[10, 5], 4.0 / 25.0)

    # 越界输入快速就地崩溃
    with pytest.raises(ValueError):
        compute_local_ant_density([(25, 5)], passable, rows, cols)


def test_dynamic_congestion_penalty_scaling():
    rows, cols = 10, 10
    passable = np.ones((rows, cols), dtype=bool)
    width_map = np.full((rows, cols), 1, dtype=np.int32)
    density_map = np.full((rows, cols), 0.8, dtype=np.float32)

    raw_pen, int_delay = compute_dynamic_congestion_penalty(
        width_map, density_map, passable, alpha=10.0, beta=1.5, max_delay=4
    )
    expected_pen = 10.0 * np.power(0.8 / 1.0, 1.5)
    assert np.isclose(raw_pen[0, 0], expected_pen)
    assert int_delay[0, 0] == min(4, int(round(expected_pen)))

    # 参数范围异常快速就地崩溃
    with pytest.raises(ValueError):
        compute_dynamic_congestion_penalty(width_map, density_map, passable, alpha=-1.0)
    with pytest.raises(ValueError):
        compute_dynamic_congestion_penalty(width_map, density_map, passable, beta=0.0)


def build_asymmetric_dual_corridor_fixture(num_ants_in_narrow: int = 0):
    rows, cols = 40, 50
    water = np.ones((rows, cols), dtype=bool)
    entrance = (15, 10)
    exit_tile = (15, 35)

    water[entrance[0], entrance[1]] = False
    water[exit_tile[0], exit_tile[1]] = False

    # 短窄走廊: 行 10 (长度 5 + 25 + 5 = 35 步, 宽度 1)
    narrow_row = 10
    for c in range(entrance[1] + 1, exit_tile[1]):
        water[narrow_row, c] = False
    for r in range(narrow_row, entrance[0] + 1):
        water[r, entrance[1]] = False
        water[r, exit_tile[1]] = False

    # 长宽走廊: 行 24..26 (宽度 3，长度 9 + 25 + 9 = 43 步)
    wide_row = 24
    for c in range(entrance[1] + 1, exit_tile[1]):
        for w in range(3):
            water[wide_row + w, c] = False
    for r in range(entrance[0], wide_row + 3):
        water[r, entrance[1]] = False
        water[r, exit_tile[1]] = False

    # 导出垂直连通的 5 颗食物与采集位，满足多蚂蚁获配需求
    water[13:18, 35] = False
    foods = []
    gather_positions = []
    for i in range(5):
        fr = 13 + i
        water[fr, 35:39] = False
        foods.append((fr, 38))
        gather_positions.append((fr, 37))

    gather_pos = (exit_tile[0], exit_tile[1] + 2)
    food_pos = (exit_tile[0], exit_tile[1] + 3)

    narrow_ants = [(narrow_row, entrance[1] + 3 + i * 3) for i in range(num_ants_in_narrow)]
    deciding_ant = entrance
    all_ants = [deciding_ant] + narrow_ants

    return {
        "rows": rows,
        "cols": cols,
        "water": water,
        "passable": ~water,
        "gather_pos": gather_pos,
        "food_pos": food_pos,
        "foods": foods,
        "gather_positions": gather_positions,
        "deciding_ant": deciding_ant,
        "narrow_ants": narrow_ants,
        "all_ants": all_ants,
    }


def test_asymmetric_dual_corridor_diversion():
    for ants_cnt, expected_action in [(0, 0), (1, 0), (4, 2)]:
        env = build_asymmetric_dual_corridor_fixture(num_ants_in_narrow=ants_cnt)
        rows, cols = env["rows"], env["cols"]
        water = env["water"]
        passable = env["passable"]
        gather_pos = env["gather_pos"]
        deciding_ant = env["deciding_ant"]
        ar, ac = deciding_ant

        width_map = compute_corridor_width(water, rows, cols)
        density_map = compute_local_ant_density(env["narrow_ants"], passable, rows, cols, radius=2)
        _, int_delay = compute_dynamic_congestion_penalty(
            width_map, density_map, passable, alpha=10.0, beta=1.5, max_delay=4
        )

        cost_matrix = compute_cost_matrix_bitmask_multibucket(
            ant_positions=[deciding_ant],
            gathering_positions=[gather_pos],
            passable_mask=passable,
            cell_delay=int_delay,
            rows=rows,
            cols=cols,
            max_depth=60,
            max_delay=4,
        )
        assert cost_matrix[0, 0] < FAR

        fields = compute_batch_torus_geodesics_multibucket(
            targets=[gather_pos],
            passable_mask=passable,
            cell_delay=int_delay,
            rows=rows,
            cols=cols,
            max_depth=60,
            max_delay=4,
        )
        dist_field = fields[0]

        cost_north = dist_field[(ar - 1) % rows, ac]
        cost_south = dist_field[(ar + 1) % rows, ac]
        chosen_action = 0 if cost_north < cost_south else 2

        assert chosen_action == expected_action, (
            f"蚂蚁数量为 {ants_cnt} 时，决策动作应为 {expected_action}，实际决策为 {chosen_action} "
            f"(North: {cost_north}, South: {cost_south})"
        )


def test_mcmf_teacher_diversion_integration():
    # 验证 MCMFTeacher 在 dynamic_congestion=True 下的完整接入行为
    # 场景 1: 窄走廊 0 蚂蚁，决策走向短窄走廊 (North)
    env_0 = build_asymmetric_dual_corridor_fixture(num_ants_in_narrow=0)
    teacher_0 = MCMFTeacher(max_depth=60, dynamic_congestion=True, congestion_alpha=10.0, congestion_beta=1.5)
    obs_0 = {
        "size": (env_0["rows"], env_0["cols"]),
        "mine": env_0["all_ants"],
        "food": env_0["foods"],
        "ids": list(range(len(env_0["all_ants"]))),
    }
    seen_0 = np.ones((env_0["rows"], env_0["cols"]), dtype=bool)
    orders_0 = teacher_0.orders(obs_0, env_0["water"], seen_0)
    assert len(orders_0) == len(env_0["all_ants"])
    assert orders_0[0] == 0  # North (短窄走廊)

    # 场景 2: 窄走廊 4 只聚集蚂蚁，决策走向长宽走廊 (South 分流)
    env_4 = build_asymmetric_dual_corridor_fixture(num_ants_in_narrow=4)
    teacher_4 = MCMFTeacher(max_depth=60, dynamic_congestion=True, congestion_alpha=10.0, congestion_beta=1.5)
    obs_4 = {
        "size": (env_4["rows"], env_4["cols"]),
        "mine": env_4["all_ants"],
        "food": env_4["foods"],
        "ids": list(range(len(env_4["all_ants"]))),
    }
    seen_4 = np.ones((env_4["rows"], env_4["cols"]), dtype=bool)
    orders_4 = teacher_4.orders(obs_4, env_4["water"], seen_4)
    assert len(orders_4) == len(env_4["all_ants"])
    assert orders_4[0] == 2  # South (长宽走廊分流)


def test_stress_dynamic_congestion_latency_under_10ms():
    rows, cols = 64, 64
    water = np.zeros((rows, cols), dtype=bool)
    water[15:25, 20:25] = True
    water[15:25, 35:40] = True
    water[40:50, 15:20] = True
    water[40:50, 45:50] = True
    for c in range(10, 50):
        water[30, c] = True
        water[32, c] = True
    water[31, 10:50] = False

    passable = ~water
    rng = np.random.default_rng(2026)

    ant_set = set()
    for c in range(20, 35):
        ant_set.add((31, c))
    while len(ant_set) < 100:
        r = int(rng.integers(0, rows))
        c = int(rng.integers(0, cols))
        if passable[r, c]:
            ant_set.add((r, c))
    ants = list(ant_set)

    food_set = set()
    while len(food_set) < 30:
        r = int(rng.integers(0, rows))
        c = int(rng.integers(0, cols))
        if passable[r, c] and (r, c) not in ant_set:
            food_set.add((r, c))
    foods = list(food_set)

    teacher = MCMFTeacher(
        max_depth=16,
        dynamic_congestion=True,
        congestion_alpha=10.0,
        congestion_beta=1.5,
    )
    obs = {
        "size": (rows, cols),
        "mine": ants,
        "food": foods,
        "ids": list(range(len(ants))),
    }
    seen = np.ones((rows, cols), dtype=bool)

    import gc
    # 预热一次并隔离垃圾回收扰动
    teacher.orders(obs, water, seen)
    gc.collect()

    # 运行 5 回合测量平均单回合耗时
    times = []
    for _ in range(5):
        t0 = time.perf_counter()
        orders = teacher.orders(obs, water, seen)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        times.append(elapsed_ms)
        assert len(orders) == 100

    avg_ms = float(np.mean(times))
    print(f"\n100 蚂蚁规模动态拥塞单回合平均耗时: {avg_ms:.3f} ms")
    assert avg_ms < 10.0, f"单回合平均耗时 ({avg_ms:.3f} ms) 必须严格小于 10.0 ms 门禁"
