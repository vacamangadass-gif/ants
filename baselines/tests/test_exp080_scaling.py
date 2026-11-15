from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tb_baselines import planes
from tb_baselines.benchmarks.run_exp080_scaling import (
    compute_collision_blocking_rate,
    compute_corridor_congestion_index,
    compute_corridor_mask,
    compute_corridor_stalling_rate,
    compute_effective_gathering_throughput,
    parse_args,
    rle_from_mask,
    run_scaling_simulation,
)


def test_corridor_width_open_field():
    rows, cols = 12, 12
    water = np.zeros((rows, cols), dtype=bool)
    mask = compute_corridor_mask(water, max_corridor_width=2)
    assert not mask.any()
    assert int(mask.sum()) == 0


def test_corridor_width_1wide_corridor():
    rows, cols = 10, 10
    water = np.zeros((rows, cols), dtype=bool)
    water[3, :] = True
    water[5, :] = True

    mask = compute_corridor_mask(water, max_corridor_width=2)
    for c in range(cols):
        assert mask[4, c]
    assert int(mask.sum()) == cols


def test_corridor_width_2wide_corridor():
    rows, cols = 12, 12
    water = np.zeros((rows, cols), dtype=bool)
    water[:, 3] = True
    water[:, 6] = True

    mask = compute_corridor_mask(water, max_corridor_width=2)
    for r in range(rows):
        assert mask[r, 4]
        assert mask[r, 5]
    assert int(mask.sum()) == rows * 2


def test_corridor_width_torus_boundary_wrapping():
    rows, cols = 10, 10
    water = np.zeros((rows, cols), dtype=bool)
    water[1, :] = True
    water[8, :] = True

    mask = compute_corridor_mask(water, max_corridor_width=2)
    for c in range(cols):
        assert mask[0, c]
        assert mask[9, c]
    assert int(mask.sum()) == cols * 2


def test_corridor_width_diagonal_chokepoint():
    rows, cols = 8, 8
    water = np.zeros((rows, cols), dtype=bool)
    water[4, :3] = True
    water[4, 5:] = True
    water[3, :4] = True
    water[5, 4:] = True

    mask = compute_corridor_mask(water, max_corridor_width=2)
    assert mask[4, 3]
    assert mask[4, 4]



def test_congestion_metric_mathematical_precision():
    stalled_turns = 12
    turns = 100
    corridor_cells = 6
    index = compute_corridor_congestion_index(
        stalled_turns_in_corridor=stalled_turns,
        turns=turns,
        corridor_cell_count=corridor_cells,
    )
    expected = 12.0 / (100.0 * 6.0)
    assert pytest.approx(index, rel=1e-6) == expected

    rate = compute_corridor_stalling_rate(
        stalled_turns_in_corridor=stalled_turns,
        total_corridor_ant_turns=48,
    )
    assert pytest.approx(rate, rel=1e-6) == 12.0 / 48.0

    assert compute_corridor_congestion_index(0, 0, 10) == 0.0
    assert compute_corridor_congestion_index(10, 100, 0) == 0.0
    assert compute_corridor_stalling_rate(10, 0) == 0.0


def test_gathering_throughput_mathematical_precision():
    food_gathered = 45
    turns = 300
    alive_ants = 50.0
    throughput = compute_effective_gathering_throughput(
        total_food_gathered=food_gathered,
        turns=turns,
        average_alive_ants=alive_ants,
    )
    expected = (45.0 * 100.0) / (300.0 * 50.0)
    assert pytest.approx(throughput, rel=1e-6) == expected

    assert compute_effective_gathering_throughput(10, 0, 50.0) == 0.0
    assert compute_effective_gathering_throughput(10, 100, 0.0) == 0.0


def test_collision_blocking_rate_precision():
    forced_holds = 18
    total_ant_turns = 600
    rate = compute_collision_blocking_rate(
        total_forced_holds=forced_holds,
        total_ant_turns=total_ant_turns,
    )
    expected = 18.0 / 600.0
    assert pytest.approx(rate, rel=1e-6) == expected
    assert compute_collision_blocking_rate(5, 0) == 0.0


def test_fast_fail_insufficient_land():
    rows, cols = 6, 6
    water = np.ones((rows, cols), dtype=bool)
    water[0, 0] = False
    water[0, 1] = False

    board_data = {
        "rows": rows,
        "cols": cols,
        "water": rle_from_mask(water),
        "hills": [[0, 0]],
        "food": [],
    }

    with pytest.raises(ValueError, match="地图合法陆地数量不足"):
        run_scaling_simulation(
            board_data=board_data,
            scale=10,
            turns=10,
            seed=42,
        )


def test_fast_fail_invalid_arguments():
    rows, cols = 8, 8
    water = np.zeros((rows, cols), dtype=bool)
    board_data = {
        "rows": rows,
        "cols": cols,
        "water": rle_from_mask(water),
        "hills": [[0, 0]],
        "food": [],
    }

    with pytest.raises(ValueError, match="蚂蚁规模必须大于 0"):
        run_scaling_simulation(board_data, scale=0, turns=10, seed=42)

    with pytest.raises(ValueError, match="模拟推进回合数必须大于 0"):
        run_scaling_simulation(board_data, scale=5, turns=0, seed=42)

    with pytest.raises(ValueError, match="未知教师模型类型"):
        run_scaling_simulation(board_data, scale=5, turns=10, seed=42, teacher_type="invalid")


def test_rle_from_mask_roundtrip():
    rows, cols = 16, 16
    rng = np.random.default_rng(20261103)
    raw = rng.random((rows, cols)) < 0.25

    rle = rle_from_mask(raw)
    b = planes.Board(rows, cols)
    recovered = b.rle(rle).astype(bool)

    assert np.array_equal(raw, recovered)


def test_run_scaling_simulation_integration_tiny_corridor():
    map_file = Path(__file__).resolve().parents[2] / "benchmarks" / "maps_2p" / "tiny-corridor-2p-1h.json"
    assert map_file.is_file(), f"未找到官方测试地图: {map_file}"

    with open(map_file, "r", encoding="utf-8") as f:
        board_data = json.load(f)

    result = run_scaling_simulation(
        board_data=board_data,
        scale=20,
        turns=5,
        seed=90000,
        teacher_type="compound",
    )

    required_keys = [
        "scale",
        "turns",
        "seed",
        "teacher_type",
        "grid_shape",
        "total_land_cells",
        "corridor_cell_count",
        "total_corridor_ant_turns",
        "stalled_corridor_turns",
        "corridor_congestion_index",
        "corridor_stalling_rate",
        "total_food_gathered",
        "effective_gathering_throughput",
        "total_ant_turns",
        "total_forced_holds",
        "collision_blocking_rate",
        "internal_collision_blocking_rate",
        "mean_latency_ms",
        "p50_latency_ms",
        "p95_latency_ms",
        "max_latency_ms",
        "min_latency_ms",
        "total_latency_ms",
        "latency_ms",
    ]

    for key in required_keys:
        assert key in result, f"结果缺少指标字段: {key}"

    assert result["scale"] == 20
    assert result["turns"] == 5
    assert result["corridor_cell_count"] > 0
    assert result["total_corridor_ant_turns"] >= 0
    assert result["mean_latency_ms"] > 0.0
    assert result["p95_latency_ms"] >= result["p50_latency_ms"]
    assert result["effective_gathering_throughput"] >= 0.0
    assert result["collision_blocking_rate"] >= 0.0


def test_zero_corridor_cells_metrics():
    rows, cols = 12, 12
    water = np.zeros((rows, cols), dtype=bool)
    corridor_mask = compute_corridor_mask(water, max_corridor_width=2)
    corridor_cells = int(corridor_mask.sum())
    assert corridor_cells == 0

    idx = compute_corridor_congestion_index(0, 100, corridor_cells)
    assert idx == 0.0
    rate = compute_corridor_stalling_rate(0, 0)
    assert rate == 0.0


def test_run_scaling_simulation_baseline_teacher():
    map_file = Path(__file__).resolve().parents[2] / "benchmarks" / "maps_2p" / "tiny-open-2p-1h.json"
    assert map_file.is_file(), f"未找到官方测试地图: {map_file}"

    with open(map_file, "r", encoding="utf-8") as f:
        board_data = json.load(f)

    result = run_scaling_simulation(
        board_data=board_data,
        scale=20,
        turns=3,
        seed=90001,
        teacher_type="baseline",
    )
    assert result["teacher_type"] == "baseline"
    assert result["scale"] == 20
    assert result["turns"] == 3
    assert result["mean_latency_ms"] > 0.0


def test_parse_args_custom_values():
    args = [
        "--scales", "30,50",
        "--maps", "benchmarks/maps_2p/tiny-open-2p-1h.json",
        "--seeds", "123,456",
        "--turns", "50",
        "--out", "runs/custom_out/",
        "--teacher", "baseline",
    ]
    parsed = parse_args(args)
    assert parsed.scales == "30,50"
    assert parsed.maps == "benchmarks/maps_2p/tiny-open-2p-1h.json"
    assert parsed.seeds == "123,456"
    assert parsed.turns == 50
    assert parsed.out == "runs/custom_out/"
    assert parsed.teacher == "baseline"
