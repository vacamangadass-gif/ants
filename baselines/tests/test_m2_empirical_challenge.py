from __future__ import annotations

import json
import random
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[2]
REPLAYS_DIR = ROOT / "replays" / "tournament_p3_9map"
METRICS_PATH = ROOT / "data" / "tournament_p3_results" / "tournament_metrics.json"

EXPECTED_DIGEST = "sha256:819166e7918117a1cbbd67dc042e7368a076a2dd45cea5122683983912d591e1"
EXPECTED_ENTRANTS = [
    "exp031_iter1", "exp031_iter2", "exp031_iter3", "exp031_iter4",
    "exp031_iter5", "exp031_iter6", "exp031_iter7", "exp031_iter8",
    "exp033_iter1", "exp033_iter2", "exp033_iter3", "exp033_iter4",
    "exp033_iter5", "exp033_iter6", "exp033_iter7", "exp033_iter8",
]
EXPECTED_MAPS = {
    "medium-crossroads-2p-1h",
    "medium-labyrinth-2p-1h",
    "medium-valley-2p-1h",
    "small-arena-2p-1h",
    "small-islands-2p-1h",
    "small-rooms-2p-1h",
    "tiny-colonnade-2p-1h",
    "tiny-corridor-2p-1h",
    "tiny-open-2p-1h",
}
MAX_ALLOWED_US = 700_000


class TestMilestone2ReplaySamplingAndIntegrity:
    # 验证回放目录总文件数量严格等于 2160
    def test_total_replay_file_count(self):
        files = [p for p in REPLAYS_DIR.glob("*.json") if "_batch_" not in p.name]
        assert len(files) == 2160, f"回放文件总数异常: 实际为 {len(files)}，预期为 2160"

    # 随机抽样 100 场回放深入校验结构完整性、turns、scores、ranks 以及 engine_digest
    def test_sample_100_replays_structure_and_digest(self):
        all_files = sorted([p for p in REPLAYS_DIR.glob("*.json") if "_batch_" not in p.name])
        rng = random.Random(20261031)
        sampled_files = rng.sample(all_files, 100)
        assert len(sampled_files) == 100

        for path in sampled_files:
            text = path.read_text(encoding="utf-8")
            data = json.loads(text)

            # 校验核心字段存在性
            required_keys = (
                "engine_digest", "engine_ranks", "map", "map_id",
                "match_id", "max_turns", "reason", "scores", "seats", "turns", "deltas"
            )
            for k in required_keys:
                assert k in data, f"文件 {path.name} 缺少字段 {k}"

            # 校验确定性引擎摘要
            assert data["engine_digest"] == EXPECTED_DIGEST, (
                f"引擎摘要不一致: {data['engine_digest']} != {EXPECTED_DIGEST}"
            )

            # 校验地图标识有效性
            assert data["map_id"] in EXPECTED_MAPS, f"未知地图标识: {data['map_id']}"

            # 校验 match_id 与文件名一致
            assert data["match_id"] == path.stem, f"对局标识与文件名不匹配: {data['match_id']}"

            # 校验回合数数值范围与帧列表长度
            turns = data["turns"]
            assert isinstance(turns, int), f"turns 必须为整型: {turns}"
            assert 0 <= turns <= 300, f"turns 超出数值边界: {turns}"
            assert len(data["deltas"]) == turns, (
                f"帧数量与回合数不匹配: {len(data['deltas'])} != {turns}"
            )

            # 校验得分与排名一致性
            scores = data["scores"]
            ranks = data["engine_ranks"]
            assert len(scores) == 2, f"得分维度必须为 2: {scores}"
            assert len(ranks) == 2, f"排名维度必须为 2: {ranks}"
            assert all(isinstance(s, int) and s >= 0 for s in scores), f"得分必须为非负整数: {scores}"
            assert all(r in (1, 2) for r in ranks), f"排名取值异常: {ranks}"

            if scores[0] > scores[1]:
                assert ranks == [1, 2], f"胜负与排名不一致: scores={scores}, ranks={ranks}"
            elif scores[0] < scores[1]:
                assert ranks == [2, 1], f"胜负与排名不一致: scores={scores}, ranks={ranks}"
            else:
                assert ranks == [1, 1], f"平局排名不一致: scores={scores}, ranks={ranks}"

            # 校验席位信息与时延上限
            seats = data["seats"]
            assert len(seats) == 2, f"席位数量必须为 2: {len(seats)}"
            seat_labels = [s["label"] for s in seats]
            assert all(label in EXPECTED_ENTRANTS for label in seat_labels), f"未知参赛者: {seat_labels}"
            assert seat_labels[0] != seat_labels[1], f"同一选手自对弈异常: {seat_labels}"

            for s in seats:
                assert s["infer_us_max"] <= MAX_ALLOWED_US, (
                    f"时延超过 700ms 上限: {s['infer_us_max']} us 在文件 {path.name}"
                )
                assert s["infer_us_total"] >= 0, f"推理总时延为负数: {s['infer_us_total']}"
                assert s["seat_turns"] == turns, f"席位回合数与对局回合数不匹配: {s['seat_turns']} vs {turns}"


class TestPairwiseReciprocalSymmetry:
    # 校验 240 组两两对抗的完整性与互易对称性，严格满足积分和为 18
    def test_all_240_pairs_points_sum_strictly_18(self):
        assert METRICS_PATH.is_file(), f"指标汇总文件不存在: {METRICS_PATH}"
        data = json.loads(METRICS_PATH.read_text(encoding="utf-8"))

        h2h = data.get("head_to_head", {})
        expected_pair_count = len(EXPECTED_ENTRANTS) * (len(EXPECTED_ENTRANTS) - 1)
        assert expected_pair_count == 240
        assert len(h2h) == 240, f"两两对抗记录数量异常: 实际为 {len(h2h)}，预期为 240"

        for p1 in EXPECTED_ENTRANTS:
            for p2 in EXPECTED_ENTRANTS:
                if p1 == p2:
                    continue
                pair_key = f"{p1}_vs_{p2}"
                rev_key = f"{p2}_vs_{p1}"
                assert pair_key in h2h, f"缺少对战对: {pair_key}"
                assert rev_key in h2h, f"缺少对战对: {rev_key}"

                h1 = h2h[pair_key]
                h2 = h2h[rev_key]

                # 对局总数必须严格等于 18 场（9 张地图双向对弈）
                assert h1["matches"] == 18, f"对局数异常: {pair_key} matches={h1['matches']}"
                assert h2["matches"] == 18, f"对局数异常: {rev_key} matches={h2['matches']}"

                # 胜平负对称性
                assert h1["p1_wins"] == h2["p2_wins"], (
                    f"胜场互易不对称: {pair_key} p1_wins={h1['p1_wins']} vs {rev_key} p2_wins={h2['p2_wins']}"
                )
                assert h1["p2_wins"] == h2["p1_wins"], (
                    f"胜场互易不对称: {pair_key} p2_wins={h1['p2_wins']} vs {rev_key} p1_wins={h2['p1_wins']}"
                )
                assert h1["draws"] == h2["draws"], (
                    f"平局互易不对称: {pair_key} draws={h1['draws']} vs {rev_key} draws={h2['draws']}"
                )

                # 胜平负之和必须等于 18
                assert h1["p1_wins"] + h1["p2_wins"] + h1["draws"] == 18
                assert h2["p1_wins"] + h2["p2_wins"] + h2["draws"] == 18

                # 积分守恒：单组对决双向总积分严格等于 18.0
                total_points = h1["p1_points"] + h2["p1_points"]
                assert abs(total_points - 18.0) < 1e-6, (
                    f"总积分不等于 18: {pair_key} 与 {rev_key} 积分和为 {total_points}"
                )

                # 内部积分定义一致性：p1_points 必须等于 p1_wins * 1.0 + draws * 0.5
                expected_p1_pts = h1["p1_wins"] * 1.0 + h1["draws"] * 0.5
                assert abs(h1["p1_points"] - expected_p1_pts) < 1e-6, (
                    f"积分计算公式不一致: {pair_key} p1_points={h1['p1_points']} vs {expected_p1_pts}"
                )


class TestEntrantMatchInvariantsAndConservation:
    # 验证单模型参赛场次严格等于 270 场，且胜平负场次之和严格等于 270
    def test_entrant_matches_strictly_270_and_sum_conserved(self):
        data = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
        leaderboard = data.get("leaderboard", [])
        assert len(leaderboard) == 16, f"参赛者排行榜数量异常: {len(leaderboard)}"

        leaderboard_names = [item["name"] for item in leaderboard]
        assert set(leaderboard_names) == set(EXPECTED_ENTRANTS)

        total_wins = 0
        total_draws = 0
        total_losses = 0
        total_points = 0.0

        for item in leaderboard:
            name = item["name"]
            matches = item["matches"]
            wins = item["wins"]
            draws = item["draws"]
            losses = item["losses"]
            points = item["points"]

            # 单模型必须严格经历 270 场对局（15 个对手 * 18 场）
            assert matches == 270, f"{name} 参赛场次异常: {matches} != 270"

            # 胜平负场次之和严格等于 270
            assert wins + draws + losses == 270, (
                f"{name} 胜平负之和不等于 270: wins={wins}, draws={draws}, losses={losses}"
            )

            # 积分必须严格满足标准计分规则
            expected_points = wins * 1.0 + draws * 0.5
            assert abs(points - expected_points) < 1e-6, (
                f"{name} 积分不匹配: {points} != {expected_points}"
            )

            # 得分率与胜率必须在 [0.0, 1.0] 范围内且计算精确
            assert 0.0 <= item["score_rate"] <= 1.0
            assert 0.0 <= item["win_rate"] <= 1.0
            assert abs(item["score_rate"] - round(points / 270.0, 4)) < 1e-4
            assert abs(item["win_rate"] - round(wins / 270.0, 4)) < 1e-4

            total_wins += wins
            total_draws += draws
            total_losses += losses
            total_points += points

        # 全局守恒性校验：总胜场必须等于总负场
        assert total_wins == total_losses, f"全局胜负不守恒: 总胜 {total_wins} != 总负 {total_losses}"

        # 全局总场次为 2160 场（每场 2 个参赛座位，计 4320 席次）
        assert total_wins + total_draws + total_losses == 270 * 16

        # 全局总积分严格等于 2160.0
        assert abs(total_points - 2160.0) < 1e-6, f"全局总积分不守恒: {total_points} != 2160.0"


class TestNumericalBoundsOnTurns:
    # 验证胜局拆毁回合与败局存活回合的数值边界合理性（0 <= turn <= 300）
    def test_raze_and_survive_turn_bounds(self):
        data = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
        leaderboard = data.get("leaderboard", [])

        for item in leaderboard:
            name = item["name"]
            raze_turns = item.get("raze_turns", [])
            loss_turns = item.get("loss_turns", [])
            hills_razed = item.get("hills_razed", 0)
            losses = item.get("losses", 0)

            # 拆毁记录数量与 hills_razed 严格一致
            assert len(raze_turns) == hills_razed, (
                f"{name} 拆毁记录长度与拆毁数不一致: {len(raze_turns)} != {hills_razed}"
            )

            # 败局存活记录数量与 losses 严格一致
            assert len(loss_turns) == losses, (
                f"{name} 败局记录长度与负场数不一致: {len(loss_turns)} != {losses}"
            )

            # 每一个拆毁回合必须严格在 [1, 300] 范围内（第 0 回合不可能拆毁）
            for t in raze_turns:
                assert isinstance(t, int), f"{name} 拆毁回合必须为整数: {t}"
                assert 1 <= t <= 300, f"{name} 拆毁回合超出边界 [1, 300]: {t}"

            # 每一个败局存活回合必须严格在 [1, 300] 范围内
            for t in loss_turns:
                assert isinstance(t, int), f"{name} 存活回合必须为整数: {t}"
                assert 1 <= t <= 300, f"{name} 存活回合超出边界 [1, 300]: {t}"

            # 均值统计量在存在记录时必须在 [1.0, 300.0] 范围内且数值准确
            if raze_turns:
                expected_mean_raze = round(sum(raze_turns) / len(raze_turns), 2)
                assert item["mean_raze_turn"] == pytest.approx(expected_mean_raze, abs=0.01)
                assert 1.0 <= item["mean_raze_turn"] <= 300.0
            else:
                assert item["mean_raze_turn"] is None

            if loss_turns:
                expected_mean_survive = round(sum(loss_turns) / len(loss_turns), 2)
                assert item["mean_survive_turn"] == pytest.approx(expected_mean_survive, abs=0.01)
                assert 1.0 <= item["mean_survive_turn"] <= 300.0
            else:
                assert item["mean_survive_turn"] is None

            # 运营与导航指标非负性与合理性
            assert item["food_gathered"] >= 0
            assert item["collisions"] >= 0
            assert item["combat_deaths"] >= 0
            assert item["total_orders"] >= item["blocked_orders"] >= 0
            assert 0.0 <= item["blocked_move_pct"] <= 100.0
            assert item["infer_us_max"] <= MAX_ALLOWED_US


class TestIndependentReplayOracleCrossValidation:
    # 独立遍历 2160 场回放元数据，独立累加计算各模型胜平负与积分，与 tournament_metrics.json 交叉核对
    def test_independent_recomputation_from_raw_replays(self):
        all_files = sorted([p for p in REPLAYS_DIR.glob("*.json") if "_batch_" not in p.name])
        assert len(all_files) == 2160

        independent_records = {
            name: {"matches": 0, "wins": 0, "draws": 0, "losses": 0, "points": 0.0}
            for name in EXPECTED_ENTRANTS
        }

        for path in all_files:
            content = json.loads(path.read_text(encoding="utf-8"))
            assert content["engine_digest"] == EXPECTED_DIGEST, f"非官方确定性引擎摘要: {path.name}"
            turns = content["turns"]
            assert 1 <= turns <= 300, f"回合数越界: {turns} 在文件 {path.name}"
            assert len(content["deltas"]) == turns, f"帧数与回合数不匹配 在文件 {path.name}"

            seats = content["seats"]
            ranks = content["engine_ranks"]
            p0 = seats[0]["label"]
            p1 = seats[1]["label"]

            independent_records[p0]["matches"] += 1
            independent_records[p1]["matches"] += 1

            if ranks[0] < ranks[1]:
                independent_records[p0]["wins"] += 1
                independent_records[p0]["points"] += 1.0
                independent_records[p1]["losses"] += 1
            elif ranks[0] > ranks[1]:
                independent_records[p1]["wins"] += 1
                independent_records[p1]["points"] += 1.0
                independent_records[p0]["losses"] += 1
            else:
                independent_records[p0]["draws"] += 1
                independent_records[p0]["points"] += 0.5
                independent_records[p1]["draws"] += 1
                independent_records[p1]["points"] += 0.5

        data = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
        reported_map = {item["name"]: item for item in data.get("leaderboard", [])}

        for name in EXPECTED_ENTRANTS:
            indep = independent_records[name]
            rep = reported_map[name]

            assert indep["matches"] == rep["matches"] == 270, f"{name} 场次不一致"
            assert indep["wins"] == rep["wins"], f"{name} 胜场不一致: 独立计算 {indep['wins']} vs 报告 {rep['wins']}"
            assert indep["draws"] == rep["draws"], f"{name} 平局不一致: 独立计算 {indep['draws']} vs 报告 {rep['draws']}"
            assert indep["losses"] == rep["losses"], f"{name} 负场不一致: 独立计算 {indep['losses']} vs 报告 {rep['losses']}"
            assert abs(indep["points"] - rep["points"]) < 1e-6, f"{name} 积分不一致: 独立计算 {indep['points']} vs 报告 {rep['points']}"
