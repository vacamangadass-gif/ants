import json
import os
import re
from PIL import Image

# 检验原始比赛数据与汇总矩阵的一致性
with open("data/ablation_tournament_results.json", "r", encoding="utf-8") as f_matches:
    tournament_matches = json.load(f_matches)

with open("data/out_ablation_matrix.json", "r", encoding="utf-8") as f_matrix:
    matrix_data = json.load(f_matrix)

leaderboard = {item["variant_id"]: item for item in matrix_data["leaderboard"]}
pairwise = {item["variant_id"]: item for item in matrix_data["pairwise_comparisons_vs_baseline"]}

assert len(tournament_matches) == 630, f"总比赛场次异常: {len(tournament_matches)}"
assert len(matrix_data["leaderboard"]) == 8, f"排行榜变体数量异常: {len(matrix_data['leaderboard'])}"
assert len(matrix_data["pairwise_comparisons_vs_baseline"]) == 7, f"对比条目数量异常: {len(matrix_data['pairwise_comparisons_vs_baseline'])}"

# 1. 验证原始对战明细逐项聚合与 out_ablation_matrix.json 完全相等
from collections import defaultdict
by_variant = defaultdict(list)
for match in tournament_matches:
    by_variant[match["variant"]].append(match)

for var_code, recs in by_variant.items():
    vid = var_code.lower().replace("_", "")
    mat_entry = leaderboard[vid]
    
    matches_count = len(recs)
    wins = sum(1 for m in recs if m["variant_won"])
    losses = sum(1 for m in recs if m["baseline_won"])
    draws = sum(1 for m in recs if m["draw"])
    points = wins + 0.5 * draws
    
    assert mat_entry["matches"] == matches_count
    assert mat_entry["wins"] == wins
    assert mat_entry["draws"] == draws
    assert mat_entry["losses"] == losses
    assert abs(mat_entry["points"] - points) < 1e-4

    food_gathered_sum = sum(m["variant_stats"]["ants_spawned"] for m in recs)
    mean_food = round(food_gathered_sum / matches_count, 2)
    turns_sum = sum(m["turns"] for m in recs)
    food_rate_100t = round((food_gathered_sum / turns_sum * 100.0) if turns_sum > 0 else 0.0, 2)
    hills_razed_sum = sum(1 for m in recs if m["variant_score"] > 0)
    mean_hills_razed = round(hills_razed_sum / matches_count, 2)
    raze_turns = [m["turns"] for m in recs if m["variant_score"] > 0]
    mean_raze_turn = round(sum(raze_turns) / len(raze_turns), 1) if raze_turns else 300.0

    assert mean_food == mat_entry["mean_food_gathered"]
    assert food_rate_100t == mat_entry["food_rate_per_100_turns"]
    assert mean_hills_razed == mat_entry["mean_hills_razed"]
    assert mean_raze_turn == mat_entry["mean_raze_turn"]

print("原始比赛对战记录聚合验证通过: 630 场对局胜平负、得分、采食、摧毁与推平回合完全无缝吻合")

# 2. 验证 4 张学术级图表的分辨率与 DPI 属性
target_images = [
    "tinybrain/ima/exp074_ablation_score_win_rates.png",
    "tinybrain/ima/exp074_stall_reduction_breakdown.png",
    "tinybrain/ima/exp074_ablation_multimetric_radar.png",
    "tinybrain/ima/exp074_map_performance_heatmap.png",
]

for image_path in target_images:
    assert os.path.isfile(image_path), f"图像文件缺失: {image_path}"
    file_size = os.path.getsize(image_path)
    assert file_size > 100000, f"图像文件过小可能损坏: {image_path} ({file_size} bytes)"
    with Image.open(image_path) as image_obj:
        image_obj.verify()
    with Image.open(image_path) as image_obj:
        width, height = image_obj.size
        assert width >= 3000 and height >= 1500, f"图像分辨率不足: {image_path} ({width}x{height})"
        dpi = image_obj.info.get("dpi")
        assert dpi is not None, f"图像缺失 DPI 元数据: {image_path}"
        assert abs(dpi[0] - 300) < 0.1 and abs(dpi[1] - 300) < 0.1, f"图像 DPI 不满足 300 要求: {dpi}"

print("图像验证通过: 全部 4 张图表均为 300 DPI 且分辨率高于 3000x1500，文件完整无损坏")

# 3. 验证 EXP-074 实验日志中的全量消融天梯总榜数值与正文描述
exp074_path = "tinybrain/04 实验日志/EXP-074.md"
with open(exp074_path, "r", encoding="utf-8") as f_exp074:
    exp074_content = f_exp074.read()

table1_match = re.search(r"### Table 1: 全矩阵消融天梯总榜.*?\n\| :---:.*?\n(.*?)\n\n", exp074_content, re.DOTALL)
assert table1_match is not None, "EXP-074.md 未定位到 Table 1"
table1_lines = [line.strip() for line in table1_match.group(1).strip().split("\n")]
assert len(table1_lines) == 8, f"Table 1 行数异常: {len(table1_lines)}"

for line in table1_lines:
    cols = [c.strip().replace("*", "").replace("`", "") for c in line.split("|")[1:-1]]
    rank = int(cols[0])
    vid = cols[1]
    branch = cols[2]
    matches = int(cols[3])
    w = int(cols[4])
    d = int(cols[5])
    l = int(cols[6])
    score_rate_pct = float(cols[7].replace("%", ""))
    win_rate_pct = float(cols[8].replace("%", ""))
    blocked_pct = float(cols[9].replace("%", ""))
    conflict_pct = float(cols[10].replace("%", ""))
    srr_pct = float(cols[11].replace("%", "").replace("+", ""))
    food_gathered = float(cols[12])
    food_rate_100t = float(cols[13])
    hills_razed = float(cols[14])
    mean_raze_turn = float(cols[15].replace("t", ""))
    casualties = float(cols[16])
    collisions = float(cols[17])

    entry = leaderboard[vid]
    assert entry["rank"] == rank, f"{vid} rank 不一致: {rank} vs {entry['rank']}"
    assert entry["matches"] == matches, f"{vid} matches 不一致: {matches} vs {entry['matches']}"
    assert entry["wins"] == w, f"{vid} wins 不一致: {w} vs {entry['wins']}"
    assert entry["draws"] == d, f"{vid} draws 不一致: {d} vs {entry['draws']}"
    assert entry["losses"] == l, f"{vid} losses 不一致: {l} vs {entry['losses']}"
    assert abs(round(entry["score_rate"] * 100, 2) - score_rate_pct) < 1e-4, f"{vid} score_rate 不一致"
    assert abs(round(entry["win_rate"] * 100, 2) - win_rate_pct) < 1e-4, f"{vid} win_rate 不一致"
    assert abs(entry["blocked_order_rate_pct"] - blocked_pct) < 1e-4, f"{vid} blocked_pct 不一致"
    assert abs(entry["conflict_rate_pct"] - conflict_pct) < 1e-4, f"{vid} conflict_pct 不一致"
    assert abs(entry["stall_reduction_ratio_pct"] - srr_pct) < 1e-4, f"{vid} srr_pct 不一致"
    assert abs(entry["mean_food_gathered"] - food_gathered) < 1e-4, f"{vid} food_gathered 不一致"
    assert abs(entry["food_rate_per_100_turns"] - food_rate_100t) < 1e-4, f"{vid} food_rate_100t 不一致"
    assert abs(entry["mean_hills_razed"] - hills_razed) < 1e-4, f"{vid} hills_razed 不一致"
    assert abs(entry["mean_raze_turn"] - mean_raze_turn) < 1e-4, f"{vid} mean_raze_turn 不一致"
    assert abs(entry["mean_casualties"] - casualties) < 1e-4, f"{vid} casualties 不一致"
    assert abs(entry["mean_collisions"] - collisions) < 1e-4, f"{vid} collisions 不一致"

print("EXP-074 Table 1 验证通过: 全部 8 个变体 18 维度指标与矩阵完全一致")

# 4. 验证 EXP-074 正文遥测与核心亮点数值
exp074_entry = leaderboard["exp074"]
exp074_telemetry = exp074_entry["telemetry_hold_stats"]
assert "57,444" in exp074_content or "57444" in exp074_content
assert exp074_telemetry["total_ant_turns"] == 57444
assert "1,150" in exp074_content or "1150" in exp074_content
assert exp074_telemetry["total_holds"] == 1150
assert "790" in exp074_content
assert exp074_telemetry["accidental_stalls"] == 790
assert "1.3753%" in exp074_content
assert abs(exp074_telemetry["global_accidental_stall_rate_pct"] - 1.3753) < 1e-4
assert "+34.75%" in exp074_content
assert abs(exp074_entry["stall_reduction_ratio_pct"] - 34.75) < 1e-4
assert "31.30%" in exp074_content
assert abs(exp074_telemetry["intentional_ratio_pct"] - 31.30) < 1e-4

# 地图特定指标验证
mb = matrix_data["map_breakdown"]
assert mb["medium-labyrinth-2p-1h.json"]["exp074"]["score_rate"] == 1.0
assert mb["medium-labyrinth-2p-1h.json"]["exp074"]["win_rate"] == 1.0
assert mb["medium-valley-2p-1h.json"]["exp074"]["score_rate"] == 0.85
assert mb["medium-valley-2p-1h.json"]["exp074"]["win_rate"] == 0.80

print("EXP-074 正文遥测指标与地图高亮验证通过")

# 5. 验证 EXP-070 正文遥测数据
exp070_path = "tinybrain/04 实验日志/EXP-070.md"
with open(exp070_path, "r", encoding="utf-8") as f_exp070:
    exp070_content = f_exp070.read()

baseline_telemetry = leaderboard["baseline"]["telemetry_hold_stats"]
assert "480,498" in exp070_content
assert baseline_telemetry["total_ant_turns"] == 480498
assert "13,593" in exp070_content
assert baseline_telemetry["total_holds"] == 13593
assert "3,465" in exp070_content
assert baseline_telemetry["intentional_holds"] == 3465
assert "25.49%" in exp070_content
assert abs(baseline_telemetry["intentional_ratio_pct"] - 25.49) < 1e-4
assert "3,291" in exp070_content
assert baseline_telemetry["reason_breakdown"]["defense"] == 3291
assert "174" in exp070_content
assert baseline_telemetry["reason_breakdown"]["harvest_wait"] == 174
assert "10,128" in exp070_content
assert baseline_telemetry["accidental_stalls"] == 10128
assert "74.51%" in exp070_content
assert abs(baseline_telemetry["accidental_ratio_pct"] - 74.51) < 1e-4
assert "2.1078%" in exp070_content
assert abs(baseline_telemetry["global_accidental_stall_rate_pct"] - 2.1078) < 1e-4

print("EXP-070 遥测指标验证通过")

# 6. 验证单变量实验日志 (EXP-071A, EXP-071B, EXP-072A, EXP-072B, EXP-073A, EXP-073B) 表格数值
single_variant_files = [
    ("exp071a", "tinybrain/04 实验日志/EXP-071A.md"),
    ("exp071b", "tinybrain/04 实验日志/EXP-071B.md"),
    ("exp072a", "tinybrain/04 实验日志/EXP-072A.md"),
    ("exp072b", "tinybrain/04 实验日志/EXP-072B.md"),
    ("exp073a", "tinybrain/04 实验日志/EXP-073A.md"),
    ("exp073b", "tinybrain/04 实验日志/EXP-073B.md"),
]

for vid, filepath in single_variant_files:
    with open(filepath, "r", encoding="utf-8") as f_log:
        content = f_log.read()
    entry = leaderboard[vid]
    pw = pairwise[vid]

    # 验证胜平负
    w_d_l_str = f"{entry['wins']} / {entry['draws']} / {entry['losses']}"
    assert w_d_l_str in content, f"{vid} 胜平负 {w_d_l_str} 未在 {filepath} 中出现"

    # 验证得分率
    score_rate_pct_str = f"{round(entry['score_rate'] * 100, 2):.2f}%"
    assert score_rate_pct_str in content, f"{vid} 得分率 {score_rate_pct_str} 未在 {filepath} 中出现"

    # 验证得分率置信区间
    ci_str = f"[{round(entry['score_rate_ci95'][0] * 100, 2):.2f}%, {round(entry['score_rate_ci95'][1] * 100, 2):.2f}%]"
    assert ci_str in content, f"{vid} 置信区间 {ci_str} 未在 {filepath} 中出现"

    # 验证纯胜率
    win_rate_pct_str = f"{round(entry['win_rate'] * 100, 2):.2f}%"
    assert win_rate_pct_str in content, f"{vid} 纯胜率 {win_rate_pct_str} 未在 {filepath} 中出现"

    # 验证停滞消除比例 (SRR) 若表格中包含
    if "停滞消除比例" in content:
        srr_sign = "+" if entry["stall_reduction_ratio_pct"] > 0 else ""
        srr_str = f"{srr_sign}{entry['stall_reduction_ratio_pct']:.2f}%"
        assert srr_str in content, f"{vid} 停滞消除比例 {srr_str} 未在 {filepath} 中出现"

print("单变量实验日志 EXP-071A..EXP-073B 表格指标验证通过")

# 7. 验证核心提案说明 Section 7.1 表 7.1
proposal_path = "tinybrain/03 Idea 构想拓扑/IDEA-007_目标执行与动作语义修复/IDEA-007_核心提案说明.md"
with open(proposal_path, "r", encoding="utf-8") as f_prop:
    proposal_content = f_prop.read()

assert "### 7.1 第一批次（Batch 1，EXP-070 至 EXP-074）实证消融矩阵与假设裁决" in proposal_content
table_7_1_match = re.search(r"### 7.1 第一批次.*?\| :--- \|.*?\n(.*?)\n\n### 7.2", proposal_content, re.DOTALL)
assert table_7_1_match is not None, "核心提案说明未定位到表 7.1"
table_7_1_lines = [line.strip() for line in table_7_1_match.group(1).strip().split("\n")]
assert len(table_7_1_lines) == 8, f"表 7.1 行数异常: {len(table_7_1_lines)}"

exp_id_to_vid = {
    "EXP-070": "baseline",
    "EXP-071A": "exp071a",
    "EXP-071B": "exp071b",
    "EXP-072A": "exp072a",
    "EXP-072B": "exp072b",
    "EXP-073A": "exp073a",
    "EXP-073B": "exp073b",
    "EXP-074": "exp074",
}

for line in table_7_1_lines:
    cols = [c.strip().replace("*", "").replace("`", "") for c in line.split("|")[1:-1]]
    exp_id = cols[0]
    vid = exp_id_to_vid[exp_id]
    entry = leaderboard[vid]

    # 检查综合得分率
    score_rate_val = round(entry["score_rate"] * 100, 2)
    score_rate_str = f"{score_rate_val:.2f}%"
    assert score_rate_str in cols[4], f"{exp_id} 表 7.1 得分率不匹配: {score_rate_str} not in {cols[4]}"

    # 检查纯胜率
    win_rate_val = round(entry["win_rate"] * 100, 2)
    win_rate_str = f"{win_rate_val:.2f}%"
    assert win_rate_str in cols[5], f"{exp_id} 表 7.1 胜率不匹配: {win_rate_str} not in {cols[5]}"

    # 检查受阻指令率
    blocked_val = entry["blocked_order_rate_pct"]
    blocked_str = f"{blocked_val:.2f}%"
    assert blocked_str in cols[6], f"{exp_id} 表 7.1 受阻率不匹配: {blocked_str} not in {cols[6]}"

    # 检查停滞消除比例
    srr_val = entry["stall_reduction_ratio_pct"]
    srr_sign = "+" if srr_val > 0 else ""
    srr_str = f"{srr_sign}{srr_val:.2f}%"
    assert srr_str in cols[7], f"{exp_id} 表 7.1 停滞消除比例不匹配: {srr_str} not in {cols[7]}"

    # 检查置信区间（EXP-071A 起）
    if exp_id != "EXP-070":
        s_ci_str = f"[{round(entry['score_rate_ci95'][0] * 100, 2):.2f}%, {round(entry['score_rate_ci95'][1] * 100, 2):.2f}%]"
        assert s_ci_str in cols[4], f"{exp_id} 表 7.1 得分率 CI 不匹配: {s_ci_str} not in {cols[4]}"
        w_ci_str = f"[{round(entry['win_rate_ci95'][0] * 100, 2):.2f}%, {round(entry['win_rate_ci95'][1] * 100, 2):.2f}%]"
        assert w_ci_str in cols[5], f"{exp_id} 表 7.1 胜率 CI 不匹配: {w_ci_str} not in {cols[5]}"

    # 检查裁决结果
    if exp_id == "EXP-072B":
        assert "REFUTED" in cols[8], f"{exp_id} 裁决应为 REFUTED: {cols[8]}"
    else:
        assert "SUPPORTED" in cols[8], f"{exp_id} 裁决应为 SUPPORTED: {cols[8]}"

print("核心提案说明 Table 7.1 验证通过: 全部 8 行数值及裁决与矩阵完全一致")
print("ALL_DATA_CONSISTENCY_CHALLENGES_PASSED_SUCCESSFULLY")
