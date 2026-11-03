import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch, Rectangle

# 配置全局中英文无衬线字体与负号解析
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = [
    "Noto Sans CJK SC",
    "Noto Sans CJK JP",
    "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False

# 检验输入数据文件存在性并就地崩溃
data_file_path = "/home/host/project/ants/data/out_ablation_matrix.json"
assert os.path.isfile(data_file_path), f"数据文件缺失: {data_file_path}"

# 检验并创建资产输出目录
output_directory = "/home/host/project/ants/tinybrain/ima"
os.makedirs(output_directory, exist_ok=True)

# 加载消融实验评测矩阵数据
with open(data_file_path, "r", encoding="utf-8") as file_handle:
    matrix_data = json.load(file_handle)

leaderboard_records = matrix_data["leaderboard"]
assert len(leaderboard_records) == 8, f"排行榜记录数量异常: {len(leaderboard_records)}"

map_breakdown_records = matrix_data["map_breakdown"]
assert len(map_breakdown_records) == 9, f"地图细分记录数量异常: {len(map_breakdown_records)}"

# 构建变体记录索引字典
variant_map = {item["variant_id"]: item for item in leaderboard_records}

# 变体标准显示名称与顺序序列
variant_display_names = {
    "baseline": "基准线 (Master)",
    "exp071a": "EXP-071A (采食位)",
    "exp071b": "EXP-071B (记忆清除)",
    "exp072a": "EXP-072A (驻留主权)",
    "exp072b": "EXP-072B (调度公平)",
    "exp073a": "EXP-073A (波前耗尽)",
    "exp073b": "EXP-073B (探索前沿)",
    "exp074": "EXP-074 (综合组合)",
}

# -------------------------------------------------------------------------
# 图表一：8变体综合得分率与纯胜率对比柱状图 (带 Wilson 95% 置信区间)
# -------------------------------------------------------------------------
sorted_by_rank = sorted(leaderboard_records, key=lambda item: item["rank"])
rank_variant_ids = [item["variant_id"] for item in sorted_by_rank]
rank_labels = [f"第{item['rank']}名\n{variant_display_names[item['variant_id']]}" for item in sorted_by_rank]

scores_pct = np.array([item["score_rate"] * 100 for item in sorted_by_rank])
wins_pct = np.array([item["win_rate"] * 100 for item in sorted_by_rank])

# 提取 Wilson 95% 置信区间上下界
score_err_low = np.array([(item["score_rate"] - item["score_rate_ci95"][0]) * 100 for item in sorted_by_rank])
score_err_high = np.array([(item["score_rate_ci95"][1] - item["score_rate"]) * 100 for item in sorted_by_rank])
score_yerr = np.vstack([score_err_low, score_err_high])

win_err_low = np.array([(item["win_rate"] - item["win_rate_ci95"][0]) * 100 for item in sorted_by_rank])
win_err_high = np.array([(item["win_rate_ci95"][1] - item["win_rate"]) * 100 for item in sorted_by_rank])
win_yerr = np.vstack([win_err_low, win_err_high])

x_indices = np.arange(len(sorted_by_rank))
bar_width = 0.38

fig_1, ax_1 = plt.subplots(figsize=(14, 6.5), dpi=300)

# 标定特定变体渲染颜色
colors_score = ["#2563eb" if vid == "exp074" else "#1e40af" for vid in rank_variant_ids]
colors_win = ["#10b981" if vid == "exp074" else "#059669" for vid in rank_variant_ids]

rects_score = ax_1.bar(
    x_indices - bar_width / 2,
    scores_pct,
    bar_width,
    yerr=score_yerr,
    capsize=4,
    color=colors_score,
    edgecolor="#0f172a",
    linewidth=1.1,
    alpha=0.92,
    label="综合得分率 Score Rate (%)"
)

rects_win = ax_1.bar(
    x_indices + bar_width / 2,
    wins_pct,
    bar_width,
    yerr=win_yerr,
    capsize=4,
    color=colors_win,
    edgecolor="#0f172a",
    linewidth=1.1,
    alpha=0.92,
    label="纯胜率 Pure Win Rate (%)"
)

# 标注柱状数值文本
for rect in rects_score:
    height = rect.get_height()
    ax_1.annotate(
        f"{height:.1f}%",
        xy=(rect.get_x() + rect.get_width() / 2, height),
        xytext=(0, 7),
        textcoords="offset points",
        ha="center",
        va="bottom",
        fontsize=8.8,
        fontweight="bold",
        color="#1e40af"
    )

for rect in rects_win:
    height = rect.get_height()
    ax_1.annotate(
        f"{height:.1f}%",
        xy=(rect.get_x() + rect.get_width() / 2, height),
        xytext=(0, 7),
        textcoords="offset points",
        ha="center",
        va="bottom",
        fontsize=8.8,
        fontweight="bold",
        color="#047857"
    )

# 绘制基准线水平参考虚线
baseline_score_val = variant_map["baseline"]["score_rate"] * 100
baseline_win_val = variant_map["baseline"]["win_rate"] * 100
ax_1.axhline(baseline_score_val, color="#1e40af", linestyle="--", linewidth=1.5, alpha=0.75, label=f"基准线得分率 ({baseline_score_val:.1f}%)")
ax_1.axhline(baseline_win_val, color="#059669", linestyle=":", linewidth=1.5, alpha=0.75, label=f"基准线纯胜率 ({baseline_win_val:.1f}%)")

# 标注综合变体性能突破文本框
exp074_idx = rank_variant_ids.index("exp074")
ax_1.annotate(
    "EXP-074 综合修复体:\n纯胜率 48.89% (全矩阵最高)\n综合得分率 59.44% (超越基线 +15.31%)\n消除单变量缺陷并达成最高对局胜势",
    xy=(exp074_idx + bar_width / 2, wins_pct[exp074_idx]),
    xytext=(exp074_idx - 1.2, 68),
    arrowprops=dict(facecolor="#b91c1c", shrink=0.08, width=1.4, headwidth=6),
    fontsize=9.5,
    fontweight="bold",
    color="#991b1b",
    bbox=dict(boxstyle="round,pad=0.45", facecolor="#fef2f2", edgecolor="#f87171", lw=1.2)
)

ax_1.set_ylabel("百分比率 (%)", fontsize=11.5, fontweight="bold")
ax_1.set_title("TinyBrains IDEA-007 消融矩阵 8 变体竞技综合得分率与纯胜率天梯 (含 Wilson 95% 置信区间)", fontsize=13.5, fontweight="bold", pad=15)
ax_1.set_xticks(x_indices)
ax_1.set_xticklabels(rank_labels, fontsize=9.5, fontweight="bold")
ax_1.set_ylim(0, 80)
ax_1.grid(True, axis="y", linestyle=":", alpha=0.55)
ax_1.legend(loc="upper right", framealpha=0.92, fontsize=10)

fig_1.tight_layout()
target_fig1_path = os.path.join(output_directory, "exp074_ablation_score_win_rates.png")
fig_1.savefig(target_fig1_path, bbox_inches="tight")
plt.close(fig_1)

# -------------------------------------------------------------------------
# 图表二：停滞削减比例 (SRR %) 与驻留原因堆叠结构分解图
# -------------------------------------------------------------------------
pipeline_order = ["baseline", "exp071a", "exp071b", "exp072a", "exp072b", "exp073a", "exp073b", "exp074"]
pipeline_labels = [variant_display_names[vid] for vid in pipeline_order]

srr_values = np.array([variant_map[vid]["stall_reduction_ratio_pct"] for vid in pipeline_order])

# 提取并计算每千兵步动作停滞频次
ant_turns_list = np.array([variant_map[vid]["telemetry_hold_stats"]["total_ant_turns"] for vid in pipeline_order], dtype=float)
reasons = ["harvest_wait", "defense", "conflict_wait", "potential_plateau"]

harvest_rate = np.array([variant_map[vid]["telemetry_hold_stats"]["reason_breakdown"]["harvest_wait"] / ant_turns_list[i] * 1000 for i, vid in enumerate(pipeline_order)])
defense_rate = np.array([variant_map[vid]["telemetry_hold_stats"]["reason_breakdown"]["defense"] / ant_turns_list[i] * 1000 for i, vid in enumerate(pipeline_order)])
conflict_rate = np.array([variant_map[vid]["telemetry_hold_stats"]["reason_breakdown"]["conflict_wait"] / ant_turns_list[i] * 1000 for i, vid in enumerate(pipeline_order)])
plateau_rate = np.array([variant_map[vid]["telemetry_hold_stats"]["reason_breakdown"]["potential_plateau"] / ant_turns_list[i] * 1000 for i, vid in enumerate(pipeline_order)])

fig_2, (ax_2a, ax_2b) = plt.subplots(1, 2, figsize=(15, 6.5), dpi=300)

# 子图 2A：停滞削减比例 (SRR %)
bar_colors_srr = []
for val in srr_values:
    if val > 15:
        bar_colors_srr.append("#059669")
    elif val > 0:
        bar_colors_srr.append("#0284c7")
    elif val == 0:
        bar_colors_srr.append("#64748b")
    else:
        bar_colors_srr.append("#dc2626")

y_pos = np.arange(len(pipeline_order))
bars_srr = ax_2a.barh(y_pos, srr_values, height=0.55, color=bar_colors_srr, edgecolor="#0f172a", linewidth=1.0, alpha=0.9)
ax_2a.axvline(0, color="#334155", linewidth=1.2, linestyle="-")

for bar in bars_srr:
    width = bar.get_width()
    offset = 2.0 if width >= 0 else -2.0
    align = "left" if width >= 0 else "right"
    sign = "+" if width > 0 else ""
    ax_2a.annotate(
        f"{sign}{width:.1f}%",
        xy=(width, bar.get_y() + bar.get_height() / 2),
        xytext=(offset, 0),
        textcoords="offset points",
        ha=align,
        va="center",
        fontsize=9.2,
        fontweight="bold",
        color="#0f172a"
    )

ax_2a.set_yticks(y_pos)
ax_2a.set_yticklabels(pipeline_labels, fontsize=10, fontweight="bold")
ax_2a.invert_yaxis()
ax_2a.set_xlabel("遥测停滞削减比例 SRR (%) [正值表示停滞减少，负值表示停滞恶化]", fontsize=10.5, fontweight="bold")
ax_2a.set_title("微观遥测停滞消除率演变 (Stall Reduction Ratio %)", fontsize=12.5, fontweight="bold", pad=12)
ax_2a.set_xlim(-70, 45)
ax_2a.grid(True, axis="x", linestyle=":", alpha=0.55)

ax_2a.annotate(
    "EXP-074 协同突破:\nSRR 达到 +34.8%\n全图探索与仲裁闭环消除单改停滞反弹",
    xy=(srr_values[-1], y_pos[-1]),
    xytext=(srr_values[-1] - 40, y_pos[-1] - 0.7),
    arrowprops=dict(facecolor="#059669", shrink=0.08, width=1.3, headwidth=5),
    fontsize=9.0,
    fontweight="bold",
    color="#065f46",
    bbox=dict(boxstyle="round,pad=0.35", facecolor="#ecfdf5", edgecolor="#6ee7b7")
)

# 子图 2B：驻留动因堆叠结构分解 (每千步频次)
bar_width_stack = 0.55
bars_harvest = ax_2b.bar(y_pos, harvest_rate, bar_width_stack, label="采食等待 (harvest_wait 意图)", color="#f59e0b", edgecolor="#0f172a", linewidth=0.9, alpha=0.9)
bars_defense = ax_2b.bar(y_pos, defense_rate, bar_width_stack, bottom=harvest_rate, label="防守驻留 (defense 意图)", color="#3b82f6", edgecolor="#0f172a", linewidth=0.9, alpha=0.9)
bars_conflict = ax_2b.bar(y_pos, conflict_rate, bar_width_stack, bottom=harvest_rate + defense_rate, label="冲突避让 (conflict_wait 协同)", color="#8b5cf6", edgecolor="#0f172a", linewidth=0.9, alpha=0.9)
bars_plateau = ax_2b.bar(y_pos, plateau_rate, bar_width_stack, bottom=harvest_rate + defense_rate + conflict_rate, label="势能死锁 (potential_plateau 意外停滞)", color="#ef4444", edgecolor="#0f172a", linewidth=0.9, alpha=0.9)

total_rates = harvest_rate + defense_rate + conflict_rate + plateau_rate
for idx, tot in enumerate(total_rates):
    ax_2b.annotate(
        f"{tot:.1f}",
        xy=(idx, tot),
        xytext=(0, 4),
        textcoords="offset points",
        ha="center",
        va="bottom",
        fontsize=8.8,
        fontweight="bold",
        color="#0f172a"
    )

ax_2b.set_xticks(y_pos)
ax_2b.set_xticklabels(pipeline_labels, fontsize=9.0, fontweight="bold", rotation=30, ha="right")
ax_2b.set_ylabel("驻留动因频次 (次 / 1,000 兵步)", fontsize=10.5, fontweight="bold")
ax_2b.set_title("单兵驻留动因微观结构分解 (意图驻留 vs 意外死锁)", fontsize=12.5, fontweight="bold", pad=12)
ax_2b.set_ylim(0, 55)
ax_2b.grid(True, axis="y", linestyle=":", alpha=0.55)
ax_2b.legend(loc="upper right", framealpha=0.92, fontsize=8.8)

# 标注势能平原死锁削减实证
ax_2b.annotate(
    "势能死锁由基线 21.1 骤降至 12.8 次/千步\n意外停滞下降 39.1%，意图采食正常释放",
    xy=(7, total_rates[7]),
    xytext=(3.5, 48),
    arrowprops=dict(facecolor="#b91c1c", shrink=0.08, width=1.3, headwidth=5),
    fontsize=9.0,
    fontweight="bold",
    color="#991b1b",
    bbox=dict(boxstyle="round,pad=0.35", facecolor="#fef2f2", edgecolor="#fca5a5")
)

fig_2.suptitle("TinyBrains IDEA-007 算法遥测指标：停滞削减比例与动因微观解耦全景", fontsize=14.0, fontweight="bold", y=0.98)
fig_2.tight_layout(rect=[0, 0.03, 1, 0.95])
target_fig2_path = os.path.join(output_directory, "exp074_stall_reduction_breakdown.png")
fig_2.savefig(target_fig2_path, bbox_inches="tight")
plt.close(fig_2)

# -------------------------------------------------------------------------
# 图表三：核心变体多维雷达图与归一化全景对比图
# -------------------------------------------------------------------------
key_variant_ids = ["baseline", "exp071a", "exp072a", "exp073b", "exp074"]
metric_labels = [
    "综合得分率\nScore Rate",
    "纯胜率\nWin Rate",
    "采食速率\nFood Gather",
    "停滞削减率\nStall Reduction",
    "摧毁蚁穴数\nHills Razed",
    "动作无冲突率\nCollision-Free",
]
num_metrics = len(metric_labels)

# 提取关键变体的物理指标数值
# 1. score_rate (%)
# 2. win_rate (%)
# 3. food_rate_per_100_turns
# 4. stall_reduction_ratio_pct (%)
# 5. mean_hills_razed
# 6. collision_free_rate (%) = 100 - blocked_order_rate_pct
raw_metrics_matrix = []
for vid in key_variant_ids:
    item = variant_map[vid]
    m_row = [
        item["score_rate"] * 100,
        item["win_rate"] * 100,
        item["food_rate_per_100_turns"],
        item["stall_reduction_ratio_pct"],
        item["mean_hills_razed"],
        100.0 - item["blocked_order_rate_pct"],
    ]
    raw_metrics_matrix.append(m_row)
raw_metrics_matrix = np.array(raw_metrics_matrix)

# 归一化映射至 [15, 100] 区间以保障雷达多边形显示稳定性
# 设定理论基准下界与上界
bounds_min = np.array([40.0, 0.0, 250.0, -60.0, 0.50, 99.0])
bounds_max = np.array([65.0, 55.0, 380.0, 40.0, 1.00, 100.0])

normalized_matrix = 15.0 + 85.0 * (raw_metrics_matrix - bounds_min) / (bounds_max - bounds_min)
normalized_matrix = np.clip(normalized_matrix, 5.0, 100.0)

# 计算雷达图环形角度
angles = np.linspace(0, 2 * np.pi, num_metrics, endpoint=False).tolist()
angles += angles[:1]

fig_3 = plt.figure(figsize=(16, 7.5), dpi=300)
ax_3a = fig_3.add_subplot(1, 2, 1, projection="polar")
ax_3b = fig_3.add_subplot(1, 2, 2)

radar_styles = {
    "baseline": {"color": "#64748b", "lw": 1.6, "ls": "--", "alpha": 0.05, "label": "基准线 Baseline"},
    "exp071a": {"color": "#f59e0b", "lw": 1.8, "ls": "-.", "alpha": 0.08, "label": "EXP-071A (采食位重构)"},
    "exp072a": {"color": "#8b5cf6", "lw": 1.8, "ls": ":", "alpha": 0.08, "label": "EXP-072A (冲突仲裁主权)"},
    "exp073b": {"color": "#0d9488", "lw": 1.8, "ls": "-.", "alpha": 0.08, "label": "EXP-073B (探索状态机)"},
    "exp074": {"color": "#1e40af", "lw": 3.0, "ls": "-", "alpha": 0.25, "label": "EXP-074 (综合修复体)"},
}

for i, vid in enumerate(key_variant_ids):
    values = normalized_matrix[i].tolist()
    values += values[:1]
    cfg = radar_styles[vid]
    ax_3a.plot(angles, values, color=cfg["color"], linewidth=cfg["lw"], linestyle=cfg["ls"], label=cfg["label"])
    ax_3a.fill(angles, values, color=cfg["color"], alpha=cfg["alpha"])

ax_3a.set_theta_offset(np.pi / 2)
ax_3a.set_theta_direction(-1)
ax_3a.set_xticks(angles[:-1])
ax_3a.set_xticklabels(metric_labels, fontsize=10.0, fontweight="bold")
ax_3a.set_ylim(0, 105)
ax_3a.set_yticks([25, 50, 75, 100])
ax_3a.set_yticklabels(["25", "50", "75", "100 (极值)"], fontsize=8.0, color="#64748b")
ax_3a.grid(True, linestyle=":", alpha=0.6)
ax_3a.set_title("核心消融变体多维能力雷达包络面 (归一化)", fontsize=12.5, fontweight="bold", pad=18)
ax_3a.legend(loc="upper right", bbox_to_anchor=(0.15, 0.05), framealpha=0.92, fontsize=9.0)

# 子图 3B：六项核心维度物理数值分组对比柱状图
bar_groups_x = np.arange(num_metrics)
width_group = 0.15
group_offsets = np.linspace(-2 * width_group, 2 * width_group, len(key_variant_ids))

# 采用各维度自适应标度绘制
for idx_v, vid in enumerate(key_variant_ids):
    cfg = radar_styles[vid]
    # 使用归一化比例展示相对强弱
    vals = normalized_matrix[idx_v]
    rects = ax_3b.bar(
        bar_groups_x + group_offsets[idx_v],
        vals,
        width_group,
        label=cfg["label"],
        color=cfg["color"],
        edgecolor="#0f172a",
        linewidth=0.8,
        alpha=0.9
    )

# 标注物理极值文本说明
annotations_text = [
    f"071A 最高\n{variant_map['exp071a']['score_rate']*100:.1f}%",
    f"074 最高\n{variant_map['exp074']['win_rate']*100:.1f}%",
    f"074 极值\n{variant_map['exp074']['food_rate_per_100_turns']:.1f}",
    f"074 突破\n+{variant_map['exp074']['stall_reduction_ratio_pct']:.1f}%",
    f"072A 最高\n{variant_map['exp072a']['mean_hills_razed']:.2f}",
    "071A/074\n100.0%",
]

for idx_m, txt in enumerate(annotations_text):
    ax_3b.annotate(
        txt,
        xy=(idx_m, 103),
        xytext=(0, 5),
        textcoords="offset points",
        ha="center",
        va="bottom",
        fontsize=8.5,
        fontweight="bold",
        color="#1e40af"
    )

ax_3b.set_xticks(bar_groups_x)
short_metric_names = ["得分率", "纯胜率", "采食率", "停滞削减", "摧毁蚁穴", "无冲突率"]
ax_3b.set_xticklabels(short_metric_names, fontsize=10.5, fontweight="bold")
ax_3b.set_ylabel("相对能力指数指数 (归一化刻度 0~100)", fontsize=10.5, fontweight="bold")
ax_3b.set_title("核心指标相对强度分组对照 (柱高对应综合能力)", fontsize=12.5, fontweight="bold", pad=12)
ax_3b.set_ylim(0, 122)
ax_3b.grid(True, axis="y", linestyle=":", alpha=0.55)
ax_3b.legend(loc="upper left", framealpha=0.92, fontsize=9.0)

fig_3.suptitle("TinyBrains IDEA-007 综合变体 (EXP-074) 与单变量核心能力多维全景对比", fontsize=14.0, fontweight="bold", y=0.98)
fig_3.tight_layout(rect=[0, 0.03, 1, 0.95])
target_fig3_path = os.path.join(output_directory, "exp074_ablation_multimetric_radar.png")
fig_3.savefig(target_fig3_path, bbox_inches="tight")
plt.close(fig_3)

# -------------------------------------------------------------------------
# 图表四：9地图 × 8变体得分率与纯胜率双核热力图
# -------------------------------------------------------------------------
ordered_maps = [
    "tiny-open-2p-1h.json",
    "tiny-corridor-2p-1h.json",
    "tiny-colonnade-2p-1h.json",
    "small-arena-2p-1h.json",
    "small-rooms-2p-1h.json",
    "small-islands-2p-1h.json",
    "medium-crossroads-2p-1h.json",
    "medium-labyrinth-2p-1h.json",
    "medium-valley-2p-1h.json",
]

map_chinese_labels = [
    "tiny-open\n(开放平原)",
    "tiny-corridor\n(狭窄走廊)",
    "tiny-colonnade\n(柱廊网格)",
    "small-arena\n(角斗竞技)",
    "small-rooms\n(连通隔间)",
    "small-islands\n(扼喉岛链)",
    "medium-crossroads\n(十字轴心)",
    "medium-labyrinth\n(复杂迷宫)",
    "medium-valley\n(环抱山谷)",
]

variant_col_ids = ["baseline", "exp071a", "exp071b", "exp072a", "exp072b", "exp073a", "exp073b", "exp074"]
variant_col_labels = [
    "基准线\nBaseline",
    "071A\n采食位",
    "071B\n记忆清除",
    "072A\n冲突仲裁",
    "072B\n调度公平",
    "073A\n波前耗尽",
    "073B\n探索前沿",
    "074\n综合体",
]

num_rows = len(ordered_maps)
num_cols = len(variant_col_ids)

heatmap_scores = np.zeros((num_rows, num_cols))
heatmap_wins = np.zeros((num_rows, num_cols))

for r_idx, map_name in enumerate(ordered_maps):
    map_entry = map_breakdown_records[map_name]
    for c_idx, vid in enumerate(variant_col_ids):
        heatmap_scores[r_idx, c_idx] = map_entry[vid]["score_rate"] * 100
        heatmap_wins[r_idx, c_idx] = map_entry[vid]["win_rate"] * 100

fig_4, (ax_4a, ax_4b) = plt.subplots(1, 2, figsize=(16, 8.5), dpi=300)

cmap_scores = plt.cm.YlGnBu
cmap_wins = plt.cm.YlOrRd

im_score = ax_4a.imshow(heatmap_scores, cmap=cmap_scores, vmin=20, vmax=100, aspect="auto")
im_win = ax_4b.imshow(heatmap_wins, cmap=cmap_wins, vmin=0, vmax=100, aspect="auto")

# 标注单元格文本
for r in range(num_rows):
    for c in range(num_cols):
        s_val = heatmap_scores[r, c]
        w_val = heatmap_wins[r, c]
        color_s = "white" if s_val >= 65 else "#0f172a"
        color_w = "white" if w_val >= 55 else "#0f172a"
        ax_4a.text(c, r, f"{s_val:.1f}%", ha="center", va="center", color=color_s, fontsize=9.0, fontweight="bold")
        ax_4b.text(c, r, f"{w_val:.1f}%", ha="center", va="center", color=color_w, fontsize=9.0, fontweight="bold")

# 设置坐标轴与标签
ax_4a.set_xticks(np.arange(num_cols))
ax_4a.set_xticklabels(variant_col_labels, fontsize=9.5, fontweight="bold")
ax_4a.set_yticks(np.arange(num_rows))
ax_4a.set_yticklabels(map_chinese_labels, fontsize=9.5, fontweight="bold")
ax_4a.set_title("9 张基准地图 × 8 变体 综合得分率矩阵 (%)", fontsize=12.5, fontweight="bold", pad=12)

ax_4b.set_xticks(np.arange(num_cols))
ax_4b.set_xticklabels(variant_col_labels, fontsize=9.5, fontweight="bold")
ax_4b.set_yticks(np.arange(num_rows))
ax_4b.set_yticklabels(map_chinese_labels, fontsize=9.5, fontweight="bold")
ax_4b.set_title("9 张基准地图 × 8 变体 纯胜率矩阵 (%)", fontsize=12.5, fontweight="bold", pad=12)

# 突出标定 medium-labyrinth 下 EXP-074 达成的 100% 极值单元格
lab_row_idx = ordered_maps.index("medium-labyrinth-2p-1h.json")
exp074_col_idx = variant_col_ids.index("exp074")

rect_highlight_a = Rectangle((exp074_col_idx - 0.48, lab_row_idx - 0.48), 0.96, 0.96, fill=False, edgecolor="#dc2626", linewidth=3.0)
rect_highlight_b = Rectangle((exp074_col_idx - 0.48, lab_row_idx - 0.48), 0.96, 0.96, fill=False, edgecolor="#dc2626", linewidth=3.0)
ax_4a.add_patch(rect_highlight_a)
ax_4b.add_patch(rect_highlight_b)

# 标注高拓扑复杂度迷宫全胜说明
ax_4a.annotate(
    "迷宫绝对优势: 100.0% 得分率",
    xy=(exp074_col_idx, lab_row_idx),
    xytext=(exp074_col_idx - 2.8, lab_row_idx - 0.7),
    arrowprops=dict(facecolor="#dc2626", shrink=0.1, width=1.4, headwidth=5),
    fontsize=9.2,
    fontweight="bold",
    color="#991b1b",
    bbox=dict(boxstyle="round,pad=0.35", facecolor="#fef2f2", edgecolor="#f87171", lw=1.2)
)

ax_4b.annotate(
    "迷宫绝对统治: 100.0% 胜率 (基线 7.1%)",
    xy=(exp074_col_idx, lab_row_idx),
    xytext=(exp074_col_idx - 3.2, lab_row_idx - 0.7),
    arrowprops=dict(facecolor="#dc2626", shrink=0.1, width=1.4, headwidth=5),
    fontsize=9.2,
    fontweight="bold",
    color="#991b1b",
    bbox=dict(boxstyle="round,pad=0.35", facecolor="#fef2f2", edgecolor="#f87171", lw=1.2)
)

cbar_a = fig_4.colorbar(im_score, ax=ax_4a, fraction=0.046, pad=0.04)
cbar_a.ax.set_ylabel("综合得分率 Score Rate (%)", rotation=-90, va="bottom", fontsize=10.0, fontweight="bold")

cbar_b = fig_4.colorbar(im_win, ax=ax_4b, fraction=0.046, pad=0.04)
cbar_b.ax.set_ylabel("纯胜率 Pure Win Rate (%)", rotation=-90, va="bottom", fontsize=10.0, fontweight="bold")

fig_4.suptitle("TinyBrains IDEA-007 基准地图拓扑适应性热力矩阵：综合修复体在复杂迷宫地图达成 100% 全胜", fontsize=14.0, fontweight="bold", y=0.98)
fig_4.tight_layout(rect=[0, 0.03, 1, 0.95])
target_fig4_path = os.path.join(output_directory, "exp074_map_performance_heatmap.png")
fig_4.savefig(target_fig4_path, bbox_inches="tight")
plt.close(fig_4)

print("EXP074_PLOTS_GENERATED_SUCCESSFULLY")
