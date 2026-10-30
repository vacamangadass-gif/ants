import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = [
    "Noto Sans CJK SC",
    "Noto Sans CJK JP",
    "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False

output_dir = "/home/host/project/ants/tinybrain/ima"
os.makedirs(output_dir, exist_ok=True)

# 1. 绘制训练收敛演化折线图 (EXP-030 vs EXP-031)
with open("/home/host/project/ants/runs/exp030/history.json", "r", encoding="utf-8") as f:
    hist030 = json.load(f)
with open("/home/host/project/ants/runs/exp031/history.json", "r", encoding="utf-8") as f:
    hist031 = json.load(f)

epochs_030 = [x["epoch"] for x in hist030["epochs"]]
train_loss_030 = [x["train_loss"] for x in hist030["epochs"]]
val_loss_030 = [x["val_loss"] for x in hist030["epochs"]]
train_acc_030 = [x["train_acc"] * 100 for x in hist030["epochs"]]
val_acc_030 = [x["val_acc"] * 100 for x in hist030["epochs"]]

epochs_031 = [x["epoch"] for x in hist031["epochs"]]
train_loss_031 = [x["train_loss"] for x in hist031["epochs"]]
val_loss_031 = [x["val_loss"] for x in hist031["epochs"]]
train_acc_031 = [x["train_acc"] * 100 for x in hist031["epochs"]]
val_acc_031 = [x["val_acc"] * 100 for x in hist031["epochs"]]

best_loss_030 = min(val_loss_030)
best_loss_031 = min(val_loss_031)
best_acc_030 = max(val_acc_030)
best_acc_031 = max(val_acc_031)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5), dpi=300)

# 子图 1：交叉熵损失对比
ax1.plot(epochs_030, train_loss_030, linestyle="--", color="#f87171", alpha=0.8, label="EXP-030 训练集损失")
ax1.plot(epochs_030, val_loss_030, linestyle="-", marker="o", color="#dc2626", linewidth=2.2, label=f"EXP-030 验证集损失 (最佳 {best_loss_030:.4f})")
ax1.plot(epochs_031, train_loss_031, linestyle="--", color="#60a5fa", alpha=0.8, label="EXP-031 训练集损失")
ax1.plot(epochs_031, val_loss_031, linestyle="-", marker="s", color="#2563eb", linewidth=2.2, label=f"EXP-031 验证集损失 (最佳 {best_loss_031:.4f})")

ax1.set_title("训练与验证交叉熵损失演化 (Loss)", fontsize=13, fontweight="bold", pad=12)
ax1.set_xlabel("训练轮次 (Epoch)", fontsize=11)
ax1.set_ylabel("交叉熵损失 (Cross-Entropy Loss)", fontsize=11)
ax1.set_xticks(epochs_030)
ax1.grid(True, linestyle=":", alpha=0.6)
ax1.legend(loc="upper right", framealpha=0.9, fontsize=9.5)

# 子图 2：动作拟合精度对比
ax2.plot(epochs_030, train_acc_030, linestyle="--", color="#f87171", alpha=0.8, label="EXP-030 训练集精度")
ax2.plot(epochs_030, val_acc_030, linestyle="-", marker="o", color="#dc2626", linewidth=2.2, label=f"EXP-030 验证集精度 (最佳 {best_acc_030:.2f}%)")
ax2.plot(epochs_031, train_acc_031, linestyle="--", color="#60a5fa", alpha=0.8, label="EXP-031 训练集精度")
ax2.plot(epochs_031, val_acc_031, linestyle="-", marker="s", color="#2563eb", linewidth=2.2, label=f"EXP-031 验证集精度 (最佳 {best_acc_031:.2f}%)")


ax2.set_title("动作决策拟合准确率演化 (Accuracy %)", fontsize=13, fontweight="bold", pad=12)
ax2.set_xlabel("训练轮次 (Epoch)", fontsize=11)
ax2.set_ylabel("动作拟合精度 (%)", fontsize=11)
ax2.set_xticks(epochs_030)
ax2.grid(True, linestyle=":", alpha=0.6)
ax2.legend(loc="lower right", framealpha=0.9, fontsize=9.5)

fig.suptitle("EXP-030 纯专家扩增 与 EXP-031 DAgger 交互聚合训练收敛曲线对比", fontsize=15, fontweight="bold", y=0.98)
fig.tight_layout(rect=[0, 0.03, 1, 0.95])
plot1_path = os.path.join(output_dir, "exp030_exp031_training_convergence.png")
fig.savefig(plot1_path, bbox_inches="tight")
plt.close(fig)

# 2. 绘制 DAgger 核心遥测图 (双轴组合图)
with open("/home/host/project/ants/runs/exp031/dagger_telemetry.json", "r", encoding="utf-8") as f:
    dag_data = json.load(f)

stages = ["基线 D_0", "迭代 1 (k=1)", "迭代 2 (k=2)", "迭代 3 (k=3)"]
x_indices = np.arange(len(stages))
cumulative_sigs = [
    dag_data["baseline_signatures_d0"],
    dag_data["cumulative_signatures"]["k1"],
    dag_data["cumulative_signatures"]["k2"],
    dag_data["cumulative_signatures"]["k3"],
]
expansion_factors = [
    1.0000,
    dag_data["expansion_factors"]["e1"],
    dag_data["expansion_factors"]["e2"],
    dag_data["expansion_factors"]["e3"],
]
# 分歧率仅在推演迭代 k=1, 2, 3 存在
disagreement_rates = [
    np.nan,
    dag_data["disagreement_rates"]["d1"] * 100,
    dag_data["disagreement_rates"]["d2"] * 100,
    dag_data["disagreement_rates"]["d3"] * 100,
]

fig, ax_left = plt.subplots(figsize=(10, 6), dpi=300)
ax_right = ax_left.twinx()

bar_colors = ["#94a3b8", "#38bdf8", "#0284c7", "#0369a1"]
bars = ax_left.bar(x_indices, cumulative_sigs, width=0.45, color=bar_colors, edgecolor="#0f172a", linewidth=1.1, alpha=0.88, label="累计状态空间指纹数 (左轴)")

for i, bar in enumerate(bars):
    h = bar.get_height()
    factor_str = f"{expansion_factors[i]:.4f}×"
    sig_str = f"{cumulative_sigs[i]:,}"
    ax_left.annotate(f"{sig_str}\n({factor_str})",
                     xy=(bar.get_x() + bar.get_width() / 2, h),
                     xytext=(0, 6),
                     textcoords="offset points",
                     ha="center", va="bottom", fontsize=9.5, fontweight="bold", color="#0f172a")

ax_left.set_ylabel("累计不重复宏观状态指纹数 (Signatures)", fontsize=11, fontweight="bold", color="#0369a1")
ax_left.set_ylim(0, 125000)
ax_left.tick_params(axis="y", labelcolor="#0369a1")

# 右轴折线图：动作分歧率
disagree_x = x_indices[1:]
disagree_y = disagreement_rates[1:]
line = ax_right.plot(disagree_x, disagree_y, color="#e11d48", marker="D", markersize=9, linewidth=2.6, label="学生-教师动作分歧率 (右轴)")

for x, y in zip(disagree_x, disagree_y):
    ax_right.annotate(f"{y:.2f}%",
                      xy=(x, y),
                      xytext=(15, 8),
                      textcoords="offset points",
                      ha="left", va="bottom", fontsize=10.5, fontweight="bold", color="#e11d48",
                      arrowprops=dict(arrowstyle="->", color="#e11d48", lw=1.2))

ax_right.set_ylabel("动作分歧率 Disagreement Rate (%)", fontsize=11, fontweight="bold", color="#e11d48")
ax_right.set_ylim(15, 30)
ax_right.tick_params(axis="y", labelcolor="#e11d48")

ax_left.set_xticks(x_indices)
ax_left.set_xticklabels(stages, fontsize=11, fontweight="bold")
ax_left.grid(True, axis="y", linestyle=":", alpha=0.5)

ax_left.set_title("EXP-031 DAgger 交互遥测：状态空间扩展倍率与动作分歧率衰减趋势", fontsize=13.5, fontweight="bold", pad=15)

lines1, labels1 = ax_left.get_legend_handles_labels()
lines2, labels2 = ax_right.get_legend_handles_labels()
ax_left.legend(lines1 + lines2, labels1 + labels2, loc="upper left", framealpha=0.9, fontsize=10)

fig.tight_layout()
plot2_path = os.path.join(output_dir, "exp031_dagger_telemetry.png")
fig.savefig(plot2_path, bbox_inches="tight")
plt.close(fig)

# 读取循环赛数据
with open("/home/host/project/ants/data/tournament_p3_results/tournament_metrics.json", "r", encoding="utf-8") as f:
    tour_data = json.load(f)

# 3. 循环赛综合得分率与纯胜率对比柱状图
models = [entry["name"] for entry in tour_data["leaderboard"]]
score_rates = [(entry["points"] / entry["matches"]) * 100 for entry in tour_data["leaderboard"]]
pure_win_rates = [(entry["wins"] / entry["matches"]) * 100 for entry in tour_data["leaderboard"]]

x = np.arange(len(models))
width = 0.35

fig, ax = plt.subplots(figsize=(10.5, 6), dpi=300)

rects1 = ax.bar(x - width/2, score_rates, width, label="综合得分率 Score Rate (%)", color="#2563eb", edgecolor="#1e3a8a", alpha=0.9)
rects2 = ax.bar(x + width/2, pure_win_rates, width, label="纯胜率 Pure Win Rate (%)", color="#10b981", edgecolor="#064e3b", alpha=0.9)

for rect in rects1:
    height = rect.get_height()
    ax.annotate(f"{height:.1f}%",
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center", va="bottom", fontsize=9.5, fontweight="bold", color="#1e3a8a")

for rect in rects2:
    height = rect.get_height()
    ax.annotate(f"{height:.1f}%",
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center", va="bottom", fontsize=9.5, fontweight="bold", color="#064e3b")

ax.set_ylabel("百分比 (%)", fontsize=11, fontweight="bold")
ax.set_title("9 图双向循环赛各模型综合得分率与纯胜率天梯榜对比 (每模型 72 场)", fontsize=13.5, fontweight="bold", pad=15)
ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=11, fontweight="bold")
ax.set_ylim(0, 88)
ax.grid(True, axis="y", linestyle=":", alpha=0.5)
ax.legend(loc="upper right", framealpha=0.9, fontsize=10.5)

# 标注核心科学发现：EXP-031 相对 EXP-030 净胜优势
ax.annotate("假说 H3 实证确证: 样本量相同(500k)\nDAgger 相对纯离线扩增得分率净胜 +21.5%",
            xy=(0 - width/2, score_rates[0]),
            xytext=(0.15, 75),
            arrowprops=dict(facecolor="#b91c1c", shrink=0.08, width=1.5, headwidth=7),
            fontsize=10, fontweight="bold", color="#b91c1c",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#fef2f2", edgecolor="#b91c1c", lw=1.2))

fig.tight_layout()
plot3_path = os.path.join(output_dir, "tournament_score_win_rates.png")
fig.savefig(plot3_path, bbox_inches="tight")
plt.close(fig)

# 4. 逆境平均存活回合与场均碰撞受阻率对比散点图
blocked_rates = []
survival_turns = []
bubble_sizes = []
model_names = []

for entry in tour_data["leaderboard"]:
    model_names.append(entry["name"])
    # 动作受阻率 = blocked_orders / total_orders * 100
    br = (entry["blocked_orders"] / entry["total_orders"]) * 100
    blocked_rates.append(br)
    # 败局平均存活回合
    loss_turns = entry["loss_turns"]
    mean_st = float(np.mean(loss_turns)) if loss_turns else 0.0
    survival_turns.append(mean_st)
    # 得分率决定气泡大小
    sr = (entry["points"] / entry["matches"]) * 100
    bubble_sizes.append(sr * 25)

fig, ax = plt.subplots(figsize=(10.5, 6.5), dpi=300)

colors = ["#2563eb", "#10b981", "#ef4444", "#f59e0b", "#8b5cf6"]

for i in range(len(model_names)):
    ax.scatter(blocked_rates[i], survival_turns[i], s=bubble_sizes[i],
               color=colors[i], alpha=0.75, edgecolors="#0f172a", linewidth=1.5,
               label=f"{model_names[i]} (得分率 {bubble_sizes[i]/25:.1f}%)")
    
    # 标注文本与偏移
    offset_x = 0.08
    offset_y = 0.8
    if model_names[i] == "exp031":
        offset_x = 0.12
        offset_y = -3.2
    elif model_names[i] == "exp030":
        offset_y = -2.2
    elif model_names[i] == "exp011":
        offset_y = -2.0
    ax.annotate(f"{model_names[i]}\n(受阻率: {blocked_rates[i]:.2f}%, 败局存活: {survival_turns[i]:.1f}t)",
                xy=(blocked_rates[i], survival_turns[i]),
                xytext=(blocked_rates[i] + offset_x, survival_turns[i] + offset_y),
                fontsize=9.5, fontweight="bold", color="#1e293b",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#f8fafc", edgecolor="#cbd5e1", alpha=0.9))

# 标定区域底色：左上方为最优区（低受阻率、高逆境存活）
ax.axvspan(0.5, 2.5, ymin=0.55, ymax=1.0, color="#dcfce7", alpha=0.25)
ax.text(0.60, 91.2, "最优自愈与畅行区域\n(低动作受阻率 + 高逆境存活韧性)",
        fontsize=9.5, color="#166534", fontweight="bold",
        bbox=dict(boxstyle="square,pad=0.4", facecolor="#f0fdf4", edgecolor="#86efac", alpha=0.8))


ax.set_xlabel("移动动作受阻率 Blocked Movement Rate (%) [越低越顺畅]", fontsize=11, fontweight="bold")
ax.set_ylabel("败局平均存活回合 Mean Survival Turns in Defeats [越高越顽强]", fontsize=11, fontweight="bold")
ax.set_title("9 图循环赛导航质量与逆境生存韧性分布关系 (气泡面积对应综合得分率)", fontsize=13.5, fontweight="bold", pad=15)
ax.set_xlim(0.5, 5.5)
ax.set_ylim(50, 95)
ax.grid(True, linestyle=":", alpha=0.6)
ax.legend(loc="lower right", framealpha=0.9, fontsize=9.5)

fig.tight_layout()
plot4_path = os.path.join(output_dir, "tournament_resilience_scatter.png")
fig.savefig(plot4_path, bbox_inches="tight")
plt.close(fig)

# 5. 两两对决综合得分率与战绩热力图 (5x5 Pairwise Heatmap)
models_order = ["exp031", "exp011", "exp030", "nano-bc-max-r", "nano-bc"]
n_models = len(models_order)
matrix_rates = np.zeros((n_models, n_models))
matrix_text = [["" for _ in range(n_models)] for _ in range(n_models)]

h2h = tour_data["head_to_head"]

for i, m1 in enumerate(models_order):
    for j, m2 in enumerate(models_order):
        if i == j:
            matrix_rates[i, j] = np.nan
            matrix_text[i][j] = "—"
        else:
            key1 = f"{m1}_vs_{m2}"
            if key1 in h2h:
                rec = h2h[key1]
                p1_wins = rec["p1_wins"]
                draws = rec["draws"]
                p2_wins = rec["p2_wins"]
                score_rate = (rec["p1_points"] / rec["matches"]) * 100
                matrix_rates[i, j] = score_rate
                matrix_text[i][j] = f"{score_rate:.1f}%\n({p1_wins}-{draws}-{p2_wins})"

fig, ax = plt.subplots(figsize=(9, 7.5), dpi=300)

cmap = plt.cm.YlGnBu.copy()
cmap.set_bad(color="#e2e8f0")

im = ax.imshow(matrix_rates, cmap=cmap, vmin=20, vmax=80)

ax.set_xticks(np.arange(n_models))
ax.set_yticks(np.arange(n_models))
ax.set_xticklabels(models_order, fontsize=11, fontweight="bold")
ax.set_yticklabels(models_order, fontsize=11, fontweight="bold")

plt.setp(ax.get_xticklabels(), rotation=30, ha="right", rotation_mode="anchor")

# 在单元格内标注文字
for i in range(n_models):
    for j in range(n_models):
        val = matrix_rates[i, j]
        txt = matrix_text[i][j]
        if np.isnan(val):
            ax.text(j, i, txt, ha="center", va="center", color="#64748b", fontsize=12, fontweight="bold")
        else:
            text_color = "white" if val >= 55 else "#0f172a"
            ax.text(j, i, txt, ha="center", va="center", color=text_color, fontsize=10, fontweight="bold")

cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.ax.set_ylabel("行模型对列模型得分率 Score Rate (%)", rotation=-90, va="bottom", fontsize=10.5, fontweight="bold")

ax.set_xlabel("对手模型 (Opponent)", fontsize=11.5, fontweight="bold", labelpad=8)
ax.set_ylabel("焦点模型 (Focal Entrant)", fontsize=11.5, fontweight="bold", labelpad=8)
ax.set_title("9 图循环赛两两对决综合得分率与战绩矩阵 (18 场双向对弈)", fontsize=13.5, fontweight="bold", pad=15)

fig.tight_layout()
plot5_path = os.path.join(output_dir, "tournament_pairwise_heatmap.png")
fig.savefig(plot5_path, bbox_inches="tight")
plt.close(fig)

print("ALL_PLOTS_GENERATED_SUCCESSFULLY")
