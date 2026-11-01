import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = [
    "Noto Sans CJK SC",
    "Noto Sans CJK JP",
    "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False

output_dir = "/home/host/project/ants/tinybrain/ima"
os.makedirs(output_dir, exist_ok=True)


def compute_kneedle(x_vals, y_vals, invert=False):
    x = np.array(x_vals, dtype=float)
    y = np.array(y_vals, dtype=float)
    x_span = np.max(x) - np.min(x)
    y_span = np.max(y) - np.min(y)
    x_norm = (x - np.min(x)) / x_span if x_span != 0 else np.zeros_like(x)
    y_norm = (y - np.min(y)) / y_span if y_span != 0 else np.zeros_like(y)
    if invert:
        d = (1.0 - x_norm) - y_norm
    else:
        d = y_norm - x_norm
    knee_idx = int(np.argmax(d))
    return d, knee_idx


def compute_marginal_derivatives(x_vals, y_vals, scale=100000):
    x = np.array(x_vals, dtype=float)
    y = np.array(y_vals, dtype=float)
    dx = np.diff(x)
    dy = np.diff(y)
    derivatives = (dy / dx) * scale
    return derivatives


# 加载对局评测与遥测数据
with open("/home/host/project/ants/data/tournament_p3_results/tournament_metrics.json", "r", encoding="utf-8") as f:
    tour_data = json.load(f)

with open("/home/host/project/ants/runs/exp031/dagger_telemetry.json", "r", encoding="utf-8") as f:
    dag_nano = json.load(f)

with open("/home/host/project/ants/runs/exp033/dagger_telemetry.json", "r", encoding="utf-8") as f:
    dag_micro = json.load(f)

leaderboard = tour_data["leaderboard"]
board_dict = {entry["name"]: entry for entry in leaderboard}

# 提取路线 A (Nano) 与路线 B (Micro) 评测序列
iters = np.arange(1, 9)
nano_names = [f"exp031_iter{k}" for k in iters]
micro_names = [f"exp033_iter{k}" for k in iters]

nano_scores = np.array([board_dict[m]["score_rate"] * 100 for m in nano_names])
nano_wins = np.array([board_dict[m]["win_rate"] * 100 for m in nano_names])
nano_blocked = np.array([board_dict[m]["blocked_move_pct"] for m in nano_names])
nano_survive = np.array([board_dict[m]["mean_survive_turn"] for m in nano_names])

micro_scores = np.array([board_dict[m]["score_rate"] * 100 for m in micro_names])
micro_wins = np.array([board_dict[m]["win_rate"] * 100 for m in micro_names])
micro_blocked = np.array([board_dict[m]["blocked_move_pct"] for m in micro_names])
micro_survive = np.array([board_dict[m]["mean_survive_turn"] for m in micro_names])

# 累计样本规模 (基线 250,000 + 轮次增量 83,333)
micro_samples = np.array([250000 + 83333 * k for k in iters])
nano_samples = np.array([250000 + 83333 * k for k in iters])

# 1. 绘制历史图表：EXP-030 vs EXP-031 训练收敛演化折线图
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
fig.savefig(os.path.join(output_dir, "exp030_exp031_training_convergence.png"), bbox_inches="tight")
plt.close(fig)

# 2. 绘制历史图表：EXP-031 DAgger 交互遥测图
stages_031 = ["基线 D_0", "迭代 1 (k=1)", "迭代 2 (k=2)", "迭代 3 (k=3)"]
x_indices_031 = np.arange(len(stages_031))
cumulative_sigs_031 = [
    dag_nano["baseline_signatures_d0"],
    dag_nano["cumulative_signatures"]["k1"],
    dag_nano["cumulative_signatures"]["k2"],
    dag_nano["cumulative_signatures"]["k3"],
]
expansion_factors_031 = [
    1.0000,
    dag_nano["expansion_factors"]["e1"],
    dag_nano["expansion_factors"]["e2"],
    dag_nano["expansion_factors"]["e3"],
]
disagreement_rates_031 = [
    np.nan,
    dag_nano["disagreement_rates"]["d1"] * 100,
    dag_nano["disagreement_rates"]["d2"] * 100,
    dag_nano["disagreement_rates"]["d3"] * 100,
]

fig, ax_left = plt.subplots(figsize=(10, 6), dpi=300)
ax_right = ax_left.twinx()
bar_colors = ["#94a3b8", "#38bdf8", "#0284c7", "#0369a1"]
bars = ax_left.bar(x_indices_031, cumulative_sigs_031, width=0.45, color=bar_colors, edgecolor="#0f172a", linewidth=1.1, alpha=0.88, label="累计状态空间指纹数 (左轴)")

for i, bar in enumerate(bars):
    h = bar.get_height()
    factor_str = f"{expansion_factors_031[i]:.4f}×"
    sig_str = f"{cumulative_sigs_031[i]:,}"
    ax_left.annotate(f"{sig_str}\n({factor_str})",
                     xy=(bar.get_x() + bar.get_width() / 2, h),
                     xytext=(0, 6),
                     textcoords="offset points",
                     ha="center", va="bottom", fontsize=9.5, fontweight="bold", color="#0f172a")

ax_left.set_ylabel("累计不重复宏观状态指纹数 (Signatures)", fontsize=11, fontweight="bold", color="#0369a1")
ax_left.set_ylim(0, 125000)
ax_left.tick_params(axis="y", labelcolor="#0369a1")

disagree_x = x_indices_031[1:]
disagree_y = disagreement_rates_031[1:]
ax_right.plot(disagree_x, disagree_y, color="#e11d48", marker="D", markersize=9, linewidth=2.6, label="学生-教师动作分歧率 (右轴)")

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
ax_left.set_xticks(x_indices_031)
ax_left.set_xticklabels(stages_031, fontsize=11, fontweight="bold")
ax_left.grid(True, axis="y", linestyle=":", alpha=0.5)
ax_left.set_title("EXP-031 DAgger 交互遥测：状态空间扩展倍率与动作分歧率衰减趋势", fontsize=13.5, fontweight="bold", pad=15)

lines1, labels1 = ax_left.get_legend_handles_labels()
lines2, labels2 = ax_right.get_legend_handles_labels()
ax_left.legend(lines1 + lines2, labels1 + labels2, loc="upper left", framealpha=0.9, fontsize=10)
fig.tight_layout()
fig.savefig(os.path.join(output_dir, "exp031_dagger_telemetry.png"), bbox_inches="tight")
plt.close(fig)

# 3. 科研图表 1：双轨闭环迭代轮次综合得分率与移动受阻率演化轨迹
fig, (ax_score, ax_block) = plt.subplots(1, 2, figsize=(14, 6), dpi=300)

# 子图 1：综合得分率演变
ax_score.plot(iters, micro_scores, marker="o", markersize=8, color="#1e40af", linewidth=2.5, label="路线 B (Micro 28 通道)")
ax_score.plot(iters, nano_scores, marker="s", markersize=8, color="#059669", linewidth=2.5, label="路线 A (Nano 12 通道)")

for i, txt in enumerate(micro_scores):
    ax_score.annotate(f"{txt:.1f}%", (iters[i], micro_scores[i]), textcoords="offset points", xytext=(0, 9), ha="center", fontsize=9.5, fontweight="bold", color="#1e40af")

for i, txt in enumerate(nano_scores):
    ax_score.annotate(f"{txt:.1f}%", (iters[i], nano_scores[i]), textcoords="offset points", xytext=(0, -15), ha="center", fontsize=9.5, fontweight="bold", color="#059669")

# 标注拐点
ax_score.annotate("路线 B 核心边际收益拐点 (k=3)\n得分率 62.22% (基线显著突破)",
                  xy=(3, micro_scores[2]), xytext=(1.8, 68),
                  arrowprops=dict(facecolor="#1e40af", shrink=0.08, width=1.5, headwidth=6),
                  fontsize=9.5, fontweight="bold", color="#1e40af",
                  bbox=dict(boxstyle="round,pad=0.4", facecolor="#eff6ff", edgecolor="#93c5fd"))

ax_score.annotate("路线 B 平台饱和拐点 (k=5)\n得分率 64.81% (高位收敛)",
                  xy=(5, micro_scores[4]), xytext=(4.3, 72),
                  arrowprops=dict(facecolor="#1e40af", shrink=0.08, width=1.5, headwidth=6),
                  fontsize=9.5, fontweight="bold", color="#1e40af",
                  bbox=dict(boxstyle="round,pad=0.4", facecolor="#eff6ff", edgecolor="#93c5fd"))

ax_score.annotate("路线 A 容量瓶颈拐点 (k=2)\n得分率 46.30% (参数容量受限)",
                  xy=(2, nano_scores[1]), xytext=(2.2, 38),
                  arrowprops=dict(facecolor="#059669", shrink=0.08, width=1.5, headwidth=6),
                  fontsize=9.5, fontweight="bold", color="#059669",
                  bbox=dict(boxstyle="round,pad=0.4", facecolor="#ecfdf5", edgecolor="#a7f3d0"))

ax_score.set_title("双轨闭环迭代综合得分率演化 (Score Rate)", fontsize=13, fontweight="bold", pad=12)
ax_score.set_xlabel("DAgger 交互迭代轮次 (Iteration k)", fontsize=11, fontweight="bold")
ax_score.set_ylabel("综合得分率 Score Rate (%)", fontsize=11, fontweight="bold")
ax_score.set_xticks(iters)
ax_score.set_xticklabels([f"迭代 {k}" for k in iters], fontsize=10, fontweight="bold")
ax_score.set_ylim(24, 76)
ax_score.grid(True, linestyle=":", alpha=0.6)
ax_score.legend(loc="lower right", framealpha=0.9, fontsize=10.5)

# 子图 2：移动受阻比例演变
ax_block.plot(iters, micro_blocked, marker="o", markersize=8, color="#1e40af", linewidth=2.5, label="路线 B (Micro 28 通道)")
ax_block.plot(iters, nano_blocked, marker="s", markersize=8, color="#059669", linewidth=2.5, label="路线 A (Nano 12 通道)")

for i, txt in enumerate(micro_blocked):
    ax_block.annotate(f"{txt:.2f}%", (iters[i], micro_blocked[i]), textcoords="offset points", xytext=(0, -14), ha="center", fontsize=9.5, fontweight="bold", color="#1e40af")

for i, txt in enumerate(nano_blocked):
    ax_block.annotate(f"{txt:.2f}%", (iters[i], nano_blocked[i]), textcoords="offset points", xytext=(0, 9), ha="center", fontsize=9.5, fontweight="bold", color="#059669")

ax_block.annotate("物理移动纠偏关键拐点 (k=2)\n路线 B 降幅 -43.7%, 路线 A 降幅 -30.6%\n两轮闭环基本消除移动碰撞阻塞",
                  xy=(2, micro_blocked[1]), xytext=(2.5, 2.5),
                  arrowprops=dict(facecolor="#dc2626", shrink=0.08, width=1.5, headwidth=6),
                  fontsize=9.5, fontweight="bold", color="#dc2626",
                  bbox=dict(boxstyle="round,pad=0.4", facecolor="#fef2f2", edgecolor="#fca5a5"))

ax_block.set_title("移动动作受阻率演化 (Blocked Movement Rate)", fontsize=13, fontweight="bold", pad=12)
ax_block.set_xlabel("DAgger 交互迭代轮次 (Iteration k)", fontsize=11, fontweight="bold")
ax_block.set_ylabel("移动动作受阻比例 (%) [越低越畅行]", fontsize=11, fontweight="bold")
ax_block.set_xticks(iters)
ax_block.set_xticklabels([f"迭代 {k}" for k in iters], fontsize=10, fontweight="bold")
ax_block.set_ylim(0.2, 3.8)
ax_block.grid(True, linestyle=":", alpha=0.6)
ax_block.legend(loc="upper right", framealpha=0.9, fontsize=10.5)

fig.suptitle("DAgger 双轨闭环推演全周期得分率与移动受阻率演化轨迹", fontsize=15, fontweight="bold", y=0.98)
fig.tight_layout(rect=[0, 0.03, 1, 0.95])
fig.savefig(os.path.join(output_dir, "dagger_iterations_evolution.png"), bbox_inches="tight")
plt.close(fig)

# 4. 科研图表 2：双轨样本规模展开得分率、离散边际导数与 Kneedle 弦距离曲线
fig, (ax_marg, ax_kneedle) = plt.subplots(1, 2, figsize=(14, 6), dpi=300)

micro_derivatives = compute_marginal_derivatives(micro_samples, micro_scores)
kneedle_dist_b, kneedle_idx_b = compute_kneedle(micro_samples, micro_scores)
kneedle_dist_block, _ = compute_kneedle(micro_samples, micro_blocked, invert=True)

# 子图 1：综合得分率与离散边际导数 (双轴)
ax_marg_right = ax_marg.twinx()

ax_marg.plot(micro_samples / 1000, micro_scores, marker="o", markersize=7.5, color="#1e40af", linewidth=2.4, label="路线 B 综合得分率 (左轴)")
ax_marg.plot(nano_samples / 1000, nano_scores, marker="s", markersize=7.5, color="#059669", linewidth=2.4, label="路线 A 综合得分率 (左轴)")

deriv_x = (micro_samples[1:] + micro_samples[:-1]) / (2 * 1000)
ax_marg_right.plot(deriv_x, micro_derivatives, marker="D", markersize=7, color="#dc2626", linestyle="--", linewidth=2.0, label="路线 B 一阶边际导数 (右轴)")

for x, y in zip(deriv_x, micro_derivatives):
    sign = "+" if y > 0 else ""
    ax_marg_right.annotate(f"{sign}{y:.2f}%", (x, y), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=8.5, fontweight="bold", color="#dc2626")

ax_marg.set_xlabel("累计样本规模 Cumulative Samples (千席位步数 k seat-turns)", fontsize=11, fontweight="bold")
ax_marg.set_ylabel("综合得分率 Score Rate (%)", fontsize=11, fontweight="bold", color="#1e40af")
ax_marg_right.set_ylabel("一阶离散边际导数 (% / 100k 样本)", fontsize=11, fontweight="bold", color="#dc2626")
ax_marg.set_ylim(25, 75)
ax_marg_right.set_ylim(-18, 16)
ax_marg.set_xticks(micro_samples / 1000)
ax_marg.set_xticklabels([f"{int(s)}k" for s in micro_samples / 1000], fontsize=9.5, fontweight="bold")
ax_marg.grid(True, linestyle=":", alpha=0.6)

ax_marg.annotate("核心边际拐点: 500k 样本 (k=3)\n前 3 轮平均导数: +6.00%/100k\n后 5 轮平均导数: +0.84%/100k\n边际效率衰减 7.1 倍",
                 xy=(500, micro_scores[2]), xytext=(350, 28),
                 arrowprops=dict(facecolor="#dc2626", shrink=0.08, width=1.5, headwidth=6),
                 fontsize=9, fontweight="bold", color="#991b1b",
                 bbox=dict(boxstyle="round,pad=0.4", facecolor="#fef2f2", edgecolor="#fca5a5"))

lines_l1, labels_l1 = ax_marg.get_legend_handles_labels()
lines_r1, labels_r1 = ax_marg_right.get_legend_handles_labels()
ax_marg.legend(lines_l1 + lines_r1, labels_l1 + labels_r1, loc="upper left", framealpha=0.9, fontsize=9.5)
ax_marg.set_title("双轨样本展开得分率与一阶离散边际导数", fontsize=13, fontweight="bold", pad=12)

# 子图 2：移动受阻比例与 Kneedle 垂直弦距离
ax_kneedle_right = ax_kneedle.twinx()

ax_kneedle.plot(micro_samples / 1000, micro_blocked, marker="o", markersize=7.5, color="#1e40af", linewidth=2.4, label="路线 B 移动受阻率 (左轴)")
ax_kneedle.plot(nano_samples / 1000, nano_blocked, marker="s", markersize=7.5, color="#059669", linewidth=2.4, label="路线 A 移动受阻率 (左轴)")

ax_kneedle_right.plot(micro_samples / 1000, kneedle_dist_b, marker="^", markersize=8, color="#7c3aed", linestyle="-.", linewidth=2.2, label="路线 B 得分率 Kneedle 弦距离 (右轴)")

for i, d in enumerate(kneedle_dist_b):
    if i in [1, 2, 4]:
        ax_kneedle_right.annotate(f"D={d:.4f}", (micro_samples[i] / 1000, d), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8.5, fontweight="bold", color="#7c3aed")

ax_kneedle_right.annotate("Kneedle 全局极大值\n(D3 = 0.4539, 500k 样本)",
                          xy=(500, kneedle_dist_b[2]), xytext=(530, 0.48),
                          arrowprops=dict(facecolor="#7c3aed", shrink=0.08, width=1.5, headwidth=6),
                          fontsize=9, fontweight="bold", color="#6b21a8",
                          bbox=dict(boxstyle="round,pad=0.4", facecolor="#faf5ff", edgecolor="#d8b4fe"))

ax_kneedle_right.annotate("次级极大值 (D5 = 0.3598)\n667k 样本容量饱和",
                          xy=(667, kneedle_dist_b[4]), xytext=(680, 0.32),
                          arrowprops=dict(facecolor="#7c3aed", shrink=0.08, width=1.5, headwidth=6),
                          fontsize=9, fontweight="bold", color="#6b21a8",
                          bbox=dict(boxstyle="round,pad=0.4", facecolor="#faf5ff", edgecolor="#d8b4fe"))

ax_kneedle.set_xlabel("累计样本规模 Cumulative Samples (千席位步数 k seat-turns)", fontsize=11, fontweight="bold")
ax_kneedle.set_ylabel("移动动作受阻比例 (%)", fontsize=11, fontweight="bold", color="#1e40af")
ax_kneedle_right.set_ylabel("Kneedle 垂直基准弦距离 $D_k$", fontsize=11, fontweight="bold", color="#7c3aed")
ax_kneedle.set_ylim(0.2, 3.8)
ax_kneedle_right.set_ylim(-0.8, 0.6)
ax_kneedle.set_xticks(micro_samples / 1000)
ax_kneedle.set_xticklabels([f"{int(s)}k" for s in micro_samples / 1000], fontsize=9.5, fontweight="bold")
ax_kneedle.grid(True, linestyle=":", alpha=0.6)

lines_l2, labels_l2 = ax_kneedle.get_legend_handles_labels()
lines_r2, labels_r2 = ax_kneedle_right.get_legend_handles_labels()
ax_kneedle.legend(lines_l2 + lines_r2, labels_l2 + labels_r2, loc="upper right", framealpha=0.9, fontsize=9.5)
ax_kneedle.set_title("移动受阻率演进与 Kneedle 曲率极大值分布", fontsize=13, fontweight="bold", pad=12)

fig.suptitle("DAgger 双轨样本规模扩展边际收益导数与 Kneedle 曲率拐点定量分析", fontsize=15, fontweight="bold", y=0.98)
fig.tight_layout(rect=[0, 0.03, 1, 0.95])
fig.savefig(os.path.join(output_dir, "dagger_samples_marginal_benefit.png"), bbox_inches="tight")
plt.close(fig)

# 5. 科研图表 3：路线 B (Micro) 算法机理遥测指标图 (双轴柱线组合图)
stages_033 = ["基线 D_0"] + [f"迭代 {k} (k={k})" for k in iters]
x_indices_033 = np.arange(len(stages_033))
cumulative_sigs_033 = [dag_micro["baseline_signatures_d0"]] + [dag_micro["cumulative_signatures"][f"k{k}"] for k in iters]
expansion_factors_033 = [1.0000] + [dag_micro["expansion_factors"][f"e{k}"] for k in iters]
disagreement_rates_033 = [np.nan] + [dag_micro["disagreement_rates"][f"d{k}"] * 100 for k in iters]

fig, ax_left = plt.subplots(figsize=(11, 6), dpi=300)
ax_right = ax_left.twinx()

bar_colors_033 = ["#94a3b8"] + [plt.cm.Blues(0.45 + 0.06 * k) for k in range(len(iters))]
bars = ax_left.bar(x_indices_033, cumulative_sigs_033, width=0.48, color=bar_colors_033, edgecolor="#0f172a", linewidth=1.1, alpha=0.9, label="累计状态空间指纹数 (左轴)")

for i, bar in enumerate(bars):
    h = bar.get_height()
    factor_str = f"{expansion_factors_033[i]:.4f}×"
    sig_str = f"{cumulative_sigs_033[i]:,}"
    ax_left.annotate(f"{sig_str}\n({factor_str})",
                     xy=(bar.get_x() + bar.get_width() / 2, h),
                     xytext=(0, 6),
                     textcoords="offset points",
                     ha="center", va="bottom", fontsize=8.5, fontweight="bold", color="#0f172a")

ax_left.set_ylabel("累计不重复宏观状态指纹数 (Signatures)", fontsize=11, fontweight="bold", color="#1e40af")
ax_left.set_ylim(0, 155000)
ax_left.tick_params(axis="y", labelcolor="#1e40af")

disagree_x_033 = x_indices_033[1:]
disagree_y_033 = disagreement_rates_033[1:]
ax_right.plot(disagree_x_033, disagree_y_033, color="#dc2626", marker="D", markersize=8.5, linewidth=2.5, label="诱导分布动作分歧率 (右轴)")

for x, y in zip(disagree_x_033, disagree_y_033):
    ax_right.annotate(f"{y:.2f}%",
                      xy=(x, y),
                      xytext=(10, 8),
                      textcoords="offset points",
                      ha="left", va="bottom", fontsize=9.5, fontweight="bold", color="#dc2626")

ax_right.set_ylabel("动作分歧率 Disagreement Rate (%)", fontsize=11, fontweight="bold", color="#dc2626")
ax_right.set_ylim(4, 15)
ax_right.tick_params(axis="y", labelcolor="#dc2626")

ax_left.set_xticks(x_indices_033)
ax_left.set_xticklabels(stages_033, fontsize=10, fontweight="bold", rotation=20, ha="right")
ax_left.grid(True, axis="y", linestyle=":", alpha=0.5)
ax_left.set_title("EXP-033 (Micro) DAgger 交互遥测：状态空间扩展倍率与动作分歧率衰减趋势", fontsize=13.5, fontweight="bold", pad=15)

lines1, labels1 = ax_left.get_legend_handles_labels()
lines2, labels2 = ax_right.get_legend_handles_labels()
ax_left.legend(lines1 + lines2, labels1 + labels2, loc="upper left", framealpha=0.9, fontsize=10)

fig.tight_layout()
fig.savefig(os.path.join(output_dir, "exp033_dagger_telemetry.png"), bbox_inches="tight")
plt.close(fig)

# 6. 科研图表 4：Nano vs Micro 动作分歧率与状态空间扩展倍率对比
disagree_nano = np.array([dag_nano["disagreement_rates"][f"d{k}"] * 100 for k in iters])
disagree_micro = np.array([dag_micro["disagreement_rates"][f"d{k}"] * 100 for k in iters])
expansion_nano = np.array([dag_nano["expansion_factors"][f"e{k}"] for k in iters])
expansion_micro = np.array([dag_micro["expansion_factors"][f"e{k}"] for k in iters])

fig, (ax_dcomp, ax_ecomp) = plt.subplots(1, 2, figsize=(13, 5.5), dpi=300)

# 子图 1：分歧率对比
ax_dcomp.plot(iters, disagree_micro, marker="o", markersize=8, color="#1e40af", linewidth=2.4, label="路线 B (Micro 28 通道)")
ax_dcomp.plot(iters, disagree_nano, marker="s", markersize=8, color="#059669", linewidth=2.4, label="路线 A (Nano 12 通道)")

for i, (m, n) in enumerate(zip(disagree_micro, disagree_nano)):
    ax_dcomp.annotate(f"{m:.2f}%", (iters[i], m), textcoords="offset points", xytext=(0, -14), ha="center", fontsize=9, fontweight="bold", color="#1e40af")
    ax_dcomp.annotate(f"{n:.2f}%", (iters[i], n), textcoords="offset points", xytext=(0, 9), ha="center", fontsize=9, fontweight="bold", color="#059669")

ax_dcomp.annotate("模型容量效应:\nMicro 动作分歧率稳定控制在 6%~12%\n相对 Nano 降低超 50% 分歧",
                  xy=(8, disagree_micro[-1]), xytext=(4.5, 9),
                  arrowprops=dict(facecolor="#1e40af", shrink=0.08, width=1.5, headwidth=6),
                  fontsize=9.5, fontweight="bold", color="#1e40af",
                  bbox=dict(boxstyle="round,pad=0.4", facecolor="#eff6ff", edgecolor="#93c5fd"))

ax_dcomp.set_title("学生-教师动作分歧率衰减对比 (Disagreement Rate)", fontsize=12.5, fontweight="bold", pad=12)
ax_dcomp.set_xlabel("交互迭代轮次 (Iteration k)", fontsize=11, fontweight="bold")
ax_dcomp.set_ylabel("动作分歧率 (%)", fontsize=11, fontweight="bold")
ax_dcomp.set_xticks(iters)
ax_dcomp.set_xticklabels([f"k={k}" for k in iters], fontsize=10, fontweight="bold")
ax_dcomp.set_ylim(4, 27)
ax_dcomp.grid(True, linestyle=":", alpha=0.6)
ax_dcomp.legend(loc="upper right", framealpha=0.9, fontsize=10)

# 子图 2：状态空间扩展倍率对比
ax_ecomp.plot(iters, expansion_micro, marker="o", markersize=8, color="#1e40af", linewidth=2.4, label="路线 B (Micro 28 通道)")
ax_ecomp.plot(iters, expansion_nano, marker="s", markersize=8, color="#059669", linewidth=2.4, label="路线 A (Nano 12 通道)")

for i, (m, n) in enumerate(zip(expansion_micro, expansion_nano)):
    ax_ecomp.annotate(f"{m:.4f}×", (iters[i], m), textcoords="offset points", xytext=(0, -14), ha="center", fontsize=8.5, fontweight="bold", color="#1e40af")
    ax_ecomp.annotate(f"{n:.4f}×", (iters[i], n), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8.5, fontweight="bold", color="#059669")

ax_ecomp.set_title("状态空间扩展倍率演变对比 (Expansion Factor)", fontsize=12.5, fontweight="bold", pad=12)
ax_ecomp.set_xlabel("交互迭代轮次 (Iteration k)", fontsize=11, fontweight="bold")
ax_ecomp.set_ylabel("相对基线 D_0 扩展倍率 (×)", fontsize=11, fontweight="bold")
ax_ecomp.set_xticks(iters)
ax_ecomp.set_xticklabels([f"k={k}" for k in iters], fontsize=10, fontweight="bold")
ax_ecomp.set_ylim(1.05, 1.78)
ax_ecomp.grid(True, linestyle=":", alpha=0.6)
ax_ecomp.legend(loc="lower right", framealpha=0.9, fontsize=10)

fig.suptitle("DAgger 双轨遥测指标对比：模型容量对策略拟合精度与状态探索范围的影响", fontsize=14.5, fontweight="bold", y=0.98)
fig.tight_layout(rect=[0, 0.03, 1, 0.95])
fig.savefig(os.path.join(output_dir, "dagger_telemetry_comparison.png"), bbox_inches="tight")
plt.close(fig)

# 7. 科研图表 5：16 模型循环赛天梯综合得分率与纯胜率柱状图
models_all = [entry["name"] for entry in leaderboard]
classes_all = [entry["model_class"] for entry in leaderboard]
scores_all = [(entry["points"] / entry["matches"]) * 100 for entry in leaderboard]
wins_all = [(entry["wins"] / entry["matches"]) * 100 for entry in leaderboard]

n_entrants = len(models_all)
x_pos = np.arange(n_entrants)
bar_w = 0.38

fig, ax = plt.subplots(figsize=(14, 6.5), dpi=300)

colors_score = ["#1e40af" if c == "micro" else "#047857" for c in classes_all]
colors_win = ["#3b82f6" if c == "micro" else "#10b981" for c in classes_all]

rects_score = ax.bar(x_pos - bar_w / 2, scores_all, bar_w, color=colors_score, edgecolor="#0f172a", linewidth=1.0, alpha=0.9)
rects_win = ax.bar(x_pos + bar_w / 2, wins_all, bar_w, color=colors_win, edgecolor="#0f172a", linewidth=1.0, alpha=0.9)

for rect in rects_score:
    h = rect.get_height()
    ax.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8.5, fontweight="bold", color="#0f172a")

for rect in rects_win:
    h = rect.get_height()
    ax.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8.5, fontweight="bold", color="#0f172a")

ax.annotate("假说 H4 实证确证: Micro 架构全面超越 Nano\n前 8 名席位全部由 Micro 模型占据\n第 8 轮 Micro (65.74%) 净胜 Nano (48.89%) +16.85% 得分率",
            xy=(0, scores_all[0]), xytext=(0.5, 74),
            arrowprops=dict(facecolor="#b91c1c", shrink=0.08, width=1.5, headwidth=6),
            fontsize=10, fontweight="bold", color="#b91c1c",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#fef2f2", edgecolor="#b91c1c", lw=1.2))

legend_elements = [
    Patch(facecolor="#1e40af", edgecolor="#0f172a", label="Micro (路线 B) 综合得分率"),
    Patch(facecolor="#3b82f6", edgecolor="#0f172a", label="Micro (路线 B) 纯胜率"),
    Patch(facecolor="#047857", edgecolor="#0f172a", label="Nano (路线 A) 综合得分率"),
    Patch(facecolor="#10b981", edgecolor="#0f172a", label="Nano (路线 A) 纯胜率"),
]
ax.legend(handles=legend_elements, loc="upper right", framealpha=0.9, fontsize=9.5)

ax.set_ylabel("百分比 (%)", fontsize=11, fontweight="bold")
ax.set_title("9 图双向席位互换循环赛 16 模型天梯榜 (每模型 270 场对局，总计 2,160 场)", fontsize=13.5, fontweight="bold", pad=15)
ax.set_xticks(x_pos)
ax.set_xticklabels(models_all, fontsize=9.5, fontweight="bold", rotation=30, ha="right")
ax.set_ylim(0, 84)
ax.grid(True, axis="y", linestyle=":", alpha=0.5)

fig.tight_layout()
fig.savefig(os.path.join(output_dir, "tournament_score_win_rates.png"), bbox_inches="tight")
plt.close(fig)

# 8. 科研图表 6：16 模型移动受阻率 vs 败局存活回合韧性散点图
blocked_all = [(entry["blocked_orders"] / entry["total_orders"]) * 100 for entry in leaderboard]
survival_all = [float(np.mean(entry["loss_turns"])) if entry["loss_turns"] else 0.0 for entry in leaderboard]
bubble_sizes = [s * 25 for s in scores_all]

fig, ax = plt.subplots(figsize=(11, 7), dpi=300)

color_palette_micro = ["#1e3a8a", "#1e40af", "#2563eb", "#3b82f6", "#4338ca", "#6366f1", "#7c3aed", "#8b5cf6"]
color_palette_nano = ["#064e3b", "#047857", "#059669", "#10b981", "#34d399", "#0d9488", "#14b8a6", "#2dd4bf"]

micro_count = 0
nano_count = 0
for i in range(n_entrants):
    m_class = classes_all[i]
    if m_class == "micro":
        c = color_palette_micro[micro_count % len(color_palette_micro)]
        micro_count += 1
    else:
        c = color_palette_nano[nano_count % len(color_palette_nano)]
        nano_count += 1

    ax.scatter(blocked_all[i], survival_all[i], s=bubble_sizes[i],
               color=c, alpha=0.8, edgecolors="#0f172a", linewidth=1.4)

    off_x = 0.05
    off_y = 0.6
    if models_all[i] in ["exp033_iter8", "exp033_iter5"]:
        off_x = 0.07
        off_y = -1.8
    elif models_all[i] in ["exp031_iter8"]:
        off_x = 0.06
        off_y = 0.8
    elif models_all[i] in ["exp031_iter7", "exp031_iter3"]:
        off_x = 0.06
        off_y = -1.6

    ax.annotate(f"{models_all[i]}\n({blocked_all[i]:.2f}%, {survival_all[i]:.1f}t)",
                xy=(blocked_all[i], survival_all[i]),
                xytext=(blocked_all[i] + off_x, survival_all[i] + off_y),
                fontsize=8.5, fontweight="bold", color="#1e293b",
                bbox=dict(boxstyle="round,pad=0.25", facecolor="#f8fafc", edgecolor="#cbd5e1", alpha=0.9))

# 标定最优畅行自愈区域
ax.axvspan(0.4, 1.5, ymin=0.2, ymax=1.0, color="#dcfce7", alpha=0.25)
ax.text(0.55, 82.5, "最优畅行与自愈区域\n(低受阻率 < 1.5% + 高抗逆存活)",
        fontsize=9.5, color="#166534", fontweight="bold",
        bbox=dict(boxstyle="square,pad=0.4", facecolor="#f0fdf4", edgecolor="#86efac", alpha=0.85))

scatter_legends = [
    Patch(facecolor="#1e40af", edgecolor="#0f172a", label="Micro (路线 B 迭代 1~8)"),
    Patch(facecolor="#059669", edgecolor="#0f172a", label="Nano (路线 A 迭代 1~8)"),
]
ax.legend(handles=scatter_legends, loc="upper right", framealpha=0.9, fontsize=9.5)

ax.set_xlabel("移动动作受阻率 Blocked Movement Rate (%) [越低越畅行]", fontsize=11, fontweight="bold")
ax.set_ylabel("败局平均存活回合 Mean Survival Turns in Defeats [越高越顽强]", fontsize=11, fontweight="bold")
ax.set_title("9 图循环赛导航质量与逆境生存韧性分布关系 (气泡面积对应综合得分率)", fontsize=13.5, fontweight="bold", pad=15)
ax.set_xlim(0.4, 3.6)
ax.set_ylim(64, 85)
ax.grid(True, linestyle=":", alpha=0.6)

fig.tight_layout()
fig.savefig(os.path.join(output_dir, "tournament_resilience_scatter.png"), bbox_inches="tight")
plt.close(fig)

# 9. 科研图表 7：16×16 两两对抗综合得分率热力图
models_order = [entry["name"] for entry in leaderboard]
n_m = len(models_order)
matrix_rates = np.zeros((n_m, n_m))
matrix_text = [["" for _ in range(n_m)] for _ in range(n_m)]

h2h = tour_data["head_to_head"]

for i, m1 in enumerate(models_order):
    for j, m2 in enumerate(models_order):
        if i == j:
            matrix_rates[i, j] = np.nan
            matrix_text[i][j] = "—"
        else:
            k_pair = f"{m1}_vs_{m2}"
            if k_pair in h2h:
                rec = h2h[k_pair]
                p1_wins = rec["p1_wins"]
                draws = rec["draws"]
                p2_wins = rec["p2_wins"]
                score_rate = (rec["p1_points"] / rec["matches"]) * 100
                matrix_rates[i, j] = score_rate
                matrix_text[i][j] = f"{score_rate:.0f}%\n({p1_wins}-{p2_wins})"

fig, ax = plt.subplots(figsize=(13, 11), dpi=300)

cmap = plt.cm.YlGnBu.copy()
cmap.set_bad(color="#e2e8f0")

im = ax.imshow(matrix_rates, cmap=cmap, vmin=15, vmax=85)

ax.set_xticks(np.arange(n_m))
ax.set_yticks(np.arange(n_m))
ax.set_xticklabels(models_order, fontsize=9.5, fontweight="bold")
ax.set_yticklabels(models_order, fontsize=9.5, fontweight="bold")

plt.setp(ax.get_xticklabels(), rotation=35, ha="right", rotation_mode="anchor")

# 标注单元格文本
for i in range(n_m):
    for j in range(n_m):
        val = matrix_rates[i, j]
        txt = matrix_text[i][j]
        if np.isnan(val):
            ax.text(j, i, txt, ha="center", va="center", color="#64748b", fontsize=11, fontweight="bold")
        else:
            txt_color = "white" if val >= 55 else "#0f172a"
            ax.text(j, i, txt, ha="center", va="center", color=txt_color, fontsize=7.5, fontweight="bold")

# 绘制 Micro 阵营与 Nano 阵营分隔线 (前 8 为 Micro，后 8 为 Nano)
ax.axhline(7.5, color="#dc2626", linewidth=2.0, linestyle="--")
ax.axvline(7.5, color="#dc2626", linewidth=2.0, linestyle="--")

cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.ax.set_ylabel("行模型对列模型得分率 Score Rate (%)", rotation=-90, va="bottom", fontsize=10.5, fontweight="bold")

ax.set_xlabel("对手模型 (Opponent)", fontsize=11.5, fontweight="bold", labelpad=8)
ax.set_ylabel("焦点模型 (Focal Entrant)", fontsize=11.5, fontweight="bold", labelpad=8)
ax.set_title("9 图循环赛 16×16 两两对决综合得分率与战绩矩阵 (每单元 18 场双向对弈)", fontsize=13.5, fontweight="bold", pad=15)

fig.tight_layout()
fig.savefig(os.path.join(output_dir, "tournament_pairwise_heatmap.png"), bbox_inches="tight")
plt.close(fig)

# 10. 科研图表 8：路线 B (EXP-033 Micro) 训练损失与验证损失收敛折线图 (迭代 1 至 8)
all_epochs = []
exp033_train_loss = []
exp033_val_loss = []
exp033_train_acc = []
exp033_val_acc = []
stage_transition_epochs = []

epoch_counter = 0
for k in range(1, 9):
    hpath = f"/home/host/project/ants/runs/exp033/k{k}/history.json"
    with open(hpath, "r", encoding="utf-8") as f:
        hk = json.load(f)
    for ep in hk["epochs"]:
        epoch_counter += 1
        all_epochs.append(epoch_counter)
        exp033_train_loss.append(ep["train_loss"])
        exp033_val_loss.append(ep["val_loss"])
        exp033_train_acc.append(ep["train_acc"] * 100)
        exp033_val_acc.append(ep["val_acc"] * 100)
    if k < 8:
        stage_transition_epochs.append(epoch_counter)

fig, (ax_l, ax_a) = plt.subplots(1, 2, figsize=(13, 5.5), dpi=300)

# 子图 1：交叉熵损失收敛
ax_l.plot(all_epochs, exp033_train_loss, linestyle="--", color="#60a5fa", alpha=0.8, label="Micro 训练集损失")
ax_l.plot(all_epochs, exp033_val_loss, linestyle="-", marker="o", markersize=4, color="#1e40af", linewidth=2.0, label="Micro 验证集损失")

for step in stage_transition_epochs:
    ax_l.axvline(step + 0.5, color="#cbd5e1", linestyle=":", linewidth=1.2)

best_loss_idx = int(np.argmin(exp033_val_loss))
best_loss_val = exp033_val_loss[best_loss_idx]
best_loss_ep = all_epochs[best_loss_idx]

ax_l.annotate(f"最优验证损失: {best_loss_val:.4f}\n(轮次 {best_loss_ep}, 迭代 8)",
              xy=(best_loss_ep, best_loss_val), xytext=(best_loss_ep - 12, best_loss_val + 0.15),
              arrowprops=dict(facecolor="#1e40af", shrink=0.08, width=1.4, headwidth=5),
              fontsize=9, fontweight="bold", color="#1e40af",
              bbox=dict(boxstyle="round,pad=0.3", facecolor="#eff6ff", edgecolor="#93c5fd"))

ax_l.set_title("EXP-033 策略网络交叉熵损失演化 (Loss)", fontsize=12.5, fontweight="bold", pad=12)
ax_l.set_xlabel("累计训练轮次 (Cumulative Epochs across k1~k8)", fontsize=11, fontweight="bold")
ax_l.set_ylabel("交叉熵损失 (Cross-Entropy Loss)", fontsize=11, fontweight="bold")
ax_l.grid(True, linestyle=":", alpha=0.6)
ax_l.legend(loc="upper right", framealpha=0.9, fontsize=9.5)

# 子图 2：动作拟合精度收敛
ax_a.plot(all_epochs, exp033_train_acc, linestyle="--", color="#60a5fa", alpha=0.8, label="Micro 训练集精度")
ax_a.plot(all_epochs, exp033_val_acc, linestyle="-", marker="s", markersize=4, color="#1e40af", linewidth=2.0, label="Micro 验证集精度")

for step in stage_transition_epochs:
    ax_a.axvline(step + 0.5, color="#cbd5e1", linestyle=":", linewidth=1.2)

best_acc_idx = int(np.argmax(exp033_val_acc))
best_acc_val = exp033_val_acc[best_acc_idx]
best_acc_ep = all_epochs[best_acc_idx]

ax_a.annotate(f"最优验证精度: {best_acc_val:.2f}%\n(轮次 {best_acc_ep}, 迭代 8)",
              xy=(best_acc_ep, best_acc_val), xytext=(best_acc_ep - 12, best_acc_val - 6),
              arrowprops=dict(facecolor="#1e40af", shrink=0.08, width=1.4, headwidth=5),
              fontsize=9, fontweight="bold", color="#1e40af",
              bbox=dict(boxstyle="round,pad=0.3", facecolor="#eff6ff", edgecolor="#93c5fd"))

ax_a.set_title("EXP-033 策略网络动作拟合精度演化 (Accuracy %)", fontsize=12.5, fontweight="bold", pad=12)
ax_a.set_xlabel("累计训练轮次 (Cumulative Epochs across k1~k8)", fontsize=11, fontweight="bold")
ax_a.set_ylabel("动作拟合精度 (%)", fontsize=11, fontweight="bold")
ax_a.grid(True, linestyle=":", alpha=0.6)
ax_a.legend(loc="lower right", framealpha=0.9, fontsize=9.5)

fig.suptitle("EXP-033 (Micro 28 通道) DAgger 交互聚合训练全周期收敛曲线 (迭代 1 至 8)", fontsize=14.5, fontweight="bold", y=0.98)
fig.tight_layout(rect=[0, 0.03, 1, 0.95])
fig.savefig(os.path.join(output_dir, "exp033_training_convergence.png"), bbox_inches="tight")
plt.close(fig)

print("ALL_PLOTS_GENERATED_SUCCESSFULLY")
