import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch, Rectangle

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = [
    "Noto Sans CJK SC",
    "Noto Sans CJK JP",
    "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False

output_dir = "/home/host/project/ants/tinybrain/ima"
os.makedirs(output_dir, exist_ok=True)

# -----------------------------------------------------------------------------
# 图表一：EXP-080 教师规模压力基准评测遥测分析图
# -----------------------------------------------------------------------------
exp080_summary_path = "/home/host/project/ants/runs/exp080_scaling/telemetry_summary.json"
exp080_jsonl_path = "/home/host/project/ants/runs/exp080_scaling/telemetry.jsonl"
assert os.path.isfile(exp080_summary_path), f"文件不存在: {exp080_summary_path}"
assert os.path.isfile(exp080_jsonl_path), f"文件不存在: {exp080_jsonl_path}"

exp080_records = []
with open(exp080_jsonl_path, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            exp080_records.append(json.loads(line))

assert len(exp080_records) == 135, f"EXP-080 记录数量异常: {len(exp080_records)}"

scales = [20, 40, 60, 80, 100]

# 子图 1：迷宫地形走廊受阻率与拥塞指数
fig, axes = plt.subplots(2, 2, figsize=(15, 11), dpi=300)

lab_records = [r for r in exp080_records if r["map_id"] == "medium-labyrinth-2p-1h"]
lab_by_scale = {s: [r for r in lab_records if r["scale"] == s] for s in scales}

mean_stalling = [np.mean([r["corridor_stalling_rate"] * 100 for r in lab_by_scale[s]]) for s in scales]
mean_congestion = [np.mean([r["corridor_congestion_index"] for r in lab_by_scale[s]]) for s in scales]

ax1 = axes[0, 0]
color_stalling = "#d9534f"
color_congestion = "#f0ad4e"

line1 = ax1.plot(scales, mean_stalling, marker="o", linewidth=2.5, color=color_stalling, label="走廊受阻率 (%)")
ax1.set_xlabel("己方蚂蚁规模 (只)", fontsize=11)
ax1.set_ylabel("走廊受阻率 (%)", fontsize=11, color=color_stalling)
ax1.tick_params(axis="y", labelcolor=color_stalling)
ax1.set_ylim(40, 100)
ax1.grid(True, linestyle="--", alpha=0.3)

for x, y in zip(scales, mean_stalling):
    ax1.annotate(f"{y:.1f}%", (x, y), textcoords="offset points", xytext=(0, 8),
                 ha="center", fontsize=9, fontweight="bold", color=color_stalling)

ax1_twin = ax1.twinx()
line2 = ax1_twin.plot(scales, mean_congestion, marker="s", linewidth=2.5, linestyle="--",
                      color=color_congestion, label="走廊拥塞指数")
ax1_twin.set_ylabel("走廊拥塞指数", fontsize=11, color=color_congestion)
ax1_twin.tick_params(axis="y", labelcolor=color_congestion)
ax1_twin.set_ylim(0.0, 0.65)

for x, y in zip(scales, mean_congestion):
    ax1_twin.annotate(f"{y:.4f}", (x, y), textcoords="offset points", xytext=(0, -14),
                      ha="center", fontsize=9, fontweight="bold", color=color_congestion)

lines = line1 + line2
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc="upper left", framealpha=0.9)
ax1.set_title("(a) 迷宫地形走廊受阻率与拥塞指数激增趋势", fontsize=12, fontweight="bold", pad=10)

# 子图 2：全地图各规模友方内部碰撞阻挡率分布
ax2 = axes[0, 1]
collision_by_scale = [[r["internal_collision_blocking_rate"] * 100 for r in exp080_records if r["scale"] == s] for s in scales]

box = ax2.boxplot(collision_by_scale, tick_labels=[f"{s}只" for s in scales], patch_artist=True,
                  boxprops=dict(facecolor="#5bc0de", color="#2a6496", alpha=0.7),
                  medianprops=dict(color="#d9534f", linewidth=2),
                  whiskerprops=dict(color="#2a6496", linestyle="--"),
                  capprops=dict(color="#2a6496"))

ax2.set_xlabel("己方蚂蚁规模 (只)", fontsize=11)
ax2.set_ylabel("内部碰撞阻挡率 (%)", fontsize=11)
ax2.set_ylim(35, 55)
ax2.grid(True, linestyle="--", alpha=0.3)
ax2.set_title("(b) 全地图 135 场全矩阵友方内部碰撞阻挡率分布", fontsize=12, fontweight="bold", pad=10)

overall_mean_collision = np.mean([r["internal_collision_blocking_rate"] * 100 for r in exp080_records])
ax2.axhline(overall_mean_collision, color="#d9534f", linestyle=":", linewidth=1.5,
            label=f"全矩阵均值 ({overall_mean_collision:.2f}%)")
ax2.legend(loc="upper right", framealpha=0.9)

# 子图 3：有效采食速率与单兵采食效率边际递减
ax3 = axes[1, 0]
select_maps = [
    ("medium-labyrinth-2p-1h", "迷宫 (labyrinth)", "#d9534f", "o"),
    ("tiny-colonnade-2p-1h", "列柱 (colonnade)", "#5cb85c", "s"),
    ("tiny-open-2p-1h", "开阔 (open)", "#0275d8", "^"),
    ("medium-crossroads-2p-1h", "十字路口 (crossroads)", "#f0ad4e", "D"),
]

for map_id, map_label, map_color, map_marker in select_maps:
    map_recs = [r for r in exp080_records if r["map_id"] == map_id]
    map_by_s = {s: [r for r in map_recs if r["scale"] == s] for s in scales}
    throughput = [np.mean([r["effective_gathering_throughput"] for r in map_by_s[s]]) for s in scales]
    ax3.plot(scales, throughput, marker=map_marker, linewidth=2, color=map_color, label=map_label)

ax3.set_xlabel("己方蚂蚁规模 (只)", fontsize=11)
ax3.set_ylabel("单兵采食吞吐量 (食物/百回合·只)", fontsize=11)
ax3.grid(True, linestyle="--", alpha=0.3)
ax3.legend(loc="upper left", framealpha=0.9, fontsize=9)
ax3.set_title("(c) 典型拓扑地形有效采食吞吐量演进曲线", fontsize=12, fontweight="bold", pad=10)

# 子图 4：单回合决策平均耗时与 P95 耗时伸缩表现
ax4 = axes[1, 1]
mean_lat_by_scale = [np.mean([r["mean_latency_ms"] for r in exp080_records if r["scale"] == s]) for s in scales]
p95_lat_by_scale = [np.mean([r["p95_latency_ms"] for r in exp080_records if r["scale"] == s]) for s in scales]

ax4.plot(scales, mean_lat_by_scale, marker="o", linewidth=2.5, color="#337ab7", label="平均单回合耗时")
ax4.plot(scales, p95_lat_by_scale, marker="s", linewidth=2.5, linestyle="--", color="#f0ad4e", label="P95 单回合耗时")
ax4.axhline(10.0, color="#d9534f", linestyle=":", linewidth=1.5, label="延迟告警线 (10.0 ms)")

for x, y in zip(scales, mean_lat_by_scale):
    ax4.annotate(f"{y:.2f}ms", (x, y), textcoords="offset points", xytext=(0, 7),
                 ha="center", fontsize=9, color="#337ab7")

for x, y in zip(scales, p95_lat_by_scale):
    ax4.annotate(f"{y:.2f}ms", (x, y), textcoords="offset points", xytext=(0, 7),
                 ha="center", fontsize=9, color="#f0ad4e")

ax4.set_xlabel("己方蚂蚁规模 (只)", fontsize=11)
ax4.set_ylabel("决策延迟 (毫秒)", fontsize=11)
ax4.set_ylim(1.0, 11.5)
ax4.grid(True, linestyle="--", alpha=0.3)
ax4.legend(loc="upper left", framealpha=0.9)
ax4.set_title("(d) 决策延迟伸缩表现 (100 蚂蚁规模内保持线性可控)", fontsize=12, fontweight="bold", pad=10)

plt.tight_layout()
fig.savefig(os.path.join(output_dir, "exp080_scaling_telemetry.png"), dpi=300)
plt.close(fig)
print("已成功生成: exp080_scaling_telemetry.png")

# -----------------------------------------------------------------------------
# 图表二：EXP-081 决策延迟拆解与性能门禁对比图
# -----------------------------------------------------------------------------
scenarios = [
    "基准评测\n(64x64/100蚁)",
    "纯网络流求解\n(100蚁/120位)",
    "端到端决策\n(对抗压力测试)",
    "多轮迭代推演\n(50轮综合统计)",
    "超大地图推演\n(80x80/50轮)",
    "持续多轮推演\n(5种子/125回合)",
    "单环境长程\n(100回合连续)",
    "高并发负载\n(8进程竞争环境)",
]

mean_latencies = [3.34, 2.48, 3.42, 3.40, 4.01, 2.96, 4.68, 5.22]
p95_latencies = [4.72, 2.63, 4.87, 4.35, 5.57, 4.23, 5.17, 6.82]

fig, (ax_lat, ax_decomp) = plt.subplots(1, 2, figsize=(15, 6), dpi=300)

x_pos = np.arange(len(scenarios))
bar_width = 0.35

rects1 = ax_lat.bar(x_pos - bar_width/2, mean_latencies, bar_width, label="平均单回合耗时",
                    color="#337ab7", edgecolor="#23527c", alpha=0.85)
rects2 = ax_lat.bar(x_pos + bar_width/2, p95_latencies, bar_width, label="P95 单回合耗时",
                    color="#f0ad4e", edgecolor="#d58512", alpha=0.85)

ax_lat.axhline(15.0, color="#d9534f", linestyle="--", linewidth=1.8, label="硬性门禁上限 (15.0 ms)")
ax_lat.axhline(10.0, color="#5cb85c", linestyle=":", linewidth=1.8, label="优化目标上限 (10.0 ms)")

ax_lat.set_xticks(x_pos)
ax_lat.set_xticklabels(scenarios, fontsize=9)
ax_lat.set_ylabel("决策延迟 (毫秒)", fontsize=11)
ax_lat.set_ylim(0, 17.5)
ax_lat.grid(True, linestyle="--", alpha=0.3, axis="y")
ax_lat.legend(loc="upper right", framealpha=0.9, fontsize=9)
ax_lat.set_title("(a) 多场景推演决策延迟与门禁标准判定对比", fontsize=12, fontweight="bold", pad=10)

for rect in rects1:
    h = rect.get_height()
    ax_lat.annotate(f"{h:.2f}", xy=(rect.get_x() + rect.get_width()/2, h),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8)

for rect in rects2:
    h = rect.get_height()
    ax_lat.annotate(f"{h:.2f}", xy=(rect.get_x() + rect.get_width()/2, h),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8, fontweight="bold")

# 子图：算法模块耗时拆解与迭代演化对比
stages = [
    "迭代1与2阶段\n(单阶段全图测地场)",
    "迭代3前中期\n(两阶段位图并行)",
    "迭代3交付状态\n(当前弧推进与剪枝)",
    "迭代4最终状态\n(环面曼哈顿距离预筛)",
]
stage_latencies = [18.38, 4.61, 3.40, 2.96]

bar_stages = ax_decomp.bar(stages, stage_latencies, width=0.5, color=["#d9534f", "#f0ad4e", "#5bc0de", "#5cb85c"],
                           edgecolor="#333333", alpha=0.85)

ax_decomp.axhline(15.0, color="#d9534f", linestyle="--", linewidth=1.8, label="硬性门禁上限 (15.0 ms)")
ax_decomp.axhline(10.0, color="#5cb85c", linestyle=":", linewidth=1.8, label="优化目标上限 (10.0 ms)")

for rect in bar_stages:
    h = rect.get_height()
    ax_decomp.annotate(f"{h:.2f} ms", xy=(rect.get_x() + rect.get_width()/2, h),
                       xytext=(0, 4), textcoords="offset points", ha="center", va="bottom",
                       fontsize=10, fontweight="bold")

# 插入内部拆解文字框
breakdown_text = (
    "迭代4单回合耗时拆解 (基准 3.34 ms):\n"
    "• 位图标量费用计算: 1.82 ms (54.5%)\n"
    "• Dial 桶队列 SSP 求解: 0.65 ms (19.5%)\n"
    "• 胜出采集位按需场广播: 0.87 ms (26.0%)\n"
    "• 整体延迟削减率: -83.9% (对比 18.38 ms)"
)
ax_decomp.text(0.52, 0.68, breakdown_text, transform=ax_decomp.transAxes,
               fontsize=9, verticalalignment="top",
               bbox=dict(boxstyle="round,pad=0.5", facecolor="#f9f9f9", edgecolor="#cccccc", alpha=0.9))

ax_decomp.set_ylabel("80x80 / 连续推演单回合耗时 (毫秒)", fontsize=11)
ax_decomp.set_ylim(0, 22)
ax_decomp.grid(True, linestyle="--", alpha=0.3, axis="y")
ax_decomp.legend(loc="upper right", framealpha=0.9, fontsize=9)
ax_decomp.set_title("(b) 核心算法优化演进路径与耗时压降幅度", fontsize=12, fontweight="bold", pad=10)

plt.tight_layout()
fig.savefig(os.path.join(output_dir, "exp081_mcmf_latency_breakdown.png"), dpi=300)
plt.close(fig)
print("已成功生成: exp081_mcmf_latency_breakdown.png")

# -----------------------------------------------------------------------------
# 图表三：EXP-081 采食冗余削减与内部碰撞阻挡彻底归零对比图
# -----------------------------------------------------------------------------
fig, (ax_red, ax_col) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)

seeds = ["随机种子 1001", "随机种子 1002", "随机种子 2026"]
compound_redundancy = [284, 218, 225]
mcmf_redundancy = [2, 1, 4]
reductions = [99.30, 99.54, 98.22]

x_idx = np.arange(len(seeds))
width = 0.35

rects_c = ax_red.bar(x_idx - width/2, compound_redundancy, width, label="复合教师策略 (Compound)",
                     color="#d9534f", edgecolor="#ac2925", alpha=0.85)
rects_m = ax_red.bar(x_idx + width/2, mcmf_redundancy, width, label="MCMF 协调教师 (EXP-081)",
                     color="#5cb85c", edgecolor="#4cae4c", alpha=0.85)

for i, (rc, rm, red) in enumerate(zip(rects_c, rects_m, reductions)):
    ax_red.annotate(f"{rc.get_height()}", xy=(rc.get_x() + rc.get_width()/2, rc.get_height()),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9)
    ax_red.annotate(f"{rm.get_height()}", xy=(rm.get_x() + rm.get_width()/2, rm.get_height()),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax_red.annotate(f"削减率\n{red:.2f}%", xy=(x_idx[i], max(rc.get_height(), rm.get_height()) / 2),
                    ha="center", va="center", fontsize=10, fontweight="bold",
                    color="#2b542c", bbox=dict(boxstyle="square,pad=0.2", facecolor="#dff0d8", edgecolor="#d6e9c6", alpha=0.9))

ax_red.set_xticks(x_idx)
ax_red.set_xticklabels(seeds, fontsize=10)
ax_red.set_ylabel("采食重叠冲突次数 (次)", fontsize=11)
ax_red.set_ylim(0, 330)
ax_red.grid(True, linestyle="--", alpha=0.3, axis="y")
ax_red.legend(loc="upper right", framealpha=0.9)
ax_red.set_title("(a) 长程推演采食重叠冗余对比 (削减率超过 98%)", fontsize=12, fontweight="bold", pad=10)

# 子图：友方内部碰撞阻挡率对比
models = ["复合教师策略\n(无容量约束基线)", "MCMF 协调教师\n(EXP-081 容量约束)"]
collision_rates = [43.52, 0.00]
colors_col = ["#d9534f", "#5cb85c"]

bars_col = ax_col.bar(models, collision_rates, width=0.45, color=colors_col, edgecolor="#333333", alpha=0.85)
ax_col.set_ylabel("友方内部碰撞阻挡率 (%)", fontsize=11)
ax_col.set_ylim(0, 55)
ax_col.grid(True, linestyle="--", alpha=0.3, axis="y")
ax_col.set_title("(b) 友方内部碰撞与移动仲裁阻挡对比 (彻底消除)", fontsize=12, fontweight="bold", pad=10)

ax_col.annotate(f"基准均值: 43.52%\n(区间: 38.45% ~ 50.70%)", xy=(bars_col[0].get_x() + bars_col[0].get_width()/2, 43.52),
                xytext=(0, 5), textcoords="offset points", ha="center", va="bottom", fontsize=10, fontweight="bold", color="#d9534f")

ax_col.annotate(f"严格归零: 0.00%\n(0 次碰撞 / 0 次阻挡)", xy=(bars_col[1].get_x() + bars_col[1].get_width()/2, 0.00),
                xytext=(0, 15), textcoords="offset points", ha="center", va="bottom", fontsize=10, fontweight="bold",
                color="#5cb85c", bbox=dict(boxstyle="round,pad=0.4", facecolor="#dff0d8", edgecolor="#d6e9c6", alpha=0.9))

plt.tight_layout()
fig.savefig(os.path.join(output_dir, "exp081_redundancy_collision_reduction.png"), dpi=300)
plt.close(fig)
print("已成功生成: exp081_redundancy_collision_reduction.png")

# -----------------------------------------------------------------------------
# 图表四：EXP-082 Gate 3-1 经验条件熵与可学习性门禁验证图
# -----------------------------------------------------------------------------
exp082_summary_path = "/home/host/project/ants/runs/exp082_gate3_1/telemetry_summary.json"
exp082_jsonl_path = "/home/host/project/ants/runs/exp082_gate3_1/telemetry.jsonl"
assert os.path.isfile(exp082_summary_path), f"文件不存在: {exp082_summary_path}"
assert os.path.isfile(exp082_jsonl_path), f"文件不存在: {exp082_jsonl_path}"

with open(exp082_summary_path, "r", encoding="utf-8") as f:
    exp082_summary = json.load(f)

exp082_records = []
with open(exp082_jsonl_path, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            exp082_records.append(json.loads(line))

assert len(exp082_records) == 135, f"EXP-082 记录数量异常: {len(exp082_records)}"

fig, axes = plt.subplots(2, 2, figsize=(15, 11), dpi=300)

# 子图 1：门禁核心裁决对比
ax1 = axes[0, 0]
gate_labels = ["原始局部观测\n(31x31感受野切片)", "特征投影补偿\n(空间引导梯度)"]
gate_values = [
    exp082_summary["raw_evaluation"]["empirical_conditional_entropy"],
    exp082_summary["compensated_evaluation"]["empirical_conditional_entropy"],
]

bars_gate = ax1.bar(gate_labels, gate_values, width=0.45, color=["#337ab7", "#5cb85c"], edgecolor="#333333", alpha=0.85)
ax1.axhline(0.50, color="#d9534f", linestyle="--", linewidth=2.0, label="Gate 3-1 门禁阈值上限 (0.50 bit)")

for rect, val in zip(bars_gate, gate_values):
    ax1.annotate(f"{val:.4f} bit\n(PASS)", xy=(rect.get_x() + rect.get_width()/2, val),
                 xytext=(0, 6), textcoords="offset points", ha="center", va="bottom",
                 fontsize=11, fontweight="bold", color="#2e6da4" if val > 0.08 else "#398439")

# 净消减标注
ax1.annotate(f"条件熵净消减: -0.0501 bit\n(相对压降 -49.20%)", xy=(0.5, 0.075), xycoords="data",
             ha="center", va="center", fontsize=10, fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.4", facecolor="#fcf8e3", edgecolor="#faebcc", alpha=0.9))

ax1.set_ylabel("经验条件熵 H(A|C) (比特)", fontsize=11)
ax1.set_ylim(0, 0.60)
ax1.grid(True, linestyle="--", alpha=0.3, axis="y")
ax1.legend(loc="upper right", framealpha=0.9)
ax1.set_title("(a) Gate 3-1 经验条件熵核心门禁裁决对比", fontsize=12, fontweight="bold", pad=10)

# 子图 2：9 张基准地图条件熵与有效覆盖率
ax2 = axes[0, 1]
map_names_raw = sorted(list(set(r["map"] for r in exp082_records)))
map_short_names = [m.replace("-2p-1h", "") for m in map_names_raw]

map_entropies = [np.mean([r["empirical_entropy"] for r in exp082_records if r["map"] == m]) for m in map_names_raw]
map_coverages = [np.mean([r["coverage_ratio"] * 100 for r in exp082_records if r["map"] == m]) for m in map_names_raw]

x_maps = np.arange(len(map_names_raw))
bars_map = ax2.bar(x_maps, map_entropies, width=0.55, color="#5bc0de", edgecolor="#28a4c9", alpha=0.85, label="经验条件熵 (bit)")

ax2.set_xticks(x_maps)
ax2.set_xticklabels(map_short_names, rotation=35, ha="right", fontsize=9)
ax2.set_ylabel("经验条件熵均值 (bit)", fontsize=11, color="#28a4c9")
ax2.tick_params(axis="y", labelcolor="#28a4c9")
ax2.set_ylim(0, 0.045)
ax2.grid(True, linestyle="--", alpha=0.3)

ax2_twin = ax2.twinx()
line_cov = ax2_twin.plot(x_maps, map_coverages, marker="o", color="#f0ad4e", linewidth=2.2, label="样本有效覆盖率 (%)")
ax2_twin.axhline(70.0, color="#d9534f", linestyle=":", linewidth=1.5, label="覆盖率门禁下限 (70%)")
ax2_twin.set_ylabel("样本有效覆盖率 (%)", fontsize=11, color="#f0ad4e")
ax2_twin.tick_params(axis="y", labelcolor="#f0ad4e")
ax2_twin.set_ylim(65, 100)

for x, y in zip(x_maps, map_entropies):
    ax2.annotate(f"{y:.3f}", (x, y), textcoords="offset points", xytext=(0, 3), ha="center", fontsize=8)

ax2.set_title("(b) 9 张基准地图经验条件熵均值与有效样本覆盖率", fontsize=12, fontweight="bold", pad=10)

# 子图 3：各蚂蚁规模下的经验条件熵与单回合耗时
ax3 = axes[1, 0]
scale_entropies = [np.mean([r["empirical_entropy"] for r in exp082_records if r["scale"] == s]) for s in scales]
scale_latencies = [np.mean([r["avg_turn_latency_ms"] for r in exp082_records if r["scale"] == s]) for s in scales]

color_ent = "#337ab7"
color_lat = "#d9534f"

line_ent = ax3.plot(scales, scale_entropies, marker="o", linewidth=2.5, color=color_ent, label="经验条件熵 (bit)")
ax3.set_xlabel("蚂蚁阶梯规模 (只)", fontsize=11)
ax3.set_ylabel("经验条件熵 (bit)", fontsize=11, color=color_ent)
ax3.tick_params(axis="y", labelcolor=color_ent)
ax3.set_ylim(0, 0.04)
ax3.grid(True, linestyle="--", alpha=0.3)

for x, y in zip(scales, scale_entropies):
    ax3.annotate(f"{y:.4f}", (x, y), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=9, color=color_ent)

ax3_twin = ax3.twinx()
line_lat = ax3_twin.plot(scales, scale_latencies, marker="s", linewidth=2.5, linestyle="--", color=color_lat, label="单回合平均耗时 (ms)")
ax3_twin.set_ylabel("单回合平均耗时 (毫秒)", fontsize=11, color=color_lat)
ax3_twin.tick_params(axis="y", labelcolor=color_lat)
ax3_twin.set_ylim(0, 65)

for x, y in zip(scales, scale_latencies):
    ax3_twin.annotate(f"{y:.1f}ms", (x, y), textcoords="offset points", xytext=(0, -14), ha="center", fontsize=9, color=color_lat)

lines_comb = line_ent + line_lat
labels_comb = [l.get_label() for l in lines_comb]
ax3.legend(lines_comb, labels_comb, loc="upper left", framealpha=0.9)
ax3.set_title("(c) 20~100 规模阶梯下的经验条件熵与决策耗时演进", fontsize=12, fontweight="bold", pad=10)

# 子图 4：离散条件类别的样本频次覆盖与长尾分布
ax4 = axes[1, 1]
raw_total_samples = exp082_summary["total_aggregated_samples"]
raw_valid_samples = exp082_summary["raw_evaluation"]["verdict"]["telemetry_details"]["valid_samples"]
raw_tail_samples = raw_total_samples - raw_valid_samples

raw_valid_cats = exp082_summary["raw_evaluation"]["valid_categories"]
raw_tail_cats = exp082_summary["raw_evaluation"]["total_categories"] - raw_valid_cats

pie_labels = [
    f"有效充分支持类别\n({raw_valid_cats:,}类, {raw_valid_samples:,}样本\n覆盖率: 97.62%)",
    f"长尾偶发扰动类别\n({raw_tail_cats:,}类, {raw_tail_samples:,}样本\n覆盖率: 2.38%)",
]
pie_sizes = [raw_valid_samples, raw_tail_samples]
pie_colors = ["#5cb85c", "#f0ad4e"]

wedges, texts, autotexts = ax4.pie(pie_sizes, labels=pie_labels, autopct="%1.2f%%", startangle=140,
                                   colors=pie_colors, explode=(0, 0.1),
                                   wedgeprops=dict(edgecolor="#333333", linewidth=1.2))

for at in autotexts:
    at.set_fontsize(11)
    at.set_fontweight("bold")

ax4.set_title("(d) 810,000 条样本离散条件分类有效覆盖与长尾分布", fontsize=12, fontweight="bold", pad=10)

plt.tight_layout()
fig.savefig(os.path.join(output_dir, "exp082_gate3_1_entropy_distribution.png"), dpi=300)
plt.close(fig)
print("已成功生成: exp082_gate3_1_entropy_distribution.png")
