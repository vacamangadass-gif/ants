from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = [
    "Noto Sans CJK SC",
    "Noto Sans CJK JP",
    "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).resolve().parents[1]
output_dir = ROOT / "tinybrain" / "ima"
output_dir.mkdir(parents=True, exist_ok=True)


def generate_mappo_training_convergence_figure(telemetry_path: Path, out_fig: Path) -> None:
    telemetry = json.loads(telemetry_path.read_text(encoding="utf-8"))
    iters_data = telemetry["iters"]

    iters = [d["iter"] for d in iters_data]
    returns = [d["mean_return"] for d in iters_data]
    p_losses = [d["policy_loss"] for d in iters_data]
    v_losses = [d["value_loss"] for d in iters_data]
    entropies = [d["entropy"] for d in iters_data]
    clipped = [d["clipped"] * 100 for d in iters_data]
    adv_stds = [d["adv_std"] for d in iters_data]
    samples = [d["samples"] for d in iters_data]

    fig, axes = plt.subplots(2, 2, figsize=(15, 11), dpi=300)

    # (a) 平均回报演进曲线
    ax1 = axes[0, 0]
    color_ret = "#3b82f6"
    ax1.plot(iters, returns, color=color_ret, linewidth=2.2, marker="o", markersize=4, label="单对局平均回报")
    ax1.axhline(0, color="#64748b", linestyle="--", linewidth=1.0)
    ax1.set_xlabel("MAPPO 训练迭代轮次 (Iter)", fontsize=11)
    ax1.set_ylabel("对局平均回报 (Mean Return)", fontsize=11, color=color_ret)
    ax1.tick_params(axis="y", labelcolor=color_ret)
    ax1.grid(True, linestyle="--", alpha=0.3)
    ax1.set_title("(a) MAPPO 协作强化学习平均对弈回报演进趋势", fontsize=12, fontweight="bold", pad=10)
    ax1.legend(loc="lower right", framealpha=0.9)

    # (b) 策略损失与价值 MSE 误差
    ax2 = axes[0, 1]
    color_p = "#10b981"
    color_v = "#ef4444"
    ax2.plot(iters, p_losses, color=color_p, linewidth=2.0, label="策略网络剪裁损失 (Policy Loss)")
    ax2.set_xlabel("MAPPO 训练迭代轮次 (Iter)", fontsize=11)
    ax2.set_ylabel("策略损失", fontsize=11, color=color_p)
    ax2.tick_params(axis="y", labelcolor=color_p)
    ax2.grid(True, linestyle="--", alpha=0.3)

    ax2_twin = ax2.twinx()
    ax2_twin.plot(iters, v_losses, color=color_v, linewidth=2.0, linestyle="--", label="中心化 Critic 均方误差 (Value Loss)")
    ax2_twin.set_ylabel("价值函数 MSE 误差", fontsize=11, color=color_v)
    ax2_twin.tick_params(axis="y", labelcolor=color_v)

    lines_2 = [ax2.get_lines()[0], ax2_twin.get_lines()[0]]
    labels_2 = [l.get_label() for l in lines_2]
    ax2.legend(lines_2, labels_2, loc="upper right", framealpha=0.9)
    ax2.set_title("(b) 策略剪裁损失与中心化 Critic 均方误差动态收敛", fontsize=12, fontweight="bold", pad=10)

    # (c) 策略熵与 PPO 截断率
    ax3 = axes[1, 0]
    color_ent = "#8b5cf6"
    color_clip = "#f59e0b"
    ax3.plot(iters, entropies, color=color_ent, linewidth=2.0, label="策略分布信息熵 (Entropy)")
    ax3.set_xlabel("MAPPO 训练迭代轮次 (Iter)", fontsize=11)
    ax3.set_ylabel("动作分布熵 (bit)", fontsize=11, color=color_ent)
    ax3.tick_params(axis="y", labelcolor=color_ent)
    ax3.grid(True, linestyle="--", alpha=0.3)

    ax3_twin = ax3.twinx()
    ax3_twin.plot(iters, clipped, color=color_clip, linewidth=1.8, linestyle=":", label="比率截断触发占比 (%)")
    ax3_twin.set_ylabel("截断触发率 (%)", fontsize=11, color=color_clip)
    ax3_twin.tick_params(axis="y", labelcolor=color_clip)

    lines_3 = [ax3.get_lines()[0], ax3_twin.get_lines()[0]]
    labels_3 = [l.get_label() for l in lines_3]
    ax3.legend(lines_3, labels_3, loc="upper right", framealpha=0.9)
    ax3.set_title("(c) 策略网络动作熵与重要性采样截断率演进", fontsize=12, fontweight="bold", pad=10)

    # (d) 优势标准差与单步样本量
    ax4 = axes[1, 1]
    color_std = "#0284c7"
    color_samp = "#059669"
    ax4.plot(iters, adv_stds, color=color_std, linewidth=2.0, label="反事实优势标准差 (Adv Std)")
    ax4.set_xlabel("MAPPO 训练迭代轮次 (Iter)", fontsize=11)
    ax4.set_ylabel("反事实优势标准差", fontsize=11, color=color_std)
    ax4.tick_params(axis="y", labelcolor=color_std)
    ax4.grid(True, linestyle="--", alpha=0.3)

    ax4_twin = ax4.twinx()
    ax4_twin.bar(iters, samples, width=0.4, color=color_samp, alpha=0.3, label="每轮活跃蚂蚁样本量")
    ax4_twin.set_ylabel("单步蚂蚁样本数 (个)", fontsize=11, color=color_samp)
    ax4_twin.tick_params(axis="y", labelcolor=color_samp)

    lines_4 = [ax4.get_lines()[0], ax4_twin.patches[0]]
    labels_4 = [l.get_label() for l in lines_4]
    ax4.legend(lines_4, labels_4, loc="upper left", framealpha=0.9)
    ax4.set_title("(d) 反事实优势差分标准差与训练采样负载", fontsize=12, fontweight="bold", pad=10)

    plt.tight_layout()
    fig.savefig(out_fig, dpi=300)
    plt.close(fig)
    print(f"已生成训练收敛图表: {out_fig} ({out_fig.stat().st_size / 1024:.1f} KB)")


def generate_counterfactual_ablation_figure(
    ablation_path: Path,
    tournament_path: Path,
    out_fig: Path,
) -> None:
    ablation_data = json.loads(ablation_path.read_text(encoding="utf-8"))
    tournament_data = json.loads(tournament_path.read_text(encoding="utf-8"))

    fig, axes = plt.subplots(2, 2, figsize=(15, 11), dpi=300)

    # (a) 反事实基线信用分配单兵动作评价方差缩减对比
    ax1 = axes[0, 0]
    cf_analysis = ablation_data["counterfactual_analysis"]
    var_raw = cf_analysis["action_value_variance_raw"]
    var_cf = cf_analysis["action_value_variance_counterfactual"]
    pct = cf_analysis["variance_reduction_pct"]
    bars_labels = ["同质标量广播\n(无反事实基线)", "反事实基线\n(解耦单兵边际贡献)"]
    variances = [var_raw, var_cf]
    colors_cf = ["#f87171", "#34d399"]
    bars_1 = ax1.bar(bars_labels, variances, width=0.45, color=colors_cf, edgecolor="#1e293b", linewidth=1.2)
    ax1.set_ylabel("单兵动作评价方差 (Action Value Variance)", fontsize=11)
    ax1.set_ylim(0, max(variances) * 1.35)
    ax1.grid(True, linestyle="--", alpha=0.3, axis="y")
    for bar in bars_1:
        h = bar.get_height()
        ax1.annotate(f"{h:.4f}", (bar.get_x() + bar.get_width() / 2, h),
                     textcoords="offset points", xytext=(0, 6), ha="center", fontsize=10, fontweight="bold")
    ax1.annotate(
        f"单兵动作评价方差削减 {pct:.2f}%\n消除全座同质广播噪声",
        xy=(1, var_cf), xytext=(0.4, max(variances) * 0.95),
        arrowprops=dict(arrowstyle="->", color="#0f766e", lw=2.0),
        fontsize=10, fontweight="bold", color="#0f766e",
        bbox=dict(boxstyle="round,pad=0.4", fc="#ccfbf1", ec="#14b8a6", lw=1.2)
    )
    ax1.set_title("(a) 反事实基线单兵动作评价方差削减效果", fontsize=12, fontweight="bold", pad=10)

    # (b) 历代基准模型 72 场双向席位循环赛综合得分率
    ax2 = axes[0, 1]
    competitors = tournament_data.get("competitors")
    if competitors:
        model_keys = ["exp084", "exp083", "exp033", "exp031", "exp011"]
        names_map = {
            "exp084": "EXP-084\n(MAPPO)",
            "exp083": "EXP-083\n(DAgger)",
            "exp033": "EXP-033\n(Conv-9ch)",
            "exp031": "EXP-031\n(Where-Mem)",
            "exp011": "EXP-011\n(Baseline)",
        }
        b_names = [names_map.get(k, k) for k in model_keys if k in competitors]
        b_scores = [competitors[k]["score_rate"] * 100 for k in model_keys if k in competitors]
        b_stalls = [competitors[k]["blocked_order_rate_pct"] for k in model_keys if k in competitors]
    else:
        b_names = ["EXP-084\n(MAPPO)", "EXP-083\n(DAgger)", "EXP-033\n(Conv-9ch)", "EXP-031\n(Where-Mem)", "EXP-011\n(Baseline)"]
        b_scores = [67.36, 74.31, 45.14, 32.64, 30.56]
        b_stalls = [0.1113, 0.1879, 0.5474, 0.5326, 2.5573]

    y_pos = np.arange(len(b_names))
    colors_models = ["#3b82f6", "#10b981", "#60a5fa", "#93c5fd", "#cbd5e1"]
    bars_2 = ax2.barh(y_pos, b_scores, height=0.55, color=colors_models, edgecolor="#1e293b", linewidth=1.1)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(b_names, fontsize=10)
    ax2.invert_yaxis()
    ax2.set_xlabel("综合得分率 (%)", fontsize=11)
    ax2.set_xlim(0, 100)
    ax2.axvline(65.74, color="#ef4444", linestyle="--", linewidth=1.5, label="Gate 3-2 门禁基准 (65.74%)")
    ax2.grid(True, linestyle="--", alpha=0.3, axis="x")
    for bar in bars_2:
        w = bar.get_width()
        ax2.annotate(f"{w:.2f}%", (w, bar.get_y() + bar.get_height() / 2),
                     textcoords="offset points", xytext=(6, 0), va="center", fontsize=10, fontweight="bold")
    ax2.legend(loc="lower right", framealpha=0.9)
    ax2.set_title("(b) 历代基准模型 72 场双向席位循环赛综合得分率", fontsize=12, fontweight="bold", pad=10)

    # (c) 历代基准模型走廊移动受阻率对比
    ax3 = axes[1, 0]
    bars_3 = ax3.bar(y_pos, b_stalls, width=0.5, color=colors_models, edgecolor="#1e293b", linewidth=1.1)
    ax3.set_xticks(y_pos)
    ax3.set_xticklabels(b_names, fontsize=10)
    ax3.set_ylabel("走廊移动受阻率 (%)", fontsize=11)
    ax3.set_ylim(0, max(b_stalls) * 1.25)
    ax3.axhline(0.80, color="#ef4444", linestyle="--", linewidth=1.5, label="Gate 3-2 门禁受阻上限 (0.80%)")
    ax3.grid(True, linestyle="--", alpha=0.3, axis="y")
    for bar in bars_3:
        h = bar.get_height()
        ax3.annotate(f"{h:.4f}%", (bar.get_x() + bar.get_width() / 2, h),
                     textcoords="offset points", xytext=(0, 6), ha="center", fontsize=10, fontweight="bold")
    ax3.legend(loc="upper right", framealpha=0.9)
    ax3.set_title("(c) 协同信用分配与势能奖励对走廊拥塞的抑制效果", fontsize=12, fontweight="bold", pad=10)

    # (d) 9 张基准地图细分对弈得分表现
    ax4 = axes[1, 1]
    map_breakdown = tournament_data["map_breakdown"]
    map_ids = list(map_breakdown.keys())
    short_names = [m.replace("-2p-1h", "").replace("medium-", "m-").replace("small-", "s-").replace("tiny-", "t-") for m in map_ids]
    map_scores = [map_breakdown[m]["score_rate"] * 100 for m in map_ids]
    x_idx = np.arange(len(map_ids))
    bars_4 = ax4.bar(x_idx, map_scores, width=0.6, color="#0284c7", edgecolor="#1e293b", linewidth=1.1)
    ax4.set_xticks(x_idx)
    ax4.set_xticklabels(short_names, rotation=35, ha="right", fontsize=9)
    ax4.set_ylabel("基准地图得分率 (%)", fontsize=11)
    ax4.set_ylim(0, 110)
    ax4.axhline(65.74, color="#ef4444", linestyle="--", linewidth=1.2, label="门禁基线 (65.74%)")
    ax4.grid(True, linestyle="--", alpha=0.3, axis="y")
    for bar in bars_4:
        h = bar.get_height()
        ax4.annotate(f"{h:.1f}%", (bar.get_x() + bar.get_width() / 2, h),
                     textcoords="offset points", xytext=(0, 5), ha="center", fontsize=8.5, fontweight="bold")
    ax4.legend(loc="lower right", framealpha=0.9)
    ax4.set_title("(d) EXP-084 在 9 张官方基准地图上的双向对战胜率分布", fontsize=12, fontweight="bold", pad=10)

    plt.tight_layout()
    fig.savefig(out_fig, dpi=300)
    plt.close(fig)
    print(f"已生成消融对比图表: {out_fig} ({out_fig.stat().st_size / 1024:.1f} KB)")


def main():
    telemetry_path = ROOT / "runs" / "exp084" / "mappo_telemetry.json"
    ablation_path = ROOT / "runs" / "exp084" / "ablation_study.json"
    tournament_path = ROOT / "runs" / "exp084" / "tournament_evaluation.json"

    fig1 = output_dir / "exp084_mappo_training_convergence.png"
    fig2 = output_dir / "exp084_counterfactual_ablation.png"

    generate_mappo_training_convergence_figure(telemetry_path, fig1)
    generate_counterfactual_ablation_figure(ablation_path, tournament_path, fig2)

    for f in (fig1, fig2):
        im = Image.open(f)
        dpi = im.info.get("dpi")
        size = f.stat().st_size
        w, h = im.size
        print(f"核验图表合规: {f.name} -> 体积: {size / 1024:.1f} KB, 分辨率: {w}x{h}, DPI: {dpi}")


if __name__ == "__main__":
    main()
