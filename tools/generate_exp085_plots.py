import json
from pathlib import Path
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

ROOT = Path(__file__).resolve().parents[1]
output_dir = ROOT / "tinybrain" / "ima"
output_dir.mkdir(parents=True, exist_ok=True)


def generate_multiseat_tournament_plot():
    multiseat_file = ROOT / "runs" / "m3_multiseat_evaluation_report.json"
    data = json.loads(multiseat_file.read_text(encoding="utf-8"))

    fig, axes = plt.subplots(1, 2, figsize=(16, 7), dpi=300)

    # 1. 各模型总体战绩对比（场均得分与排名）
    ax1 = axes[0]
    models = ["exp085", "exp021", "exp033", "exp011", "exp083"]
    model_labels = ["EXP-085\n(改进协同)", "EXP-021\n(强攻基线)", "EXP-033\n(天梯标杆)", "EXP-011\n(早期基线)", "EXP-083\n(前代模型)"]
    mean_scores = [data["models"][m]["mean_score"] for m in models]
    std_scores = [data["models"][m]["std_score"] for m in models]
    colors = ["#2b7bba", "#e67e22", "#27ae60", "#95a5a6", "#e74c3c"]

    bars = ax1.bar(model_labels, mean_scores, yerr=std_scores, capsize=6, color=colors, alpha=0.85, edgecolor="black", linewidth=1.2)
    ax1.set_ylabel("多席位锦标赛场均得分", fontsize=12, fontweight="bold")
    ax1.set_title("92 场全对称多席位锦标赛跨模型场均得分对比 (516 席位)", fontsize=13, fontweight="bold", pad=12)
    ax1.grid(axis="y", linestyle="--", alpha=0.5)

    for bar, score in zip(bars, mean_scores):
        height = bar.get_height()
        ax1.annotate(
            f"{score:.2f} 分",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )

    # 2. 基础棋盘分席位得分热力对比
    ax2 = axes[1]
    boards = ["2p (24x24)", "3p (36x36)", "4p (48x64)", "6p (80x96)", "8p (120x124)"]
    board_keys = ["2p", "3p", "4p", "6p", "8p"]
    x = np.arange(len(boards))
    width = 0.22

    score_085 = [data["per_board"][b].get("exp085", {}).get("mean_score", 0.0) for b in board_keys]
    score_033 = [data["per_board"][b].get("exp033", {}).get("mean_score", 0.0) for b in board_keys]
    score_083 = [data["per_board"][b].get("exp083", {}).get("mean_score", 0.0) for b in board_keys]

    b1 = ax2.bar(x - width, score_085, width, label="EXP-085 (改进模型)", color="#2b7bba", edgecolor="black", alpha=0.9)
    b2 = ax2.bar(x, score_033, width, label="EXP-033 (天梯标杆)", color="#27ae60", edgecolor="black", alpha=0.9)
    b3 = ax2.bar(x + width, score_083, width, label="EXP-083 (前代模型)", color="#e74c3c", edgecolor="black", alpha=0.9)

    ax2.set_xlabel("官方基础棋盘规格与席位数", fontsize=12, fontweight="bold")
    ax2.set_ylabel("场均得分", fontsize=12, fontweight="bold")
    ax2.set_title("各规格基础棋盘分项对抗表现对比", fontsize=13, fontweight="bold", pad=12)
    ax2.set_xticks(x)
    ax2.set_xticklabels(boards, fontsize=10)
    ax2.legend(frameon=True, fontsize=10)
    ax2.grid(axis="y", linestyle="--", alpha=0.5)

    # 在 8p 上突出显示巨大优势
    ax2.annotate(
        "8 席位超大棋盘\nEXP-085 取得 4.81 分 (领先 4.04 倍)",
        xy=(4 - width, score_085[4]),
        xytext=(3.2, 5.2),
        arrowprops=dict(facecolor="#2b7bba", shrink=0.08, width=1.5, headwidth=6),
        fontsize=10,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#e8f4f8", edgecolor="#2b7bba"),
    )

    plt.tight_layout()
    out_file = output_dir / "exp085_multiseat_tournament.png"
    plt.savefig(out_file, dpi=300)
    plt.close()
    print(f"生成多席位锦标赛图表成功: {out_file}")


def generate_h2h_comparison_plot():
    h2h_file = ROOT / "runs" / "m3_h2h_exp085_vs_exp033" / "evaluation_report.json"
    h2h_data = json.loads(h2h_file.read_text(encoding="utf-8"))

    fig, axes = plt.subplots(1, 2, figsize=(16, 7), dpi=300)

    # 1. 180 场对局胜负与得分分布饼图 / 条形图
    ax1 = axes[0]
    outcomes = ["EXP-085 获胜 (140 场)", "双方战平 (6 场)", "EXP-033 获胜 (34 场)"]
    counts = [140, 6, 34]
    pie_colors = ["#2b7bba", "#f39c12", "#e74c3c"]
    wedges, texts, autotexts = ax1.pie(
        counts,
        labels=outcomes,
        autopct="%1.1f%%",
        startangle=140,
        colors=pie_colors,
        explode=(0.05, 0.0, 0.0),
        textprops=dict(fontsize=11, fontweight="bold"),
        wedgeprops=dict(edgecolor="black", linewidth=1.2),
    )
    for at in autotexts:
        at.set_color("white")
        at.set_fontsize(12)
    ax1.set_title("EXP-085 对决 EXP-033 双人基准循环赛战果 (180 场)", fontsize=13, fontweight="bold", pad=12)

    # 2. 9 张基准地图分项得分率对比
    ax2 = axes[1]
    map_breakdown = h2h_data["map_breakdown"]
    map_names = list(map_breakdown.keys())
    short_names = [m.replace("-2p-1h", "") for m in map_names]
    exp085_win_rates = [
        (map_breakdown[m]["exp085_wins"] / map_breakdown[m]["matches"]) * 100
        for m in map_names
    ]
    exp085_score_rates = [map_breakdown[m]["exp085_score_rate"] * 100 for m in map_names]

    y = np.arange(len(map_names))
    height = 0.35

    ax2.barh(y - height / 2, exp085_win_rates, height, label="EXP-085 胜率 (%)", color="#2b7bba", edgecolor="black", alpha=0.9)
    ax2.barh(y + height / 2, exp085_score_rates, height, label="EXP-085 得分率 (%)", color="#3498db", edgecolor="black", alpha=0.7)

    ax2.axvline(70.0, color="#e74c3c", linestyle="--", linewidth=1.5, label="验收基准线 (70.0%)")
    ax2.set_xlabel("百分比 (%)", fontsize=12, fontweight="bold")
    ax2.set_title("9 张官方双人基准地图细分胜率与得分率", fontsize=13, fontweight="bold", pad=12)
    ax2.set_yticks(y)
    ax2.set_yticklabels(short_names, fontsize=10)
    ax2.legend(loc="lower left", frameon=True, fontsize=10)
    ax2.grid(axis="x", linestyle="--", alpha=0.5)

    plt.tight_layout()
    out_file = output_dir / "exp085_h2h_comparison.png"
    plt.savefig(out_file, dpi=300)
    plt.close()
    print(f"生成双人对抗对比图表成功: {out_file}")


if __name__ == "__main__":
    generate_multiseat_tournament_plot()
    generate_h2h_comparison_plot()
