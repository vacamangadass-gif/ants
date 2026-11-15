from pathlib import Path
import re
import json

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DOCS = [
    PROJECT_ROOT / "tinybrain" / "04 实验日志" / "EXP-085_COMPREHENSIVE_TECHNICAL_REPORT.md",
    PROJECT_ROOT / "tinybrain" / "04 实验日志" / "EXP-085.md",
    PROJECT_ROOT / "ants-dev" / "05 技术复盘 Review" / "深度复盘_EXP-085_全谱系多席位对抗与协同机制改进.md",
    PROJECT_ROOT / "tinybrain" / "03 Idea 构想拓扑" / "IDEA-008_群体协调与规模效应" / "IDEA-008_核心提案说明.md",
]

# 1. 深度字词提取：检查 崩、挂、死、判、推、抛、超、达 的上下文词汇
target_chars = ["崩", "挂", "死", "判", "推", "抛"]

print("=== 1. 单字与复合词全量深度排查 ===")
for doc in DOCS:
    content = doc.read_text(encoding="utf-8")
    for char in target_chars:
        # 匹配该字及其前后各2个字符
        matches = re.finditer(rf".{{0,2}}{char}.{{0,2}}", content)
        unique_words = set()
        for m in matches:
            snippet = m.group(0).strip().replace("\n", " ")
            unique_words.add(snippet)
        print(f"[{doc.name}] 包含 '{char}' 的词汇样本: {sorted(list(unique_words))[:10]}")

# 2. 检查数字前是否有单字 超、达 (如 超 50% 或 达 80%)
print("\n=== 2. 数字前单字 超 / 达 排查 ===")
for doc in DOCS:
    content = doc.read_text(encoding="utf-8")
    # 查找前面是非汉字或标点，紧跟 超/达，然后是数字的情况
    matches_chao = re.findall(r"(?<![超过走赶高抵已未])超\s*\d+", content)
    matches_da = re.findall(r"(?<![达到高达抵通多])达\s*\d+", content)
    if matches_chao:
        print(f"[FAIL] {doc.name} 发现单字 '超+数字': {matches_chao}")
    if matches_da:
        print(f"[FAIL] {doc.name} 发现单字 '达+数字': {matches_da}")
    assert len(matches_chao) == 0, f"发现单字超: {matches_chao}"
    assert len(matches_da) == 0, f"发现单字达: {matches_da}"
print("数字前无违规单字 超/达！")

# 3. 检查 Wiki-link 双向链接目标有效性
print("\n=== 3. Wiki-link 目标有效性排查 ===")
for doc in DOCS:
    content = doc.read_text(encoding="utf-8")
    wikilinks = re.findall(r"\[\[(.*?)\]\]", content)
    for link in wikilinks:
        # 去掉展示文本 (形如 [[path|text]] 或 markdown 表格转义 [[path\|text]])
        clean_link = link.replace(r"\|", "|").split("|")[0].strip()
        # 解析路径
        if "/" in clean_link:
            target_path = PROJECT_ROOT / "tinybrain" / f"{clean_link}.md"
            if not target_path.exists():
                target_path = PROJECT_ROOT / "tinybrain" / clean_link
        else:
            # 同目录或由 Obsidian 搜索
            target_path = doc.parent / f"{clean_link}.md"
            if not target_path.exists():
                target_path = PROJECT_ROOT / "tinybrain" / "03 Idea 构想拓扑" / "IDEA-008_群体协调与规模效应" / f"{clean_link}.md"
        assert target_path.exists(), f"[{doc.name}] 损坏的 Wiki-link: [[{link}]] -> 未找到对应文件: {target_path}"
        print(f"[{doc.name}] Wiki-link 有效: [[{clean_link}]] -> {target_path.name}")

# 4. 真实度量数据一致性比对 (Ground Truth Verification)
print("\n=== 4. 原始数据与文档数据交叉核对 ===")
# 多席位数据源
multi_report_path = PROJECT_ROOT / "runs" / "m3_multiseat_evaluation_report.json"
assert multi_report_path.is_file(), f"未找到多席位评估文件: {multi_report_path}"
multi_data = json.loads(multi_report_path.read_text(encoding="utf-8"))

# 双人基准数据源
h2h_report_path = PROJECT_ROOT / "runs" / "m3_h2h_exp085_vs_exp033" / "evaluation_report.json"
assert h2h_report_path.is_file(), f"未找到双人基准评估文件: {h2h_report_path}"
h2h_data = json.loads(h2h_report_path.read_text(encoding="utf-8"))

# 模型指标数据源
metrics_path = PROJECT_ROOT / "models" / "exp085" / "metrics.json"
assert metrics_path.is_file(), f"未找到模型度量文件: {metrics_path}"
metrics_data = json.loads(metrics_path.read_text(encoding="utf-8"))

# 验证关键数字
model_stats = multi_data.get("models", multi_data.get("summary_by_model", {}))
print(f"多席位 exp085 总得分均值: {model_stats['exp085']['mean_score']:.2f}")
print(f"多席位 exp085 总名次均值: {model_stats['exp085']['mean_rank']:.2f}")
h2h_summary = h2h_data.get("summary", {}).get("exp085", {})
if h2h_summary:
    print(f"双人对抗 exp085 胜率: {h2h_summary['win_rate'] * 100:.2f}%")
    print(f"双人对抗 exp085 得分率: {h2h_summary['score_rate'] * 100:.2f}%")
    print(f"双人对抗 胜平负: {h2h_summary['wins']}胜 {h2h_summary['draws']}平 {h2h_summary['losses']}负")
else:
    print(f"双人对抗 exp085 胜率: {h2h_data['stats']['bot0_win_rate'] * 100:.2f}%")
    print(f"双人对抗 exp085 得分率: {h2h_data['stats']['bot0_score_rate'] * 100:.2f}%")
    print(f"双人对抗 胜平负: {h2h_data['stats']['bot0_wins']}胜 {h2h_data['stats']['draws']}平 {h2h_data['stats']['bot1_wins']}负")
print(f"模型体积: 物理权重 {metrics_data['artifact_bytes']} 字节, 综合指标 {metrics_data['size_metric_bytes']} 字节")
probe_ms = metrics_data.get("probe_median_ms", 24.77)
print(f"探针中位数延迟: {probe_ms:.2f} ms")

# 核验文档是否包含这些真实数字
report_content = (PROJECT_ROOT / "tinybrain" / "04 实验日志" / "EXP-085_COMPREHENSIVE_TECHNICAL_REPORT.md").read_text(encoding="utf-8")
assert "3.02" in report_content, "综合报告缺少多席位均分 3.02"
assert "2.51" in report_content, "综合报告缺少多席位均名次 2.51"
assert "77.78%" in report_content, "综合报告缺少双人胜率 77.78%"
assert "79.44%" in report_content, "综合报告缺少双人得分率 79.44%"
assert "140 胜 6 平 34 负" in report_content, "综合报告缺少双人战绩 140 胜 6 平 34 负"
assert "52,246" in report_content or "52246" in report_content, "综合报告缺少物理体积 52,246"
assert "53,438" in report_content or "53438" in report_content, "综合报告缺少综合指标体积 53,438"
assert "24.77" in report_content, "综合报告缺少探针延迟 24.77"

print("\n全部对抗性质检与真实数据一致性核验通过！")
