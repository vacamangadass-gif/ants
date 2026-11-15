from pathlib import Path
import re

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DOCS = [
    PROJECT_ROOT / "tinybrain" / "04 实验日志" / "EXP-085_COMPREHENSIVE_TECHNICAL_REPORT.md",
    PROJECT_ROOT / "tinybrain" / "04 实验日志" / "EXP-085.md",
    PROJECT_ROOT / "ants-dev" / "05 技术复盘 Review" / "深度复盘_EXP-085_全谱系多席位对抗与协同机制改进.md",
    PROJECT_ROOT / "tinybrain" / "03 Idea 构想拓扑" / "IDEA-008_群体协调与规模效应" / "IDEA-008_核心提案说明.md",
]

# 检查黑话、俗语、缩写
SUSPICIOUS = [
    "写死",
    "秒破",
    "送死",
    "挂零",
    "破零",
    "抓手",
    "赋能",
    "背书",
    "拉齐",
    "闭环",
    "打法",
    "壁垒",
    "心智",
    "解耦",  # 解耦是专业技术词汇
    "颗粒度",
    "链路",
]

for doc in DOCS:
    content = doc.read_text(encoding="utf-8")
    lines = content.splitlines()
    for idx, line in enumerate(lines, 1):
        for word in SUSPICIOUS:
            if word in line:
                print(f"[{doc.name}:{idx}] 出现 '{word}': {line.strip()[:100]}")
