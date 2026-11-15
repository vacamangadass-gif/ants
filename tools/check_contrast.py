from pathlib import Path
import re

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DOCS = [
    PROJECT_ROOT / "tinybrain" / "04 实验日志" / "EXP-085_COMPREHENSIVE_TECHNICAL_REPORT.md",
    PROJECT_ROOT / "tinybrain" / "04 实验日志" / "EXP-085.md",
    PROJECT_ROOT / "ants-dev" / "05 技术复盘 Review" / "深度复盘_EXP-085_全谱系多席位对抗与协同机制改进.md",
    PROJECT_ROOT / "tinybrain" / "03 Idea 构想拓扑" / "IDEA-008_群体协调与规模效应" / "IDEA-008_核心提案说明.md",
]

CONTRAST_PATTERNS = [
    r"不是.{1,30}而是",
    r"要.{1,30}而不是",
    r"而不是",
    r"而非",
    r"并非.{1,30}而是",
    r"并不是",
    r"与其.{1,30}不如",
    r"不再.{1,30}而是",
]

for doc in DOCS:
    content = doc.read_text(encoding="utf-8")
    lines = content.splitlines()
    for idx, line in enumerate(lines, 1):
        for pat in CONTRAST_PATTERNS:
            m = re.search(pat, line)
            if m:
                print(f"[{doc.name}:{idx}] 匹配对比句式 '{pat}': {line.strip()}")
