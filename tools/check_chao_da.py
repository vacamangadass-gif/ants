from pathlib import Path
import re

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DOCS = [
    PROJECT_ROOT / "tinybrain" / "04 实验日志" / "EXP-085_COMPREHENSIVE_TECHNICAL_REPORT.md",
    PROJECT_ROOT / "tinybrain" / "04 实验日志" / "EXP-085.md",
    PROJECT_ROOT / "ants-dev" / "05 技术复盘 Review" / "深度复盘_EXP-085_全谱系多席位对抗与协同机制改进.md",
    PROJECT_ROOT / "tinybrain" / "03 Idea 构想拓扑" / "IDEA-008_群体协调与规模效应" / "IDEA-008_核心提案说明.md",
]

for doc in DOCS:
    content = doc.read_text(encoding="utf-8")
    lines = content.splitlines()
    for idx, line in enumerate(lines, 1):
        # 寻找 "达" 或 "超" 紧跟数字的情况
        matches = re.finditer(r"(.)([达超])\s*(\d+)", line)
        for m in matches:
            prev_char = m.group(1)
            verb = m.group(2)
            num = m.group(3)
            # 如果前面不是 "高"、"多"、"抵"、"到"、"出" 等合理双字词前缀
            full_match = m.group(0)
            print(f"[{doc.name}:{idx}] {full_match} -> 整行: {line.strip()}")
