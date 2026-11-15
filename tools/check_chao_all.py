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
        for m in re.finditer(r".{0,2}超.{0,2}", line):
            snippet = m.group(0).strip()
            # 排除 超大、超越、超过、超出、超距、超标、超额、超参数
            if not any(w in snippet for w in ["超大", "超越", "超过", "超出", "超距", "超标", "超额", "超参", "超级"]):
                print(f"[{doc.name}:{idx}] 待检 '超': '{snippet}' in: {line.strip()}")
