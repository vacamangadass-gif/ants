from pathlib import Path
import re
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DOCS = [
    PROJECT_ROOT / "tinybrain" / "04 实验日志" / "EXP-085_COMPREHENSIVE_TECHNICAL_REPORT.md",
    PROJECT_ROOT / "tinybrain" / "04 实验日志" / "EXP-085.md",
    PROJECT_ROOT / "ants-dev" / "05 技术复盘 Review" / "深度复盘_EXP-085_全谱系多席位对抗与协同机制改进.md",
    PROJECT_ROOT / "tinybrain" / "03 Idea 构想拓扑" / "IDEA-008_群体协调与规模效应" / "IDEA-008_核心提案说明.md",
]

# 1. 检验所有待核验文档存在性
for doc_path in DOCS:
    assert doc_path.is_file(), f"文档不存在: {doc_path}"

# 2. 对抗性扫描模式定义
FORBIDDEN_PATTERNS = [
    (r"不是[^，。\n]+而是", "禁用句式：不是...而是..."),
    (r"要[^，。\n]+而不是", "禁用句式：要...而不是..."),
    (r"而不是", "禁用对比句式：而不是"),
    (r"而非", "禁用对比句式：而非"),
    (r"并非[^，。\n]+而是", "禁用句式：并非...而是..."),
    (r"不是[^，。\n]+是", "禁用句式：不是...是..."),
    (r"落地", "禁用互联网黑话：落地"),
    (r"钉死", "禁用互联网黑话：钉死"),
    (r"对齐", "禁用互联网黑话：对齐"),
    (r"栈", "禁用包含栈字的词汇：栈"),
    (r"两字版本", "禁用缩写表达：两字版本"),
    (r"单字版本", "禁用缩写表达：单字版本"),
    (r"一句话总结", "禁用套话：一句话总结"),
    (r"上述内容是", "禁用套话：上述内容是"),
    (r"总的来说", "禁用套话：总的来说"),
    (r"综上所述", "禁用套话：综上所述"),
]

SINGLE_CHAR_VERB_PATTERNS = [
    (r"(?<![\u4e00-\u9fa5a-zA-Z0-9])崩(?![\u4e00-\u9fa5a-zA-Z0-9])", "单字缩写：崩"),
    (r"(?<![\u4e00-\u9fa5a-zA-Z0-9])挂(?![\u4e00-\u9fa5a-zA-Z0-9])", "单字缩写：挂"),
    (r"(?<![\u4e00-\u9fa5a-zA-Z0-9])死(?![\u4e00-\u9fa5a-zA-Z0-9])", "单字缩写：死"),
    (r"(?<![\u4e00-\u9fa5a-zA-Z0-9])判(?![\u4e00-\u9fa5a-zA-Z0-9])", "单字缩写：判"),
    (r"(?<![\u4e00-\u9fa5a-zA-Z0-9])推(?![\u4e00-\u9fa5a-zA-Z0-9])", "单字缩写：推"),
    (r"(?<![\u4e00-\u9fa5a-zA-Z0-9])抛(?![\u4e00-\u9fa5a-zA-Z0-9])", "单字缩写：抛"),
]

ASCII_ART_PATTERNS = [
    (r"\+[-=]+\+", "ASCII 流程图/边框: +---+"),
    (r"\| {3,}\|", "ASCII 流程图内部列: |   |"),
    (r"\+---\+", "ASCII 边框"),
]

# 3. 逐个文件执行扫描
for doc_path in DOCS:
    content = doc_path.read_text(encoding="utf-8")
    lines = content.splitlines()

    # 扫描禁用模式
    for pat, desc in FORBIDDEN_PATTERNS:
        matches = list(re.finditer(pat, content))
        if len(matches) > 0:
            for m in matches:
                start = max(0, m.start() - 30)
                end = min(len(content), m.end() + 30)
                print(f"[FAIL] {doc_path.name}: 命中 {desc} -> ...{content[start:end]}...")
            assert False, f"{doc_path.name} 命中 {desc}"

    # 扫描单字动词
    for pat, desc in SINGLE_CHAR_VERB_PATTERNS:
        matches = list(re.finditer(pat, content))
        if len(matches) > 0:
            for m in matches:
                start = max(0, m.start() - 30)
                end = min(len(content), m.end() + 30)
                print(f"[FAIL] {doc_path.name}: 命中 {desc} -> ...{content[start:end]}...")
            assert False, f"{doc_path.name} 命中 {desc}"

    # 扫描 ASCII Art (排除代码块和 mermaid)
    in_code_block = False
    for line_idx, line in enumerate(lines, 1):
        stripped = line.strip()
        if stripped.startswith("```"):
            in_code_block = not in_code_block
            continue
        if in_code_block:
            continue
        for pat, desc in ASCII_ART_PATTERNS:
            if re.search(pat, line):
                print(f"[FAIL] {doc_path.name}:{line_idx}: 命中 {desc} -> {line}")
                assert False, f"{doc_path.name}:{line_idx} 命中 {desc}"

    # 扫描图片与图注规范
    img_matches = list(re.finditer(r"!\[(.*?)\]\((.*?)\)", content))
    for m in img_matches:
        alt_text = m.group(1).strip()
        rel_path = m.group(2).strip()
        assert len(alt_text) > 0, f"{doc_path.name} 图片缺少 alt 文本: {m.group(0)}"
        img_abs = (doc_path.parent / rel_path).resolve()
        assert img_abs.is_file(), f"{doc_path.name} 图片链接无效: {rel_path} -> {img_abs}"
        
        # 验证物理文件属性
        size = img_abs.stat().st_size
        assert size > 50000, f"{img_abs.name} 体积小于 50KB: {size} 字节"
        with Image.open(img_abs) as im:
            assert im.format == "PNG", f"{img_abs.name} 格式非 PNG: {im.format}"
            dpi = im.info.get("dpi")
            assert dpi is not None and abs(dpi[0] - 300) < 1.0, f"{img_abs.name} DPI 非 300: {dpi}"
            w, h = im.size
            assert w >= 1000 and h >= 600, f"{img_abs.name} 尺寸异常: {w}x{h}"

        # 验证斜体图注
        end_idx = m.end()
        after_text = content[end_idx:end_idx + 600].strip()
        assert after_text.startswith("*") and "*" in after_text[1:], f"{doc_path.name} 图片后缺少斜体图注: {m.group(0)}"

    print(f"全量规则核验通过: {doc_path.name}")

print("4 份新生成/更新核心文档红线规则全量扫描全部通过！")
