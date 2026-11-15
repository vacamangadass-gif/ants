import os
import re
from pathlib import Path
from PIL import Image

PROJECT_ROOT = Path("/home/host/project/ants")
VAULT_ROOT = PROJECT_ROOT / "tinybrain"
PROPOSAL_DOC = VAULT_ROOT / "03 Idea 构想拓扑" / "IDEA-008_群体协调与规模效应" / "IDEA-008_核心提案说明.md"
LOGS_DIR = VAULT_ROOT / "04 实验日志"
IMA_DIR = VAULT_ROOT / "ima"

# 1. 检验双向链接完整性与目标文件有效性
assert PROPOSAL_DOC.is_file(), f"提案文档不存在: {PROPOSAL_DOC}"
proposal_content = PROPOSAL_DOC.read_text(encoding="utf-8")

exp_ids = ["EXP-080", "EXP-081", "EXP-082", "EXP-083", "EXP-084", "EXP-085"]
for eid in exp_ids:
    # 检验提案正文第四节与第六节中包含该实验的 wiki-link
    pat_section4 = rf"\[\[04 实验日志/{eid}\|{eid}\]\]"
    pat_section6 = rf"\[\[04 实验日志/{eid}\\|{eid}\]\]"
    
    has_sec4 = re.search(pat_section4, proposal_content) is not None
    has_sec6 = re.search(pat_section6, proposal_content) is not None
    assert has_sec4, f"提案正文缺少第四节对应实验链接: {eid}"
    assert has_sec6, f"提案第六节决策表格缺少对应实验链接: {eid}"
    
    # 检验目标实验日志文件存在
    log_file = LOGS_DIR / f"{eid}.md"
    assert log_file.is_file(), f"关联实验日志文件不存在: {log_file}"
    
    # 检验实验日志反向链接回到 IDEA-008 提案
    log_content = log_file.read_text(encoding="utf-8")
    backlink_meta = 'related_idea: "[[03 Idea 构想拓扑/IDEA-008_群体协调与规模效应/IDEA-008_核心提案说明]]"'
    backlink_sec1 = "[[IDEA-008_核心提案说明]]"
    assert backlink_meta in log_content, f"实验日志 {eid} 元数据缺少反向链接: {backlink_meta}"
    assert backlink_sec1 in log_content, f"实验日志 {eid} 第一节缺少反向链接: {backlink_sec1}"
    print(f"双向链接校验通过: {eid} <-> IDEA-008")

# 2. 检验 Markdown 图片链接与图片物理属性
img_pattern = r"!\[(.*?)\]\((.*?)\)"
img_matches = list(re.finditer(img_pattern, proposal_content))
assert len(img_matches) >= 8, f"提案文档图片引用数量不足 8 张: 仅发现 {len(img_matches)} 张"

expected_prefixes = ["exp080_", "exp081_", "exp082_", "exp083_", "exp084_", "exp085_"]
found_prefixes = set()

for m in img_matches:
    alt_text = m.group(1).strip()
    rel_path = m.group(2).strip()
    assert len(alt_text) > 0, f"图片链接缺少替代文本 (alt text): {m.group(0)}"
    
    # 检验路径解析
    target_img_path = (PROPOSAL_DOC.parent / rel_path).resolve()
    assert target_img_path.is_file(), f"图片文件不存在: {rel_path} -> {target_img_path}"
    
    filename = target_img_path.name
    matched_prefix = None
    for pfx in expected_prefixes:
        if filename.startswith(pfx):
            matched_prefix = pfx
            break
    assert matched_prefix is not None, f"发现非预期图片文件: {filename}"
    found_prefixes.add(matched_prefix)
    
    # 检验文件体积 > 50 KB (50,000 字节)
    file_size = target_img_path.stat().st_size
    assert file_size > 50000, f"图片文件体积小于 50 KB: {filename} ({file_size} 字节)"
    
    # 检验图片格式与 DPI >= 300
    with Image.open(target_img_path) as img:
        img_format = img.format
        assert img_format == "PNG", f"图片格式非 PNG: {filename} ({img_format})"
        dpi = img.info.get("dpi")
        assert dpi is not None, f"图片缺失 DPI 元数据: {filename}"
        assert abs(dpi[0] - 300) < 1.0 and abs(dpi[1] - 300) < 1.0, f"图片 DPI 偏离 300: {filename} ({dpi})"
        width, height = img.size
        assert width >= 1000 and height >= 600, f"图片像素尺寸异常: {filename} ({width}x{height})"
    
    # 检验图片后紧随斜体图注
    end_idx = m.end()
    post_text = proposal_content[end_idx:end_idx + 600].strip()
    assert post_text.startswith("*图") and "*" in post_text[1:], f"图片后缺少规范斜体图注: {filename}"
    print(f"图片资产验证合规: {filename} (大小: {file_size / 1024:.1f} KB, DPI: {dpi[0]:.0f}, 尺寸: {width}x{height})")

for pfx in expected_prefixes:
    assert pfx in found_prefixes, f"缺少前缀为 {pfx} 的图片引用"

# 3. 对抗性文本文风与禁用词扫描
forbidden_regexes = [
    (r"不是[^，。\n]+而是", "禁用句式：不是...而是..."),
    (r"要[^，。\n]+而不是", "禁用句式：要...而不是..."),
    (r"而不是", "禁用对比句式：而不是"),
    (r"而非", "禁用对比句式：而非"),
    (r"落地", "禁用互联网黑话：落地"),
    (r"钉死", "禁用互联网黑话：钉死"),
    (r"对齐", "禁用互联网黑话：对齐"),
    (r"技术栈", "禁用黑话：技术栈"),
    (r"模型栈", "禁用黑话：模型栈"),
    (r"两字版本", "禁用缩写表达：两字版本"),
    (r"单字版本", "禁用缩写表达：单字版本"),
    (r"一句话总结", "禁用套话：一句话总结"),
    (r"上述内容是", "禁用套话：上述内容是"),
    (r"总的来说", "禁用套话：总的来说"),
    (r"综上所述", "禁用套话：综上所述"),
]

docs_to_scan = [
    PROPOSAL_DOC,
    LOGS_DIR / "EXP-080.md",
    LOGS_DIR / "EXP-081.md",
    LOGS_DIR / "EXP-082.md",
    LOGS_DIR / "EXP-083.md",
    LOGS_DIR / "EXP-084.md",
    LOGS_DIR / "EXP-085.md",
]

for doc_path in docs_to_scan:
    text = doc_path.read_text(encoding="utf-8")
    for pattern, desc in forbidden_regexes:
        matches = list(re.finditer(pattern, text))
        assert len(matches) == 0, f"文件 {doc_path.name} 命中 {desc}: {[m.group(0) for m in matches]}"

    # 单字动词/缩写针对性扫描：例如独立的单字 判、推、抛、挂、崩、死
    single_char_patterns = [
        (r"(?<![\u4e00-\u9fa5a-zA-Z0-9])崩(?![\u4e00-\u9fa5a-zA-Z0-9])", "单字缩写：崩"),
        (r"死锁|死局", None),  # 合法双字词
        (r"(?<![\u4e00-\u9fa5a-zA-Z0-9])挂(?![\u4e00-\u9fa5a-zA-Z0-9])", "单字缩写：挂"),
    ]
    for pat, desc in single_char_patterns:
        if desc is None:
            continue
        matches = list(re.finditer(pat, text))
        assert len(matches) == 0, f"文件 {doc_path.name} 命中 {desc}: {[m.group(0) for m in matches]}"
    print(f"文风合规扫描通过: {doc_path.name}")

# 4. 检验 Canvas 拓扑图完整性
canvas_file = PROPOSAL_DOC.parent / "IDEA-008_演化决策树.canvas"
assert canvas_file.is_file(), f"Canvas 文件不存在: {canvas_file}"
import json
canvas_data = json.loads(canvas_file.read_text(encoding="utf-8"))
node_ids = {n["id"] for n in canvas_data.get("nodes", [])}
for eid in ["exp080", "exp081", "exp082", "exp083", "exp084", "exp085"]:
    target_nid = f"i8-{eid}"
    assert target_nid in node_ids, f"Canvas 决策树缺少节点: {target_nid}"
print("Canvas 拓扑节点完整性校验通过")

print("所有静态检测与对抗性验证项全部通过！")
