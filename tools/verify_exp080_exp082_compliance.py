import os
import re
import json

files_to_check = [
    "/home/host/project/ants/tinybrain/04 实验日志/EXP-080.md",
    "/home/host/project/ants/tinybrain/04 实验日志/EXP-081.md",
    "/home/host/project/ants/tinybrain/04 实验日志/EXP-082.md",
    "/home/host/project/ants/tinybrain/03 Idea 构想拓扑/IDEA-008_群体协调与规模效应/IDEA-008_核心提案说明.md",
]

# 检查图表资产存在性与大小
images_to_check = [
    "/home/host/project/ants/tinybrain/ima/exp080_scaling_telemetry.png",
    "/home/host/project/ants/tinybrain/ima/exp081_mcmf_latency_breakdown.png",
    "/home/host/project/ants/tinybrain/ima/exp081_redundancy_collision_reduction.png",
    "/home/host/project/ants/tinybrain/ima/exp082_gate3_1_entropy_distribution.png",
]

from PIL import Image

for img in images_to_check:
    assert os.path.isfile(img), f"图片缺失: {img}"
    size = os.path.getsize(img)
    assert size > 50000, f"图片大小过小: {img} ({size} bytes)"
    im = Image.open(img)
    dpi = im.info.get("dpi")
    assert dpi is not None and abs(dpi[0] - 300) < 1.0, f"图片 DPI 非 300: {img} ({dpi})"
    print(f"图片正常: {os.path.basename(img)} ({size / 1024:.1f} KB, DPI {dpi[0]:.0f})")

# 检查禁用词与句式
forbidden_patterns = [
    r"不是[^，。\n]+而是",
    r"要[^，。\n]+而不是",
    r"落地",
    r"钉死",
    r"对齐",
    r"栈",
    r"两字版本",
    r"单字版本",
    r"一句话总结",
    r"上述内容是",
    r"[\u4e00-\u9fa5]达\s*[\d]",
    r"[\u4e00-\u9fa5]超\s*[\d]",
]

for filepath in files_to_check:
    assert os.path.isfile(filepath), f"文件不存在: {filepath}"
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    for pat in forbidden_patterns:
        matches = re.findall(pat, content)
        assert len(matches) == 0, f"文件 {filepath} 包含禁用词/句式: {pat} (匹配: {matches})"

    # 检查图片引用与斜体图注
    img_matches = list(re.finditer(r"!\[(.*?)\]\((.*?)\)", content))
    assert len(img_matches) > 0, f"文件 {filepath} 未包含任何图片引用"

    for match in img_matches:
        alt_text = match.group(1)
        rel_path = match.group(2)
        assert len(alt_text) > 0, f"文件 {filepath} 图片缺少说明: {match.group(0)}"

        # 检查解析路径
        doc_dir = os.path.dirname(filepath)
        abs_img_path = os.path.normpath(os.path.join(doc_dir, rel_path))
        assert os.path.isfile(abs_img_path), f"文件 {filepath} 中图片链接无效: {rel_path} -> {abs_img_path}"

        # 检查其后紧跟斜体图注
        end_pos = match.end()
        after_text = content[end_pos:end_pos + 400].strip()
        assert after_text.startswith("*") and "*" in after_text[1:], f"文件 {filepath} 图片后缺少斜体图注: {match.group(0)}"

    print(f"规范与图片检查通过: {os.path.basename(filepath)}")

# 检查实验日志的 YAML 与 5 个章节
for exp_file in files_to_check[:3]:
    with open(exp_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    assert lines[0].strip() == "---", f"{exp_file} 缺少开头的 YAML 分隔符"
    yaml_end = -1
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            yaml_end = i
            break
    assert yaml_end > 0, f"{exp_file} 缺少结束的 YAML 分隔符"

    yaml_text = "".join(lines[1:yaml_end])
    assert "id:" in yaml_text, f"{exp_file} YAML 缺少 id"
    assert "related_idea:" in yaml_text, f"{exp_file} YAML 缺少 related_idea"
    assert "date:" in yaml_text, f"{exp_file} YAML 缺少 date"
    assert "outcome:" in yaml_text, f"{exp_file} YAML 缺少 outcome"

    full_text = "".join(lines)
    assert "## 1. 实验目标与关联" in full_text, f"{exp_file} 缺少第 1 节"
    assert "## 2. 环境与参数配置" in full_text, f"{exp_file} 缺少第 2 节"
    assert "## 3. 过程记录与异动排查" in full_text, f"{exp_file} 缺少第 3 节"
    assert "## 4. 评测数据" in full_text, f"{exp_file} 缺少第 4 节"
    assert "## 5. 证据分析与直接推论" in full_text, f"{exp_file} 缺少第 5 节"

# 检查 Idea 提案的 YAML 与 6 列决策表格
idea_file = files_to_check[3]
with open(idea_file, "r", encoding="utf-8") as f:
    idea_text = f.read()

assert "status: 实验中" in idea_text or "status: 已完成" in idea_text, f"Idea 提案状态非合法值: 实验中 或 已完成"
assert "| 实验编号 | 验证目标 | 关联日志 | 实验执行状态 | 假设检验结果 | 后续行动记录 |" in idea_text, f"Idea 提案第 6 节表格表头不符合规范六列定义"

print("全部检查 100% 通过！")
