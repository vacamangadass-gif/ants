import glob
import json
import os
import re
from PIL import Image

# 1. 验证 4 张 300 DPI 学术级图表
image_files = [
    "tinybrain/ima/exp074_ablation_score_win_rates.png",
    "tinybrain/ima/exp074_stall_reduction_breakdown.png",
    "tinybrain/ima/exp074_ablation_multimetric_radar.png",
    "tinybrain/ima/exp074_map_performance_heatmap.png",
]

for image_path in image_files:
    assert os.path.isfile(image_path), f"图像文件缺失: {image_path}"
    assert os.path.getsize(image_path) > 100000, f"图像文件大小过小: {image_path}"
    with Image.open(image_path) as image_obj:
        dpi = image_obj.info.get("dpi")
        assert dpi is not None, f"图像缺失 DPI 元数据: {image_path}"
        assert round(dpi[0]) == 300 and round(dpi[1]) == 300, f"图像 DPI 不满足 300 DPI 要求: {dpi}"
print("4 张学术级图表验证通过 (300 DPI)")

# 2. 验证 8 篇实验日志完整性与结构要素
log_files = [
    "tinybrain/04 实验日志/EXP-070.md",
    "tinybrain/04 实验日志/EXP-071A.md",
    "tinybrain/04 实验日志/EXP-071B.md",
    "tinybrain/04 实验日志/EXP-072A.md",
    "tinybrain/04 实验日志/EXP-072B.md",
    "tinybrain/04 实验日志/EXP-073A.md",
    "tinybrain/04 实验日志/EXP-073B.md",
    "tinybrain/04 实验日志/EXP-074.md",
]

required_sections = [
    "## 1. 完整执行命令",
    "## 2.",
    "## 3. 评价指标与原始数据路径",
    "## 4. 评测结果与指标波动区间",
    "## 5. 对假设的支持或反驳程度判定",
    "## 6. 后续行动与决策依据",
]

forbidden_words = ["不是", "落地", "钉死", "对齐", "栈"]

for log_path in log_files:
    assert os.path.isfile(log_path), f"日志文件缺失: {log_path}"
    with open(log_path, "r", encoding="utf-8") as file_handle:
        content = file_handle.read()

    for sec in required_sections:
        assert sec in content, f"日志 {log_path} 缺失小节: {sec}"

    for word in forbidden_words:
        assert word not in content, f"日志 {log_path} 包含禁用词: {word}"

    # 检查图像引用解析有效性与斜体图注
    img_matches = re.findall(r"!\[(.*?)\]\((.*?)\)", content)
    for title, rel_path in img_matches:
        abs_img_path = os.path.normpath(os.path.join(os.path.dirname(log_path), rel_path))
        assert os.path.isfile(abs_img_path), f"日志 {log_path} 图像引用无效: {rel_path} -> {abs_img_path}"
        caption_pattern = rf"!\[.*?\]\({re.escape(rel_path)}\)\s*\n\s*\*(.*?)\*"
        assert re.search(caption_pattern, content) is not None, f"日志 {log_path} 图像 {rel_path} 缺少紧随其后的斜体图注"

print("8 篇实验日志验证通过 (结构完备、图像引用解析有效、图注健全、无禁用词)")

# 3. 验证核心提案说明第 7 节
proposal_file = "tinybrain/03 Idea 构想拓扑/IDEA-007_目标执行与动作语义修复/IDEA-007_核心提案说明.md"
assert os.path.isfile(proposal_file), f"提案文件缺失: {proposal_file}"
with open(proposal_file, "r", encoding="utf-8") as file_handle:
    proposal_content = file_handle.read()

assert "## 7. 阶段性结论与决策记录" in proposal_content, "提案缺失第 7 节"
assert "### 7.1 第一批次（Batch 1，EXP-070 至 EXP-074）实证消融矩阵与假设裁决" in proposal_content
assert "### 7.2 演化门禁 1（Gate 1）达标裁决" in proposal_content
assert "### 7.3 技术演进决策与后续批次分化" in proposal_content
for word in forbidden_words:
    assert word not in proposal_content[proposal_content.find("## 7."):], f"提案第 7 节包含禁用词: {word}"
print("核心提案说明第 7 节验证通过")

# 4. 验证演化决策树 Canvas
canvas_file = "tinybrain/03 Idea 构想拓扑/IDEA-007_目标执行与动作语义修复/IDEA-007_演化决策树.canvas"
assert os.path.isfile(canvas_file), f"Canvas 文件缺失: {canvas_file}"
with open(canvas_file, "r", encoding="utf-8") as file_handle:
    canvas_data = json.load(file_handle)

node_ids = {n["id"]: n for n in canvas_data["nodes"]}
assert "i7-exp070" in node_ids and node_ids["i7-exp070"]["color"] == "4"
assert "i7-exp071a" in node_ids and node_ids["i7-exp071a"]["color"] == "4"
assert "i7-exp071b" in node_ids and node_ids["i7-exp071b"]["color"] == "4"
assert "i7-exp072a" in node_ids and node_ids["i7-exp072a"]["color"] == "4"
assert "i7-exp072b" in node_ids and node_ids["i7-exp072b"]["color"] == "1"
assert "i7-exp073a" in node_ids and node_ids["i7-exp073a"]["color"] == "4"
assert "i7-exp073b" in node_ids and node_ids["i7-exp073b"]["color"] == "4"
assert "i7-exp074" in node_ids and node_ids["i7-exp074"]["color"] == "4"
assert "i7-gate1" in node_ids and node_ids["i7-gate1"]["color"] == "4"
assert len(canvas_data["nodes"]) == 18, f"节点总数不符合预期: {len(canvas_data['nodes'])}"
assert len(canvas_data["edges"]) == 22, f"连线总数不符合预期: {len(canvas_data['edges'])}"
print("演化决策树 Canvas 验证通过 (18 节点, 22 连线, 状态颜色与语义标签精确匹配)")

print("ALL_M6_DELIVERABLES_VERIFIED_SUCCESSFULLY")
