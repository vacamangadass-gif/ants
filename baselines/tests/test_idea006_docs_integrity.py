from pathlib import Path
import json
import re
import pytest

ROOT = Path(__file__).resolve().parents[2]
DOC_PATH = ROOT / "tinybrain" / "03 Idea 构想拓扑" / "IDEA-006_数据集聚合模仿学习与分布偏移缓解" / "IDEA-006_核心提案说明.md"
CANVAS_PATH = ROOT / "tinybrain" / "03 Idea 构想拓扑" / "IDEA-006_数据集聚合模仿学习与分布偏移缓解" / "IDEA-006_演化决策树.canvas"


def test_idea006_doc_structure_and_sections():
    assert DOC_PATH.is_file(), f"核心提案文件不存在: {DOC_PATH}"
    content = DOC_PATH.read_text(encoding="utf-8")

    # 验证元数据头部
    assert content.startswith("---"), "文档必须以 YAML frontmatter 开头"
    assert "id: IDEA-006" in content
    assert "status: 验证通过" in content

    # 验证全部六大核心一级章节及容量阶梯小节存在
    required_sections = [
        "## 1. 研究问题与事实依据",
        "### 1.4 容量阶梯效应理论推导与双轨饱和分化机制",
        "## 2. 核心科学假设",
        "## 3. 双轨长程技术路线与硬核工程门禁",
        "## 4. 双轨长程梯次对照实验设计",
        "## 5. 验证标准、评测数据与学术图表体系",
        "## 6. 阶段性结论与决策记录",
    ]
    for sec in required_sections:
        assert sec in content, f"缺失必选章节: {sec}"


def test_idea006_doc_wikilinks_exist():
    content = DOC_PATH.read_text(encoding="utf-8")
    tinybrain_root = ROOT / "tinybrain"
    wikilinks = re.findall(r'\[\[(.*?)\]\]', content)
    assert len(wikilinks) >= 5, f"双向链接数量不足: {len(wikilinks)}"
    for link in wikilinks:
        target = link.split("|")[0].strip()
        target_path = tinybrain_root / f"{target}.md"
        assert target_path.is_file(), f"双向链接指向的文件不存在: {target_path} (from [[{link}]])"


def test_idea006_doc_image_references_and_captions():
    content = DOC_PATH.read_text(encoding="utf-8")
    doc_dir = DOC_PATH.parent

    # 提取全部图像引用
    matches = re.findall(r'!\[(.*?)\]\((.*?)\)', content)
    assert len(matches) == 8, f"必须且只能嵌入 8 张科研图表，实际提取到 {len(matches)} 张"

    expected_images = [
        "../../ima/dagger_telemetry_comparison.png",
        "../../ima/exp033_dagger_telemetry.png",
        "../../ima/exp033_training_convergence.png",
        "../../ima/tournament_score_win_rates.png",
        "../../ima/tournament_pairwise_heatmap.png",
        "../../ima/tournament_resilience_scatter.png",
        "../../ima/dagger_iterations_evolution.png",
        "../../ima/dagger_samples_marginal_benefit.png",
    ]

    extracted_urls = [m[1] for m in matches]
    for exp in expected_images:
        assert exp in extracted_urls, f"缺失目标科研图表: {exp}"
        abs_img = (doc_dir / exp).resolve()
        assert abs_img.is_file(), f"图像文件在本地文件系统中不存在: {abs_img}"

    # 验证每张图像下方均配置斜体图注
    for alt, url in matches:
        pattern = re.compile(rf'!\[{re.escape(alt)}\]\({re.escape(url)}\)\s*\n\*(图 \d+：.*?)\*', re.DOTALL)
        assert pattern.search(content), f"图表缺少斜体图注: {url}"


def test_idea006_doc_metric_accuracy():
    content = DOC_PATH.read_text(encoding="utf-8")

    # 核心评测数据检验
    key_metrics = [
        "65.74%",  # exp033_iter8 得分率
        "48.89%",  # exp031_iter8 得分率
        "+16.85%",  # Micro 相比 Nano 净胜优势
        "2,160",   # 循环赛总对局数
        "0.4539",  # Kneedle D_3
        "0.3598",  # Kneedle D_5
        "7.1 倍",  # 边际效率衰减倍率
        "-43.7%",  # 迭代 2 受阻率最大降幅
        "0.87%",   # 迭代 8 受阻率
        "916,664", # 路线 B 最终席位步数
        "916,667", # 路线 A 最终席位步数
        "models/exp031",
        "models/exp033",
    ]
    for metric in key_metrics:
        assert metric in content, f"缺失核心定量指标: {metric}"


def test_idea006_doc_banned_words_and_phrases():
    content = DOC_PATH.read_text(encoding="utf-8")
    banned = ["不是", "落地", "钉死", "对齐", "栈", "一句话总结", "上述内容是", "死阻"]
    for word in banned:
        assert word not in content, f"核心提案文档违反禁令，包含违禁词: {word}"


def test_idea006_canvas_integrity_and_semantics():
    assert CANVAS_PATH.is_file(), f"Canvas 文件不存在: {CANVAS_PATH}"
    with open(CANVAS_PATH, "r", encoding="utf-8") as f:
        canvas = json.load(f)

    nodes = canvas["nodes"]
    edges = canvas["edges"]
    assert len(nodes) >= 14, f"Canvas 节点数量不足: {len(nodes)}"
    assert len(edges) >= 18, f"Canvas 连线数量不足: {len(edges)}"

    node_dict = {n["id"]: n for n in nodes}

    # 必选关键节点存在性
    required_node_ids = [
        "node-doc",
        "node-theory",
        "node-baseline",
        "node-root",
        "node-gate",
        "node-route-a",
        "node-route-b",
        "node-exp030",
        "node-exp031",
        "node-exp033-pre",
        "i6-D",
        "node-tournament",
        "node-marginal",
        "node-hypotheses",
        "node-conclusion",
    ]
    for nid in required_node_ids:
        assert nid in node_dict, f"Canvas 缺少核心节点: {nid}"

    # 验证核心确证节点采用规范 Color 4 (绿色)
    confirmed_nodes = [
        "node-theory",
        "node-baseline",
        "node-root",
        "node-route-a",
        "node-route-b",
        "node-exp030",
        "node-exp031",
        "node-exp033-pre",
        "i6-D",
        "node-tournament",
        "node-marginal",
        "node-hypotheses",
        "node-conclusion",
    ]
    for nid in confirmed_nodes:
        assert node_dict[nid].get("color") == "4", f"节点 {nid} 颜色代码必须为 4"

    # 验证硬门禁节点为 Color 1 (红色)
    assert node_dict["node-gate"].get("color") == "1", "门禁节点必须为 Color 1"

    # 验证全部连线端点有效且包含清晰语义标签
    edge_ids = set()
    for edge in edges:
        edge_ids.add(edge["id"])
        assert edge["fromNode"] in node_dict, f"连线起点无效: {edge}"
        assert edge["toNode"] in node_dict, f"连线终点无效: {edge}"
        assert "label" in edge and len(edge["label"]) > 0, f"连线缺少语义标签: {edge}"

    # 验证因果边精确性：EXP-030 不属于 16 模型循环赛参评模型
    assert "edge-exp030-tournament" not in edge_ids, "EXP-030 未参与 16 模型双向循环赛，连线关系不实"
    assert "edge-exp030-exp031" in edge_ids, "缺失 EXP-030 对 EXP-031 的消融对照连线"

    # 验证无节点重叠 (简单矩形相交检测)
    for i, n1 in enumerate(nodes):
        for j, n2 in enumerate(nodes):
            if i >= j:
                continue
            # 检测 n1 与 n2 是否完全相交重叠
            x1_min, x1_max = n1["x"], n1["x"] + n1["width"]
            y1_min, y1_max = n1["y"], n1["y"] + n1["height"]
            x2_min, x2_max = n2["x"], n2["x"] + n2["width"]
            y2_min, y2_max = n2["y"], n2["y"] + n2["height"]

            overlap_x = min(x1_max, x2_max) - max(x1_min, x2_min)
            overlap_y = min(y1_max, y2_max) - max(y1_min, y2_min)
            assert not (overlap_x > 0 and overlap_y > 0), f"节点发生物理重叠: {n1['id']} 与 {n2['id']}"

    # Canvas 违禁词检测
    canvas_str = json.dumps(canvas, ensure_ascii=False)
    banned = ["不是", "落地", "钉死", "对齐", "栈", "死阻"]
    for word in banned:
        assert word not in canvas_str, f"Canvas 画布违反禁令，包含违禁词: {word}"
