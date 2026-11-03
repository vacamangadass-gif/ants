import json
import os
import re
from pathlib import Path
from PIL import Image
import pytest

PROJECT_ROOT = Path("/home/host/project/ants")
VAULT_ROOT = PROJECT_ROOT / "tinybrain"
LOGS_DIR = VAULT_ROOT / "04 实验日志"
PROPOSAL_DOC = VAULT_ROOT / "03 Idea 构想拓扑" / "IDEA-007_目标执行与动作语义修复" / "IDEA-007_核心提案说明.md"
CANVAS_FILE = VAULT_ROOT / "03 Idea 构想拓扑" / "IDEA-007_目标执行与动作语义修复" / "IDEA-007_演化决策树.canvas"

LOG_FILES = [
    LOGS_DIR / "EXP-070.md",
    LOGS_DIR / "EXP-071A.md",
    LOGS_DIR / "EXP-071B.md",
    LOGS_DIR / "EXP-072A.md",
    LOGS_DIR / "EXP-072B.md",
    LOGS_DIR / "EXP-073A.md",
    LOGS_DIR / "EXP-073B.md",
    LOGS_DIR / "EXP-074.md",
]


@pytest.fixture(scope="module")
def canvas_data():
    assert CANVAS_FILE.exists(), f"Canvas 文件缺失: {CANVAS_FILE}"
    with open(CANVAS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


class TestCanvasTopologyIntegrity:

    def test_canvas_nodes_and_edges_count(self, canvas_data):
        nodes = canvas_data["nodes"]
        edges = canvas_data["edges"]
        assert len(nodes) == 18, f"节点总数不符合预期: {len(nodes)}"
        assert len(edges) == 22, f"连线总数不符合预期: {len(edges)}"

    def test_canvas_node_coordinates_and_non_overlapping(self, canvas_data):
        nodes = canvas_data["nodes"]
        node_map = {}
        for n in nodes:
            nid = n["id"]
            assert nid not in node_map, f"节点 ID 重复: {nid}"
            node_map[nid] = n
            for prop in ["x", "y", "width", "height"]:
                assert prop in n, f"节点 {nid} 缺少属性 {prop}"
                assert isinstance(n[prop], (int, float)), f"节点 {nid} 属性 {prop} 非数值"
            assert n["width"] > 0
            assert n["height"] > 0

        # 对抗性重叠检测
        node_list = list(node_map.values())
        for i in range(len(node_list)):
            n1 = node_list[i]
            r1_l, r1_r = n1["x"], n1["x"] + n1["width"]
            r1_t, r1_b = n1["y"], n1["y"] + n1["height"]
            for j in range(i + 1, len(node_list)):
                n2 = node_list[j]
                r2_l, r2_r = n2["x"], n2["x"] + n2["width"]
                r2_t, r2_b = n2["y"], n2["y"] + n2["height"]

                overlap_x = r1_l < r2_r and r1_r > r2_l
                overlap_y = r1_t < r2_b and r1_b > r2_t
                assert not (overlap_x and overlap_y), f"节点重叠冲突: {n1['id']} 与 {n2['id']}"

    def test_canvas_edges_endpoints_and_labels(self, canvas_data):
        nodes = {n["id"]: n for n in canvas_data["nodes"]}
        edges = canvas_data["edges"]
        connected = set()

        for edge in edges:
            eid = edge["id"]
            from_node = edge["fromNode"]
            to_node = edge["toNode"]
            assert from_node in nodes, f"连线 {eid} fromNode 未定义: {from_node}"
            assert to_node in nodes, f"连线 {eid} toNode 未定义: {to_node}"
            assert "label" in edge and len(edge["label"].strip()) > 0, f"连线 {eid} 缺少语义标签"
            connected.add(from_node)
            connected.add(to_node)

        orphans = set(nodes.keys()) - connected
        assert len(orphans) == 0, f"发现脱节孤立节点: {orphans}"

    def test_canvas_file_nodes_and_wiki_links(self, canvas_data):
        for node in canvas_data["nodes"]:
            nid = node["id"]
            if node.get("type") == "file":
                rel_path = node["file"]
                target = VAULT_ROOT / rel_path
                assert target.exists(), f"文件节点 {nid} 目标不存在: {target}"
            elif node.get("type") == "text":
                text = node.get("text", "")
                wiki_links = re.findall(r"\[\[(.*?)\]\]", text)
                for link in wiki_links:
                    clean_link = link.split("|")[0].split("#")[0].strip()
                    target1 = VAULT_ROOT / clean_link
                    target2 = VAULT_ROOT / f"{clean_link}.md"
                    assert target1.exists() or target2.exists(), f"Canvas 节点 {nid} wiki 链接失效: {link}"

    def test_canvas_color_semantics(self, canvas_data):
        nodes = {n["id"]: n for n in canvas_data["nodes"]}
        expected_colors = {
            "i7-root": "2",
            "i7-hypotheses": "3",
            "i7-exp070": "4",
            "i7-exp071a": "4",
            "i7-exp071b": "4",
            "i7-exp072a": "4",
            "i7-exp072b": "1",
            "i7-exp073a": "4",
            "i7-exp073b": "4",
            "i7-exp074": "4",
            "i7-gate1": "4",
            "i7-exp075": "3",
            "i7-exp076": "3",
            "i7-exp077": "3",
            "i7-gate2": "5",
        }
        for nid, color in expected_colors.items():
            assert nodes[nid].get("color") == color, f"节点 {nid} 颜色不符: {nodes[nid].get('color')} != {color}"


class TestLinkAndImageIntegrity:
    @pytest.mark.parametrize("log_path", LOG_FILES)
    def test_log_wiki_links_and_images(self, log_path: Path):
        assert log_path.exists(), f"实验日志缺失: {log_path}"
        content = log_path.read_text(encoding="utf-8")

        # 校验 Wiki-links
        wiki_links = re.findall(r"\[\[(.*?)\]\]", content)
        for link in wiki_links:
            clean_link = link.split("|")[0].split("#")[0].strip()
            target1 = VAULT_ROOT / clean_link
            target2 = VAULT_ROOT / f"{clean_link}.md"
            assert target1.exists() or target2.exists(), f"{log_path.name} wiki 链接失效: {link}"

        # 校验图片路径与属性
        images = re.findall(r"!\[(.*?)\]\((.*?)\)", content)
        assert len(images) > 0, f"{log_path.name} 缺少内嵌学术图表"
        for alt_text, img_rel in images:
            img_abs = (log_path.parent / img_rel).resolve()
            assert img_abs.exists(), f"{log_path.name} 图片文件缺失: {img_rel}"
            assert img_abs.is_file()
            assert img_abs.stat().st_size > 100000

            with Image.open(img_abs) as img_obj:
                dpi = img_obj.info.get("dpi")
                assert dpi is not None, f"{img_rel} 缺少 DPI 元数据"
                assert round(dpi[0]) == 300 and round(dpi[1]) == 300, f"{img_rel} DPI 非 300: {dpi}"

            caption_pattern = rf"!\[.*?\]\({re.escape(img_rel)}\)\s*\n\s*\*(.*?)\*"
            assert re.search(caption_pattern, content) is not None, f"{log_path.name} 图片 {img_rel} 缺少斜体图注"

    def test_proposal_section7_wiki_links(self):
        assert PROPOSAL_DOC.exists()
        content = PROPOSAL_DOC.read_text(encoding="utf-8")
        sec7_idx = content.find("## 7. 阶段性结论与决策记录")
        assert sec7_idx != -1
        sec7_content = content[sec7_idx:]

        wiki_links = re.findall(r"\[\[(.*?)\]\]", sec7_content)
        assert len(wiki_links) >= 8, f"提案第 7 节 wiki 链接数量不足: {len(wiki_links)}"
        for link in wiki_links:
            clean_link = link.split("|")[0].split("#")[0].strip()
            target1 = VAULT_ROOT / clean_link
            target2 = VAULT_ROOT / f"{clean_link}.md"
            assert target1.exists() or target2.exists(), f"提案第 7 节链接失效: {link}"


class TestLanguageAndForbiddenWordsAudit:
    @pytest.mark.parametrize("log_path", LOG_FILES)
    def test_log_forbidden_words_strict_scan(self, log_path: Path):
        content = log_path.read_text(encoding="utf-8")
        content_no_code = re.sub(r"`[^`]+`", "", content)

        forbidden_patterns = [
            (r"不是.*?而是", "不允许使用'不是...而是...'句式"),
            (r"要.*?而不是", "不允许使用'要...而不是...'句式"),
            (r"而不是", "不允许使用'而不是'"),
            (r"落地", "禁用非技术黑话'落地'"),
            (r"钉死", "禁用非技术黑话'钉死'"),
            (r"对齐", "禁用非技术黑话'对齐'"),
            (r"技术栈", "禁用'技术栈'"),
            (r"模型栈", "禁用'模型栈'"),
            (r"第一版先", "禁止分步暂定措辞'第一版先'"),
            (r"一句话总结", "禁止总结性套话'一句话总结'"),
            (r"两字版本", "禁止单字缩略用法'两字版本'"),
            (r"单字版本", "禁止单字缩略用法'单字版本'"),
        ]

        for pat, desc in forbidden_patterns:
            matches = re.findall(pat, content_no_code)
            assert len(matches) == 0, f"{log_path.name} 包含禁用模式 ({desc}): {matches}"

        # 检查汉字'栈'
        stack_matches = re.findall(r"[\u4e00-\u9fa5]{0,3}栈[\u4e00-\u9fa5]{0,3}", content_no_code)
        assert len(stack_matches) == 0, f"{log_path.name} 包含'栈'字: {stack_matches}"

        # 检查非规范单字动词
        single_char_checks = [
            (r"(?<![崩])崩(?![溃])", "单字缩写'崩'"),
            (r"(?<![抛])抛(?![出])", "单字缩写'抛'"),
            (r"(?<![挂])挂(?![起])", "单字缩写'挂'"),
        ]
        for pat, desc in single_char_checks:
            m = re.findall(rf"[\u4e00-\u9fa5]{{0,2}}{pat}[\u4e00-\u9fa5]{{0,2}}", content_no_code)
            assert len(m) == 0, f"{log_path.name} 包含违规{desc}: {m}"

    def test_proposal_section7_forbidden_words_strict_scan(self):
        content = PROPOSAL_DOC.read_text(encoding="utf-8")
        sec7_idx = content.find("## 7. 阶段性结论与决策记录")
        assert sec7_idx != -1
        sec7_content = content[sec7_idx:]
        content_no_code = re.sub(r"`[^`]+`", "", sec7_content)

        forbidden_patterns = [
            (r"不是.*?而是", "不允许使用'不是...而是...'句式"),
            (r"要.*?而不是", "不允许使用'要...而不是...'句式"),
            (r"而不是", "不允许使用'而不是'"),
            (r"落地", "禁用非技术黑话'落地'"),
            (r"钉死", "禁用非技术黑话'钉死'"),
            (r"对齐", "禁用非技术黑话'对齐'"),
            (r"技术栈", "禁用'技术栈'"),
            (r"模型栈", "禁用'模型栈'"),
            (r"第一版先", "禁止分步暂定措辞'第一版先'"),
            (r"一句话总结", "禁止总结性套话'一句话总结'"),
            (r"两字版本", "禁止单字缩略用法'两字版本'"),
            (r"单字版本", "禁止单字缩略用法'单字版本'"),
        ]

        for pat, desc in forbidden_patterns:
            matches = re.findall(pat, content_no_code)
            assert len(matches) == 0, f"提案第 7 节包含禁用模式 ({desc}): {matches}"

        stack_matches = re.findall(r"[\u4e00-\u9fa5]{0,3}栈[\u4e00-\u9fa5]{0,3}", content_no_code)
        assert len(stack_matches) == 0, f"提案第 7 节包含'栈'字: {stack_matches}"
