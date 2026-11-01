from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import List, Tuple

import pytest

ROOT = Path("/home/host/project/ants")
TB_DIR = ROOT / "tinybrain"
CANVAS_PATH = TB_DIR / "03 Idea 构想拓扑/IDEA-006_数据集聚合模仿学习与分布偏移缓解/IDEA-006_演化决策树.canvas"
DOC_PATHS = [
    TB_DIR / "04 实验日志/EXP-033.md",
    TB_DIR / "04 实验日志/EXP-031.md",
    TB_DIR / "03 Idea 构想拓扑/IDEA-006_数据集聚合模仿学习与分布偏移缓解/IDEA-006_核心提案说明.md",
]
WASM_PATH = ROOT / "dist/tb-ants.wasm"
EXPECTED_WASM_DIGEST = "819166e7918117a1cbbd67dc042e7368a076a2dd45cea5122683983912d591e1"


class TestCanvasSchemaAndTopology:
    # 检验 Canvas 文件 JSON 有效性
    def test_canvas_valid_json(self):
        assert CANVAS_PATH.is_file(), f"Canvas 文件不存在: {CANVAS_PATH}"
        raw_text = CANVAS_PATH.read_text(encoding="utf-8")
        data = json.loads(raw_text)
        assert isinstance(data, dict), "Canvas 根结构必须为 JSON 字典对象"
        assert "nodes" in data, "Canvas 缺少 nodes 节点集合"
        assert "edges" in data, "Canvas 缺少 edges 连线集合"
        assert len(data["nodes"]) >= 8, f"Canvas 节点数量不足: {len(data['nodes'])}"
        assert len(data["edges"]) >= 10, f"Canvas 连线数量不足: {len(data['edges'])}"

    # 检验每个节点包含必须字段且格式合法
    def test_canvas_nodes_schema(self):
        data = json.loads(CANVAS_PATH.read_text(encoding="utf-8"))
        nodes = data["nodes"]
        node_ids = set()

        for idx, node in enumerate(nodes):
            assert "id" in node, f"节点 [{idx}] 缺失 id 字段"
            assert isinstance(node["id"], str) and len(node["id"].strip()) > 0, f"节点 [{idx}] id 为空"
            assert node["id"] not in node_ids, f"节点 id 重复: {node['id']}"
            node_ids.add(node["id"])

            assert "type" in node, f"节点 {node['id']} 缺失 type 字段"
            assert node["type"] in ("text", "file", "group", "link"), f"节点 {node['id']} type 非法: {node['type']}"

            assert "x" in node and isinstance(node["x"], int), f"节点 {node['id']} x 坐标必须为整数"
            assert "y" in node and isinstance(node["y"], int), f"节点 {node['id']} y 坐标必须为整数"
            assert "width" in node and isinstance(node["width"], int) and node["width"] > 0, f"节点 {node['id']} width 必须为正整数"
            assert "height" in node and isinstance(node["height"], int) and node["height"] > 0, f"节点 {node['id']} height 必须为正整数"

            if node["type"] == "text":
                assert "text" in node, f"文本节点 {node['id']} 缺失 text 字段"
                assert isinstance(node["text"], str) and len(node["text"].strip()) > 0, f"文本节点 {node['id']} text 内容为空"
            elif node["type"] == "file":
                assert "file" in node, f"文件节点 {node['id']} 缺失 file 字段"
                linked_file = TB_DIR / node["file"]
                assert linked_file.is_file(), f"文件节点 {node['id']} 引用的文件不存在: {linked_file}"

    # 检验核心节点 i6-D 属性与绿色标记
    def test_canvas_node_i6_d_verified(self):
        data = json.loads(CANVAS_PATH.read_text(encoding="utf-8"))
        node_map = {n["id"]: n for n in data["nodes"]}

        assert "i6-D" in node_map, "缺少核心主线节点 i6-D"
        node_d = node_map["i6-D"]

        # 检验绿色 verified hypothesis 标记 (color 必须为 "4")
        assert node_d.get("color") == "4", f"节点 i6-D 颜色未标记为 4 (绿色验证成功): {node_d.get('color')}"

        # 检验描述文本包含完成指标与 EXP-033
        text = node_d.get("text", "")
        assert "EXP-033" in text, "节点 i6-D 文本未包含 EXP-033 标识"
        assert "已完成" in text, "节点 i6-D 文本未标注已完成状态"
        assert "65.74%" in text, "节点 i6-D 文本未包含 65.74% 综合得分率"
        assert "H1/H2/H3/H4" in text, "节点 i6-D 文本未标注假说确证结论"

    # 检验每条连线端点引用真实有效
    def test_canvas_edges_integrity(self):
        data = json.loads(CANVAS_PATH.read_text(encoding="utf-8"))
        node_ids = {n["id"] for n in data["nodes"]}
        edges = data["edges"]

        for idx, edge in enumerate(edges):
            assert "id" in edge, f"连线 [{idx}] 缺失 id 字段"
            assert isinstance(edge["id"], str) and len(edge["id"].strip()) > 0, f"连线 [{idx}] id 为空"

            assert "fromNode" in edge, f"连线 {edge['id']} 缺失 fromNode"
            assert edge["fromNode"] in node_ids, f"连线 {edge['id']} 源节点不存在: {edge['fromNode']}"

            assert "toNode" in edge, f"连线 {edge['id']} 缺失 toNode"
            assert edge["toNode"] in node_ids, f"连线 {edge['id']} 目标节点不存在: {edge['toNode']}"


class TestMarkdownSyntaxAndIntegrity:
    # 检验 Markdown 标题语法合规
    @pytest.mark.parametrize("doc_path", DOC_PATHS)
    def test_markdown_headings(self, doc_path: Path):
        assert doc_path.is_file(), f"文档不存在: {doc_path}"
        lines = doc_path.read_text(encoding="utf-8").splitlines()

        in_code_block = False
        for line_no, line in enumerate(lines, 1):
            stripped = line.strip()
            if stripped.startswith("```"):
                in_code_block = not in_code_block
                continue
            if in_code_block:
                continue

            if stripped.startswith("#"):
                # 标题必须符合 # 到 ###### 且后跟至少一个空格和内容
                match = re.match(r"^(#{1,6})\s+(\S.*)$", stripped)
                assert match is not None, f"文档 {doc_path.name}:{line_no} 标题语法错误: '{line}'"

    # 检验 Markdown 代码块围栏闭合
    @pytest.mark.parametrize("doc_path", DOC_PATHS)
    def test_markdown_code_fences(self, doc_path: Path):
        lines = doc_path.read_text(encoding="utf-8").splitlines()
        fence_lines = [idx for idx, line in enumerate(lines, 1) if line.strip().startswith("```")]
        assert len(fence_lines) % 2 == 0, f"文档 {doc_path.name} 代码块围栏未成对闭合，出现 {len(fence_lines)} 处标记"

    # 检验 Markdown 方括号与链接闭合完整性
    @pytest.mark.parametrize("doc_path", DOC_PATHS)
    def test_markdown_brackets_closure(self, doc_path: Path):
        text = doc_path.read_text(encoding="utf-8")

        # 移除多行代码块
        clean_text = re.sub(r"```[\s\S]*?```", "", text)
        # 移除行内代码
        clean_text = re.sub(r"`[^`\n]*`", "", clean_text)
        # 移除数学公式块
        clean_text = re.sub(r"\$\$[\s\S]*?\$\$", "", clean_text)
        clean_text = re.sub(r"\$[^$\n]*\$", "", clean_text)

        lines = clean_text.splitlines()
        for line_no, line in enumerate(lines, 1):
            # 检验双中括号 Obsidian Wiki 链接成对
            wiki_open = line.count("[[")
            wiki_close = line.count("]]")
            assert wiki_open == wiki_close, f"文档 {doc_path.name}:{line_no} WikiLink 未正确闭合: '{line}'"

            # 移除已闭合的 Wiki 链接以简化后续检查
            temp_line = re.sub(r"\[\[.*?\]\]", "", line)

            # 检验标准 Markdown 链接 [text](url)
            md_links = re.findall(r"!?\[([^\]]*)\]\(([^)]*)\)", temp_line)
            # 移除有效 Markdown 链接
            temp_line = re.sub(r"!?\[[^\]]*\]\([^)]*\)", "", temp_line)

            # 检查剩余孤立中括号
            single_open_sq = temp_line.count("[")
            single_close_sq = temp_line.count("]")
            assert single_open_sq == single_close_sq, f"文档 {doc_path.name}:{line_no} 存在未闭合中括号: '{line}'"


class TestImageLinksAndResolvability:
    # 提取并验证全部图片引用物理文件存在且体积大于 100 KB
    @pytest.mark.parametrize("doc_path", DOC_PATHS)
    def test_image_links_exist_and_exceed_100kb(self, doc_path: Path):
        content = doc_path.read_text(encoding="utf-8")
        images = re.findall(r"!\[([^\]]*)\]\(([^)]+)\)", content)

        assert len(images) >= 2, f"文档 {doc_path.name} 图片引用数量过少: {len(images)}"

        for alt, rel_path in images:
            # 移除 URL 中的锚点或查询参数
            clean_rel = rel_path.split("?")[0].split("#")[0]
            resolved = (doc_path.parent / clean_rel).resolve()

            assert resolved.is_file(), f"文档 {doc_path.name} 引用图片不存在: {rel_path} -> {resolved}"
            file_size = resolved.stat().st_size
            min_size = 100 * 1024  # 100 KB
            assert file_size >= min_size, f"图片体积低于 100 KB 门槛: {resolved.name} ({file_size} 字节 < {min_size} 字节)"

            # 验证 PNG 文件头魔数
            with open(resolved, "rb") as f:
                header = f.read(8)
            assert header == b"\x89PNG\r\n\x1a\n", f"图片格式非标准 PNG 文件: {resolved}"


class TestRuleComplianceAndZeroAsciiArt:
    # 严格检验零 ASCII Art 字符及边框
    @pytest.mark.parametrize("doc_path", DOC_PATHS + [CANVAS_PATH])
    def test_zero_ascii_art(self, doc_path: Path):
        content = doc_path.read_text(encoding="utf-8")

        # 检验 Unicode 框线字符集 (\u2500 - \u257F)
        box_chars = re.findall(r"[\u2500-\u257F]", content)
        assert len(box_chars) == 0, f"文件 {doc_path.name} 包含非法 Unicode 框线字符: {box_chars[:10]}"

        # 检验 ASCII 边框艺术 (如 +---+ 或 |---| 或 +====+)
        lines = content.splitlines()
        for line_no, line in enumerate(lines, 1):
            stripped = line.strip()
            # 排除正常 Markdown 表格对齐分割行 (例如 | :---: | :--- |)
            if re.match(r"^\|(\s*:?-+:?\s*\|)+$", stripped):
                continue
            # 排除 Mermaid 语法连线与横线
            if stripped.startswith("---") or stripped.startswith("-->") or "-->" in stripped:
                continue

            ascii_art_pattern = re.findall(r"\+[-=]{2,}|\+[-=]+\+|\|[-=]{2,}\|", line)
            assert len(ascii_art_pattern) == 0, f"文件 {doc_path.name}:{line_no} 包含 ASCII Art 边框: '{line}'"

    # 严格检验 rule.md 禁用短语
    @pytest.mark.parametrize("doc_path", DOC_PATHS + [CANVAS_PATH])
    def test_zero_forbidden_phrases(self, doc_path: Path):
        lines = doc_path.read_text(encoding="utf-8").splitlines()

        for line_no, line in enumerate(lines, 1):
            # 禁止 "不是...而是..."
            if "不是" in line and "而是" in line:
                raise AssertionError(f"文件 {doc_path.name}:{line_no} 出现禁用句式 '不是...而是...': '{line.strip()}'")

            # 禁止黑话词汇
            for jargon in ("落地", "钉死", "对齐"):
                if jargon in line:
                    raise AssertionError(f"文件 {doc_path.name}:{line_no} 出现禁用黑话 '{jargon}': '{line.strip()}'")

            # 禁止技术栈/模型栈/单独的栈
            if "栈" in line:
                raise AssertionError(f"文件 {doc_path.name}:{line_no} 出现禁用字 '栈': '{line.strip()}'")


class TestPlatformEnvelopeAndTelemetry:
    # 验证 Wasm 确定性游戏引擎散列契约
    def test_wasm_engine_digest_invariant(self):
        assert WASM_PATH.is_file(), f"底层引擎不存在: {WASM_PATH}"
        data = WASM_PATH.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        assert digest == EXPECTED_WASM_DIGEST, f"底层引擎散列被篡改: {digest} != {EXPECTED_WASM_DIGEST}"

    # 验证模型规格包大小平台门禁
    @pytest.mark.parametrize(
        "model_dir,max_bytes",
        [
            ("models/exp033", 128 * 1024),
            ("models/exp033_iter7", 128 * 1024),
            ("models/exp033_iter8", 128 * 1024),
            ("models/exp031", 12800),  # 12.8 KB = 13,107 字节
        ],
    )
    def test_model_size_envelope(self, model_dir: str, max_bytes: int):
        target = ROOT / model_dir / "model.onnx"
        assert target.is_file(), f"模型文件不存在: {target}"
        size = target.stat().st_size
        assert size < max_bytes, f"模型文件体积超限: {target} ({size} >= {max_bytes})"

    # 验证过程遥测诊断记录覆盖完整 8 轮交互
    def test_dagger_telemetry_json_integrity(self):
        telemetry_file = ROOT / "runs/exp033/dagger_telemetry.json"
        assert telemetry_file.is_file(), f"遥测文件不存在: {telemetry_file}"
        data = json.loads(telemetry_file.read_text(encoding="utf-8"))

        assert data.get("baseline_signatures_d0") == 77171, "基准状态指纹数量不符合契约 77171"
        assert "disagreement_rates" in data, "遥测数据缺少 disagreement_rates"
        assert "expansion_factors" in data, "遥测数据缺少 expansion_factors"
        assert "cumulative_signatures" in data, "遥测数据缺少 cumulative_signatures"

        # 检验 8 轮迭代明细
        for k in range(1, 9):
            key = f"k{k}"
            assert key in data, f"缺少第 {k} 轮迭代记录: {key}"
            entry = data[key]
            assert entry["iteration"] == k
            assert entry["rows"] == 83333
            assert "disagreement_rate" in entry
            assert 0.0 < entry["disagreement_rate"] < 0.20
            d_key = f"d{k}"
            assert entry["disagreement_rate"] == data["disagreement_rates"][d_key]
            assert data["cumulative_signatures"][key] >= 77171
            assert data["expansion_factors"][f"e{k}"] > 1.0


class TestInflectionMathAndModelValidation:
    # 验证 Kneedle 曲率极大值数学闭式解
    def test_kneedle_inflection_math(self):
        # 路线 B 8 轮累计样本规模 (seat-turns)
        x_pts = [250000 + k * 83333 for k in range(1, 9)]
        # 路线 B 8 轮对弈综合得分率 (来自 Table 1)
        y_pts = [0.5222, 0.6019, 0.6222, 0.5889, 0.6481, 0.5296, 0.5963, 0.6574]

        x_min, x_max = x_pts[0], x_pts[-1]
        y_min, y_max = y_pts[0], y_pts[-1]

        # 归一化基准弦垂直距离 D_k = y_norm - x_norm
        d_values = []
        for x, y in zip(x_pts, y_pts):
            x_norm = (x - x_min) / (x_max - x_min)
            y_norm = (y - y_min) / (y_max - y_min)
            d_values.append(y_norm - x_norm)

        # 验证全局极大值拐点在 k=3 (500k 样本) 处取得
        max_idx = max(range(len(d_values)), key=lambda i: d_values[i])
        assert max_idx == 2, f"全局极大值拐点未在 k=3 处取得: idx {max_idx}"
        assert abs(d_values[2] - 0.4539) < 0.005, f"D_3 理论数值计算不符: {d_values[2]} != 0.4539"

        # 验证次级平台极大值在 k=5 (667k 样本) 处取得
        assert abs(d_values[4] - 0.3598) < 0.005, f"D_5 理论数值计算不符: {d_values[4]} != 0.3598"

    # 验证一阶离散边际导数与衰减倍率计算
    def test_discrete_marginal_derivative_decay(self):
        # 前 3 轮平均导数: (y_3 - y_1) / (x_3 - x_1) * 100k
        delta_y_early = 0.6222 - 0.5222  # +10.00%
        delta_x_early_100k = (500000 - 333333) / 100000  # 1.66667
        deriv_early = delta_y_early / delta_x_early_100k * 100  # % / 100k
        assert abs(deriv_early - 6.00) < 0.05, f"前 3 轮边际导数计算不符: {deriv_early}"

        # 后 5 轮平均导数: (y_8 - y_3) / (x_8 - x_3) * 100k
        delta_y_late = 0.6574 - 0.6222  # +3.52%
        delta_x_late_100k = (916664 - 500000) / 100000  # 4.16664
        deriv_late = delta_y_late / delta_x_late_100k * 100  # % / 100k
        assert abs(deriv_late - 0.84) < 0.05, f"后 5 轮边际导数计算不符: {deriv_late}"

        # 边际效率衰减倍率
        decay_ratio = deriv_early / deriv_late
        assert abs(decay_ratio - 7.1) < 0.2, f"边际效率衰减倍率计算不符: {decay_ratio}"

    # 验证 ONNX 模型结构与合法性
    @pytest.mark.parametrize("model_name", ["exp033", "exp033_iter7", "exp033_iter8", "exp031"])
    def test_onnx_model_validity(self, model_name: str):
        import onnx
        onnx_file = ROOT / "models" / model_name / "model.onnx"
        assert onnx_file.is_file(), f"ONNX 模型缺失: {onnx_file}"
        model = onnx.load(str(onnx_file))
        onnx.checker.check_model(model)
        assert len(model.graph.input) > 0, "ONNX 计算图缺少输入节点"
        assert len(model.graph.output) > 0, "ONNX 计算图缺少输出节点"

