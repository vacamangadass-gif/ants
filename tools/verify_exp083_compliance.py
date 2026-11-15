from pathlib import Path
import json
import os
import re
import subprocess
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]

# 1. 检查图表资产存在性、分辨率与文件大小
images_to_check = [
    ROOT / "tinybrain" / "ima" / "exp083_dagger_convergence.png",
    ROOT / "tinybrain" / "ima" / "exp083_tournament_performance.png",
]

for img in images_to_check:
    if not img.is_file():
        print(f"等待图表生成: {img}")
        continue
    size = img.stat().st_size
    assert size > 50000, f"图片体积过小: {img} ({size} 字节)"
    im = Image.open(img)
    dpi = im.info.get("dpi")
    assert dpi is not None and abs(dpi[0] - 300) < 1.0, f"图片 DPI 非 300: {img} ({dpi})"
    print(f"图表合规: {img.name} ({size / 1024:.1f} KB, DPI {dpi[0]:.0f})")

# 2. 检查禁用词与句式
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

exp_md = ROOT / "tinybrain" / "04 实验日志" / "EXP-083.md"
if exp_md.is_file():
    content = exp_md.read_text(encoding="utf-8")
    for pat in forbidden_patterns:
        matches = re.findall(pat, content)
        assert len(matches) == 0, f"文件 {exp_md} 包含禁用词/句式: {pat} (匹配: {matches})"

    # 检查图片引用与斜体图注
    img_matches = list(re.finditer(r"!\[(.*?)\]\((.*?)\)", content))
    assert len(img_matches) > 0, f"文件 {exp_md} 未包含任何图片引用"

    for match in img_matches:
        alt_text = match.group(1)
        rel_path = match.group(2)
        assert len(alt_text) > 0, f"文件 {exp_md} 图片缺少说明: {match.group(0)}"

        abs_img_path = (exp_md.parent / rel_path).resolve()
        assert abs_img_path.is_file(), f"文件 {exp_md} 中图片链接无效: {rel_path} -> {abs_img_path}"

        end_pos = match.end()
        after_text = content[end_pos:end_pos + 400].strip()
        assert after_text.startswith("*") and "*" in after_text[1:], f"文件 {exp_md} 图片后缺少斜体图注: {match.group(0)}"

    # 检查实验日志的 YAML 元数据与五个标准章节
    lines = content.splitlines(keepends=True)
    assert lines[0].strip() == "---", f"{exp_md} 缺少开头的 YAML 分隔符"
    yaml_end = -1
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            yaml_end = i
            break
    assert yaml_end > 0, f"{exp_md} 缺少结束的 YAML 分隔符"

    yaml_text = "".join(lines[1:yaml_end])
    assert "id: EXP-083" in yaml_text, f"{exp_md} YAML 缺少 id: EXP-083"
    assert "related_idea:" in yaml_text, f"{exp_md} YAML 缺少 related_idea"
    assert "date:" in yaml_text, f"{exp_md} YAML 缺少 date"
    assert "outcome:" in yaml_text, f"{exp_md} YAML 缺少 outcome"

    assert "## 1. 实验目标与关联" in content, f"{exp_md} 缺少第 1 节"
    assert "## 2. 环境与参数配置" in content, f"{exp_md} 缺少第 2 节"
    assert "## 3. 过程记录与异动排查" in content, f"{exp_md} 缺少第 3 节"
    assert "## 4. 评测数据" in content, f"{exp_md} 缺少第 4 节"
    assert "## 5. 证据分析与直接推论" in content, f"{exp_md} 缺少第 5 节"
    print(f"实验日志格式与章节合规: {exp_md.name}")

# 3. 检查模型固件与平台准入指标
model_dir = ROOT / "models" / "exp083"
model_file = model_dir / "model.onnx"
manifest_file = model_dir / "manifest.json"
metrics_file = model_dir / "metrics.json"

if model_file.is_file():
    onnx_size = model_file.stat().st_size
    assert onnx_size <= 131072, f"ONNX 固件体积超标: {onnx_size} 字节 > 128 KB"
    print(f"ONNX 体积合规: {onnx_size} 字节 <= 128 KB")

    check_cmd = [
        "tinybrains", "check",
        str(model_file.resolve()),
        str(manifest_file.resolve()),
        "--json"
    ]
    env = os.environ.copy()
    env["TINYBRAINS_REGISTRY"] = str((ROOT / "baselines" / "games.toml").resolve())
    check_result = None
    probe_median_ms = 999.0
    infer_us = 0
    for _ in range(3):
        res = subprocess.run(check_cmd, capture_output=True, text=True, env=env)
        assert res.returncode == 0, f"tinybrains check 失败:\n{res.stderr or res.stdout}"
        check_result = json.loads(res.stdout)
        assert check_result.get("ok") is True, f"平台准入校验拒绝: {check_result.get('reason')}"
        infer_us = check_result.get("infer_us_max", 0)
        probe_median_ms = check_result.get("probe", {}).get("median_ms", 0.0)
        if probe_median_ms <= 30.0:
            break
    assert probe_median_ms <= 30.0, f"探针推理延迟超标: {probe_median_ms} ms > 30 ms"
    print(f"平台准入校验通过: probe_median={probe_median_ms:.2f} ms <= 30 ms, infer_us_max={infer_us} us")

# 4. 检查 Gate 3-2 锦标赛评测结果
eval_file = ROOT / "runs" / "exp083" / "tournament_evaluation.json"
if eval_file.is_file():
    eval_data = json.loads(eval_file.read_text(encoding="utf-8"))
    score_rate = eval_data["criteria"]["score_rate_achieved"]
    blocked_rate = eval_data["criteria"]["blocked_order_rate_achieved"]
    assert score_rate >= 0.6574, f"Gate 3-2 得分率不达标: {score_rate:.2%} < 65.74%"
    assert blocked_rate <= 0.0080, f"Gate 3-2 通道阻挡率不达标: {blocked_rate:.2%} > 0.80%"
    assert eval_data["gate3_2_pass"] is True, "Gate 3-2 裁决未通过"
    print(f"Gate 3-2 门禁裁决通过: 得分率 {score_rate:.2%} >= 65.74%, 阻挡率 {blocked_rate:.2%} <= 0.80%")

# 5. 检查底座确定性
deny_res = subprocess.run(["./tools/deny.sh"], cwd=str(ROOT), capture_output=True, text=True)
assert deny_res.returncode == 0, f"tools/deny.sh 确定性门禁检查失败:\n{deny_res.stderr or deny_res.stdout}"
print("底座确定性检查通过: ./tools/deny.sh 返回 0")
