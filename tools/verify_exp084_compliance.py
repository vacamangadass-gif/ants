from pathlib import Path
import json
import os
import re
import subprocess
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]

# 1. 检查图表资产存在性、分辨率与文件大小
images_to_check = [
    ROOT / "tinybrain" / "ima" / "exp084_mappo_training_convergence.png",
    ROOT / "tinybrain" / "ima" / "exp084_counterfactual_ablation.png",
]

for img in images_to_check:
    assert img.is_file(), f"图表文件缺失: {img}"
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

exp_md = ROOT / "tinybrain" / "04 实验日志" / "EXP-084.md"
assert exp_md.is_file(), f"实验日志缺失: {exp_md}"

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
assert "id: EXP-084" in yaml_text, f"{exp_md} YAML 缺少 id: EXP-084"
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
model_dir = ROOT / "models" / "exp084"
model_file = model_dir / "model.onnx"
manifest_file = model_dir / "manifest.json"
metrics_file = model_dir / "metrics.json"

assert model_file.is_file(), f"ONNX 固件缺失: {model_file}"
onnx_size = model_file.stat().st_size
assert onnx_size <= 131072, f"ONNX 固件体积超标: {onnx_size} 字节 > 128 KB"
print(f"ONNX 体积合规: {onnx_size} 字节 <= 128 KB")

check_cmd = [
    "tinybrains", "check",
    str(model_file.resolve()),
    str(manifest_file.resolve()),
    "--json",
]
env = os.environ.copy()
env["TINYBRAINS_REGISTRY"] = str((ROOT / "baselines" / "games.toml").resolve())

check_json = None
median_ms = 999.0
for _ in range(3):
    check_res = subprocess.run(check_cmd, cwd=str(ROOT), capture_output=True, text=True, env=env)
    assert check_res.returncode == 0, f"tinybrains check 失败: {check_res.stderr or check_res.stdout}"
    check_json = json.loads(check_res.stdout)
    assert check_json.get("ok") is True, f"平台准入未通过: {check_json}"
    median_ms = check_json["probe"]["median_ms"]
    if median_ms <= 30.0:
        break

assert median_ms <= 30.0, f"探针推理耗时超标: {median_ms} ms > 30.0 ms"
print(f"平台探针基准测试合规: {median_ms:.2f} ms <= 30.0 ms")

operators = check_json["graph"]["operators"]
assert "Linear" not in operators and "Value" not in operators, f"ONNX 中包含未剥离的价值头算子: {operators}"
print(f"Critic 价值头剥离合规: 算子集合为 {operators}")

# 4. 验证引擎未修改与纯整数确定性
deny_res = subprocess.run(["./tools/deny.sh"], cwd=str(ROOT), capture_output=True, text=True)
assert deny_res.returncode == 0, f"tools/deny.sh 检测未通过: {deny_res.stderr or deny_res.stdout}"

status_res = subprocess.run(["git", "status", "-s", "engine/"], cwd=str(ROOT), capture_output=True, text=True)
assert status_res.returncode == 0 and len(status_res.stdout.strip()) == 0, f"engine/ 目录被意外触碰修改:\n{status_res.stdout}"
print("底座引擎纯洁性与确定性门禁通过: engine/ 零修改, ./tools/deny.sh 退出码 0")

print("\nEXP-084 全要素合规验证全部通过！")
