from pathlib import Path
from PIL import Image
import numpy as np
import pytest

IMAGE_DIR = Path("/home/host/project/ants/tinybrain/ima")

TARGET_IMAGES = [
    "dagger_iterations_evolution.png",
    "dagger_samples_marginal_benefit.png",
    "exp033_dagger_telemetry.png",
    "dagger_telemetry_comparison.png",
    "tournament_score_win_rates.png",
    "tournament_resilience_scatter.png",
    "tournament_pairwise_heatmap.png",
    "exp033_training_convergence.png",
    "exp030_exp031_training_convergence.png",
    "exp031_dagger_telemetry.png",
]

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def test_target_image_list_completeness():
    assert len(TARGET_IMAGES) == 10
    for name in TARGET_IMAGES:
        img_path = IMAGE_DIR / name
        assert img_path.exists(), f"目标图像缺失: {name}"
        assert img_path.is_file(), f"目标路径不是常规文件: {name}"


@pytest.mark.parametrize("image_name", TARGET_IMAGES)
def test_png_magic_bytes_and_file_size(image_name):
    img_path = IMAGE_DIR / image_name
    assert img_path.exists(), f"文件不存在: {image_name}"

    file_bytes = img_path.read_bytes()
    magic = file_bytes[:8]
    assert magic == PNG_MAGIC, f"{image_name} 的 PNG 魔数错误: 实际为 {magic!r}"

    # 验证文件体积大于 100 KB
    file_size_kb = len(file_bytes) / 1024
    assert file_size_kb >= 100.0, (
        f"{image_name} 文件体积过小: {file_size_kb:.2f} KB < 100 KB"
    )


@pytest.mark.parametrize("image_name", TARGET_IMAGES)
def test_image_mode_and_dimensions(image_name):
    img_path = IMAGE_DIR / image_name
    im = Image.open(img_path)

    # 校验 PIL 模式为 RGB 或 RGBA
    assert im.mode in ("RGB", "RGBA"), f"{image_name} 颜色模式不符合预期: {im.mode}"

    # 校验像素尺寸: 宽至少 2000px，高至少 1000px
    width, height = im.size
    assert width >= 2000, f"{image_name} 图像宽度不合规: {width}px < 2000px"
    assert height >= 1000, f"{image_name} 图像高度不合规: {height}px < 1000px"


@pytest.mark.parametrize("image_name", TARGET_IMAGES)
def test_image_dpi_strictly_300(image_name):
    img_path = IMAGE_DIR / image_name
    im = Image.open(img_path)

    dpi = im.info.get("dpi")
    assert dpi is not None, f"{image_name} 缺失 DPI 元数据"

    dpi_x = round(float(dpi[0]))
    dpi_y = round(float(dpi[1]))
    assert (dpi_x, dpi_y) == (300, 300), (
        f"{image_name} DPI 不符合要求: 实际为 {dpi}, 期望 (300, 300)"
    )


@pytest.mark.parametrize("image_name", TARGET_IMAGES)
def test_non_monochrome_pixel_entropy_and_variance(image_name):
    img_path = IMAGE_DIR / image_name
    im = Image.open(img_path)
    arr = np.asarray(im)

    # 提取有效色彩通道（忽略 RGBA 的 Alpha 通道若全为 255）
    if arr.ndim == 3 and arr.shape[2] >= 3:
        color_arr = arr[:, :, :3]
    else:
        color_arr = arr

    # 计算标准差，非单色画布的标准差必须显著大于 0
    std_val = float(np.std(color_arr))
    assert std_val >= 30.0, (
        f"{image_name} 色彩标准差过低 ({std_val:.2f})，疑似单色空白画布"
    )

    # 计算香农信息熵，排除全黑全白等单一像素值的零熵情况
    flat = color_arr.flatten()
    counts = np.bincount(flat)
    non_zero_probs = counts[counts > 0] / flat.size
    entropy = -float(np.sum(non_zero_probs * np.log2(non_zero_probs)))
    assert entropy >= 0.70, (
        f"{image_name} 像素熵过低 ({entropy:.2f} bits)，疑似单色空白画布"
    )

    # 独立色阶数量至少为 100 种
    unique_vals = np.unique(flat)
    assert len(unique_vals) >= 100, f"{image_name} 色阶种类过少 ({len(unique_vals)})"
