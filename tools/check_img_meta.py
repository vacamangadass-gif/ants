from pathlib import Path
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMA_DIR = PROJECT_ROOT / "tinybrain" / "ima"

imgs = [
    IMA_DIR / "exp085_multiseat_tournament.png",
    IMA_DIR / "exp085_h2h_comparison.png",
]

for img_path in imgs:
    assert img_path.is_file(), f"图片缺失: {img_path}"
    size = img_path.stat().st_size
    with Image.open(img_path) as im:
        dpi = im.info.get("dpi")
        print(f"[{img_path.name}] 体积: {size} 字节 ({size/1024:.1f} KB), 尺寸: {im.size}, 模式: {im.mode}, DPI: {dpi}, 格式: {im.format}")
