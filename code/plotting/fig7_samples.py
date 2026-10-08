"""Figure 7: qualitative sample grids for conditional image generation.

Assembles the eight stored sample grids in `output/fig7_samples/panels.npz`.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
PANELS_NPZ = PACKAGE_ROOT / "output" / "fig7_samples" / "panels.npz"
OUT_PNG = PACKAGE_ROOT / "output" / "figures" / "fig7_tds_samples.png"

CANVAS = (3226, 1251)
PANEL = (586, 586)
XS = [477, 1197, 1917, 2638]
YS = [3, 661]

LAYOUT = [
    ["vanilla_mnist3", "vanilla_mnist7", "vanilla_cifar_airplane", "vanilla_cifar_cat"],
    ["twdr_mnist3", "twdr_mnist7", "twdr_cifar_airplane", "twdr_cifar_cat"],
]
ROW_LABELS = ["Vanilla TDS", "TDS-T-WDR"]
LABEL_MAX_SIZE = 60


def fit_font(draw, texts, max_width):
    """The largest default face whose widest label fits the band."""
    for size in range(LABEL_MAX_SIZE, 7, -1):
        font = ImageFont.load_default(size=size)
        if max(draw.textlength(text, font=font) for text in texts) <= max_width:
            return font
    return ImageFont.load_default()


def draw_row_labels(canvas):
    """One label per row, centred in the band left of the first column."""
    draw = ImageDraw.Draw(canvas)
    font = fit_font(draw, ROW_LABELS, 0.88 * XS[0])
    for y, text in zip(YS, ROW_LABELS):
        left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
        draw.text((0.5 * (XS[0] - (right - left)) - left,
                   y + 0.5 * (PANEL[1] - (bottom - top)) - top),
                  text, fill="black", font=font)


def main() -> None:
    panels = np.load(PANELS_NPZ)
    canvas = Image.new("RGB", CANVAS, "white")
    draw_row_labels(canvas)
    for r, row in enumerate(LAYOUT):
        for c, name in enumerate(row):
            panel = Image.fromarray(panels[name])
            canvas.paste(panel.resize(PANEL, Image.BICUBIC), (XS[c], YS[r]))
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(OUT_PNG)
    print(f"wrote {OUT_PNG}")


if __name__ == "__main__":
    main()
