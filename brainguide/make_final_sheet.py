#!/usr/bin/env python3
"""Curation sheet: rows = ROI, cols = seeds 0-4. Baseline config."""
from pathlib import Path
from PIL import Image, ImageDraw

SRC = Path("outputs/final")
ROIS = ["faces", "bodies", "words", "places", "earlyvisual"]
SEEDS = [0, 1, 2, 3, 4]
TH = 300
PADL, PADT = 130, 26

def load(roi, seed):
    p = SRC / f"{roi}_seed{seed}_0.png"
    if not p.exists():
        return Image.new("RGB", (TH, TH), (40, 40, 40))
    im = Image.open(p).convert("RGB").resize((TH, TH))
    if p.stat().st_size < 2000:
        d = ImageDraw.Draw(im); d.rectangle([0, 0, TH-1, TH-1], outline=(220, 0, 0), width=6)
    return im

W = PADL + len(SEEDS) * TH
H = PADT + 20 + len(ROIS) * TH
canvas = Image.new("RGB", (W, H), (245, 245, 245))
d = ImageDraw.Draw(canvas)
d.text((6, 4), "Baseline (empty prompt, clip=0.30) - rows=ROI, cols=seed", fill=(0, 0, 0))
for c, s in enumerate(SEEDS):
    d.text((PADL + c * TH + 6, PADT + 2), f"seed {s}", fill=(0, 0, 0))
y0 = PADT + 20
for r, roi in enumerate(ROIS):
    d.text((6, y0 + r * TH + TH // 2), roi, fill=(0, 0, 0))
    for c, s in enumerate(SEEDS):
        canvas.paste(load(roi, s), (PADL + c * TH, y0 + r * TH))
out = SRC / "curation_sheet.png"
canvas.save(out)
print("wrote", out)
