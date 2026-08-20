#!/usr/bin/env python3
"""Contact sheets for the quality-lever matrix.
Per ROI: rows = conditions (A baseline / B neg / C pos / D both), cols = seeds.
Also one big overview: rows = ROIs, cols = the 4 conditions (seed 0)."""
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

SRC = Path("outputs/matrix")
OUT = Path("outputs/matrix/contact")
OUT.mkdir(parents=True, exist_ok=True)

ROIS = ["faces", "bodies", "words", "places", "earlyvisual"]
CONDS = [("A", "baseline (empty)"), ("B", "neg-prompt"),
         ("C", "quality-prompt"), ("D", "both")]
SEEDS = [0, 1]
TH = 256          # thumb size
PADL, PADT = 150, 26   # left label gutter, top label strip

def load(cond, roi, seed):
    p = SRC / f"{cond}_{roi}_seed{seed}_0.png"
    if not p.exists():
        return Image.new("RGB", (TH, TH), (40, 40, 40))
    im = Image.open(p).convert("RGB").resize((TH, TH))
    # flag blank/near-black (failed) tiles with a red border
    if p.stat().st_size < 2000:
        d = ImageDraw.Draw(im); d.rectangle([0, 0, TH-1, TH-1], outline=(220, 0, 0), width=6)
    return im

def sheet(cols_labels, rows_labels, cell, title, path):
    ncol, nrow = len(cols_labels), len(rows_labels)
    W = PADL + ncol * TH
    H = PADT + 20 + nrow * TH
    canvas = Image.new("RGB", (W, H), (245, 245, 245))
    d = ImageDraw.Draw(canvas)
    d.text((6, 4), title, fill=(0, 0, 0))
    for c, cl in enumerate(cols_labels):
        d.text((PADL + c * TH + 6, PADT + 2), cl, fill=(0, 0, 0))
    y0 = PADT + 20
    for r, rl in enumerate(rows_labels):
        d.text((6, y0 + r * TH + TH // 2), rl, fill=(0, 0, 0))
        for c in range(ncol):
            canvas.paste(cell(r, c), (PADL + c * TH, y0 + r * TH))
    canvas.save(path)
    print("wrote", path)

# per-ROI sheets: rows=conditions, cols=seeds
for roi in ROIS:
    sheet([f"seed {s}" for s in SEEDS],
          [f"{c}:{name}" for c, name in CONDS],
          lambda r, c: load(CONDS[r][0], roi, SEEDS[c]),
          f"ROI = {roi}   (rows: levers, cols: seeds)",
          OUT / f"roi_{roi}.png")

# overview: rows=ROIs, cols=conditions (seed 0)
sheet([name for _, name in CONDS], ROIS,
      lambda r, c: load(CONDS[c][0], ROIS[r], 0),
      "Overview  rows=ROI, cols=lever (seed 0)",
      OUT / "overview_seed0.png")
sheet([name for _, name in CONDS], ROIS,
      lambda r, c: load(CONDS[c][0], ROIS[r], 1),
      "Overview  rows=ROI, cols=lever (seed 1)",
      OUT / "overview_seed1.png")
