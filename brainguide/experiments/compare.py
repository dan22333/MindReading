#!/usr/bin/env python3
"""Build one comparison contact sheet across baseline + test1 + test2.
Rows = ROIs, columns = methods. Run after pulling VM results into ./pulled/.

Expected layout (see run_tests.sh + pull step):
  pulled/baseline/sub1_<roi>_seed0_0.png     # the "like we tried before" images
  pulled/test1/t1_<roi>_seed0_0.png          # gradient hygiene
  pulled/test2/t2ctrl_<roi>_seed0_0.png      # ddim control (reps=1)
  pulled/test2/t2tt_<roi>_seed0_0.png        # ddim time-travel (reps=4)
"""
from pathlib import Path
from PIL import Image, ImageDraw

BASE = Path(__file__).resolve().parent / "pulled"
OUT = Path(__file__).resolve().parent / "comparison.png"

ROIS = ["faces", "places", "words", "bodies", "earlyvisual"]
COLS = [
    ("baseline (DPM)",       "baseline/sub1_{roi}_seed0_0.png"),
    ("test1 hygiene",        "test1/t1_{roi}_seed0_0.png"),
    ("test2 ctrl (DDIM)",    "test2/t2ctrl_{roi}_seed0_0.png"),
    ("test2 time-travel",    "test2/t2tt_{roi}_seed0_0.png"),
]
TH = 256
PADL, PADT = 130, 30


def load(rel):
    p = BASE / rel
    if not p.exists():
        im = Image.new("RGB", (TH, TH), (35, 35, 35))
        ImageDraw.Draw(im).text((8, TH // 2), "missing", fill=(200, 120, 120))
        return im
    im = Image.open(p).convert("RGB").resize((TH, TH))
    if p.stat().st_size < 2000:   # blank/failed tile
        ImageDraw.Draw(im).rectangle([0, 0, TH - 1, TH - 1], outline=(220, 0, 0), width=6)
    return im


def main():
    ncol, nrow = len(COLS), len(ROIS)
    W, H = PADL + ncol * TH, PADT + nrow * TH
    canvas = Image.new("RGB", (W, H), (245, 245, 245))
    d = ImageDraw.Draw(canvas)
    d.text((6, 8), "brainguide — universal (DINOv2) encoder: fix comparison (seed 0)", fill=(0, 0, 0))
    for c, (label, _) in enumerate(COLS):
        d.text((PADL + c * TH + 6, PADT - 14), label, fill=(0, 0, 0))
    for r, roi in enumerate(ROIS):
        d.text((6, PADT + r * TH + TH // 2), roi, fill=(0, 0, 0))
        for c, (_, tmpl) in enumerate(COLS):
            canvas.paste(load(tmpl.format(roi=roi)), (PADL + c * TH, PADT + r * TH))
    canvas.save(OUT)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
