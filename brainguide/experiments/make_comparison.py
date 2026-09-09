#!/usr/bin/env python3
"""One comparison contact sheet across baseline + plans 1/2/4.
Rows = ROIs, columns = methods. Reads from ./pulled/.

  pulled/baseline/sub1_<roi>_seed0_0.png   # universal-encoder baseline (DPM)
  pulled/test1/t1_<roi>_seed0_0.png        # plan 1: gradient hygiene
  pulled/test2/t2ctrl_<roi>_seed0_0.png    # plan 2: DDIM control (reps=1)
  pulled/test2/t2tt_<roi>_seed0_0.png      # plan 2: DDIM time-travel (reps=4)
  pulled/test4/t4_<roi>_seed0_0.png        # plan 4: DINO->CLIP bridge + unCLIP
"""
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

BASE = Path(__file__).resolve().parent / "pulled"
OUT = Path(__file__).resolve().parent / "comparison.png"

ROIS = ["faces", "places", "words", "bodies", "earlyvisual"]
COLS = [
    ("baseline (DPM)",     "baseline/sub1_{roi}_seed0_0.png"),
    ("plan1 hygiene",      "test1/t1_{roi}_seed0_0.png"),
    ("plan2 ctrl (DDIM)",  "test2/t2ctrl_{roi}_seed0_0.png"),
    ("plan2 time-travel",  "test2/t2tt_{roi}_seed0_0.png"),
    ("plan4 bridge+IP-Adapter","test4/t4_{roi}_seed0_0.png"),
]
TH = 256
PADL, PADT = 150, 48


def _font(sz):
    for p in ("/System/Library/Fonts/Supplemental/Arial.ttf",
              "/System/Library/Fonts/Helvetica.ttc"):
        try:
            return ImageFont.truetype(p, sz)
        except Exception:
            pass
    return ImageFont.load_default()


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
    canvas = Image.new("RGB", (W, H), (250, 250, 250))
    d = ImageDraw.Draw(canvas)
    f_title, f_lbl = _font(15), _font(14)
    d.text((8, 12), "brainguide — universal (DINOv2) encoder: plans 1/2/4 vs baseline (seed 0)",
           fill=(0, 0, 0), font=f_title)
    for c, (label, _) in enumerate(COLS):
        d.text((PADL + c * TH + 6, PADT - 18), label, fill=(0, 0, 0), font=f_lbl)
    for r, roi in enumerate(ROIS):
        d.text((8, PADT + r * TH + TH // 2 - 6), roi, fill=(0, 0, 0), font=f_lbl)
        for c, (_, tmpl) in enumerate(COLS):
            canvas.paste(load(tmpl.format(roi=roi)), (PADL + c * TH, PADT + r * TH))
    canvas.save(OUT)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
