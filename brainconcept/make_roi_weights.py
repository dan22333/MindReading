#!/usr/bin/env python
"""Slice the full ridge encoder W_s1.pt into per-ROI weight files that coach.py
loads directly as brain_W (ckpt['W'] must be [num_roi_voxels, 768]).

W_s1.pt holds the full W [39548, 768] plus roi_idx = {name: indices}. This makes
one file per requested ROI: {'W': W[roi_idx[name]], 'roi': name, ...}. No retrain.

Usage (on the VM, from brainconcept/):
    python make_roi_weights.py --full weights/W_s1.pt --rois bodies places
"""
import argparse
from pathlib import Path
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", default="weights/W_s1.pt",
                    help="full ridge checkpoint with W [V,768] and roi_idx dict")
    ap.add_argument("--rois", nargs="+", default=["bodies", "places"])
    ap.add_argument("--out_dir", default="weights")
    args = ap.parse_args()

    ckpt = torch.load(args.full, map_location="cpu")
    W = torch.as_tensor(ckpt["W"], dtype=torch.float32)          # [V_all, 768]
    roi_idx = ckpt["roi_idx"]
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    for name in args.rois:
        if name not in roi_idx:
            raise KeyError(f"ROI '{name}' not in {list(roi_idx)}")
        idx = torch.as_tensor(roi_idx[name], dtype=torch.long)
        W_roi = W[idx].contiguous()                              # [V_roi, 768]
        out = out_dir / f"W_{name}_s1.pt"
        torch.save({"W": W_roi, "roi": name,
                    "alpha": ckpt.get("alpha"),
                    "clip": ckpt.get("clip", "ViT-L/14"),
                    "l2norm": ckpt.get("l2norm", True)}, out)
        print(f"[roi] {name:10s} W {tuple(W_roi.shape)} -> {out}")


if __name__ == "__main__":
    main()
