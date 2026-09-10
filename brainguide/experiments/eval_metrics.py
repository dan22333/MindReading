#!/usr/bin/env python
"""Step A — quantitative evaluation: selectivity matrix per method.

Scores every generated image with the FROZEN universal (DINOv2) brain encoder on
all 5 ROIs, and builds, per method, a 5x5 selectivity matrix:

    M[i][j] = mean predicted activation, over images generated FOR ROI i,
              measured ON ROI j's voxels.

A selective method has a dominant diagonal (ROI-i images drive ROI i, not others).
Because raw per-ROI activation scales differ, we z-score each column (ROI j) across
the method's own images, so the diagonal test is scale-fair.

  python experiments/eval_metrics.py --base exp_out --config configs/universal.yaml \
      --out exp_out/eval

Encoder is used ONLY as a score — its weights are never touched.
"""
import argparse, glob, json
from pathlib import Path
import numpy as np
import torch
from PIL import Image

import sys
BG_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BG_ROOT))
from run import build

ROIS = ["faces", "places", "words", "bodies"]
N = len(ROIS)
# method name -> (subdir, filename prefix)
METHODS = {
    "baseline":  ("baseline", "base"),
    "plan1":     ("test1", "t1"),
    "plan2ctrl": ("test2", "t2ctrl"),
    "plan2tt":   ("test2", "t2tt"),
    "plan4":     ("test4", "t4"),
    "readout":   ("test3", "t3"),
    "ddpo":      ("test5", "t5"),
}


def load01(paths, device):
    ims = []
    for p in paths:
        im = Image.open(p).convert("RGB").resize((224, 224))
        ims.append(torch.from_numpy(np.asarray(im)).float().permute(2, 0, 1) / 255.0)
    return torch.stack(ims).to(device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="exp_out", help="dir containing baseline/test1/test2/test4")
    ap.add_argument("--config", default="configs/universal.yaml")
    ap.add_argument("--out", default="exp_out/eval")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    base = Path(args.base)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    import yaml
    cfg = yaml.safe_load(Path(args.config).read_text())
    _, encoder, _, _ = build(cfg)
    roi_idx = {r: torch.as_tensor(np.load(BG_ROOT / "brainguide" / "roi" / f"sub1_{r}.npy").astype("int64"),
                                  device=device) for r in ROIS}

    @torch.no_grad()
    def score(imgs01, roi):                       # mean activation on ROI's voxels, per image
        vox = roi_idx[roi].unsqueeze(0).expand(imgs01.shape[0], -1)
        return encoder(imgs01, vox).mean(1).cpu().numpy()   # [B]

    report = {}
    for m, (sub, pre) in METHODS.items():
        # raw[i][j] activation, and keep per-image for z-scoring per column
        per_img = {i: [] for i in ROIS}           # per_img[roi_i] -> list of images
        for roi_i in ROIS:
            files = sorted(glob.glob(str(base / sub / f"{pre}_{roi_i}_seed*_0.png")))
            if not files:
                continue
            per_img[roi_i] = files
        if not any(per_img.values()):
            print(f"[eval] {m}: no images, skipping"); continue

        # score every image on every ROI j
        raw = np.zeros((N, N)); allcol = {j: [] for j in ROIS}  # for z-score
        rowvals = {i: {j: [] for j in ROIS} for i in ROIS}
        for i, roi_i in enumerate(ROIS):
            if not per_img[roi_i]:
                raw[i, :] = np.nan; continue
            imgs = load01(per_img[roi_i], device)
            for j, roi_j in enumerate(ROIS):
                s = score(imgs, roi_j)            # [n_imgs]
                rowvals[roi_i][roi_j] = s.tolist()
                raw[i, j] = float(s.mean())
                allcol[roi_j].extend(s.tolist())
        # z-score each column across this method's images
        mu = {j: float(np.mean(allcol[j])) for j in ROIS}
        sd = {j: float(np.std(allcol[j]) + 1e-8) for j in ROIS}
        zmat = np.zeros((N, N))
        for i, roi_i in enumerate(ROIS):
            for j, roi_j in enumerate(ROIS):
                zmat[i, j] = (raw[i, j] - mu[roi_j]) / sd[roi_j]
        # selectivity: fraction of rows whose diagonal is the argmax (z-space)
        hits = sum(1 for i in range(N) if not np.isnan(zmat[i]).any() and int(np.argmax(zmat[i])) == i)
        diag = float(np.nanmean([zmat[i, i] for i in range(N)]))
        offdiag = float(np.nanmean([zmat[i, j] for i in range(N) for j in range(N) if i != j]))
        report[m] = {"raw": raw.tolist(), "z": zmat.tolist(),
                     "diag_argmax_hits": hits, "diag_z_mean": diag,
                     "offdiag_z_mean": offdiag, "selectivity_contrast": diag - offdiag}
        print(f"[eval] {m:9s}  diag-argmax {hits}/{N}   diagZ {diag:+.2f}  "
              f"offdiagZ {offdiag:+.2f}  contrast {diag-offdiag:+.2f}")

    (out / "selectivity.json").write_text(json.dumps({"rois": ROIS, "methods": report}, indent=2))
    print(f"[eval] wrote {out/'selectivity.json'}")


if __name__ == "__main__":
    main()
