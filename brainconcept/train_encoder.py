#!/usr/bin/env python
"""Stage 1 — train the fMRI encoder W:  CLIP ViT-L/14 image embedding -> fMRI.

W maps an L2-normalized CLIP ViT-L/14 image embedding (768-d) to per-voxel fMRI
response over ALL functional voxels (~39,548 for S1). That 768-d embedding is
EXACTLY what Kandinsky 2.1's prior outputs, so the trained W drops straight into
the brain objective as  -mean_ROI(W . z).

Design (per the plan): we train ONE general image->brain encoder on ALL voxels.
Ridge is per-voxel independent, so this is a single solve. ROIs are NOT used to
restrict training — they are only applied LATER (Stage 2) to select which rows
of W we average. Here we use the ROI index files only to REPORT a per-ROI metric
breakdown, and we save them alongside W for Stage 2.

Pipeline:
    NSD stimulus images -(CLIP ViT-L/14, frozen)-> 768-d emb -(L2norm)-> X
    NSD fMRI betas (all voxels) ---------------------------------------> Y
    RidgeCV(alpha grid) on train -> validate (median Pearson r) -> test -> save W

Deliverables (proof the encoder is solid):
    - test median voxel Pearson r + full distribution
    - noise-ceiling-normalized r (r / sqrt(nc)), fraction of voxels beating floor
    - per-ROI median r table (faces/bodies/places/words/earlyvisual)
    - predicted-vs-actual scatter for top voxels
    - metrics.json + PNG plots + W_s1.pt

Run on the VM (needs a GPU for CLIP). Defaults match prep_s1.py's outputs.

    python train_encoder.py \
        --data_dir data \
        --roi_dir ../brainguide/brainguide/roi \
        --out weights/W_s1.pt
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

ROI_NAMES = ["faces", "bodies", "places", "words", "earlyvisual"]


def embed_images_clip(images_uint8, device, batch=64):
    """images_uint8: [N,H,W,3] uint8 -> [N,768] L2-normed CLIP ViT-L/14 image embeddings.

    Uses OpenAI CLIP 'ViT-L/14' weights via open_clip (identical vector space to
    Kandinsky 2.1's prior target) — avoids the openai/CLIP git dependency.
    """
    import open_clip
    from PIL import Image
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-L-14", pretrained="openai")
    model = model.to(device).eval()
    feats = []
    with torch.no_grad():
        for i in range(0, len(images_uint8), batch):
            chunk = images_uint8[i:i + batch]
            px = torch.stack([preprocess(Image.fromarray(im)) for im in chunk]).to(device)
            emb = model.encode_image(px).float()
            emb = emb / emb.norm(p=2, dim=-1, keepdim=True)   # L2-normalize (matches the prior)
            feats.append(emb.cpu())
    return torch.cat(feats).numpy()


def _pearson_cols(A, B):
    A = A - A.mean(0, keepdims=True); B = B - B.mean(0, keepdims=True)
    num = (A * B).sum(0)
    den = np.sqrt((A**2).sum(0) * (B**2).sum(0)) + 1e-8
    return num / den


def ridgecv_fit(X_tr, Y_tr, X_va, Y_va, alphas):
    """One RidgeCV over ALL voxels. Returns (W [V,768], best_alpha, val_r [V])."""
    from sklearn.linear_model import Ridge
    best = None
    for a in alphas:
        reg = Ridge(alpha=a, fit_intercept=False)
        reg.fit(X_tr, Y_tr)                        # coef_: [V,768]
        r = _pearson_cols(X_va @ reg.coef_.T, Y_va)
        score = np.nanmedian(r)
        print(f"  alpha={a:>9.3g}  val median r={score:.4f}", flush=True)
        if best is None or score > best[0]:
            best = (score, a, reg.coef_.astype(np.float32), r)
    _, alpha, W, r = best
    print(f"[ridge] best alpha={alpha:g}  val median r={np.nanmedian(r):.4f}")
    return W, alpha, r


def load_data(args):
    """Load betas (all voxels), images, split, ncsnr, and the ROI index dict."""
    d = Path(args.data_dir)
    images = np.load(d / "s1_images.npy")                    # [N,256,256,3] uint8
    betas = np.load(d / "s1_betas.npy").astype(np.float32)   # [N, V_all]
    sp = np.load(d / "s1_split.npz")
    ncsnr = np.load(d / "s1_ncsnr.npy") if (d / "s1_ncsnr.npy").exists() else None
    roi = {}
    rd = Path(args.roi_dir)
    for name in ROI_NAMES:
        f = rd / f"sub1_{name}.npy"
        if f.exists():
            roi[name] = np.load(f).astype(np.int64)
    return images, betas, ncsnr, roi, sp["train_idx"], sp["val_idx"], sp["test_idx"]


def noise_ceiling(ncsnr, n_reps=3):
    """NSD noise ceiling as a fraction of variance: nc = ncsnr^2/(ncsnr^2 + 1/n).
    Max achievable Pearson r for repeat-averaged responses ~ sqrt(nc)."""
    nc = ncsnr**2 / (ncsnr**2 + 1.0 / n_reps)
    return nc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--roi_dir", default="../brainguide/brainguide/roi")
    ap.add_argument("--out", default="weights/W_s1.pt")
    ap.add_argument("--report_dir", default="report")
    ap.add_argument("--alphas", default="100,1000,5000,10000,50000,100000,500000")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    alphas = [float(a) for a in args.alphas.split(",")]

    images, Y, ncsnr, roi, tr, va, te = load_data(args)
    V = Y.shape[1]
    print(f"[data] images={images.shape} betas={Y.shape} "
          f"| train={len(tr)} val={len(va)} test={len(te)} | ROIs={list(roi)}", flush=True)

    print("[clip] embedding images with OpenAI ViT-L/14 ...", flush=True)
    X = embed_images_clip(images, device)                   # [N,768]

    W, alpha, val_r = ridgecv_fit(X[tr], Y[tr], X[va], Y[va], alphas)

    # ---- held-out test (shared-1000) ----
    test_pred = X[te] @ W.T
    test_r = _pearson_cols(test_pred, Y[te])
    med = float(np.nanmedian(test_r))
    print(f"[test] median voxel Pearson r = {med:.4f}", flush=True)

    metrics = {"alpha": alpha, "n_voxels": int(V),
               "n_train": int(len(tr)), "n_val": int(len(va)), "n_test": int(len(te)),
               "test_median_r": med,
               "test_mean_r": float(np.nanmean(test_r)),
               "frac_voxels_r_gt_0.1": float(np.mean(test_r > 0.1)),
               "frac_voxels_r_gt_0.2": float(np.mean(test_r > 0.2))}

    # ---- noise ceiling ----
    nc = None
    if ncsnr is not None:
        nc = noise_ceiling(ncsnr)                            # fraction of variance
        ceil_r = np.sqrt(np.clip(nc, 1e-6, 1.0))             # max achievable r
        well_defined = nc > 0.05                             # voxels with real signal
        norm_r = test_r[well_defined] / ceil_r[well_defined]
        metrics["noise_ceiling_median_frac"] = float(np.nanmedian(nc[well_defined]))
        metrics["normalized_r_median"] = float(np.nanmedian(norm_r))
        metrics["frac_voxels_signal(nc>0.05)"] = float(np.mean(well_defined))
        print(f"[ceiling] median normalized r (r/ceiling) over signal voxels = "
              f"{metrics['normalized_r_median']:.3f}  "
              f"(n_signal={int(well_defined.sum())})", flush=True)

    # ---- per-ROI breakdown ----
    roi_table = {}
    for name, idx in roi.items():
        rr = test_r[idx]
        entry = {"n": int(len(idx)), "median_r": float(np.nanmedian(rr)),
                 "mean_r": float(np.nanmean(rr))}
        if nc is not None:
            cr = np.sqrt(np.clip(nc[idx], 1e-6, 1.0))
            entry["normalized_r_median"] = float(np.nanmedian(rr / cr))
        roi_table[name] = entry
        print(f"[roi] {name:12s} n={entry['n']:6d}  median r={entry['median_r']:.3f}"
              + (f"  norm_r={entry.get('normalized_r_median', float('nan')):.3f}"
                 if nc is not None else ""), flush=True)
    metrics["per_roi"] = roi_table

    # ---- save artifacts ----
    outp = Path(args.out); outp.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"W": torch.from_numpy(W), "roi_idx": roi, "alpha": alpha,
                "val_r": val_r, "test_r": test_r,
                "ncsnr": ncsnr, "clip": "ViT-L/14", "l2norm": True}, outp)
    rep = Path(args.report_dir); rep.mkdir(parents=True, exist_ok=True)
    with open(rep / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    np.save(rep / "test_r.npy", test_r)
    _plots(rep, test_r, nc, test_pred, Y[te], roi_table)
    print(f"[done] saved {outp}  (W {W.shape}); report in {rep}/", flush=True)
    print(json.dumps(metrics, indent=2))


def _plots(rep, test_r, nc, pred, actual, roi_table):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[plots] skipped ({e})"); return
    # 1) histogram of test r
    plt.figure(figsize=(6, 4))
    plt.hist(test_r[~np.isnan(test_r)], bins=80, color="#4576b5")
    plt.axvline(np.nanmedian(test_r), color="k", ls="--",
                label=f"median={np.nanmedian(test_r):.3f}")
    plt.xlabel("test voxel Pearson r"); plt.ylabel("# voxels"); plt.legend()
    plt.title("Encoder test accuracy (all voxels)"); plt.tight_layout()
    plt.savefig(rep / "hist_test_r.png", dpi=120); plt.close()
    # 2) per-ROI bar
    if roi_table:
        plt.figure(figsize=(6, 4))
        names = list(roi_table); vals = [roi_table[n]["median_r"] for n in names]
        plt.bar(names, vals, color="#59a14f")
        plt.ylabel("median test r"); plt.title("Per-ROI encoder accuracy")
        plt.xticks(rotation=20); plt.tight_layout()
        plt.savefig(rep / "per_roi_r.png", dpi=120); plt.close()
    # 3) pred-vs-actual scatter for the single best voxel
    best = int(np.nanargmax(test_r))
    plt.figure(figsize=(4.5, 4.5))
    plt.scatter(actual[:, best], pred[:, best], s=6, alpha=0.4)
    plt.xlabel("actual response"); plt.ylabel("predicted")
    plt.title(f"Best voxel #{best}  r={test_r[best]:.3f}"); plt.tight_layout()
    plt.savefig(rep / "scatter_best_voxel.png", dpi=120); plt.close()
    print(f"[plots] wrote hist_test_r.png, per_roi_r.png, scatter_best_voxel.png")


if __name__ == "__main__":
    main()
