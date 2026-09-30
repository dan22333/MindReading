#!/usr/bin/env python
"""Phase 2 — build the complement encoder + its natural-image baseline.

Slices W_comp = W_s1.pt rows for the complement ROIs (default bodies+places+words —
"rest of high-level vision") and computes its z-normalization baseline (mean,std of the
per-image mean voxel response over the 300 natural NSD test images) using the SAME encoder
path as the scorer: OpenCLIP ViT-L/14 openai, L2-normed image embedding @ W.t().

  python slice_complement.py \
      --w_s1 weights/W_s1.pt --out weights/W_comp_s1.pt \
      --rois bodies places words \
      --baseline_images data/s1_images.npy --baseline_split data/s1_split.npz --baseline_n 300

Prints comp_baseline_mean / comp_baseline_std (paste into the config) and re-confirms the
faces baseline (should be ~ -0.0476 / 0.2003) as a sanity check that the path matches.
"""
import argparse
from pathlib import Path
import numpy as np
import torch


def baseline_stats(W, ours_emb, imgs, idx, batch=16):
    bs = []
    for i in range(0, len(idx), batch):
        chunk = [np.asarray(imgs[j]).astype("uint8") for j in idx[i:i + batch]]
        e = ours_emb(chunk)
        bs.append((e @ W.t()).mean(dim=1).cpu().numpy())
    bs = np.concatenate(bs)
    return float(bs.mean()), float(bs.std() + 1e-8), len(idx)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--w_s1", default="weights/W_s1.pt")
    ap.add_argument("--out", default="weights/W_comp_s1.pt")
    ap.add_argument("--rois", nargs="+", default=["bodies", "places", "words"])
    ap.add_argument("--baseline_images", default="data/s1_images.npy")
    ap.add_argument("--baseline_split", default="data/s1_split.npz")
    ap.add_argument("--baseline_n", type=int, default=300)
    ap.add_argument("--faces_weights", default="weights/W_faces_s1.pt",
                    help="for the sanity re-check of the faces baseline")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(args.w_s1, map_location="cpu")
    W_full = torch.as_tensor(ck["W"], dtype=torch.float32)             # [39548, 768]
    roi_idx = ck["roi_idx"]
    # union of complement ROI voxels (dedup + sort so rows are stable/reproducible)
    vox = np.unique(np.concatenate([np.asarray(roi_idx[r]) for r in args.rois])).astype(np.int64)
    W_comp = W_full[torch.as_tensor(vox)]                             # [Vc, 768]
    print(f"[slice] complement ROIs {args.rois} -> {W_comp.shape[0]} voxels (of {W_full.shape[0]})")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"W": W_comp, "roi": "complement:" + "+".join(args.rois),
                "voxels": vox, "clip": ck.get("clip"), "l2norm": ck.get("l2norm")}, args.out)
    print(f"[slice] saved {args.out}  W{tuple(W_comp.shape)}")

    # --- baseline over 300 natural test images (same path as score_floor_all.ours_emb) ---
    import open_clip
    from PIL import Image
    clip_model, _, preprocess = open_clip.create_model_and_transforms("ViT-L-14", pretrained="openai")
    clip_model = clip_model.eval().to(device)

    def ours_emb(uint8_imgs):
        px = torch.stack([preprocess(Image.fromarray(x)) for x in uint8_imgs]).to(device)
        with torch.no_grad():
            e = clip_model.encode_image(px).float()
            return e / e.norm(p=2, dim=-1, keepdim=True)

    imgs = np.load(args.baseline_images, mmap_mode="r")
    idx = (np.load(args.baseline_split)["test_idx"] if Path(args.baseline_split).exists()
           else np.arange(imgs.shape[0]))[: args.baseline_n]

    cm, cs, n = baseline_stats(W_comp.to(device), ours_emb, imgs, idx)
    print(f"\n[COMPLEMENT baseline] comp_baseline_mean={cm:.4f} comp_baseline_std={cs:.4f} n={n}")

    if Path(args.faces_weights).exists():
        Wf = torch.as_tensor(torch.load(args.faces_weights, map_location="cpu")["W"],
                             dtype=torch.float32, device=device)
        fm, fsd, _ = baseline_stats(Wf, ours_emb, imgs, idx)
        print(f"[FACES baseline  ] roi_baseline_mean={fm:.4f} roi_baseline_std={fsd:.4f} "
              f"(expect ~ -0.0476 / 0.2003)")
    print("SLICE_COMPLEMENT_DONE")


if __name__ == "__main__":
    main()
