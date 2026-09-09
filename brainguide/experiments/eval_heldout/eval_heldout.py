#!/usr/bin/env python3
"""Held-out (independent) selectivity — the honest judge.

Scores our brainguide panel (baseline/plan1/plan2/plan4) with the reproduced
ORIGINAL BrainDiVE encoder (OpenCLIP ViT-B/16 + linear read-out, subject 1) — a
DIFFERENT encoder than the DINOv2 one plans 1/2 optimized. If a method merely
hacked our encoder, it drops here; if it found real ROI content, it holds.

Its own experiment; runs in braindive_venv (timm), NO diffusers needed:
  ~/braindive_venv/bin/python eval_heldout.py \
     --bd-root ~/BrainDiVE --repro ~/BrainDiVE/repro \
     --ckpt ~/BrainDiVE/repro/braindive_encoder_weights/S1/00100.chkpt \
     --base ~/exp_outputs_grid --out ~/exp_outputs_grid/eval_heldout
"""
import argparse, glob, json, os, sys
import numpy as np, torch
from PIL import Image

ROIS = ["faces", "places", "words", "bodies", "earlyvisual"]
HIGHER = {"faces", "places", "words", "bodies"}      # scored via forward_higher; early via forward_early
METHODS = {"baseline": ("baseline", "base"), "plan1": ("test1", "t1"),
           "plan2ctrl": ("test2", "t2ctrl"), "plan2tt": ("test2", "t2tt"),
           "plan4": ("test4", "t4"), "readout": ("test3", "t3"), "ddpo": ("test5", "t5")}
CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073])[None, :, None, None]
CLIP_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711])[None, :, None, None]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bd-root", required=True)
    ap.add_argument("--repro", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--subject", type=int, default=1)
    a = ap.parse_args()
    subj = a.subject
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(a.out, exist_ok=True)
    sys.path.insert(0, os.path.expanduser(a.bd_root))
    import encoder_model_vit as mv

    backbone = mv.feature_extractor_vit().eval().to(dev)
    for p in backbone.parameters():
        p.requires_grad = False
    ck = torch.load(os.path.expanduser(a.ckpt), map_location="cpu", weights_only=True)
    net = ck["network"] if "network" in ck else ck
    n_early = net[f"final_{subj}_early.weight"].shape[0]
    n_higher = net[f"final_{subj}_higher.weight"].shape[0]
    print(f"[heldout] S{subj} dims early={n_early} higher={n_higher}")
    bm = mv.downproject_CLIP_split_linear(num_early_output={subj: n_early},
                                          num_higher_output={subj: n_higher})
    bm.load_state_dict(net, strict=True)
    bm.eval().to(dev)
    for p in bm.parameters():
        p.requires_grad = False

    R = os.path.expanduser(a.repro)
    early_m = torch.from_numpy(np.load(f"{R}/early_vis_mask_S1.npy")).bool().to(dev)
    higher_m = torch.from_numpy(np.load(f"{R}/higher_vis_mask_S1.npy")).bool().to(dev)
    assert int(early_m.sum()) == n_early and int(higher_m.sum()) == n_higher, "mask/encoder mismatch!"
    roi_m = {r: torch.from_numpy(np.load(f"{R}/masks/{r}_S1.npy")).bool().to(dev) for r in ROIS}
    mean = CLIP_MEAN.to(dev); std = CLIP_STD.to(dev)
    L = higher_m.shape[0]                                 # functional-voxel length

    @torch.no_grad()
    def score_one(path):
        im = Image.open(path).convert("RGB").resize((224, 224))
        x = torch.from_numpy(np.asarray(im)).float().permute(2, 0, 1)[None] / 255.0
        x = ((x.to(dev) - mean) / std)
        feats = backbone(x)
        ph = bm.forward_higher(feats[0][0], feats[0][1], feats[1], [subj]).reshape(-1)
        pe = bm.forward_early(feats[0][0], feats[0][1], feats[1], [subj]).reshape(-1)
        rh = torch.zeros(L, device=dev); rh[higher_m] = ph
        re = torch.zeros(L, device=dev); re[early_m] = pe
        return {j: float((re if j == "earlyvisual" else rh)[roi_m[j]].mean()) for j in ROIS}

    report = {}
    for m, (sub, pre) in METHODS.items():
        raw = np.full((5, 5), np.nan); allcol = {j: [] for j in ROIS}
        for i, roi_i in enumerate(ROIS):
            files = sorted(glob.glob(f"{os.path.expanduser(a.base)}/{sub}/{pre}_{roi_i}_seed*_0.png"))
            if not files:
                continue
            cols = {j: [] for j in ROIS}
            for f in files:
                s = score_one(f)
                for j in ROIS:
                    cols[j].append(s[j]); allcol[j].append(s[j])
            for j, roi_j in enumerate(ROIS):
                raw[i, j] = float(np.mean(cols[roi_j]))
        if np.isnan(raw).all():
            print(f"[heldout] {m}: no images"); continue
        mu = {j: float(np.mean(allcol[j])) for j in ROIS}
        sd = {j: float(np.std(allcol[j]) + 1e-8) for j in ROIS}
        z = np.array([[(raw[i, j] - mu[ROIS[j]]) / sd[ROIS[j]] for j in range(5)] for i in range(5)])
        hits = sum(1 for i in range(5) if not np.isnan(z[i]).any() and int(np.argmax(z[i])) == i)
        diag = float(np.nanmean([z[i, i] for i in range(5)]))
        off = float(np.nanmean([z[i, j] for i in range(5) for j in range(5) if i != j]))
        report[m] = {"raw": raw.tolist(), "z": z.tolist(), "diag_argmax_hits": hits,
                     "diag_z_mean": diag, "offdiag_z_mean": off, "selectivity_contrast": diag - off}
        print(f"[heldout] {m:9s} diag-argmax {hits}/5  contrast {diag-off:+.2f}")

    Path = os.path.join(a.out, "selectivity_heldout.json")
    with open(Path, "w") as fh:
        json.dump({"rois": ROIS, "encoder": "original BrainDiVE OpenCLIP-ViT-B/16 (S1)",
                   "methods": report}, fh, indent=2)
    print(f"[heldout] wrote {Path}")


if __name__ == "__main__":
    main()
