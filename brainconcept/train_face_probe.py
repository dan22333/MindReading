#!/usr/bin/env python
"""Phase 1 (round 2) — train + VALIDATE a linear face probe on prior-space embeddings.

Positives = face_bank.pt, negatives = nonface_bank.pt (both prior-output CLIP image embeddings,
the exact space the optimizer works in). Trains logistic regression logit = e·w + b, P=sigmoid.

GATE: held-out AUC must be >= --min_auc (default 0.97) or it exits non-zero (no wasted sweep).
Functional test: also scores the 300 NSD natural images (open_clip) so we see P(face) on real
non-face photos. Saves weights/face_probe.pt {w:[768], b, auc, acc}.

  python train_face_probe.py --face weights/face_bank.pt --nonface weights/nonface_bank.pt \
      --out weights/face_probe.pt --min_auc 0.97
"""
import argparse, sys
from pathlib import Path
import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--face", default="weights/face_bank.pt")
    ap.add_argument("--nonface", default="weights/nonface_bank.pt")
    ap.add_argument("--out", default="weights/face_probe.pt")
    ap.add_argument("--min_auc", type=float, default=0.97)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--val_frac", type=float, default=0.2)
    ap.add_argument("--baseline_images", default="data/s1_images.npy")
    ap.add_argument("--baseline_split", default="data/s1_split.npz")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    F = torch.as_tensor(torch.load(args.face, map_location="cpu")["img_set"], dtype=torch.float32)
    N = torch.as_tensor(torch.load(args.nonface, map_location="cpu")["img_set"], dtype=torch.float32)
    F = F / F.norm(dim=-1, keepdim=True); N = N / N.norm(dim=-1, keepdim=True)
    X = torch.cat([F, N], 0); y = torch.cat([torch.ones(len(F)), torch.zeros(len(N))])
    print(f"[data] faces {len(F)} nonfaces {len(N)} dim {X.shape[1]}")

    # deterministic split (no RNG — argsort of a fixed hash-like key)
    key = ((torch.arange(len(X)) * 2654435761) % 1000).float() / 1000.0
    val = key < args.val_frac
    Xtr, ytr = X[~val].to(device), y[~val].to(device)
    Xva, yva = X[val].to(device), y[val].to(device)

    w = torch.zeros(X.shape[1], device=device, requires_grad=True)
    b = torch.zeros(1, device=device, requires_grad=True)
    opt = torch.optim.Adam([w, b], lr=0.05, weight_decay=1e-3)
    lossf = torch.nn.BCEWithLogitsLoss()
    for e in range(args.epochs):
        opt.zero_grad()
        loss = lossf(Xtr @ w + b, ytr)
        loss.backward(); opt.step()

    with torch.no_grad():
        pv = torch.sigmoid(Xva @ w + b).cpu().numpy(); yv = yva.cpu().numpy()
        # AUC via rank statistic
        order = pv.argsort()
        ranks = np.empty_like(order, dtype=float); ranks[order] = np.arange(1, len(pv) + 1)
        npos, nneg = yv.sum(), (1 - yv).sum()
        auc = (ranks[yv == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg)
        acc = ((pv > 0.5) == (yv > 0.5)).mean()
        ptr_face = torch.sigmoid(F.to(device) @ w + b).mean().item()
        ptr_non = torch.sigmoid(N.to(device) @ w + b).mean().item()
    print(f"[probe] val AUC={auc:.4f} acc={acc:.3f} | mean P(face): faces={ptr_face:.3f} nonfaces={ptr_non:.3f}")

    # functional test on real NSD naturals (open_clip space — cross-check generalization)
    try:
        import open_clip
        from PIL import Image
        cm, _, pre = open_clip.create_model_and_transforms("ViT-L-14", pretrained="openai")
        cm = cm.eval().to(device)
        imgs = np.load(args.baseline_images, mmap_mode="r")
        idx = (np.load(args.baseline_split)["test_idx"] if Path(args.baseline_split).exists()
               else np.arange(imgs.shape[0]))[:300]
        ps = []
        for i in range(0, len(idx), 16):
            px = torch.stack([pre(Image.fromarray(np.asarray(imgs[j]).astype("uint8"))) for j in idx[i:i+16]]).to(device)
            with torch.no_grad():
                e = cm.encode_image(px).float(); e = e / e.norm(dim=-1, keepdim=True)
                ps.append(torch.sigmoid(e @ w + b).cpu().numpy())
        ps = np.concatenate(ps)
        print(f"[probe] NSD naturals (mostly non-face) mean P(face)={ps.mean():.3f} "
              f"(frac>0.5 = {100*np.mean(ps>0.5):.0f}%)  [want low]")
    except Exception as ex:
        print(f"[probe] NSD functional test skipped: {ex}")

    torch.save({"w": w.detach().cpu(), "b": float(b.detach().cpu()), "auc": float(auc),
                "acc": float(acc), "type": "linear"}, args.out)
    print(f"saved {args.out}")
    if auc < args.min_auc:
        print(f"PROBE_GATE_FAIL: AUC {auc:.4f} < {args.min_auc}"); sys.exit(2)
    print("TRAIN_FACE_PROBE_DONE")


if __name__ == "__main__":
    main()
