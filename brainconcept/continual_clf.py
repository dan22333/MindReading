#!/usr/bin/env python
"""Continual 'forbidden-driver' classifier for the iterative novelty loop.

Positive class = the ARCHIVE: a dict {mode_name -> embeddings[N,768]} holding {faces} plus every
discovered mode. Negative class = cached clean non-drivers (neg_emb.pt). Trains a logistic
regression on z-embeddings (logit = w.z + b, P = sigmoid). Replay = train on ALL modes each
round (optionally subsample old modes). Reports held-out AUC + PER-MODE recall (so we can see
the classifier still flags every past class). Saves weights/face_probe.pt {w,b,auc,recall}.

  python continual_clf.py --archive weights/archive.pt --neg weights/neg_emb.pt \
      --out weights/face_probe.pt --min_auc 0.97
"""
import argparse
from pathlib import Path
import numpy as np
import torch

BC = Path(__file__).resolve().parent


def split_mask(n, val_frac=0.2):
    key = ((torch.arange(n) * 2654435761) % 1000).float() / 1000.0   # deterministic, no RNG
    return key >= val_frac, key < val_frac   # train_mask, val_mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", default=str(BC / "weights/archive.pt"))
    ap.add_argument("--neg", default=str(BC / "weights/neg_emb.pt"))
    ap.add_argument("--out", default=str(BC / "weights/face_probe.pt"))
    ap.add_argument("--min_auc", type=float, default=0.97)
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--max_per_mode", type=int, default=400, help="replay cap per old mode")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    archive = torch.load(args.archive, map_location="cpu")        # {mode: [N,768]}
    neg = torch.load(args.neg, map_location="cpu")["emb"].float()
    neg = neg / neg.norm(dim=-1, keepdim=True)

    # build positives with per-mode train/val split (for per-mode recall + replay cap)
    pos_tr, pos_va, mode_va = [], [], {}
    for mode, e in archive.items():
        e = torch.as_tensor(e, dtype=torch.float32)
        e = e / e.norm(dim=-1, keepdim=True)
        tr, va = split_mask(len(e))
        etr = e[tr]
        if len(etr) > args.max_per_mode:                         # replay cap on large/old modes
            etr = etr[torch.linspace(0, len(etr) - 1, args.max_per_mode).long()]
        pos_tr.append(etr); pos_va.append(e[va]); mode_va[mode] = e[va]
    Ptr = torch.cat(pos_tr); Pva = torch.cat(pos_va)
    ntr, nva = split_mask(len(neg)); Ntr, Nva = neg[ntr], neg[nva]

    Xtr = torch.cat([Ptr, Ntr]).to(device)
    ytr = torch.cat([torch.ones(len(Ptr)), torch.zeros(len(Ntr))]).to(device)
    Xva = torch.cat([Pva, Nva]).to(device)
    yva = torch.cat([torch.ones(len(Pva)), torch.zeros(len(Nva))]).to(device)
    print(f"[clf] modes={list(archive.keys())}  pos_tr={len(Ptr)} neg_tr={len(Ntr)}", flush=True)

    w = torch.zeros(768, device=device, requires_grad=True)
    b = torch.zeros(1, device=device, requires_grad=True)
    opt = torch.optim.Adam([w, b], lr=0.05, weight_decay=1e-3)
    posw = torch.tensor([len(Ntr) / max(1, len(Ptr))], device=device)   # class balance
    lossf = torch.nn.BCEWithLogitsLoss(pos_weight=posw)
    for _ in range(args.epochs):
        opt.zero_grad(); lossf(Xtr @ w + b, ytr).backward(); opt.step()

    with torch.no_grad():
        pv = torch.sigmoid(Xva @ w + b).cpu().numpy(); yv = yva.cpu().numpy()
        order = pv.argsort(); ranks = np.empty_like(order, float); ranks[order] = np.arange(1, len(pv) + 1)
        npos, nneg = yv.sum(), (1 - yv).sum()
        auc = (ranks[yv == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg)
        # per-mode recall (fraction of that mode's held-out flagged forbidden, P>0.5)
        recall = {}
        for mode, e in mode_va.items():
            if len(e) == 0:
                recall[mode] = None; continue
            p = torch.sigmoid(e.to(device) @ w + b)
            recall[mode] = float((p > 0.5).float().mean())
        neg_fpr = float((torch.sigmoid(Nva.to(device) @ w + b) > 0.5).float().mean())
    print(f"[clf] val AUC={auc:.4f}  neg false-positive={neg_fpr:.3f}", flush=True)
    for m, r in recall.items():
        print(f"      recall[{m}] = {r if r is None else round(r,3)}", flush=True)

    torch.save({"w": w.detach().cpu(), "b": float(b.detach().cpu()), "type": "linear",
                "auc": float(auc), "recall": recall, "neg_fpr": neg_fpr,
                "modes": list(archive.keys())}, args.out)
    print(f"[clf] saved {args.out}", flush=True)
    low = [m for m, r in recall.items() if r is not None and r < 0.8]
    if low:
        print(f"[clf] WARNING low recall on past modes: {low} (forgetting — raise replay cap)", flush=True)
    if auc < args.min_auc:
        print(f"CLF_GATE_FAIL AUC {auc:.4f} < {args.min_auc}", flush=True)
    print("CONTINUAL_CLF_DONE", flush=True)


if __name__ == "__main__":
    main()
