#!/usr/bin/env python
"""TEST 3 (option 3) — Readout-head guidance, stage 1: harvest + train.

Teach a small head to predict the DINOv2 brain-encoder's per-ROI score from SD's
OWN UNet features phi_t(x_t). The encoder is used ONLY to make labels (frozen);
the UNet is frozen (we read its mid-block). At sampling time we then guide by the
head's gradient, which flows through renderable UNet features -> on-manifold.

  python experiments/test3_readout_head/train_heads.py \
      --images /app/brainguide/exp_out/imagebank --config configs/universal.yaml \
      --out experiments/test3_readout_head --epochs 40 --bs 8

Its own experiment. Runs in the brainguide container.
"""
import argparse, glob, os, sys
from pathlib import Path
import numpy as np, torch, torch.nn as nn, yaml
from PIL import Image

BG = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BG))
from run import build

ROIS = ["faces", "places", "words", "bodies", "earlyvisual"]
SD15 = "runwayml/stable-diffusion-v1-5"
TIMESTEPS = [50, 150, 300, 500, 700, 900]      # noise levels to train the head across


def list_images(d):
    fs = []
    for e in ("*.jpg", "*.jpeg", "*.png", "*.JPEG"):
        fs += glob.glob(os.path.join(d, "**", e), recursive=True)
    return sorted(fs)


def load01(paths, size, device):
    ims = []
    for p in paths:
        im = Image.open(p).convert("RGB").resize((size, size))
        ims.append(torch.from_numpy(np.asarray(im)).float().permute(2, 0, 1) / 255.0)
    return torch.stack(ims).to(device)


class Head(nn.Module):                          # phi (pooled 1280) + t -> per-ROI score
    def __init__(self, din=1280, nroi=5, h=512):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(din + 1, h), nn.GELU(),
                                 nn.Linear(h, h), nn.GELU(), nn.Linear(h, nroi))

    def forward(self, feat, t):                  # feat [B,1280], t [B] in [0,1]
        return self.net(torch.cat([feat, t[:, None]], 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--config", default="configs/universal.yaml")
    ap.add_argument("--out", default="experiments/test3_readout_head")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--max-images", type=int, default=1000)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    files = list_images(args.images)[: args.max_images]
    assert files, f"no images in {args.images}"
    print(f"[t3] bank: {len(files)} images")

    cfg = yaml.safe_load(Path(args.config).read_text())
    _, encoder, _, _ = build(cfg)               # frozen DINOv2 brain encoder (labels only)
    roi_idx = {r: torch.as_tensor(np.load(BG / "brainguide" / "roi" / f"sub1_{r}.npy").astype("int64"),
                                  device=dev) for r in ROIS}

    from diffusers import StableDiffusionPipeline
    pipe = StableDiffusionPipeline.from_pretrained(SD15, torch_dtype=torch.float16, token=False)
    pipe = pipe.to(dev); pipe.safety_checker = None
    unet, vae, sched = pipe.unet.eval(), pipe.vae.eval(), pipe.scheduler
    tok, tenc = pipe.tokenizer, pipe.text_encoder
    with torch.no_grad():                       # empty-prompt cond embeds (BrainDiVE-style)
        ids = tok("", padding="max_length", max_length=tok.model_max_length,
                  truncation=True, return_tensors="pt").input_ids.to(dev)
        empty = tenc(ids)[0]                     # [1,77,768]

    # hook the mid-block output
    feats = {}
    def hook(m, i, o): feats["mid"] = o
    unet.mid_block.register_forward_hook(hook)

    X, T, Y = [], [], []                         # pooled feats, t/1000, labels
    for i in range(0, len(files), args.bs):
        batch = files[i:i + args.bs]
        imgs01 = load01(batch, 512, dev)
        with torch.no_grad():
            # labels: DINOv2 encoder per-ROI (encoder wants [0,1]+voxel idx, does own norm)
            lab = torch.stack([encoder(imgs01, roi_idx[r].unsqueeze(0).expand(imgs01.shape[0], -1)).mean(1)
                               for r in ROIS], 1)   # [B,5]
            lat = vae.encode(imgs01.half() * 2 - 1).latent_dist.mean * vae.config.scaling_factor  # [B,4,64,64]
            for t in TIMESTEPS:
                tt = torch.full((lat.shape[0],), t, device=dev, dtype=torch.long)
                noise = torch.randn_like(lat)
                x_t = sched.add_noise(lat, noise, tt)
                emb = empty.repeat(lat.shape[0], 1, 1)
                unet(x_t, tt, encoder_hidden_states=emb)      # fills feats["mid"]
                pooled = feats["mid"].float().mean(dim=(2, 3))  # [B,1280]
                X.append(pooled.cpu()); T.append(torch.full((lat.shape[0],), t / 1000.0))
                Y.append(lab.float().cpu())
        print(f"[t3] harvested {min(i+args.bs, len(files))}/{len(files)}", end="\r")
    X = torch.cat(X); T = torch.cat(T); Y = torch.cat(Y)
    print(f"\n[t3] dataset: X{tuple(X.shape)} Y{tuple(Y.shape)}")

    ymu, ysd = Y.mean(0), Y.std(0) + 1e-6        # normalize labels per ROI
    Yn = (Y - ymu) / ysd
    head = Head(X.shape[1], len(ROIS)).to(dev)
    opt = torch.optim.Adam(head.parameters(), lr=1e-3, weight_decay=1e-4)
    Xd, Td, Yd = X.to(dev), T.to(dev), Yn.to(dev)
    n = len(Xd)
    for ep in range(args.epochs):
        perm = torch.randperm(n, device=dev); tot = 0.0
        for j in range(0, n, 256):
            idx = perm[j:j + 256]
            pred = head(Xd[idx], Td[idx])
            loss = torch.nn.functional.mse_loss(pred, Yd[idx])
            opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item() * len(idx)
        if ep % 5 == 0 or ep == args.epochs - 1:
            # per-ROI correlation as a sanity check
            with torch.no_grad():
                p = head(Xd, Td)
                cors = [float(torch.corrcoef(torch.stack([p[:, k], Yd[:, k]]))[0, 1]) for k in range(5)]
            print(f"[t3] ep{ep:3d} mse {tot/n:.4f}  r=" + " ".join(f"{r[:3]}:{c:.2f}" for r, c in zip(ROIS, cors)))
    torch.save({"state": head.state_dict(), "ymu": ymu, "ysd": ysd, "rois": ROIS,
                "din": X.shape[1]}, out / "readout_head.pt")
    print(f"[t3] saved {out/'readout_head.pt'}")


if __name__ == "__main__":
    main()
