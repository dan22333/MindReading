#!/usr/bin/env python
"""TEST 4 — DINO->CLIP bridge + Stable unCLIP 2.1 (no gradient guidance).

Pipeline (per the teaching guide, Part VI option 4):
  1. Score an image bank with the universal (DINOv2) brain encoder -> per-ROI activation.
  2. For each ROI, take the top-K images, average their DINOv2 CLS embeddings -> e_dino(roi).
  3. Bridge B: a small MLP mapping DINOv2 CLS -> OpenCLIP ViT-H/14 image embedding,
     trained on (dino, clip) pairs over the whole bank.
  4. e_clip = B(e_dino); feed as image_embeds to Stable unCLIP 2.1 -> generate.

Everything here uses the encoder only as a SCORE (top-K) and a feature extractor —
never as a gradient — so the CLIP-vs-DINO manifold mismatch cannot appear.

Runs in the brainguide container (transformers + torch.hub; no open_clip needed).
  python experiments/test4_dino_clip_bridge/run_test4.py \
      --images /app/brainguide/exp_out/imagebank \
      --config configs/universal.yaml \
      --out exp_out/test4 --topk 20 --epochs 30
"""
import argparse
import glob
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from PIL import Image

# run.py lives at the brainguide/ root (two levels up from this script). When this
# file is launched directly (`python experiments/.../run_test4.py`) Python puts the
# SCRIPT's dir on sys.path — NOT the brainguide root — so `from run import build`
# would ModuleNotFoundError. Add the root explicitly. (The container's PYTHONPATH
# only carries brainit-fmri, not brainguide, so this is required on the VM too.)
import sys
BG_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BG_ROOT))

from run import build  # reuse the universal-encoder builder

CLIP_H_ID = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
# unCLIP (stabilityai/stable-diffusion-2-1-unclip) is GATED on HF (401 anonymous),
# so option 4's generator here is the other public half: SD1.5 + IP-Adapter (ViT-H),
# which consumes the SAME OpenCLIP-ViT-H image embeds the bridge produces.
SD15_ID = "runwayml/stable-diffusion-v1-5"
IPADAPTER_ID = "h94/IP-Adapter"
ROIS = ["faces", "places", "words", "bodies", "earlyvisual"]


# ---- image bank ------------------------------------------------------------
def list_images(d):
    exts = ("*.jpg", "*.jpeg", "*.png", "*.JPEG", "*.PNG")
    files = []
    for e in exts:
        files += glob.glob(os.path.join(d, "**", e), recursive=True)
    return sorted(files)


def load_batch(paths, size, device):
    ims = []
    for p in paths:
        im = Image.open(p).convert("RGB").resize((size, size))
        ims.append(torch.from_numpy(np.asarray(im)).float().permute(2, 0, 1) / 255.0)
    return torch.stack(ims).to(device)


# ---- feature extractors ----------------------------------------------------
def dino_cls_fn(device):
    """dinov2_vitl14_reg CLS embedding (1024-d), matching the encoder's backbone."""
    m = torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14_reg").to(device).eval()
    mean = torch.tensor([0.485, 0.456, 0.406], device=device)[None, :, None, None]
    std = torch.tensor([0.229, 0.224, 0.225], device=device)[None, :, None, None]

    @torch.no_grad()
    def f(imgs01):  # imgs01: B,3,H,W in [0,1]
        x = torch.nn.functional.interpolate(imgs01, size=224, mode="bilinear", align_corners=False)
        x = (x - mean) / std
        return m(x)  # B,1024 (CLS after norm)
    return f


def clip_h_fn(device):
    from transformers import CLIPVisionModelWithProjection
    # token=False forces anonymous access: these repos are public, and a stale/
    # invalid cached HF token otherwise 401s the (uncached) Hub metadata fetch.
    m = CLIPVisionModelWithProjection.from_pretrained(CLIP_H_ID, token=False).to(device).eval()
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=device)[None, :, None, None]
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=device)[None, :, None, None]

    @torch.no_grad()
    def f(imgs01):
        x = torch.nn.functional.interpolate(imgs01, size=224, mode="bilinear", align_corners=False)
        x = (x - mean) / std
        return m(pixel_values=x).image_embeds  # B,1024
    return f


# ---- the bridge ------------------------------------------------------------
class Bridge(nn.Module):
    def __init__(self, din=1024, dout=1024, h=2048):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(din, h), nn.GELU(),
            nn.Linear(h, h), nn.GELU(),
            nn.Linear(h, dout),
        )

    def forward(self, x):
        return self.net(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True, help="image-bank dir")
    ap.add_argument("--config", default="configs/universal.yaml")
    ap.add_argument("--out", default="exp_out/test4")
    ap.add_argument("--topk", type=int, default=20)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--ip-scale", dest="ip_scale", type=float, default=0.8,
                    help="IP-Adapter conditioning strength (1.0 = strongest brain-target signal)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seeds", default="", help="space-separated seeds; multi-sample per ROI (overrides --seed)")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    files = list_images(args.images)
    assert files, f"no images in {args.images}"
    print(f"[t4] image bank: {len(files)} images")

    # 1. encoders
    cfg = yaml.safe_load(Path(args.config).read_text())
    _, encoder, _, _ = build(cfg)               # universal brain encoder (frozen)
    dino = dino_cls_fn(device)
    clip = clip_h_fn(device)

    # ROI voxel index sets (real ROIs). Anchor to the brainguide root so the load
    # doesn't depend on the process cwd (the .npy live at brainguide/roi/).
    roi_idx = {r: torch.as_tensor(np.load(BG_ROOT / "brainguide" / "roi" / f"sub1_{r}.npy").astype("int64"),
                                  device=device) for r in ROIS}

    # 2. sweep the bank: DINO CLS, CLIP img emb, and per-ROI activation.
    # NOTE the encoder contract (encoders/base.py): UniversalEncoder.forward takes
    # a [0,1] image PLUS an explicit voxel-index selection, and does its OWN resize
    # + ImageNet normalization inside. So we feed the raw [0,1] batch (not a CLIP-
    # normalized one — that would double-normalize) and pass each ROI's indices.
    D, C = [], []
    scores = {r: [] for r in ROIS}
    for i in range(0, len(files), args.bs):
        batch = load_batch(files[i:i + args.bs], 224, device)   # [B,3,224,224] in [0,1]
        D.append(dino(batch).cpu())                             # ImageNet-norm done inside
        C.append(clip(batch).cpu())                             # CLIP-norm done inside
        with torch.no_grad():
            for r in ROIS:
                # The released encoder builds voxel_embed = voxel_embed[voxel_ind],
                # so voxel_ind's leading dim must match the image batch (its internal
                # map-attention bmms voxel-side against image-side per batch element).
                # The pipeline only ever ran B=1; here B>1, so tile indices to [B, V].
                vox = roi_idx[r].unsqueeze(0).expand(batch.shape[0], -1)   # [B, |roi_r|]
                acts = encoder(batch, vox)                                 # [B, |roi_r|]
                scores[r].append(acts.mean(1).cpu())
        print(f"[t4] scored {min(i+args.bs, len(files))}/{len(files)}", end="\r")
    D = torch.cat(D); C = torch.cat(C)
    scores = {r: torch.cat(v) for r, v in scores.items()}
    print(f"\n[t4] features: DINO {tuple(D.shape)}  CLIP {tuple(C.shape)}")

    # 3. train the bridge DINO->CLIP  (cosine + MSE)
    bridge = Bridge(D.shape[1], C.shape[1]).to(device)
    opt = torch.optim.Adam(bridge.parameters(), lr=1e-3)
    Dd, Cd = D.to(device), C.to(device)
    Cn = torch.nn.functional.normalize(Cd, dim=1)
    n = len(Dd)
    for ep in range(args.epochs):
        perm = torch.randperm(n, device=device)
        tot = 0.0
        for i in range(0, n, args.bs):
            idx = perm[i:i + args.bs]
            pred = bridge(Dd[idx])
            cos = 1 - (torch.nn.functional.normalize(pred, dim=1) * Cn[idx]).sum(1).mean()
            mse = torch.nn.functional.mse_loss(pred, Cd[idx])
            loss = cos + 0.5 * mse
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(idx)
        if ep % 5 == 0 or ep == args.epochs - 1:
            print(f"[t4] bridge epoch {ep:3d}  loss {tot/n:.4f}")
    torch.save(bridge.state_dict(), out / "bridge.pt")

    # free scoring models before loading unCLIP
    del encoder
    torch.cuda.empty_cache()

    # 4. Generate from bridged targets via SD1.5 + IP-Adapter (ViT-H).
    from diffusers import StableDiffusionPipeline
    pipe = StableDiffusionPipeline.from_pretrained(SD15_ID, torch_dtype=torch.float16, token=False)
    pipe = pipe.to(device); pipe.safety_checker = None
    # IP-Adapter's ImageProjection maps a 1024-d OpenCLIP-ViT-H image embed into the
    # UNet's cross-attention. We feed it precomputed embeds (bridge output), so the
    # pipeline's own CLIP image encoder is never invoked.
    # base ip-adapter_sd15 is the ViT-H variant (1024-d CLIP image embeds).
    pipe.load_ip_adapter(IPADAPTER_ID, subfolder="models",
                         weight_name="ip-adapter_sd15.safetensors")
    pipe.set_ip_adapter_scale(args.ip_scale)
    bridge.eval()

    seeds = [int(x) for x in str(args.seeds).split()] if args.seeds.strip() else [args.seed]
    for r in ROIS:
        topk = torch.topk(scores[r], min(args.topk, n)).indices
        e_dino = D[topk].mean(0, keepdim=True).to(device)             # avg DINO CLS of top-K
        with torch.no_grad():
            e_clip = bridge(e_dino)                                   # -> OpenCLIP-ViT-H space [1,1024]
        cond = e_clip.reshape(1, 1, e_clip.shape[-1]).to(torch.float16)
        neg = torch.zeros_like(cond)
        ip_embeds = torch.cat([neg, cond], dim=0)                     # [2,1,1024]: (neg, cond) for CFG
        # bridge target is fixed per ROI; seeds only vary the diffusion noise -> a sample set.
        for s in seeds:
            g = torch.Generator(device=device).manual_seed(s)
            img = pipe(prompt="", ip_adapter_image_embeds=[ip_embeds],
                       num_inference_steps=40, guidance_scale=7.5, generator=g).images[0]
            img.save(out / f"t4_{r}_seed{s}_0.png")
        print(f"[t4] {r}: saved {len(seeds)} seeds  (top-K mean score {scores[r][topk].mean():.3f})")


if __name__ == "__main__":
    main()
