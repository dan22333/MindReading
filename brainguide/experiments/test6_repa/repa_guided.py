#!/usr/bin/env python
"""TEST 6 (option 6) — REPA: gradient guidance with the universal DINOv2 encoder on a
DINOv2-ALIGNED generator (SiT-XL/2 trained with REPA). REPA trains the diffusion
transformer so its internal features match DINOv2; a DINOv2 brain-encoder gradient is
therefore (more) on-manifold -> higher alpha than SD1.5. This is exactly "gradient
guidance you can use with your universal encoder" on a model that fits it.

Encoder frozen (used only as a differentiable score); SiT + VAE frozen. Guidance is on
the predicted clean latent x0 = x - t*v (velocity detached, so no backprop through SiT).

  python experiments/test6_repa/repa_guided.py --repa ~/REPA --config configs/universal.yaml \
      --rois faces places words bodies --seeds 0 1 2 3 4 --steps 50 --scale 0.3 --out exp_out/test6
"""
import argparse, os, sys
from pathlib import Path
import numpy as np, torch, yaml
from PIL import Image

BG = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(BG))
ROIS = ["faces", "places", "words", "bodies"]
SCALE = 0.18215


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repa", default=os.path.expanduser("~/REPA"))
    ap.add_argument("--config", default="configs/universal.yaml")
    ap.add_argument("--rois", nargs="+", default=ROIS)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--scale", type=float, default=0.3)   # guidance strength (frac of |v|)
    ap.add_argument("--lo", type=float, default=0.10)      # semantic window (in t, which runs 1->0)
    ap.add_argument("--hi", type=float, default=0.85)
    ap.add_argument("--out", default="exp_out/test6")
    a = ap.parse_args()
    dev = "cuda"
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    # IMPORTANT: brainit-fmri (universal encoder) AND REPA BOTH ship a top-level
    # `models` package. Build the encoder FIRST so unpickling binds `models` to
    # brainit-fmri; then purge `models*` from sys.modules and put REPA on the path so
    # `from models.sit import ...` resolves to REPA. The already-instantiated encoder
    # module doesn't re-import `models`, so it keeps working afterwards.
    from run import build
    cfg = yaml.safe_load(Path(a.config).read_text())
    print("[t6] building universal DINOv2 brain encoder …")
    _, enc, _, _ = build(cfg)                     # frozen; used only as a differentiable score
    roi_idx = {r: torch.as_tensor(np.load(BG / "brainguide" / "roi" / f"sub1_{r}.npy").astype("int64"),
                                  device=dev) for r in a.rois}

    for _m in [m for m in list(sys.modules) if m == "models" or m.startswith("models.")]:
        del sys.modules[_m]
    # brainit-fmri/models is a REAL package (__init__.py) and would win over REPA's
    # namespace models/ dir even with REPA first on the path — so drop brainit-fmri
    # from sys.path now (the encoder is already instantiated and won't re-import).
    sys.path[:] = [p for p in sys.path if "brainit-fmri" not in p]
    sys.path.insert(0, os.path.expanduser(a.repa))
    from models.sit import SiT_models
    from utils import download_model
    from diffusers.models import AutoencoderKL

    print("[t6] building SiT-XL/2 + loading REPA checkpoint …")
    model = SiT_models["SiT-XL/2"](input_size=32, num_classes=1000, use_cfg=True,
                                   z_dims=[768], encoder_depth=8,
                                   fused_attn=False, qk_norm=False).to(dev)
    sd = download_model('last.pt')
    model.load_state_dict(sd); model.eval()
    for p in model.parameters(): p.requires_grad_(False)
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema").to(dev).eval()
    for p in vae.parameters(): p.requires_grad_(False)

    def decode01(lat):                           # [1,4,32,32] -> [1,3,256,256] in [0,1]
        img = vae.decode(lat / SCALE).sample
        return ((img + 1) / 2).clamp(0, 1)

    tsteps = torch.linspace(1, 0, a.steps + 1)
    y_null = torch.tensor([1000], device=dev)    # CFG null token = unconditional (pure brain discovery)

    for roi in a.rois:
        vox = roi_idx[roi].unsqueeze(0)          # [1,V]
        for s in a.seeds:
            g = torch.Generator(device=dev).manual_seed(s)
            x = torch.randn(1, 4, 32, 32, generator=g, device=dev, dtype=torch.float32)
            for i in range(a.steps):
                t_cur = float(tsteps[i]); t_next = float(tsteps[i + 1])
                with torch.no_grad():            # velocity (fp32, detached: no backprop through SiT)
                    v = model(x, torch.full((1,), t_cur, device=dev, dtype=torch.float32),
                              y=y_null)[0]
                x_next = x + (t_next - t_cur) * v
                if a.lo <= t_cur <= a.hi:         # guide on predicted x0 in the semantic window
                    xg = x.detach().requires_grad_(True)
                    x0 = xg - t_cur * v           # v detached -> grad flows only through decode+encoder
                    score = enc(decode01(x0).float(), vox).mean()   # maximize ROI activation
                    grad = torch.autograd.grad(score, xg)[0]
                    gn = grad.flatten(1).norm(dim=1).clamp_min(1e-8).view(-1, 1, 1, 1)
                    vn = v.flatten(1).norm(dim=1).view(-1, 1, 1, 1)
                    x_next = x_next + a.scale * vn * (grad / gn)     # bounded, unit-direction step
                x = x_next.detach()
            with torch.no_grad():
                img = decode01(x)[0].permute(1, 2, 0).cpu().numpy()
            Image.fromarray((img * 255).astype("uint8")).save(out / f"t6_{roi}_seed{s}_0.png")
            print(f"[t6] saved t6_{roi}_seed{s}_0.png")
    print("[t6] DONE")


if __name__ == "__main__":
    main()
