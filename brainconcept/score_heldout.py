#!/usr/bin/env python
"""Held-out honesty check for the brainconcept ConceptLab loop.

Scores decoded images (the loop's per-step grids, split into tiles) through one of
three brain encoders and z-scores each response against a natural-image baseline.

  encoder=ours       our ridge  (open_clip ViT-L-14 openai -> L2norm -> @ W_roi.T)
  encoder=braindive  original BrainDiVE-CLIP (OpenCLIP ViT-B/16 + linear readout, S1)
  encoder=dino       universal DINOv2 encoder (UniversalEncoder, brainguide)

"ours" is the encoder the token was optimized against (in-sample). "dino" is the
PRIMARY honesty check (different backbone, can't be hacked through CLIP). "braindive"
is a secondary CLIP-based check.

z = (roi_mean(image) - baseline_mean) / baseline_std, baseline = natural NSD images.
If ours-z stays high but dino-z collapses -> the token reward-hacked our encoder.

Run "ours" and "braindive" in ~/braindive_venv; run "dino" in the brainguide Docker
container (needs dinov2). Output: one JSON per (encoder, roi).
"""
import argparse, glob, json, os, sys
from pathlib import Path
import numpy as np


# ----------------------------------------------------------------------------- tiling
def load_grid_tiles(path):
    """A step grid is N square tiles laid out horizontally. Return list of HxWx3 uint8."""
    from PIL import Image
    im = np.asarray(Image.open(path).convert("RGB"))
    h, w = im.shape[:2]
    n = max(1, int(round(w / h)))
    ts = w // n
    return [im[:, i * ts:(i + 1) * ts] for i in range(n)]


def step_key(path):
    """Sort key: init -> 0, step_1250 -> 1250, 1250_step_images -> 1250."""
    b = Path(path).stem
    if "init" in b:
        return 0
    digits = "".join(c if c.isdigit() else " " for c in b).split()
    return int(digits[0]) if digits else 10 ** 9


# ----------------------------------------------------------------------------- backends
class OursBackend:
    """our ridge: open_clip ViT-L-14 (openai) -> L2norm -> @ W_roi.T ; mean over ROI."""
    def __init__(self, roi, args, device):
        import torch, open_clip
        self.torch = torch
        self.device = device
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            "ViT-L-14", pretrained="openai")
        self.model = self.model.eval().to(device)
        ck = torch.load(Path(args.weights_dir) / f"W_{roi}_s1.pt", map_location="cpu")
        self.W = torch.as_tensor(ck["W"], dtype=torch.float32, device=device)  # [V,768]
        # Novelty anchor: CLIP-text embedding of the "obvious category" for this ROI.
        self.anchor = None
        if getattr(args, "category_anchor", None):
            import open_clip
            tok = open_clip.get_tokenizer("ViT-L-14")
            with torch.no_grad():
                a = self.model.encode_text(tok([args.category_anchor]).to(device)).float()
                self.anchor = (a / a.norm(p=2, dim=-1, keepdim=True))   # [1,768]

    @property
    def name(self): return "ours"

    def _img_emb(self, uint8_imgs):
        from PIL import Image
        import torch
        px = torch.stack([self.preprocess(Image.fromarray(a)) for a in uint8_imgs]).to(self.device)
        with torch.no_grad():
            emb = self.model.encode_image(px).float()
            emb = emb / emb.norm(p=2, dim=-1, keepdim=True)
        return emb                                     # [B,768]

    def score(self, uint8_imgs):
        emb = self._img_emb(uint8_imgs)
        resp = emb @ self.W.t()                        # [B, V]
        return resp.mean(dim=1).cpu().numpy()          # [B]

    def catsim(self, uint8_imgs):
        """cosine(image, CLIP-text(category anchor)) — novelty axis: low = out-of-category."""
        if self.anchor is None:
            return None
        emb = self._img_emb(uint8_imgs)
        return (emb @ self.anchor.t()).squeeze(1).cpu().numpy()   # [B]


class BrainDiVEBackend:
    """original BrainDiVE-CLIP encoder, mean over the ROI's higher-visual mask."""
    def __init__(self, roi, args, device):
        import torch
        sys.path.insert(0, args.braindive_root)
        import encoder_model_vit as mv
        self.torch = torch
        self.device = device
        subj = 1
        R = os.path.join(args.braindive_root, "repro")
        self.backbone = mv.feature_extractor_vit().eval().to(device)
        for p in self.backbone.parameters():
            p.requires_grad = False
        net = torch.load(f"{R}/braindive_encoder_weights/S1/00100.chkpt",
                         map_location="cpu", weights_only=True)["network"]
        n_e = net[f"final_{subj}_early.weight"].shape[0]
        n_h = net[f"final_{subj}_higher.weight"].shape[0]
        self.bm = mv.downproject_CLIP_split_linear({subj: n_e}, {subj: n_h})
        self.bm.load_state_dict(net, strict=True); self.bm.eval().to(device)
        self.subj = subj
        self.higher_m = torch.from_numpy(np.load(f"{R}/higher_vis_mask_S1.npy")).bool().to(device)
        self.early_m = torch.from_numpy(np.load(f"{R}/early_vis_mask_S1.npy")).bool().to(device)
        self.roi_m = torch.from_numpy(np.load(f"{R}/masks/{roi}_S1.npy")).bool().to(device)
        self.L = self.higher_m.shape[0]
        self.is_early = (roi == "earlyvisual")
        self.mean = torch.tensor([0.48145466, 0.4578275, 0.40821073])[None, :, None, None].to(device)
        self.std = torch.tensor([0.26862954, 0.26130258, 0.27577711])[None, :, None, None].to(device)

    @property
    def name(self): return "braindive"

    def score(self, uint8_imgs):
        from PIL import Image
        import torch
        xs = []
        for a in uint8_imgs:
            im = Image.fromarray(a).resize((224, 224))
            xs.append(torch.from_numpy(np.asarray(im)).float().permute(2, 0, 1) / 255.0)
        x = ((torch.stack(xs).to(self.device)) - self.mean) / self.std
        out = []
        with torch.no_grad():
            for i in range(x.shape[0]):                 # one at a time (readout wants [subj])
                feats = self.backbone(x[i:i + 1])
                if self.is_early:
                    p = self.bm.forward_early(feats[0][0], feats[0][1], feats[1], [self.subj]).reshape(-1)
                    resp = torch.zeros(self.L, device=self.device); resp[self.early_m] = p
                else:
                    p = self.bm.forward_higher(feats[0][0], feats[0][1], feats[1], [self.subj]).reshape(-1)
                    resp = torch.zeros(self.L, device=self.device); resp[self.higher_m] = p
                out.append(float(resp[self.roi_m].mean()))
        return np.array(out)


class DinoBackend:
    """universal DINOv2 encoder (brainguide), mean over the ROI voxel indices."""
    def __init__(self, roi, args, device):
        import torch
        from brainguide import build_encoder
        self.torch = torch
        self.device = device
        self.enc = build_encoder(
            "universal", code_path="brainit-fmri",
            checkpoint="brainit-fmri/results/saved_models/encoder_ch128.pth",
            backbone="dinov2_vitl14_reg", inner_ch=128,
            select_layers=[1, 6, 12, 18, 24], num_reg_tokens=4,
            include_reg_tokens=False, backbone_image_size=224, device=device)
        idx = np.load(f"brainguide/roi/sub1_{roi}.npy").astype("int64")
        self.roi = torch.as_tensor(idx, dtype=torch.long, device=device)  # [V]

    @property
    def name(self): return "dino"

    def score(self, uint8_imgs):
        import torch
        x = torch.stack([torch.from_numpy(a).float().permute(2, 0, 1) / 255.0
                         for a in uint8_imgs]).to(self.device)      # [B,3,H,W] in [0,1]
        vox = self.roi.unsqueeze(0).expand(x.shape[0], -1)          # [B,V]
        with torch.no_grad():
            resp = self.enc(x, vox)                                 # [B,V]
        return resp.float().mean(dim=1).cpu().numpy()


BACKENDS = {"ours": OursBackend, "braindive": BrainDiVEBackend, "dino": DinoBackend}


# ----------------------------------------------------------------------------- baseline
def baseline_stats(backend, args, batch=16):
    imgs = np.load(args.baseline_images, mmap_mode="r")             # [N,H,W,3] uint8
    n = imgs.shape[0]
    if args.baseline_split and Path(args.baseline_split).exists():
        idx = np.load(args.baseline_split)["test_idx"]
    else:
        idx = np.arange(n)
    idx = np.asarray(idx)[: args.baseline_n]
    scores = []
    for i in range(0, len(idx), batch):
        chunk = [np.asarray(imgs[j]).astype("uint8") for j in idx[i:i + batch]]
        scores.append(backend.score(chunk))
    scores = np.concatenate(scores)
    return float(scores.mean()), float(scores.std() + 1e-8), int(len(idx))


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", required=True, choices=list(BACKENDS))
    ap.add_argument("--roi", required=True)
    ap.add_argument("--images", required=True, help="glob of step grid jpegs")
    ap.add_argument("--out", required=True)
    ap.add_argument("--weights_dir", default="weights")
    ap.add_argument("--braindive_root", default="/home/danielnurieli/BrainDiVE")
    ap.add_argument("--baseline_images", default="data/s1_images.npy")
    ap.add_argument("--baseline_split", default="data/s1_split.npz")
    ap.add_argument("--baseline_n", type=int, default=300)
    ap.add_argument("--category_anchor", default=None,
                    help="ours-only: CLIP-text anchor for the obvious category; logs cat_sim (novelty axis)")
    args = ap.parse_args()

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    backend = BACKENDS[args.encoder](args.roi, args, device)

    bmean, bstd, bn = baseline_stats(backend, args)
    print(f"[baseline] {args.encoder}/{args.roi}: mean={bmean:.4f} std={bstd:.4f} n={bn}", flush=True)

    files = sorted(glob.glob(args.images), key=step_key)
    steps = {}
    do_cat = args.encoder == "ours" and getattr(backend, "anchor", None) is not None
    for f in files:
        tiles = load_grid_tiles(f)
        raw = backend.score(tiles)                                  # [n_tiles]
        z = (raw - bmean) / bstd
        entry = {
            "file": os.path.basename(f), "n_tiles": len(tiles),
            "raw_mean": float(raw.mean()), "raw_tiles": [float(v) for v in raw],
            "z_mean": float(z.mean()), "z_max": float(z.max()),
            "z_tiles": [float(v) for v in z],
        }
        if do_cat:
            cs = backend.catsim(tiles)
            entry["cat_sim"] = float(cs.mean())
            entry["cat_sim_tiles"] = [float(v) for v in cs]
        steps[str(step_key(f))] = entry
        catmsg = f"  cat_sim={entry['cat_sim']:+.3f}" if do_cat else ""
        print(f"  step {step_key(f):>5}: raw={raw.mean():.4f}  z={z.mean():+.2f}{catmsg}", flush=True)

    out = {"encoder": args.encoder, "roi": args.roi,
           "baseline": {"mean": bmean, "std": bstd, "n": bn},
           "steps": steps}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"[done] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
