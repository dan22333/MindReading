#!/usr/bin/env python
"""Entry point: config -> encoder -> pipe.brain_tweak -> generate -> save.

    # cheap: validate the scorer + gradient contract, no SD download
    python run.py --config configs/braindive_clip.yaml --smoke

    # full run: guided Stable Diffusion generation
    python run.py --config configs/braindive_clip.yaml --prompt "a photograph of a face"

Everything about *which brain the sampler steers toward* lives in the config's
``encoder`` block; swapping ``encoder.name`` from ``braindive_clip`` to
``universal`` changes the science without touching this file.
"""
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import yaml

from brainguide import build_encoder, make_brain_tweak


class ReadoutHead(nn.Module):
    """TEST 3 head: pooled UNet mid-block features (+ t) -> per-ROI score.
    Architecture must match experiments/test3_readout_head/train_heads.py::Head."""
    def __init__(self, din=1280, nroi=5, h=512):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(din + 1, h), nn.GELU(),
                                 nn.Linear(h, h), nn.GELU(), nn.Linear(h, nroi))

    def forward(self, feat, t):
        return self.net(torch.cat([feat, t[:, None]], 1))


def pick_device(requested: str) -> str:
    if requested and requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_voxels(spec, num_voxels):
    """Turn a config ``voxels`` spec into a LongTensor target-ROI selection.

    Accepts a **path to a .npy** file of indices (a real ROI, e.g.
    ``roi/sub1_faces.npy``), an explicit list, a ``{start,stop}`` range, or
    ``null`` (all voxels up to ``num_voxels``).
    """
    if spec is None:
        return torch.arange(num_voxels, dtype=torch.long)
    if isinstance(spec, str):
        import numpy as np
        p = Path(spec)
        if not p.is_absolute():
            p = Path(__file__).resolve().parent / spec   # relative to run.py
        return torch.as_tensor(np.load(p).astype("int64"), dtype=torch.long)
    if isinstance(spec, dict):
        return torch.arange(spec["start"], spec["stop"], dtype=torch.long)
    return torch.as_tensor(list(spec), dtype=torch.long)


def build(cfg, voxels_override=None):
    device = pick_device(cfg.get("device", "auto"))
    enc_cfg = dict(cfg["encoder"])
    name = enc_cfg.pop("name")
    enc_cfg.setdefault("device", device)
    encoder = build_encoder(name, **enc_cfg)

    steer = cfg.get("steer", {})
    spec = voxels_override if voxels_override is not None else steer.get("voxels")
    voxels = resolve_voxels(spec, enc_cfg.get("num_voxels", 1000))
    tweak = make_brain_tweak(
        encoder,
        voxels=voxels,
        sign=float(steer.get("sign", 1.0)),
        reduce=steer.get("reduce", "mean"),
    )
    return device, encoder, voxels, tweak


def smoke(cfg):
    """Prove the image->scalar contract and that gradients reach the image."""
    device, encoder, voxels, tweak = build(cfg)
    print(f"[smoke] device={device}  encoder={cfg['encoder']['name']}  "
          f"target voxels={len(voxels)}")
    # brain_tweak receives a CLIP-normalized image (what the sampler hands it).
    img = torch.randn(1, 3, 224, 224, device=device, requires_grad=True)
    score = tweak(img)
    score.backward()
    assert score.ndim == 0, f"brain_tweak must return a scalar, got {tuple(score.shape)}"
    assert img.grad is not None and torch.isfinite(img.grad).all(), "no finite gradient"
    gnorm = img.grad.norm().item()
    print(f"[smoke] score={score.item():+.4f}  grad_norm={gnorm:.4e}  "
          f"grad_flows={'yes' if gnorm > 0 else 'NO'}")
    print("[smoke] OK — encoder is a valid, differentiable pipe.brain_tweak.")


def generate(cfg, args):
    from brainguide.pipeline import mypipelineSAG
    from diffusers import DPMSolverMultistepScheduler, DDIMScheduler

    device, encoder, voxels, tweak = build(cfg, voxels_override=args.voxels)
    gen_cfg = dict(cfg.get("generate", {}))
    model_id = cfg.get("model_id", "runwayml/stable-diffusion-v1-5")
    tag = args.tag or cfg["encoder"]["name"]
    print(f"[gen] target='{tag}'  voxels={len(voxels)}")

    dtype = torch.float16 if device == "cuda" else torch.float32
    print(f"[gen] loading {model_id} on {device} ({dtype}) …")
    pipe = mypipelineSAG.from_pretrained(model_id, torch_dtype=dtype)
    # BrainDiVE pure-discovery: we want the raw brain-guided image, not a
    # filtered one. SD's NSFW safety checker fires on some ROIs (e.g. faces,
    # which generates a close-up human face) and replaces the output with a
    # black 842-byte image — a false positive that looks identical to a failed
    # generation. Disable it so every ROI returns its actual image.
    pipe.safety_checker = None
    # Scheduler: DPM-Solver++ is the tuned baseline. Time-travel/resampling
    # (test 2) needs a stateless Markovian step, so it selects DDIM.
    sched_name = str(args.scheduler if args.scheduler is not None else gen_cfg.get("scheduler", "dpm")).lower()
    if sched_name == "ddim":
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    else:
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)
    pipe.brain_tweak = tweak  # <-- the entire brain->image coupling

    # --- Test 3 (readout-head guidance): differentiate a head on the UNet's own
    # mid-block features instead of the encoder. Gated by --readout-head.
    if args.readout_head:
        ck = torch.load(args.readout_head, map_location=device, weights_only=False)
        head = ReadoutHead(din=int(ck.get("din", 1280)), nroi=len(ck["rois"]))
        head.load_state_dict(ck["state"]); head.to(device).eval()
        for p in head.parameters():
            p.requires_grad_(False)
        pipe._mid_feats = {}
        pipe.unet.mid_block.register_forward_hook(
            lambda m, i, o: pipe._mid_feats.__setitem__("h", o))
        pipe.readout_head = head
        roi_name = args.readout_roi or "faces"
        pipe.readout_roi = ck["rois"].index(roi_name)
        print(f"[gen] readout-head ON: target ROI '{roi_name}' (idx {pipe.readout_roi} of {ck['rois']})")

    # --- Test 1 (gradient hygiene) + Test 2 (time-travel) knobs from config ---
    # Every one defaults to the baseline value, so an untouched config == before.
    hyg = dict(cfg.get("hygiene", {}))
    pipe.hygiene_cutouts = int(args.cutouts if args.cutouts is not None else hyg.get("cutouts", 0))
    pipe.hygiene_cut_pow = float(hyg.get("cut_pow", 1.0))
    pipe.hygiene_tv      = float(args.tv if args.tv is not None else hyg.get("tv", 0.0))
    pipe.hygiene_range   = float(args.range_w if args.range_w is not None else hyg.get("range", 0.0))
    tt = dict(cfg.get("timetravel", {}))
    pipe.timetravel_reps = int(args.tt_reps if args.tt_reps is not None else tt.get("reps", 1))
    pipe.timetravel_lo   = float(tt.get("lo", 0.10))
    pipe.timetravel_hi   = float(tt.get("hi", 0.60))
    if pipe.hygiene_cutouts or pipe.timetravel_reps > 1 or pipe.hygiene_tv or pipe.hygiene_range:
        print(f"[gen] hygiene: cutouts={pipe.hygiene_cutouts} tv={pipe.hygiene_tv} "
              f"range={pipe.hygiene_range} | timetravel: reps={pipe.timetravel_reps} "
              f"win=[{pipe.timetravel_lo},{pipe.timetravel_hi}] sched={sched_name}")

    # BrainDiVE-style: empty prompt (pass --prompt "") lets the brain gradient,
    # not text, drive the content. args.prompt="" is honored (only None falls back).
    prompt = args.prompt if args.prompt is not None else gen_cfg.get("prompt", "")
    # Negative prompt is a content-neutral quality lever: it steers AWAY from
    # ugliness (blur, artifacts) without injecting any object category, so it
    # improves fidelity uniformly across ROIs without confounding discovery.
    negative_prompt = (args.negative_prompt if args.negative_prompt is not None
                       else gen_cfg.get("negative_prompt", None))
    seed = args.seed if args.seed is not None else gen_cfg.get("seed", 0)
    generator = torch.Generator(device=device).manual_seed(int(seed))

    out = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=gen_cfg.get("num_inference_steps", 50),
        guidance_scale=gen_cfg.get("guidance_scale", 7.5),
        sag_scale=gen_cfg.get("sag_scale", 0.0),
        clip_guidance_scale=(args.clip_scale if args.clip_scale is not None
                             else gen_cfg.get("clip_guidance_scale", 0.15)),
        num_images_per_prompt=gen_cfg.get("num_images_per_prompt", 1),
        generator=generator,
    )
    out_dir = Path(args.out or cfg.get("out_dir", "outputs"))
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, im in enumerate(out.images):
        p = out_dir / f"{tag}_seed{seed}_{i}.png"
        im.save(p)
        print(f"[gen] saved {p}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--smoke", action="store_true",
                    help="validate encoder + gradient flow without loading SD")
    ap.add_argument("--prompt", default=None,
                    help='text prompt; pass "" for BrainDiVE-style brain-only generation')
    ap.add_argument("--negative-prompt", dest="negative_prompt", default=None,
                    help="content-neutral quality negative prompt (steers away from "
                         "blur/artifacts without biasing ROI content)")
    ap.add_argument("--voxels", default=None,
                    help="override steer.voxels: path to a .npy index file (a real ROI)")
    ap.add_argument("--tag", default=None, help="label for output filenames")
    ap.add_argument("--clip-scale", dest="clip_scale", type=float, default=None,
                    help="override generate.clip_guidance_scale (guidance strength)")
    # experiment knobs (test 1 / test 2) — override the config's hygiene/timetravel
    ap.add_argument("--cutouts", type=int, default=None, help="test1: N augmentation crops (0=off)")
    ap.add_argument("--tv", type=float, default=None, help="test1: total-variation weight")
    ap.add_argument("--range", dest="range_w", type=float, default=None, help="test1: out-of-range penalty")
    ap.add_argument("--tt-reps", dest="tt_reps", type=int, default=None, help="test2: time-travel repeats (1=off)")
    ap.add_argument("--scheduler", default=None, help="dpm|ddim (ddim required for time-travel)")
    ap.add_argument("--readout-head", dest="readout_head", default=None,
                    help="test3: path to readout_head.pt (guide via UNet-feature head)")
    ap.add_argument("--readout-roi", dest="readout_roi", default=None,
                    help="test3: target ROI name for the readout head")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    if args.smoke:
        smoke(cfg)
    else:
        generate(cfg, args)


if __name__ == "__main__":
    main()
