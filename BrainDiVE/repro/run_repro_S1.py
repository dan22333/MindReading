#!/usr/bin/env python3
"""Faithful ORIGINAL-BrainDiVE runner for subject 1, cleaned of the author's
cluster paths. Uses their released encoder checkpoint + their exact generation
recipe, driving the SAME ROIs as our brainguide panel so results are comparable.

Reproduction proof: the functional voxel set rebuilt in prep_masks.py matches the
released S1 checkpoint EXACTLY (4657 early + 22961 higher), so the per-voxel
readout aligns with the ROI masks with no ambiguity.

Run (on the GPU VM, in the pre-May-2023 diffusers env):
    python run_repro_S1.py --ckpt /path/S1/00100.chkpt --repro . \
        --out outputs --rois faces bodies places words earlyvisual --seeds 0 1 2 3 4
"""
import argparse, os, sys
import numpy as np
import torch

# their modules live one dir up (the cloned BrainDiVE repo root)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import encoder_model_vit as model_vit          # noqa: E402
from brain_guide_pipeline import mypipelineSAG  # noqa: E402
from diffusers import DPMSolverMultistepScheduler  # noqa: E402

HIGHER_ROIS = {"faces", "bodies", "places", "words"}  # steered via forward_higher
# earlyvisual is steered via forward_early


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="S1/00100.chkpt (released weights)")
    ap.add_argument("--repro", default=".", help="dir with functional_S1.npy, *_vis_mask_S1.npy, masks/")
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--subject", type=int, default=1)
    ap.add_argument("--rois", nargs="+", default=["faces", "bodies", "places", "words", "earlyvisual"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--clip-guidance-scale", type=float, default=130.0)  # their OpenCLIP ViT-B default
    ap.add_argument("--sag-scale", type=float, default=0.75)
    ap.add_argument("--steps", type=int, default=50)
    # SD2.1-base source. stabilityai's repo is now HF-gated; sd2-community is an
    # un-gated, byte-identical diffusers mirror (same weights, incl. fp16 variant).
    ap.add_argument("--model-id", default="sd2-community/stable-diffusion-2-1-base")
    args = ap.parse_args()
    subj = args.subject
    os.makedirs(args.out, exist_ok=True)

    # --- CLIP ViT backbone (their laion2b ViT-B/16 feature extractor) ---
    print("Loading CLIP ViT backbone")
    backbone = model_vit.feature_extractor_vit()
    backbone.eval().cuda()
    for p in backbone.parameters():
        p.requires_grad = False

    # --- encoder: read voxel counts straight from the checkpoint (no neural data needed) ---
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=True)
    net = ck["network"] if "network" in ck else ck
    n_early = net[f"final_{subj}_early.weight"].shape[0]
    n_higher = net[f"final_{subj}_higher.weight"].shape[0]
    print(f"S{subj} encoder dims: early={n_early} higher={n_higher}")
    brain_model = model_vit.downproject_CLIP_split_linear(
        num_early_output={subj: n_early}, num_higher_output={subj: n_higher})
    brain_model.load_state_dict(net, strict=True)
    brain_model.eval().cuda()
    for p in brain_model.parameters():
        p.requires_grad = False

    # --- masks over the functional voxel set ---
    R = args.repro
    early_vis_mask = torch.from_numpy(np.load(f"{R}/early_vis_mask_S1.npy")).bool().cuda()
    higher_vis_mask = torch.from_numpy(np.load(f"{R}/higher_vis_mask_S1.npy")).bool().cuda()
    assert int(early_vis_mask.sum()) == n_early and int(higher_vis_mask.sum()) == n_higher, \
        "mask/encoder voxel-count mismatch — reproduction is NOT aligned!"

    def make_loss(mask_tensor, kind):
        def loss_higher(image_input):
            feats = backbone(image_input)
            pred = brain_model.forward_higher(feats[0][0], feats[0][1], feats[1], [subj]).reshape(-1)
            resp = torch.zeros(higher_vis_mask.shape, dtype=torch.float32, device="cuda")
            resp[higher_vis_mask] = pred
            return -torch.mean(resp[mask_tensor])
        def loss_early(image_input):
            feats = backbone(image_input)
            pred = brain_model.forward_early(feats[0][0], feats[0][1], feats[1], [subj]).reshape(-1)
            resp = torch.zeros(higher_vis_mask.shape, dtype=torch.float32, device="cuda")
            resp[early_vis_mask] = pred
            return -torch.mean(resp[mask_tensor])
        return loss_higher if kind == "higher" else loss_early

    # --- SD2.1-base pipeline, their exact config ---
    print(f"Loading SD2.1-base pipeline from {args.model_id}")
    try:  # fp16 weights live as a variant on the mirror's main branch
        pipe = mypipelineSAG.from_pretrained(args.model_id,
                                             torch_dtype=torch.float16, variant="fp16")
    except Exception as e:  # fall back to fp32 weights cast to fp16 (same math for inference)
        print(f"fp16 variant unavailable ({type(e).__name__}); loading default weights in fp16")
        pipe = mypipelineSAG.from_pretrained(args.model_id, torch_dtype=torch.float16)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to("cuda")

    for roi in args.rois:
        mask = torch.from_numpy(np.load(f"{R}/masks/{roi}_S1.npy")).bool().cuda()
        kind = "higher" if roi in HIGHER_ROIS else "early"
        pipe.brain_tweak = make_loss(mask, kind)
        print(f"\n=== ROI {roi} ({kind}); {int(mask.sum())} voxels ===")
        for seed in args.seeds:
            out = os.path.join(args.out, f"{roi}_seed{seed}.png")
            if os.path.exists(out):
                print("skip", out); continue
            g = torch.Generator(device="cuda").manual_seed(int(seed))
            image = pipe("", sag_scale=args.sag_scale, guidance_scale=0.0,
                         num_inference_steps=args.steps, generator=g,
                         clip_guidance_scale=args.clip_guidance_scale)
            image.images[0].save(out, format="PNG")
            print("wrote", out)


if __name__ == "__main__":
    main()
