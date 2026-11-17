#!/usr/bin/env python3
"""CONTROL run: plain SD2.1-base (their exact model/scheduler) with a text prompt
per ROI group, NO brain encoder / NO guidance. Purpose: confirm the diffusion
pipeline itself produces coherent images, independent of the encoder steering.

This deliberately uses a vanilla StableDiffusionPipeline (not their SAG/brain
pipeline) so nothing but text conditioning drives the image — the cleanest test
that "the diffusion works." Same base weights + DPM scheduler as the real runner.

Run (GPU VM, braindive env):
    python run_control_plainSD.py --out outputs_control --seeds 0 1 2 3 4
"""
import argparse, os
import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

# One neutral prompt per ROI group — the categories our brain-guided run targets.
# Kept plain/literal so a coherent render clearly means the pipeline is healthy.
PROMPTS = {
    "faces": "a photograph of a human face",
    "bodies": "a photograph of a human body",
    "places": "a photograph of a place, an outdoor scene",
    "words": "a photograph of printed words and text",
    "earlyvisual": "a colorful high-frequency abstract texture pattern",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs_control")
    ap.add_argument("--rois", nargs="+", default=list(PROMPTS.keys()))
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--guidance", type=float, default=7.5)  # normal text-CFG
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    print("Loading SD2.1-base (plain text2img, no brain encoder)")
    pipe = StableDiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2-1-base", torch_dtype=torch.float16, revision="fp16")
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to("cuda")
    pipe.set_progress_bar_config(disable=True)

    for roi in args.rois:
        prompt = PROMPTS[roi]
        print(f"\n=== CONTROL {roi}: '{prompt}' ===")
        for seed in args.seeds:
            out = os.path.join(args.out, f"{roi}_seed{seed}.png")
            if os.path.exists(out):
                print("skip", out); continue
            g = torch.Generator(device="cuda").manual_seed(int(seed))
            img = pipe(prompt, num_inference_steps=args.steps,
                       guidance_scale=args.guidance, generator=g).images[0]
            img.save(out, format="PNG")
            print("wrote", out)


if __name__ == "__main__":
    main()
