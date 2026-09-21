#!/usr/bin/env python
"""Honest novelty judge for the floor-loss runs: BLIP-2 face yes/no.

cos-to-anchor is gameable (the model can decorate a face to slide the cosine while
staying a face). The TRUE question is categorical: "is there still a human face?".
We ask BLIP-2 (flan-t5-xl) directly, per tile, and report the face RATE.

Each ConceptLab step image is an N-tile horizontal grid (one tile per inference seed).
We judge every tile of every step grid matched by --images, and summarise per step and
for the final step (the delivered concept).

Runs in ~/braindive_venv (transformers + BLIP-2, already used by the loop).
"""
import argparse, glob, json, os
from pathlib import Path
import numpy as np


def load_grid_tiles(path):
    from PIL import Image
    im = np.asarray(Image.open(path).convert("RGB"))
    h, w = im.shape[:2]
    n = max(1, int(round(w / h)))
    ts = w // n
    return [im[:, i * ts:(i + 1) * ts] for i in range(n)]


def step_key(path):
    b = Path(path).stem
    if "init" in b:
        return 0
    digits = "".join(c if c.isdigit() else " " for c in b).split()
    return int(digits[0]) if digits else 10 ** 9


QUESTION = "Question: Is there a human face in this image? Answer yes or no. Answer:"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True, help="glob of step grid jpegs")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="Salesforce/blip2-flan-t5-xl")
    ap.add_argument("--question", default=QUESTION)
    args = ap.parse_args()

    import torch
    from PIL import Image
    from transformers import Blip2Processor, Blip2ForConditionalGeneration
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    proc = Blip2Processor.from_pretrained(args.model)
    model = Blip2ForConditionalGeneration.from_pretrained(args.model, torch_dtype=dtype).to(device).eval()

    def is_face(tile_uint8):
        im = Image.fromarray(tile_uint8)
        inp = proc(images=im, text=args.question, return_tensors="pt").to(device, dtype)
        with torch.no_grad():
            out = model.generate(**inp, max_new_tokens=5)
        ans = proc.batch_decode(out, skip_special_tokens=True)[0].strip().lower()
        return ans.startswith("yes"), ans

    files = sorted(glob.glob(args.images), key=step_key)
    steps = {}
    for f in files:
        tiles = load_grid_tiles(f)
        flags, answers = [], []
        for t in tiles:
            yes, ans = is_face(t)
            flags.append(bool(yes)); answers.append(ans)
        rate = float(np.mean(flags)) if flags else 0.0
        steps[str(step_key(f))] = {"file": os.path.basename(f), "n_tiles": len(tiles),
                                   "face_rate": rate, "flags": flags, "answers": answers}
        print(f"  step {step_key(f):>5}: face_rate={rate:.2f}  ({sum(flags)}/{len(flags)})", flush=True)

    final_key = max((int(k) for k in steps), default=None)
    final_rate = steps[str(final_key)]["face_rate"] if final_key is not None else None
    out = {"images_glob": args.images, "question": args.question,
           "final_step": final_key, "final_face_rate": final_rate, "steps": steps}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"[done] final_face_rate={final_rate} -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
