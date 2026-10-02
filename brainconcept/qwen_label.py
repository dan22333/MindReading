#!/usr/bin/env python
"""Qwen2-VL labeller for review gates — richer discovery captions than BLIP.

Parses a round's filmstrips -> tiles -> asks Qwen2-VL-7B two questions per tile:
  (1) "What is the main subject? Answer in 1-3 words."
  (2) "Is there any face or face-like structure (human, animal, or object pareidolia)? yes/no."
Writes RESULTS/loop/qwen_round_{R}.json. Degrades gracefully if the env can't load Qwen2-VL
(old transformers) — prints what's missing so we can spin a dedicated qwen venv.

  python qwen_label.py --round 1   [--int4]
"""
import argparse, glob, json, sys
from pathlib import Path
from collections import Counter
import numpy as np

BC = Path(__file__).resolve().parent


def tiles(path):
    from PIL import Image
    im = np.asarray(Image.open(path).convert("RGB"))
    h, w = im.shape[:2]; n = max(1, round(w / h)); ts = w // n
    return [Image.fromarray(im[:, i*ts:(i+1)*ts]).resize((448, 448)) for i in range(n)]


def step_of(p):
    d = "".join(c if c.isdigit() else " " for c in Path(p).stem).split()
    return int(d[0]) if d else -1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", type=int, required=True)
    ap.add_argument("--int4", action="store_true")
    ap.add_argument("--model", default="Qwen/Qwen2-VL-7B-Instruct")
    ap.add_argument("--max_tiles", type=int, default=24)
    args = ap.parse_args()
    R = args.round
    try:
        import torch
        from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
    except Exception as e:
        print(f"QWEN_ENV_MISSING: {e}\n-> needs transformers>=4.45 + qwen-vl-utils; "
              f"create a dedicated venv (python -m venv ~/qwen_venv; pip install 'transformers>=4.45' "
              f"qwen-vl-utils accelerate bitsandbytes torch).", flush=True)
        sys.exit(3)

    device = "cuda"
    kw = dict(torch_dtype=torch.float16, device_map="auto")
    if args.int4:
        from transformers import BitsAndBytesConfig
        kw = dict(quantization_config=BitsAndBytesConfig(load_in_4bit=True,
                  bnb_4bit_compute_dtype=torch.float16), device_map="auto")
    print(f"[qwen] loading {args.model} ({'int4' if args.int4 else 'fp16'}) ...", flush=True)
    model = Qwen2VLForConditionalGeneration.from_pretrained(args.model, **kw).eval()
    proc = AutoProcessor.from_pretrained(args.model)

    run_dirs = sorted(glob.glob(str(BC / f"vendor_conceptlab/outputs/loop_r{R}_s*")))
    imgs = []
    for d in run_dirs:
        cands = [x for x in glob.glob(d + "/images/*_step_images.jpeg") if "init" not in x]
        if cands:
            imgs += tiles(max(cands, key=step_of))
    imgs = imgs[: args.max_tiles]
    print(f"[qwen] round {R}: {len(imgs)} tiles", flush=True)

    def ask(img, q):
        msg = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": q}]}]
        text = proc.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
        inp = proc(text=[text], images=[img], return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(**inp, max_new_tokens=16)
        return proc.batch_decode(out[:, inp.input_ids.shape[1]:], skip_special_tokens=True)[0].strip()

    Q1 = "What is the main subject of this image? Answer in 1 to 3 words."
    Q2 = "Is there any face or face-like structure (human, animal, or object pareidolia)? Answer yes or no."
    subjects, faces = [], []
    for im in imgs:
        s = ask(im, Q1).lower(); f = ask(im, Q2).lower().startswith("y")
        subjects.append(s); faces.append(f)
    theme = Counter([s.split()[-1] for s in subjects if s]).most_common(5)
    facelike = float(np.mean(faces)) if faces else float("nan")
    out = {"round": R, "n": len(imgs), "face_like_rate": facelike,
           "top_subjects": theme, "subjects": subjects, "faces": faces}
    (BC / "RESULTS/loop").mkdir(parents=True, exist_ok=True)
    json.dump(out, open(BC / f"RESULTS/loop/qwen_round_{R}.json", "w"), indent=2)
    print(f"[qwen] round {R}: face-like {facelike:.2f}  top={theme}", flush=True)
    print("QWEN_LABEL_DONE", flush=True)


if __name__ == "__main__":
    main()
