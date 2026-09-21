#!/usr/bin/env python
"""Load-once scorer for the floor sweep (BLIP-2 + ours). Loads BLIP-2 and OpenCLIP
ONCE and the natural-image baseline ONCE, then iterates all runs -> writes
blip_<run>.json and ours_<run>.json (same schema as judge_blip_face / score_heldout).
Replaces the per-run model reloads. Runs in ~/braindive_venv.
"""
import argparse, glob, json, os
from pathlib import Path
import numpy as np

BC = Path.home() / "MindReading/brainconcept"


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
    d = "".join(c if c.isdigit() else " " for c in b).split()
    return int(d[0]) if d else 10 ** 9


QUESTION = "Question: Is there a human face in this image? Answer yes or no. Answer:"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--roi", default="faces")
    ap.add_argument("--category_anchor", default="a human face")
    ap.add_argument("--out", default=str(BC / "RESULTS/floor"))
    ap.add_argument("--weights_dir", default=str(BC / "weights"))
    ap.add_argument("--baseline_images", default=str(BC / "data/s1_images.npy"))
    ap.add_argument("--baseline_split", default=str(BC / "data/s1_split.npz"))
    ap.add_argument("--baseline_n", type=int, default=300)
    ap.add_argument("--blip_model", default="Salesforce/blip2-flan-t5-xl")
    args = ap.parse_args()
    outdir = Path(args.out); outdir.mkdir(parents=True, exist_ok=True)

    import torch, open_clip
    from PIL import Image
    from transformers import Blip2Processor, Blip2ForConditionalGeneration
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    # --- load ONCE: ours (OpenCLIP + W + text anchor) ---
    clip_model, _, preprocess = open_clip.create_model_and_transforms("ViT-L-14", pretrained="openai")
    clip_model = clip_model.eval().to(device)
    W = torch.as_tensor(torch.load(Path(args.weights_dir) / f"W_{args.roi}_s1.pt",
                                   map_location="cpu")["W"], dtype=torch.float32, device=device)
    tok = open_clip.get_tokenizer("ViT-L-14")
    with torch.no_grad():
        a = clip_model.encode_text(tok([args.category_anchor]).to(device)).float()
        text_anchor = a / a.norm(p=2, dim=-1, keepdim=True)

    def ours_emb(uint8_imgs):
        px = torch.stack([preprocess(Image.fromarray(x)) for x in uint8_imgs]).to(device)
        with torch.no_grad():
            e = clip_model.encode_image(px).float()
            return e / e.norm(p=2, dim=-1, keepdim=True)

    # --- baseline ONCE ---
    imgs = np.load(args.baseline_images, mmap_mode="r")
    idx = (np.load(args.baseline_split)["test_idx"] if Path(args.baseline_split).exists()
           else np.arange(imgs.shape[0]))[: args.baseline_n]
    bs = []
    for i in range(0, len(idx), 16):
        chunk = [np.asarray(imgs[j]).astype("uint8") for j in idx[i:i + 16]]
        e = ours_emb(chunk)
        bs.append((e @ W.t()).mean(dim=1).cpu().numpy())
    bs = np.concatenate(bs)
    bmean, bstd = float(bs.mean()), float(bs.std() + 1e-8)
    print(f"[baseline] ours/{args.roi}: mean={bmean:.4f} std={bstd:.4f} n={len(idx)}", flush=True)

    # --- load ONCE: BLIP-2 --- (use_fast=False: this venv's tokenizers lib can't parse the
    # newer flan-t5 fast tokenizer.json -> "PyPreTokenizerTypeWrapper"; slow tokenizer is fine)
    proc = Blip2Processor.from_pretrained(args.blip_model, use_fast=False)
    blip = Blip2ForConditionalGeneration.from_pretrained(args.blip_model, torch_dtype=dtype).to(device).eval()

    def is_face(uint8_img):
        inp = proc(images=Image.fromarray(uint8_img), text=QUESTION, return_tensors="pt").to(device, dtype)
        with torch.no_grad():
            out = blip.generate(**inp, max_new_tokens=5)
        ans = proc.batch_decode(out, skip_special_tokens=True)[0].strip().lower()
        return ans.startswith("yes"), ans

    # --- iterate runs ---
    for run in args.runs:
        d = BC / f"vendor_conceptlab/outputs/{run}/images"
        files = sorted(glob.glob(str(d / "*images.jpeg")), key=step_key)
        if not files:
            print(f"skip {run} (no images)", flush=True); continue
        ours_steps, blip_steps = {}, {}
        for f in files:
            tiles = load_grid_tiles(f)
            emb = ours_emb(tiles)
            raw = (emb @ W.t()).mean(dim=1).cpu().numpy()
            z = (raw - bmean) / bstd
            cs = (emb @ text_anchor.t()).squeeze(1).cpu().numpy()
            ours_steps[str(step_key(f))] = {"file": os.path.basename(f), "n_tiles": len(tiles),
                "raw_mean": float(raw.mean()), "z_mean": float(z.mean()), "z_max": float(z.max()),
                "z_tiles": [float(v) for v in z], "cat_sim": float(cs.mean()),
                "cat_sim_tiles": [float(v) for v in cs]}
            flags, answers = [], []
            for t in tiles:
                yes, ans = is_face(t); flags.append(bool(yes)); answers.append(ans)
            blip_steps[str(step_key(f))] = {"file": os.path.basename(f), "n_tiles": len(tiles),
                "face_rate": float(np.mean(flags)) if flags else 0.0, "flags": flags, "answers": answers}
        fk = max((int(k) for k in ours_steps), default=None)
        json.dump({"encoder": "ours", "roi": args.roi, "baseline": {"mean": bmean, "std": bstd, "n": len(idx)},
                   "steps": ours_steps}, open(outdir / f"ours_{run}.json", "w"), indent=2)
        json.dump({"question": QUESTION, "final_step": fk,
                   "final_face_rate": blip_steps[str(fk)]["face_rate"] if fk is not None else None,
                   "steps": blip_steps}, open(outdir / f"blip_{run}.json", "w"), indent=2)
        print(f"[{run}] final z={ours_steps[str(fk)]['z_mean']:+.2f} "
              f"cat_sim={ours_steps[str(fk)]['cat_sim']:+.3f} "
              f"face_rate={blip_steps[str(fk)]['face_rate']:.2f}", flush=True)
    print("SCORE_FLOOR_ALL_DONE", flush=True)


if __name__ == "__main__":
    main()
