#!/usr/bin/env python
"""Load-once scorer for the contrast (E1/E2 v2) sweep. Loads OpenCLIP + BLIP-2 + both
ridge encoders (faces, complement) + the E1/E2 anchor sets ONCE, and the natural-image
baseline ONCE, then iterates all runs -> RESULTS/contrast/{ours,blip}_<run>.json.

Per checkpoint filmstrip (6 tiles) it records, on the DECODED images (open_clip re-encode):
  z_faces, z_comp, selectivity = z_faces - z_comp   (drive axis; each z vs its own baseline)
  img_cos_mean / img_cos_max  (novelty vs the run's E1/E2 image anchor set)
  cat_sim  (image vs CLIP-text "a human face")
  face_rate (BLIP-2 yes/no — the honest novelty metric)
and parses drift_cos (decode->re-encode) from the training log ~/contrast_<run>.log.

DINOv2 realness is scored separately (dino_all.py, docker). Runs in ~/braindive_venv.
"""
import argparse, glob, json, os, re
from pathlib import Path
import numpy as np

BC = Path.home() / "MindReading/brainconcept"
QUESTION = "Question: Is there a human face in this image? Answer yes or no. Answer:"


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


def parse_drift(logpath):
    """Map checkpoint step -> drift_cos, read from the training log. The drift line
    (`[drift] cos(X, enc(dec(X))) = <f>`) follows the `For step #<n>` of that iteration;
    the checkpoint file is named for <n>+1, so align to that."""
    out = {}
    if not Path(logpath).exists():
        return out
    cur = 0
    for line in open(logpath, errors="ignore"):
        m = re.search(r"For step #(\d+)", line)
        if m:
            cur = int(m.group(1)); continue
        m = re.search(r"\[drift\] cos\(X, enc\(dec\(X\)\)\) = ([\-0-9.]+)", line)
        if m:
            out[str(cur + 1)] = float(m.group(1))
    return out


def parse_pface(logpath):
    """Map step -> P(face) from the [contrast] training line (E3 runs only)."""
    out = {}
    if not Path(logpath).exists():
        return out
    cur = 0
    for line in open(logpath, errors="ignore"):
        m = re.search(r"For step #(\d+)", line)
        if m:
            cur = int(m.group(1)); continue
        m = re.search(r"p_face ([0-9.]+)", line)
        if m:
            out[str(cur + 1)] = float(m.group(1))
    return out


def anchor_for(run):
    if run.startswith("r2_"):
        return "bank"          # round 2 uses the large face bank as the monitoring anchor
    return "e1" if "_e1_" in run or run.endswith("_e1") else "e2"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out", default=str(BC / "RESULTS/contrast"))
    ap.add_argument("--weights_dir", default=str(BC / "vendor_conceptlab/weights"))
    ap.add_argument("--faces_weights", default=str(BC / "weights/W_faces_s1.pt"))
    ap.add_argument("--comp_weights", default=str(BC / "vendor_conceptlab/weights/W_comp_s1.pt"))
    ap.add_argument("--baseline_images", default=str(BC / "data/s1_images.npy"))
    ap.add_argument("--baseline_split", default=str(BC / "data/s1_split.npz"))
    ap.add_argument("--baseline_n", type=int, default=300)
    ap.add_argument("--blip_model", default="Salesforce/blip2-flan-t5-xl")
    ap.add_argument("--no_blip", action="store_true")
    args = ap.parse_args()
    outdir = Path(args.out); outdir.mkdir(parents=True, exist_ok=True)

    import torch, open_clip
    from PIL import Image
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    clip_model, _, preprocess = open_clip.create_model_and_transforms("ViT-L-14", pretrained="openai")
    clip_model = clip_model.eval().to(device)
    tok = open_clip.get_tokenizer("ViT-L-14")
    Wf = torch.as_tensor(torch.load(args.faces_weights, map_location="cpu")["W"],
                         dtype=torch.float32, device=device)
    Wc = torch.as_tensor(torch.load(args.comp_weights, map_location="cpu")["W"],
                         dtype=torch.float32, device=device)
    with torch.no_grad():
        a = clip_model.encode_text(tok(["a human face"]).to(device)).float()
        text_anchor = a / a.norm(p=2, dim=-1, keepdim=True)
    anchors = {}       # e -> image anchor set [N,768]
    anchors_txt = {}   # e -> text  anchor set [K,768]
    bankp = Path(args.weights_dir) / "face_bank.pt"
    if bankp.exists():
        im = torch.as_tensor(torch.load(bankp, map_location="cpu")["img_set"], dtype=torch.float32, device=device)
        anchors["bank"] = im / im.norm(p=2, dim=-1, keepdim=True)
    for e in ["e1", "e2"]:
        p = Path(args.weights_dir) / f"anchor_{e}.pt"
        if p.exists():
            s = torch.load(p, map_location="cpu")
            im = torch.as_tensor(s["img_set"], dtype=torch.float32, device=device)
            anchors[e] = im / im.norm(p=2, dim=-1, keepdim=True)
            if s.get("text_set") is not None:
                tx = torch.as_tensor(s["text_set"], dtype=torch.float32, device=device)
                anchors_txt[e] = tx / tx.norm(p=2, dim=-1, keepdim=True)

    def emb(uint8_imgs):
        px = torch.stack([preprocess(Image.fromarray(x)) for x in uint8_imgs]).to(device)
        with torch.no_grad():
            e = clip_model.encode_image(px).float()
            return e / e.norm(p=2, dim=-1, keepdim=True)

    # baselines (faces + complement) over the 300 natural test images
    imgs = np.load(args.baseline_images, mmap_mode="r")
    idx = (np.load(args.baseline_split)["test_idx"] if Path(args.baseline_split).exists()
           else np.arange(imgs.shape[0]))[: args.baseline_n]
    bf, bc = [], []
    for i in range(0, len(idx), 16):
        chunk = [np.asarray(imgs[j]).astype("uint8") for j in idx[i:i + 16]]
        e = emb(chunk)
        bf.append((e @ Wf.t()).mean(dim=1).cpu().numpy())
        bc.append((e @ Wc.t()).mean(dim=1).cpu().numpy())
    bf, bc = np.concatenate(bf), np.concatenate(bc)
    fm, fs = float(bf.mean()), float(bf.std() + 1e-8)
    cm, cs = float(bc.mean()), float(bc.std() + 1e-8)
    print(f"[baseline] faces mean={fm:.4f} std={fs:.4f} | comp mean={cm:.4f} std={cs:.4f} n={len(idx)}",
          flush=True)

    blip = proc = None
    if not args.no_blip:
        from transformers import Blip2Processor, Blip2ForConditionalGeneration
        proc = Blip2Processor.from_pretrained(args.blip_model, use_fast=False)
        blip = Blip2ForConditionalGeneration.from_pretrained(args.blip_model, torch_dtype=dtype).to(device).eval()

    def is_face(uint8_img):
        inp = proc(images=Image.fromarray(uint8_img), text=QUESTION, return_tensors="pt").to(device, dtype)
        with torch.no_grad():
            out = blip.generate(**inp, max_new_tokens=5)
        ans = proc.batch_decode(out, skip_special_tokens=True)[0].strip().lower()
        return ans.startswith("yes"), ans

    for run in args.runs:
        d = BC / f"vendor_conceptlab/outputs/{run}/images"
        files = sorted(glob.glob(str(d / "*images.jpeg")), key=step_key)
        if not files:
            print(f"skip {run} (no images)", flush=True); continue
        A = anchors[anchor_for(run)]
        Atxt = anchors_txt.get(anchor_for(run))
        drift = parse_drift(Path.home() / f"contrast_{run}.log")
        pface = parse_pface(Path.home() / f"contrast_{run}.log")
        ours_steps, blip_steps = {}, {}
        for f in files:
            tiles = load_grid_tiles(f)
            e = emb(tiles)
            zf = ((e @ Wf.t()).mean(dim=1).cpu().numpy() - fm) / fs
            zc = ((e @ Wc.t()).mean(dim=1).cpu().numpy() - cm) / cs
            sel = zf - zc
            ic = (e @ A.t())                                       # [tiles, N]
            ic_mean = ic.mean(dim=1).cpu().numpy()
            ic_max = ic.max(dim=1).values.cpu().numpy()
            cs_ = (e @ text_anchor.t()).squeeze(1).cpu().numpy()
            if Atxt is not None:
                tc = (e @ Atxt.t())                                   # [tiles, K]
                tc_mean = float(tc.mean(dim=1).mean()); tc_max = float(tc.max(dim=1).values.mean())
            else:
                tc_mean = tc_max = None
            k = str(step_key(f))
            ours_steps[k] = {"file": os.path.basename(f), "n_tiles": len(tiles),
                "z_faces": float(zf.mean()), "z_comp": float(zc.mean()),
                "selectivity": float(sel.mean()), "selectivity_max": float(sel.max()),
                "img_cos_mean": float(ic_mean.mean()), "img_cos_max": float(ic_max.mean()),
                "txt_cos_mean": tc_mean, "txt_cos_max": tc_max, "p_face": pface.get(k),
                "cat_sim": float(cs_.mean()), "drift_cos": drift.get(k),
                "z_faces_tiles": [float(v) for v in zf], "sel_tiles": [float(v) for v in sel]}
            if blip is not None:
                flags = [bool(is_face(t)[0]) for t in tiles]
                blip_steps[k] = {"file": os.path.basename(f), "n_tiles": len(tiles),
                    "face_rate": float(np.mean(flags)) if flags else 0.0, "flags": flags}
        fk = max((int(k) for k in ours_steps), default=None)
        json.dump({"encoder": "ours+comp", "run": run, "anchor": anchor_for(run),
                   "baseline": {"faces": [fm, fs], "comp": [cm, cs]},
                   "steps": ours_steps}, open(outdir / f"ours_{run}.json", "w"), indent=2)
        if blip is not None:
            json.dump({"question": QUESTION, "final_step": fk,
                       "final_face_rate": blip_steps[str(fk)]["face_rate"] if fk is not None else None,
                       "steps": blip_steps}, open(outdir / f"blip_{run}.json", "w"), indent=2)
        fr = blip_steps[str(fk)]["face_rate"] if (blip is not None and fk is not None) else float("nan")
        o = ours_steps[str(fk)]
        print(f"[{run}] sel={o['selectivity']:+.2f} (zf {o['z_faces']:+.2f} zc {o['z_comp']:+.2f}) "
              f"img_cos m/M={o['img_cos_mean']:.2f}/{o['img_cos_max']:.2f} "
              f"drift={o['drift_cos']} face_rate={fr:.2f}", flush=True)
    print("SCORE_CONTRAST_ALL_DONE", flush=True)


if __name__ == "__main__":
    main()
