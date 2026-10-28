#!/usr/bin/env python
"""Build thumbnail assets for the 2-round check report (run in braindive_venv on the L4).

Produces ~small JPEGs under report_assets/:
  faces_sample/   real NSD faces  -> round-0 classifier POSITIVES
  neg_sample/     clean NSD imgs  -> classifier NEGATIVES (no face, no animal)
  animal_sample/  NSD animals     -> DROPPED (neither pos nor neg)
  r1/  r2/        the round's strongest held-out DINOv2 DRIVERS (= that round's new positives)
Writes report_assets/manifest.json with per-image labels/scores.
"""
import glob, json, os
from pathlib import Path
from collections import Counter
import numpy as np
import torch
from PIL import Image

BC = Path(__file__).resolve().parent
OUT = BC / "report_assets"
for d in ["faces_sample", "neg_sample", "animal_sample", "r1", "r2"]:
    (OUT / d).mkdir(parents=True, exist_ok=True)
device = "cuda"
TH = 256   # thumbnail size


def save_thumb(arr, path, q=82):
    im = Image.fromarray(arr).convert("RGB")
    im.thumbnail((TH, TH))
    im.save(path, "JPEG", quality=q)


def tiles(path):
    im = np.asarray(Image.open(path).convert("RGB"))
    h, w = im.shape[:2]; n = max(1, round(w / h)); ts = w // n
    return [im[:, i*ts:(i+1)*ts] for i in range(n)]


def step_of(p):
    d = "".join(c if c.isdigit() else " " for c in Path(p).stem).split()
    return int(d[0]) if d else -1


manifest = {"faces_sample": [], "neg_sample": [], "animal_sample": [], "r1": [], "r2": []}

# ---- round drivers (strongest held-out DINOv2 tiles) ----
for R, key in [(1, "r1"), (2, "r2")]:
    run_dirs = sorted(glob.glob(str(BC / f"vendor_conceptlab/outputs/loop_r{R}_s*")))
    pool = []  # (z, tile_arr, run)
    for d in run_dirs:
        run = Path(d).name
        cands = [x for x in glob.glob(d + "/images/*_step_images.jpeg") if "init" not in x]
        if not cands:
            continue
        T = tiles(max(cands, key=step_of))
        dj = BC / f"RESULTS/heldout/dino_{run}.json"
        if dj.exists():
            dd = json.load(open(dj)); k = str(max(int(x) for x in dd["steps"]))
            z = np.asarray(dd["steps"][k]["z_tiles"])
        else:
            z = np.full(len(T), np.nan)
        for i in range(min(len(T), len(z))):
            pool.append((float(z[i]), T[i], run))
    pool = [p for p in pool if not np.isnan(p[0])]
    pool.sort(key=lambda x: -x[0])
    for rank, (z, arr, run) in enumerate(pool[:12]):
        fn = f"{key}_{rank:02d}_z{z:+.2f}.jpg"
        save_thumb(arr, OUT / key / fn)
        manifest[key].append({"file": f"{key}/{fn}", "dino_z": round(z, 2), "run": run})
    print(f"[{key}] saved {min(12,len(pool))} driver tiles; top z={pool[0][0]:+.2f}" if pool else f"[{key}] none", flush=True)

# ---- classifier positives/negatives (round 0 training composition) via BLIP on NSD sample ----
from transformers import Blip2Processor, Blip2ForConditionalGeneration
proc = Blip2Processor.from_pretrained("Salesforce/blip2-flan-t5-xl", use_fast=False)
blip = Blip2ForConditionalGeneration.from_pretrained(
    "Salesforce/blip2-flan-t5-xl", torch_dtype=torch.float16).to(device).eval()


def ask(pil, q):
    inp = proc(images=pil, text=q, return_tensors="pt").to(device, torch.float16)
    with torch.no_grad():
        o = blip.generate(**inp, max_new_tokens=4)
    return proc.batch_decode(o, skip_special_tokens=True)[0].strip().lower()


QF = "Question: Is there a human face in this image? Answer yes or no. Answer:"
QA = "Question: Is there an animal in this image? Answer yes or no. Answer:"
QO = "Question: What is the main object in this image? Answer:"
nsd = np.load(BC / "data/s1_images.npy", mmap_mode="r")
rng = np.linspace(0, nsd.shape[0]-1, 160).astype(int)   # spread sample, deterministic
need = {"faces_sample": 8, "neg_sample": 8, "animal_sample": 6}
for j in rng:
    if all(len(manifest[k]) >= need[k] for k in need):
        break
    arr = np.asarray(nsd[j]).astype("uint8")
    pil = Image.fromarray(arr).convert("RGB").resize((224, 224))
    face = ask(pil, QF).startswith("yes"); animal = ask(pil, QA).startswith("yes")
    if face and len(manifest["faces_sample"]) < need["faces_sample"]:
        bucket, label = "faces_sample", ask(pil, QO)
    elif animal and not face and len(manifest["animal_sample"]) < need["animal_sample"]:
        bucket, label = "animal_sample", ask(pil, QO)
    elif (not face and not animal) and len(manifest["neg_sample"]) < need["neg_sample"]:
        bucket, label = "neg_sample", ask(pil, QO)
    else:
        continue
    fn = f"{bucket}_{len(manifest[bucket]):02d}.jpg"
    save_thumb(arr, OUT / bucket / fn)
    manifest[bucket].append({"file": f"{bucket}/{fn}", "label": label})
    print(f"[{bucket}] {fn}: {label}", flush=True)

json.dump(manifest, open(OUT / "manifest.json", "w"), indent=2)
print("REPORT_ASSETS_DONE", flush=True)
