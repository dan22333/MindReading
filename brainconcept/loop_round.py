#!/usr/bin/env python
"""Per-round processor for the iterative novelty loop (runs in braindive_venv).

Given this round's generated filmstrips (outputs/loop_r{R}_s*/images/1000_step_images.jpeg, 20
tiles each) and the docker DINOv2 scores (RESULTS/heldout/dino_loop_r{R}_s*.json, z_tiles):
  1. CLIP-encode every tile (fp16);  z_faces (ours) per tile.
  2. Select DRIVERS: tiles with held-out DINOv2 z >= --drive_thresh (real, not a hack).
  3. BLIP label drivers: human-face? + "what is the main object?" -> round theme + face-rate.
  4. Intra-round DIVERSITY: mean pairwise CLIP cos among drivers (doubles as the seed-diversity check).
  5. Expand archive.pt with the drivers' embeddings under mode  r{R}_{theme}.
  6. Write RESULTS/loop/round_{R}.json + a stop signal.

  python loop_round.py --round 1 --drive_thresh 1.5
"""
import argparse, glob, json, os
from collections import Counter
from pathlib import Path
import numpy as np
import torch

BC = Path(__file__).resolve().parent
FM, FS = -0.0476, 0.2003


def tiles(path):
    from PIL import Image
    im = np.asarray(Image.open(path).convert("RGB"))
    h, w = im.shape[:2]; n = max(1, round(w / h)); ts = w // n
    return [im[:, i*ts:(i+1)*ts] for i in range(n)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", type=int, required=True)
    ap.add_argument("--drive_thresh", type=float, default=1.5)
    ap.add_argument("--archive", default=str(BC / "weights/archive.pt"))
    ap.add_argument("--qwen", action="store_true")
    args = ap.parse_args()
    R = args.round
    outdir = BC / "RESULTS/loop"; outdir.mkdir(parents=True, exist_ok=True)
    device = "cuda"
    import open_clip
    from PIL import Image

    cm, _, pre = open_clip.create_model_and_transforms("ViT-L-14", pretrained="openai")
    cm = cm.eval().to(device)
    Wf = torch.as_tensor(torch.load(BC/"weights/W_faces_s1.pt", map_location="cpu")["W"],
                         dtype=torch.float32, device=device)

    @torch.no_grad()
    def enc(imgs):
        px = torch.stack([pre(Image.fromarray(x)) for x in imgs]).to(device)
        with torch.autocast("cuda", dtype=torch.float16):
            e = cm.encode_image(px)
        e = e.float(); return e / e.norm(dim=-1, keepdim=True)

    def step_of(p):
        d = "".join(c if c.isdigit() else " " for c in Path(p).stem).split()
        return int(d[0]) if d else -1
    # restrict to THIS round's runs (runs.txt) — a loose glob also matches stale same-name dirs
    runs_file = BC/f"loop_r{R}_runs.txt"
    if runs_file.exists():
        names = [l.strip() for l in open(runs_file) if l.strip()]
        run_dirs = [str(BC/f"vendor_conceptlab/outputs/{n}") for n in names]
    else:
        run_dirs = sorted(glob.glob(str(BC/f"vendor_conceptlab/outputs/loop_r{R}_s*")))
    runs = []
    for d in run_dirs:
        cands = [x for x in glob.glob(d + "/images/*_step_images.jpeg") if "init" not in x]
        if cands:
            runs.append(max(cands, key=step_of))     # the final (max-step) filmstrip
    if not runs:
        print(f"[round {R}] no generated images found -> nothing to process", flush=True)
        return
    allE, allzf, alldino, keep_imgs = [], [], [], []
    for f in runs:
        run = Path(f).parts[-3]
        T = tiles(f); e = enc(T)
        zf = ((e @ Wf.t()).mean(1) - FM) / FS
        dj = BC/f"RESULTS/heldout/dino_{run}.json"
        if dj.exists():
            dd = json.load(open(dj)); k = str(max(int(x) for x in dd["steps"]))
            dino = np.asarray(dd["steps"][k]["z_tiles"])
        else:
            dino = np.full(len(T), np.nan)
        for i in range(len(T)):
            allE.append(e[i].cpu()); allzf.append(float(zf[i])); alldino.append(float(dino[i])); keep_imgs.append(T[i])
    allE = torch.stack(allE); allzf = np.array(allzf); alldino = np.array(alldino)
    print(f"[round {R}] {len(allE)} tiles; dino z mean {np.nanmean(alldino):+.2f} max {np.nanmax(alldino):+.2f}", flush=True)

    # select real drivers by held-out DINOv2
    drv = np.where(alldino >= args.drive_thresh)[0]
    print(f"[round {R}] drivers (dino>= {args.drive_thresh}): {len(drv)}/{len(allE)}", flush=True)

    # BLIP label the drivers. NOTE: this is a LABEL for the archive, NOT a stop gate.
    # The question is BROAD (any face-like structure incl. animal/doll/mask/pareidolia) so
    # doll/mask faces are not mistaken for "non-face" — but it never halts the loop.
    face_like_rate, theme = float("nan"), "none"
    if len(drv):
        from transformers import Blip2Processor, Blip2ForConditionalGeneration
        proc = Blip2Processor.from_pretrained("Salesforce/blip2-flan-t5-xl", use_fast=False)
        blip = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-flan-t5-xl",
               torch_dtype=torch.float16).to(device).eval()

        def ask(img, q):
            inp = proc(images=Image.fromarray(img).resize((224,224)), text=q, return_tensors="pt").to(device, torch.float16)
            with torch.no_grad(): o = blip.generate(**inp, max_new_tokens=6)
            return proc.batch_decode(o, skip_special_tokens=True)[0].strip().lower()
        QF = ("Question: Is there any face or face-like structure (human, animal, doll, mask, "
              "or object pareidolia) in this image? Answer yes or no. Answer:")
        QO = "Question: What is the main object in this image? Answer:"
        faces, objs = [], []
        for i in drv:
            faces.append(ask(keep_imgs[i], QF).startswith("yes"))
            objs.append(ask(keep_imgs[i], QO))
        face_like_rate = float(np.mean(faces))
        theme = Counter([o.split()[-1] if o else "?" for o in objs]).most_common(1)[0][0]
        print(f"[round {R}] face_like_rate {face_like_rate:.2f}  theme '{theme}'  objs={Counter(objs).most_common(3)}", flush=True)

    # intra-round diversity (seed-diversity check): mean pairwise cos among drivers
    diversity = None
    if len(drv) > 1:
        D = allE[drv]; C = (D @ D.t()); n = len(drv)
        diversity = float((C.sum() - n) / (n*(n-1)))   # mean off-diagonal cosine (1=identical)
        print(f"[round {R}] intra-round mean pairwise cos = {diversity:.3f} (1.0=seeds NOT diversifying)", flush=True)

    # expand archive
    arch = torch.load(args.archive, map_location="cpu") if Path(args.archive).exists() else {}
    mode = f"r{R}_{theme}"
    if len(drv):
        arch[mode] = allE[drv].clone()
        torch.save(arch, args.archive)
        print(f"[round {R}] archive += '{mode}' ({len(drv)} embs); archive modes={list(arch.keys())}", flush=True)

    # stop signal: EXHAUSTION ONLY. A round is "dry" when no tile clears the held-out
    # DINOv2 drive bar (--drive_thresh). The loop stops on 2 consecutive dry rounds
    # (ROI can no longer be driven by anything novel) — NOT on any face judgement.
    # face_like_rate is recorded purely as a label describing what this round's drivers are.
    dry = bool(len(drv) == 0)
    meta = {"round": R, "n_tiles": int(len(allE)), "n_drivers": int(len(drv)),
            "drive_thresh": float(args.drive_thresh),
            "dino_mean": float(np.nanmean(alldino)), "dino_max": float(np.nanmax(alldino)),
            "face_like_rate": face_like_rate, "theme": theme, "diversity": diversity,
            "dry_round": dry, "archive_modes": list(arch.keys())}
    json.dump(meta, open(outdir/f"round_{R}.json", "w"), indent=2)
    print(f"[round {R}] {'DRY ROUND (no novel driver >= %.1f)' % args.drive_thresh if dry else 'driver(s) found -> archived + forbidden'}", flush=True)
    print("LOOP_ROUND_DONE", flush=True)


if __name__ == "__main__":
    main()
