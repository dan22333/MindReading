#!/usr/bin/env python
"""Recompute the image<->image anchor cosine on the DECODED final images (the loss's
real novelty signal), for all 12 runs. E1 runs use anchor_face_e1.pt, E2 runs anchor_face_e2.pt.
open_clip ViT-L-14 (openai) == the same CLIP space the Kandinsky prior/anchor live in."""
import glob, json
from pathlib import Path
import numpy as np, torch, open_clip
from PIL import Image

BC = Path("/home/danielnurieli/MindReading/brainconcept")
dev = "cuda"
model,_,pre = open_clip.create_model_and_transforms("ViT-L-14", pretrained="openai")
model = model.eval().to(dev)

def load_anchor(p):
    a = torch.load(p, map_location="cpu"); a = a["anchor"] if isinstance(a,dict) else a
    a = a.float(); return (a/a.norm()).to(dev)              # [768]

anchors = {"e1": load_anchor(BC/"vendor_conceptlab/weights/anchor_face_e1.pt"),
           "e2": load_anchor(BC/"vendor_conceptlab/weights/anchor_face_e2.pt")}

def tiles(path):
    im = np.asarray(Image.open(path).convert("RGB")); h,w = im.shape[:2]
    n = max(1, round(w/h)); ts = w//n
    return [Image.fromarray(im[:, i*ts:(i+1)*ts]) for i in range(n)]

runs = [("floor_faces_"+k,"e1") for k in ["z1_w1","z2_w1","z3_w1","z4_w1","z2_w05","z2_w2"]] + \
       [("floor_e2_faces_"+k,"e2") for k in ["z1_w1","z2_w1","z3_w1","z4_w1","z2_w05","z2_w2"]]

for run, fam in runs:
    fs = glob.glob(str(BC/f"vendor_conceptlab/outputs/{run}/images/2000_step_images.jpeg"))
    if not fs: print(run,"MISSING"); continue
    ts = tiles(fs[0])
    px = torch.stack([pre(t) for t in ts]).to(dev)
    with torch.no_grad():
        e = model.encode_image(px).float(); e = e/e.norm(dim=-1,keepdim=True)
        cos = (e @ anchors[fam]).mean().item()
    print(f"{run} anchor_cos={cos:+.3f}")
