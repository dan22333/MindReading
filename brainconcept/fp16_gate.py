#!/usr/bin/env python
"""fp16-agreement GATE for the CLIP ('ours') encoder + face probe.

Encodes the same images in fp32 and under autocast-fp16, then checks that the
embeddings / z_faces / P(face) agree within tolerance. Prints PASS/FAIL per metric
and a fp32-vs-fp16 timing. Only switch the encode path to fp16 if it PASSES.

Run: ~/braindive_venv/bin/python fp16_gate.py   (from brainconcept/)
"""
import time
from pathlib import Path
import numpy as np
import torch, open_clip
from PIL import Image

BC = Path(__file__).resolve().parent
N = 300
FM, FS = -0.0476, 0.2003   # faces baseline (ours)

device = "cuda"
clip_model, _, preprocess = open_clip.create_model_and_transforms("ViT-L-14", pretrained="openai")
clip_model = clip_model.eval().to(device)                       # fp32 weights
Wf = torch.as_tensor(torch.load(BC / "weights/W_faces_s1.pt", map_location="cpu")["W"],
                     dtype=torch.float32, device=device)
pb = torch.load(BC / "vendor_conceptlab/weights/face_probe.pt", map_location="cpu")
pw = torch.as_tensor(pb["w"], dtype=torch.float32, device=device).reshape(-1)
pbias = float(pb["b"])

imgs = np.load(BC / "data/s1_images.npy", mmap_mode="r")
idx = (np.load(BC / "data/s1_split.npz")["test_idx"])[:N]
px = torch.stack([preprocess(Image.fromarray(np.asarray(imgs[j]).astype("uint8"))) for j in idx]).to(device)


@torch.no_grad()
def enc(use_fp16):
    outs = []
    for i in range(0, len(px), 64):
        chunk = px[i:i + 64]
        if use_fp16:
            with torch.autocast("cuda", dtype=torch.float16):
                e = clip_model.encode_image(chunk)
        else:
            e = clip_model.encode_image(chunk)
        e = e.float(); e = e / e.norm(p=2, dim=-1, keepdim=True)
        outs.append(e)
    return torch.cat(outs, 0)


def zf(e):   # normalized faces-ROI z per image
    return ((e @ Wf.t()).mean(1) - FM) / FS


def pface(e):
    return torch.sigmoid(e @ pw + pbias)


# warm up, then time
_ = enc(False); torch.cuda.synchronize()
t0 = time.time(); e32 = enc(False); torch.cuda.synchronize(); t32 = time.time() - t0
t0 = time.time(); e16 = enc(True);  torch.cuda.synchronize(); t16 = time.time() - t0

cos = (e32 * e16).sum(-1)                              # per-image fp32-vs-fp16 agreement
dz = (zf(e32) - zf(e16)).abs()
dp = (pface(e32) - pface(e16)).abs()

print(f"[fp16 gate] N={N}")
print(f"  cos(fp32,fp16):  mean {cos.mean():.6f}  min {cos.min():.6f}   (want >= 0.9995)")
print(f"  |d z_faces|:     mean {dz.mean():.5f}  max {dz.max():.5f}     (want < 0.01)")
print(f"  |d P(face)|:     mean {dp.mean():.5f}  max {dp.max():.5f}     (want < 0.01)")
print(f"  encode time:     fp32 {t32:.3f}s   fp16 {t16:.3f}s   speedup {t32/max(t16,1e-9):.2f}x")
ok = (cos.min() >= 0.9995) and (dz.max() < 0.01) and (dp.max() < 0.01)
print("  RESULT:", "PASS -> safe to use fp16 for CLIP encode" if ok else "FAIL -> keep fp32")
print("FP16_GATE_DONE")
