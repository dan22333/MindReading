#!/usr/bin/env python
"""Phase 0 — build the negative pool + real-face baseline for the iterative novelty loop.

One image pool (NSD naturals we have + optional COCO val2017) → BLIP-labelled once into:
  - clean NEGATIVES   (no human/animal face)      -> cache embeddings  weights/neg_emb.pt
  - real FACE set     (BLIP: human face)          -> cache embeddings  weights/face_ref_emb.pt
Then compute the real-face z_faces baseline (ours, fp16) as the yardstick.

All CLIP encoding is fp16 (autocast) via a DataLoader(num_workers, pin_memory)+non_blocking
pipeline with pre-resize. DINOv2/BrainDiVE baselines are done separately (docker), fp16.

  python prep_phase0.py --coco 1 --n_nsd 10000 --out weights/
Dry run: python prep_phase0.py --coco 0 --n_nsd 200 --limit 200
"""
import argparse, os, glob, time
from pathlib import Path
import numpy as np
import torch
from PIL import Image

BC = Path(__file__).resolve().parent
FM, FS = -0.0476, 0.2003   # placeholder faces baseline (recomputed below in fp16)
COCO_URL = "http://images.cocodataset.org/zips/val2017.zip"


def ensure_coco(root: Path):
    d = root / "coco_val2017"
    if (d / "val2017").exists():
        return sorted(glob.glob(str(d / "val2017" / "*.jpg")))
    d.mkdir(parents=True, exist_ok=True)
    print("[coco] downloading val2017 (~1GB) ...", flush=True)
    os.system(f"cd {d} && wget -q {COCO_URL} && unzip -q val2017.zip && rm val2017.zip")
    return sorted(glob.glob(str(d / "val2017" / "*.jpg")))


class ImgDS(torch.utils.data.Dataset):
    """Yields (preprocessed CLIP tensor, index). Sources: NSD array rows + COCO jpg paths."""
    def __init__(self, nsd_imgs, nsd_idx, coco_paths, preprocess):
        self.nsd_imgs, self.nsd_idx = nsd_imgs, nsd_idx
        self.coco_paths = coco_paths
        self.pre = preprocess
        self.n_nsd = len(nsd_idx)

    def __len__(self):
        return self.n_nsd + len(self.coco_paths)

    def __getitem__(self, i):
        if i < self.n_nsd:
            arr = np.asarray(self.nsd_imgs[self.nsd_idx[i]]).astype("uint8")
            img = Image.fromarray(arr).convert("RGB")
        else:
            img = Image.open(self.coco_paths[i - self.n_nsd]).convert("RGB")
        return self.pre(img), i


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coco", type=int, default=1)
    ap.add_argument("--n_nsd", type=int, default=10000)
    ap.add_argument("--limit", type=int, default=0, help="cap total pool (dry run)")
    ap.add_argument("--out", default=str(BC / "weights"))
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--no_blip", action="store_true")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    device = "cuda"
    import open_clip

    clip_model, _, preprocess = open_clip.create_model_and_transforms("ViT-L-14", pretrained="openai")
    clip_model = clip_model.eval().to(device)
    Wf = torch.as_tensor(torch.load(BC / "weights/W_faces_s1.pt", map_location="cpu")["W"],
                         dtype=torch.float32, device=device)

    nsd = np.load(BC / "data/s1_images.npy", mmap_mode="r")
    nsd_idx = np.arange(min(args.n_nsd, nsd.shape[0]))
    coco_paths = ensure_coco(Path(args.out).parent / "data") if args.coco else []
    ds = ImgDS(nsd, nsd_idx, coco_paths, preprocess)
    keep = len(ds) if not args.limit else min(args.limit, len(ds))
    print(f"[pool] NSD {len(nsd_idx)} + COCO {len(coco_paths)} = {len(ds)} (using {keep})", flush=True)

    dl = torch.utils.data.DataLoader(ds, batch_size=args.batch, num_workers=args.workers,
                                     pin_memory=True, shuffle=False)
    embs = torch.empty(len(ds), 768, dtype=torch.float32)
    t0 = time.time(); done = 0
    with torch.no_grad():
        for px, ii in dl:
            px = px.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16):
                e = clip_model.encode_image(px)
            e = e.float(); e = e / e.norm(p=2, dim=-1, keepdim=True)
            embs[ii] = e.cpu()
            done += len(ii)
            if done >= keep:
                break
    print(f"[encode] {done} imgs in {time.time()-t0:.1f}s (fp16) = {done/(time.time()-t0):.0f} img/s", flush=True)
    embs = embs[:keep]

    # --- BLIP label: human face? / animal? --- (batched)
    is_face = np.zeros(keep, bool); is_animal = np.zeros(keep, bool)
    if not args.no_blip:
        from transformers import Blip2Processor, Blip2ForConditionalGeneration
        proc = Blip2Processor.from_pretrained("Salesforce/blip2-flan-t5-xl", use_fast=False)
        blip = Blip2ForConditionalGeneration.from_pretrained(
            "Salesforce/blip2-flan-t5-xl", torch_dtype=torch.float16).to(device).eval()

        def ask(pil_batch, q):
            inp = proc(images=pil_batch, text=[q]*len(pil_batch), return_tensors="pt",
                       padding=True).to(device, torch.float16)
            with torch.no_grad():
                o = blip.generate(**inp, max_new_tokens=3)
            return [a.strip().lower().startswith("yes") for a in proc.batch_decode(o, skip_special_tokens=True)]

        QF = "Question: Is there a human face in this image? Answer yes or no. Answer:"
        QA = "Question: Is there an animal in this image? Answer yes or no. Answer:"
        BS = 32
        for i in range(0, keep, BS):
            pil = []
            for j in range(i, min(i+BS, keep)):
                if j < len(nsd_idx):
                    pil.append(Image.fromarray(np.asarray(nsd[nsd_idx[j]]).astype("uint8")).convert("RGB").resize((224,224)))
                else:
                    pil.append(Image.open(coco_paths[j-len(nsd_idx)]).convert("RGB").resize((224,224)))
            is_face[i:i+len(pil)] = ask(pil, QF)
            is_animal[i:i+len(pil)] = ask(pil, QA)
            if i % (BS*20) == 0:
                print(f"[blip] {i}/{keep}", flush=True)

    neg_mask = ~(is_face | is_animal)          # clean negatives: no face, no animal
    face_mask = is_face                         # real-face reference
    neg_emb = embs[torch.from_numpy(neg_mask)]
    face_emb = embs[torch.from_numpy(face_mask)]
    torch.save({"emb": neg_emb}, out / "neg_emb.pt")
    torch.save({"emb": face_emb}, out / "face_ref_emb.pt")
    print(f"[split] negatives {neg_emb.shape[0]}  real-faces {face_emb.shape[0]}  "
          f"(animals dropped {int(is_animal.sum())})", flush=True)

    # --- real-face baseline (ours, fp16-encoded embeddings) ---
    if face_emb.shape[0] > 0:
        z = ((face_emb.to(device) @ Wf.t()).mean(1) - FM) / FS
        print(f"[REAL-FACE baseline] ours z_faces: mean {z.mean():.2f}  median {z.median():.2f}  "
              f"p10 {z.kthvalue(max(1,int(0.1*len(z))))[0]:.2f}  n={len(z)}", flush=True)
    print("PREP_PHASE0_DONE", flush=True)


if __name__ == "__main__":
    main()
