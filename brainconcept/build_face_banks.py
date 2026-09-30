#!/usr/bin/env python
"""Phase 0 (round 2) — build face + non-face embedding banks in the PRIOR (training) space.

Generates CLIP image embeddings by running the Kandinsky prior on many face and non-face
phrasings — the exact space the optimizer works in. Outputs:
  weights/face_bank.pt     {img_set:[Nf,768], terms}   (also the E2 "large bank" anchor_set)
  weights/nonface_bank.pt  {img_set:[Nn,768], terms}   (classifier negatives)

Run from vendor_conceptlab/ with PYTHONPATH=.  (env ~/braindive_venv).
"""
import argparse
from pathlib import Path
import torch
from kandinsky2 import load_utils
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_anchor import prior_image_embs   # reuse the exact prior path

FACE_TERMS = [
    "a human face", "a face", "a portrait of a person", "a close-up of a human face", "a headshot",
    "the face of a man", "the face of a woman", "a face with eyes, nose and mouth", "a human facial portrait",
    "a person looking at the camera", "a smiling face", "a crying face", "an angry face", "a child's face",
    "an old person's face", "a baby face", "a face in profile", "a bearded man's face", "a woman's face with makeup",
    "a face wearing glasses", "a selfie", "a mugshot", "a passport photo", "a face with open mouth",
    "a surprised face", "a face with closed eyes", "a laughing face", "a serious face", "a face looking sideways",
    "a wrinkled elderly face", "a teenager's face", "a face with freckles", "a painted portrait of a face",
    "a statue of a human head", "a cartoon face", "a face mask of a person", "a doll's face",
    "a face partially in shadow", "two people's faces", "a crowd of faces",
]
NONFACE_TERMS = [
    "a car", "a mountain landscape", "a bowl of fruit", "a city street", "a wooden chair", "a laptop computer",
    "a dog running", "a cat sleeping", "a plate of food", "a forest", "an ocean beach", "a bookshelf",
    "a bicycle", "a cup of coffee", "a flower garden", "a bridge", "a train station", "a kitchen",
    "a pair of shoes", "a guitar", "a tree", "a river", "a building", "a clock", "a keyboard", "a bottle",
    "a bird flying", "a fish", "a snowy field", "a desert", "a page of text", "handwritten letters",
    "a road sign", "a human hand", "a person's legs", "a soccer ball", "clouds in the sky", "a rocky cliff",
    "a bunch of grapes", "an abstract texture",
]


def build(model, terms, per_term, batch, device):
    out = []
    for t in terms:
        got = 0
        while got < per_term:
            b = min(batch, per_term - got)
            e = prior_image_embs(model, t, batch_size=b)
            e = e / e.norm(p=2, dim=-1, keepdim=True)
            out.append(e.to(device)); got += b
        print(f"  {t!r}: {per_term}")
    return torch.cat(out, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per_term", type=int, default=25)
    ap.add_argument("--batch", type=int, default=25)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--cache_root", default="/tmp/kandinsky2")
    ap.add_argument("--out_face", default="weights/face_bank.pt")
    ap.add_argument("--out_nonface", default="weights/nonface_bank.pt")
    args = ap.parse_args()
    print("loading Kandinsky ...")
    model = load_utils.get_model(Path(args.cache_root), args.device)
    print(f"FACE bank: {len(FACE_TERMS)} terms x {args.per_term}")
    face = build(model, FACE_TERMS, args.per_term, args.batch, args.device)
    torch.save({"img_set": face.cpu(), "terms": FACE_TERMS}, args.out_face)
    print(f"saved {args.out_face}  {tuple(face.shape)}")
    print(f"NON-FACE bank: {len(NONFACE_TERMS)} terms x {args.per_term}")
    non = build(model, NONFACE_TERMS, args.per_term, args.batch, args.device)
    torch.save({"img_set": non.cpu(), "terms": NONFACE_TERMS}, args.out_nonface)
    print(f"saved {args.out_nonface}  {tuple(non.shape)}")
    print("BUILD_FACE_BANKS_DONE")


if __name__ == "__main__":
    main()
