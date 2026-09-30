#!/usr/bin/env python
"""Build a ~1000-phrase face TEXT bank (CLIP text embeddings) for the text-repel experiment.

Uses the Kandinsky model's own CLIP text encoder (same space as training) via build_anchor.text_embs.
Saves weights/text_bank.pt {img_set:[N,768], terms} — 'img_set' key so the existing anchor_set /
E2 image-cosine code path loads it unchanged; the loss then computes cos(image_emb, text_j).
"""
import argparse, itertools
from pathlib import Path
import torch, sys
from kandinsky2 import load_utils
sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_anchor import text_embs

ADJ = ["", "young", "old", "elderly", "smiling", "laughing", "crying", "angry", "surprised", "serious",
       "bearded", "wrinkled", "freckled", "happy", "sad", "tired", "excited", "calm", "frowning",
       "grinning", "sleepy", "worried", "confident", "shy", "makeup-wearing", "sunburned", "pale"]
SUBJ = ["man", "woman", "child", "person", "boy", "girl", "elderly man", "elderly woman", "baby",
        "teenager", "old lady", "young man", "middle-aged man", "toddler", "grandmother", "soldier"]
VIEW = ["face", "portrait", "headshot", "close-up of the face", "face in profile", "facial portrait",
        "face looking at the camera", "face with eyes nose and mouth", "selfie", "passport photo of the face"]
EXTRA = ["a human face", "a face", "a cartoon face", "a doll's face", "a statue of a human head",
         "a painted portrait", "a face mask", "a crowd of faces", "two faces", "a face partially in shadow"]


def phrasings(n_target):
    out = list(EXTRA)
    for adj, subj, view in itertools.product(ADJ, SUBJ, VIEW):
        p = f"a {adj+' ' if adj else ''}{subj}'s {view}".replace("  ", " ")
        out.append(p)
        if len(out) >= n_target:
            return out[:n_target]
    return out[:n_target]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--out", default="weights/text_bank.pt")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--cache_root", default="/tmp/kandinsky2")
    args = ap.parse_args()
    terms = phrasings(args.n)
    print(f"[text_bank] {len(terms)} phrasings (e.g. {terms[10:13]})")
    model = load_utils.get_model(Path(args.cache_root), args.device)
    embs = text_embs(model, terms, args.device)   # [N,768] normed CLIP text
    torch.save({"img_set": embs.cpu(), "terms": terms}, args.out)
    print(f"[text_bank] saved {args.out} {tuple(embs.shape)}")
    print("BUILD_TEXT_BANK_DONE")


if __name__ == "__main__":
    main()
