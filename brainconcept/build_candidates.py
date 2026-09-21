#!/usr/bin/env python
"""Build a CLIP concept vocabulary (WordNet nouns + xkcd colors + shapes/textures/materials),
embed it once with CLIP-text, then for each run rank its final images against the vocab and
print the top-K nearest concepts (what the learned token is 'close to' in CLIP space).

1-vs-all matmul — no KD-tree needed (vocab is ~tens of thousands × 768; a query is one matmul).
Run on the VM (open_clip in ~/braindive_venv). GPU if available else CPU.
"""
import glob, json, re, sys
from pathlib import Path
import numpy as np
import torch, open_clip
from PIL import Image

DEV = "cuda" if torch.cuda.is_available() else "cpu"
SHAPES = ("circle ring oval sphere star spiral blob crescent triangle square rectangle hexagon "
          "teardrop lattice grid stripe dot cross arch cylinder cube cone pyramid disc loop knot "
          "wave zigzag").split()
TEXTURES = ("furry hairy fuzzy fluffy bumpy rough smooth glossy matte shiny metallic wooden glassy "
            "plastic rubbery scaly feathery woven knitted cracked wrinkled spiky slimy velvety "
            "striped spotted dotted checkered marbled").split()
MATERIALS = "metal plastic wood glass fur feathers rubber ceramic paper fabric leather stone clay wax".split()


def build_vocab():
    words = set()
    try:
        import nltk
        from nltk.corpus import wordnet as wn
        try:
            wn.all_synsets
        except LookupError:
            nltk.download("wordnet", quiet=True); nltk.download("omw-1.4", quiet=True)
        for s in wn.all_synsets(pos="n"):
            for l in s.lemma_names():
                if "_" not in l and l.isalpha() and 3 <= len(l) <= 18:
                    words.add(l.lower())
    except Exception as e:
        print(f"[vocab] WordNet unavailable ({e}); using a small fallback noun list", flush=True)
        words.update("person man woman child girl boy face dog cat bear owl bird fish horse cow "
                     "mask doll clown toy camera bottle car house building tree flower letter word "
                     "hand eye mouth teeth hat coat dress shirt ball bat racket book sign".split())
    try:
        from matplotlib._color_data import XKCD_COLORS
        for name in XKCD_COLORS:
            words.add(name.replace("xkcd:", "").lower())
    except Exception:
        words.update("red orange yellow green blue purple pink brown black white gray gold silver".split())
    words.update(SHAPES + TEXTURES + MATERIALS)
    vocab = sorted(w for w in words if w.isascii())
    return vocab


def embed_text(model, tok, vocab, bs=2048):
    outs = []
    with torch.no_grad():
        for i in range(0, len(vocab), bs):
            chunk = [f"a photo of a {w}" for w in vocab[i:i + bs]]
            t = model.encode_text(tok(chunk).to(DEV)).float()
            outs.append((t / t.norm(dim=-1, keepdim=True)).cpu())
    return torch.cat(outs)


def step_key(p):
    s = Path(p).stem
    return 0 if "init" in s else int((re.findall(r"\d+", s) or ["0"])[0])


def img_emb(model, prep, run_dir, last_only=True):
    grids = sorted(glob.glob(f"{run_dir}/*images.jpeg"), key=step_key)
    if not grids:
        return None
    files = grids[-1:] if last_only else grids
    tiles = []
    for g in files:
        im = np.asarray(Image.open(g).convert("RGB"))
        h, w = im.shape[:2]; n = max(1, round(w / h)); ts = w // n
        tiles += [im[:, i * ts:(i + 1) * ts] for i in range(n)]
    px = torch.stack([prep(Image.fromarray(a)) for a in tiles]).to(DEV)
    with torch.no_grad():
        e = model.encode_image(px).float()
        e = e / e.norm(dim=-1, keepdim=True)
    return e.mean(0, keepdim=True).cpu()   # [1,768] mean over the final grid's samples


def main():
    runs = sys.argv[1:]
    model, _, prep = open_clip.create_model_and_transforms("ViT-L-14", pretrained="openai")
    model = model.eval().to(DEV)
    tok = open_clip.get_tokenizer("ViT-L-14")
    vocab = build_vocab()
    print(f"[vocab] {len(vocab)} concepts on {DEV}", flush=True)
    V = embed_text(model, tok, vocab).to(DEV)   # [N,768]
    out = {}
    for run in runs:
        d = f"vendor_conceptlab/outputs/{run}/images"
        z = img_emb(model, prep, d)
        if z is None:
            print(f"{run}: no images"); continue
        sims = (z.to(DEV) @ V.T).squeeze(0)              # [N]
        top = sims.topk(20)
        cands = [(vocab[i], round(float(sims[i]), 3)) for i in top.indices.tolist()]
        out[run] = cands
        print(f"\n=== {run} — nearest concepts ===")
        print("  " + ", ".join(f"{w}({s})" for w, s in cands))
    Path("RESULTS/heldout").mkdir(parents=True, exist_ok=True)
    json.dump(out, open("RESULTS/heldout/candidates.json", "w"), indent=1)


if __name__ == "__main__":
    main()
