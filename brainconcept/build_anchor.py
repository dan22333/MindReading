"""Build a CLIP *image*-space anchor for the floor-loss experiments (E1/E2).

Why image space: the optimized quantity is the diffusion-prior output z (a CLIP
ViT-L/14 *image* embedding). Comparing it to a CLIP *text* embedding of "a human face"
suffers CLIP's modality gap (image<->text cos squished to ~0.18 max), so the novelty
term has no leverage. This script samples the prior for one or more face phrasings,
L2-normalizes and averages the samples -> a stable image-space "face direction" whose
image<->image cosine has real range (~0..0.8). Text-derived, so NO photo bank.

  E1 (single term):   python build_anchor.py --preset e1 --n_samples 20 --out weights/anchor_face_e1.pt
  E2 (10-term avg):   python build_anchor.py --preset e2 --n_samples 20 --out weights/anchor_face_e2.pt

The prior is stochastic (samples from noise), so N calls give N distinct points on the
face manifold; averaging N cuts noise ~1/sqrt(N). We match training's prior settings
(cf_scale=4, timestep_respacing="5") so the anchor sits on the same manifold as z.

Run from vendor_conceptlab/ (same place scripts.train runs), env `concept-lab`.
"""
import argparse
from pathlib import Path

import torch
from kandinsky2 import load_utils

# ---- face phrasings ---------------------------------------------------------
PRESETS = {
    "e1": ["a human face"],
    "e2": [
        "a human face",
        "a face",
        "a portrait of a person",
        "a close-up of a human face",
        "a headshot",
        "the face of a man",
        "the face of a woman",
        "a face with eyes, nose, and mouth",
        "a human facial portrait",
        "a person looking at the camera",
    ],
}


@torch.no_grad()
def prior_image_embs(model, prompt: str, batch_size: int,
                     prior_cf_scale: int = 4, prior_steps: str = "5") -> torch.Tensor:
    """Return [batch_size, 768] un-normed CLIP image embeddings sampled from the prior.

    This mirrors Coach.generate_clip_emb(apply_prior=True) exactly (text transformer ->
    prior diffusion), minus the learnable-token path, so anchors match training samples.
    """
    t2_eos_token_id = len(model.tokenizer2.encoder) - 1
    prompts_batch = [prompt for _ in range(batch_size)]
    cf_scales = torch.tensor([prior_cf_scale] * batch_size, device=model.device)

    max_txt_length = model.prior.model.text_ctx
    tok, mask = model.tokenizer2.padded_tokens_and_mask(prompts_batch, max_txt_length)
    cf_token, cf_mask = model.tokenizer2.padded_tokens_and_mask([""], max_txt_length)
    if cf_token.shape != tok.shape:
        cf_token = cf_token.expand(tok.shape[0], -1)
        cf_mask = cf_mask.expand(tok.shape[0], -1)
    tok = torch.cat([tok, cf_token], dim=0).to(model.device)
    mask = torch.cat([mask, cf_mask], dim=0).to(model.device)

    x = model.clip_model.token_embedding(tok).type(model.clip_model.dtype)
    x = x + model.clip_model.positional_embedding.type(model.clip_model.dtype)
    x = x.permute(1, 0, 2)
    x = model.clip_model.transformer(x)
    x = x.permute(1, 0, 2)
    x = model.clip_model.ln_final(x).type(model.clip_model.dtype)
    txt_feat_seq = x
    txt_feat = (x[torch.arange(x.shape[0]), tok.eq(t2_eos_token_id).nonzero()[:1, -1]]
                @ model.clip_model.text_projection)
    txt_feat = txt_feat.float().to(model.device)
    txt_feat_seq = txt_feat_seq.float().to(model.device)

    img_feat = model.prior(txt_feat, txt_feat_seq, mask, cf_scales,
                           timestep_respacing=prior_steps)
    return img_feat.float()  # [batch_size, 768]


@torch.no_grad()
def text_embs(model, terms, device):
    """Return [K, 768] L2-normed CLIP *text* embeddings (pooled text projection), one per term.
    Mirrors the txt_feat path in generate_clip_emb so the text anchors sit in the same CLIP
    joint space as the optimized image embedding (image<->text cosine, modality-gap-limited)."""
    t2_eos_token_id = len(model.tokenizer2.encoder) - 1
    max_txt_length = model.prior.model.text_ctx
    out = []
    for term in terms:
        tok, mask = model.tokenizer2.padded_tokens_and_mask([term], max_txt_length)
        tok = tok.to(model.device)
        x = model.clip_model.token_embedding(tok).type(model.clip_model.dtype)
        x = x + model.clip_model.positional_embedding.type(model.clip_model.dtype)
        x = x.permute(1, 0, 2)
        x = model.clip_model.transformer(x)
        x = x.permute(1, 0, 2)
        x = model.clip_model.ln_final(x).type(model.clip_model.dtype)
        txt_feat = (x[torch.arange(x.shape[0]), tok.eq(t2_eos_token_id).nonzero()[:1, -1]]
                    @ model.clip_model.text_projection).float()
        txt_feat = txt_feat / txt_feat.norm(p=2, dim=-1, keepdim=True)
        out.append(txt_feat.to(device))
    return torch.cat(out, dim=0)                                        # [K, 768]


def build(model, terms, n_samples, batch, device):
    """Return the FULL individual image-prior anchor set (not just the centroid).

    img_set = [len(terms)*n_samples, 768] L2-normed prior samples. The contrast loss scores
    the optimized embedding against each row individually (mean+max), so we keep them all;
    the averaged centroid is also returned for sanity/back-compat.
    """
    all_normed = []
    for term in terms:
        got = 0
        while got < n_samples:
            b = min(batch, n_samples - got)
            emb = prior_image_embs(model, term, batch_size=b)          # [b, 768]
            emb = emb / emb.norm(p=2, dim=-1, keepdim=True)            # L2-norm each
            all_normed.append(emb.to(device))
            got += b
            print(f"  [{term!r}] {got}/{n_samples}")
    stacked = torch.cat(all_normed, dim=0)                             # [len(terms)*n, 768]
    anchor = stacked.mean(dim=0)                                       # centroid
    anchor = anchor / anchor.norm(p=2)                                 # renorm
    # sanity: mean pairwise cos of samples to the anchor (tightness of the centroid)
    tight = (stacked @ anchor).mean().item()
    return anchor, stacked, tight


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", choices=list(PRESETS), default=None)
    ap.add_argument("--terms", type=str, default=None,
                    help="semicolon-separated phrasings, overrides --preset")
    ap.add_argument("--n_samples", type=int, default=20, help="prior samples PER term")
    ap.add_argument("--batch", type=int, default=10, help="prior samples per forward call")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--cache_root", type=Path, default=Path("/tmp/kandinsky2"))
    ap.add_argument("--device", type=str, default="cuda:0")
    args = ap.parse_args()

    if args.terms:
        terms = [t.strip() for t in args.terms.split(";") if t.strip()]
    elif args.preset:
        terms = PRESETS[args.preset]
    else:
        raise SystemExit("pass --preset e1|e2 or --terms 'a;b;c'")

    print(f"Loading Kandinsky2.1 ({args.device}) ...")
    model = load_utils.get_model(args.cache_root, args.device)
    print(f"Building anchor from {len(terms)} term(s) x {args.n_samples} samples:")
    for t in terms:
        print(f"  - {t}")

    anchor, img_set, tight = build(model, terms, args.n_samples, args.batch, args.device)
    text_set = text_embs(model, terms, args.device)                    # [K, 768] normed CLIP text
    n_total = img_set.shape[0]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"anchor": anchor.cpu(),                                # centroid (back-compat)
                "img_set": img_set.cpu(),                              # [N,768] individual image priors
                "text_set": text_set.cpu(),                            # [K,768] individual CLIP-text
                "terms": terms, "n_total": n_total,
                "tightness_cos": tight}, args.out)
    print(f"Saved anchor set img_set{tuple(img_set.shape)} text_set{tuple(text_set.shape)} "
          f"from {len(terms)} term(s) -> {args.out}")
    print(f"Centroid tightness (mean sample->anchor cos): {tight:.3f} "
          f"(higher = tighter; expect lower for the 10-term e2)")


if __name__ == "__main__":
    main()
