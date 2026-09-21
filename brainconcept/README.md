# brainconcept — find the token whose diffusion-prior output maximizes a person's ROI fMRI

**Your goal, verbatim intent:** train a CLIP-vision→fMRI encoder for one patient (ridge,
with validation + hyper-parameter tuning + test); then **learn a token** whose image
lights up a chosen ROI; generate 10–20 images from it; have a VLM name what's in them;
re-optimize the token to keep maximizing the ROI **while its CLIP-image embedding stays
far from the CLIP-text of the concepts found**; repeat until the ROI signal can't stay high.

**Method = [ConceptLab](https://arxiv.org/abs/2308.02669) with the positive term replaced by
a brain term.** We use the **new diffusion-prior models** ([Kandinsky 2.1](https://github.com/ai-forever/Kandinsky-2)
/ [Karlo](https://github.com/kakaobrain/karlo) — the open unCLIP/DALLE-2 priors), **not** the
old BrainDiVE Stable-Diffusion-guidance path. Background: `../ConceptLab_to_fMRI_explainer.html`.

---

## 1. The pipeline (one picture)

```
 learnable token v*
        │
        ▼
 [ text encoder ] ──► [ DIFFUSION PRIOR  P ] ──►  z   ( a CLIP IMAGE embedding )
   (frozen)              Kandinsky 2.1 / Karlo      │
                          (frozen)                  │
        ┌──────────────────────────────────────────┼────────────────────────────────┐
        │  POSITIVE (drive the brain)               │   NEGATIVE (stay novel)         │
        │  maximize  mean_ROI( W · L2norm(z) )       │   minimize  cos( z, t_c )        │
        │  W = ridge encoder CLIP-img → fMRI         │   t_c = CLIP-text(found concept) │
        └───────────────────────────────────────────┴──────────────────────────────────┘
        │
        ▼ (only to look at results + run the VLM)
 [ DECODER ] ──► images ──► VLM ("what's in this?") ──► new concepts ──► add to negatives ──┐
        ▲                                                                                    │
        └──────────────────────── next round: same brain goal, more negatives ◄─────────────┘
```

Everything in the optimization loop lives in **CLIP-image-embedding space** (the prior's
output). The prior is small and the encoder is linear, so the token gradient is **cheap** —
no backprop through a 50-step pixel decoder. We only decode `z → image` to visualize and to
feed the VLM.

**The objective (ConceptLab Eq. 3, brain edition):**
```
Loss(v*) = − mean_ROI( W · L2norm(P(v*)) )                     # drive the ROI
           + λ · [ mean_c cos(P(v*), t_c) + max_c cos(P(v*), t_c) ]/2   # flee found concepts
```

---

## 2. Which models — the chosen stack

| Piece | Choice | Why |
|---|---|---|
| Diffusion prior + decoder | **Kandinsky 2.1** (ConceptLab's) — Karlo as drop-in | ConceptLab's **public code already optimizes a token in Kandinsky's prior output** and runs the BLIP-2 negative loop. Forking it = the least pipeline. Both use CLIP ViT-L/14. |
| Patient encoder | **ridge: CLIP ViT-L/14 image embedding → fMRI** | Your "vision-clip→fmri with ridge." **Must** be trained in the SAME CLIP space the prior outputs (ViT-L/14) so the brain term is computable directly on `z`. |
| VLM (names contents) | **BLIP-2** (ConceptLab already wires it) | `--live_negatives` already: caption image → add answer as negative. We reuse it. |
| Negative distance | `cos(z, CLIP_text(concept))` | `z` is already a CLIP image embedding → no re-encoding needed. |
| Novelty metric (opt.) | **CREward** | Auto-log new territory per round. |

**Base to fork:** [`kfirgoldberg/ConceptLab`](https://github.com/kfirgoldberg/ConceptLab).
We change **one thing**: swap the positive category-similarity loss for `−mean_ROI(W·z)`.

> Note on "full tokens, not CLS": the prior's output `z` is a **single pooled** CLIP image
> embedding by construction, and the ridge encoder consumes exactly that. Full-token
> conditioning was only available on the (rejected) Stable-Diffusion route. On this
> prior route we commit to the pooled embedding — consistent with the ridge-encoder step.

---

## 3. What we build vs reuse

| Already in ConceptLab (reuse) | New (we write) |
|---|---|
| Token `v*` optimization in the prior's output space | The **ridge brain encoder** `W` (CLIP-L/14 → fMRI) |
| Positive + negative CLIP-space losses + closest-negative penalty | Swap positive term → `−mean_ROI(W·z)` |
| BLIP-2 loop that names images and grows negatives | Feed our ROI + negatives config; adjust stop rule |
| Decoder to generate the 10–20 images | Eval: hit-rate, novelty, held-out fMRI check |

---

## 4. Phased plan

```
Phase A  Fork ConceptLab; get token optimization + BLIP-2 loop running on Kandinsky.  [~2–3 days]
Phase B  Download NSD (S1); train ridge CLIP-ViT-L/14 → fMRI (alpha search, val, test). [download + ~1 day]
Phase C  Replace ConceptLab's positive loss with −mean_ROI(W·z); wire ROI + λ.          [~2–3 days]
Phase D  Run the loop for an ROI; the BLIP-2 negatives now drive brain-novelty.         [~2–3 days]
Phase E  Evaluate: category hit-rate, CREward novelty, held-out fMRI spot-check.        [~2–4 days]
```
**Total ~2–3 weeks + NSD download.** Leaner than the SD route because ConceptLab already
implements the token loop and the negative loop — we swap a single term.

---

## 5. Would it work? — honest verdict

- **Token optimization in the prior + BLIP-2 negatives:** high confidence — it's ConceptLab's
  shipped code.
- **Brain term on `z`:** high confidence and *cheap* — `W·L2norm(z)` is linear and fully
  differentiable; gradient flows to `v*` through the small prior only.
- **Main dependency:** a **CLIP ViT-L/14 → fMRI ridge encoder trained on the patient**, which
  needs **NSD fMRI data** (see §6). This is the real gating cost.
- **Main scientific risk:** *encoder exploitation* — `v*` could find a `z` that fools the linear
  encoder without being a real brain driver. Mitigate: keep ConceptLab's similarity clamps,
  validate each round that decoded images look ROI-appropriate (VLM gate), and hold out fMRI.

**Stop the outer loop when:** best `mean_ROI(W·z)` under current negatives drops below ~0.7× the
round-1 max for 2 rounds; OR the VLM stops naming new concepts; OR a hard cap (8–12 rounds).

---

## 6. Data reality (blocks Phase B)

- `../brainit-fmri/data/nsd_data/` currently has **only the README** — no fMRI, no images.
- To train the patient ridge encoder you need NSD **fMRI betas + the stimulus images** for **S1**
  (most data + ROI masks already here), behind the
  [NSD data-access agreement](https://naturalscenesdataset.org/).
- The prior/decoder/BLIP-2 (Phase A) need **no** NSD data — start there in parallel.

---

## 7. Layout (as built)

```
brainconcept/
  README.md                 ← the plan (this file)
  VM_RUN.md                 ← exact commands to run every stage on the T4 VM
  train_encoder.py          ← Stage 1: CLIP ViT-L/14 image emb → fMRI ridge (RidgeCV, val, test)
  vendor_conceptlab/        ← forked github.com/kfirgoldberg/ConceptLab, minimally patched:
    training/train_config.py  · + brain_objective, brain_encoder_weights fields
    training/coach.py         · + brain_positive_loss(); positive term gated on brain_objective;
                                  generic VLM question in brain mode
    configs/brain_faces_s1.yaml · the brain run config (Kandinsky 2.1 + W + live_negatives)
```

**The entire diff vs stock ConceptLab is ~40 lines** (one new method, one gated branch, two
config fields, one VLM-question case) plus `train_encoder.py`. Everything else — token
optimization in the prior, the BLIP-2 negative loop, generation — is reused unchanged.

> Status: **code complete, not yet run.** Authored and byte-compiled on the host; the heavy
> stages (model downloads, encoder training, the loop) run on the VM per `VM_RUN.md`. Stage 1
> needs the NSD download (only you can do that). `train_encoder.py::load_nsd_s1` has one TODO
> to match your NSD file layout.
