# MindReading — project learnings

Source of truth for what we've learned building this project. **Markdown is authoritative**;
`MindReading_learnings.pdf` is a render target only — never edit the PDF.

**Project goal.** Generate an image that is X but not Y — e.g. for a suspected food-selective
brain region, generate an image containing *no food* that nevertheless activates that region.

**How this file is organised**

- **Part A — Project learnings log.** Every entry in full, newest first. The authoritative record.
- **Part C — Topic notes.** One chapter per topic, holding one-line pointers back into Part A,
  so re-reading a chapter surfaces everything learned about it. Chapters are created on demand.

There is no Part B: this project had no pre-existing reference document to fold in.

**Phases.** This project has no formal plan file, so the phase field uses the 5-paper reading arc
that structures the work:

| Phase | Meaning |
|---|---|
| `Paper 1 — Universal Encoder` | Beliy et al., *The Wisdom of a Crowd of Brains* (arXiv 2406.12179) |
| `Paper 2 — NeuroGen` | Gu et al., *NeuroImage* 247:118812 (2022) |
| `Paper 3 — BrainDiVE` | Luo et al., NeurIPS 2023 Oral (arXiv 2306.03089) |
| `Paper 4 — BrainExplore` | Wasserman et al. (arXiv 2512.08560) |
| `Paper 5 — BrainCause` | Golbari et al. (arXiv 2605.23895) |
| `Project design` | Decisions about our own experiment |
| `general` | Phase-agnostic |

**Claim provenance.** Entries tag evidence as: stated in the paper / verified in code (with
`path:line`) / our own inference / unavailable. Where paper and code disagree, both are recorded.

**Repos in this folder** — all clean upstream clones, no local modifications:
`universal-brain-encoder-code` (WeizmannVision), `NeuroGen` (zijin-gu), `BrainDiVE` (aluo-x).

---

# Part A — Project learnings log

Newest first. Entry format:

```markdown
<a id="l-YYYYMMDD-short-slug"></a>

## YYYY-MM-DD — Title stating the claim
**Type:** insight | gotcha | pattern | decision | resource
**Phase:** <phase>

The learning in 1-4 sentences.

**Why it matters:** consequence.
**How to apply:** concrete action.
**Source:** file:line, paper section, or context.
```


<a id="l-20260929-cosine-vs-classifier"></a>

## 2026-09-29 — A cosine-to-prototype repel cannot leave a category; only a two-class discriminant can
**Type:** insight  
**Phase:** Project design

We want an image that maximises a face-ROI response yet is *not a face*: a drive term $\mathrm{ReLU}(z^{\*}-\text{drive})$ pulls the embedding $z$ into the high-face-ROI region while a **novelty** term pushes it out of face-space. We tried two novelty terms and — crucially — only their *direction* differs.

**Cosine repel (E2).** Penalise the (hinged) similarity to $N\!\approx\!1000$ face-image anchors $a_i$. With $z$ and each $a_i$ unit-normalised, $\cos(z,a_i)=z\cdot a_i$, so the mean term is **linear**:
$$S(z)=\frac1N\sum_i \cos(z,a_i)=z\cdot\Big(\tfrac1N\sum_i a_i\Big)=z\cdot\bar a,\qquad \nabla_z S=\bar a .$$
So "cosine to each anchor, then average" **equals** "cosine to the centroid $\bar a$" (up to the scalar $|\bar a|$), and the repel descends along $-\bar a$ — straight away from the *average* face. Note $\bar a$ is **not** unit: $|\bar a|^2=\frac1{N^2}\big(N+\sum_{i\neq j}a_i\!\cdot a_j\big)\le 1$, with $|\bar a|$ = the anchor-cluster tightness.

**Classifier repel (E3).** Penalise $P(\text{face})=\sigma(w^\top z+b)$; it descends along $-w$. Modelling each class as a Gaussian with shared covariance $\Sigma$ and taking the log-posterior-ratio, the $z^\top\Sigma^{-1}z$ terms cancel (⇒ linear boundary) and leave the LDA/logistic optimum
$$w=\Sigma^{-1}\,(\mu_{\text{face}}-\mu_{\text{nonface}}).$$
$\Sigma^{-1}$ is **whitening** ($\Sigma=Q\Lambda Q^\top\Rightarrow\Sigma^{-1}=Q\Lambda^{-1}Q^\top$: down-weight high-variance/noisy directions, up-weight tight ones), and class separability is the Mahalanobis distance $d^2=(\mu_{\text{face}}-\mu_{\text{nonface}})^\top\Sigma^{-1}(\mu_{\text{face}}-\mu_{\text{nonface}})$ — larger $d$ ⇒ steeper, more confident sigmoid (our probe: held-out AUC $=1.0$, so large $d$).

**The one-line difference.** The cosine repel is the classifier with two pieces deleted: no $\mu_{\text{nonface}}$ (it only knows the face mean) and no $\Sigma^{-1}$ (no whitening) — strip them from $w$ and you are left with $\approx\mu_{\text{face}}=\bar a$. So $-\bar a$ = "away from the average face," but the face region has width along $-\bar a$ (atypical/decorated faces have low centroid-cosine yet are still faces), so you slide to the *weird edge of face-space and stay a face*. $-w$ = "toward the other class, across the whitened boundary," so it actually exits.

**Why it matters:** This is the mathematical reason E2 drove $\cos$ to its hinge ($\approx0.31$) yet produced only **decorated human faces**, while E3 left human-face-space. To *leave* a set you need a signal informed by the complement (negatives) and the covariance — not distance to a prototype.
**How to apply:** For any "generate X-absent" objective use a discriminatively-trained (X vs not-X) probe as the novelty term, not cosine to an X-example bank. Normalising $\bar a$ only rescales the gradient magnitude ($\bar a$ vs $\bar a/|\bar a|$), never its direction — so it is not a fix. The negative class must span what "not-X" means: see [full entry](#l-20260929-classifier-relocates).
**Source:** derivations in artifacts `claude.ai/code/artifact/4d6d12dd-4a76-459e-882d-0311523178dd` (cosine vs classifier) and `.../5e6a3913-5dc4-427f-8dc4-5f7f6b92c91a` (LDA); loss in `brainconcept/vendor_conceptlab/training/coach.py` (`classifier_repel` branch).


---

<a id="l-20260929-classifier-relocates"></a>

## 2026-09-29 — A classifier repel moves toward whatever you label "not-X" — we labelled animals non-face, so it made animal faces
**Type:** gotcha  
**Phase:** Project design

E3's classifier drove $P(\text{human face})\to 0.07$ and BLIP's "is there a *human* face?" to 0.00 on every run — yet the images were unmistakable **animal faces** (badger/hedgehog/raccoon-like creatures), and held-out DINOv2 still read the face ROI at $+1.7$ to $+2.8$. Face cortex responds to animal and pareidolic faces, so these *are* faces — just not human. The cause is mechanical: $-w$ points toward $\mu_{\text{nonface}}$, and our negative set contained animals (`dog/cat/bird/fish`), so "not-a-face" literally included animal faces. E3 did exactly what the labels said. It exploited the gap between "human face" (penalised) and "face structure" (what drives the ROI), not a genuine non-face solution.

**Why it matters:** "Works" is defined entirely by the class boundary you draw. A clean-looking escape (face-rate 0.00) can be definition-gaming; we only caught it by looking at the pixels, not the BLIP flag. It is also the strongest evidence so far for a possible biological ceiling — strong face-ROI drive may *require* face structure.
**How to apply:** To test a genuine non-face driver, move animal + pareidolic faces into the **positive** class (so $\mu_{\text{face}}$ includes them and $\mu_{\text{nonface}}$ is pure objects) and change the honest judge to "any face-like pattern (human/animal/object)?". Excluding animals from *both* classes does **not** work — it leaves them unpenalised. Always eyeball the images; never trust a single binary judge.
**Source:** `brainconcept/build_face_banks.py` `FACE_TERMS`/`NONFACE_TERMS`; report artifact `claude.ai/code/artifact/1e904e9e-74bd-493c-933d-2e4ff33cfd85`.


---

<a id="l-20260928-comp-clamp"></a>

## 2026-09-28 — Contrast selectivity is gamed by suppressing the complement; clamp it at zero to force real drive
**Type:** gotcha  
**Phase:** Project design

We defined drive as a **selectivity contrast** $z_{\text{faces}}-z_{\text{comp}}$ (faces ROI minus the complement = bodies+places+words, each z-scored to its own natural-image baseline) and floored it: $\mathrm{ReLU}(z^{\*}-(z_{\text{faces}}-z_{\text{comp}}))$. Because $z_{\text{comp}}$ can go negative, the contrast is inflated by *positive minus negative*: e.g. round-1 E1 at $z^{\*}=1$ reached "selectivity" $+1.86$ from $z_{\text{faces}}=+0.08$ and $z_{\text{comp}}=-1.78$ — i.e. $\approx 96\%$ complement-suppression and near-zero real face drive. The floor was satisfied without driving faces at all.

**Fix (`comp_clamp`):** use $z_{\text{faces}}-\max(z_{\text{comp}},0)$. Driving the complement *below* baseline earns no free selectivity, so the floor can only be met by real face drive. Effect in round 2: $z_{\text{comp}}\!\approx\!0$ everywhere and $z_{\text{faces}}$ genuinely reached $z^{\*}$ ($z^{\*}=2\Rightarrow z_{\text{faces}}\approx2.3$, vs the fake $1.1$ before).

**Why it matters:** A difference-of-two-normalised-quantities objective silently rewards tanking the subtrahend; the metric then certifies "selective" images that do not drive the target. Any contrast/selectivity loss needs this guard.
**How to apply:** Clamp the complement term at its baseline ($\max(\cdot,0)$ after z-scoring), or add a separate direct floor on $z_{\text{faces}}$. Config flag `comp_clamp` in `train_config.py`; branch in `coach.py`.
**Source:** `brainconcept/vendor_conceptlab/training/coach.py` contrast branch; round-1 vs round-2 tables in report artifacts.


---

<a id="l-20260928-round2-result"></a>

## 2026-09-28 — Image-bank → human faces, classifier → animal faces, text → inert: every strong ROI driver is some kind of face
**Type:** insight  
**Phase:** Project design

Round 2 (24 runs, $z^{\*}\!\in\!\{2,2.5,3,3.5\}\times w\!\in\!\{1,1.5,2\}$, `comp_clamp` on) plus a text control, all with the same drive term and held-out validation:

| novelty repel | anchor | outcome | BLIP human-face | held-out DINOv2 $z$ |
|---|---|---|---|---|
| E2 cosine | 1000 face **images** | human faces | 1.00 | $+2.2\ldots+3.5$ |
| Text cosine | 1000 face **text** phrasings | human faces | 0.50 | up to $+3.39$ (highest) |
| E3 classifier | face probe on $z$ | **animal faces** | 0.00 | $+1.7\ldots+2.8$ |

All three produce **real** ROI drivers (a second, non-CLIP encoder agrees), and all three are faces of some kind. Two calibration facts: a real/generated face sits at $z_{\text{faces}}\!\approx\!+1.8$, so these optimised images ($+2.5$ to $+3.4$) are **super-stimuli** — above a natural face. And the text repel is **inert**: image$\leftrightarrow$text cosine is pinned at $\sim0.06$–$0.13$ by CLIP's modality gap, so driving it to $0.02$ moved $z$ along a direction nearly orthogonal to image-faceness and produced the *strongest* face driver of all.

**Why it matters:** With an honest metric and held-out validation, we could not yet make a *non-face* strong driver — only human-face, animal-face, or (via text) human-face super-stimuli. Novelty must be measured in **image space** (the text axis is orthogonal to it), and even the image-space classifier only relocated to animal faces ([full entry](#l-20260929-classifier-relocates)).
**How to apply:** Keep DINOv2 (non-CLIP) as the honest drive check; drop the text repel; treat $z_{\text{faces}}\approx1.8$ as the "one real face" yardstick when reading magnitudes.
**Source:** `brainconcept/RESULTS/Round2_E2_vs_E3_report.pdf`; artifact `claude.ai/code/artifact/1e904e9e-74bd-493c-933d-2e4ff33cfd85`.


---

<a id="l-20260928-imgcos-decoupled"></a>

## 2026-09-28 — Anchor cosine is decoupled from perceptual faceness — the repel hits its hinge yet the image stays a face
**Type:** insight  
**Phase:** Project design

The cosine repel *succeeds on its own metric and fails at the goal*. In E2 at $z^{\*}=2$, $\cos(z,\text{bank})$ fell from $0.58$ (init) to $\approx0.31$ by step 500 and **plateaued at the hinge** $c_{\text{img}}=0.30$ — a value comparable to a natural *non-face* photo (naturals sit at $0.32$–$0.44$ to the bank) — yet BLIP still reported a human face. Even more telling: the $z^{\*}=3$ run reached a *lower* $\cos=0.23$ than the escaped $z^{\*}=1$ run ($0.32$) while being **more** of a face. So low anchor-cosine $\neq$ non-face. Escape in round 1 came from a **low drive floor** (weak drive lets $z$ drift off faces), not from the repel reducing the cosine; and more steps do not help — the hinge zeroes the gradient below $c_{\text{img}}$ by design, so the cosine simply parks there.

**Why it matters:** A repel that provably drives its surrogate to target while the true property is unchanged is the textbook symptom of a mis-specified proxy — here explained by [the one-class-vs-discriminant entry](#l-20260929-cosine-vs-classifier). It also rules out "lower the threshold" and "train longer" as fixes.
**How to apply:** Validate a novelty proxy against the actual property (a held-out judge on the decoded image), not against the proxy's own value; if the proxy is a linear/cosine one-class score, expect this decoupling.
**Source:** per-step trajectories in `brainconcept/RESULTS/contrast/ours_r2_e2_*.json`; round-1 report `StageB_E1E2_frontier_report.pdf`.


---

<a id="l-20260927-contrast-loss"></a>

## 2026-09-27 — The contrast/floor loss: a drive release-valve, mean+max hinged repel, and a decode→re-encode realness monitor
**Type:** pattern  
**Phase:** Project design

The loss that underlies all of the above (ConceptLab loop, faces ROI, subject S1), optimising one learnable token's embedding $z$ (the diffusion-prior output):
$$L=\underbrace{\mathrm{ReLU}\!\big(z^{\*}-(z_{\text{faces}}-\max(z_{\text{comp}},0))\big)}_{\text{drive (selectivity floor)}}+\;w\cdot\underbrace{\big[\operatorname{mean}_i \mathrm{clamp}(\cos(z,a_i),c)+\max_i \mathrm{clamp}(\cos(z,a_i),c)\big]}_{\text{novelty repel}} .$$
Three design choices earned their place: (1) the **floor** is a release-valve — once $z_{\text{faces}}-z_{\text{comp}}\ge z^{\*}$ the drive gradient is exactly zero, so the repel can act unopposed; without it the maximise-drive term is insatiable and rebounds into face-space. Measured: at convergence with the floor met, drive contributes $0$ to the loss and the residual is $\approx60$–$95\%$ the image-repel term. (2) **mean + max** hinged aggregation (ConceptLab's negative-loss form): mean leaves the anchor centroid, max is a hard-negative that stops the image hugging a single anchor; the per-anchor `clamp(cos, c)` gives zero gradient below $c$ so each anchor is pushed only until "far enough." (3) A **drift monitor** $\cos(z,\,\mathrm{encode}(\mathrm{decode}(z)))$ logged at checkpoints (not in the loss) catches off-manifold degeneration — healthy runs sit $\approx0.75$–$0.9$; it falls as you push $z^{\*}$ up or $c$ down.

**Why it matters:** These are reusable ingredients for constrained generation: a floor to make a "reach a level then stop" target (avoids the maximise-rebound), mean+max hinges for repelling from a set, and a cheap decode→re-encode consistency check for realness that needs no extra model.
**How to apply:** Prefer $\mathrm{ReLU}(\text{target}-\text{quantity})$ over raw maximise/minimise whenever "enough" is well-defined; log decode→re-encode cosine as a free off-manifold alarm.
**Source:** `brainconcept/vendor_conceptlab/training/coach.py` (contrast branch, `decode_reencode_cos` in `kandinsky2_1_model.py`); background in [[C.4 — Guided generation mechanics]].


---

<a id="l-20260810-gap-analysis"></a>

## 2026-08-10 — Our project is the unoccupied cell between BrainDiVE and BrainCause — and BrainCause names it as future work
**Type:** decision  
**Phase:** Project design

Component inventory across the five papers: a differentiable image->voxel encoder exists (Paper 1);
gradient-guided diffusion that *maximises* an ROI exists (Paper 3); maximise-one-ROI-while-minimising-another
exists (Paper 2 sec 3.4, but with BigGAN and ROI-means); *removing* a concept and testing whether the response
survives exists (Paper 5, via ~10 LLM-proposed edits per concept). **Nobody has done constrained optimisation:
maximise ROI activation subject to verified concept absence.** BrainCause tests a finite hand-proposed list of
alternatives; we would *search* the manifold — the difference between checking 10 candidates and running
gradient descent.

**Why it matters:** BrainCause explicitly flags both the gap and the plan: *"some discovered representations may remain causal only
with respect to the alternatives considered by the current pipeline"* and *"An important direction for future work
is to make the pipeline more iterative, allowing current results and activation patterns to guide the proposal of
new counterfactuals and semantic negatives."* That is a citation justifying our project — and a warning, since the
authors are in the same institute.

**How to apply:** Before committing, ask Roman Beliy / Navve Wasserman / Yuval Golbari what is already in progress — a 10-minute
conversation that could redirect months. Then frame the work as a stress test of BrainCause's causal claims rather
than an independent discovery attempt, so every outcome is a result.

**Source:** BrainCause arXiv 2605.23895 sec 5; cross-checked against BrainDiVE sec 4.4 and NeuroGen sec 3.4


---

<a id="l-20260810-three-rungs"></a>

## 2026-08-10 — Run retrieval on measured data before writing any generator — it de-risks the project and hands you the confound
**Type:** pattern  
**Phase:** Project design

The escalation is retrieval -> prompt-and-rank -> gradient-guided, and each rung buys evidence the next cannot.
Rung 1: filter NSD's ~73k images for *no food* (public COCO category labels) and rank by **measured** food-ROI beta.
Zero adversarial risk (real photographs, real measured responses), uses measured rather than predicted signal, and
if the top qualifying images share a property the data has handed us the hypothesis to generate. Rung 2: generate
with a plain diffusion model and let the encoder only *rank* — never provide a gradient — which is structurally
near-immune to exploitation.

**Why it matters:** Rung 1 costs days and no GPU. If nothing among 73,000 natural images drives the food region without food, we learn
the region is genuinely food-specific in week 1 rather than month 3. Rung 2 plausibly gets 60-70% of the effect for
10% of the risk; many projects should stop there.

**How to apply:** Do rung 1 first, be willing to stop at rung 2, and only escalate to gradients once we can state what rung 2 failed
to deliver.

**Source:** Our analysis; NSD image set + public COCO annotations


---

<a id="l-20260810-two-encoders"></a>

## 2026-08-10 — Lock a structurally different validation encoder and never let it see a gradient — not once
**Type:** decision  
**Phase:** Project design

Discovery encoder = the Universal Brain Encoder (DINOv2 + attention head), gradients flow, guides generation.
Validation encoder = BrainDiVE's encoder (CLIP ViT-B/16 + one linear layer), which **never receives a gradient,
is never used in generation, and is inspected only after images are final and frozen.** Maximally independent:
different backbone, different head, different training pipeline. Three levels of evidence follow: (1) the image
fools *our* encoder — near-worthless, that is what gradient ascent does; (2) it also fools the locked encoder —
the effect is a property of image->brain mappings, not one network's quirk; (3) a real brain responds — the claim.

**Why it matters:** Without a locked encoder, "we found an adversarial example" and "we discovered something about the brain" are
indistinguishable, and the project is unfalsifiable. Level 1 is trivially achievable and looks identical to a
discovery in a figure.

**How to apply:** Never tune a hyperparameter on it, never "just check" mid-run. Reimplementing BrainDiVE's head is ~5 minutes
(one `Linear(512, N)` on an L2-normalised CLIP embedding) but needs weights — either request them or fit our own on NSD.

**Source:** BrainDiVE encoder_model_vit.py:487,509-512; our analysis


---

<a id="l-20260810-both-outcomes"></a>

## 2026-08-10 — Frame the experiment so all three outcomes are publishable, then a null result is a contribution
**Type:** pattern  
**Phase:** Project design

"I will find a food-free image that activates the food region" has one success branch and one dead end. Reframed
as "does BrainCause's causal claim survive gradient search over the whole manifold of food-free images?", every
branch is a result: **find nothing** -> a stronger causality claim than minimal edits can provide
("optimisation-based causal validation"); **find something that transfers** to the locked encoder -> a discovery
about the brain; **find something that does not transfer** -> a methods result about encoder exploitability, plus
the perceptual-cost-vs-activation-gain curve, which has never been published for any brain encoder.

**Why it matters:** Same experiments, same code, same compute — only the framing differs, and the framing decides whether a null
result is a failure or a paper. It also forces us to build the locked encoder, because branches 2 and 3 are
undecidable without it.

**How to apply:** Write the three-branch table before running anything and check each branch is a paper. If one branch reads "then
I have nothing", change the framing until it doesn't.

**Source:** Our analysis


---

<a id="l-20260810-fmri-contrast"></a>

## 2026-08-10 — Design the confirmatory fMRI as a between-condition contrast, not a correlation, because contrasts need no noise ceiling
**Type:** decision  
**Phase:** Project design

A correlation with measured betas needs repeats to be interpretable (see the noise-ceiling entry), and repeats
cost scanner time. A contrast does not. So structure the scan as conditions: **A** real food images (positive
control), **B** our optimised food-absent images, **C** prompt-matched *unoptimised* food-absent controls,
**D** images optimised for a different ROI (specificity), **E** counterfactual removal pairs, **F** neutral/random.
Primary pre-registered contrast: **B > C in the food ROI AND B ~= C in control ROIs.**

**Why it matters:** Novel synthetic stimuli will realistically get 1 repeat each, so measured correlations will be far below the ~0.52
seen in the papers for reasons unrelated to our method. Both halves of the primary contrast matter — the second is
what rules out "we just made everything more exciting".

**How to apply:** Pre-register the contrast before generating a single image. Report ROI size in voxels at every threshold used.

**Source:** Our analysis; Paper 1 App C on repeat averaging


---

<a id="l-20260810-food-mask"></a>

## 2026-08-10 — There is no food localizer and BrainDiVE's food masks are not in its repo — resolve the food ROI in week 1 or be blocked
**Type:** gotcha  
**Phase:** Project design

Faces/places/bodies/words have standard localizers; **food does not**, because until 2022 nobody thought a food
region existed. BrainDiVE's masks come from Jain et al. 2023 (*Comm. Biol.* 6:1, its ref [5]) and the code expects
`subj_{}_food_mask.npy` and `food_s{}.npy` on the authors' cluster — neither file ships (the repo is 372 KB, no
`.npy`). Also: `utils/roi_utils.py` in Paper 1's repo has **no food entry** in `ROI_CLASSES`.

**Why it matters:** Every downstream step depends on which voxels count as "the food region", and email latency is the bottleneck, not
compute.

**How to apply:** Three paths, do 1 and 3 the same day: (1) email Tarr/Wehbe (CMU) for the per-subject masks; (2) reproduce Khosla's
decomposition ourselves (~a week, adds our own noise); (3) **placeholder now** — take NSD images with/without COCO
food labels, compute a t-contrast on measured betas per subject, threshold. Then compare masks when the real one
arrives; the overlap is itself a sanity check.

**Source:** BrainDiVE run_brain_guidance_category.py:49,204,237 and ref [5]; universal-brain-encoder-code utils/roi_utils.py:11-18


---

<a id="l-20260810-mask-overlap"></a>

## 2026-08-10 — Check the food-mask / Algonauts-mask overlap before optimising, or we may be optimising a truncated region
**Type:** gotcha  
**Phase:** Project design

Paper 1's encoder covers ~39.5k voxels selected by the Algonauts mask — about 12% of the 327,684 fsaverage
vertices. Everything outside it is unrepresentable by that encoder. BrainDiVE evidently judged the standard mask
too small: it unions the NSD category ROIs with **45 HCP-MMP1 parcels** chosen for low noise. And food cortex was
discovered precisely by examining voxels nobody had labelled.

**Why it matters:** If a meaningful part of food-selective cortex falls outside the ~39.5k, we would be optimising a truncated ROI
without any error message.

**How to apply:** A 20-minute check that could save a month: compute the intersection of the food mask with the Algonauts mask and
report the fraction retained. Do it before any generation code.

**Source:** Paper 1 App B.1 and Table S3; BrainDiVE run_brain_guidance_category.py:42,62-70


---

<a id="l-20260810-voxel-centric"></a>

## 2026-08-10 — Making the output one scalar per (image, voxel) is what removes subject count, voxel count and scanner from the problem
**Type:** insight  
**Phase:** Paper 1 — Universal Encoder

Prior encoders are "fMRI-centric": one example = (image, entire 39,548-vector), so the output layer is welded to
one subject's voxel count *and* ordering, and 8 subjects are 8 separate problems. Paper 1 makes one example =
(image, ONE voxel) -> one scalar. Every voxel of every subject gets a private 256-d embedding in one table
(315,647 x 256 ~= 81M params, **97% of all trainable weights**); the other ~2.4M are shared. Passing 5,000 voxel
indices at once is pure parallelism — voxels never attend to each other, so it is mathematically identical to
5,000 separate calls.

**Why it matters:** This is the entire mechanism for pooling subjects who saw **no shared images** on **different scanners**. Shared
weights get gradients from every image; only the sampled embedding rows get subject-specific gradients. Pooling
helped the *noisiest* subject most (S8 +15% vs S1 +7%) — the fingerprint of genuine data sharing rather than added
capacity. Replacing it with a shared latent is the worst of three ablations, on all 8 subjects, both metrics.

**How to apply:** When an output layer is welded to one instance's dimensionality, try refactoring the output unit into an input.
Cost scales as (instances x units), so ~316k rows is fine and 10^10 is not. Performance was flat in embedding dim
from 64->1024 (0.5170 -> 0.5243), so the win is the factorisation, not capacity — we could shrink to 64 for a
cheaper differentiable encoder.

**Source:** Paper 1 Abstract, Related Work p.3, Fig S9, Table S6; models/encoder_models.py:55,75-113


---

<a id="l-20260810-shared-latent-asymmetry"></a>

## 2026-08-10 — A shared latent bottleneck works when many inputs map to one output space, and fails when one input maps to many
**Type:** insight  
**Phase:** Paper 1 — Universal Encoder

MindEye2 / MindBridge solve multi-subject fMRI *decoding* by projecting every brain into one shared latent, then
using one shared decoder. That cannot reverse for *encoding*. In decoding the inputs differ per subject but the
output space is shared (image space), so subject quirks are nuisance you *want* to strip. In encoding the input is
shared (an image) and the outputs are subject-specific fMRI in completely different spaces — **the
subject-specific variability IS the target**, so compressing it away deletes the answer.

**Why it matters:** The cleanest available argument for why the fashionable trick from the adjacent literature does not transfer, and
it is empirically confirmed by the shared-latent ablation being worst on all 8 subjects. Good related-work paragraph
for our own writeup.

**How to apply:** Before importing a bottleneck/alignment trick from a neighbouring task, ask which END of the pipeline is shared.
Analogy: many-languages -> English tolerates a shared bottleneck; English -> many-languages does not, because
producing correct French *is* the task.

**Source:** Paper 1 App A (appendix p.1); Fig S9


---

<a id="l-20260810-embedding-used-twice"></a>

## 2026-08-10 — Reusing one per-voxel vector as both attention query and readout weights forces a coherent description of the voxel
**Type:** pattern  
**Phase:** Paper 1 — Universal Encoder

The same 256-d embedding is used twice: (1) as the cross-attention **query** over image patches, which learns
that voxel's receptive field ("where"); (2) multiplied by a shared `[256, 640]` dictionary to produce that voxel's
personal 640 readout weights, dot-producted with the features ("which"). The second step is literally
`W = voxel_embed @ feat_embed; out = (features * W).sum(-1)` — no softmax, no nonlinearity, so
"query-conditioned linear readout" is more accurate than the paper's "functional attention".

**Why it matters:** Two coupled jobs from one vector means the model cannot learn incoherent combinations — a face voxel is one that
looks at face-sized regions **and** reads semantic features, both facts about the same voxel. The shared `[256,640]`
dictionary is also the regulariser that makes 315,647 voxels learnable from 70k images: a voxel cannot invent an
arbitrary tuning, only express one in the shared 256-word vocabulary.

**How to apply:** When a per-unit embedding could be injected in several places, inject the SAME vector at each rather than learning
separate ones — free, and it constrains the solution space.

**Source:** models/encoder_models.py:100 (query) and :110-112 (readout); Paper 1 p.5


---

<a id="l-20260810-multi-depth-taps"></a>

## 2026-08-10 — Tap several depths of a frozen backbone and let each output unit learn which depth it needs
**Type:** pattern  
**Phase:** Paper 1 — Universal Encoder

The image passes through DINOv2's 24 blocks **once**, with snapshots after blocks 1, 6, 12, 18, 24; then five
independent parallel read-out heads run and their outputs are concatenated to `[V, 640]`. Each voxel's readout
weights select which of the 5x128 features matter. Unsupervised: a V1 voxel's embedding learns early-layer weights
(edges/texture), an FFA voxel's learns late-layer ones (semantics).

**Why it matters:** The brain is a hierarchy and so is the backbone; forcing everything through the last layer makes early-visual
prediction impossible because the spatial detail has been discarded and is not recoverable. Evidence it works:
V1/V2 reach r ~ 0.65-0.70 while word areas sit at ~0.25-0.40.

**How to apply:** For any "many heterogeneous outputs from one frozen backbone" problem, tap multiple depths and let a learned
per-output weighting choose. One sequential pass, N parallel heads — cheap. Note the number of taps was **not**
ablated, so 5 is unjustified.

**Source:** models/encoder_models.py:86-107; train/train_encoder.py:17,165; Paper 1 p.4, Fig S10a


---

<a id="l-20260810-lora-adapter-not-lora"></a>

## 2026-08-10 — Paper 1's “LoRA” is a rank-7 adapter on each block's output, not a low-rank update to W_o as the paper states
**Type:** gotcha  
**Phase:** Paper 1 — Universal Encoder

App B.2 says *"We only modify the output projection weights (W_o) of the self attention block."* The code adds
`x = x + lora_A[i](lora_B[i](x)) * lora_scale` **after the whole block** — after attention *and* MLP *and*
residual. That is an adapter, not a low-rank weight update, and the two are not equivalent. `lora_B` is
zero-initialised so the term starts at exactly 0 (the model begins as pure DINOv2), `r = 7`,
`lora_scale = alpha/r = 0.01/7 ~= 0.00143` — a very gentle correction. 344K trainable params, 0.1% of the backbone.

**Why it matters:** If we reimplement from the paper we build a different model. Also: the adapter placement means DINOv2's code is
never touched — the wrapper just loops over `.blocks` and adds a term, which is the easy way to adapt any frozen
backbone.

**How to apply:** Trust the code; it produced the released checkpoint. `r`, `alpha`, and LoRA-on/off were **not ablated**, so those
values are unjustified and worth sweeping if we retrain.

**Source:** models/encoder_models.py:39-49,70-73,86-88 vs Paper 1 App B.2; train/train_encoder.py:6-7


---

<a id="l-20260810-vertices-not-voxels"></a>

## 2026-08-10 — This literature says “voxel” but Paper 1's data is fsaverage surface vertices, and per-subject masking destroys the index correspondence
**Type:** insight  
**Phase:** Paper 1 — Universal Encoder

Two stages. (1) FreeSurfer resamples every subject onto one fixed mesh — **163,842 vertices per hemisphere,
identical count for everyone**, and vertex #50,000 IS the same anatomical spot in all subjects. (2) A per-subject
visual-cortex mask then keeps ~39.5k (39,548 for S1-S5, 39,198 for S6-S7, 39,511 for S8 — the differences are just
scanner field-of-view coverage). Because the masks differ and the data is then **flattened to 1D**, from the first
missing vertex onward index N refers to a different location in each subject. The nice correspondence from stage 1
is gone.

**Why it matters:** Explains the confusing "same count / different count" story, and shows the model uses **no** cross-subject voxel
correspondence anywhere — it does not need one. Also the download path proves the space:
`s3://.../subj0N/fsaverage/betas_.../lh.betas_session01.mgh`.

**How to apply:** Never assume index equality across subjects. If we ever need correspondence, learn it (similar behaviour ->
similar embedding) rather than asserting it anatomically.

**Source:** data/scripts/download/download_fmri:33; data/scripts/data_processing/prepare_fmri.py:28-41; Paper 1 Table S3, App B.1


---

<a id="l-20260810-anatomy-neq-function"></a>

## 2026-08-10 — Anatomical registration gives an index correspondence that is functionally wrong more than half the time
**Type:** insight  
**Phase:** Paper 1 — Universal Encoder

Even with indices perfectly fixed, functional areas sit in different places relative to the folds across people.
Paper 1 measures it: **more than 50% of voxel indices in each discovered functional cluster differ between subjects
1 and 2, in anatomically aligned fsaverage space.**

**Why it matters:** It kills the shortcut "vertex N is the same voxel in both subjects, so train one model on both". It also makes the
cross-subject cluster findings *credible* rather than suspicious — had the clusters been index-aligned, a skeptic
could dismiss them as a registration artifact.

**How to apply:** Cite this number whenever justifying a learned rather than asserted correspondence. For our project it also means
food-ROI definitions must be built per subject, never shared by index.

**Source:** Paper 1 p.8


---

<a id="l-20260810-betas-are-solved-for"></a>

## 2026-08-10 — An fMRI beta is solved for from all timepoints jointly — you never read it off at a chosen timepoint
**Type:** insight  
**Phase:** general

The HRF is ~20 s long while NSD showed an image every 4 s, so ~5 responses overlap at every moment and **no
timepoint is clean**. You build `y = X.beta + eps` where `y` = the measured timecourse (known), `X` = the design
matrix (known — constructed from the experiment log's onset times convolved with a chosen HRF shape; it contains no
measurements), and `beta` = one coefficient per image (the unknown, `beta_hat = (X'X)^-1 X'y`). What makes the
system solvable is timing jitter: images at different phases give distinguishable columns of `X`.

**Why it matters:** The natural wrong model — "image 5 was at t=20s, HRF peaks at +5s, so beta_5 = the value at t=25s" — is the
biggest blocker to reading this literature, and no better timepoint exists. It also explains why encoders predict
one scalar per (image, voxel): the beta is time-independent by construction.

**How to apply:** Remember `X` is fully known before any data is seen, and only `beta` is fit. In practice we never run this: NSD
published the betas (`betas_fithrf_GLMdenoise_RR` = per-voxel HRF from a 20-shape library + data-driven nuisance
regressors + ridge). Our pipeline starts from a downloaded array.

**Source:** Friston et al. ~1994 (SPM); Allen et al. 2022 (NSD); NeuroGen p.5; DATA_README


---

<a id="l-20260810-noise-ceiling"></a>

## 2026-08-10 — Repeats give the noise variance for free, so r-squared over the noise ceiling is the only honest accuracy number
**Type:** insight  
**Phase:** general

Show the same stimulus 3x and the measurements disagree; since the input was held **fixed**, that within-stimulus
variance is pure noise (`Var_k(s_i + eps_ik) = Var(eps)` because `s_i` is constant). Average it over stimuli to get
`sigma2_noise`. Pool **all** measurements and you get `sigma2_total = sigma2_signal + sigma2_noise` (the cross term
vanishes under independence). Subtract; `NC = sigma2_signal / sigma2_total`. Since a stimulus-only model can never
predict the noise, **the maximum achievable r-squared IS the noise ceiling**, so
`encoding accuracy = r2/NC = (what you got)/(the best possible)`.

**Why it matters:** r = 0.63 reads as mediocre, but if NC = 0.40 then r2/NC = 1.00 and the model is at ceiling — the residual is the
scanner's fault. The two computations are the same numbers grouped two different ways (per-stimulus variances
averaged, vs one variance over everything pooled). After z-scoring, `sigma2_total = 1` by construction so
`NC = sigma2_signal` directly. Cross-check: Paper 1's Table S6 median r = 0.5221 and Table 1 accuracy 0.69 imply
NC ~ 0.40, mutually consistent.

**How to apply:** Report r2/NC for absolute claims or across subjects with different noise; plain r for two models on identical data.
Two traps: (1) averaging 3 repeats into the target raises achievable r a lot, so "r = 0.52" means "vs a 3-repeat
average" — always say which; (2) with 1 repeat you cannot compute NC at all, which is why our confirmatory scan
should be a contrast. **No noise-ceiling code exists in Paper 1's repo** — `encoder_inference.py:167-172` computes
raw Pearson only, so reproducing Table 1 means fetching NSD's `ncsnr` ourselves.

**Source:** Allen et al. 2022; Paper 1 p.6, Table 1 vs Table S6; inference/encoder_inference.py:167-172


---

<a id="l-20260810-t-is-a-dial"></a>

## 2026-08-10 — ROI boundaries come from a t-threshold nobody treats as a p-value — so report every result at multiple thresholds
**Type:** gotcha  
**Phase:** general

An ROI like FFA comes from a separate localizer scan: compute
`t = c'beta_hat / sqrt(sigma2_hat * c'(X'X)^-1 c)` per voxel, threshold, draw a boundary. The `t` exists to make
voxels **comparable** — a raw beta-difference is not, because a voxel beside a large vessel swings +-8 while a clean
one swings +-0.5. But with ~100k voxels, p<0.05 alone yields ~5,000 false positives, and fMRI noise is temporally
autocorrelated so the t-distribution assumption is already violated. In practice: NSD used **t>0** inside a
hand-drawn polygon; BrainDiVE called that *"very generous"* and used **t>2** (t>5 for figures).

**Why it matters:** "The food region" is a threshold choice, not a fixed object. Any effect that depends on the threshold is not an
effect, and a reviewer will ask. NeuroGen flags the same circularity about itself: face ROIs were defined by a
face>non-face contrast, then used to study face preferences.

**How to apply:** Run every analysis at t in {2, 3, 5} and report ROI size in voxels at each. Also keep the t-statistic mentally
separate from z-scored betas: the t is computed once, offline, by other people, from a **different** experiment;
our model never sees it, only the resulting index list.

**Source:** NeuroGen p.5, p.15; BrainDiVE p.5; standard OLS inference (Gauss-Markov + Student 1908)


---

<a id="l-20260810-region-name-is-inference"></a>

## 2026-08-10 — A discovered “X-selective region” is an inference from a list of top images, not a measurement of X
**Type:** insight  
**Phase:** Paper 4 — BrainExplore

Food cortex has no localizer. It was found by hypothesis-free decomposition of measured fMRI: factor
`B[images x voxels] ~ C[images x k] @ P[k x voxels]`, sort images by a component's coefficient `C[:, j]`, look at
the top ones, and name the component after what you see. The chain is (1) these voxels co-vary — measured;
(2) the top images contain food — measured; (3) **therefore the region represents food — an inference**, and step 3
does not follow. Three independent papers found it this way in 2022-23 (Khosla et al. 2022; Pennock et al. 2023;
Jain et al. 2023).

**Why it matters:** Food images are also warm-toned, high-saturation, close-up, round, glossy and plated. Any of those could be the
real tuning variable. **This gap is exactly what our project attacks**, and it is why BrainCause found ~73% of
activation-based localizations to be false positives. BrainExplore industrialises the decomposition step
(PCA/NMF/ICA/SAE + VLM labelling) and confirms all decompositions are learned from fMRI alone, no image features.

**How to apply:** Whenever reading "region/neuron selective for X", ask whether X was independently manipulated or merely observed
in the top-activating set. If only observed, the label is a hypothesis, and the fix is a matched contrast against
non-X items equated on the plausible confounds.

**Source:** Khosla, Ratan Murty & Kanwisher 2022 Curr Biol 32:4159; Pennock et al. 2023 Curr Biol 33:134; Jain et al. 2023 Comm Biol 6:1 (BrainDiVE ref [5]); BrainExplore arXiv 2512.08560 sec 3.2, Table 1


---

<a id="l-20260810-mask-drops-88-percent"></a>

## 2026-08-10 — The standard ~40k voxel mask discards 88% of cortex, and food cortex was found precisely by looking at unlabelled voxels
**Type:** gotcha  
**Phase:** Paper 1 — Universal Encoder

Paper 1 keeps ~39.5k of 327,684 fsaverage vertices, using the Algonauts (Gifford et al. 2023) vision mask — a
convention, chosen for comparability, not a discovery. Dropped: motor, auditory, prefrontal, somatosensory,
cerebellum, subcortex, most parietal and anterior temporal. Defensible (most of it genuinely does not respond to
images; cost; comparability) but not free.

**Why it matters:** Three real costs, none discussed in the paper: BrainDiVE explicitly used a **larger** set (NSD ROIs unioned with
45 HCP-MMP1 parcels), so a peer group judged the standard mask too small; food cortex was discovered by examining
unlabelled voxels; and object/scene information demonstrably exists in parietal and prefrontal cortex, invisible
here.

**How to apply:** We cannot make any claim about voxels outside the mask. Check the food-mask overlap before building anything.

**Source:** Paper 1 App B.1, Table S3; BrainDiVE run_brain_guidance_category.py:42,62-70


---

<a id="l-20260810-correlation-not-cosine"></a>

## 2026-08-10 — MSE prefers the pattern-blind model when a global factor dominates, and raw cosine barely fixes it — demeaned cosine does
**Type:** gotcha  
**Phase:** general

Four rooms, truth `[21, 19, 22, 18]`. Model A reports the building average `[20,20,20,20]` — useless for "which
room is warmest" — and gets **MSE 2.5**. Model B is 30% miscalibrated but the *order* is perfect (`0.7 x truth`)
and gets **MSE 36.2**. MSE says the useless model is **14x better**, because MSE measures absolute distance and has
no concept of ordering. Cosine is scale-invariant so it prefers B — but `cos_A = 0.9970` vs `cos_B = 1.0000`, a gap
of 0.003, so at a typical weighting it barely helps. **Demean first** (cosine of demeaned vectors IS Pearson
correlation) and A collapses to `r = 0.0` while B is `r = 1.0` — a gap of 1.0, ~300x stronger.

**Why it matters:** Decompose the response as `r_iv = g_i + d_iv`: a global drive shared by all voxels (large, easy to predict) plus
the voxel-specific deviation (small, hard, and the only part we care about). Plain MSE takes the shortcut — nail
`g_i`, set `d_iv ~ 0` — yielding respectable MSE, respectable per-voxel correlation, and a model that cannot answer
"did region X respond more than region Y?". Cosine is scale-invariant but **not shift-invariant**, and the global
drive is a shift, which is exactly why raw cosine is weak here.

**How to apply:** Use a magnitude term *and* a demeaned shape term. **Apply the same rule to our optimisation objective:** never
maximise a raw ROI mean — maximise a contrast, or demean before averaging. Ask of every objective "can this be
satisfied by making everything larger?" If yes, it will be. NeuroGen sec 3.4 is the working precedent: maximising
FFA while *minimising* V1 visibly removes texture from the generated images.

**Source:** utils/train_utils_enc.py:66; Paper 1 App B.3; NeuroGen sec 2.4 p.7, sec 3.4 p.12


---

<a id="l-20260810-mse-1-baseline"></a>

## 2026-08-10 — With z-scored targets, MSE = 1.0 is the free “I learned nothing” baseline
**Type:** insight  
**Phase:** Paper 1 — Universal Encoder

`MSE = mean((0 - target)^2) = mean(target^2)`, and z-scored targets have mean 0 and std 1, so
`mean(target^2) = variance = 1`. A model that ignores the input entirely and always outputs 0 therefore scores
exactly 1.0. Paper 1 reports 0.34-0.47 (Fig S12).

**Why it matters:** A reference point you get for free, with no baseline to train. Also a reminder of what z-scoring means for our
outputs: `+2.0` = "two standard deviations above this voxel's own session mean", and **absolute activation levels
have been removed by construction** — we can only ask "did this voxel respond more than its own average", never
"is FFA more active than PPA overall". That shapes which claims are even expressible.

**How to apply:** Report MSE against 1.0 rather than against nothing. And when writing our objective, remember absolute magnitudes
are not available — only relative ones.

**Source:** Paper 1 Fig S12; prepare_fmri.py:47-50


---

<a id="l-20260810-gradient-at-x0"></a>

## 2026-08-10 — To steer diffusion with an off-the-shelf predictor, take the gradient at the predicted clean sample, never at the noisy latent
**Type:** pattern  
**Phase:** Paper 3 — BrainDiVE

The predictor was trained on clean images, so evaluating it on `x_t` (mostly noise, off-manifold) gives garbage
gradients — the paper says convergence *"is poor without changes to the guidance"*. Fix:
`x0_hat = (x_t - sqrt(1-a)*eps) / sqrt(a)`, then blend
`x_t' = sqrt(1-a)*x0_hat + (1 - sqrt(1-a))*x_t`, decode **that** to pixels, take the gradient there, and set
`eps' = eps - sqrt(1-a) * grad`. Early in sampling (a small) the blend is mostly `x0_hat` so the predictor sees a
plausible image; late (a -> 1) it is mostly `x_t`.

**Why it matters:** This is the generic recipe for plugging **any** differentiable scorer into a pretrained sampler without retraining
anything — the entire API is one function, image -> scalar, assigned to `pipe.brain_tweak`. It also implicitly
guards against exploitation, because the scorer only ever sees plausible images.

**How to apply:** Needs one extra UNet pass per step so the graph connects `latents -> eps -> x0_hat`, inside an
`@torch.enable_grad()` block within an otherwise `@torch.no_grad()` sampler. Guide in **latent** space, not pixels.
Settings that worked for them: SD 2.1-base, **null prompt `""`**, `guidance_scale=0.0` (brain is the only
conditioning), `sag_scale=0.75`, DPM-Solver++ 2nd order, 50 steps, no reranking, fixed 50 steps as the only
stopping rule.

**Source:** Paper 3 sec 3.3; brain_guide_pipeline.py:454-525 (`cond_fn`) and :768-780; run_brain_guidance_category.py:282


---

<a id="l-20260810-four-defenses"></a>

## 2026-08-10 — Guided diffusion has four structural defenses against exploiting the guide — prefer them to a tuned penalty term
**Type:** pattern  
**Phase:** Paper 3 — BrainDiVE

(1) The diffusion prior pulls the latent back toward the image manifold every step, so the scorer gradient is only
a perturbation. (2) The gradient is taken at the predicted clean sample, so the scorer never sees noise.
(3) **L2-normalising the scorer's input embedding removes the radial gradient** — "make the embedding bigger" is
unavailable, only rotation is; the paper notes it *"improves stability of gradient magnitudes w.r.t. the image"*.
(4) Guiding in the 8x-downsampled latent means the VAE decoder lowpass-filters every perturbation, so
high-frequency adversarial texture is unwritable.

**Why it matters:** Defenses 3 and 4 are nearly free and structural, which beats an OOD penalty we would have to tune. Framed
geometrically: BrainDiVE's encoder is one linear layer on a normalised embedding, so "images that maximise voxel i"
is exactly "images whose normalised CLIP embedding aligns with row i of W" — a **cone on the sphere**, not a point.
Many images satisfy it, and **that degree of freedom is precisely what our project exploits**: find the food-free
member of the cone.

**How to apply:** Normalise any embedding we put a head on; keep guidance in latent space; cap the guidance scale. Note **none of the
four is measured** in the paper — the perceptual-cost-vs-activation-gain curve is an easy contribution. Deliberately
run at 10x the intended scale once so we know what exploitation looks like before it happens accidentally.

**Source:** Paper 3 sec 3.2-3.3; encoder_model_vit.py:509-512; README notes 4-5 (use gamma ~100 for OpenCLIP ViT-B, not the 130 in code)


---

<a id="l-20260810-discrete-class-two-stage"></a>

## 2026-08-10 — A class-conditional generator forces a two-stage search because the class is not differentiable — diffusion removes that constraint
**Type:** insight  
**Phase:** Paper 2 — NeuroGen

BigGAN-deep needs `G(c, z)`: `c` is a one-hot over 1,000 ImageNet classes (discrete, non-differentiable) and `z` is
a 128-d noise vector (differentiable). So NeuroGen must (1) *enumerate* — generate 100 images per class for all 1,000
classes, average predicted activation, keep the top 10; then (2) *optimise* — Adam on `z` within each top class,
1,000 steps, lr 0.01, objective `-activation + 0.001*||z||^2`. Only `z` is trained; generator and encoder are frozen.
Stopping is not convergence: run 1,000 steps and `argmax` over all of them.

**Why it matters:** It caps what can be discovered at "the best member of 1,000 ImageNet classes". BrainDiVE's central complaint is
exactly this (*"restricted to ImageNet images, which are primarily images of single objects"*), and replacing it with
a LAION-2B diffusion prior *"over three magnitudes larger"* is what made sub-ROI discovery possible. For our project
this matters doubly: we need images that are *definitionally unusual* (concept-absent yet region-activating), and a
1,000-class prior is a bad tool for that.

**How to apply:** Use diffusion, not a class-conditional GAN. Keep NeuroGen's honest bits: the explicit `||z||` regulariser and the
`argmax`-over-steps selection. Discrepancies to know: paper uses truncation **0.4**, the CLI default is **1**; and
`closure()` is called twice per step (`neurogen.py:85-86`), doubling compute and logging a post-update loss.

**Source:** NeuroGen sec 2.4 p.7, sec 3.2; neurogen.py:32-105,27; BrainDiVE sec 2


---

<a id="l-20260810-generator-prior-biases-discovery"></a>

## 2026-08-10 — In a generate-and-discover loop the generator's training distribution decides what you are able to discover
**Type:** gotcha  
**Phase:** Paper 2 — NeuroGen

NeuroGen's headline finding is that face-selective regions respond strongly to **dog** faces, with the dog/human
ratio varying by individual and region. The authors then note: 120 of ~1000 ImageNet classes are dog breeds, and
ImageNet dog images are typically close-up faces. Their biggest discovery is in the single category that is 12% of
their generator's training data.

**Why it matters:** The finding survives — they validated the ratio against measured fMRI t-statistics on natural images, entirely
independent of the generator — but the *direction* of discovery was steered by the prior. Activation maximisation
reports "what the model likes, restricted to what the generator can draw.”

**How to apply:** Before claiming a discovery from a generate-and-rank loop, ask what the generator over- and under-represents, and
validate on data the generator had no hand in (e.g. rank **natural** images by **measured** response and check the
same pattern). Expect modern diffusion priors to be biased toward aesthetic, centred, well-lit subjects — which for
us means biased toward *plated, beautiful* food.

**Source:** NeuroGen Discussion p.14


---

<a id="l-20260810-synthesis-amplifies"></a>

## 2026-08-10 — Optimising against a learned model can amplify a real signal that is statistically invisible in raw measurements
**Type:** insight  
**Phase:** Paper 2 — NeuroGen

NeuroGen's strongest result is not that synthetic images beat natural ones on *predicted* activation (they do, in
192/192 subject x region cases — but predicted, never measured). It is that the dog-vs-human ratio in the top-10
**synthetic** images correlates with the measured fMRI dog-vs-human t-statistic at Spearman **r = 0.67
(p = 1.2e-5)**, rising to **0.75 (p = 2e-7)** for top-100 — while ranking the top-10 **natural** images by
**measured** fMRI gives **r = 0.29, p = 0.086, not significant**.

**Why it matters:** The encoder acts as a denoiser: its predictions are smoother than single measurements, so optimising against it
surfaces individual differences buried in measurement noise. That is a quantified reason to prefer synthesis over
retrieval — and a different argument from "synthesis reaches higher activation".

**How to apply:** Legitimate amplifier **provided** the amplified pattern is validated against an independent measured statistic, as
they did. Amplification without that check is just model-fitting. Relevant to our rung-1-vs-rung-3 tradeoff:
retrieval is safer, synthesis is more sensitive.

**Source:** NeuroGen sec 3.2-3.3 p.10-11


---

<a id="l-20260810-activation-neq-representation"></a>

## 2026-08-10 — Ranking units by activation gives ~73% false positives; ranking by a counterfactual difference drops it to 23%
**Type:** insight  
**Phase:** Paper 5 — BrainCause

BrainCause tested 260 concepts. For each it built positives; **semantic negatives** (LLM-proposed concepts that
co-occur with the target but are distinct — for *human hands*: human body, human legs, robot hands); and
**counterfactual negatives** (edit the positive to remove the concept, keep everything else). A voxel scores on
activation if it fires for positives; it scores on **causality** only if it fires *more* for positives than for the
hardest semantic negative and the hardest edit. Activation-only: FPR 73.4%, TPR 26.6%. Causal: FPR 23.0%,
TPR 38.7% — fewer false positives **and** more true ones.

**Why it matters:** Generalises far past fMRI: "this neuron/feature fires for X" is the standard interpretability claim, and ~3 in 4
such claims are about something correlated with X. The failure is invisible without an explicit contrast, because
the positives genuinely do drive the unit. Validation: top-100 causal voxels land inside the expected NSD ROI for
bodies 99%, words 99%, faces 90%, places 74%.

**How to apply:** For any selectivity claim, add both negative sets and report the **difference**, not the activation. Their pipeline
is worth copying wholesale for our controls: Gemma-3-27B proposes, FLUX.2 generates/edits, **Qwen3-VL-8B verifies**
concept absence with a second pass under a different prompt.

**Source:** BrainCause arXiv 2605.23895 sec 3.1-3.2, Fig 4, Tables 1-2


---

<a id="l-20260810-shared-representation-metric"></a>

## 2026-08-10 — If your model and your metric read the same representation, the metric cannot detect the failure mode you care about
**Type:** gotcha  
**Phase:** Paper 3 — BrainDiVE

BrainDiVE's encoder is one linear layer on a CLIP image embedding, and its headline metric is CLIP zero-shot
5-way classification of the generated images. Both branches read the same CLIP space, so an image that adversarially
rotates the embedding toward "food" will *simultaneously* raise the encoder's food-region prediction **and** be
classified as food. The metric is structurally blind to exactly the failure mode it is used to rule out. Partial
mitigation: they probe with CoCa ViT-L/14 while the encoder uses ViT-B/16 — different model, same family, same
LAION-2B data. Their human 2AFC studies are the real evidence.

**Why it matters:** Shared-representation evaluation looks rigorous and produces numbers that cannot fail. Their reported gains are
large (BrainDiVE-100 mean 77.0/78.8 vs NSD top-100 53.0/63.6) and the "all stimuli" row is exactly 20.0 = 5-way
chance, which is a nice internal check — but the circularity caps how much the metric can prove.

**How to apply:** **Do not use a CLIP-based classifier to verify "no food present"** if our scorer is CLIP- or DINOv2-based. Use a
VLM from a different family, with a second verification pass. Keep at least one human-judgement or
real-measurement check in the chain.

**Source:** Paper 3 Table 1, sec 4.1; BrainCause sec 3.1


---

<a id="l-20260810-precommit-seeds"></a>

## 2026-08-10 — Commit the random-seed list to git before generating anything, so you cannot cherry-pick and can say so
**Type:** pattern  
**Phase:** Paper 3 — BrainDiVE

BrainDiVE loads a fixed `random_seeds.pkl` indexed by (subject, region) and generates ~1,000 images per cell from
those seeds. The authors therefore cannot select favourable seeds after the fact, and "not cherry-picked" becomes
verifiable rather than asserted.

**Why it matters:** Generative results are uniquely vulnerable to silent seed selection and a reader usually cannot detect it.
Pre-commitment converts an unfalsifiable promise into a checkable artifact at zero cost.

**How to apply:** Write the seed list to a file and commit it before the first generation run. Report N generated, not N shown. Same
discipline for the ROI threshold sweep — fix the set {2, 3, 5} up front and report all of them.

**Source:** run_brain_guidance_category.py:35,91-92,179


---

<a id="l-20260810-leakage-audit"></a>

## 2026-08-10 — Paper 1's train/test split is clean, but normalisation and voxel selection both touched test data
**Type:** insight  
**Phase:** Paper 1 — Universal Encoder

Five checks. **Clean:** the ~1,000 shared test images (`type_sample==2`) never enter training for any subject
(`embeds_single = embeds[type_sample == 1]`), and checkpoint selection is on a 2,500-image val split while the
tables report test. **Not clean:** (a) *transductive normalisation* — the per-session z-score computes mean/std over
all ~750 trials including test trials, and ~10% of trials per session are test trials (1,000 shared x 3 repeats /
30,000 trials); (b) *voxel-selection circularity* — the Algonauts mask selected reliably-responsive voxels using
NSD data that includes the test images.

**Why it matters:** Both non-clean items are standard practice and their practical impact is small (a shift in a normalisation constant
is not a leak a model can exploit; the voxel mask inflates all methods equally so rankings stay fair). But both are
valid reviewer objections, and (b) means absolute r values are optimistic relative to unselected cortex.

**How to apply:** In our own confirmatory analysis, compute normalisation statistics from **training trials only** — costs nothing,
removes the objection. Trust published rankings; discount absolute values. Also note the naming trap: the object
called `test_generator` is built from `fmri_dataset_val`, so "test" in the code means validation.

**Source:** train/train_encoder.py:69-79,140,197-200; prepare_fmri.py:49,121,125,131; Paper 1 App B.1


---

<a id="l-20260810-verify-loss-in-code"></a>

## 2026-08-10 — Verify the loss and every hyperparameter in the training entry point — papers and class defaults both lie
**Type:** gotcha  
**Phase:** Paper 1 — Universal Encoder

Paper 1 App B.3 states `L = a*MSE - (1-a)*cos` with `a = 0.1`, i.e. cosine weighted **9x** MSE. The code is
`mse_loss - 0.1*cosine_similarity`, i.e. MSE weighted **10x** cosine — a ~90x difference in relative weighting, and
it changes which model wins. Separately, the `encoder_param` class defaults (`embed_dim_vox=64`, `inner_ch=32`,
`in_channels=1280`) are **all overridden** by the training script to the paper's values (256, 128, 1024), so reading
the class alone gives three wrong hyperparameters.

**Why it matters:** The loss is the single most important line in a training repo and the one most often written from memory in an
appendix. Reimplementing from the paper would train a different model with no obvious symptom.

**How to apply:** Grep the training entry point, not the model class and not the paper. When they disagree, the code produced the
released checkpoint — trust the code, and test both if we retrain. Also unrecorded in the paper: AdamW-equivalent
details, `ReduceLROnPlateau(mode='max', factor=0.1, patience=4, threshold=0.003)` on the 75th-percentile val
correlation, 30 epochs, dropout 0.25, +-3 px shift augmentation, ~1 day on one Quadro RTX 8000, ~50 ms/image inference.

**Source:** utils/train_utils_enc.py:66 vs App B.3; models/encoder_models.py:12-22 vs train/train_encoder.py:11-12,55-57,96-101,164,190-204; App B.3


---

<a id="l-20260810-index-convention-bug"></a>

## 2026-08-10 — Two scripts in Paper 1's repo disagree on 0- vs 1-based subject IDs, silently swapping two subjects and crashing on the eighth
**Type:** gotcha  
**Phase:** Paper 1 — Universal Encoder

`prepare_fmri.py:109-114` writes subject IDs as **1..8**; `utils/datasets.py:31-32` indexes a length-8 array with
them; `train/train_encoder.py:78,139` passes them through with **no `-1`**. Effect: subject 7 gets subject 8's voxel
count (39,511 instead of 39,198) *and* subject 8's embedding-table offset — so two subjects' identities silently
swap for the whole run — and subject 8 raises `IndexError` within the first few shuffled batches. That 0-based is
intended is confirmed by `train_encoder_transfer.py:91,102`, which builds `np.zeros(...)` labels with a length-1
count array.

**Why it matters:** It blocks retraining from the public release, though not the released checkpoint (trained internally, presumably
with 0-based data). Most likely a release-packaging mismatch, not a science flaw.

**How to apply:** Fix: `single_sub = fmri_data['single_sub'] - 1`. Verify in 10 seconds:
`print(d['single_sub'].min(), d['single_sub'].max())` — "1 8" means apply the `-1`, "0 7" means leave it. **General
rule: before any retrain from released research code, print min/max of every index array and check against the
length of what it indexes.** Related: validation uses `sample=False` so voxel vectors have per-subject lengths and
default collation will fail across mixed subjects — expect to need `batch_size=1` for val.

**Source:** prepare_fmri.py:109-114; utils/datasets.py:31-38; train/train_encoder.py:78,139; train_transfer/train_encoder_transfer.py:91,102


---

<a id="l-20260810-normalization-footguns"></a>

## 2026-08-10 — Preprocessing mismatches between these repos fail silently — the symptom is near-identical predictions for different inputs
**Type:** gotcha  
**Phase:** Project design

Two traps. (1) Paper 1's **training** transform divides by 255 *inside* the function; both **inference** scripts do
not, and the caller does `trans_imgs_shift(images[i] / 255.0)`. So the inference scripts require uint8 0-255 input —
hand them already-`[0,1]` floats, exactly what a diffusion VAE decoder outputs, and you divide by 255 twice and feed
a near-black image with no error. (2) BrainDiVE's `cond_fn` normalises with **OpenAI CLIP** statistics because its
encoder is CLIP-based; Paper 1's encoder wants **ImageNet** statistics. Porting one into the other requires
un-normalising then re-normalising.

**Why it matters:** Both produce plausible-looking outputs and plausible-looking numbers. This is the single most likely way our port
breaks, and it will not raise an exception.

**How to apply:** Diagnose with **variance, not error**: compute `pred.std(axis=0).mean()` across a batch of *different* inputs; near
zero means the model is effectively seeing one input. Make that a permanent smoke test, in CI, not once. When wiring
model A into model B, write both conventions down and convert in exactly one place.

**Source:** train/train_encoder.py:118 vs inference/encoder_inference.py:34-37,145 and inference/predict_fmri_for_images.py:33-36,86; brain_guide_pipeline.py:496-504


---

<a id="l-20260810-braindive-release-broken"></a>

## 2026-08-10 — BrainDiVE's release can generate but cannot train, and needs two fixes before it imports
**Type:** gotcha  
**Phase:** Paper 3 — BrainDiVE

(1) `run_brain_guidance_category.py:17` does `import encoder_model_vit` but lines 27 and 121 call `model_vit.*` —
`NameError`. `encoder_trainer.py:10` and `encoder_eval.py:7` import a `model_vit` module that is **not in the repo**
(the commit history shows a `"Refactor names"` pass with inconsistent renames). Fix:
`import encoder_model_vit as model_vit`. (2) `downproject_CLIP_split_linear.__init__` creates **only**
`final_*_higher`, but its `forward` references `self.fuse1`, `self.act1`, `self.reduce_1`, `self.squeeze`,
`final_*_early` — none of which exist. So `forward_higher()` works (and is what generation uses),
while `forward()` and `forward_early()` raise `AttributeError`.

**Why it matters:** We can use BrainDiVE as a frozen validation encoder or port its pipeline, but **cannot retrain its encoder from the
release as-is**. Consistent with README note 1 ("in the paper only used the higher visual part"). Also missing:
encoder weights ("download here" with no link), the food/refined-ROI masks, and both demo notebooks; all paths are
hardcoded to `/ocean/projects/soc220007p/aluo/`.

**How to apply:** Do not budget time for end-to-end BrainDiVE reproduction. Instead port `brain_guide_pipeline.py` to use Paper 1's
encoder as `brain_tweak` — the pipeline is encoder-agnostic by design and needs only one `image -> scalar` function.
Reimplementing the head is one `Linear(512, N)` on a normalised CLIP embedding.

**Source:** run_brain_guidance_category.py:17,27,121; encoder_model_vit.py:482-524; encoder_trainer.py:10,168; encoder_eval.py:7; README notes


---

<a id="l-20260810-zero-nsd-stub"></a>

## 2026-08-10 — You can run Paper 1's pretrained encoder on arbitrary images today, with no NSD access, by stubbing one npz key
**Type:** pattern  
**Phase:** Paper 1 — Universal Encoder

`inference/predict_fmri_for_images.py` reads `fmri_v2.npz` for exactly one thing: `num_voxels_subjects`. So
fabricate it from Table S3 —
`np.savez("data/nsd_data/fmri_v2.npz", num_voxels_subjects=np.array([39548]*5+[39198]*2+[39511]))` — then
`python inference/predict_fmri_for_images.py --images my.npy --subjects 1 --region full` produces a
`(N, 39548)` prediction array. The checkpoint comes from HuggingFace `RomanBeliy/Brain-IT`,
`checkpoints/encoder_ch128.pth`.

**Why it matters:** Unblocks all encoder work before the NSD Data Access Agreement clears, which is the long pole. Region-restricted
prediction (`--region floc-faces`) additionally needs the Drive ROI masks; `region="full"` needs nothing.

**How to apply:** Gotchas: the checkpoint is a **pickled whole model**, so run from the repo root (`models.encoder_models` must be
importable) and note `torch.hub.load("...dinov2_vits14_reg")` exists purely to register classes for unpickling even
though the model is ViT-L. `env.yml` pins `torch==2.1.0`; on torch >= 2.6 add `weights_only=False`. `env.yml` is a
Linux/CUDA export and will not solve on macOS — hand-install instead. Also: the README references
`bash data/scripts/run_all_data_processing`, which **does not exist**.

**Source:** inference/predict_fmri_for_images.py:101-110; data/scripts/download/download_encoder.py:11-12; Paper 1 Table S3; env.yml:276; README:56


---


# Part C — Topic notes

Each chapter collects pointers into Part A. Chapters are created on demand.

- **C.1** — fMRI fundamentals: voxels, betas, noise and ROIs
- **C.2** — Brain-encoder architectures
- **C.3** — Loss and objective design
- **C.4** — Guided generation mechanics
- **C.5** — Validity, causality and evaluation
- **C.6** — Released-code archaeology
- **C.7** — Our project: concept-absent activation
- **C.8** — Workflow, tooling and repo conventions


## C.1 — fMRI fundamentals: voxels, betas, noise and ROIs

*Scope: prerequisites for reading any paper in this folder. Phase: general / Paper 1.*


> **Learned 2026-08-10 —** This literature says “voxel” but Paper 1's data is fsaverage surface vertices, and per-subject masking destroys the index correspondence  
> → [full entry](#l-20260810-vertices-not-voxels)

> **Learned 2026-08-10 —** Anatomical registration gives an index correspondence that is functionally wrong more than half the time  
> → [full entry](#l-20260810-anatomy-neq-function)

> **Learned 2026-08-10 —** An fMRI beta is solved for from all timepoints jointly — you never read it off at a chosen timepoint  
> → [full entry](#l-20260810-betas-are-solved-for)

> **Learned 2026-08-10 —** Repeats give the noise variance for free, so r-squared over the noise ceiling is the only honest accuracy number  
> → [full entry](#l-20260810-noise-ceiling)

> **Learned 2026-08-10 —** ROI boundaries come from a t-threshold nobody treats as a p-value — so report every result at multiple thresholds  
> → [full entry](#l-20260810-t-is-a-dial)

> **Learned 2026-08-10 —** A discovered “X-selective region” is an inference from a list of top images, not a measurement of X  
> → [full entry](#l-20260810-region-name-is-inference)

> **Learned 2026-08-10 —** The standard ~40k voxel mask discards 88% of cortex, and food cortex was found precisely by looking at unlabelled voxels  
> → [full entry](#l-20260810-mask-drops-88-percent)


## C.2 — Brain-encoder architectures

*Scope: how image->fMRI encoders are built and why. Phase: Paper 1.*


> **Learned 2026-08-10 —** Making the output one scalar per (image, voxel) is what removes subject count, voxel count and scanner from the problem  
> → [full entry](#l-20260810-voxel-centric)

> **Learned 2026-08-10 —** A shared latent bottleneck works when many inputs map to one output space, and fails when one input maps to many  
> → [full entry](#l-20260810-shared-latent-asymmetry)

> **Learned 2026-08-10 —** Reusing one per-voxel vector as both attention query and readout weights forces a coherent description of the voxel  
> → [full entry](#l-20260810-embedding-used-twice)

> **Learned 2026-08-10 —** Tap several depths of a frozen backbone and let each output unit learn which depth it needs  
> → [full entry](#l-20260810-multi-depth-taps)

> **Learned 2026-08-10 —** Paper 1's “LoRA” is a rank-7 adapter on each block's output, not a low-rank update to W_o as the paper states  
> → [full entry](#l-20260810-lora-adapter-not-lora)


## C.3 — Loss and objective design

*Scope: what to optimise, and the shortcuts a naive objective invites. Phase: Paper 1 / Project design.*


> **Learned 2026-08-10 —** MSE prefers the pattern-blind model when a global factor dominates, and raw cosine barely fixes it — demeaned cosine does  
> → [full entry](#l-20260810-correlation-not-cosine)

> **Learned 2026-08-10 —** With z-scored targets, MSE = 1.0 is the free “I learned nothing” baseline  
> → [full entry](#l-20260810-mse-1-baseline)


## C.4 — Guided generation mechanics

*Scope: turning a frozen encoder into an image generator. Phase: Papers 2-3.*


> **Learned 2026-08-10 —** To steer diffusion with an off-the-shelf predictor, take the gradient at the predicted clean sample, never at the noisy latent  
> → [full entry](#l-20260810-gradient-at-x0)

> **Learned 2026-08-10 —** Guided diffusion has four structural defenses against exploiting the guide — prefer them to a tuned penalty term  
> → [full entry](#l-20260810-four-defenses)

> **Learned 2026-08-10 —** A class-conditional generator forces a two-stage search because the class is not differentiable — diffusion removes that constraint  
> → [full entry](#l-20260810-discrete-class-two-stage)

> **Learned 2026-08-10 —** In a generate-and-discover loop the generator's training distribution decides what you are able to discover  
> → [full entry](#l-20260810-generator-prior-biases-discovery)

> **Learned 2026-08-10 —** Optimising against a learned model can amplify a real signal that is statistically invisible in raw measurements  
> → [full entry](#l-20260810-synthesis-amplifies)


## C.5 — Validity, causality and evaluation

*Scope: telling a discovery apart from an adversarial example. Phase: Papers 3-5.*


> **Learned 2026-08-10 —** Ranking units by activation gives ~73% false positives; ranking by a counterfactual difference drops it to 23%  
> → [full entry](#l-20260810-activation-neq-representation)

> **Learned 2026-08-10 —** If your model and your metric read the same representation, the metric cannot detect the failure mode you care about  
> → [full entry](#l-20260810-shared-representation-metric)

> **Learned 2026-08-10 —** Commit the random-seed list to git before generating anything, so you cannot cherry-pick and can say so  
> → [full entry](#l-20260810-precommit-seeds)

> **Learned 2026-08-10 —** Paper 1's train/test split is clean, but normalisation and voxel selection both touched test data  
> → [full entry](#l-20260810-leakage-audit)


## C.6 — Released-code archaeology

*Scope: what actually runs in the three cloned repos, and what silently does not. Phase: all.*


> **Learned 2026-08-10 —** Verify the loss and every hyperparameter in the training entry point — papers and class defaults both lie  
> → [full entry](#l-20260810-verify-loss-in-code)

> **Learned 2026-08-10 —** Two scripts in Paper 1's repo disagree on 0- vs 1-based subject IDs, silently swapping two subjects and crashing on the eighth  
> → [full entry](#l-20260810-index-convention-bug)

> **Learned 2026-08-10 —** Preprocessing mismatches between these repos fail silently — the symptom is near-identical predictions for different inputs  
> → [full entry](#l-20260810-normalization-footguns)

> **Learned 2026-08-10 —** BrainDiVE's release can generate but cannot train, and needs two fixes before it imports  
> → [full entry](#l-20260810-braindive-release-broken)

> **Learned 2026-08-10 —** You can run Paper 1's pretrained encoder on arbitrary images today, with no NSD access, by stubbing one npz key  
> → [full entry](#l-20260810-zero-nsd-stub)


## C.7 — Our project: concept-absent activation

*Scope: scope, gap analysis and design decisions for the food-absent experiment. Phase: Project design.*


> **Learned 2026-08-10 —** Our project is the unoccupied cell between BrainDiVE and BrainCause — and BrainCause names it as future work  
> → [full entry](#l-20260810-gap-analysis)

> **Learned 2026-08-10 —** Run retrieval on measured data before writing any generator — it de-risks the project and hands you the confound  
> → [full entry](#l-20260810-three-rungs)

> **Learned 2026-08-10 —** Lock a structurally different validation encoder and never let it see a gradient — not once  
> → [full entry](#l-20260810-two-encoders)

> **Learned 2026-08-10 —** Frame the experiment so all three outcomes are publishable, then a null result is a contribution  
> → [full entry](#l-20260810-both-outcomes)

> **Learned 2026-08-10 —** Design the confirmatory fMRI as a between-condition contrast, not a correlation, because contrasts need no noise ceiling  
> → [full entry](#l-20260810-fmri-contrast)

> **Learned 2026-08-10 —** There is no food localizer and BrainDiVE's food masks are not in its repo — resolve the food ROI in week 1 or be blocked  
> → [full entry](#l-20260810-food-mask)

> **Learned 2026-08-10 —** Check the food-mask / Algonauts-mask overlap before optimising, or we may be optimising a truncated region  
> → [full entry](#l-20260810-mask-overlap)


## C.8 — Workflow, tooling and repo conventions

*Scope: catch-all for one-off learnings. Phase: general.*


*No entries yet.*


## C.9 — Contrast-loss experiments: escaping a category while driving its ROI

*Scope: the brainconcept faces experiments — the loss, the drive fix, E2/E3/text results, and the theory of why cosine can't leave a class. Phase: Project design.*


> **Learned 2026-09-29 —** A cosine-to-prototype repel cannot leave a category; only a two-class discriminant can  
> → [full entry](#l-20260929-cosine-vs-classifier)

> **Learned 2026-09-29 —** A classifier repel moves toward whatever you label "not-X" — we labelled animals non-face, so it made animal faces  
> → [full entry](#l-20260929-classifier-relocates)

> **Learned 2026-09-28 —** Contrast selectivity is gamed by suppressing the complement; clamp it at zero to force real drive  
> → [full entry](#l-20260928-comp-clamp)

> **Learned 2026-09-28 —** Image-bank → human faces, classifier → animal faces, text → inert: every strong ROI driver is some kind of face  
> → [full entry](#l-20260928-round2-result)

> **Learned 2026-09-28 —** Anchor cosine is decoupled from perceptual faceness — the repel hits its hinge yet the image stays a face  
> → [full entry](#l-20260928-imgcos-decoupled)

> **Learned 2026-09-27 —** The contrast/floor loss: a drive release-valve, mean+max hinged repel, and a decode→re-encode realness monitor  
> → [full entry](#l-20260927-contrast-loss)
