#!/usr/bin/env python3
"""Merge: full original teaching guide (MathJax inlined) + a new Results part."""
import base64, io, re
from pathlib import Path
from PIL import Image

ROOT = Path("/Users/danielnurieli/AI work/MindReading")
SCR = Path("/private/tmp/claude-501/-Users-danielnurieli-AI-work-MindReading/158d213d-99e6-4714-affb-65a69598a469/scratchpad")
PULLED = ROOT / "brainguide/experiments/pulled"
ORIG = (ROOT / "session_answers_2026-08-25.html").read_text()
# clarify the dense "Jacobian rotates & stretches" sentence with a concrete analogy
ORIG = ORIG.replace(
    'Because latent-space and pixel-space have different sizes and shapes, this chart also '
    '<b>rotates and stretches</b> any direction you send through it — a clean latent direction '
    'can come out as a lopsided pixel direction, and vice-versa.</p>',
    'Because latent space (~16k numbers) and pixel space (~786k) differ in size and wiring, this '
    'chart <b>rotates and stretches</b> every direction. Picture the decoder as a <b>mixing board</b>: '
    'the latent is ~16k knobs, the image ~786k light-bulbs, and <b>no knob controls a single bulb</b> — '
    'turning one knob shifts a whole coordinated <i>pattern</i> of bulbs. So a simple, clean latent move '
    'comes out as a complicated, spread-out pixel pattern (the “rotate”), and some knobs are <i>loud</i> '
    '(a tiny turn floods the stage) while others are <i>quiet</i> (cranking them barely does anything) — '
    'that is the “stretch.” A clean latent direction can therefore land as a lopsided pixel direction, '
    'and vice-versa.</p>')
# clarify WHY CLIP wishes are reachable but DINOv2's often are not
ORIG = ORIG.replace(
    'A CLIP encoder tends to wish for reachable things; a DINOv2 encoder often wishes for '
    'unreachable ones. Hold that thought — Step 2 makes "reachable vs. not" exact.</p>',
    'A CLIP encoder tends to wish for <b>reachable</b> things; a DINOv2 encoder often for '
    '<b>unreachable</b> ones — here is <i>why</i>. An encoder\'s "wish" is whatever change most raises '
    'its score, which depends on what it <i>notices</i>. CLIP was trained matching images to captions, '
    'so it notices <b>semantic content</b> ("make this look more like a face") — a real-image change SD '
    'can render (SD was literally built around CLIP). DINOv2 was trained by self-supervised '
    '<b>patch-matching</b>, so it is sensitive to fine texture / patch statistics; its <i>cheapest</i> '
    'way to spike a score is often a precise high-frequency texture tweak — which is not a coherent real '
    'image, so the decoder cannot make it. In short: <b>reachable = "stays among real photos" = the '
    'language SD speaks (CLIP)</b>. (It is a tendency, not a law — a DINOv2 wish has a reachable part too; '
    'the trouble is the unreachable part is large.) Hold that thought — Step 2 makes "reachable vs. not" '
    'exact.</p>')
# insert a full "notation key" box right after the Part-0 "two words" intro
ORIG = ORIG.replace(
    '\n  <h4>The one-paragraph version of what goes wrong',
    '\n  <div class="cbox"><h4>Notation key — every symbol, in one place</h4>'
    '<p><b>The objects.</b> \\(x\\) = the image (pixels); \\(z\\) = the latent (the compressed code SD '
    'edits); \\(D\\) = the decoder, \\(x=D(z)\\); \\(r\\) = the reward (one number: how hard the ROI '
    'fires = average of the encoder\'s per-voxel predictions); \\(\\mathcal M\\) = the manifold (the set '
    'of real-looking images).</p>'
    '<p><b>Gradients (calculus).</b> \\(\\nabla\\) = gradient, "direction of steepest increase"; '
    '\\(\\nabla_x r\\) = gradient w.r.t. <em>pixels</em> = the "pixel-wish" (one value per pixel); '
    '\\(\\nabla_z\\) = gradient w.r.t. the <em>latent</em>.</p>'
    '<p><b>The translator (linear algebra).</b> \\(J_D\\) = the decoder\'s Jacobian, the table "nudge '
    'this latent → these pixels move" (\\(\\partial D/\\partial z\\)); \\(J_D^{\\top}\\) = its transpose '
    '— go backward, pixel-direction → latent-direction (what backprop applies).</p>'
    '<p><b>The split (differential geometry).</b> \\(P_{\\mathcal M}\\) = projection onto the manifold\'s '
    'tangent plane ("keep only the part that slides <em>along</em> real-image space"); \\(I\\) = identity, '
    'so \\(I-P_{\\mathcal M}\\) = "the part that pokes <em>off</em> it"; \\(g_\\parallel=P_{\\mathcal M}'
    '\\nabla_x r\\) = honest/on-manifold half; \\(g_\\perp=(I-P_{\\mathcal M})\\nabla_x r\\) = '
    'off-manifold/"cheat" half; \\(\\lVert\\cdot\\rVert\\) = the length of a vector; '
    '\\(\\alpha=\\lVert g_\\parallel\\rVert/\\lVert\\nabla_x r\\rVert\\) = fraction of the wish that is '
    'honest (0→1) — this guide\'s own defined ratio.</p>'
    '<p><b>⚠ One clash:</b> Part II\'s diffusion math writes \\(\\bar\\alpha_t\\) for the '
    '<em>noise-schedule</em> weight at step \\(t\\) — a <b>different</b> thing from the alignment '
    '\\(\\alpha\\) above; same Greek letter, unrelated meaning.</p></div>\n'
    '\n  <h4>The one-paragraph version of what goes wrong')
# clarify "normalize the step" (Part II item 5)
ORIG = ORIG.replace(
    '<h4>5 · The one knob that matters: normalize the step</h4>',
    '<h4>5 · The one knob that matters: normalize the step</h4>'
    '<div class="cbox"><h4>In plain words</h4>'
    '<p>The raw guidance gradient\'s <em>size</em> is unpredictable — it swings across denoising steps '
    'and ROIs; too small does nothing, too large explodes into texture/blank. So no single fixed '
    '"guidance scale" works everywhere. <b>Fix:</b> don\'t fix the step size — tie it to the model\'s own '
    'update size. Take only the gradient\'s <em>direction</em> and give it a magnitude equal to a fixed '
    'fraction of the denoiser\'s own noise-prediction \\(\\lVert\\varepsilon\\rVert\\): '
    '\\(\\text{guidance}= \\text{scale}\\cdot\\lVert\\varepsilon\\rVert\\cdot \\nabla r/\\lVert\\nabla r'
    '\\rVert\\). Now one dimensionless number works for <em>every</em> ROI and timestep. '
    '<b>Analogy:</b> cruise control — always nudge by X% of whatever the model is doing right now (early '
    'steps make big moves, late steps tiny ones). This ROI-independent normalized step is literally what '
    'made your images appear at all.</p></div>')
# proof-honesty note at Step 3 (the alpha claim)
ORIG = ORIG.replace(
    'Why CLIP is aligned (high',
    '<div class="cbox"><h4>Is Step 3 proved, or a model? (honest)</h4>'
    '<p>Read \\(\\alpha\\) as a <b>model</b>, not a measured constant: we do not numerically compute '
    '\\(g_\\parallel/g_\\perp\\) (projecting onto the manifold is hard). What is <em>proved</em> is the '
    'scale-invariance below (turning the knob up scales both halves equally, so \\(\\alpha\\) cannot '
    'change — pure algebra). What is an <em>architectural argument</em> is CLIP-high / DINOv2-low '
    '(next). What is <em>measured</em> is the consequence: our controlled A/B (same guidance, CLIP '
    'coherent vs DINOv2 texture — see Part VII) and the held-out reward-hacking. So the quality gap is a '
    'fact; "\\(\\alpha\\) explains it" is the best-supported explanation.</p></div>'
    'Why CLIP is aligned (high')
MJS = (SCR / "mathjax-tex-svg.js").read_text()
OUT = SCR / "merged.html"
ROIS = ["faces", "places", "words", "bodies"]

def datauri(rel, px=240, q=78):
    p = PULLED / rel
    if not p.exists():
        return ""
    im = Image.open(p).convert("RGB").resize((px, px), Image.LANCZOS)
    b = io.BytesIO(); im.save(b, "JPEG", quality=q)
    return "data:image/jpeg;base64," + base64.b64encode(b.getvalue()).decode()

def cell(rel, label=None):
    src = datauri(rel)
    inner = (f'<img loading="lazy" alt="{label or rel}" src="{src}">' if src
             else '<span class="miss">generating…</span>')
    return f'<div class="rc">{inner}</div>'

def grid(cols, rows, tpl):
    """cols/rows: label lists. tpl(colkey,rowkey)->relpath. cols are (key,label)."""
    n = len(cols)
    head = '<div class="rrow rhead"><div class="rl"></div>' + "".join(
        f'<div class="ch">{lbl}</div>' for _, lbl in cols) + "</div>"
    body = ""
    for rk, rl in rows:
        body += f'<div class="rrow" style="--n:{n}"><div class="rl">{rl}</div>' + "".join(
            cell(tpl(ck, rk), f"{ck} {rk}") for ck, _ in cols) + "</div>"
    return f'<div class="rgw"><div class="rg" style="--n:{n}">{head}{body}</div></div>'

# --- the three result figures ---
master = grid(
    cols=[("baseline","baseline"),("t1","plan1 hygiene"),("t2ctrl","plan2 ctrl"),
          ("t2tt","plan2 t-travel"),("t4","plan4 bridge")],
    rows=[(r,r) for r in ROIS],
    tpl=lambda c,r: {
        "baseline":f"baseline/base_{r}_seed0_0.png","t1":f"test1/t1_{r}_seed0_0.png",
        "t2ctrl":f"test2/t2ctrl_{r}_seed0_0.png","t2tt":f"test2/t2tt_{r}_seed0_0.png",
        "t4":f"test4/t4_{r}_seed0_0.png"}[c])

# ---- selectivity heatmaps (from eval_metrics output) ----
import json as _json
SELJSON = ROOT / "brainguide/experiments/pulled/eval/selectivity.json"
HELDJSON = ROOT / "brainguide/experiments/pulled/eval/selectivity_heldout.json"
def sel_heatmaps(path):
    if not path.exists():
        return "<p class='rcap'>(matrix not found)</p>"
    d = _json.loads(path.read_text()); rois = d["rois"]
    order = ["baseline","plan1","plan2ctrl","plan2tt","plan4","readout","ddpo","repa"]
    lbl = {"baseline":"baseline","plan1":"plan 1","plan2ctrl":"plan 2 ctrl",
           "plan2tt":"plan 2 t-travel","plan4":"plan 4","readout":"readout (3)","ddpo":"DDPO (1)","repa":"REPA (6)"}
    def col(z):
        z = max(-2.0, min(2.0, z))
        if z >= 0: return f"rgba(46,158,143,{z/2*0.75:.2f})"
        return f"rgba(201,86,63,{-z/2*0.55:.2f})"
    # ranking bar
    rows = sorted(((m, d["methods"][m]) for m in order if m in d["methods"]),
                  key=lambda kv: -kv[1]["selectivity_contrast"])
    rank = '<table class="seltab"><tr><th>method</th><th>diag-argmax</th><th>contrast (z)</th></tr>'
    for m, v in rows:
        rank += (f'<tr><td>{lbl[m]}</td><td>{v["diag_argmax_hits"]}/{len(rois)}</td>'
                 f'<td>{v["selectivity_contrast"]:+.2f}</td></tr>')
    rank += "</table>"
    # per-method mini heatmaps
    hm = '<div class="hmwrap">'
    for m in order:
        if m not in d["methods"]: continue
        z = d["methods"][m]["z"]
        t = f'<div class="hm"><div class="hmt">{lbl[m]}</div><table class="hmtab">'
        t += '<tr><td class="hc"></td>' + "".join(f'<td class="hc">{r[:4]}</td>' for r in rois) + '</tr>'
        for i, ri in enumerate(rois):
            t += f'<tr><td class="hc">{ri[:4]}</td>'
            for j in range(len(rois)):
                b = col(z[i][j]); edge = "outline:2px solid var(--accent);" if i==j else ""
                t += f'<td style="background:{b};{edge}">{z[i][j]:+.1f}</td>'
            t += "</tr>"
        hm += t + "</table></div>"
    hm += "</div>"
    return rank + hm

def compare_table():
    if not (SELJSON.exists() and HELDJSON.exists()):
        return ""
    o = _json.loads(SELJSON.read_text())["methods"]
    h = _json.loads(HELDJSON.read_text())["methods"]
    lbl = {"baseline":"baseline","plan1":"plan 1 hygiene","plan2ctrl":"plan 2 ctrl",
           "plan2tt":"plan 2 t-travel","plan4":"plan 4 bridge","readout":"readout-head (3)","ddpo":"DDPO (1)","repa":"REPA-naive (6)"}
    order = ["baseline","plan1","plan2ctrl","plan2tt","plan4","readout","ddpo","repa"]
    t = ('<table class="seltab"><tr><th>method</th><th>our encoder</th><th>held-out</th>'
         '<th>drop = hacking</th></tr>')
    for m in order:
        if m not in o or m not in h: continue
        oc = o[m]["selectivity_contrast"]; hc = h[m]["selectivity_contrast"]; dr = oc - hc
        flag = " ✓ honest" if abs(dr) <= 0.1 else (" ⚠ inflated" if dr > 0.25 else "")
        t += (f'<tr><td>{lbl[m]}</td><td>{oc:+.2f}</td><td>{hc:+.2f}</td>'
              f'<td>{dr:+.2f}{flag}</td></tr>')
    return t + "</table>"

def summary_table():
    if not HELDJSON.exists():
        return ""
    h = _json.loads(HELDJSON.read_text())["methods"]
    # (key, label, family, one-line mechanism, verdict)
    rows = [
        ("baseline", "baseline", "—", "raw encoder gradient into the latent every step",
         "the low-α failure; floor, partly hacked"),
        ("plan1", "plan 1 · hygiene", "R", "crop-averaged gradient + TV/range regularizers",
         "honest &amp; selective, but flattens to mush"),
        ("plan2tt", "plan 2 · time-travel", "R", "re-noise/re-denoise so the prior re-absorbs kicks",
         "sharper; still partly encoder-hacked"),
        ("plan4", "plan 4 · bridge+IP-Adapter", "C", "score → top-K DINO centroid → CLIP → condition SD",
         "now 4/4 &amp; honest (held-out +1.69); best realism"),
        ("readout", "readout-head (opt 3)", "R", "head on SD's own UNet features; guide via ∇head",
         "gradient guidance that works with your encoder — 4/4, honest"),
        ("ddpo", "DDPO (opt 1)", "S", "RL: encoder as reward, PPO-update a LoRA",
         "underperformed at demo scale (sample-hungry)"),
        ("repa", "REPA-naive (opt 6)", "R/G", "pixel-gradient guidance on a DINOv2-aligned SiT",
         "high on our encoder but inflated (+0.38) &amp; texture-y — naive guidance hacks it"),
    ]
    t = ('<table class="seltab"><tr><th>method</th><th>family</th><th>how it beats the mismatch</th>'
         '<th>held-out sel.</th><th>verdict</th></tr>')
    for k, lab, fam, mech, verd in rows:
        hc = h.get(k, {}).get("selectivity_contrast"); hh = h.get(k, {}).get("diag_argmax_hits")
        sel = f'{hh}/{len(ROIS)}, {hc:+.2f}' if hc is not None else '—'
        t += (f'<tr><td><b>{lab}</b></td><td class="k">{fam}</td><td>{mech}</td>'
              f'<td>{sel}</td><td>{verd}</td></tr>')
    return t + ('</table><p class="rcap">Family: <b>S</b>=score-only (α irrelevant) · '
                '<b>R</b>=realign gradient (raise α) · <b>C</b>=condition natively. '
                'Held-out sel. = diagonal-argmax hits &amp; contrast on the independent BrainDiVE-CLIP encoder.</p>')

SEL_HTML = sel_heatmaps(SELJSON)
HELDOUT_HTML = sel_heatmaps(HELDJSON)
COMPARE_HTML = compare_table()
SUMMARY_HTML = summary_table()

SEEDCOLS = [(f"s{s}", f"seed {s}") for s in range(5)]
ROIROWS = [(r, r) for r in ROIS]
plan4_samples = grid(SEEDCOLS, ROIROWS, lambda c, r: f"test4/t4_{r}_seed{c[1:]}_0.png")
plan1_samples = grid(SEEDCOLS, ROIROWS, lambda c, r: f"test1/t1_{r}_seed{c[1:]}_0.png")
plan2_samples = grid(SEEDCOLS, ROIROWS, lambda c, r: f"test2/t2tt_{r}_seed{c[1:]}_0.png")
base_samples  = grid(SEEDCOLS, ROIROWS, lambda c, r: f"baseline/base_{r}_seed{c[1:]}_0.png")
readout_samples = grid(SEEDCOLS, ROIROWS, lambda c, r: f"test3/t3_{r}_seed{c[1:]}_0.png")
ddpo_samples = grid(SEEDCOLS, ROIROWS, lambda c, r: f"test5/t5_{r}_seed{c[1:]}_0.png")
repa_samples = grid(SEEDCOLS, ROIROWS, lambda c, r: f"test6/t6_{r}_seed{c[1:]}_0.png")
def _abs_datauri(p):
    import base64 as _b
    q=Path(p)
    return ("data:image/png;base64,"+_b.b64encode(q.read_bytes()).decode()) if q.exists() else ""
DDPO_REWARD_IMG = _abs_datauri(SCR / "ddpo_reward.png")

caveat = grid(
    cols=[("baseline","baseline"),("t4","plan4")],
    rows=[("faces","faces"),("bodies","bodies")],
    tpl=lambda c,r: (f"baseline/sub1_{r}_seed0_0.png" if c=="baseline"
                     else f"test4/t4_{r}_seed0_0.png"))

# the controlled proof: SAME SD gradient guidance, CLIP encoder vs DINOv2 encoder
clipdino = grid(
    cols=[("clip","CLIP encoder (orig. BrainDiVE)"),("dino","DINOv2 encoder (our baseline)")],
    rows=[(r, r) for r in ["faces", "places", "words", "bodies"]],
    tpl=lambda c,r: (f"braindive_clip/{r}_seed0.png" if c=="clip"
                     else f"baseline/base_{r}_seed0_0.png"))

RESULTS = f'''
<div class="part" id="results">
  <h2>Part VII</h2>
  <h3>Results — what actually came out</h3>
  <p class="lede">Universal (DINOv2) encoder · 4 semantic ROIs (earlyvisual dropped) · SD&nbsp;1.5 · L4 GPU. This is the empirical
  test of everything in Parts I–VI: does cleaning the gradient (plans&nbsp;1–2) or abandoning it
  (plan&nbsp;4) beat the raw baseline?</p>

  <h4 class="rh">The seven methods we tried — at a glance</h4>
  {SUMMARY_HTML}

  <div class="cbox"><h4>Is the "α / off-manifold" story actually established? (yes) — and a key FAQ</h4>
  <p><b>It's a known phenomenon, not a hunch.</b> Three established results underpin it: (1) the gradient
  of any network points <em>off</em> the data manifold — the mechanism of adversarial examples (Szegedy
  et al. 2013; Goodfellow et al. 2015); (2) only adversarially-<em>robust</em> encoders have on-manifold
  ("perceptually-aligned") gradients (Santurkar et al. 2019; Kawar, Ganz &amp; Elad, <em>Robust Classifier
  Guidance</em>, TMLR 2022); (3) CLIP-gradient guidance specifically is provably adversarial-prone
  (Shen et al., NeurIPS 2024) — which is why the Crowson/Disco-Diffusion practice adds cutouts + TV. That
  SD "speaks CLIP" is architectural fact (its cross-attention is conditioned on CLIP; Rombach et al. 2022),
  and REPA (Yu et al., ICLR 2025) is direct evidence that <em>aligning</em> a generator to DINOv2 fixes it.
  <b>Honest caveat — assumed vs. measured:</b> we did <em>not</em> numerically measure α (projecting a
  gradient onto the manifold's tangent plane is hard), so <b>α is an explanatory model, not a measured
  number</b> — the tangent/normal split is standard geometry, but the ratio "α" is this guide's framing.
  What we <em>did</em> measure is the outcome it predicts. So the CLIP-vs-DINOv2 quality gap is a
  <b>measured fact</b>; "it's because DINOv2's gradient is more off-manifold" is the <b>best-supported
  explanation</b> (matching the adversarial-gradient literature and REPA's premise), not a proven theorem.
  Our results bear it out: the raw DINOv2 gradient made texture, on-manifold/score-only methods made
  photos, and the held-out encoder exposed the gradient methods as partly reward-hacking.</p>
  <p><b>FAQ — "BrainDiVE used the gradient; why can't I?"</b> You can — our <em>baseline</em> is exactly
  that. Gradient guidance isn't forbidden; its quality depends on the encoder↔generator match.
  <b>BrainDiVE's encoder is CLIP — the same language SD obeys</b> — so its gradient is on-manifold (high α)
  and looks good. <b>Yours is DINOv2</b>, a space SD never learned, so the <em>same</em> method gives
  texture (low α). Everything here is about making the DINOv2 gradient usable: clean it (plans 1–2), route
  it through SD's own features (readout-head), or use a DINOv2-aligned generator (REPA).</p></div>

  <h4 class="rh">The proof, controlled: same guidance, swap only the encoder</h4>
  <p class="rcap">Both columns use identical SD gradient guidance on subject 1's ROIs. Left: the original
  BrainDiVE <b>CLIP</b> encoder (reproduced here). Right: our <b>DINOv2</b> universal encoder (baseline).
  Only the encoder differs — and CLIP is coherent while DINOv2 collapses to texture. That is the α claim,
  demonstrated, not asserted.</p>
  {clipdino}

  <h4 class="rh">Every method, one sample per ROI (seed 0)</h4>
  {master}
  <p class="rcap">Reading the rows through the \\(\\alpha\\) lens: <b>baseline</b> is the low-\\(\\alpha\\)
  failure (washed / texture-leaning). <b>plan&nbsp;1</b> calms texture but flattens — hygiene raised
  \\(\\alpha\\) yet hit the ceiling. <b>plan&nbsp;2</b> is sharper (prior re-projection) but still shows
  the DINOv2 off-manifold signature. <b>plan&nbsp;4</b> is the only unambiguously photographic column —
  it never took the gradient, so \\(\\alpha\\) is irrelevant.</p>

  <p class="rcap">Below: five samples per ROI for each plan (rows = ROIs, columns = seeds 0–4), so you
  can judge the <em>spread</em>, not one lucky draw. 100 images total; all runs complete.</p>

  <h4 class="rh">Baseline — raw gradient (5 samples/ROI)</h4>
  {base_samples}
  <p class="rcap">Low-\\(\\alpha\\) failure mode: washed, texture-leaning, content barely legible across
  every seed — the problem all three plans are trying to beat.</p>

  <h4 class="rh">Plan 1 — gradient hygiene (5 samples/ROI)</h4>
  {plan1_samples}
  <p class="rcap">Crop-averaging + TV/range kills the per-pixel adversarial half, so texture calms — but
  it flattens toward the painterly ceiling that \\(\\alpha\\) imposes on any gradient method.</p>

  <h4 class="rh">Plan 2 — time-travel (5 samples/ROI)</h4>
  {plan2_samples}
  <p class="rcap">Re-noise/re-denoise re-projects onto the manifold each pass — sharper and more
  structured than the baseline, yet the DINOv2 off-manifold signature still shows.</p>

  <h4 class="rh">Plan 4 — bridge + IP-Adapter (5 samples/ROI)</h4>
  <p class="rcap">The bridge target is fixed per ROI (a top-K centroid); the five seeds vary only the
  diffusion noise. The only unambiguously photographic column — it never took the gradient, so
  \\(\\alpha\\) is irrelevant.</p>
  {plan4_samples}

  <h4 class="rh">Readout-head (option 3) — 5 samples/ROI</h4>
  <p class="rcap">The Part-VIII plan, now run: guide on a small head trained on SD's <em>own</em> UNet
  features (encoder used only for labels). More photographic than the raw-gradient baseline (α raised —
  real faces and scenes, not static), while staying ROI-specific. Not as pristine as plan 4, but far more
  selective — see the numbers below.</p>
  {readout_samples}

  <h4 class="rh">DDPO (option 1) — RL reward fine-tune (5 samples/ROI)</h4>
  <p class="rcap">The other Part-VIII plan, run as a <b>compact demo</b>: 30 rounds × 8 samples/ROI, a
  1.2M-param LoRA on SD, the DINOv2 encoder used only as a scalar reward. Reward climbed on some ROIs
  (words +0.42, faces +0.11) but fell on others (places, bodies) — net noisy across ROIs.</p>
  <img class="rewardfig" src="{DDPO_REWARD_IMG}" alt="DDPO reward vs round">
  {ddpo_samples}
  <div class="cbox"><h4>Honest read — DDPO underperformed here, and why</h4>
  <p>Its selectivity landed at just <b>2/4 (contrast +0.19 our encoder, +0.27 held-out)</b> — the weakest
  of every method. This isn't a flaw in the method but in its <b>budget</b>: DDPO is <em>sample-hungry</em>
  (real runs use hundreds of rounds / thousands of images), and 240 samples cannot move a 1.2M-param LoRA
  decisively. The survey warned of exactly this. What the demo <em>does</em> prove is that the machinery
  runs end-to-end — sample → score with the frozen encoder → policy-gradient (REINFORCE + PPO clip) →
  update the LoRA → generate. Realising option 1's promise needs a full-scale run.</p></div>

  <h4 class="rh">Why we dropped earlyvisual — and plan 4's one remaining looseness</h4>
  <p>Plan 4 builds each ROI's target by averaging the top-K liked embeddings into one centroid,
  \\( \\bar e_{{\\text{{ROI}}}} = \\tfrac1K\\sum_k e_k \\), then rendering it. A centroid captures the
  region's central tendency but blurs anything the top-K don't share. Its clearest failure was
  <b>earlyvisual</b> — a <em>texture</em> region, not a semantic category — which the centroid rendered as
  a generic object. Since earlyvisual isn't a "concept" the way faces/places/words/bodies are, we drop it
  and evaluate on the <b>4 semantic ROIs</b>. That single change lifts plan 4 from 4/5 to <b>4/4</b>. The
  one looseness that remains is <b>faces↔bodies</b>: plan 4's "faces" are often whole people.</p>
  {caveat}
  <p class="rcap">The images alone can't rank selectivity vs. realism, so we scored every image two ways —
  on our own encoder and on an independent held-out encoder — below.</p>

  <h4 class="rh">Selectivity — the numbers (our encoder)</h4>
  <p>Eyeballing can't separate "looks real" from "drives the right region." So we <em>measure</em>: show
  every generated image to the frozen encoder and ask, for each, "how hard would each of the four
  regions fire?" Arrange the answers in a grid — one <b>row per region we generated for</b>, one
  <b>column per region we scored on</b>.</p>
  <ul class="gl">
    <li><b>Why z-scores?</b> Raw firing numbers aren't comparable across regions (each has its own
    baseline and scale). So we convert each column to a <b>z-score</b>: how many standard deviations above
    the <em>typical</em> image this one drives that region. \(z=+1.5\) means "1.5 SD harder than average";
    \(z=0\) is average; negative is below average.</li>
    <li><b>How to read the heatmap.</b> The <b>diagonal</b> cell of each row (outlined) is "did the
    faces-images drive the <em>faces</em> region?" A selective method has the diagonal as the reddest
    (highest-z) cell in its row — the image drives its own region and <em>no other</em>.</li>
    <li><b>Contrast</b> = mean(diagonal) − mean(off-diagonal), in z. Bigger = cleaner selectivity.
    <b>Diag-argmax</b> = how many of the 4 rows have their diagonal as the row-max (4/4 is perfect).</li>
  </ul>
  <p class="anlg"><b>Worked example — plan 4's "faces" row.</b> faces is the top (correct) cell, with
  <em>bodies</em> close behind — plan 4's face images fire the bodies region almost as hard, because they
  are whole people, not face close-ups. On the 4 semantic ROIs plan 4 gets all four diagonals right
  (4/4).</p>
  {SEL_HTML}
  <div class="cbox"><h4>Read this carefully — it is the <em>biased</em> view</h4>
  <p>On <em>our own</em> encoder the gradient methods score highest (plan 2 t-travel +1.89) and the
  score-only methods lower (plan 4 +1.53, readout +1.58). But the gradient methods <em>optimize this exact
  encoder's gradient</em> — so their high scores are partly circular (reward hacking), not proof of better
  content. The decisive, unbiased test is the <b>held-out encoder</b> — next.</p></div>

  <h4 class="rh">The honest judge — held-out encoder (independent BrainDiVE-CLIP)</h4>
  <p>We re-scored all 125 images with a <em>completely different</em> encoder — the reproduced original
  BrainDiVE model (OpenCLIP ViT-B/16 + linear read-out, subject 1, exact 4657/22961 voxel match). No
  method optimized <em>this</em> one, so it can't be gamed. The tell is the <b>drop</b> = our-encoder
  contrast − held-out contrast: a big drop means the method's selectivity was inflated by hacking our
  encoder; near-zero means it found genuinely ROI-driving content.</p>
  {COMPARE_HTML}
  <div class="cbox"><h4>What the held-out test reveals (4 semantic ROIs)</h4>
  <p><b>The gradient baselines partly evaporate.</b> plan 2 ctrl drops the most (+1.81→+1.38, −0.43),
  baseline (−0.28) and plan 2 time-travel (−0.19) also fall — their scores on our own encoder were partly
  circular, as predicted.</p>
  <p><b>The score-only / on-manifold methods are honest and now on top.</b> plan 4 <em>rises</em> on the
  independent judge (+1.53→<b>+1.69</b>, drop −0.16) and is now <b>4/4</b> once earlyvisual is removed;
  readout-head is +1.58→<b>+1.64</b> (−0.06), also 4/4. Because neither hacked our encoder, there was
  nothing to lose on the held-out one.</p>
  <p><b>For your goal — gradient guidance with the universal encoder — readout-head (option 3) is the
  built answer.</b> It <em>is</em> gradient guidance driven by your DINOv2 encoder (through a head on SD's
  own features), and it lands 4/4, honest (−0.06), and more photographic than the raw-gradient baselines.
  The higher-ceiling version is REPA (Part IX): a generator whose features are DINOv2-aligned, so you can
  differentiate your encoder <em>directly</em> and stay on-manifold.</p>
  <p><b>Bottom line (7 methods, 4 semantic ROIs).</b> On the held-out judge: plan 2 t-travel +1.70 ≈
  <b>plan 4 +1.69</b> > plan 1 +1.66 > <b>readout +1.64</b> > baseline +1.51 > plan 2 ctrl +1.38 > DDPO
  +0.27. But weighting <em>honesty + realism</em>, the winners are <b>plan 4</b> (best realism, honest,
  4/4) and <b>readout-head</b> (gradient-guided, honest, 4/4). The gradient baselines look strong only on
  their own encoder; DDPO underperformed at demo scale (sample-hungry, not broken). Practical rec:
  <b>readout-head for gradient guidance today; REPA for the higher ceiling; a full-scale DDPO
  as the higher-ceiling bet.</b></p></div>
  <div class="hmwrap-label rcap">Held-out selectivity matrices (z, diagonal accented):</div>
  {HELDOUT_HTML}

  <h4 class="rh">Footnote — unCLIP → IP-Adapter</h4>
  <p>The survey's option&nbsp;4 said “bridge + IP-Adapter / <b>unCLIP</b>.” Stable&nbsp;unCLIP is
  <b>gated</b> on Hugging&nbsp;Face (401 anonymous), so plan&nbsp;4 uses the public equivalent —
  SD&nbsp;1.5&nbsp;+&nbsp;IP-Adapter — which consumes the same CLIP-ViT-H image embedding the bridge
  produces. Same idea, reachable weights.</p>
</div>
'''

NEXTSTEPS = r'''
<div class="part" id="next">
  <h2>Part VIII</h2>
  <h3>Two more methods we built (options 3 &amp; 1) — raising \(\alpha\) without touching the encoder</h3>
  <p class="lede">Plan 4 wins on realism but its ROI-selectivity is loose (whole people instead of face
  close-ups). So we built two more methods aimed
  at being <em>both</em> photographic <em>and</em> sharply ROI-specific — under one hard rule:
  <span class="frozen">the brain encoder's weights are never touched.</span> Both are now implemented,
  run, and scored — this section explains the math and intuition; their results are in Part&nbsp;VII.</p>

  <div class="cbox">
    <h4>The constraint, and what it rules out</h4>
    <p>Recall §I: the whole problem is the alignment number \(\alpha\) — the fraction of the brain
    gradient that lies <em>on</em> the image manifold. There are only three escapes: raise \(\alpha\),
    make \(\alpha\) irrelevant, or retrain the generator. Keeping the encoder frozen rules out
    <b>option&nbsp;2 (robustify the encoder)</b>, which would adversarially fine-tune its read-out. That
    leaves two frozen-encoder routes, both of which we will build: <b>readout-head guidance</b> (raise
    \(\alpha\)) and <b>DDPO</b> (make \(\alpha\) irrelevant).</p>
  </div>

  <h4 class="rh">Step A — first, <em>measure</em> (you can't improve what you can't score)</h4>
  <p>Judging by eye conflates realism with selectivity. We score every generated image with the frozen
  encoder and build a <b>selectivity matrix</b>: entry \(M_{ij}\) is the mean predicted activation, over
  the images generated <em>for</em> ROI&nbsp;\(i\), measured <em>on</em> ROI&nbsp;\(j\)'s voxels.</p>
  \[ M_{ij}=\operatorname*{mean}_{x\ \in\ \text{images for ROI } i}\ a_{\text{ROI } j}(x),
     \qquad\text{selective}\ \Longleftrightarrow\ \arg\max_j M_{ij}=i\ \ \text{(strong diagonal).} \]
  <p><b>Intuition:</b> a good method makes ROI-\(i\) images that fire ROI&nbsp;\(i\) <em>and no other</em>
  — so the matrix's diagonal should dominate each row. This is where plan&nbsp;4's averaging would show a
  cost. Crucially we <em>also</em> re-score with a <b>held-out, independent encoder</b> — the reproduced
  original BrainDiVE CLIP model — because plans&nbsp;1&nbsp;&amp;&nbsp;2 optimize <em>our</em> encoder's
  gradient, so scoring them on our own encoder is circular (reward hacking); an independent judge is the
  honest test.</p>

  <h4 class="rh">Step B — Readout-head guidance (option 3): raise \(\alpha\) by changing <em>what we differentiate</em></h4>
  <p class="one">In one sentence</p>
  <p>Instead of differentiating the foreign DINOv2 encoder, we teach a tiny helper to predict the brain
  score from Stable Diffusion's <em>own</em> internal features — and we guide by differentiating that
  helper instead. Because the helper reads the generator's native language, its gradient can't push
  off-manifold.</p>

  <p class="one">Why this should work</p>
  <p>Go back to §I: the brain gradient is only partly renderable because it is born in DINOv2's feature
  space, which the UNet doesn't share (low \(\alpha\)). But the UNet's <em>own</em> hidden activations at
  a denoising step — call them \(\phi_t(x_t)\) — are by definition directions the generator can move
  along cleanly: nudging the latent to change \(\phi_t\) is exactly the kind of change SD renders as a
  real image. So if a small head can predict "how hard would the ROI fire?" <em>from those internal
  activations</em>, then differentiating the head sends us along on-manifold directions.</p>

  <p class="anlg"><b>Analogy.</b> Last time (§I) you phoned a painter (the decoder) to translate a wish
  written in a foreign language (DINOv2) — and got a garbled result. Here you instead <b>hire a native
  speaker</b>: a small head that reads the generator's own internal language and tells you which way to
  nudge. Same goal, but now every instruction is one the painter can actually execute.</p>

  <p class="one">The two phases</p>
  <ul class="gl">
    <li><b>Offline — teach the head.</b> Take a big pile of images; run the frozen DINOv2 brain-encoder
    once to get each one's ROI score \(a_V(x)\) (this is the <em>only</em> place DINOv2 is used — as a
    label maker). Also run SD's noising process and grab its internal features \(\phi_t\) at many noise
    levels \(t\). Fit a small head \(h_\psi\) (weights \(\psi\)) to predict the score from the features.</li>
    <li><b>Online — guide with the head.</b> During generation, at each step read the UNet's features
    \(\phi_t(z_t)\), pass them through \(h_\psi\), and step the latent \(z_t\) up the head's gradient.</li>
  </ul>
  \[ \underbrace{h_\psi\!\big(\phi_t(x_t),\,t\big)\approx a_V(x)}_{\text{train offline (UNet + encoder frozen)}}
     \qquad\Longrightarrow\qquad
     \underbrace{g_t=\nabla_{z_t}\,h_\psi\!\big(\phi_t(z_t),\,t\big)}_{\text{guide online}}. \]

  <p class="one">What each symbol means</p>
  <ul class="gl">
    <li>\(x_t,\,z_t\) — the (noisy) image / its latent at denoising step \(t\).</li>
    <li>\(\phi_t(\cdot)\) — SD's own hidden activations at step \(t\) (the mid-block + a couple of
    up-blocks are the standard picks). The "native language."</li>
    <li>\(h_\psi\) — the small trained head; \(\psi\) its weights. The "native speaker."</li>
    <li>\(a_V(x)\) — the frozen brain-encoder's ROI score; used offline only, as a training label.</li>
    <li>\(g_t\) — the guidance nudge added to the latent this step.</li>
  </ul>
  <p><b>Why \(\alpha\!\to\!1\):</b> the gradient now flows <em>only</em> through features SD can render,
  so it lands almost entirely in the honest, on-manifold half \(g_\parallel\). <span class="frozen">The
  encoder is frozen (labels only) and the UNet is frozen (we read it and train a small head)</span> — and
  it reuses our existing guided sampler almost verbatim: we just swap "differentiate the encoder" for
  "differentiate the head."</p>
  <p class="rcap"><b>Result (Part VII):</b> readout-head scored held-out <b>4/4, +1.64, drop −0.06</b> —
  honest gradient guidance driven by your universal encoder; tied with plan 4 as the study's best balance.</p>

  <h4 class="rh">Step C — DDPO (option 1): make \(\alpha\) <em>irrelevant</em> by never taking the gradient at all</h4>
  <p class="one">In one sentence</p>
  <p>Stop asking the encoder "which way should I nudge the pixels?" (a gradient). Instead just let SD
  make a whole image, ask the encoder "how good was that?" (a single number), and use trial-and-error to
  make SD produce high-scoring images more often. No encoder gradient is ever computed, so its
  off-manifold geometry can't hurt us.</p>

  <p class="one">The reframe: generation as trial-and-error</p>
  <p>Think of making an image as a sequence of small moves (the ~50 denoising steps). After the last
  move you have a finished image \(x_0\), and only then do you collect a <b>reward</b> — the brain score:</p>
  \[ R(x_0)=a_V(x_0)=\frac{1}{|V|}\sum_{v\in V} f_v(x_0)\quad\text{(mean predicted firing over the ROI's voxels).} \]
  <p class="anlg"><b>Analogy.</b> It's training a dog with treats. You don't grab its legs and move them
  (that's the gradient — invasive, and it fights you). You let it do a whole trick, then give a treat if
  you liked it. Do that many times and the behaviours that earned treats become more frequent — without
  ever micromanaging a single motion.</p>

  <p class="one">The update rule, unpacked</p>
  <p>Generate a batch of images, score them, and nudge a small set of trainable weights \(\theta\) (a
  <b>LoRA</b> adapter bolted onto SD) so the moves that led to <em>above-average</em> images become more
  likely:</p>
  \[ \nabla_\theta J=\mathbb{E}\!\Big[\textstyle\sum_t\big(R(x_0)-b\big)\,\nabla_\theta\log
     p_\theta(x_{t-1}\mid x_t)\Big]\;-\;\beta\,\mathrm{KL}\!\big(p_\theta\,\Vert\,p_\text{base}\big). \]
  <ul class="gl">
    <li>\(\theta\) — the LoRA weights we train (rank ~8 on the UNet's attention). <span class="frozen">SD's
    backbone and the encoder stay frozen.</span></li>
    <li>\(\nabla_\theta\log p_\theta(x_{t-1}\mid x_t)\) — "how do I make the model more likely to take the
    move it just took?" Multiply that by the reward and you <em>reinforce</em> good moves.</li>
    <li>\(b\) — a <b>baseline</b>, the batch-average reward. Subtracting it means "reward relative to
    typical," i.e. <b>graded on a curve</b>: above-average images get pushed up, below-average down. It's
    the single trick that makes training stable.</li>
    <li>\(\beta\,\mathrm{KL}(p_\theta\Vert p_\text{base})\) — a <b>leash</b> tying the fine-tuned model to
    the original SD. This is your <b>realism dial</b>: turn \(\beta\) up the moment images start looking
    "off," and it can't wander off the natural-image manifold to chase reward.</li>
  </ul>

  <p><b>Why \(\alpha\) disappears entirely:</b> the encoder shows up only as the plain number \(R\) — it
  is <em>never differentiated</em>. So its adversarial directions, its low \(\alpha\), its whole geometry
  are simply <em>gone</em> (not dampened like plans 1–2 — gone), because the thing being optimized is SD
  itself, which already knows what real images look like. Optimizing the score <em>directly</em> while
  leashed to realism is why DDPO is the one route that can plausibly win on realism <em>and</em>
  selectivity at once — the survey's top pick.</p>
  <p class="anlg"><b>The catch.</b> Trial-and-error is sample-hungry (many generations per round), and the
  dog may learn to <em>cheat</em> — find weird images that spike our encoder without being meaningful
  (reward hacking). That is exactly why Step&nbsp;A's <b>held-out encoder</b> is the guardrail: if the
  independent judge disagrees, the reward was hacked.</p>
  <p class="rcap"><b>Result (Part VII):</b> at demo scale (30 rounds × 8 = 240 samples) DDPO reached only
  <b>2/4 our, +0.27 held-out</b> — sample-starved, not broken. Reward moved on some ROIs; a full-scale run
  (hundreds of rounds) is what realizes option 1's ceiling.</p>

  <div class="cbox">
    <h4>Why this order</h4>
    <p><b>Measure → readout-head → DDPO.</b> The scoring harness comes first because it is <em>also</em>
    the reward for DDPO and the label source for the readout head — and it is how we will know whether
    any of this actually beat plan&nbsp;4. Readout-head is next because it reuses our sampler and is
    plain guidance (fast, low-risk). DDPO is last: the biggest lift and the most compute, taken on only
    if we still need more selectivity than readout-head delivers.</p>
  </div>
</div>
'''

NEXTGEN = r'''
<div class="part" id="bettergen">
  <h2>Part IX</h2>
  <h3>Using a better generator — the real ceiling-lift</h3>
  <p class="lede">Look at the photos: baseline is static, plan 1 is mush, plan 2 is distorted, DDPO
  (at demo scale) is weak. Only plan 4 (and, more selectively, readout-head) produced real images. The
  common cause is <b>not</b> the encoder — it's that <b>Stable Diffusion 1.5 only understands CLIP</b>.
  Every method so far worked <em>around</em> that. The bigger win is to change the generator itself.</p>

  <div class="cbox"><h4>Why "a better generator" is the true fix</h4>
  <p>Recall \(\alpha\) (§I): the fraction of the brain gradient that is renderable. It is a property of
  the <em>pairing</em> (encoder ↔ generator). We held the generator fixed (SD 1.5, a CLIP model) and
  fought a DINOv2 encoder, so \(\alpha\) was capped. Swap in a generator that <em>shares</em> your
  encoder's language — or is simply far stronger — and the ceiling moves.</p></div>

  <h4 class="rh">Lever A — the cheapest win: plan 4 on a stronger backbone (no training)</h4>
  <p><b>In one sentence:</b> keep the exact score-only recipe of plan 4, but replace SD 1.5 with a much
  better painter. Our plan 4 pipeline (rank images by ROI score → average the top-K → bridge to the
  generator's image-embedding space → condition) transfers directly to <b>SDXL</b>, <b>SD 3</b>, or
  <b>FLUX.1</b> via their IP-Adapters. Nothing about the encoder or the method changes — only the image
  quality of the base model. This is the single highest-return next step, and it needs <em>zero</em>
  training.</p>
  <p class="anlg"><b>Why it helps:</b> plan 4 already sidesteps \(\alpha\) (it never takes a gradient).
  Its only weakness was SD 1.5's modest fidelity and the coarse top-K average. A stronger backbone fixes
  the fidelity half for free; a finer bridge (fewer-K, or IP-Adapter-Plus) fixes the rest.</p>

  <h4 class="rh">Lever B — REPA: train a generator that natively speaks DINOv2 (option 6)</h4>
  <p class="who" style="font-family:var(--mono);font-size:12px;color:var(--faint)">Yu et al., “Representation
  Alignment for Generation,” ICLR 2025 (Oral).</p>
  <p><b>In one sentence:</b> while training a diffusion transformer, add a loss that forces its internal
  features to <em>match</em> a frozen DINOv2's features of the same image. The generator then thinks in
  DINOv2's language, so a DINOv2 brain-gradient is renderable by construction.</p>
  \[ \mathcal{L}=\mathcal{L}_{\text{diffusion}} \;+\; \lambda\underbrace{\big(1-\operatorname{sim}\!\big(
     P_\phi(h_t),\; f_{\text{DINO}}(x)\big)\big)}_{\text{align DiT hidden }h_t\text{ to DINOv2}} \]
  <ul class="gl">
    <li>\(h_t\) — a hidden layer of the diffusion transformer at step \(t\); \(P_\phi\) a small projector.</li>
    <li>\(f_{\text{DINO}}(x)\) — frozen DINOv2 features of the clean image. \(\operatorname{sim}\) = cosine.</li>
    <li>\(\lambda\) — alignment strength. (In the paper this also makes training much faster/better.)</li>
  </ul>
  <p><b>Why it's the principled cure:</b> after REPA, the generator's own representation <em>is</em>
  DINOv2-shaped — so your brain encoder (which reads DINOv2) points along directions the generator can
  render: \(\alpha\to1\). Cost: you must <b>train a DiT</b> (this is the G-family "retrain the generator"
  budget you'd ruled out for SD — but REPA is notably cheaper than training a DiT from scratch, and small
  DiT/SiT models are tractable).</p>

  <div class="cbox"><h4>We actually tested REPA — and it taught us something important</h4>
  <p>We took the public REPA <b>SiT-XL/2</b> checkpoint and guided it with your universal encoder
  (5 seeds × 4 ROIs). Result: on our encoder it scored the <em>highest</em> selectivity of any method
  (<b>4/4, +1.91</b>) — but held-out only <b>+1.53</b> (<b>drop +0.38 → inflated</b>), and the images are
  <b>texture-y</b>, not clean:</p>
  {repa_samples}
  <p><b>Why — and the real lesson.</b> Our guidance decoded the latent to pixels and took the encoder
  gradient <em>through the VAE</em> — the <em>same</em> pixel→latent path as SD. It never used REPA's
  DINOv2-<em>aligned internal features</em>. So it reward-hacked exactly like the SD gradient baselines.</p>
  <p><b>Would a different guidance strength fix it? No — and this matters.</b> We tried scale 0.3 (20
  steps) and 0.2 (50 steps); both texture. Scale changes the <em>magnitude</em> of the push, not its
  <em>direction</em> — it scales the on-manifold and off-manifold halves together, so α (the ratio) is
  unchanged (exactly the Part-I argument for why "just push harder" fails on SD). The fix has to change
  <em>where the gradient comes from</em>, not how hard you push.</p>
  <p><b>The fix — "a readout-head on the SiT":</b> apply option 3 to REPA. Train a tiny head on the
  <em>SiT's own internal features</em> — which REPA trained to <em>be</em> DINOv2 — to predict the ROI
  score, then guide by that head's gradient. Because those features already <em>are</em> DINOv2, the head
  reads your brain signal from <em>inside</em> the generator, so its gradient is on-manifold by
  construction: the aligned generator (REPA) <em>and</em> aligned guidance (readout-head) together. That
  is the concrete next experiment; our naive pixel-space port is an honest negative result that isolates
  exactly <em>why</em> a better generator alone isn't enough.</p></div>

  <h4 class="rh">Lever C — RCG: generate in representation space (option 5)</h4>
  <p class="who" style="font-family:var(--mono);font-size:12px;color:var(--faint)">Li, Katabi &amp; He,
  “Return of Unconditional Generation,” NeurIPS 2024.</p>
  <p><b>In one sentence:</b> split generation in two — first a small model samples a self-supervised
  <em>representation</em> \(z_{\text{rep}}\) (DINOv2-like), then a pixel generator renders an image
  conditioned on it. To drive an ROI, you optimize/search \(z_{\text{rep}}\) to maximize the brain score,
  then decode — all in the representation space your encoder already lives in.</p>
  \[ z_{\text{rep}}^\star=\arg\max_{z}\; a_V\big(G(z)\big),\qquad \text{image}=G(z_{\text{rep}}^\star) \]
  <p>It's plan 4's idea, but with a purpose-built representation→image generator instead of a bridge into
  a CLIP model — so the round-trip is faithful rather than approximate. Cost: use/train the RCG stack.</p>

  <table class="seltab">
    <tr><th>lever</th><th>touches</th><th>training</th><th>expected payoff</th></tr>
    <tr><td>A · plan 4 on SDXL/SD3/FLUX</td><td>base model only</td><td>none</td><td>big realism jump, today</td></tr>
    <tr><td>B · REPA DiT</td><td>new generator</td><td>train a DiT (align to DINOv2)</td><td>native DINOv2 guidance, α→1</td></tr>
    <tr><td>C · RCG</td><td>new generator</td><td>train/adopt RCG stack</td><td>faithful rep-space ROI search</td></tr>
  </table>
  <div class="cbox"><h4>Recommendation</h4>
  <p><b>Do Lever A first</b> — plan 4 (or readout-head) on a stronger backbone is the cheapest, fastest,
  no-training way to turn the "kind-of-works" photos into genuinely good ones. If you're willing to train
  a generator, <b>REPA</b> is the principled endgame: it makes your DINOv2 encoder a first-class steering
  wheel instead of a foreign one.</p></div>
</div>
'''

PAPERS = r'''
<div class="part" id="papers">
  <h2>Part X</h2>
  <h3>Related papers we already have — the generator × encoder landscape</h3>
  <p class="lede">Five papers in this project bear directly on "drive a brain ROI with a generator."
  They span every generator family — and two are from the <em>same lab</em> as your DINOv2 encoder.</p>
  <div class="gwrap"><table class="seltab" style="min-width:640px">
    <tr><th>paper</th><th>generator</th><th>encoder</th><th>approach</th><th>relation to your goal</th></tr>
    <tr><td><b>NeuroGen</b> (Gu et al.)</td><td>BigGAN</td><td>CNN (GNet)</td>
        <td>optimize the GAN latent/class to maximize predicted voxel activation</td>
        <td>the GAN-era ancestor of gradient guidance</td></tr>
    <tr><td><b>BrainDiVE</b> (Luo, NeurIPS'23)</td><td>Stable Diffusion</td><td>CLIP</td>
        <td>gradient-guide the sampler by the encoder's gradient</td>
        <td>the exact method you're extending (high α because CLIP↔CLIP)</td></tr>
    <tr><td><b>BrainExplore</b> (Weizmann/MIT)</td><td>— (score-based)</td><td>encoder + SAEs</td>
        <td>per unit: top natural images + a language description of their shared meaning</td>
        <td>encoder-as-<em>score</em> — like plan 4; no α problem</td></tr>
    <tr><td><b>From Activation to Causality</b> (BrainCause, Weizmann/MIT)</td><td>diffusion editing</td>
        <td>image→fMRI encoder</td><td>counterfactual edits that remove a concept to test <em>causation</em></td>
        <td>validation layer: proves a region <em>represents</em> a concept, not just correlates</td></tr>
    <tr><td><b>Wisdom of a Crowd of Brains</b> (Weizmann)</td><td>—</td><td>DINOv2 universal</td>
        <td>one encoder across many subjects on frozen DINOv2 features</td>
        <td><b>your encoder</b></td></tr>
  </table></div>
  <div class="cbox"><h4>The key point for your question</h4>
  <p>You asked for <b>gradient guidance you can use with your universal encoder</b>. The gradient-guidance
  lineage is <b>NeuroGen → BrainDiVE</b> (and our plans 1–3). BrainDiVE works because its CLIP encoder
  matches SD's CLIP conditioning (high α); yours is DINOv2, so the direct gradient is capped — which is
  why <b>readout-head</b> (route the gradient through SD's own features) and <b>REPA</b> (a generator whose
  features are DINOv2-aligned) are the two ways to keep gradient guidance <em>and</em> use your encoder.</p>
  <p><b>Also worth knowing:</b> <b>BrainExplore</b> and <b>BrainCause</b> come from the same lab as your
  encoder and deliberately <em>avoid</em> gradient guidance — score-based discovery and counterfactual
  editing. If the goal is discovery/validation rather than steering, those are the encoder-native routes
  that never hit the α wall at all.</p></div>
</div>
'''

PAPERS_EXPANDED = r'''
<div class="part" id="papers-expanded">
  <h2>Part XI</h2>
  <h3>Every paper, expanded — goal · how · result · what it means for you</h3>
  <p class="lede">The reference list from Part III, each entry filled out: what the paper was trying to
  do, what it did technically, what it found, and specifically how it bears on <b>your</b> case —
  generating images that drive ROI voxels from the universal DINOv2 encoder.</p>

  <p class="one">A · Brain-stimulus synthesis (your direct lineage)</p>

  <div class="pex"><h4>BrainDiVE — Luo et al., NeurIPS 2023 (Oral)</h4>
    <p><b>Goal:</b> synthesize images that maximally drive a chosen visual ROI, to <em>discover</em> what
    that region encodes.</p>
    <p><b>How:</b> linear readout on frozen OpenCLIP ViT-B/16 → per-voxel activation; each denoising step,
    decode the predicted-\(x_0\), backprop mean ROI activation into the latent; SAG 0.75, scale ≈130,
    empty prompt, SD 2.1.</p>
    <p><b>Result:</b> images with clear per-ROI signatures (faces/places/words/bodies) matching known
    selectivity, plus finer discoveries (e.g. food-selective cortex).</p>
    <p class="foryou"><b>For you:</b> this <em>is</em> your pipeline. It works because CLIP↔CLIP is high-α;
    swapping in DINOv2 is what broke it. Everything else here is about fixing that.</p></div>

  <div class="pex"><h4>BrainSCUBA — Luo et al., ICLR 2024</h4>
    <p><b>Goal:</b> describe what a voxel prefers in <em>words</em>, not pixels.</p>
    <p><b>How:</b> map voxel → a "soft" CLIP embedding, project onto the CLIP sphere, caption it with a
    CLIP-conditioned language model.</p>
    <p><b>Result:</b> fine-grained, interpretable per-region captions; semantic gradients across cortex.</p>
    <p class="foryou"><b>For you:</b> a ready-made concept-namer for your loop — after you generate an
    ROI image, this labels <em>what</em> it depicts. Proof the voxel→text direction works.</p></div>

  <div class="pex"><h4>Voxel-Weighted Activation Maximization — Shinkle &amp; Lescroart, CVPR 2025 wksp</h4>
    <p><b>Goal:</b> maximize voxel activation with <em>no</em> generator.</p>
    <p><b>How:</b> gradient-ascent directly in pixels on a voxel-weighted objective (DeepDream at cortex);
    no image prior.</p>
    <p><b>Result:</b> feature-visualization textures, not photos.</p>
    <p class="foryou"><b>For you:</b> the α-irrelevant-but-ugly baseline — shows exactly the off-manifold
    failure when nothing constrains the image to look real.</p></div>

  <div class="pex"><h4>NEvo — EPFL, 2026 (public HF pipeline)</h4>
    <p><b>Goal:</b> synthesize <em>dynamic</em> stimuli (videos, and images) that maximally drive a region.</p>
    <p><b>How:</b> evolutionary search over a structured prompt space; render with SDXL-Turbo / LTX-Video;
    score each candidate with a V-JEPA2-based encoder; iterate. The encoder is a <b>black-box score</b> —
    never differentiated. Reports BrainDiVE "fails to generate meaningful stimuli" in the dynamic setting.</p>
    <p><b>Result:</b> first model-based framework for dynamic visual selectivity.</p>
    <p class="foryou"><b>For you — usable, not just a principle:</b> it's a public <code>Diffusers</code>
    pipeline that <em>also does still images</em> and lets you <b>swap in your own encoder</b>. So it's
    essentially turnkey "search-based plan 4": plug your DINOv2 ROI encoder in as the scorer. Only
    adaptation: its named-ROI targeting expects fsaverage5 output, so you wrap your encoder's score.
    <a href="https://nevo-project.epfl.ch/" target="_blank" rel="noopener">project</a> ·
    <a href="https://arxiv.org/abs/2607.02317" target="_blank" rel="noopener">arXiv 2607.02317</a> ·
    <a href="https://huggingface.co/epfl-neuroai/NEvo" target="_blank" rel="noopener">HF</a></p></div>

  <div class="pex"><h4>BrainTRACE / “From Activation to Causality” — Golbari et al., 2026</h4>
    <p><b>Goal:</b> test whether a region <em>causally represents</em> a concept, not just correlates.</p>
    <p><b>How:</b> generate concept images (FLUX) + a VLM (Qwen-VL) confirms the concept is present;
    FLUX-edit a counterfactual that removes it, VLM confirms removal; score all through an image→fMRI
    encoder; keep only voxels passing the three-way test.</p>
    <p><b>Result:</b> false positives cut from ~70% to ~23%.</p>
    <p class="foryou"><b>For you:</b> validates the "generate → edit → check activation drop" logic. Your
    novelty is the open-ended, voxel-first, recursive-decomposition direction they don't automate. (This
    is one of the PDFs you have.)</p></div>

  <div class="pex"><h4>BrainCLIP — Liu et al., 2023</h4>
    <p><b>Goal:</b> map fMRI natively into CLIP space.</p>
    <p><b>How:</b> train an fMRI→CLIP encoder with visual + textual (contrastive) supervision.</p>
    <p class="foryou"><b>For you:</b> the "brain-CLIP" option — a readout that targets CLIP embeddings would
    be high-α (renderable) and plug straight into captioning / IP-Adapter. A different route than fixing
    the DINOv2 gradient.</p></div>

  <p class="one">B · The guidance machinery (the theory behind BrainDiVE)</p>

  <div class="pex"><h4>Classifier &amp; Classifier-Free Guidance — Dhariwal&amp;Nichol 2021; Ho&amp;Salimans 2022</h4>
    <p><b>Goal / How:</b> steer sampling with a classifier gradient (CG), or bake conditioning into
    training (CFG). <b>Result:</b> the foundation of all guided diffusion.</p>
    <p class="foryou"><b>For you:</b> BrainDiVE is literally CG with an ROI "classifier" — this is the root
    of your method (and where the "scale" knob comes from).</p></div>

  <div class="pex"><h4>DPS — Diffusion Posterior Sampling — Chung et al., ICLR 2023</h4>
    <p><b>Goal/How:</b> guide with <em>any</em> differentiable objective evaluated on the predicted-\(x_0\).</p>
    <p class="foryou"><b>For you:</b> the formal parent of BrainDiVE's step — your guidance is a special
    case of DPS.</p></div>

  <div class="pex"><h4>FreeDoM — Yu et al., ICCV 2023</h4>
    <p><b>Goal/How:</b> training-free energy guidance + the <b>time-travel / resampling</b> trick that
    re-noises and re-denoises to heal off-manifold damage. <b>Result:</b> cleaner training-free guidance.</p>
    <p class="foryou"><b>For you:</b> this is our <b>plan 2</b> — the biggest quality win you can bolt onto
    a gradient pipeline without training.</p></div>

  <div class="pex"><h4>MCG &amp; MPGD — manifold-constrained / -preserving guidance — Chung 2022; He 2024</h4>
    <p><b>Goal/How:</b> <em>name your exact problem</em> and fix it — project the guidance gradient onto
    the data manifold so it never pushes into "impossible image" directions (keep only \(g_\parallel\)).</p>
    <p class="foryou"><b>For you:</b> the principled cure for a low-α encoder <em>without training</em> —
    the "R" option we flagged but didn't build. A strong next experiment for keeping gradients.</p></div>

  <div class="pex"><h4>Universal Guidance — Bansal et al., 2023 · TFG — Ye et al., NeurIPS 2024</h4>
    <p><b>Goal/How:</b> UG adds inner optimization steps per denoising step with any predictor; TFG shows
    FreeDoM/UG/DPS are one framework with a few knobs. <b>Result / honest verdict:</b> TFG finds these
    methods "often lack theoretical grounding" — the gradient family is <em>finicky by nature</em>.</p>
    <p class="foryou"><b>For you:</b> confirms the texture/instability you saw isn't your bug — it's
    intrinsic to gradient guidance, which is why score-only routes (plan 4 / NEvo / DDPO) are attractive.</p></div>

  <p class="one">C · Fixing the mismatch by training (if you insist on gradients)</p>

  <div class="pex"><h4>Perceptually-aligned gradients / robust synthesis — Santurkar, Engstrom et al., 2019</h4>
    <p><b>Goal/How:</b> adversarially-robust models have gradients that look like real image edits;
    robustifying raises α. <b>Result:</b> robust classifiers can synthesize recognizable images by plain
    gradient ascent.</p>
    <p class="foryou"><b>For you:</b> a cheap fine-tune (LoRA/linear-probe + PGD, or a Jacobian-smoothness
    penalty) that <em>keeps DINOv2</em> but makes its gradient steerable — the "R" option 2. (You said no
    encoder-weight changes, so this is off-limits unless you relax that.)</p></div>

  <div class="pex"><h4>Textual Inversion &amp; IP-Adapter — Gal et al. 2022; Ye et al. 2023</h4>
    <p><b>Goal/How:</b> turn a target into <em>conditioning</em> instead of a gradient — learn a small map
    ROI → CLIP embedding and use SD's normal conditioning path (on-manifold by construction).</p>
    <p class="foryou"><b>For you:</b> this is exactly our <b>plan 4</b> (IP-Adapter). α stops mattering
    because you never take the gradient.</p></div>

  <div class="pex"><h4>Reward fine-tuning — DRaFT, DDPO, DPOK — 2023</h4>
    <p><b>Goal/How:</b> train SD to produce high-reward images. DRaFT backprops the reward through
    sampling; DDPO/DPOK use RL with the reward as a <em>black-box scalar</em> — no encoder gradient, so the
    α mismatch vanishes.</p>
    <p class="foryou"><b>For you:</b> our <b>DDPO</b> (option 1); the robust modern answer if you generate
    many images per ROI. Same insight as NEvo — encoder as a score, not a gradient.</p></div>
</div>
'''

FIXSTACK = r'''
<div class="part" id="fixstack">
  <h2>Part XII</h2>
  <h3>Making guidance better — the fix-stack, and exactly what we tried</h3>
  <p class="lede">If you keep gradient guidance, there's a standard toolkit for making it cleaner (the
  Crowson/Disco + FreeDoM lineage). Here is each tool — what it does, why it helps — mapped to <b>what we
  actually ran</b> and what happened.</p>
  <div class="gwrap"><table class="seltab" style="min-width:720px">
    <tr><th>technique</th><th>what it does</th><th>why it helps</th><th>we tried it?</th><th>result</th></tr>
    <tr><td><b>Cutouts</b> (aug-averaged gradient)</td><td>average the brain gradient over many random
      crops of \(\hat x_0\)</td><td>per-pixel adversarial directions disagree across crops and cancel;
      only coherent semantic signal survives</td><td class="ours">✔ plan 1 (8 cutouts)</td>
      <td>texture calmed, but flattened toward mush</td></tr>
    <tr><td><b>Guide on \(\hat x_0\)</b>, not \(x_t\)</td><td>score the Tweedie clean-image estimate, not
      the noisy latent</td><td>the encoder is only in-distribution on clean images</td>
      <td class="ours">✔ always (our cond_fn)</td><td>necessary; baseline behavior</td></tr>
    <tr><td><b>Noise-normalized step</b> (adaptive)</td><td>scale guidance to the model's own
      \(\lVert\varepsilon\rVert\) each step</td><td>keeps guidance commensurate at every t/ROI — the
      finicky-killer</td><td class="ours">✔ (the ROI-independent step)</td>
      <td><b>made images appear at all</b></td></tr>
    <tr><td><b>Semantic window</b></td><td>guide only in the middle of sampling, not early/late</td>
      <td>early = too noisy, late = only adds texture</td><td class="ours">✔ plan 2 window</td>
      <td>focuses the push where content is decided</td></tr>
    <tr><td><b>Time-travel / resampling</b></td><td>re-noise \(x_{t-1}\!\to\!t\) and re-denoise a few
      times</td><td>the diffusion prior re-absorbs off-manifold kicks each pass</td>
      <td class="ours">✔ plan 2 (DDIM)</td><td>sharper &amp; more structured than baseline, but still
      shows DINOv2 texture; partly hacked on held-out</td></tr>
    <tr><td><b>TV + range regularizers</b></td><td>penalize adjacent-pixel jumps &amp; out-of-[0,1]</td>
      <td>kills checkerboard texture and blown-out contrast</td><td class="ours">✔ plan 1</td>
      <td>reduces high-frequency artifacts</td></tr>
    <tr><td><b>Spherical-distance loss</b></td><td>great-circle loss between unit embeddings</td>
      <td>well-behaved near the optimum (vs. flat cosine)</td><td>✖ not used</td>
      <td>N/A — we maximize activation, not match an embedding</td></tr>
    <tr><td><b>Manifold projection</b> (MPGD/MCG)</td><td>project the gradient onto the manifold's tangent
      plane</td><td>keeps only \(g_\parallel\) — no step ever leaves real-image space</td>
      <td>✖ not tried</td><td>the principled cure for low-α <em>while keeping gradients</em> — a strong
      next experiment</td></tr>
    <tr><td><b>SAG</b> (self-attention guidance)</td><td>blur-and-contrast on the attended regions</td>
      <td>sharpens structure at moderate scale</td><td>⚠ off in our config</td>
      <td>diffusers-0.27 add_noise incompatibility; BrainDiVE uses 0.75</td></tr>
  </table></div>
  <div class="cbox"><h4>The honest summary of the fix-stack</h4>
  <p>We ran the biggest, cheapest wins — cutouts + TV/range (plan 1), the noise-normalized step, and
  time-travel (plan 2). They <b>helped</b> (the normalized step is what made anything appear; hygiene and
  time-travel reduced texture) but <b>hit the α ceiling</b>: none turned the DINOv2 gradient fully
  photographic, and the held-out encoder showed the gradient methods partly reward-hack. The one fix-stack
  tool we <em>haven't</em> tried and that directly targets α is <b>manifold projection (MPGD)</b> — the
  natural next gradient-side experiment. Beyond that, the bigger wins are leaving gradients behind (plan 4
  / readout-head / DDPO).</p></div>
</div>
'''

EXTRA_CSS = '''
<style id="results-style">
  .pex{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:14px 16px;margin:12px 0}
  .pex h4{margin:0 0 8px;font-size:15px;color:var(--ink)}
  .pex p{margin:0 0 7px;font-size:14px;color:var(--muted)}
  .pex p:last-child{margin-bottom:0}
  .pex .foryou{color:var(--ink);border-top:1px solid var(--line);padding-top:8px;margin-top:4px}
  .pex a{color:var(--accent)}
  .rh{font-size:15px;color:var(--accent2);margin:26px 0 6px;letter-spacing:.02em}
  .rcap{color:var(--muted);font-size:14px;margin:10px 0 4px}
  .rnote,.cbox{background:var(--panel);border:1px solid var(--line);border-left:3px solid var(--warn);
    border-radius:10px;padding:14px 16px;margin:18px 0;color:var(--ink);font-size:14.5px}
  .cbox{border-left-color:var(--accent2)}
  .cbox h4{margin:0 0 8px;font-size:15px;color:var(--ink)}
  .cbox p{margin:0 0 9px} .cbox p:last-child{margin:0}
  .frozen{color:var(--good);font-weight:700}
  .rewardfig{display:block;max-width:520px;width:100%;height:auto;border:1px solid var(--line);
    border-radius:8px;margin:10px 0;background:#fff}
  .gl{margin:6px 0 14px;padding-left:20px;font-size:14.5px;color:var(--muted)}
  .gl li{margin:6px 0} .gl b{color:var(--ink)}
  .anlg{color:var(--muted);border-left:2px solid var(--accent2);padding:2px 0 2px 14px;margin:12px 0;
    font-size:14.5px}
  .anlg b{color:var(--ink)}
  .one{font-family:ui-monospace,Menlo,monospace;font-size:12px;letter-spacing:.04em;
    text-transform:uppercase;color:var(--accent);margin:14px 0 2px}
  .seltab{border-collapse:collapse;margin:8px 0 16px;font-size:13.5px}
  .seltab th,.seltab td{border:1px solid var(--line);padding:6px 14px;text-align:left;
    font-variant-numeric:tabular-nums}
  .seltab th{font-family:ui-monospace,Menlo,monospace;font-size:11px;color:var(--muted);
    text-transform:uppercase;letter-spacing:.05em}
  .hmwrap{display:flex;flex-wrap:wrap;gap:14px;margin:12px 0}
  .hm{border:1px solid var(--line);border-radius:8px;padding:8px;background:var(--panel)}
  .hmt{font:11px ui-monospace,Menlo,monospace;color:var(--accent2);margin-bottom:5px;text-align:center}
  .hmtab{border-collapse:collapse}
  .hmtab td{width:34px;height:26px;text-align:center;font:10px ui-monospace,Menlo,monospace;
    color:var(--ink);border:1px solid var(--line)}
  .hmtab td.hc{background:transparent;color:var(--muted);border:none;font-size:9px}
  .rgw{overflow-x:auto;margin:12px 0}
  .rg{min-width:calc(96px + var(--n)*128px)}
  .rrow{display:grid;grid-template-columns:90px repeat(var(--n),1fr);gap:8px;align-items:center;margin-top:8px}
  .rhead{margin-top:0}
  .ch{font:12px ui-monospace,SFMono-Regular,Menlo,monospace;color:var(--muted);text-align:center;line-height:1.3}
  .rl{font:12px ui-monospace,SFMono-Regular,Menlo,monospace;color:var(--ink);text-align:right;padding-right:2px}
  .rc{aspect-ratio:1;border-radius:7px;overflow:hidden;background:var(--panel2);border:1px solid var(--line)}
  .rc img{display:block;width:100%;height:100%;object-fit:cover}
  .rc .miss{display:flex;height:100%;align-items:center;justify-content:center;
    font:11px ui-monospace,monospace;color:var(--muted)}
</style>
'''

# ---- assemble: strip <head> wrapper, inline MathJax, inject results + css ----
body = ORIG[ORIG.find("<body"):ORIG.rfind("</body>")]
body = body[body.find(">")+1:]                       # drop the <body ...> tag itself
# insert Results before the last top-level part (References)
idx = body.rfind('class="part"')
open_idx = body.rfind('<', 0, idx)
body = body[:open_idx] + RESULTS + NEXTSTEPS + NEXTGEN + PAPERS + PAPERS_EXPANDED + FIXSTACK + body[open_idx:]
# add a TOC entry (insert before the toc's closing </ol>)
tj = body.find('class="toc"')
if tj > 0:
    ol_end = body.find('</ol>', tj)
    toc_add = ('<li><a href="#results">Part VII &middot; Results (with images)</a></li>'
               '<li><a href="#next">Part VIII &middot; Next steps &mdash; math &amp; intuition</a></li>'
               '<li><a href="#bettergen">Part IX &middot; Using a better generator (REPA / RCG / stronger backbones)</a></li><li><a href="#papers">Part X &middot; Related papers we have (generator × encoder landscape)</a></li><li><a href="#fixstack">Part XII &middot; Making guidance better — fix-stack &amp; what we tried</a></li><li><a href="#papers-expanded">Part XI &middot; Every paper, expanded</a></li>')
    body = body[:ol_end] + toc_add + body[ol_end:]

# extract original <style> and MathJax config from head
style = re.search(r'<style>.*?</style>', ORIG, re.S).group(0)
mjcfg = re.search(r'<script>\s*window\.MathJax.*?</script>', ORIG, re.S).group(0)

page = (
    '<title>Guiding a Brain Encoder to Synthesize Images — teaching guide + results</title>\n'
    + mjcfg + '\n'
    + '<script>\n' + MJS + '\n</script>\n'
    + style + '\n'
    + EXTRA_CSS + '\n'
    + body
)
OUT.write_text(page)
print("wrote", OUT, f"({len(page)//1024} KB)  images≈{page.count('data:image/jpeg')}")
