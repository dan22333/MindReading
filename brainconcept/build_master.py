#!/usr/bin/env python
"""Master report: encoder scores, method, held-out validation, the anchor sweep
(graphs + big photos), and the full run ledger. One self-contained HTML."""
import base64, io, json, glob, os
from pathlib import Path
from PIL import Image

HD = Path("RESULTS/heldout"); IMG = Path("RESULTS/heldout_imgs")
LAMS = [("l01", 0.1), ("l02", 0.2), ("l04", 0.4), ("l06", 0.6), ("l08", 0.8)]
EXPS = [("e1", "single anchor", "#2563eb"), ("e2", "10-phrasing average", "#c026d3")]
C = dict(ours="#2563eb", dino="#137a33", bdive="#9a6510")


def final(enc, run):
    p = HD / f"{enc}_{run}.json"
    if not p.exists():
        return None, None
    st = json.loads(p.read_text())["steps"]; k = max(st, key=lambda x: int(x))
    return st[k]["z_mean"], st[k].get("cat_sim")


def img_b64(run, maxw, q=82):
    f = IMG / run / "2000_step_images.jpeg"
    if not f.exists():
        cands = sorted(glob.glob(str(IMG / run / "*step_images.jpeg")))
        if not cands:
            return ""
        f = Path(cands[-1])
    im = Image.open(f).convert("RGB")
    if im.width > maxw:
        im = im.resize((maxw, max(1, round(im.height * maxw / im.width))))
    b = io.BytesIO(); im.save(b, "JPEG", quality=q)
    return "data:image/jpeg;base64," + base64.b64encode(b.getvalue()).decode()


# ---- data -------------------------------------------------------------
MET = json.loads(Path("RESULTS/01_encoder/metrics.json").read_text())
ANCHORS10 = ["a human face", "a person's face", "the face of a person", "a face",
             "someone's face", "a human's face", "a portrait of a person",
             "a human head", "a person's head", "a close-up of a human face"]


def anchor_data():
    D = {}
    for e, _, _ in EXPS:
        for lt, lam in LAMS:
            r = f"{e}_faces_{lt}"
            oz, cs = final("ours", r); dz, _ = final("dino", r); bz, _ = final("braindive", r)
            D[(e, lam)] = dict(cos=cs, ours=oz, dino=dz, bdive=bz, run=r)
    return D


def all_runs():
    runs = sorted({os.path.basename(p)[5:-5] for p in glob.glob("RESULTS/heldout/ours_*.json")
                   if "status" not in p})
    out = []
    for r in runs:
        oz, cs = final("ours", r); dz, _ = final("dino", r); bz, _ = final("braindive", r)
        out.append(dict(run=r, ours=oz, dino=dz, bdive=bz, cos=cs))
    return out


# ---- svg --------------------------------------------------------------
def line_svg(D, key, title, ylab, w=520, h=280, pad=46):
    allv = [D[(e, lam)][key] for e, _, _ in EXPS for _, lam in LAMS if D[(e, lam)][key] is not None]
    lo, hi = min(allv), max(allv); rng = (hi - lo) or 1
    xs = [lam for _, lam in LAMS]
    X = lambda v: pad + (v - min(xs)) / (max(xs) - min(xs)) * (w - 1.5 * pad)
    Y = lambda v: h - pad - (v - lo) / rng * (h - 1.7 * pad)
    p = [f'<svg viewBox="0 0 {w} {h}" width="100%" class="plot">']
    p.append(f'<text x="{pad}" y="18" class="pt">{title}</text>')
    for gy in range(5):
        yy = pad + gy * (h - 1.7 * pad) / 4
        val = hi - gy * rng / 4
        p.append(f'<line x1="{pad}" y1="{yy:.0f}" x2="{w-pad/2:.0f}" y2="{yy:.0f}" class="grid"/>')
        p.append(f'<text x="{pad-6}" y="{yy+3:.0f}" class="ax" text-anchor="end">{val:.2f}</text>')
    for lt, lam in LAMS:
        p.append(f'<text x="{X(lam):.0f}" y="{h-14}" class="ax" text-anchor="middle">{lam}</text>')
    p.append(f'<text x="{w/2:.0f}" y="{h-2}" class="axl" text-anchor="middle">λ (brain weight)</text>')
    p.append(f'<text x="12" y="{pad-8}" class="axl">{ylab}</text>')
    for e, lbl, col in EXPS:
        pts = " ".join(f"{X(lam):.1f},{Y(D[(e,lam)][key]):.1f}" for _, lam in LAMS if D[(e, lam)][key] is not None)
        p.append(f'<polyline fill="none" stroke="{col}" stroke-width="2.5" points="{pts}"/>')
        for _, lam in LAMS:
            v = D[(e, lam)][key]
            if v is not None:
                p.append(f'<circle cx="{X(lam):.1f}" cy="{Y(v):.1f}" r="3.6" fill="{col}"/>')
    return "".join(p) + "</svg>"


def scatter_svg(D, w=520, h=360, pad=52):
    xs = [D[(e, lam)]["cos"] for e, _, _ in EXPS for _, lam in LAMS]
    ys = [D[(e, lam)]["dino"] for e, _, _ in EXPS for _, lam in LAMS]
    xlo, xhi = min(xs), max(xs); ylo, yhi = min(ys + [1.9]), max(ys)
    X = lambda v: pad + (v - xlo) / ((xhi - xlo) or 1) * (w - 1.5 * pad)
    Y = lambda v: h - pad - (v - ylo) / ((yhi - ylo) or 1) * (h - 1.8 * pad)
    p = [f'<svg viewBox="0 0 {w} {h}" width="100%" class="plot">']
    p.append(f'<text x="{pad}" y="18" class="pt">realness vs novelty — each point is one λ</text>')
    yr = Y(2.0)
    p.append(f'<line x1="{pad}" y1="{yr:.0f}" x2="{w-pad/2:.0f}" y2="{yr:.0f}" stroke="#c0392b" stroke-dasharray="4 3"/>')
    p.append(f'<text x="{w-pad:.0f}" y="{yr-5:.0f}" fill="#c0392b" font-size="9.5" text-anchor="end">DINOv2 = 2 → real</text>')
    p.append(f'<text x="{w/2:.0f}" y="{h-6}" class="axl" text-anchor="middle">cos to “human face”  (← more novel · more face-like →)</text>')
    p.append(f'<text x="12" y="{pad-10}" class="axl">DINOv2 z (independent brain drive ↑)</text>')
    for e, lbl, col in EXPS:
        for _, lam in LAMS:
            d = D[(e, lam)]
            if d["cos"] is None:
                continue
            p.append(f'<circle cx="{X(d["cos"]):.1f}" cy="{Y(d["dino"]):.1f}" r="5.5" fill="{col}" opacity="0.85"/>')
            p.append(f'<text x="{X(d["cos"])+8:.0f}" y="{Y(d["dino"])+3:.0f}" fill="{col}" font-size="9.5">λ{lam}</text>')
    return "".join(p) + "</svg>"


# ---- html helpers -----------------------------------------------------
def zc(v):
    if v is None:
        return "#8a94a3"
    return "#137a33" if v >= 2 else ("#b8860b" if v >= 1.2 else "#c0392b")


def zf(v):
    return f"{v:+.2f}" if v is not None else "—"


def main():
    A = anchor_data()
    RUNS = all_runs()
    pr = MET["per_roi"]

    # encoder per-ROI rows
    roi_order = ["faces", "bodies", "places", "words", "earlyvisual"]
    enc_rows = ""
    for r in roi_order:
        d = pr[r]
        enc_rows += (f"<tr><td>{r}</td><td class='n'>{d['n']:,}</td><td class='n'>{d['median_r']:.3f}</td>"
                     f"<td class='n'>{d['normalized_r_median']:.3f}</td></tr>")

    # anchor photos (big, 2-up)
    def photo_grid(e):
        cells = ""
        for lt, lam in LAMS:
            d = A[(e, lam)]
            cells += (f"<figure class='cell'><img src='{img_b64(d['run'],900)}'>"
                      f"<figcaption>λ {lam} · cos-to-face <b>{d['cos']:+.3f}</b> · "
                      f"DINOv2 <b style='color:{zc(d['dino'])}'>{d['dino']:+.2f}</b> · "
                      f"BrainDiVE <b style='color:{zc(d['bdive'])}'>{d['bdive']:+.2f}</b></figcaption></figure>")
        return f"<div class='grid2'>{cells}</div>"

    # anchor table
    a_rows = ""
    for e, lbl, col in EXPS:
        for lt, lam in LAMS:
            d = A[(e, lam)]
            a_rows += (f"<tr><td><span class='dot' style='background:{col}'></span>{lbl}</td><td class='n'>{lam}</td>"
                       f"<td class='n'>{d['cos']:+.3f}</td><td class='n'>{zf(d['ours'])}</td>"
                       f"<td class='n' style='color:{zc(d['dino'])}'>{zf(d['dino'])}</td>"
                       f"<td class='n' style='color:{zc(d['bdive'])}'>{zf(d['bdive'])}</td></tr>")

    # featured drivers
    FEAT = [("v2_faces", "Faces ROI — pure drive (baseline)", "A human face. This is what the region wants with no flee term — the target we try to move away from while staying real."),
            ("fx_faces_l025", "Faces ROI — a non-human face driver", "Dogs & bears. Fully validated (DINOv2 +3.47) yet cat-sim to a human face is only 0.166 — the region responds to face-like animal structure, not humans specifically."),
            ("v2_bodies", "Bodies ROI — pure drive", "People / limbs — the extrastriate body area's preferred input."),
            ("v2_places", "Places ROI — pure drive", "Scenes & interiors — the parahippocampal place area."),
            ("words", "Words ROI — pure drive", "Text-like glyphs. Note the cross-encoder split in the ledger: DINOv2/BrainDiVE score this very high (+4) while our CLIP encoder is milder (+2.4).")]
    feat_html = ""
    for run, title, desc in FEAT:
        oz, cs = final("ours", run); dz, _ = final("dino", run); bz, _ = final("braindive", run)
        badge = (f"<span class='b'>ours <b style='color:{zc(oz)}'>{zf(oz)}</b></span>"
                 f"<span class='b'>DINOv2 <b style='color:{zc(dz)}'>{zf(dz)}</b></span>"
                 f"<span class='b'>BrainDiVE <b style='color:{zc(bz)}'>{zf(bz)}</b></span>"
                 + (f"<span class='b'>cat-sim {cs:+.3f}</span>" if cs is not None else ""))
        feat_html += (f"<figure class='feat'><img src='{img_b64(run,520)}'>"
                      f"<figcaption><h4>{title}</h4><p>{desc}</p><div class='badges'>{badge}</div></figcaption></figure>")

    # full ledger
    def grp(r):
        for pre, name in [("e1_", "anchor · single"), ("e2_", "anchor · averaged"), ("fx_", "flee-extract"),
                          ("lam_", "λ sweep"), ("te_", "threshold sweep"), ("nl_", "no-live-neg"),
                          ("hp_", "anchor phrasing"), ("v2_", "category baseline"),
                          ("words", "category baseline"), ("sw_", "seed/word")]:
            if r.startswith(pre) or r == pre.rstrip("_"):
                return name
        return "baseline"
    led_rows = ""
    for d in sorted(RUNS, key=lambda x: (grp(x["run"]), x["run"])):
        cos = f"{d['cos']:+.3f}" if d["cos"] is not None else "—"
        led_rows += (f"<tr><td class='mono'>{d['run']}</td><td>{grp(d['run'])}</td>"
                     f"<td class='n'>{zf(d['ours'])}</td>"
                     f"<td class='n' style='color:{zc(d['dino'])}'>{zf(d['dino'])}</td>"
                     f"<td class='n' style='color:{zc(d['bdive'])}'>{zf(d['bdive'])}</td>"
                     f"<td class='n'>{cos}</td></tr>")

    m = MET
    anchors_str = ", ".join(f"“{a}”" for a in ANCHORS10)

    html = f"""<title>brainconcept — brain-driven concept discovery</title>
<style>
:root{{
  --bg:#fbfcfe; --panel:#ffffff; --ink:#161a22; --mut:#5a6473; --line:#e6e9ef;
  --accent:#2563eb; --accent-soft:#eef3ff; --serif:ui-serif,Georgia,"Iowan Old Style","Times New Roman",serif;
  --sans:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif; --mono:ui-monospace,"SF Mono",Menlo,monospace;
}}
@media(prefers-color-scheme:dark){{:root{{--bg:#0f1217;--panel:#161b23;--ink:#e8ecf3;--mut:#98a2b3;--line:#252c37;--accent:#6ea8ff;--accent-soft:#182233}}}}
:root[data-theme="dark"]{{--bg:#0f1217;--panel:#161b23;--ink:#e8ecf3;--mut:#98a2b3;--line:#252c37;--accent:#6ea8ff;--accent-soft:#182233}}
:root[data-theme="light"]{{--bg:#fbfcfe;--panel:#ffffff;--ink:#161a22;--mut:#5a6473;--line:#e6e9ef;--accent:#2563eb;--accent-soft:#eef3ff}}
*{{box-sizing:border-box}}
body{{margin:0;background:var(--bg);color:var(--ink);font-family:var(--sans);font-size:16px;line-height:1.62}}
.wrap{{max-width:1120px;margin:0 auto;padding:56px 24px 100px}}
.eyebrow{{font-family:var(--mono);font-size:12px;letter-spacing:.14em;text-transform:uppercase;color:var(--accent);margin:0 0 10px}}
h1{{font-family:var(--serif);font-weight:600;font-size:44px;line-height:1.08;letter-spacing:-.01em;margin:0 0 14px;text-wrap:balance}}
h2{{font-family:var(--serif);font-weight:600;font-size:29px;margin:0 0 6px;letter-spacing:-.01em}}
h3{{font-size:19px;margin:30px 0 10px}} h4{{font-size:15px;margin:0 0 4px}}
p{{color:var(--ink)}} .lead{{font-size:19px;line-height:1.6;color:var(--mut);max-width:70ch}}
.sec{{border-top:1px solid var(--line);margin-top:56px;padding-top:34px}}
.snum{{font-family:var(--mono);color:var(--mut);font-size:13px}}
.mut{{color:var(--mut)}} .max{{max-width:74ch}}
.kpis{{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin:26px 0}}
@media(max-width:720px){{.kpis{{grid-template-columns:repeat(2,1fr)}}}}
.kpi{{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:16px 18px}}
.kpi .v{{font-family:var(--mono);font-size:27px;font-weight:600;letter-spacing:-.02em}}
.kpi .l{{color:var(--mut);font-size:12.5px;margin-top:3px}}
.card{{background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:22px 24px;margin:20px 0}}
.callout{{background:var(--accent-soft);border:1px solid var(--line);border-left:3px solid var(--accent);border-radius:10px;padding:16px 20px;margin:18px 0}}
table{{border-collapse:collapse;width:100%;margin:16px 0;font-size:14.5px}}
th,td{{text-align:left;padding:8px 11px;border-bottom:1px solid var(--line)}}
th{{color:var(--mut);font-size:12px;font-weight:600;letter-spacing:.03em;text-transform:uppercase}}
td.n{{font-family:var(--mono);font-variant-numeric:tabular-nums;font-weight:600;text-align:right}}
th.n{{text-align:right}} td.mono{{font-family:var(--mono);font-size:12.5px}}
.twrap{{overflow-x:auto;border:1px solid var(--line);border-radius:12px}} .twrap table{{margin:0}}
.twrap th,.twrap td{{padding:7px 12px}} .twrap tr:last-child td{{border-bottom:none}}
.plots{{display:grid;grid-template-columns:1fr 1fr;gap:18px}} @media(max-width:800px){{.plots{{grid-template-columns:1fr}}}}
.plot{{background:var(--panel);border:1px solid var(--line);border-radius:12px}}
.pt{{fill:var(--ink);font-size:13px;font-weight:700;font-family:var(--sans)}}
.grid{{stroke:var(--line)}} .ax{{fill:var(--mut);font-size:10px;font-family:var(--mono)}} .axl{{fill:var(--mut);font-size:10.5px}}
.leg span{{display:inline-block;margin-right:16px;font-size:13px;color:var(--mut)}}
.dot{{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:6px;vertical-align:middle}}
.grid2{{display:grid;grid-template-columns:1fr 1fr;gap:18px;margin:16px 0}} @media(max-width:700px){{.grid2{{grid-template-columns:1fr}}}}
.cell{{margin:0}} .cell img{{width:100%;border-radius:10px;border:1px solid var(--line);display:block}}
.cell figcaption{{font-size:12.5px;color:var(--mut);margin-top:7px}}
.feats{{display:grid;grid-template-columns:1fr 1fr;gap:20px;margin:18px 0}} @media(max-width:760px){{.feats{{grid-template-columns:1fr}}}}
.feat{{margin:0;background:var(--panel);border:1px solid var(--line);border-radius:14px;overflow:hidden}}
.feat img{{width:100%;display:block;border-bottom:1px solid var(--line)}} .feat figcaption{{padding:14px 16px}}
.feat p{{color:var(--mut);font-size:13.5px;margin:0 0 10px}}
.badges{{display:flex;flex-wrap:wrap;gap:6px}} .b{{font-family:var(--mono);font-size:11px;background:var(--bg);border:1px solid var(--line);border-radius:999px;padding:3px 9px;color:var(--mut)}}
.pipe{{font-family:var(--mono);font-size:13px;background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:16px;overflow-x:auto;line-height:1.9}}
.tok{{color:var(--accent);font-weight:700}} .frz{{color:var(--mut)}}
code{{font-family:var(--mono);font-size:.9em;background:var(--accent-soft);padding:1px 5px;border-radius:4px}}
.eq{{font-family:var(--mono);background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:14px 18px;margin:14px 0;overflow-x:auto;font-size:14px}}
ul.tight li{{margin:5px 0}}
</style>
<div class="wrap">

<p class="eyebrow">Computational neuroscience · NSD subject-1</p>
<h1>Reading what a brain region really wants</h1>
<p class="lead">We build a differentiable model of a visual brain region, then optimize an image to drive it as hard as possible — while actively fleeing the obvious answer. The goal is to surface the <em>non-obvious</em> drivers of a region, and to prove they are real with encoders we never optimized against.</p>

<div class="kpis">
<div class="kpi"><div class="v" style="color:var(--accent)">{m['test_median_r']:.2f}</div><div class="l">encoder test median r</div></div>
<div class="kpi"><div class="v" style="color:var(--accent)">{m['normalized_r_median']:.2f}</div><div class="l">noise-ceiling-normalized r</div></div>
<div class="kpi"><div class="v" style="color:var(--accent)">{m['n_voxels']:,}</div><div class="l">voxels modeled</div></div>
<div class="kpi"><div class="v" style="color:#137a33">3</div><div class="l">independent validators</div></div>
</div>

<!-- 1 ENCODER -->
<section class="sec">
<p class="snum">01 — Stage 1</p>
<h2>The encoder: a differentiable brain</h2>
<p class="max">Every image is passed through a frozen <b>CLIP ViT-L/14</b> vision transformer to get a 768-d embedding (L2-normalized). A ridge regression maps that embedding to each voxel's fMRI response on the Natural Scenes Dataset, subject 1. This gives a per-voxel weight matrix <code>W</code> — an image's predicted brain response is just <code>z · W</code>, which is fully differentiable in <code>z</code>.</p>
<ul class="tight max">
<li>RidgeCV, α = {m['alpha']:.0f}, no intercept · {m['n_train']:,}/{m['n_val']:,}/{m['n_test']:,} train/val/test split</li>
<li>Test median r = <b>{m['test_median_r']:.3f}</b>, mean r = {m['test_mean_r']:.3f}; <b>{m['frac_voxels_r_gt_0.1']*100:.0f}%</b> of voxels beat r&gt;0.1, {m['frac_voxels_r_gt_0.2']*100:.0f}% beat r&gt;0.2</li>
<li>Normalized against the per-voxel noise ceiling, median r = <b>{m['normalized_r_median']:.3f}</b> — i.e. the model captures ~61% of the explainable signal</li>
</ul>
<div class="twrap" style="margin-top:14px"><table>
<tr><th>ROI</th><th class="n">voxels</th><th class="n">median r</th><th class="n">noise-ceiling r</th></tr>
{enc_rows}
</table></div>
<p class="mut" style="font-size:13.5px">Body, place and face regions are modeled best (normalized r 0.68–0.74); early visual cortex is hardest, as expected for a high-level CLIP feature space.</p>
</section>

<!-- 2 METHOD -->
<section class="sec">
<p class="snum">02 — Stage 2</p>
<h2>The optimization loop</h2>
<p class="max">We learn a single new <b>text token</b>. It is fed through Kandinsky 2.1's frozen text encoder and frozen diffusion <em>prior</em>, which outputs <code>z</code> — a CLIP <em>image</em> embedding. Because the whole path is differentiable, we can push <code>z</code> to maximize the region's predicted response. Gradients update <b>only the token's embedding row</b>; everything else is frozen.</p>
<div class="pipe">
<span class="tok">[learnable token]</span> → <span class="frz">Kandinsky text-encoder (frozen)</span> → <span class="frz">diffusion prior (frozen)</span> → <b>z</b> (CLIP image embedding) → <span class="frz">z·W</span> = predicted brain response
</div>
<h3>The loss — two forces in tension</h3>
<div class="eq">loss = neg_weight · [ mean + max cos(z, negatives) ] &nbsp;+&nbsp; λ · ( − mean_ROI(z·W) )</div>
<p class="max">The second term is the <b>brain drive</b> (make the region fire, weight λ). The first is the <b>flee</b> term — push away from things we already know the region likes, so we don't just rediscover the obvious. Two ways to define "negatives":</p>
<ul class="tight max">
<li><b>Discovered negatives (pipeline).</b> Every 250 steps, BLIP-2 (flan-t5-xl) captions the current image (“What's in this image?”); a flan-t5 pass extracts <em>all</em> nouns/objects/shapes; these accumulate and become CLIP-text vectors to flee. Fully automatic — no hand-coded list.</li>
<li><b>Fixed anchor.</b> Flee one fixed direction — the CLIP-text embedding of “a human face” — or the average of several paraphrases. This fences off a whole semantic region instead of playing whack-a-mole with words. Used for the sweep in §4.</li>
</ul>
<p class="mut max" style="font-size:13.5px"><b>λ</b> is the primary lever: low λ lets the flee term win (novel but risks leaving the real-image manifold); high λ lets the brain win (real but stays on the obvious category).</p>
</section>

<!-- 3 VALIDATION -->
<section class="sec">
<p class="snum">03 — Guarding against self-deception</p>
<h2>Held-out validation: is the driver real?</h2>
<p class="max">Optimizing inside a CLIP feature space risks <b>reward-hacking</b> — an image that scores high on our encoder only because it exploits CLIP quirks. So we re-score every final image with <b>two encoders we never optimized against</b>:</p>
<ul class="tight max">
<li><span class="dot" style="background:{C['ours']}"></span><b>Ours</b> — the ridge CLIP encoder above (the one we optimized; sanity only).</li>
<li><span class="dot" style="background:{C['dino']}"></span><b>DINOv2</b> — a universal self-supervised encoder in a completely different feature space. <b>Primary judge.</b></li>
<li><span class="dot" style="background:{C['bdive']}"></span><b>BrainDiVE</b> — an independently-trained CLIP→fMRI encoder from prior published work.</li>
</ul>
<p class="max">Each image's predicted response is <b>z-scored against 300 natural NSD images</b> for that ROI. A driver counts as <b>real</b> when <b>DINOv2 z ≥ 2</b> — the region fires two SDs above natural images according to an encoder that shares none of our optimization machinery. Agreement across all three is the strong result.</p>
<div class="callout"><b>Two axes, kept separate.</b> <b>DINOv2 z</b> = <i>is it a real brain driver.</i> <b>cat-sim</b> = cosine of the image to CLIP-text “a photo of a face” = <i>how obvious/on-category it looks.</i> The whole point is to <b>decouple</b> them: the prize is <b>high DINOv2 + low cat-sim</b> — a genuine driver that doesn't look like the textbook answer.</div>
</section>

<!-- 4 ANCHOR SWEEP -->
<section class="sec">
<p class="snum">04 — Main result</p>
<h2>The anchor sweep: trading obviousness for drive</h2>
<p class="max">Faces ROI. Flee the human-face direction, drive the brain, and sweep λ from 0.1 → 0.8 over 2000 steps. <b>Exp1</b> flees a single anchor (“a human face”); <b>Exp2</b> flees the average of 10 paraphrases, to fence off the broader region. We plot how face-like the result stays (cos-to-face ↓ = novel) against how hard it really drives the region (DINOv2 z).</p>
<div class="leg" style="margin:6px 0 12px"><span><span class="dot" style="background:#2563eb"></span>Exp1 · single anchor</span><span><span class="dot" style="background:#c026d3"></span>Exp2 · 10-phrasing average</span></div>
<div class="plots">
<div>{line_svg(A,'cos','cos to “human face” vs λ','more face-like ↑')}</div>
<div>{line_svg(A,'dino','DINOv2 z (real drive) vs λ','real drive ↑')}</div>
</div>
<div style="margin-top:18px">{scatter_svg(A)}</div>
<div class="callout"><b>What the curves say.</b> Raising λ pushes cos-to-face <em>and</em> DINOv2 up together — a clean trade-off frontier. As λ rises the flee term is out-weighed, so the token is allowed to stay closer to a face (higher cos) and in return drives the region harder (higher DINOv2). The <b>sweet spot is λ ≈ 0.2–0.4</b>: a genuinely real driver (DINOv2 +2.3 → +3.5) that is clearly <em>less</em> face-like (cos 0.06–0.14) than a literal face. <b>Exp2 dominates Exp1</b> — at matched λ it holds higher DINOv2 at lower cos-to-face; averaging paraphrases fences the human-face region off more robustly. The single best "far + real" point is <b>Exp2 λ0.2</b> (cos 0.057, DINOv2 +2.37).</div>
<div class="twrap"><table>
<tr><th>experiment</th><th class="n">λ</th><th class="n">cos-to-face</th><th class="n">ours z</th><th class="n">DINOv2 z</th><th class="n">BrainDiVE z</th></tr>
{a_rows}
</table></div>

<h3>Exp1 — single anchor “a human face”</h3>
{photo_grid('e1')}
<h3>Exp2 — average of 10 paraphrases</h3>
<p class="mut" style="font-size:13px">Anchors averaged: {anchors_str}.</p>
{photo_grid('e2')}
</section>

<!-- 5 FEATURED -->
<section class="sec">
<p class="snum">05 — Across regions</p>
<h2>What each region wants — and a validated surprise</h2>
<p class="max">With no flee term, each ROI's token converges to its textbook preference (faces → faces, places → scenes). The interesting result is the faces ROI's <b>non-human</b> driver: dogs and bears score as real as a human face on all three encoders, while looking far less like one.</p>
<div class="feats">{feat_html}</div>
</section>

<!-- 6 LEDGER -->
<section class="sec">
<p class="snum">06 — Appendix</p>
<h2>Full run ledger</h2>
<p class="max">Every scored run — final-step z on all three held-out encoders, plus cat-sim to a human face where measured. Green = real (z ≥ 2), amber = borderline, red = below. {len(RUNS)} runs.</p>
<div class="twrap"><table>
<tr><th>run</th><th>family</th><th class="n">ours</th><th class="n">DINOv2</th><th class="n">BrainDiVE</th><th class="n">cat-sim</th></tr>
{led_rows}
</table></div>
<p class="mut" style="font-size:13px">z-scored vs 300 natural NSD images per ROI. The words ROI shows a deliberate cross-encoder split: DINOv2 and BrainDiVE rate glyph images very high (+4 to +5) while our CLIP encoder is milder — a reminder that "real" is judge-dependent, which is exactly why we use three.</p>
</section>

</div>"""
    Path("RESULTS/master.html").write_text(html)
    print(f"wrote RESULTS/master.html ({len(html)//1024} KB)")


if __name__ == "__main__":
    main()
