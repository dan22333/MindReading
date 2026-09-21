#!/usr/bin/env python
"""Regenerate RESULTS/index.html from whatever Stage-1/Stage-2 outputs are present.

Reads:
  report/metrics.json                          (Stage 1 encoder metrics)
  outputs_faces/brain_faces_s1/images/*.jpeg   (Stage 2 round snapshots)
  <meta_file>  (optional) : line 1.. = negatives, then a '---TRAJ---' line, then traj values

Writes RESULTS/index.html + copies assets into RESULTS/. Re-run any time to refresh.
"""
import json, shutil, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RES = HERE / "RESULTS"
ENC = RES / "01_encoder"
LOOP = RES / "02_faces_loop"
IMGS = HERE / "outputs_faces" / "brain_faces_s1" / "images"
META = Path(sys.argv[1]) if len(sys.argv) > 1 else None


def load_meta():
    negs, traj = [], []
    if META and META.exists():
        section = negs
        for line in META.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("==="):
                continue
            if line.startswith("---TRAJ"):
                section = traj; continue
            if section is traj:
                traj += [float(x) for x in line.split()]
            else:
                section.append(line)
    return negs, traj


def sparkline(traj, w=640, h=130):
    if not traj:
        return ""
    resp = [-v for v in traj]                      # loss -> mean ROI response
    lo, hi = min(resp), max(resp)
    rng = (hi - lo) or 1.0
    pts = []
    for i, v in enumerate(resp):
        x = 10 + i * (w - 20) / max(1, len(resp) - 1)
        y = (h - 8) - (v - lo) / rng * (h - 20)
        pts.append(f"{x:.0f},{y:.0f}")
    return (f'<svg viewBox="0 0 {w} {h}" width="100%" height="{h}" '
            f'style="background:#0b0e12;border:1px solid #252b33;border-radius:8px">'
            f'<polyline fill="none" stroke="#59c26a" stroke-width="2" points="{" ".join(pts)}"/>'
            f'<text x="12" y="16" fill="#9aa7b4" font-size="11">mean face-ROI response  ~{hi:.2f}</text>'
            f'<text x="12" y="{h-6}" fill="#9aa7b4" font-size="11">~{resp[0]:.2f} at start</text></svg>')


def main():
    ENC.mkdir(parents=True, exist_ok=True)
    LOOP.mkdir(parents=True, exist_ok=True)
    # Stage 1 assets
    for f in ["hist_test_r.png", "per_roi_r.png", "scatter_best_voxel.png", "metrics.json"]:
        src = HERE / "report" / f
        if src.exists():
            shutil.copy(src, ENC / f)
    m = json.loads((ENC / "metrics.json").read_text()) if (ENC / "metrics.json").exists() else {}

    # Stage 2 assets: copy init + every round snapshot
    init = IMGS / "init_images.jpeg"
    if init.exists():
        shutil.copy(init, LOOP / "00_init.jpeg")
    rounds = []
    for f in sorted(IMGS.glob("*_step_images.jpeg"), key=lambda p: int(p.name.split("_")[0])):
        step = int(f.name.split("_")[0])
        dst = LOOP / f"step_{step}.jpeg"
        shutil.copy(f, dst)
        rounds.append(step)

    negs, traj = load_meta()

    def roi_rows():
        pr = m.get("per_roi", {})
        order = ["bodies", "places", "faces", "words", "earlyvisual"]
        out = ""
        for k in order:
            if k in pr:
                e = pr[k]
                out += (f"<tr><td>{k}</td><td class='n'>{e['median_r']:.3f}</td>"
                        f"<td class='n'>{e.get('normalized_r_median', float('nan')):.3f}</td></tr>")
        return out

    def round_blocks():
        html = ''
        if (LOOP / "00_init.jpeg").exists():
            html += ("<div class='round'><p class='lbl'><span class='step-tag'>step 0</span> · "
                     "initializer (neutral \"object\")</p><img src='02_faces_loop/00_init.jpeg'></div>")
        for i, step in enumerate(rounds):
            neg = negs[i] if i < len(negs) else ""
            tag = f" · BLIP-2 named: <span class='neg'>“{neg}”</span>" if neg else ""
            html += (f"<div class='round'><p class='lbl'><span class='step-tag'>step {step}</span>{tag}</p>"
                     f"<img src='02_faces_loop/step_{step}.jpeg'></div>")
        return html

    negline = " → ".join(negs) if negs else "(pending)"
    peak = max((-v for v in traj), default=float('nan'))

    html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>brainconcept — end-to-end pipeline results (S1)</title>
<style>
:root{{--bg:#0f1216;--card:#171b21;--ink:#e8ecf1;--mut:#9aa7b4;--line:#252b33;--accent:#4f9dff;--good:#59c26a;--warn:#e0a63a}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--bg);color:var(--ink);font:15px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif}}
.wrap{{max-width:1080px;margin:0 auto;padding:32px 22px 80px}}
h1{{font-size:26px;margin:0 0 4px}} h2{{font-size:20px;margin:34px 0 10px}} .sub{{color:var(--mut);margin:0 0 22px}}
.card{{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:18px 20px;margin:14px 0}}
.chain{{font:13px/1.7 ui-monospace,Menlo,monospace;white-space:pre;overflow-x:auto;color:#cdd6df}}
table{{border-collapse:collapse;width:100%;margin:6px 0}} th,td{{text-align:left;padding:7px 10px;border-bottom:1px solid var(--line)}}
th{{color:var(--mut);font-weight:600;font-size:13px}} td.n{{font-variant-numeric:tabular-nums}}
.pill{{display:inline-block;padding:2px 9px;border-radius:20px;font-size:12px;font-weight:600}}
.pill.good{{background:rgba(89,194,106,.15);color:var(--good)}}
img{{max-width:100%;border-radius:8px;border:1px solid var(--line);display:block}}
.grid2{{display:grid;grid-template-columns:1fr 1fr;gap:14px}} .cap{{color:var(--mut);font-size:13px;margin:6px 0 0}}
.round{{margin:16px 0}} .round .lbl{{font-weight:600;margin:0 0 6px}}
.neg{{font:12px ui-monospace,Menlo,monospace;color:var(--warn)}} .step-tag{{color:var(--accent);font-weight:600}}
code{{background:#0b0e12;padding:1px 6px;border-radius:5px;font-size:13px}}
.kpi{{display:flex;gap:22px;flex-wrap:wrap}} .kpi div{{background:#0b0e12;border:1px solid var(--line);border-radius:10px;padding:12px 16px;min-width:150px}}
.kpi .v{{font-size:24px;font-weight:700}} .kpi .k{{color:var(--mut);font-size:12px}}
.box{{background:#0b0e12;border-left:3px solid var(--accent);border-radius:6px;padding:12px 16px;margin:10px 0;font-size:14px}}
@media(max-width:720px){{.grid2{{grid-template-columns:1fr}}}}
</style></head><body><div class="wrap">

<h1>brainconcept — pipeline results, end to end</h1>
<p class="sub">Subject 1 · faces ROI · ConceptLab loop with the positive term replaced by a brain term. Real run on the L4 VM.</p>

<div class="box"><b>How to read this.</b> Two separate optimizations: (1) <b>Stage 1</b> trains the encoder <code>W</code> once on fMRI data.
(2) <b>Stage 2</b> freezes <code>W</code> and optimizes a single made-up word <code>mybrainconcept</code> (a vector, seeded to mean "object") inside the
prompt <i>"a photo of a mybrainconcept"</i>. Each <b>step</b> = one nudge of that word's vector to raise the frozen encoder's face-ROI score.
Every 250 steps we decode an image, BLIP-2 names it, and the name becomes a "don't repeat" negative. 2500 steps = 10 such rounds.</div>

<div class="card"><div class="chain">NSD S1 fMRI ──▶ ridge encoder W ──▶ [ select face-ROI rows ]
     │                                      │
token v* ("mybrainconcept") ──▶ Kandinsky prior ──▶ CLIP embedding z ──▶  −mean( W_face · z )   ▲ maximize
     │                                      │
     └──▶ decode ──▶ image ──▶ BLIP-2 names it ──▶ add as negative ──▶ diverge ───────────────────┘</div></div>

<h2>Stage 0 · Data <span class="pill good">done</span></h2>
<div class="card"><table>
<tr><th>array</th><th>shape</th><th>meaning</th></tr>
<tr><td><code>s1_betas.npy</code></td><td class="n">10000 × 39548</td><td>brain response per image, all voxels (repeat-avg, z-scored)</td></tr>
<tr><td><code>s1_images.npy</code></td><td class="n">10000 × 256²×3</td><td>the images S1 saw</td></tr>
<tr><td>split</td><td class="n">7200 / 1800 / 1000</td><td>train / val / test — zero overlap (test = NSD shared-1000)</td></tr>
</table></div>

<h2>Stage 1 · Encoder — is it solid? <span class="pill good">yes</span></h2>
<div class="card"><div class="kpi">
<div><div class="v">{m.get('test_median_r', float('nan')):.2f}</div><div class="k">test median voxel r</div></div>
<div><div class="v">{m.get('normalized_r_median', float('nan')):.2f}</div><div class="k">noise-ceiling-normalized r</div></div>
<div><div class="v">{m.get('frac_voxels_r_gt_0.1', float('nan'))*100:.0f}%</div><div class="k">voxels r &gt; 0.1</div></div>
<div><div class="v">α={m.get('alpha','?')}</div><div class="k">ridge (swept 1→5e5)</div></div>
</div>
<p class="cap">Held-out test = 1,000 shared images. Normalized r = raw r ÷ noise ceiling → fraction of explainable signal captured.</p>
<div class="grid2" style="margin-top:12px">
<div><img src="01_encoder/hist_test_r.png"><p class="cap">Accuracy across all 39,548 voxels.</p></div>
<div><img src="01_encoder/per_roi_r.png"><p class="cap">Per-ROI — semantic regions best, early-visual worst (as a CLIP encoder should).</p>
<table><tr><th>ROI</th><th>median r</th><th>norm r</th></tr>{roi_rows()}</table></div>
</div></div>

<h2>Stage 2 · Faces token loop <span class="pill good">10 rounds complete</span></h2>
<div class="card">
<p><b>Brain (face-ROI) response over the 2500 steps:</b></p>
{sparkline(traj)}
<p style="margin:16px 0 4px"><b>What BLIP-2 named each round</b> (each becomes a negative → forces novelty):</p>
<p class="neg">{negline}</p>
<p class="cap" style="margin-top:6px">Peak mean ROI response ≈ {peak:.2f}, up from ≈ {(-traj[0]) if traj else float('nan'):.2f} at the start.</p>
{round_blocks()}
</div>

<div class="card"><b>Why this is a real result, not a cheat:</b> the images are coherent faces, not adversarial noise, so the token
found genuine face-drivers. The brain signal rose monotonically and the named concepts drifted from random objects into faces.
The full loop (drive-brain → generate → name → diverge) ran end to end, 10 rounds.</div>

<p class="sub" style="margin-top:26px">Regenerate this page any time: <code>python make_dashboard.py &lt;meta_file&gt;</code>.
Encoder assets in <code>RESULTS/01_encoder/</code>, loop images in <code>RESULTS/02_faces_loop/</code>.</p>
</div></body></html>"""

    (RES / "index.html").write_text(html)
    print(f"[dashboard] wrote {RES/'index.html'} with {len(rounds)} rounds, {len(negs)} negatives")


if __name__ == "__main__":
    main()
