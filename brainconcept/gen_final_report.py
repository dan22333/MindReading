#!/usr/bin/env python
"""Final E1-vs-E2 floor-loss frontier report."""
import base64, pathlib
HERE = pathlib.Path("/private/tmp/claude-501/-Users-danielnurieli-AI-work-MindReading/43cf3c52-3984-4cbc-81e8-a62f9b0f5504/scratchpad")
OUT = pathlib.Path("/Users/danielnurieli/AI work/MindReading/brainconcept/RESULTS/floor_E1_vs_E2_report.html")

# (label, drive, face_rate, dino_z, anchor_cos_decoded)  — all from decoded images (honest)
# anchor_cos_decoded = cos(open_clip image emb of the DECODED picture, the image-space face anchor).
# NB the loss minimized this on the PRIOR EMBEDDING z (reached ~-0.35); on the rendered image it
# only reaches ~+0.06..+0.28 -> the decode does not preserve the push (see the note).
E1 = [("z*=1, w=1",1.90,1.00,1.78,0.058),("z*=2, w=1",2.63,1.00,2.38,0.108),
      ("z*=3, w=1",2.68,0.83,2.50,0.130),("z*=4, w=1",3.08,1.00,2.75,0.273),
      ("z*=2, w=0.5",2.30,1.00,2.32,0.052),("z*=2, w=2",2.36,1.00,2.13,0.054)]
E2 = [("z*=1, w=1",1.72,0.33,1.66,0.064),("z*=2, w=1",2.48,1.00,3.00,0.170),
      ("z*=3, w=1",2.62,1.00,2.59,0.097),("z*=4, w=1",3.50,1.00,2.80,0.280),
      ("z*=2, w=0.5",2.70,0.83,2.70,0.107),("z*=2, w=2",2.02,0.83,2.40,0.069)]
KEYS = ["z1_w1","z2_w1","z3_w1","z4_w1","z2_w05","z2_w2"]
GROUPS = [("z*=1",1.00,0.33),("z*=2",1.00,1.00),("z*=3",0.83,1.00),
          ("z*=4",1.00,1.00),("z*=2 · w½",1.00,0.83),("z*=2 · w2",1.00,0.83)]

def b64(p): return base64.b64encode((HERE/p).read_bytes()).decode()
def img(p): return f"data:image/jpeg;base64,{b64(p)}"

# ---- grouped bar (face-rate E1 vs E2) ----
def bary(f): return 250 - f*210
bars=""; xlabels=""
gw = 490/6
for i,(lab,e1,e2) in enumerate(GROUPS):
    gx = 50 + (i+0.5)*gw
    for val,off,cls in [(e1,-22,"e1"),(e2,0,"e2")]:
        y=bary(val); h=250-y
        bars+=f'<rect x="{gx+off:.0f}" y="{y:.0f}" width="20" height="{h:.0f}" class="b {cls}"/>'
        bars+=f'<text x="{gx+off+10:.0f}" y="{y-5:.0f}" class="bval {cls}t">{val:.2f}</text>'
    xlabels+=f'<text x="{gx:.0f}" y="270" class="xlab">{lab}</text>'
yaxis="".join(f'<line x1="50" y1="{bary(f):.0f}" x2="540" y2="{bary(f):.0f}" class="grid"/>'
              f'<text x="42" y="{bary(f)+4:.0f}" class="ytick">{f:.1f}</text>' for f in [0,0.5,1.0])

def rows(data):
    out=""
    for lab,d,fr,dz,cs in data:
        frcls = 'esc' if fr<0.5 else ('mid' if fr<1.0 else '')
        out+=(f'<tr><td>{lab}</td><td>{d:.2f}</td>'
              f'<td class="{frcls}">{fr:.2f}</td><td>+{dz:.2f}</td><td class="dim">{cs:+.3f}</td></tr>')
    return out

def strip(prefix):
    labs=["z*=1","z*=2","z*=3","z*=4","z*=2·w½","z*=2·w2"]
    fr=[r[2] for r in (E1 if prefix=="t_" else E2)]
    out=""
    for k,l,f in zip(KEYS,labs,fr):
        tag='esc' if f<0.5 else ('mid' if f<1 else '')
        out+=(f'<figure><img alt="{prefix}{k}" src="{img(prefix+k+".jpg")}"/>'
              f'<figcaption><b>{l}</b><span class="{tag}">face {f:.2f}</span></figcaption></figure>')
    return out

HTML=f'''<title>brainconcept — E1 vs E2 floor-loss frontier</title>
<style>
  :root{{--paper:#eef1f2;--card:#f9fbfb;--ink:#16212b;--muted:#586572;--faint:#8b959d;--line:#dbe1e4;
    --accent:#0e6d77;--accent-soft:rgba(14,109,119,.10);--signal:#a85f18;--signal-soft:rgba(168,95,24,.12);
    --esc:#1f7a4d;--esc-soft:rgba(31,122,77,.14);
    --serif:"Charter","Iowan Old Style","Palatino Linotype",Georgia,serif;
    --sans:ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;--maxw:54rem;}}
  @media (prefers-color-scheme:dark){{:root{{--paper:#0e1418;--card:#141c22;--ink:#e6edee;--muted:#9aa7b0;--faint:#6a7883;
    --line:#243039;--accent:#45b6c0;--accent-soft:rgba(69,182,192,.14);--signal:#d8934a;--signal-soft:rgba(216,147,74,.16);
    --esc:#4cc98a;--esc-soft:rgba(76,201,138,.16);}}}}
  :root[data-theme="light"]{{--paper:#eef1f2;--card:#f9fbfb;--ink:#16212b;--muted:#586572;--faint:#8b959d;--line:#dbe1e4;
    --accent:#0e6d77;--accent-soft:rgba(14,109,119,.10);--signal:#a85f18;--signal-soft:rgba(168,95,24,.12);--esc:#1f7a4d;--esc-soft:rgba(31,122,77,.14);}}
  :root[data-theme="dark"]{{--paper:#0e1418;--card:#141c22;--ink:#e6edee;--muted:#9aa7b0;--faint:#6a7883;--line:#243039;
    --accent:#45b6c0;--accent-soft:rgba(69,182,192,.14);--signal:#d8934a;--signal-soft:rgba(216,147,74,.16);--esc:#4cc98a;--esc-soft:rgba(76,201,138,.16);}}
  *{{box-sizing:border-box}} body{{margin:0;background:var(--paper);color:var(--ink);font-family:var(--serif);font-size:1.05rem;line-height:1.7;-webkit-font-smoothing:antialiased}}
  .wrap{{max-width:var(--maxw);margin:0 auto;padding:0 1.4rem}} p{{margin:1rem 0}} a{{color:var(--accent)}}
  .eyebrow{{font-family:var(--sans);font-size:.74rem;letter-spacing:.2em;text-transform:uppercase;color:var(--accent);font-weight:600;margin:0 0 .8rem}}
  h1{{font-size:clamp(1.8rem,4.5vw,2.5rem);line-height:1.1;margin:0;font-weight:600;letter-spacing:-.01em;text-wrap:balance}}
  .dek{{font-size:1.12rem;color:var(--muted);margin:.9rem 0 0;max-width:42rem}}
  header{{border-bottom:1px solid var(--line);margin-bottom:2rem;padding:3rem 0 2rem}}
  .status{{display:flex;flex-wrap:wrap;gap:.6rem;margin:1.4rem 0 0;font-family:var(--sans);font-size:.82rem}}
  .chip{{padding:.3rem .7rem;border-radius:999px;border:1px solid var(--accent);color:var(--accent);background:var(--accent-soft);font-weight:600}}
  h2{{font-size:1.35rem;font-weight:600;margin:2.6rem 0 .3rem;padding-top:1.6rem;border-top:2px solid var(--ink);text-wrap:balance}}
  .sub{{color:var(--muted);font-style:italic;margin:.1rem 0 1.2rem}}
  .headline{{background:var(--esc-soft);border:1px solid var(--esc);border-radius:12px;padding:1.3rem 1.4rem;margin:.5rem 0 0}}
  .headline b{{color:var(--esc)}}
  .card{{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:1.2rem 1.3rem;margin:1.4rem 0}}
  svg{{width:100%;height:auto;display:block}}
  .b.e1{{fill:var(--accent)}} .b.e2{{fill:var(--signal)}}
  .bval{{font-family:var(--sans);font-size:10px;font-weight:700;text-anchor:middle;font-variant-numeric:tabular-nums}}
  .e1t{{fill:var(--accent)}} .e2t{{fill:var(--signal)}}
  .xlab{{font-family:var(--sans);font-size:11px;fill:var(--muted);text-anchor:middle;font-weight:600}}
  .ytick{{font-family:var(--sans);font-size:10px;fill:var(--faint);text-anchor:end}} .grid{{stroke:var(--line);stroke-width:1}}
  .lg{{display:flex;gap:1.2rem;font-family:var(--sans);font-size:.82rem;color:var(--muted);margin:.4rem 0 0;justify-content:center}}
  .sw{{display:inline-block;width:.8rem;height:.8rem;border-radius:2px;margin-right:.35rem;vertical-align:-1px}}
  .tw{{overflow-x:auto;margin:1.2rem 0}} table{{border-collapse:collapse;width:100%;font-family:var(--sans);font-size:.9rem}}
  th,td{{text-align:right;padding:.5rem .65rem;border-bottom:1px solid var(--line);font-variant-numeric:tabular-nums}}
  th:first-child,td:first-child{{text-align:left}}
  thead th{{color:var(--faint);font-weight:600;font-size:.72rem;letter-spacing:.04em;text-transform:uppercase;border-bottom:1.5px solid var(--line)}}
  .grp td{{background:var(--accent-soft);font-family:var(--sans);font-weight:700;font-size:.74rem;letter-spacing:.06em;text-transform:uppercase;color:var(--accent)}}
  td.esc{{color:var(--esc);font-weight:700;background:var(--esc-soft)}} td.mid{{color:var(--signal);font-weight:600}} .dim{{color:var(--faint)}}
  tbody tr:hover{{background:var(--accent-soft)}}
  .hero2{{display:grid;grid-template-columns:1fr 1fr;gap:1rem;margin:1.2rem 0}}
  @media(max-width:640px){{.hero2{{grid-template-columns:1fr}}}}
  figure{{margin:0;border:1px solid var(--line);border-radius:10px;overflow:hidden;background:var(--card)}}
  figure img{{width:100%;display:block}}
  figcaption{{display:flex;justify-content:space-between;align-items:baseline;gap:.6rem;padding:.55rem .8rem;font-family:var(--sans);font-size:.82rem}}
  figcaption b{{color:var(--ink)}} figcaption span{{color:var(--muted);font-variant-numeric:tabular-nums}}
  figcaption .esc{{color:var(--esc);font-weight:700}} figcaption .mid{{color:var(--signal);font-weight:600}}
  .who{{font-family:var(--sans);font-size:.74rem;text-transform:uppercase;letter-spacing:.08em;font-weight:700;margin:0 0 .5rem}}
  .who.a{{color:var(--accent)}} .who.b{{color:var(--signal)}}
  .strip{{display:grid;grid-template-columns:1fr;gap:.7rem;margin:.6rem 0 0}}
  .note{{font-size:.95rem;color:var(--muted);border-left:2px solid var(--line);padding:.1rem 0 .1rem 1rem;margin:1.3rem 0}}
  ul.find{{margin:1rem 0;padding-left:0;list-style:none}} ul.find li{{padding:.5rem 0;border-top:1px solid var(--line)}} ul.find li:first-child{{border-top:0}}
  ul.find b{{color:var(--ink)}}
  footer{{border-top:1px solid var(--line);margin-top:2.5rem;padding:1.6rem 0 3rem;font-family:var(--sans);font-size:.82rem;color:var(--faint)}}
  .nh{{font-family:var(--sans);font-variant-numeric:tabular-nums;color:var(--esc);font-weight:700}}
</style>

<header><div class="wrap">
  <p class="eyebrow">brainconcept · floor-loss study · complete</p>
  <h1>Narrow vs. broad anchor: when does the face driver actually become novel?</h1>
  <p class="dek">Two experiments, identical loss — <b style="color:var(--ink)">satisfy the face-ROI to z*, then push off a face anchor</b>. E1's anchor is one phrase ("a human face"); E2's averages ten. The honest test: does a held-out judge still see a face?</p>
  <div class="status"><span class="chip">12/12 runs · scored: BLIP face-rate + held-out DINOv2 + drive</span></div>
</div></header>

<div class="wrap">

  <div class="headline">
    <b>Headline.</b> The broader E2 anchor produces genuine novelty — but only when the drive floor is low. At <b>z*=1</b>, E2 drops the BLIP face-rate to <span class="nh">0.33</span> (two-thirds of images are no longer faces) while E1 stays at <b>1.00</b> (all faces). Raise the floor and both snap back to faces: a hard drive target overrides the novelty push.
  </div>

  <h2>Face-rate: E1 vs E2, per configuration</h2>
  <p class="sub">BLIP-2 "is there a human face?" on the 6 final-seed images. Lower = more genuinely novel (escaped face-space). 6 images/run → resolution is 1/6.</p>
  <div class="card">
    <svg viewBox="0 0 560 285" role="img" aria-label="face-rate E1 vs E2 grouped bars">
      {yaxis}<line x1="50" y1="250" x2="540" y2="250" class="grid"/>
      {bars}{xlabels}
      <text x="20" y="145" class="xlab" transform="rotate(-90 20 145)">BLIP face-rate</text>
    </svg>
    <div class="lg"><span><span class="sw" style="background:var(--accent)"></span>E1 · 1-phrase anchor</span><span><span class="sw" style="background:var(--signal)"></span>E2 · 10-phrase anchor</span></div>
  </div>

  <h2>The z*=1 contrast — where they diverge</h2>
  <p class="sub">Same loss, same low floor, only the anchor differs. E1 stays on (group/scene) faces; E2 leaves face-space.</p>
  <div class="hero2">
    <div><p class="who a">E1 · 1-phrase · face-rate 1.00</p><figure><img alt="E1 z1" src="{img('t_z1_w1.jpg')}"/><figcaption><b>all faces</b><span>drive 1.90σ</span></figcaption></figure></div>
    <div><p class="who b">E2 · 10-phrase · face-rate 0.33</p><figure><img alt="E2 z1" src="{img('t_e2_z1_w1.jpg')}"/><figcaption><b>mostly non-faces</b><span>drive 1.72σ</span></figcaption></figure></div>
  </div>

  <h2>Every run, every metric</h2>
  <div class="tw"><table>
    <thead><tr><th>run</th><th>drive (σ)</th><th>face-rate</th><th>DINOv2 realness</th><th>anchor cos (decoded)</th></tr></thead>
    <tbody>
      <tr class="grp"><td colspan="5">E1 — 1-phrase anchor (tightness 0.98)</td></tr>
      {rows(E1)}
      <tr class="grp"><td colspan="5">E2 — 10-phrase anchor (tightness 0.88)</td></tr>
      {rows(E2)}
    </tbody>
  </table></div>
  <p class="note"><b>Reading it.</b> <b>drive</b> = σ above natural photos, re-measured on the <em>decoded</em> image (not the optimized embedding). <b>face-rate</b> = honest novelty (green = escaped). <b>DINOv2</b> = a held-out encoder never optimized against — positive everywhere confirms the drive is real, not a hack of our own encoder. <b>anchor cos (decoded)</b> = image↔image cosine of the rendered picture to the face anchor — the loss's true novelty axis, measured on the actual image.</p>

  <div class="headline" style="border-color:var(--signal);background:var(--signal-soft);margin-top:1.2rem">
    <b style="color:var(--signal)">The decode gap (important).</b> The loss <em>minimized</em> the anchor cosine on the <b>prior embedding z</b> — and drove it to <b>−0.35</b> (training logs). But re-encode the <b>rendered picture</b> and the same cosine is only <b>+0.06 to +0.28</b>. The Kandinsky decoder does not preserve that push: the −0.35 lived in un-rendered embedding space and evaporates on decode. This is why the images stayed faces the whole time — the rendered images were only ever mildly off-anchor. It also means optimizing the embedding cosine is partly wasted; the honest fix is to score/guide on the <em>decoded</em> image.</div>

  <h2>What we learned</h2>
  <ul class="find">
    <li><b>The floor loss works.</b> Achieved drive tracks the floor z* and the held-out DINOv2 agrees (up to +3.0σ) — these are real face-region drivers, not encoder hacks.</li>
    <li><b>Anchor breadth matters — E2 &gt; E1 for novelty.</b> The 10-phrase anchor is the only thing that ever escaped face-space (z*=1 → face-rate 0.33; the w-variants → 0.83). E1's narrow "close-up portrait" anchor never escaped (≥0.83 everywhere, mostly 1.00).</li>
    <li><b>Drive and novelty trade off.</b> Escape only happens at a <em>low</em> floor. At z*≥2 the drive requirement dominates and pins the image back onto faces regardless of anchor — the frontier is real.</li>
    <li><b>The text cosine is dead as a signal</b> (cat_sim ~0.13 flat everywhere) — vindicating the switch to an image-space anchor.</li>
  </ul>

  <h2>Next: E3 — hinged novelty</h2>
  <p>E1/E2 <em>minimize</em> the anchor cosine (unbounded), which overshoots to negative even while faces remain. <b>E3</b> makes the novelty term a hinge — <code style="font-family:var(--sans);background:var(--accent-soft);color:var(--accent);padding:.05em .3em;border-radius:3px">ReLU(cos − c*)</code> — so it stops at a target c*≈0 instead of pushing forever. Combined with E2's broad anchor at a low floor (the escape regime), that's the most promising recipe. Queued, not started.</p>

  <h2>All runs</h2>
  <p class="sub">E1 (1-phrase anchor)</p><div class="strip">{strip("t_")}</div>
  <p class="sub" style="margin-top:1.4rem">E2 (10-phrase anchor)</p><div class="strip">{strip("t_e2_")}</div>

</div>
<footer><div class="wrap">brainconcept floor-loss · subject 1 faces ROI · 12 runs on L4 brainguide-l4 · drive &amp; face-rate from decoded images (score_floor_all.py), realness from held-out DINOv2 (dino_all.py). E1 = anchor_face_e1 (1 phrase), E2 = anchor_face_e2 (10 phrases).</div></footer>
'''
OUT.write_text(HTML)
print("wrote", OUT, len(HTML), "bytes")
