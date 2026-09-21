#!/usr/bin/env python
"""Generate the live floor-loss results HTML (E1 sweep done, E2 in progress)."""
import base64, pathlib

HERE = pathlib.Path("/private/tmp/claude-501/-Users-danielnurieli-AI-work-MindReading/43cf3c52-3984-4cbc-81e8-a62f9b0f5504/scratchpad")
OUT = pathlib.Path("/Users/danielnurieli/AI work/MindReading/brainconcept/RESULTS/floor_live_results.html")

# run -> (z*, w, achieved drive z_norm, final anchor_cos, dino realness z or None)
RUNS = [
    ("z1_w1", 1, 1.0, 1.578, -0.349, 1.78),
    ("z2_w1", 2, 1.0, 2.329, -0.245, 2.38),
    ("z3_w1", 3, 1.0, 3.208, -0.276, None),
    ("z4_w1", 4, 1.0, 4.168, -0.117, None),
    ("z2_w05", 2, 0.5, 2.559, -0.321, None),
    ("z2_w2", 2, 2.0, 2.093, -0.323, None),
]
INIT_COS = 0.686  # every run starts here (on the face anchor)

def b64(name):
    return base64.b64encode((HERE / f"sm_{name}.jpg").read_bytes()).decode()

# ---- SVG frontier (drive x vs anchor_cos y) ----
def xy(d, c):
    x = 60 + d / 5.0 * 420
    y = 290 - (c + 0.40) / 0.50 * 250
    return x, y

zsweep = [r for r in RUNS if r[1] == 1.0]            # w=1: z*=1..4
zsweep.sort(key=lambda r: r[0])
wvar = [r for r in RUNS if r[2] != 1.0]              # z*=2, w=0.5 / 2

poly = " ".join(f"{xy(d,c)[0]:.1f},{xy(d,c)[1]:.1f}" for _,_,_,d,c,_ in zsweep)
pts_z = "".join(
    f'<circle cx="{xy(d,c)[0]:.1f}" cy="{xy(d,c)[1]:.1f}" r="6" fill="var(--accent)"/>'
    f'<text x="{xy(d,c)[0]:.1f}" y="{xy(d,c)[1]-12:.1f}" class="pt">z*={z}</text>'
    for _,z,_,d,c,_ in zsweep)
pts_w = "".join(
    f'<circle cx="{xy(d,c)[0]:.1f}" cy="{xy(d,c)[1]:.1f}" r="6" fill="var(--signal)"/>'
    f'<text x="{xy(d,c)[0]:.1f}" y="{xy(d,c)[1]+20:.1f}" class="pt sig">w={w:g}</text>'
    for _,z,w,d,c,_ in wvar)
# axes ticks
xticks = "".join(f'<text x="{60+d/5*420:.0f}" y="312" class="tick">{d}</text>'
                 f'<line x1="{60+d/5*420:.0f}" y1="290" x2="{60+d/5*420:.0f}" y2="294" class="axl"/>'
                 for d in range(0,6))
yticks = "".join(f'<text x="48" y="{290-(c+0.40)/0.5*250+4:.0f}" class="tick" text-anchor="end">{c:+.1f}</text>'
                 f'<line x1="56" y1="{290-(c+0.40)/0.5*250:.0f}" x2="480" y2="{290-(c+0.40)/0.5*250:.0f}" class="grid"/>'
                 for c in [-0.4,-0.3,-0.2,-0.1,0.0,0.1])

rows = ""
for name,z,w,d,c,dino in RUNS:
    dino_s = f'<span class="nh">+{dino:.2f}</span>' if dino is not None else '<span class="dim">pending</span>'
    hi = ' class="hl"' if (name=="z4_w1") else ''
    rows += (f'<tr{hi}><td>z*={z}, w={w:g}</td><td>{d:.2f}</td>'
             f'<td>{c:+.3f}</td><td>{dino_s}</td></tr>')

imgs = ""
order = [("z1_w1","z*=1"),("z2_w1","z*=2"),("z3_w1","z*=3"),("z4_w1","z*=4"),
         ("z2_w05","z*=2 · w=0.5"),("z2_w2","z*=2 · w=2")]
for name,lab in order:
    _,z,w,d,c,dino = next(r for r in RUNS if r[0]==name)
    imgs += (f'<figure><img alt="floor {name} final" src="data:image/jpeg;base64,{b64(name)}"/>'
             f'<figcaption><b>{lab}</b><span>drive {d:.2f}σ · anchor cos {c:+.2f}</span></figcaption></figure>')

HTML = f'''<title>brainconcept — floor-loss live results</title>
<style>
  :root{{--paper:#eef1f2;--card:#f9fbfb;--ink:#16212b;--muted:#586572;--faint:#8b959d;
    --line:#dbe1e4;--accent:#0e6d77;--accent-soft:rgba(14,109,119,.10);
    --signal:#a85f18;--signal-soft:rgba(168,95,24,.12);--noise:#a3adb4;
    --serif:"Charter","Iowan Old Style","Palatino Linotype",Georgia,serif;
    --sans:ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
    --mono:ui-monospace,"SF Mono","JetBrains Mono",Menlo,Consolas,monospace;--maxw:52rem;}}
  @media (prefers-color-scheme:dark){{:root{{--paper:#0e1418;--card:#141c22;--ink:#e6edee;--muted:#9aa7b0;
    --faint:#6a7883;--line:#243039;--accent:#45b6c0;--accent-soft:rgba(69,182,192,.14);
    --signal:#d8934a;--signal-soft:rgba(216,147,74,.16);--noise:#4e5d67;}}}}
  :root[data-theme="light"]{{--paper:#eef1f2;--card:#f9fbfb;--ink:#16212b;--muted:#586572;--faint:#8b959d;
    --line:#dbe1e4;--accent:#0e6d77;--accent-soft:rgba(14,109,119,.10);--signal:#a85f18;--signal-soft:rgba(168,95,24,.12);--noise:#a3adb4;}}
  :root[data-theme="dark"]{{--paper:#0e1418;--card:#141c22;--ink:#e6edee;--muted:#9aa7b0;--faint:#6a7883;
    --line:#243039;--accent:#45b6c0;--accent-soft:rgba(69,182,192,.14);--signal:#d8934a;--signal-soft:rgba(216,147,74,.16);--noise:#4e5d67;}}
  *{{box-sizing:border-box}} body{{margin:0;background:var(--paper);color:var(--ink);font-family:var(--serif);
    font-size:1.05rem;line-height:1.7;-webkit-font-smoothing:antialiased}}
  .wrap{{max-width:var(--maxw);margin:0 auto;padding:0 1.4rem}}
  a{{color:var(--accent)}} p{{margin:1rem 0}}
  .eyebrow{{font-family:var(--sans);font-size:.74rem;letter-spacing:.2em;text-transform:uppercase;color:var(--accent);font-weight:600;margin:0 0 .8rem}}
  h1{{font-size:clamp(1.8rem,4.5vw,2.5rem);line-height:1.1;margin:0;font-weight:600;letter-spacing:-.01em;text-wrap:balance}}
  .dek{{font-size:1.1rem;color:var(--muted);margin:.9rem 0 0;max-width:40rem}}
  header{{border-bottom:1px solid var(--line);margin-bottom:2rem;padding:3rem 0 2rem}}
  .status{{display:flex;flex-wrap:wrap;gap:.6rem;margin:1.4rem 0 0;font-family:var(--sans);font-size:.82rem}}
  .chip{{padding:.3rem .7rem;border-radius:999px;border:1px solid var(--line);background:var(--card);color:var(--muted);font-weight:600}}
  .chip b{{color:var(--ink)}} .chip.on{{border-color:var(--accent);color:var(--accent);background:var(--accent-soft)}}
  .chip.warn{{border-color:var(--signal);color:var(--signal);background:var(--signal-soft)}}
  h2{{font-size:1.35rem;font-weight:600;margin:2.6rem 0 .3rem;padding-top:1.6rem;border-top:2px solid var(--ink);text-wrap:balance}}
  .sub{{color:var(--muted);font-style:italic;margin:.1rem 0 1.2rem}}
  .nh{{font-family:var(--sans);font-variant-numeric:tabular-nums;color:var(--signal);font-weight:600}}
  .dim{{color:var(--faint);font-family:var(--sans);font-size:.85rem}}
  .card{{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:1.2rem 1.3rem;margin:1.4rem 0}}
  .tw{{overflow-x:auto;margin:1.2rem 0}} table{{border-collapse:collapse;width:100%;font-family:var(--sans);font-size:.92rem}}
  th,td{{text-align:right;padding:.55rem .7rem;border-bottom:1px solid var(--line);font-variant-numeric:tabular-nums}}
  th:first-child,td:first-child{{text-align:left}}
  thead th{{color:var(--faint);font-weight:600;font-size:.74rem;letter-spacing:.04em;text-transform:uppercase;border-bottom:1.5px solid var(--line)}}
  tbody tr:hover{{background:var(--accent-soft)}} tr.hl{{background:var(--signal-soft)}} tr.hl:hover{{background:var(--signal-soft)}}
  svg{{width:100%;height:auto;display:block}}
  .pt{{font-family:var(--sans);font-size:11px;font-weight:600;fill:var(--accent);text-anchor:middle}}
  .pt.sig{{fill:var(--signal)}} .tick{{font-family:var(--sans);font-size:11px;fill:var(--faint)}}
  .axl{{stroke:var(--faint);stroke-width:1}} .grid{{stroke:var(--line);stroke-width:1}}
  .axtitle{{font-family:var(--sans);font-size:12px;fill:var(--muted);font-weight:600}}
  .grid-imgs{{display:grid;grid-template-columns:1fr;gap:1rem;margin:1.2rem 0}}
  figure{{margin:0;border:1px solid var(--line);border-radius:10px;overflow:hidden;background:var(--card)}}
  figure img{{width:100%;display:block}}
  figcaption{{display:flex;justify-content:space-between;align-items:baseline;gap:1rem;padding:.6rem .9rem;font-family:var(--sans);font-size:.84rem}}
  figcaption b{{color:var(--ink)}} figcaption span{{color:var(--muted);font-variant-numeric:tabular-nums}}
  .note{{font-size:.95rem;color:var(--muted);border-left:2px solid var(--line);padding:.1rem 0 .1rem 1rem;margin:1.4rem 0}}
  footer{{border-top:1px solid var(--line);margin-top:2.5rem;padding:1.6rem 0 3rem;font-family:var(--sans);font-size:.82rem;color:var(--faint)}}
</style>

<header><div class="wrap">
  <p class="eyebrow">brainconcept · floor-loss study · live</p>
  <h1>Driving the face ROI: the drive-vs-novelty frontier</h1>
  <p class="dek">Same loss everywhere — <b style="color:var(--ink)">satisfy the ROI to z*, then push the image off the face anchor</b>. Sweeping the floor z* traces how hard we can drive the brain while staying selective.</p>
  <div class="status">
    <span class="chip on">E1 (1-phrase anchor) · <b>6/6 done</b></span>
    <span class="chip warn">E2 (10-phrase anchor) · <b>in progress</b></span>
    <span class="chip">full BLIP + DINOv2 scoring · after all 12</span>
  </div>
</div></header>

<div class="wrap">

  <h2>The E1 frontier — six runs</h2>
  <p class="sub">Loss: ReLU(z* − drive) + w·cos(image, face-anchor). Drive in σ above natural photos; anchor cos in image↔image space (starts at +{INIT_COS:.2f}).</p>
  <div class="card">
    <svg viewBox="0 0 520 340" role="img" aria-label="drive vs anchor-cosine frontier">
      {yticks}{xticks}
      <line x1="60" y1="40" x2="60" y2="290" class="axl"/>
      <line x1="60" y1="290" x2="480" y2="290" class="axl"/>
      <polyline points="{poly}" fill="none" stroke="var(--accent)" stroke-width="2" stroke-dasharray="4 3" opacity=".6"/>
      {pts_z}{pts_w}
      <text x="270" y="332" class="axtitle" text-anchor="middle">achieved drive  (σ above natural photos)  →</text>
      <text x="16" y="165" class="axtitle" text-anchor="middle" transform="rotate(-90 16 165)">← anchor cosine (novelty)</text>
    </svg>
    <p class="dim" style="margin:.6rem 0 0">Blue = z*-sweep at w=1 (the frontier). Ochre = novelty-weight variants at z*=2. Up-and-right = drives harder while staying nearer the face; down = pushed further off-face.</p>
  </div>

  <div class="tw"><table>
    <thead><tr><th>run</th><th>achieved drive</th><th>anchor cos</th><th>DINOv2 realness</th></tr></thead>
    <tbody>{rows}</tbody>
  </table></div>
  <p class="note"><b>What the frontier says.</b> Raising the floor z* 1→4 lifts achieved drive almost exactly onto it (1.58 → 2.33 → 3.21 → <span class="nh">4.17σ</span>) while the image stays progressively <em>closer</em> to the face anchor (cos −0.35 → −0.12): a hard drive target leaves less slack for novelty. The held-out <b>DINOv2</b> encoder (never optimized against) also reads these as strong face-drivers (+1.78, +2.38σ) — so the drive is real, not a hack of our own encoder. The novelty-weight sweep (w=0.5→2 at z*=2) mostly trades drive (2.56→2.09σ) for a touch more push-off.</p>

  <h2>What it actually generated (final step, 6 seeds each)</h2>
  <p class="sub">E1's single-phrase anchor is a narrow "close-up face" direction — pushing off it lands on <em>group / scene</em> faces rather than leaving face-space. That's the hypothesis E2's broader anchor tests.</p>
  <div class="grid-imgs">{imgs}</div>

  <h2>Still cooking</h2>
  <p>The <b>E2</b> runs (10-phrase anchor, tightness 0.88 vs E1's 0.98 — deliberately broader) are training now, 2-at-a-time on the L4. Once all 12 finish, the load-once scorer runs <b>BLIP-2 "is it a face?"</b> (the honest, un-gameable novelty metric) plus DINOv2 realness on every run, and I'll build the final <b>E1-vs-E2 frontier</b>: face-rate vs achieved drive.</p>

</div>
<footer><div class="wrap">brainconcept floor-loss · subject 1 faces ROI · L4 <span class="dim">brainguide-l4</span> · drive/anchor from training logs, realness from held-out DINOv2 (RESULTS/heldout). Live snapshot — E2 + full scoring pending.</div></footer>
'''

OUT.write_text(HTML)
print("wrote", OUT, len(HTML), "bytes")
