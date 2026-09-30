#!/usr/bin/env python
"""Floor-loss trajectory report (self-contained HTML).

For every E1 / E2 floor run it plots the per-step trajectory of:
  - the floor LOSS         = drive_term + w * anchor_cos   (from the training log)
  - the IMAGE-embedding cos = anchor_cos                    (novelty term, we push it DOWN)
  - the TEXT-embedding  cos = cat_sim                       (image vs category text, scorer JSON)
plus the drive z (z_norm) for context.

Sources
  RESULTS/floor_logs/log_floor[_e2]_faces_z{Z}_w{W}.log   dense 2000-step: pos_loss, z_norm, drive_term, anchor_cos
  RESULTS/floor/ours_floor[_e2]_faces_z{Z}_w{W}.json      9 checkpoints: z_mean, cat_sim (text cos)
"""
import re, glob, os, json, html, base64, io
from pathlib import Path
from PIL import Image

ROOT = Path(__file__).resolve().parent
LOGDIR = ROOT / "RESULTS" / "floor_logs"
SCOREDIR = ROOT / "RESULTS" / "floor"
IMGDIR = ROOT / "RESULTS" / "floor_imgs"
OUT = ROOT / "RESULTS" / "floor_trajectories.html"

CHECKPOINTS = [0, 250, 500, 750, 1000, 1250, 1500, 1750, 2000]


def img_data_uri(path, target_w=2400, q=80):
    """Downscale a checkpoint strip and return a base64 data URI, or None."""
    if not path.exists():
        return None
    im = Image.open(path).convert("RGB")
    if im.width > target_w:
        h = round(im.height * target_w / im.width)
        im = im.resize((target_w, h), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=q)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def run_dir(stem):
    # stem already carries the e2_ prefix for E2 runs; dirs are floor_<stem>
    return IMGDIR / ("floor_" + stem) / "images"

W_MAP = {"05": 0.5, "1": 1.0, "2": 2.0}


def parse_cfg(stem):
    # stem like  faces_z2_w05   or   e2_faces_z3_w1
    exp = "E2" if stem.startswith("e2_") else "E1"
    s = stem[3:] if stem.startswith("e2_") else stem
    m = re.match(r"faces_z(\d+)_w(\d+)", s)
    z = int(m.group(1)); w = W_MAP[m.group(2)]
    return exp, z, w, s  # s = "faces_z2_w05"


def parse_log(fn, w):
    steps = []
    cur = None
    with open(fn) as f:
        for line in f:
            m = re.match(r"For step #(\d+)", line)
            if m:
                if cur is not None:
                    steps.append(cur)
                cur = {"step": int(m.group(1))}
                continue
            if cur is None:
                continue
            m = re.search(r"pos_loss.*?:\s*(-?[\d.]+)", line)
            if m:
                cur["pos_loss"] = float(m.group(1))
            m = re.search(r"z_norm\s+(-?[\d.]+)\s+drive_term\s+(-?[\d.]+)\s+anchor_cos\s+(-?[\d.]+)", line)
            if m:
                cur["z_norm"] = float(m.group(1))
                cur["drive_term"] = float(m.group(2))
                cur["anchor_cos"] = float(m.group(3))
    if cur is not None:
        steps.append(cur)
    # loss = drive_term + w * anchor_cos
    for d in steps:
        if "drive_term" in d and "anchor_cos" in d:
            d["loss"] = d["drive_term"] + w * d["anchor_cos"]
    return steps


def load_scorer(stem):
    """Return sparse checkpoints {step: {z_mean, cat_sim}} from ours_floor_*.json."""
    fn = SCOREDIR / f"ours_floor_{stem}.json"
    if not fn.exists():
        return None
    d = json.load(open(fn))
    out = {}
    for k, v in d["steps"].items():
        out[int(k)] = {"z_mean": v.get("z_mean"), "cat_sim": v.get("cat_sim")}
    return {"baseline": d.get("baseline"), "steps": out}


# ---------- tiny inline-SVG line chart (dual axis: loss left, cosine right) ----------
def svg_chart(dense, sparse, z_target, w):
    W, H = 620, 300
    ml, mr, mt, mb = 48, 48, 16, 34
    pw, ph = W - ml - mr, H - mt - mb

    def dsample(seq, n=240):
        if len(seq) <= n:
            return seq
        step = len(seq) / n
        return [seq[int(i * step)] for i in range(n)]

    dense_ds = dsample(dense) if dense else []
    xs = [d["step"] for d in dense_ds] if dense_ds else [s for s in sorted(sparse["steps"])] if sparse else [0]
    xmax = 2000.0

    # left axis: loss.  right axis: cosine in [-1, 1]
    losses = [d["loss"] for d in dense_ds if "loss" in d]
    lmax = max(losses) if losses else (z_target + w)
    lmin = min(0.0, min(losses) if losses else 0.0)
    lmax = max(lmax, 0.5)

    def X(step):
        return ml + pw * (step / xmax)

    def YL(v):  # loss axis
        return mt + ph * (1 - (v - lmin) / (lmax - lmin))

    def YR(v):  # cosine axis [-1,1]
        return mt + ph * (1 - (v + 1) / 2.0)

    parts = [f'<svg viewBox="0 0 {W} {H}" width="100%" preserveAspectRatio="xMidYMid meet" role="img">']
    # zero-cos gridline
    parts.append(f'<line x1="{ml}" y1="{YR(0):.1f}" x2="{ml+pw}" y2="{YR(0):.1f}" class="grid0"/>')
    # frame
    parts.append(f'<rect x="{ml}" y="{mt}" width="{pw}" height="{ph}" class="frame"/>')

    # x ticks
    for s in range(0, 2001, 500):
        x = X(s)
        parts.append(f'<line x1="{x:.1f}" y1="{mt+ph}" x2="{x:.1f}" y2="{mt+ph+4}" class="tick"/>')
        parts.append(f'<text x="{x:.1f}" y="{mt+ph+16}" class="axlab" text-anchor="middle">{s}</text>')
    # left (loss) ticks
    for i in range(4):
        v = lmin + (lmax - lmin) * i / 3
        y = YL(v)
        parts.append(f'<text x="{ml-6}" y="{y+3:.1f}" class="axlab" text-anchor="end" fill="#b45309">{v:.1f}</text>')
    # right (cos) ticks
    for v in (-1, 0, 1):
        y = YR(v)
        parts.append(f'<text x="{ml+pw+6}" y="{y+3:.1f}" class="axlab" text-anchor="start" fill="#1f7a3d">{v:g}</text>')

    def polyline(points, cls):
        if not points:
            return ""
        d = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
        return f'<polyline points="{d}" class="{cls}"/>'

    if dense_ds:
        loss_pts = [(X(d["step"]), YL(d["loss"])) for d in dense_ds if "loss" in d]
        img_pts = [(X(d["step"]), YR(d["anchor_cos"])) for d in dense_ds if "anchor_cos" in d]
        z_pts = [(X(d["step"]), YL(min(d["z_norm"], lmax))) for d in dense_ds if "z_norm" in d]
        parts.append(polyline(z_pts, "zline"))
        parts.append(polyline(loss_pts, "lossline"))
        parts.append(polyline(img_pts, "imgline"))

    # sparse text-cos (green) markers + line
    if sparse:
        tpts = []
        for st in sorted(sparse["steps"]):
            cs = sparse["steps"][st].get("cat_sim")
            if cs is None:
                continue
            tpts.append((X(st), YR(cs)))
        parts.append(polyline(tpts, "txtline"))
        for x, y in tpts:
            parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3" class="txtdot"/>')

    # z* target line on loss axis
    yt = YL(min(z_target, lmax))
    parts.append(f'<line x1="{ml}" y1="{yt:.1f}" x2="{ml+pw}" y2="{yt:.1f}" class="ztarget"/>')
    parts.append(f'<text x="{ml+pw-4}" y="{yt-4:.1f}" class="axlab" text-anchor="end" fill="#2563eb">z*={z_target}</text>')

    parts.append("</svg>")
    return "".join(parts)


def main():
    stems = sorted({os.path.basename(f)[len("ours_floor_"):-len(".json")]
                    for f in glob.glob(str(SCOREDIR / "ours_floor_*.json"))})
    runs = []
    for stem in stems:
        exp, z, w, s = parse_cfg(stem)
        prefix = "log_floor_e2_" if exp == "E2" else "log_floor_"
        logfn = LOGDIR / f"{prefix}{s}.log"
        dense = parse_log(logfn, w) if logfn.exists() else []
        sparse = load_scorer(stem)
        runs.append({"exp": exp, "z": z, "w": w, "stem": stem, "dense": dense, "sparse": sparse,
                     "has_log": logfn.exists()})
    runs.sort(key=lambda r: (r["exp"], r["z"], r["w"]))

    # build cards
    cards = []
    for r in runs:
        title = f'{r["exp"]} · faces · z*={r["z"]}σ · w={r["w"]:g}'
        dense = r["dense"]
        if dense:
            fin = dense[-1]
            fin_loss = fin.get("loss")
            fin_img = fin.get("anchor_cos")
            fin_z = fin.get("z_norm")
        else:
            fin_loss = fin_img = fin_z = None
        fin_txt = None
        if r["sparse"]:
            last = max(r["sparse"]["steps"])
            fin_txt = r["sparse"]["steps"][last].get("cat_sim")
        chart = svg_chart(dense, r["sparse"], r["z"], r["w"])
        badges = []
        def b(lbl, val, fmt="{:.3f}", cls=""):
            v = "—" if val is None else fmt.format(val)
            return f'<span class="badge {cls}"><em>{lbl}</em>{v}</span>'
        badges.append(b("final loss", fin_loss))
        badges.append(b("final drive z", fin_z, cls="z"))
        badges.append(b("final image cos", fin_img, cls="img"))
        badges.append(b("final text cos", fin_txt, cls="txt"))
        note = "" if r["has_log"] else '<div class="warn">no training log on VM for this config — text-cosine checkpoints only (dense loss / image-cos unavailable).</div>'
        # per-step image strip
        idir = run_dir(r["stem"])
        sp = r["sparse"]["steps"] if r["sparse"] else {}
        thumbs = []
        for st in CHECKPOINTS:
            fn = idir / ("init_images.jpeg" if st == 0 else f"{st}_step_images.jpeg")
            uri = img_data_uri(fn)
            if uri is None:
                continue
            zv = sp.get(st, {}).get("z_mean")
            cv = sp.get(st, {}).get("cat_sim")
            sub = []
            if zv is not None:
                sub.append(f'<span class="z">z {zv:.2f}</span>')
            if cv is not None:
                sub.append(f'<span class="txt">txt {cv:.2f}</span>')
            thumbs.append(
                f'<figure class="thumb"><span class="stp">step {st}</span>'
                f'<img loading="lazy" src="{uri}" alt="step {st}">'
                f'<figcaption>{" · ".join(sub)}</figcaption></figure>')
        imgrow = f'<div class="imgstrip">{"".join(thumbs)}</div>' if thumbs else ""
        cards.append(f"""
        <div class="card">
          <h3>{html.escape(title)}</h3>
          <div class="badges">{''.join(badges)}</div>
          {note}
          <div class="chartwrap">{chart}</div>
          {imgrow}
        </div>""")

    # summary table
    rows = []
    for r in runs:
        dense = r["dense"]
        fl = f'{dense[-1]["loss"]:.3f}' if dense else "—"
        fi = f'{dense[-1]["anchor_cos"]:.3f}' if dense else "—"
        fz = f'{dense[-1]["z_norm"]:.2f}' if dense else "—"
        ft = "—"
        if r["sparse"]:
            last = max(r["sparse"]["steps"]); v = r["sparse"]["steps"][last].get("cat_sim")
            ft = f"{v:.3f}" if v is not None else "—"
        rows.append(f"<tr><td>{r['exp']}</td><td>z*={r['z']}σ</td><td>w={r['w']:g}</td>"
                    f"<td class='num'>{fl}</td><td class='num z'>{fz}</td>"
                    f"<td class='num img'>{fi}</td><td class='num txt'>{ft}</td></tr>")

    doc = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Floor-loss trajectories — E1 &amp; E2</title>
<style>
:root {{ color-scheme: light dark; --bg:#ffffff; --fg:#1a1a1a; --mut:#666; --card:#fafafa; --bd:#e3e3e3;
        --loss:#b45309; --img:#d97706; --txt:#1f7a3d; --z:#2563eb; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg:#141414; --fg:#e8e8e8; --mut:#9a9a9a; --card:#1d1d1d; --bd:#2f2f2f; }} }}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--bg); color:var(--fg); font:15px/1.55 -apple-system,Segoe UI,Roboto,sans-serif; }}
.wrap {{ max-width:1200px; margin:0 auto; padding:28px 20px 80px; }}
h1 {{ font-size:24px; margin:0 0 4px; }}
.sub {{ color:var(--mut); margin:0 0 22px; }}
.legend {{ display:flex; gap:18px; flex-wrap:wrap; margin:14px 0 26px; font-size:13px; }}
.legend span {{ display:inline-flex; align-items:center; gap:7px; }}
.sw {{ width:22px; height:3px; border-radius:2px; display:inline-block; }}
.grid {{ display:grid; grid-template-columns:1fr; gap:18px; }}
.card {{ background:var(--card); border:1px solid var(--bd); border-radius:12px; padding:14px 14px 12px; }}
.card .chartwrap {{ max-width:640px; }}
.lossbox {{ background:var(--card); border:1px solid var(--bd); border-left:4px solid var(--loss); border-radius:10px; padding:14px 18px; margin:0 0 24px; }}
.lossbox h2 {{ margin:0 0 8px; font-size:16px; }}
.lossbox .eq {{ font-size:16px; background:rgba(120,120,120,.1); border-radius:8px; padding:10px 14px; margin:8px 0; overflow-x:auto; }}
.lossbox ul {{ margin:8px 0 0; padding-left:20px; }} .lossbox li {{ margin:3px 0; }}
.imgstrip {{ display:flex; flex-direction:column; gap:14px; padding:10px 0 4px; margin-top:10px; border-top:1px solid var(--bd); }}
.thumb {{ margin:0; width:100%; text-align:left; }}
.thumb .stp {{ display:inline-block; font-size:13px; font-weight:600; color:var(--fg); margin-bottom:4px; }}
.thumb img {{ width:100%; height:auto; border-radius:6px; border:1px solid var(--bd); display:block; }}
.thumb figcaption {{ font-size:12px; margin-top:4px; display:flex; gap:12px; }}
.thumb .z {{ color:var(--z); }} .thumb .txt {{ color:var(--txt); }}
.card h3 {{ margin:0 0 8px; font-size:15px; }}
.badges {{ display:flex; flex-wrap:wrap; gap:6px; margin-bottom:8px; }}
.badge {{ font-size:11px; background:rgba(120,120,120,.1); border:1px solid var(--bd); border-radius:6px; padding:2px 7px; }}
.badge em {{ color:var(--mut); font-style:normal; margin-right:5px; }}
.badge.img {{ border-color:var(--img); }} .badge.txt {{ border-color:var(--txt); }} .badge.z {{ border-color:var(--z); }}
.chartwrap {{ margin-top:4px; }}
.warn {{ font-size:12px; color:#b45309; background:rgba(180,83,9,.08); border-radius:6px; padding:5px 8px; margin-bottom:8px; }}
svg text.axlab {{ font-size:10px; fill:var(--mut); }}
svg .frame {{ fill:none; stroke:var(--bd); }}
svg .grid0 {{ stroke:var(--bd); stroke-dasharray:2 3; }}
svg .tick {{ stroke:var(--bd); }}
svg .lossline {{ fill:none; stroke:var(--loss); stroke-width:2; }}
svg .imgline {{ fill:none; stroke:var(--img); stroke-width:1.6; opacity:.9; }}
svg .txtline {{ fill:none; stroke:var(--txt); stroke-width:1.4; opacity:.85; }}
svg .txtdot {{ fill:var(--txt); }}
svg .zline {{ fill:none; stroke:var(--z); stroke-width:1.2; opacity:.45; stroke-dasharray:3 3; }}
svg .ztarget {{ stroke:var(--z); stroke-width:1; stroke-dasharray:5 4; opacity:.6; }}
table {{ border-collapse:collapse; width:100%; margin:8px 0 30px; font-size:13px; }}
th,td {{ border-bottom:1px solid var(--bd); padding:6px 10px; text-align:left; }}
td.num {{ text-align:right; font-variant-numeric:tabular-nums; }}
td.img {{ color:var(--img); }} td.txt {{ color:var(--txt); }} td.z {{ color:var(--z); }}
.note {{ color:var(--mut); font-size:13px; max-width:760px; }}
.note code {{ background:rgba(120,120,120,.14); padding:1px 5px; border-radius:4px; }}
</style></head><body><div class="wrap">
<h1>Floor-loss trajectories — E1 &amp; E2</h1>
<p class="sub">faces ROI · every step 0→2000 · loss + image-embedding cosine (novelty) + text-embedding cosine + drive z</p>
<div class="lossbox">
  <h2>The loss we optimize</h2>
  <div class="eq"><b>L</b> = <span style="color:var(--z)">ReLU(z* − z<sub>drive</sub>)</span> &nbsp;+&nbsp; <span style="color:var(--img)">w · cos(img_emb, anchor)</span></div>
  <ul>
    <li><b><span style="color:var(--z)">Drive floor</span></b> — <code>z<sub>drive</sub></code> is the generated image's response in the target ROI, in units of σ over the ROI baseline. <code>ReLU(z* − z<sub>drive</sub>)</code> pushes the drive <i>up to</i> the target <code>z*</code> and then goes flat (=0): once the ROI is driven hard enough, this term stops caring. A floor, not a ceiling.</li>
    <li><b><span style="color:var(--img)">Novelty</span></b> — <code>anchor</code> is the CLIP image-space embedding of the <i>prior-averaged</i> face (the "generic face" the model defaults to). Minimizing <code>cos(img_emb, anchor)</code> pushes each generated image <i>away</i> from that average face — driving novelty. <code>w</code> weights how hard.</li>
    <li><b>Not in the loss:</b> the <span style="color:var(--txt)">text-embedding cosine</span> (image vs the category word "face") is only <i>measured</i>, never optimized — it is our check that we stayed on-category while gaining novelty.</li>
  </ul>
</div>
<div class="lossbox" style="border-left-color:var(--img)">
  <h2>The anchor vocabulary — how E1 and E2 differ</h2>
  <p style="margin:2px 0 10px">Both experiments use the <i>same</i> loss. The only difference is how the face <b>anchor</b> (the thing we push away from) is built: E1 embeds a <b>single</b> phrase; E2 embeds <b>10 phrasings</b> of "a face" and averages them, giving a broader, less-brittle notion of "generic face."</p>
  <div style="display:flex;gap:24px;flex-wrap:wrap">
    <div><b>E1 anchor — 1 phrase</b><ol style="margin:6px 0 0;padding-left:22px"><li><code>a human face</code></li></ol></div>
    <div><b>E2 anchor — 10 phrases (averaged)</b><ol style="margin:6px 0 0;padding-left:22px">
      <li><code>a human face</code></li>
      <li><code>a face</code></li>
      <li><code>a portrait of a person</code></li>
      <li><code>a close-up of a human face</code></li>
      <li><code>a headshot</code></li>
      <li><code>the face of a man</code></li>
      <li><code>the face of a woman</code></li>
      <li><code>a face with eyes, nose, and mouth</code></li>
      <li><code>a human facial portrait</code></li>
      <li><code>a person looking at the camera</code></li>
    </ol></div>
  </div>
</div>
<p class="note">The optimizer
satisfies the <b>drive floor</b> (push drive z up to z*, at which point <code>drive_term→0</code>) and then keeps
lowering the <b>image-embedding cosine</b> to the prior-averaged anchor — pushing the generated image <i>away</i>
from the average face (novelty). <b>E1</b> uses a 1-token concept; <b>E2</b> uses 2 tokens.
The <span style="color:var(--img)">image cos</span> and <span style="color:var(--txt)">text cos</span> read on the
right axis (−1…1); <span style="color:var(--loss)">loss</span> and <span style="color:var(--z)">drive z</span> read on
the left axis.</p>
<div class="legend">
  <span><i class="sw" style="background:var(--loss)"></i>floor loss (left axis)</span>
  <span><i class="sw" style="background:var(--img)"></i>image-emb cos → anchor (right, ↓=novel)</span>
  <span><i class="sw" style="background:var(--txt)"></i>text-emb cos (category, right)</span>
  <span><i class="sw" style="background:var(--z);opacity:.5"></i>drive z (left)</span>
  <span><i class="sw" style="background:var(--z)"></i>z* target</span>
</div>
<table>
<thead><tr><th>exp</th><th>drive floor</th><th>novelty w</th><th>final loss</th><th>final drive z</th><th>final image cos</th><th>final text cos</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table>
<div class="grid">{''.join(cards)}</div>
</div></body></html>"""

    OUT.write_text(doc)
    print("wrote", OUT, f"({len(runs)} runs, {sum(1 for r in runs if r['has_log'])} with dense logs)")


if __name__ == "__main__":
    main()
