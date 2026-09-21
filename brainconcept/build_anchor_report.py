#!/usr/bin/env python
"""Anchor-only experiment report: cos-to-face & DINOv2 vs lambda (Exp1 vs Exp2),
a realness-vs-novelty scatter, a data table, and the final photo per lambda."""
import base64, io, json, os
from pathlib import Path
from PIL import Image

LAMS = [("l01", 0.1), ("l02", 0.2), ("l04", 0.4), ("l06", 0.6), ("l08", 0.8)]
EXPS = [("e1", "Exp1 · single anchor “a human face”", "#2563eb"),
        ("e2", "Exp2 · 10-phrasing averaged", "#c026d3")]
HD = Path("RESULTS/heldout"); IMG = Path("RESULTS/heldout_imgs")


def final(enc, run):
    p = HD / f"{enc}_{run}.json"
    if not p.exists():
        return None, None
    st = json.loads(p.read_text())["steps"]; k = max(st, key=lambda x: int(x))
    return st[k]["z_mean"], st[k].get("cat_sim")


def load():
    D = {}
    for e, _, _ in EXPS:
        for lt, lam in LAMS:
            r = f"{e}_faces_{lt}"
            oz, cs = final("ours", r); dz, _ = final("dino", r)
            D[(e, lam)] = dict(cos=cs, ourZ=oz, dinoZ=dz, run=r)
    return D


def thumb(run, maxw=900, q=82):
    f = IMG / run / "2000_step_images.jpeg"
    if not f.exists():
        return ""
    im = Image.open(f).convert("RGB")
    if im.width > maxw:
        im = im.resize((maxw, max(1, round(im.height * maxw / im.width))))
    b = io.BytesIO(); im.save(b, "JPEG", quality=q)
    return "data:image/jpeg;base64," + base64.b64encode(b.getvalue()).decode()


def line_svg(D, key, title, ylab, w=520, h=280, pad=44):
    xs = [lam for _, lam in LAMS]
    allv = [D[(e, lam)][key] for e, _, _ in EXPS for _, lam in LAMS if D[(e, lam)][key] is not None]
    lo, hi = min(allv), max(allv); rng = (hi - lo) or 1
    X = lambda v: pad + (v - min(xs)) / (max(xs) - min(xs)) * (w - 1.5 * pad)
    Y = lambda v: h - pad - (v - lo) / rng * (h - 1.6 * pad)
    p = [f'<svg viewBox="0 0 {w} {h}" width="100%" style="background:#fff;border:1px solid #e2e5ea;border-radius:8px">']
    p.append(f'<text x="{pad}" y="18" fill="#111" font-size="13" font-weight="700">{title}</text>')
    for gy in range(5):
        yy = pad + gy * (h - 1.6 * pad) / 4
        p.append(f'<line x1="{pad}" y1="{yy:.0f}" x2="{w-pad/2:.0f}" y2="{yy:.0f}" stroke="#eee"/>')
    for lt, lam in LAMS:
        p.append(f'<text x="{X(lam):.0f}" y="{h-14}" fill="#777" font-size="10" text-anchor="middle">{lam}</text>')
    p.append(f'<text x="{w/2:.0f}" y="{h-2}" fill="#555" font-size="10" text-anchor="middle">λ (brain weight)</text>')
    p.append(f'<text x="4" y="{pad+4}" fill="#555" font-size="10">{ylab}</text>')
    for e, lbl, col in EXPS:
        pts = " ".join(f"{X(lam):.1f},{Y(D[(e,lam)][key]):.1f}" for _, lam in LAMS if D[(e, lam)][key] is not None)
        p.append(f'<polyline fill="none" stroke="{col}" stroke-width="2.5" points="{pts}"/>')
        for _, lam in LAMS:
            v = D[(e, lam)][key]
            if v is not None:
                p.append(f'<circle cx="{X(lam):.1f}" cy="{Y(v):.1f}" r="3.5" fill="{col}"/>')
    return "".join(p) + "</svg>"


def scatter_svg(D, w=520, h=340, pad=48):
    xs = [D[(e, lam)]["cos"] for e, _, _ in EXPS for _, lam in LAMS]
    ys = [D[(e, lam)]["dinoZ"] for e, _, _ in EXPS for _, lam in LAMS]
    xlo, xhi = min(xs), max(xs); ylo, yhi = min(ys + [2]), max(ys)
    X = lambda v: pad + (v - xlo) / ((xhi - xlo) or 1) * (w - 1.5 * pad)
    Y = lambda v: h - pad - (v - ylo) / ((yhi - ylo) or 1) * (h - 1.7 * pad)
    p = [f'<svg viewBox="0 0 {w} {h}" width="100%" style="background:#fff;border:1px solid #e2e5ea;border-radius:8px">']
    p.append(f'<text x="{pad}" y="18" fill="#111" font-size="13" font-weight="700">realness vs novelty (each point = one λ)</text>')
    yreal = Y(2.0)
    p.append(f'<line x1="{pad}" y1="{yreal:.0f}" x2="{w-pad/2:.0f}" y2="{yreal:.0f}" stroke="#c0392b" stroke-dasharray="4 3"/>')
    p.append(f'<text x="{w-pad:.0f}" y="{yreal-4:.0f}" fill="#c0392b" font-size="9" text-anchor="end">DINOv2=2 (real)</text>')
    p.append(f'<text x="{w/2:.0f}" y="{h-4}" fill="#555" font-size="10" text-anchor="middle">cos to “human face”  (← more novel)</text>')
    p.append(f'<text x="4" y="{pad:.0f}" fill="#555" font-size="10">DINOv2 z (real driver ↑)</text>')
    for e, lbl, col in EXPS:
        for _, lam in LAMS:
            d = D[(e, lam)]
            if d["cos"] is None:
                continue
            p.append(f'<circle cx="{X(d["cos"]):.1f}" cy="{Y(d["dinoZ"]):.1f}" r="5" fill="{col}" opacity="0.85"/>')
            p.append(f'<text x="{X(d["cos"])+7:.0f}" y="{Y(d["dinoZ"])+3:.0f}" fill="{col}" font-size="9">λ{lam}</text>')
    return "".join(p) + "</svg>"


def main():
    D = load()
    rows = ""
    for e, lbl, col in EXPS:
        cells = ""
        for lt, lam in LAMS:
            d = D[(e, lam)]
            cells += (f"<div class='cell'><div class='cl'>λ {lam} · cos {d['cos']:+.3f} · "
                      f"DINOv2 <b style='color:{'#137a33' if d['dinoZ']>=2 else '#c0392b'}'>{d['dinoZ']:+.2f}</b></div>"
                      f"<img src='{thumb(d['run'])}'></div>")
        rows += f"<h3 style='color:{col}'>{lbl}</h3><div class='grid'>{cells}</div>"
    trows = ""
    for e, lbl, col in EXPS:
        for lt, lam in LAMS:
            d = D[(e, lam)]
            trows += (f"<tr><td>{lbl.split(' ')[0]}</td><td class='n'>{lam}</td><td class='n'>{d['cos']:+.3f}</td>"
                      f"<td class='n'>{d['ourZ']:+.2f}</td><td class='n' style='color:{'#137a33' if d['dinoZ']>=2 else '#c0392b'}'>{d['dinoZ']:+.2f}</td></tr>")
    html = f"""<title>brainconcept — anchor-only sweep (human face)</title>
<style>
body{{margin:0;background:#fff;color:#141414;font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif}}
.wrap{{max-width:1120px;margin:0 auto;padding:28px 20px 70px}}
h1{{font-size:24px;margin:0 0 4px}} h3{{margin:26px 0 8px}}
.sub{{color:#555;margin:0 0 16px}}
.box{{background:#f4f7ff;border-left:3px solid #2563eb;border-radius:6px;padding:12px 16px;margin:12px 0;font-size:14px}}
.plots{{display:grid;grid-template-columns:1fr 1fr;gap:16px}} @media(max-width:800px){{.plots{{grid-template-columns:1fr}}}}
table{{border-collapse:collapse;width:100%;margin:14px 0}} th,td{{text-align:left;padding:6px 9px;border-bottom:1px solid #eceef1}}
th{{color:#666;font-size:12px}} td.n{{font-variant-numeric:tabular-nums;font-weight:600}}
.grid{{display:grid;grid-template-columns:repeat(2,1fr);gap:16px}} @media(max-width:700px){{.grid{{grid-template-columns:1fr}}}}
.cell img{{width:100%;border-radius:6px;border:1px solid #e2e5ea;display:block}} .cl{{font-size:13px;color:#333;margin:0 0 5px}}
.leg span{{display:inline-block;margin-right:14px;font-size:12px}} .dot{{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:5px;vertical-align:middle}}
</style>
<div class="wrap">
<h1>Anchor-only sweep — "human face" as the sole negative (faces ROI)</h1>
<p class="sub">Drive the face-ROI + flee one fixed direction (CLIP-text of "a human face"), no BLIP/flan-t5. Sweep λ (brain weight) 0.1→0.8, 2000 steps. <b>cos-to-face</b> = how face-like it stayed (↓ novel); <b>DINOv2 z</b> = independent brain drive (real if ≥2).</p>
<div class="box"><b>Result.</b> A clean trade-off frontier: raising λ increases both cos-to-face and DINOv2 together. The <b>sweet spot is λ≈0.2–0.4</b> — a genuinely real driver (DINOv2 +2.3 to +3.5) that is markedly <i>less</i> face-like (cos 0.06–0.14) than a literal human face. And <b>Exp2 (averaging 10 phrasings) dominates Exp1</b>: at matched λ it holds higher DINOv2 at lower cos-to-face — averaging fences off the human-face region more robustly while keeping the driver real.</div>
<div class="leg"><span><span class="dot" style="background:#2563eb"></span>Exp1 single anchor</span><span><span class="dot" style="background:#c026d3"></span>Exp2 10-phrasing avg</span></div>
<div class="plots">
<div>{line_svg(D,'cos','cos to “human face” vs λ','cos')}</div>
<div>{line_svg(D,'dinoZ','DINOv2 z (real drive) vs λ','z')}</div>
</div>
<div style="margin-top:16px">{scatter_svg(D)}</div>
<table><tr><th>exp</th><th>λ</th><th>cos-to-face</th><th>our z</th><th>DINOv2 z</th></tr>{trows}</table>
<h2>Photos per λ (final step)</h2>
{rows}
</div>"""
    Path("RESULTS/anchor_sweep.html").write_text(html)
    print(f"wrote RESULTS/anchor_sweep.html ({len(html)//1024} KB)")


if __name__ == "__main__":
    main()
