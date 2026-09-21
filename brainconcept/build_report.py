#!/usr/bin/env python
"""Assemble the held-out validation dashboard (self-contained HTML).

Consumes score_heldout.py JSONs: RESULTS/heldout/{encoder}_{roi}.json for
encoder in {ours,dino,braindive}. For each run it shows: an honesty verdict, a
trajectory chart (ours/dino/braindive z over steps), a final-score table, and a
STEP-BY-STEP image evolution (every 250-step checkpoint, thumbnailed) with the
per-step z badges and the held-out saturation point marked.
"""
import argparse, base64, io, json
from pathlib import Path
from PIL import Image

ENCS = ["ours", "dino", "braindive"]
ENC_LABEL = {"ours": "our ridge (in-sample)", "dino": "DINOv2 universal (held-out, primary)",
             "braindive": "BrainDiVE-CLIP (held-out)"}
ENC_SHORT = {"ours": "ours", "dino": "DINOv2", "braindive": "BrainDiVE"}
ENC_COLOR = {"ours": "#2563eb", "dino": "#137a33", "braindive": "#9a6510"}

ROI_DESC = {
    "faces": "face-selective cortex (FFA / OFA)",
    "bodies": "body-selective cortex (EBA)",
    "places": "scene-selective cortex (PPA / RSC)",
    "words": "visual word form area (VWFA) — 1 learnable token",
    "words2": "visual word form area (VWFA) — 2 learnable tokens (Experiment 2)",
}
ROI_TITLE = {"faces": "faces · v1 (original ConceptLab)", "bodies": "bodies · v1 (original ConceptLab)",
             "places": "places · v1 (original ConceptLab)",
             "words": "words · v1 · 1 token", "words2": "words · v1 · 2 tokens",
             "v2_faces": "faces", "v2_bodies": "bodies", "v2_places": "places",
             "v2_words_1tok_full": "words · 1 token · full-image BLIP",
             "v2_words_2tok_full": "words · 2 tokens · full-image BLIP",
             "v2_words_1tok_crop": "words · 1 token · crop BLIP",
             "v2_words_2tok_crop": "words · 2 tokens · crop BLIP",
             "v2_faces_crop": "faces · crop BLIP", "v2_bodies_crop": "bodies · crop BLIP",
             "v2_places_crop": "places · crop BLIP"}
_V2ROI = {"v2_faces": "faces", "v2_bodies": "bodies", "v2_places": "places",
          "v2_words_1tok_full": "words", "v2_words_2tok_full": "words",
          "v2_words_1tok_crop": "words", "v2_words_2tok_crop": "words",
          "v2_faces_crop": "faces", "v2_bodies_crop": "bodies", "v2_places_crop": "places"}
for _k, _r in _V2ROI.items():
    ROI_DESC[_k] = ROI_DESC[_r].split(" — ")[0] + ' — v2: fixed sentence "A photo of a {token}", 10 photos/step'

# anchor-only experiment cells: e1 (single "human face" anchor), e2 (10-phrasing averaged)
for _e, _lbl in (("e1", 'single anchor "a human face"'), ("e2", "10-phrasing averaged anchor")):
    for _lt, _lm in (("l01", "0.1"), ("l02", "0.2"), ("l04", "0.4"), ("l06", "0.6"), ("l08", "0.8")):
        _k = f"{_e}_faces_{_lt}"
        ROI_TITLE[_k] = f"faces · λ{_lm} · {_lbl} · anchor-only (no BLIP)"
        ROI_DESC[_k] = "face-selective cortex (FFA / OFA) — sole negative = CLIP(human face), drive ROI, λ=" + _lm

# automatic-pipeline cells: auto_<roi>_<floor>_<lambda>  (all-nouns, neg_top_k=0, no anchor)
for _roi in ("faces", "bodies"):
    for _ft, _fl in (("f00", "0.0"), ("fn01", "−0.1")):
        for _lt, _lm in (("l01", "0.1"), ("l025", "0.25")):
            _k = f"auto_{_roi}_{_ft}_{_lt}"
            ROI_TITLE[_k] = f"{_roi} · λ{_lm} · floor {_fl} · AUTO (all-nouns, no cap, no anchor)"
            ROI_DESC[_k] = ROI_DESC[_roi].split(" — ")[0] + f" — automatic pipeline: all-nouns extraction, neg_top_k=0, no anchor, λ={_lm}, floor {_fl}"

# high-pressure comparison cells (faces λ0.1, fixed extractor)
ROI_TITLE["hp_w_faces_l01"] = "faces · λ0.1 · neg_weight 2.0 (MORE flee pressure)"
ROI_TITLE["hp_a_faces_l01"] = "faces · λ0.1 · CATEGORY-ANCHOR (flee the face direction)"
ROI_TITLE["hp_wa_faces_l01"] = "faces · λ0.1 · weight 1.5 + anchor (both)"
for _k, _t in (("hp_w_faces_l01", "neg_weight=2.0"), ("hp_a_faces_l01", "category_anchor='a photo of a face', weight 1.0"),
               ("hp_wa_faces_l01", "neg_weight=1.5 + anchor weight 1.0")):
    ROI_DESC[_k] = ROI_DESC["faces"].split(" — ")[0] + f" — high-pressure test: {_t}, λ=0.1, fixed extractor"

# fixed-extractor cells: fx_<roi>_<lambda>  (main-2 extractor that flees the human subject)
for _roi, _lm, _tag in (("faces", "0.1", "l01"), ("faces", "0.25", "l025"), ("bodies", "0.5", "l05")):
    _k = f"fx_{_roi}_{_tag}"
    ROI_TITLE[_k] = f"{_roi} · λ {_lm} · FIXED extractor (flees the person)"
    ROI_DESC[_k] = ROI_DESC[_roi].split(" — ")[0] + f" — corrected main-2 extractor (captures people), λ={_lm}, floor 0.0"
# no-limit cells: nl_<roi>  (neg_top_k=0, accumulate every negative forever)
for _roi, _lm, _tag in (("faces", "0.1", "l01"), ("faces", "0.25", "l025"), ("bodies", "0.5", "l05")):
    _k = f"nl_{_roi}_{_tag}"
    ROI_TITLE[_k] = f"{_roi} · λ {_lm} · NO-LIMIT negatives"
    ROI_DESC[_k] = ROI_DESC[_roi].split(" — ")[0] + f" — neg_top_k=0 (accumulate all negatives, never drop), λ={_lm}"

# lambda-sweep cells: lam_<roi>_<lambda>  (term negatives, floor 0.0, λ<1)
for _roi in ("faces", "bodies"):
    for _tag, _lm in (("l05", "0.5"), ("l025", "0.25"), ("l01", "0.1")):
        _k = f"lam_{_roi}_{_tag}"
        ROI_TITLE[_k] = f"{_roi} · λ {_lm} · term-negatives"
        ROI_DESC[_k] = ROI_DESC[_roi].split(" — ")[0] + f" — λ-sweep, pos_to_neg_loss_factor={_lm}, floor 0.0, term-extraction"

# term-extraction cells: te_<roi>_<floor>  (clean term negatives, top-8 recurring, λ=1)
for _roi in ("faces", "bodies"):
    for _tag, _fl in (("t01", "0.1"), ("t00", "0.0"), ("tn01", "−0.1")):
        _k = f"te_{_roi}_{_tag}"
        ROI_TITLE[_k] = f"{_roi} · floor {_fl} · term-negatives (λ1)"
        ROI_DESC[_k] = ROI_DESC[_roi].split(" — ")[0] + f" — term-extraction, top-8 recurring negatives, min_cosine_thr={_fl}, λ=1"

# sweep cells: sw_<roi>_<floor>_<lambda>  (drop-stop long runs)
_SW = [("t02_l1", "0.2", "1"), ("t01_l1", "0.1", "1"), ("t00_l1", "0.0", "1"),
       ("tn01_l1", "−0.1", "1"), ("t00_l05", "0.0", "0.5"), ("t00_l2", "0.0", "2")]
for _roi in ("faces", "bodies"):
    for _tag, _fl, _lm in _SW:
        _k = f"sw_{_roi}_{_tag}"
        ROI_TITLE[_k] = f"{_roi} · floor {_fl} · λ {_lm}"
        ROI_DESC[_k] = ROI_DESC[_roi].split(" — ")[0] + f" — drop-stop run, min_cosine_thr={_fl}, λ={_lm}"


def load(roi, hd):
    d = {}
    for e in ENCS:
        p = hd / f"{e}_{roi}.json"
        if p.exists():
            d[e] = json.loads(p.read_text())
    return d


def thumb_b64(path, maxw=1180, q=72):
    if not (path and Path(path).exists()):
        return ""
    im = Image.open(path).convert("RGB")
    if im.width > maxw:
        im = im.resize((maxw, max(1, round(im.height * maxw / im.width))))
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=q)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def sat_step(dino_obj):
    """Step where held-out DINOv2 z first reaches 95% of its max (honest saturation)."""
    if not dino_obj:
        return None
    st = dino_obj["steps"]
    zs = sorted([(int(k), v["z_mean"]) for k, v in st.items()])
    zmax = max(z for _, z in zs)
    for s, z in zs:
        if z >= 0.95 * zmax:
            return s
    return zs[-1][0]


def sparkline(series_by_enc, sat=None, w=560, h=150, pad=26):
    allpts = [z for s in series_by_enc.values() for _, z in s]
    if not allpts:
        return ""
    lo, hi = min(allpts + [0]), max(allpts + [0])
    steps = sorted({st for s in series_by_enc.values() for st, _ in s})
    smin, smax = min(steps), max(steps)

    def X(st): return pad + (0 if smax == smin else (st - smin) / (smax - smin)) * (w - 2 * pad)
    def Y(z): return h - pad - (0 if hi == lo else (z - lo) / (hi - lo)) * (h - 2 * pad)

    parts = [f'<svg viewBox="0 0 {w} {h}" width="100%" style="background:#f7f8fa;border:1px solid #e2e5ea;border-radius:8px">']
    y0 = Y(0)
    parts.append(f'<line x1="{pad}" y1="{y0:.0f}" x2="{w-pad}" y2="{y0:.0f}" stroke="#c9ced6" stroke-dasharray="3 3"/>')
    parts.append(f'<text x="{pad}" y="{y0-4:.0f}" fill="#888" font-size="10">z=0 (natural-image baseline)</text>')
    if sat is not None and smax > smin:
        xs = X(sat)
        parts.append(f'<line x1="{xs:.0f}" y1="{pad-6}" x2="{xs:.0f}" y2="{h-pad}" stroke="#1a8a3a" stroke-width="1" stroke-dasharray="2 3" opacity="0.7"/>')
        parts.append(f'<text x="{xs+3:.0f}" y="{pad+2}" fill="#137a33" font-size="9">held-out saturates (~{sat})</text>')
    for e, s in series_by_enc.items():
        if not s:
            continue
        pts = " ".join(f"{X(st):.1f},{Y(z):.1f}" for st, z in s)
        parts.append(f'<polyline fill="none" stroke="{ENC_COLOR[e]}" stroke-width="2" points="{pts}"/>')
        st, z = s[-1]
        parts.append(f'<circle cx="{X(st):.1f}" cy="{Y(z):.1f}" r="3" fill="{ENC_COLOR[e]}"/>')
    parts.append(f'<text x="{pad}" y="14" fill="#555" font-size="11">z-scored ROI response over optimization steps</text>')
    parts.append("</svg>")
    return "".join(parts)


def verdict(d):
    if "ours" not in d or "dino" not in d:
        return ("incomplete", "warn", "held-out encoder missing")
    def final_z(e):
        st = d[e]["steps"]; k = max(st, key=lambda x: int(x)); return st[k]["z_mean"]
    o, dn = final_z("ours"), final_z("dino")
    drop = o - dn; rel = drop / max(o, 1e-6)
    if dn >= 2.0 and rel <= 0.35:
        return ("REAL driver", "good",
                f"held-out DINOv2 z={dn:+.2f} confirms it — {rel*100:.0f}% of our-encoder signal transfers to an independent non-CLIP backbone (drop {drop:+.2f})")
    if dn >= 1.0:
        return ("partly real", "warn",
                f"held-out DINOv2 z={dn:+.2f} (our z={o:+.2f}, drop {drop:+.2f}, {rel*100:.0f}%) — real signal plus some encoder-specific inflation")
    return ("likely HACKED", "bad",
            f"our z={o:+.2f} but held-out DINOv2 z={dn:+.2f} (drop {drop:+.2f}) — signal does NOT transfer to an independent encoder")


def zbadges(d, k):
    """z chips for all encoders at step key k."""
    chips = ""
    for e in ENCS:
        if e in d and k in d[e]["steps"]:
            z = d[e]["steps"][k]["z_mean"]
            chips += f"<span class='zchip' style='color:{ENC_COLOR[e]};border-color:{ENC_COLOR[e]}44'>{ENC_SHORT[e]} {z:+.1f}</span>"
    return chips


def evolution(d, img_dir, sat, blip, terms=None):
    """Row per step: label + BLIP caption + extracted terms + z badges + thumbnailed grid."""
    ref = d.get("ours") or next(iter(d.values()))
    keys = sorted(ref["steps"], key=lambda x: int(x))
    rows = ""
    for i, k in enumerate(keys):
        step = int(k)
        fname = ref["steps"][k]["file"]
        img = thumb_b64(Path(img_dir) / fname) if img_dir else ""
        lbl = "init" if step == 0 else f"step {step}"
        mark = "<span class='satmark'>← honest peak</span>" if (sat is not None and step == sat) else ""
        name = blip[i] if blip and i < len(blip) else ""
        if isinstance(name, list):
            caps = "<br>".join(f"“{c}”" for c in name)
        else:
            caps = f"“{name}”" if name else "—"
        # LLM (flan-t5) side: new terms this step + cumulative fled set
        cum = terms[i] if terms and i < len(terms) else None
        if cum:
            cum = cum if isinstance(cum, list) else [cum]
            prev = terms[i - 1] if (terms and 0 < i <= len(terms) - 1 and isinstance(terms[i - 1], list)) else []
            new = [t for t in cum if t not in prev]
            llm_html = (f"<div><b>new this step:</b> {', '.join(new) if new else '—'}</div>"
                        f"<div class='termscum'>cumulative fled (top-8): {', '.join(cum)}</div>")
        else:
            llm_html = "<span class='mut'>— (no term-extraction in this run)</span>"
        pair = (f"<div class='pair'>"
                f"<div class='side blipside'><span class='mtag'>BLIP-2 · VLM (raw caption)</span>{caps}</div>"
                f"<div class='side llmside'><span class='mtag'>flan-t5 · LLM (extracted terms)</span>{llm_html}</div>"
                f"</div>")
        rows += f"""<div class="evrow{' sat' if (sat is not None and step==sat) else ''}">
  <div class="evmeta"><span class="evstep">{lbl}</span>{mark}<div class="evz">{zbadges(d,k)}</div></div>
  {pair}
  <img loading="lazy" src="{img}">
</div>"""
    return rows


def pending_section(roi, img_dir, note, blip_map=None, terms_map=None):
    """Run whose images exist but held-out scoring hasn't run yet: full step strip
    (image + BLIP caption ‖ flan-t5 terms per step), no z scores."""
    def sk(p):
        s = p.stem
        if "init" in s:
            return 0
        digs = "".join(c if c.isdigit() else " " for c in s).split()
        return int(digs[0]) if digs else 10 ** 9
    blip = (blip_map or {}).get(roi)
    terms = (terms_map or {}).get(roi)
    rows = ""
    if img_dir and Path(img_dir).exists():
        for i, g in enumerate(sorted(Path(img_dir).glob("*images.jpeg"), key=sk)):
            step = sk(g)
            lbl = "init" if step == 0 else f"step {step}"
            name = blip[i] if blip and i < len(blip) else ""
            caps = "<br>".join(f"“{c}”" for c in name) if isinstance(name, list) else (f"“{name}”" if name else "—")
            cum = terms[i] if terms and i < len(terms) else None
            if cum:
                cum = cum if isinstance(cum, list) else [cum]
                prev = terms[i - 1] if (0 < i and isinstance(terms[i - 1], list)) else []
                new = [t for t in cum if t not in prev]
                llm = (f"<div><b>new this step:</b> {', '.join(new) if new else '—'}</div>"
                       f"<div class='termscum'>cumulative fled ({len(cum)}): {', '.join(cum)}</div>")
            else:
                llm = "<span class='mut'>—</span>"
            pair = (f"<div class='pair'><div class='side blipside'><span class='mtag'>BLIP-2 · VLM (raw caption)</span>{caps}</div>"
                    f"<div class='side llmside'><span class='mtag'>flan-t5 · LLM (extracted terms)</span>{llm}</div></div>")
            rows += (f"<div class='evrow'><div class='evmeta'><span class='evstep'>{lbl}</span></div>"
                     f"{pair}<img loading='lazy' src='{thumb_b64(g)}'></div>")
    return f"""
<h2>{ROI_TITLE.get(roi, roi)} <span class="pill run">scoring pending</span></h2>
<div class="card">
  <p class="roidesc">{ROI_DESC.get(roi,'')}</p>
  <p class="cap" style="margin:0 0 10px">{note}</p>
  <div class="evo">{rows}</div>
</div>"""


def roi_section(roi, hd, img_dir, blip_map, terms_map=None):
    d = load(roi, hd)
    if not d:
        return ""
    series = {e: sorted([(int(k), v["z_mean"]) for k, v in obj["steps"].items()]) for e, obj in d.items()}
    lab, css, vtext = verdict(d)
    sat = sat_step(d.get("dino"))
    # novelty (cat_sim) from the ours json final step, if present
    csim = None
    if "ours" in d:
        _st = d["ours"]["steps"]; _k = max(_st, key=lambda x: int(x)); csim = _st[_k].get("cat_sim")
    csim_html = (f"<span class='zchip' style='color:#7c3aed;border-color:#7c3aed55'>novelty cat_sim {csim:+.3f} "
                 f"(lower = less category-typical)</span>" if csim is not None else "")
    rows = ""
    for e in ENCS:
        if e in d:
            fz = series[e][-1][1] if series[e] else float("nan")
            mx = max(z for _, z in series[e]) if series[e] else float("nan")
            rows += (f"<tr><td><span style='color:{ENC_COLOR[e]}'>&#9632;</span> {ENC_LABEL[e]}</td>"
                     f"<td class='n'>{fz:+.2f}</td><td class='n'>{mx:+.2f}</td></tr>")
    sat_txt = (f"Held-out signal saturates by <b>step {sat}</b> — later steps add gain our encoder sees "
               f"but the independent DINOv2 does not (mild inflation), and the image keeps drifting via the "
               f"novelty loop. For the strongest honest driver, stop near here." if sat else "")
    return f"""
<h2>{ROI_TITLE.get(roi, roi)} <span class="pill {css}">{lab}</span></h2>
<div class="card">
  <p class="roidesc">{ROI_DESC.get(roi,'')}</p>
  <p class="verdict {css}">{vtext}</p>
  <p style="margin:0 0 10px">{csim_html}</p>
  <div class="grid2">
    <div>{sparkline(series, sat)}</div>
    <div><table><tr><th>encoder</th><th>final z</th><th>peak z</th></tr>{rows}</table>
      <p class="cap">z = SD above the mean natural-image response. High on our ridge but low on DINOv2 = encoder hacking.</p></div>
  </div>
  <p class="satnote">{sat_txt}</p>
  <p class="cap" style="margin-top:12px"><b>Image evolution — every 250-step checkpoint.</b> Each row: the generated samples,
  the BLIP-2 caption(s), the new + cumulative extracted terms, and the encoder z at that step.</p>
    <div class="evo">{evolution(d, img_dir, sat, blip_map.get(roi), (terms_map or {}).get(roi))}</div>
</div>"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--heldout_dir", default="RESULTS/heldout")
    ap.add_argument("--img_dirs", nargs="*", default=[])
    ap.add_argument("--rois", nargs="+", default=["faces", "bodies", "places", "words", "words2"])
    ap.add_argument("--rois_file", default=None, help="file with one run per line (overrides --rois)")
    ap.add_argument("--img_base", default=None, help="if set, img dir for each run = img_base/<run>")
    ap.add_argument("--pending", nargs="*", default=[], help="roi=imgdir::note for scored-pending runs")
    ap.add_argument("--pending_note", default="New v2 protocol run — images ready; held-out validation scores are computed when the full batch finishes.")
    ap.add_argument("--artifact", action="store_true")
    ap.add_argument("--summary", default=None, help="override the top summary card (raw HTML)")
    ap.add_argument("--methods_file", default=None, help="HTML file inserted before the run sections")
    ap.add_argument("--title", default="brainconcept — held-out validation")
    ap.add_argument("--out", default="RESULTS/heldout_report.html")
    args = ap.parse_args()
    hd = Path(args.heldout_dir)
    if args.rois_file and Path(args.rois_file).exists():
        args.rois = [ln.strip() for ln in Path(args.rois_file).read_text().splitlines() if ln.strip()]
    img_map = dict(p.split("=", 1) for p in args.img_dirs)
    if args.img_base:
        for r in args.rois:
            img_map.setdefault(r, f"{args.img_base}/{r}")
    blip_path = hd / "blip_negatives.json"
    blip_map = json.loads(blip_path.read_text()) if blip_path.exists() else {}
    terms_path = hd / "blip_terms.json"
    terms_map = json.loads(terms_path.read_text()) if terms_path.exists() else {}
    pend = ""
    for spec in args.pending:
        roi, rest = spec.split("=", 1)
        imgdir, _, note = rest.partition("::")
        pend += pending_section(roi, imgdir, note or args.pending_note, blip_map, terms_map)
    methods = Path(args.methods_file).read_text() if args.methods_file and Path(args.methods_file).exists() else ""
    body = methods + pend + "".join(roi_section(r, hd, img_map.get(r), blip_map, terms_map) for r in args.rois)

    summary = args.summary if args.summary else ('<div class="card"><b>v2 protocol — 10 runs, all held-out validated.</b> Fixed sentence '
               '<i>"A photo of a {token}"</i>, 10 photos/step, BLIP-2 on the full image vs 2×2 crops, 1 vs 2 tokens (words). '
               'Every decoded image is re-scored by DINOv2 (a non-CLIP encoder that can\'t be hacked through CLIP). '
               '<b>9/10 REAL, 1 partly real.</b>'
               '<br><br><b>1. The fixed sentence made faces more honest.</b> v1 faces drifted to <i>bees</i> with a +0.90 '
               'our-vs-DINOv2 gap; v2 faces stay on <i>faces</i> (girl / clown / mickey mouse) and the gap collapses to '
               '−0.03 (ours +3.64, DINOv2 +3.67). Dropping the painting templates kept the token on the encoder\'s '
               'photographic distribution.'
               '<br><b>2. Crop-BLIP did NOT strengthen the driver.</b> It surfaced richer <i>local</i> concepts '
               '(letters/text for words — "pastry with a letter on it", "jar with the word geco"; bees for faces) but '
               'held-out z was equal-or-slightly-lower than full-image on every ROI. Good for interpretability, not for drive.'
               '<br><b>3. 2 tokens ≈ 1 token (words).</b> DINOv2 +4.23 (2-tok) vs +4.75 (1-tok). No gain from extra capacity.'
               '<br><b>Winner: full-image BLIP + fixed sentence + 1 token.</b> Words remain the strongest held-out driver (DINOv2 +4.75).</div>')

    head = (f'<title>{args.title}</title>' if args.artifact
            else f'<!doctype html><meta charset="utf-8"><title>{args.title}</title>')
    html = f"""{head}
<style>
body{{margin:0;background:#ffffff;color:#141414;font:15px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif}}
.wrap{{max-width:940px;margin:0 auto;padding:30px 20px 70px}}
h1{{font-size:25px;margin:0 0 4px;color:#000}} h2{{font-size:20px;margin:34px 0 8px;color:#1a1a1a}}
.sub{{color:#555;margin:0 0 18px}}
.card{{background:#f7f8fa;border:1px solid #e2e5ea;border-radius:12px;padding:16px 18px;margin:12px 0}}
.grid2{{display:grid;grid-template-columns:1fr 1fr;gap:16px;align-items:center}}
@media(max-width:720px){{.grid2{{grid-template-columns:1fr}}}}
table{{border-collapse:collapse;width:100%}} th,td{{text-align:left;padding:6px 8px;border-bottom:1px solid #e2e5ea}}
th{{color:#555;font-size:13px}} td.n{{font-variant-numeric:tabular-nums;font-weight:600}}
img{{max-width:100%;border-radius:6px;border:1px solid #e2e5ea;display:block}}
.cap{{color:#555;font-size:13px}}
.chain{{color:#333}}
.pill{{display:inline-block;padding:2px 10px;border-radius:20px;font-size:13px;font-weight:600;vertical-align:middle}}
.pill.good{{background:rgba(26,138,58,.14);color:#137a33}} .pill.warn{{background:rgba(181,121,26,.16);color:#9a6510}}
.pill.bad{{background:rgba(192,57,43,.14);color:#b0281c}}
.roidesc{{color:#777;font-size:12px;text-transform:uppercase;letter-spacing:.06em;margin:0 0 8px}}
.verdict{{font-size:14px;margin:0 0 12px}} .verdict.good{{color:#137a33}} .verdict.warn{{color:#9a6510}} .verdict.bad{{color:#b0281c}}
.satnote{{color:#444;font-size:13px;background:#eef6ef;border-left:3px solid #1a8a3a;border-radius:6px;padding:9px 13px;margin:12px 0}}
.box{{background:#eef3fe;border-left:3px solid #2563eb;border-radius:6px;padding:12px 16px;margin:12px 0;font-size:14px}}
details{{margin-top:8px}} summary{{cursor:pointer;color:#555;font-size:13px;padding:4px 0}}
.evo{{display:flex;flex-direction:column;gap:6px;margin-top:8px}}
.evrow{{display:block;padding:10px 0 12px;border-top:1px solid #eceef1;page-break-inside:avoid;break-inside:avoid}}
.evrow.sat{{background:rgba(26,138,58,.06)}}
.evmeta{{margin:0 0 6px}}
.evstep{{font-weight:700;font-size:13px;color:#000}} .satmark{{color:#137a33;font-size:11px;margin-left:6px}}
.blip{{color:#7a5e12;font-size:12px;margin:4px 0 0}} .blip b{{color:#5e480c}}
.terms{{color:#15521f;font-size:12.5px;margin:3px 0 0}} .terms b{{color:#0d3a16}}
.termscum{{color:#5c7a63;font-size:10.5px;margin:2px 0 0}}
.pair{{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin:5px 0 4px}}
.side{{border-radius:6px;padding:7px 10px;font-size:12px;line-height:1.5}}
.blipside{{background:#fff7e6;border:1px solid #f0e0b8;color:#5e480c}}
.llmside{{background:#eaf5ec;border:1px solid #cbe6cf;color:#123f1c}}
.mtag{{display:block;font-size:9.5px;text-transform:uppercase;letter-spacing:.05em;font-weight:700;margin:0 0 3px;opacity:.75}}
.mut{{color:#999}}
@media(max-width:640px){{.pair{{grid-template-columns:1fr}}}}
.evz{{display:flex;flex-wrap:wrap;gap:4px;margin:5px 0 0}}
.zchip{{font-size:10.5px;font-variant-numeric:tabular-nums;border:1px solid;border-radius:5px;padding:1px 5px}}
</style>
<div class="wrap">
<h1>{args.title}</h1>
<p class="sub">Every decoded image is re-scored by two independent encoders on identical pixels.</p>
<div class="box"><b>The test.</b> <span style="color:#2563eb">&#9632; our ridge</span> is the encoder the token was
optimized against (CLIP ViT-L/14 → fMRI). <span style="color:#137a33">&#9632; DINOv2 universal</span> is the primary
honesty check — a totally different backbone that can't be fooled through CLIP.
<span style="color:#9a6510">&#9632; BrainDiVE-CLIP</span> is a secondary check. All three score the SAME decoded images,
z-scored against natural NSD images. <b>Agreement = real driver; our-ridge-only = encoder hacking.</b></div>
{summary}
{body}
</div>"""
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(html)
    print(f"[done] wrote {args.out} ({len(html)//1024} KB)")


if __name__ == "__main__":
    main()
