#!/usr/bin/env python
"""Parse ~/te.log: per-step BLIP captions (top-3) + status for each te_* run."""
import ast, json, re
from pathlib import Path

RUNS = [f"te_{r}_{t}" for r in ("faces", "bodies") for t in ("t01", "t00", "tn01")]
text = (Path.home() / "te.log").read_text(errors="ignore")
blip, status = {}, {}
for run in RUNS:
    m = re.search(rf"START {re.escape(run)} .*?(?:DONE {re.escape(run)} |\Z)", text, re.S)
    if not m:
        continue
    seg = m.group(0)
    steps = []
    for line in re.findall(r"Adding negative class\(es\): (\[.*\])", seg):
        try:
            names = ast.literal_eval(line)
        except Exception:
            names = [line]
        seen, uniq = set(), []
        for n in names:
            if n not in seen:
                seen.add(n); uniq.append(n)
        steps.append(uniq if len(uniq) > 1 else (uniq[0] if uniq else ""))
    blip[run] = steps
    rois = [-float(x) for x in re.findall(r"mean ROI response\): (-?[0-9.]+)", seg)]
    dropped = re.search(r"drop-stop STOP @ step (\d+)", seg)
    terms = re.findall(r"\[negatives<-terms\] (\[.*\])", seg)
    status[run] = {
        "done": f"DONE {run} " in seg,
        "roi_peak": round(max(rois), 3) if rois else None,
        "roi_final": round(rois[-1], 3) if rois else None,
        "dropped_at": int(dropped.group(1)) if dropped else None,
        "n_checkpoints": len(steps),
        "final_terms": terms[-1] if terms else None,
    }

Path("RESULTS/heldout").mkdir(parents=True, exist_ok=True)
Path("RESULTS/heldout/blip_te.json").write_text(json.dumps(blip, indent=1))
Path("RESULTS/heldout/te_status.json").write_text(json.dumps(status, indent=1))
for r, s in status.items():
    print(f"{r:16s} done={s['done']} peak={s['roi_peak']} final={s['roi_final']} "
          f"dropped@={s['dropped_at']} terms={s['final_terms']}")
