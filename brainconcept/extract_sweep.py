#!/usr/bin/env python
"""Parse ~/sweep.log for each sw_* cell: per-step top-3 BLIP concepts + ROI trajectory
(peak, final, whether/where it drop-stopped, # distinct concepts). Writes
RESULTS/heldout/blip_sweep.json and RESULTS/heldout/sweep_status.json.
"""
import ast, json, re
from pathlib import Path

RUNS = [f"sw_{r}_{t}" for r in ("faces", "bodies")
        for t in ("t02_l1", "t01_l1", "t00_l1", "tn01_l1", "t00_l05", "t00_l2")]
text = (Path.home() / "sweep.log").read_text(errors="ignore")

blip, status = {}, {}
for run in RUNS:
    m = re.search(rf"START {re.escape(run)} .*?(?:DONE {re.escape(run)} |\Z)", text, re.S)
    if not m:
        continue
    seg = m.group(0)
    done = f"DONE {run} " in seg
    # per-step concepts (each checkpoint = one "Adding negative class(es): [...]" line)
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
    # roi trajectory
    rois = [float(x) for x in re.findall(r"mean ROI response\): (-?[0-9.]+)", seg)]
    rois = [-x for x in rois]              # stored as -mean; flip to +mean response
    dropped = re.search(r"drop-stop STOP @ step (\d+)", seg)
    allc = set()
    for s in steps:
        allc.update(s if isinstance(s, list) else [s])
    status[run] = {
        "done": done,
        "roi_peak": round(max(rois), 3) if rois else None,
        "roi_final": round(rois[-1], 3) if rois else None,
        "dropped_at": int(dropped.group(1)) if dropped else None,
        "n_checkpoints": len(steps),
        "n_concepts": len(allc - {""}),
    }

Path("RESULTS/heldout").mkdir(parents=True, exist_ok=True)
Path("RESULTS/heldout/blip_sweep.json").write_text(json.dumps(blip, indent=1))
Path("RESULTS/heldout/sweep_status.json").write_text(json.dumps(status, indent=1))
for r, s in status.items():
    print(f"{r:20s} done={s['done']} peak={s['roi_peak']} final={s['roi_final']} "
          f"dropped@={s['dropped_at']} concepts={s['n_concepts']}")
