#!/usr/bin/env python
"""Parse the v2 training logs into per-run, per-step BLIP concept lists.
Reads ~/v2.log (batch1 7 runs) and ~/pipeline.log (3 crop-on-region runs),
splits by START/DONE markers, collects the ordered
  Adding negative class(es): [...]
lines, and writes RESULTS/heldout/blip_v2.json = {run: [per-step (list of names)]}.
"""
import ast, json, re
from pathlib import Path

LOGS = [Path.home() / "v2.log", Path.home() / "pipeline.log"]
RUNS = ["v2_faces", "v2_bodies", "v2_places", "v2_words_1tok_full", "v2_words_2tok_full",
        "v2_words_1tok_crop", "v2_words_2tok_crop", "v2_faces_crop", "v2_bodies_crop", "v2_places_crop"]

text = "\n".join(p.read_text(errors="ignore") for p in LOGS if p.exists())
out = {}
for run in RUNS:
    # slice between "START <run> " and the next "DONE <run> "
    m = re.search(rf"START {re.escape(run)} .*?DONE {re.escape(run)} ", text, re.S)
    if not m:
        continue
    seg = m.group(0)
    steps = []
    for line in re.findall(r"Adding negative class\(es\): (\[.*\])", seg):
        try:
            names = ast.literal_eval(line)
        except Exception:
            names = [line]
        # dedupe within a step, keep order
        seen, uniq = set(), []
        for n in names:
            if n not in seen:
                seen.add(n); uniq.append(n)
        steps.append(uniq if len(uniq) > 1 else (uniq[0] if uniq else ""))
    out[run] = steps

Path("RESULTS/heldout").mkdir(parents=True, exist_ok=True)
Path("RESULTS/heldout/blip_v2.json").write_text(json.dumps(out, indent=1))
for r, s in out.items():
    print(f"{r:22s} {len(s)} steps")
