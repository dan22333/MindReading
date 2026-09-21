#!/usr/bin/env python
"""Per-step EXTRACTED TERMS (the flan-t5 top-8 recurring negatives) for each
term-extraction run, from ~/te.log and ~/lam.log. Writes blip_terms.json =
{run: [ per-step list-of-terms ]}."""
import ast, json, re
from pathlib import Path

LOGS = {"te": Path.home() / "te.log", "lam": Path.home() / "lam.log"}
GROUPS = {
    "te": [f"te_{r}_{t}" for r in ("faces", "bodies") for t in ("t01", "t00", "tn01")],
    "lam": [f"lam_{r}_{t}" for r in ("faces", "bodies") for t in ("l05", "l025", "l01")],
}
out = {}
for key, runs in GROUPS.items():
    text = LOGS[key].read_text(errors="ignore") if LOGS[key].exists() else ""
    for run in runs:
        m = re.search(rf"START {re.escape(run)} .*?(?:DONE {re.escape(run)} |\Z)", text, re.S)
        if not m:
            continue
        steps = []
        for line in re.findall(r"\[negatives<-terms\] (\[.*\])", m.group(0)):
            try:
                steps.append(ast.literal_eval(line))
            except Exception:
                steps.append([line])
        out[run] = steps
        print(f"{run}: {len(steps)} term-snapshots")
Path("RESULTS/heldout/blip_terms.json").write_text(json.dumps(out, indent=1))
