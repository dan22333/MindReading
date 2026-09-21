#!/usr/bin/env python
"""Per-step BLIP captions for each lam_* run from ~/lam.log."""
import ast, json, re
from pathlib import Path
RUNS = [f"lam_{r}_{t}" for r in ("faces", "bodies") for t in ("l05", "l025", "l01")]
text = (Path.home() / "lam.log").read_text(errors="ignore")
blip = {}
for run in RUNS:
    m = re.search(rf"START {re.escape(run)} .*?(?:DONE {re.escape(run)} |\Z)", text, re.S)
    if not m:
        continue
    steps = []
    for line in re.findall(r"Adding negative class\(es\): (\[.*\])", m.group(0)):
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
    print(f"{run}: {len(steps)} steps")
Path("RESULTS/heldout/blip_lam.json").write_text(json.dumps(blip, indent=1))
