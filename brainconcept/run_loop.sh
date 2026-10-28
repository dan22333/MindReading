#!/bin/bash
# Iterative novelty loop — SELF-BOOTSTRAPPING orchestrator.
#   Round 0: GENERATE mode-0 (pure drive, no repel) -> the canonical driver (faces). No external face set.
#            Negatives = a fixed bank of random-object generated images (neg_gen.pt), filtered of self-drivers.
#   Round r: gen K seeds (classifier-repel) -> DINOv2 gate -> label/archive -> retrain (prune overlapping negs) -> stop.
#   Stop: EXHAUSTION only (2 consecutive dry rounds at DINOv2 z>=THRESH) or MAX_ROUNDS.
#   bash run_loop.sh <MAX_ROUNDS> <K> <DRIVE_THRESH> <DRIVE_FLOOR> <JITTER> <START>
#     START=0 -> fresh (round 0 + 1..MAX);  START=r -> resume from round r (uses persisted archive+probe)
set -uo pipefail
BC=~/MindReading/brainconcept; cd "$BC"
MAXR="${1:-6}"; K="${2:-8}"; THRESH="${3:-1.5}"; ZFLOOR="${4:-2.0}"; JIT="${5:-0.0}"; START="${6:-0}"
CM=-0.0055; CS=0.0296
PY=~/braindive_venv/bin/python
NEG=vendor_conceptlab/weights/neg_gen.pt           # generated negative bank (fixed, reused every round)
mkdir -p RESULTS/loop

# GPU logging in the background
nvidia-smi dmon -s pucm -d 5 -o DT > ~/loop_gpu.log 2>&1 &
DMON=$!; trap 'kill $DMON 2>/dev/null' EXIT

run_round () {   # $1=round  $2=extra gen_round args (repel/no_repel)
  local r="$1"; shift
  echo "=== $(date) ROUND $r: generate $K seeds (drive_floor=$ZFLOOR, jitter=$JIT) $* ==="
  rm -rf vendor_conceptlab/outputs/loop_r${r}_s*          # clean stale dirs (no cross-round contamination)
  $PY gen_round.py --round $r --K $K --z $ZFLOOR --comp_mean $CM --comp_std $CS --jitter $JIT "$@" >/dev/null
  bash run_contrast.sh "$BC/loop_r${r}_runs.txt" 2
  echo "=== $(date) ROUND $r: DINOv2 held-out (docker) ==="
  local RUNS; RUNS=$(cat loop_r${r}_runs.txt | tr '\n' ' ')
  bash run_score_dino_all.sh $RUNS > ~/loop_dino_r${r}.log 2>&1
  echo "=== $(date) ROUND $r: process + label + archive ==="
  $PY loop_round.py --round $r --drive_thresh $THRESH 2>&1 | tail -12
  echo "=== $(date) ROUND $r: retrain classifier (prune overlapping negs + per-mode recall) ==="
  $PY continual_clf.py --archive weights/archive.pt --neg $NEG --out vendor_conceptlab/weights/face_probe.pt --min_auc 0.97 2>&1 | tail -18
  cp vendor_conceptlab/weights/face_probe.pt RESULTS/loop/face_probe_r${r}.pt
  cp weights/archive.pt RESULTS/loop/archive_r${r}.pt
}

if [ "$START" -le 0 ]; then
  rm -f weights/archive.pt                                # fresh bootstrap: archive starts empty
  echo "=== $(date) build-time: drop generated negatives that THEMSELVES drive the ROI ==="
  $PY - <<PY
import torch
e=torch.load("$NEG",map_location="cpu")["emb"].float(); e=e/e.norm(dim=-1,keepdim=True)
Wf=torch.as_tensor(torch.load("weights/W_faces_s1.pt",map_location="cpu")["W"],dtype=torch.float32)
Wc=torch.as_tensor(torch.load("vendor_conceptlab/weights/W_comp_s1.pt",map_location="cpu")["W"],dtype=torch.float32)
FM,FS=-0.0476,0.2003; CM,CS=-0.0055,0.0296
zf=((e@Wf.t()).mean(1)-FM)/FS; zc=((e@Wc.t()).mean(1)-CM)/CS; sel=zf-zc.clamp(min=0)
keep=sel<1.0
print(f"neg_gen self-driver filter: keep {int(keep.sum())}/{len(e)} (dropped {int((~keep).sum())} with selectivity>1.0)")
torch.save({"emb":e[keep]}, "$NEG")
PY
  echo "=== $(date) ROUND 0: bootstrap mode-0 (generate the canonical driver, NO repel) ==="
  run_round 0 --no_repel
fi

LOOP_START=$START; [ "$LOOP_START" -lt 1 ] && LOOP_START=1
DRY=0
for ((r=LOOP_START; r<=MAXR; r++)); do
  run_round $r --probe weights/face_probe.pt    # path is relative to vendor_conceptlab/ (training CWD)
  # stop: EXHAUSTION only — 2 consecutive dry rounds (no tile clears DINOv2 z>=THRESH) or MAX_ROUNDS.
  META=RESULTS/loop/round_${r}.json
  DRYR=$($PY -c "import json;print(json.load(open('$META'))['dry_round'])")
  if [ "$DRYR" = "True" ]; then DRY=$((DRY+1)); else DRY=0; fi
  if [ "$DRY" -ge 2 ]; then echo "=== $(date) STOP: 2 dry rounds -> ROI exhausted (round $r) ==="; break; fi
done

echo "=== $(date) LOOP_DONE ==="
