#!/bin/bash
# Iterative novelty loop — master orchestrator (one command, auto phase/model switching).
# Round 0: archive={faces}, train base forbidden-classifier.
# Round r: gen K seeds (classifier-repel) -> DINOv2 (docker) -> process/label/archive -> retrain -> stop.
#   bash run_loop.sh <MAX_ROUNDS> <K> <DRIVE_THRESH> <JITTER>
set -uo pipefail
BC=~/MindReading/brainconcept; cd "$BC"
MAXR="${1:-5}"; K="${2:-8}"; THRESH="${3:-1.5}"; JIT="${4:-0.0}"; START="${5:-1}"
CM=-0.0055; CS=0.0296
PY=~/braindive_venv/bin/python
mkdir -p RESULTS/loop

# GPU logging in the background
nvidia-smi dmon -s pucm -d 5 -o DT > ~/loop_gpu.log 2>&1 &
DMON=$!; trap 'kill $DMON 2>/dev/null' EXIT

if [ "$START" -le 1 ]; then
  echo "=== $(date) ROUND 0: base classifier (faces vs clean negatives) ==="
  $PY - <<PY
import torch
fr = torch.load("weights/face_ref_emb.pt", map_location="cpu")["emb"]
torch.save({"faces": fr}, "weights/archive.pt")
print("archive.pt init: faces", tuple(fr.shape))
PY
  $PY continual_clf.py --archive weights/archive.pt --neg weights/neg_emb.pt --out vendor_conceptlab/weights/face_probe.pt --min_auc 0.97 2>&1 | tail -15
else
  echo "=== $(date) RESUME from round $START (using persisted archive + probe) ==="
fi

DRY=0
for ((r=START; r<=MAXR; r++)); do
  echo "=== $(date) ROUND $r: generate $K seeds (jitter=$JIT) ==="
  $PY gen_round.py --round $r --K $K --probe weights/face_probe.pt --comp_mean $CM --comp_std $CS --jitter $JIT >/dev/null
  bash run_contrast.sh "$BC/loop_r${r}_runs.txt" 2

  echo "=== $(date) ROUND $r: DINOv2 held-out (docker) ==="
  RUNS=$(cat loop_r${r}_runs.txt | tr '\n' ' ')
  bash run_score_dino_all.sh $RUNS > ~/loop_dino_r${r}.log 2>&1

  echo "=== $(date) ROUND $r: process + label + archive ==="
  $PY loop_round.py --round $r --drive_thresh $THRESH 2>&1 | tail -12

  echo "=== $(date) ROUND $r: retrain classifier (replay + per-class recall) ==="
  $PY continual_clf.py --archive weights/archive.pt --neg weights/neg_emb.pt --out vendor_conceptlab/weights/face_probe.pt --min_auc 0.97 2>&1 | tail -15

  # stop logic
  META=RESULTS/loop/round_${r}.json
  SUCCESS=$($PY -c "import json;print(json.load(open('$META'))['non_face_driver'])")
  DRYR=$($PY -c "import json;print(json.load(open('$META'))['dry_round'])")
  if [ "$SUCCESS" = "True" ]; then echo "=== $(date) STOP: NON-FACE DRIVER FOUND (round $r) ==="; break; fi
  if [ "$DRYR" = "True" ]; then DRY=$((DRY+1)); else DRY=0; fi
  if [ "$DRY" -ge 2 ]; then echo "=== $(date) STOP: 2 dry rounds -> ceiling (round $r) ==="; break; fi
done

echo "=== $(date) LOOP_DONE ==="
