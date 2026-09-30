#!/bin/bash
# Round 2 SWEEP + score for a given run-list file (arg1). 2-parallel; scores ours+BLIP+drift+p_face + DINOv2.
set -uo pipefail
BC=~/MindReading/brainconcept; cd "$BC"
RUNLIST="${1:-$BC/round2_runs.txt}"
echo "=== $(date) sweep start: $(wc -l < "$RUNLIST") runs from $RUNLIST ==="
bash run_contrast.sh "$RUNLIST" 2
ALL=$(cat "$RUNLIST" | tr '\n' ' ')
echo "=== $(date) scoring (ours+BLIP+drift+p_face) ==="
~/braindive_venv/bin/python score_contrast_all.py --runs $ALL > ~/score_r2.log 2>&1
echo "contrast scoring exit $?"
echo "=== $(date) DINOv2 held-out ==="
bash run_score_dino_all.sh $ALL > ~/dino_r2.log 2>&1
echo "dino scoring exit $?"
echo "=== $(date) ROUND2_SWEEP_DONE ($RUNLIST) ==="
