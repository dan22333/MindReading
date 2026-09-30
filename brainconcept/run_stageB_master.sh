#!/bin/bash
# Server-side master: run Stage B, then score EVERYTHING (contrast + DINOv2), all in one
# process so nothing stalls between steps. Emits STAGEB_MASTER_ALL_DONE at the end.
set -u
cd ~/MindReading/brainconcept

echo "=== $(date) STAGE B sweep start ==="
bash run_contrast.sh "$HOME/MindReading/brainconcept/contrast_stageB_runs.txt" 2

echo "=== $(date) scoring ALL runs (contrast: ours+BLIP+drift) ==="
ALL=$(cat contrast_stageA_runs.txt contrast_stageB_runs.txt | tr '\n' ' ')
~/braindive_venv/bin/python score_contrast_all.py --runs $ALL > ~/score_contrast_all.log 2>&1
echo "contrast scoring exit $?"

echo "=== $(date) DINOv2 held-out on Stage B (A already scored) ==="
BONLY=$(cat contrast_stageB_runs.txt | tr '\n' ' ')
bash run_score_dino_all.sh $BONLY > ~/dino_stageB.log 2>&1
echo "dino scoring exit $?"

echo "=== $(date) STAGEB_MASTER_ALL_DONE ==="
