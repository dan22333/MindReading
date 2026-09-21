#!/bin/bash
# Auto-scorer for the floor-loss sweep. Waits for run_floor.sh, then per run:
#   - BLIP-2 face yes/no  -> RESULTS/floor/blip_<run>.json   (honest novelty)
#   - ours (drive z + cat_sim) -> RESULTS/floor/ours_<run>.json
#   - DINOv2 held-out realness  -> RESULTS/heldout/dino_floor_<run>.json (docker)
set -u
cd ~/MindReading/brainconcept
mkdir -p RESULTS/floor RESULTS/heldout
while pgrep -f "[r]un_floor.sh" >/dev/null; do sleep 30; done
echo "=== $(date) training done -> scoring floor sweep ==="

RUNS="floor_faces_z1_w1 floor_faces_z2_w1 floor_faces_z3_w1 floor_faces_z4_w1 floor_faces_z2_w05 floor_faces_z2_w2"
for run in $RUNS; do
  d=vendor_conceptlab/outputs/$run/images
  if [ ! -d "$d" ]; then echo "skip $run (no images)"; continue; fi
  echo "=== $(date) BLIP $run ==="
  ~/braindive_venv/bin/python judge_blip_face.py \
    --images "$d/*images.jpeg" --out RESULTS/floor/blip_${run}.json
  echo "=== $(date) OURS $run ==="
  ~/braindive_venv/bin/python score_heldout.py --encoder ours --roi faces \
    --category_anchor "a human face" \
    --images "$d/*images.jpeg" --out RESULTS/floor/ours_${run}.json
  echo "=== $(date) DINO $run ==="
  bash run_score_dino.sh faces "/work/vendor_conceptlab/outputs/$run/images/*images.jpeg" floor_${run}
done
echo "ALL_FLOOR_SCORING_DONE"
