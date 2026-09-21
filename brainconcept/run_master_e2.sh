#!/bin/bash
# Master: after E1 sweep -> build E2 anchor -> run E2 sweep -> score all 12 runs.
set -u
cd ~/MindReading/brainconcept
mkdir -p RESULTS/floor RESULTS/heldout

# 1. wait for the E1 sweep to finish (GPU frees)
while pgrep -f "[r]un_floor.sh" >/dev/null; do sleep 30; done
echo "=== $(date) E1 sweep done -> build E2 anchor ==="

# 2. build the 10-phrase E2 image anchor (Kandinsky cached now, fast)
cd ~/MindReading/brainconcept/vendor_conceptlab
PYTHONPATH=. ~/braindive_venv/bin/python ../build_anchor.py --preset e2 --n_samples 20 \
  --out weights/anchor_face_e2.pt --cache_root /tmp/kandinsky2 --device cuda:0
cd ~/MindReading/brainconcept

# 3. run the E2 sweep (same 6 configs, E2 anchor)
echo "=== $(date) E2 anchor built -> run E2 sweep ==="
bash run_floor_e2.sh

# 4. score all 12 runs: BLIP face-rate + ours(drive z, cat_sim) + DINOv2 realness
echo "=== $(date) both sweeps done -> scoring all 12 ==="
RUNS="floor_faces_z1_w1 floor_faces_z2_w1 floor_faces_z3_w1 floor_faces_z4_w1 floor_faces_z2_w05 floor_faces_z2_w2 \
floor_e2_faces_z1_w1 floor_e2_faces_z2_w1 floor_e2_faces_z3_w1 floor_e2_faces_z4_w1 floor_e2_faces_z2_w05 floor_e2_faces_z2_w2"
for run in $RUNS; do
  d=vendor_conceptlab/outputs/$run/images
  [ -d "$d" ] || { echo "skip $run (no images)"; continue; }
  echo "=== $(date) BLIP $run ==="
  ~/braindive_venv/bin/python judge_blip_face.py \
    --images "$d/*images.jpeg" --out RESULTS/floor/blip_${run}.json
  echo "=== $(date) OURS $run ==="
  ~/braindive_venv/bin/python score_heldout.py --encoder ours --roi faces \
    --category_anchor "a human face" \
    --images "$d/*images.jpeg" --out RESULTS/floor/ours_${run}.json
  echo "=== $(date) DINO $run ==="
  bash run_score_dino.sh faces "/work/vendor_conceptlab/outputs/$run/images/*images.jpeg" ${run}
done
echo "ALL_FLOOR_SCORING_DONE"
