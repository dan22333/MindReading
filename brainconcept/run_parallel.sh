#!/bin/bash
# 2-way parallel orchestrator for the floor sweep + load-once scoring.
# Waits for the in-flight serial run to finish, builds the E2 anchor, then runs
# every not-yet-done config 2-at-a-time, then scores all 12 with load-once scorers.
set -u
BC=~/MindReading/brainconcept
VC=$BC/vendor_conceptlab
PY=~/braindive_venv/bin/python
# NOTE: this venv's torch predates the 'expandable_segments' allocator option — setting it
# crashes every CUDA process at init. 2 runs x ~9GB fit in the L4's 23GB without it.
cd $BC
mkdir -p RESULTS/floor RESULTS/heldout

ALL="floor_faces_z1_w1 floor_faces_z2_w1 floor_faces_z3_w1 floor_faces_z4_w1 floor_faces_z2_w05 floor_faces_z2_w2 \
floor_e2_faces_z1_w1 floor_e2_faces_z2_w1 floor_e2_faces_z3_w1 floor_e2_faces_z4_w1 floor_e2_faces_z2_w05 floor_e2_faces_z2_w2"

# 0. wait for any in-flight training (the serial run 1) to finish so GPU is clean
while pgrep -f "[s]cripts.train" >/dev/null; do sleep 15; done
echo "=== $(date) GPU clear -> build E2 anchor ==="

# 1. build the E2 (10-phrase) anchor alone (needs GPU briefly)
( cd $VC && PYTHONPATH=. $PY ../build_anchor.py --preset e2 --n_samples 20 \
    --out weights/anchor_face_e2.pt --cache_root /tmp/kandinsky2 --device cuda:0 )
echo "=== $(date) E2 anchor built ==="

# 2. queue = configs whose output isn't already complete (>=9 step grids incl init)
QUEUE=""
for r in $ALL; do
  n=$(ls $VC/outputs/$r/images/*images.jpeg 2>/dev/null | wc -l)
  if [ "$n" -ge 9 ]; then echo "skip $r (done: $n imgs)"; else QUEUE="$QUEUE $r"; fi
done
echo "=== $(date) queue:$QUEUE ==="

# 3. run 2-at-a-time (stagger 25s so Kandinsky loads / decode checkpoints don't collide)
run_one(){
  local cfg=$1
  echo "=== $(date) START $cfg ==="
  ( cd $VC && PYTHONPATH=. $PY -m scripts.train --config_path configs/${cfg}.yaml ) > ~/log_${cfg}.log 2>&1
  echo "=== $(date) DONE $cfg (exit $?) ==="
}
MAXJ=2
for cfg in $QUEUE; do
  while [ "$(jobs -rp | wc -l)" -ge "$MAXJ" ]; do sleep 10; done
  run_one "$cfg" &
  sleep 25
done
wait
echo "=== $(date) ALL TRAINING DONE -> scoring ==="

# 4. load-once scoring: BLIP+ours in one process, DINO in one docker pass
$PY score_floor_all.py --runs $ALL --out RESULTS/floor
bash run_score_dino_all.sh $ALL
echo "ALL_FLOOR_SCORING_DONE"
