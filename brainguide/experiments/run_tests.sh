#!/usr/bin/env bash
# Run TEST 1 (gradient hygiene) and TEST 2 (time-travel) on the universal encoder,
# across the 5 ROIs, seed 0 — directly comparable to the existing baseline
# (~/outputs/sub1_<roi>_seed0_0.png). Runs ON the GPU VM with live-mounted code.
#   bash ~/MindReading/brainguide/experiments/run_tests.sh
set -euo pipefail

OUT=~/exp_outputs
mkdir -p "$OUT/test1" "$OUT/test2"
cd ~/MindReading

ROIS=${ROIS:-"faces places words bodies earlyvisual"}
SEED=${SEED:-0}

dr() {  # docker run with live-mounted code; all args -> run.py
  sudo docker run --gpus all --rm \
    -v ~/MindReading/brainguide:/app/brainguide \
    -v ~/MindReading/brainit-fmri/results:/app/brainit-fmri/results \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -v ~/.cache/torch:/root/.cache/torch \
    -v "$OUT":/app/brainguide/exp_out \
    brainguide "$@" 2>&1 | tail -3
}

for roi in $ROIS; do
  V="brainguide/roi/sub1_${roi}.npy"
  echo "=================== ROI=$roi  TEST 1 (hygiene) ==================="
  dr --config experiments/test1_gradient_hygiene/config.yaml \
     --prompt "" --voxels "$V" --tag "t1_${roi}" --out exp_out/test1 --seed "$SEED"

  echo "=================== ROI=$roi  TEST 2 control (ddim reps=1) ==================="
  dr --config experiments/test2_timetravel/config.yaml \
     --prompt "" --voxels "$V" --tag "t2ctrl_${roi}" --out exp_out/test2 --tt-reps 1 --seed "$SEED"

  echo "=================== ROI=$roi  TEST 2 (ddim time-travel reps=4) ==================="
  dr --config experiments/test2_timetravel/config.yaml \
     --prompt "" --voxels "$V" --tag "t2tt_${roi}" --out exp_out/test2 --seed "$SEED"
done
echo "=================== DONE — results in $OUT ==================="
