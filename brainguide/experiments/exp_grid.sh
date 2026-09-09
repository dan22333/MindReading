#!/usr/bin/env bash
# 5-samples-per-ROI grid: baseline + plan1 (hygiene) + plan2 (ddim ctrl & time-travel),
# 5 ROIs x 5 seeds, on the GPU VM with live-mounted code. (Plan 4 is separate: run_test4.sh
# with --seeds, since it scores+trains the bridge once then samples cheaply.)
#   bash ~/MindReading/brainguide/experiments/exp_grid.sh
set -uo pipefail

OUT=~/exp_outputs_grid
mkdir -p "$OUT"/{baseline,test1,test2}
cd ~/MindReading

ROIS=${ROIS:-"faces places words bodies earlyvisual"}
SEEDS=${SEEDS:-"0 1 2 3 4"}

dr() {  # docker run, live-mounted code; all args -> run.py
  sudo docker run --gpus all --rm \
    -v ~/MindReading/brainguide:/app/brainguide \
    -v ~/MindReading/brainit-fmri/results:/app/brainit-fmri/results \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -v ~/.cache/torch:/root/.cache/torch \
    -v "$OUT":/app/brainguide/exp_out \
    -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    brainguide "$@" 2>&1 | tail -2
}

for roi in $ROIS; do
  V="brainguide/roi/sub1_${roi}.npy"
  for s in $SEEDS; do
    echo "======== roi=$roi seed=$s : baseline ========"
    dr --config configs/universal.yaml \
       --prompt "" --voxels "$V" --tag "base_${roi}" --out exp_out/baseline --seed "$s"
    echo "======== roi=$roi seed=$s : plan1 hygiene ========"
    dr --config experiments/test1_gradient_hygiene/config.yaml \
       --prompt "" --voxels "$V" --tag "t1_${roi}" --out exp_out/test1 --seed "$s"
    echo "======== roi=$roi seed=$s : plan2 ctrl (ddim reps=1) ========"
    dr --config experiments/test2_timetravel/config.yaml \
       --prompt "" --voxels "$V" --tag "t2ctrl_${roi}" --out exp_out/test2 --tt-reps 1 --seed "$s"
    echo "======== roi=$roi seed=$s : plan2 time-travel (ddim reps=4) ========"
    dr --config experiments/test2_timetravel/config.yaml \
       --prompt "" --voxels "$V" --tag "t2tt_${roi}" --out exp_out/test2 --seed "$s"
  done
done
echo "======== GUIDED GRID DONE ($(echo $ROIS | wc -w) ROIs x $(echo $SEEDS | wc -w) seeds) ========"
