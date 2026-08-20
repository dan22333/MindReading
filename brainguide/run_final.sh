#!/usr/bin/env bash
# Final generation: best-known config = baseline (empty prompt, no negative,
# clip=0.30). 5 seeds x 5 ROIs = 25 images for curation.
set -euo pipefail
OUT=~/outputs_final
mkdir -p "$OUT"
cd ~/MindReading
for roi in faces bodies words places earlyvisual; do
  for seed in 0 1 2 3 4; do
    echo ">>> roi=$roi seed=$seed"
    sudo docker run --gpus all --rm \
      -v ~/MindReading/brainguide:/app/brainguide \
      -v ~/MindReading/brainit-fmri/results:/app/brainit-fmri/results \
      -v ~/.cache/huggingface:/root/.cache/huggingface \
      -v ~/.cache/torch:/root/.cache/torch \
      -v "$OUT":/app/brainguide/outputs \
      brainguide --config configs/universal.yaml --prompt "" \
        --voxels brainguide/roi/sub1_${roi}.npy \
        --tag ${roi} --clip-scale 0.30 --seed "$seed" 2>&1 | tail -1
  done
done
echo "=================== DONE ==================="
ls "$OUT"/*.png | wc -l | xargs echo "png count:"
cd "$OUT" && tar czf ~/final.tgz *.png && echo "tar ready: ~/final.tgz"
