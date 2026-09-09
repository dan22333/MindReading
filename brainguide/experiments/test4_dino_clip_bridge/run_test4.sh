#!/usr/bin/env bash
# TEST 4 on the VM: fetch a small image bank, then run the bridge + Stable unCLIP.
#   bash ~/MindReading/brainguide/experiments/test4_dino_clip_bridge/run_test4.sh
set -euo pipefail
OUT=~/exp_outputs
BANK=~/exp_outputs/imagebank
mkdir -p "$OUT/test4"

# 1. image bank (1000 diverse ImageNet samples, ~200 MB) — for bridge training + top-K target
if [ ! -d "$BANK/.git" ] && [ -z "$(ls -A "$BANK" 2>/dev/null || true)" ]; then
  echo "[t4] cloning image bank..."
  git clone --depth 1 https://github.com/EliSchwartz/imagenet-sample-images "$BANK"
fi
echo "[t4] image bank: $(find "$BANK" -iname '*.jpg' -o -iname '*.png' -o -iname '*.jpeg' | wc -l) images"

cd ~/MindReading
sudo docker run --gpus all --rm \
  -v ~/MindReading/brainguide:/app/brainguide \
  -v ~/MindReading/brainit-fmri/results:/app/brainit-fmri/results \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -v ~/.cache/torch:/root/.cache/torch \
  -v "$OUT":/app/brainguide/exp_out \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  --entrypoint python \
  brainguide experiments/test4_dino_clip_bridge/run_test4.py \
    --images /app/brainguide/exp_out/imagebank \
    --config configs/universal.yaml \
    --out exp_out/test4 --topk 20 --epochs 30 --bs 8 --seeds "${SEEDS:-0 1 2 3 4}"
echo "[t4] DONE — results in $OUT/test4"
