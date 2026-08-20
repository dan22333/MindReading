#!/usr/bin/env bash
# Quality-lever experiment for brainguide. 4 conditions x 5 ROIs x 2 seeds = 40 imgs.
#   A baseline: empty prompt, no negative (control)
#   B neg only: empty prompt + quality negative prompt
#   C pos only: neutral photo prompt, no negative
#   D both:     neutral photo prompt + quality negative prompt
set -euo pipefail

QUALITY_NEG="blurry, low quality, deformed, distorted, jpeg artifacts, watermark, text, oversaturated, grainy"
QUALITY_POS="a high-quality photograph, sharp focus, natural lighting, detailed"

OUT=~/outputs_matrix
mkdir -p "$OUT"
cd ~/MindReading

run() {  # $1=cond $2=prompt $3=neg(may be empty) $4=roi $5=seed
  local cond="$1" prompt="$2" neg="$3" roi="$4" seed="$5"
  local negflag=()
  [ -n "$neg" ] && negflag=(--negative-prompt "$neg")
  sudo docker run --gpus all --rm \
    -v ~/MindReading/brainguide:/app/brainguide \
    -v ~/MindReading/brainit-fmri/results:/app/brainit-fmri/results \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -v ~/.cache/torch:/root/.cache/torch \
    -v "$OUT":/app/brainguide/outputs \
    brainguide --config configs/universal.yaml \
      --prompt "$prompt" "${negflag[@]}" \
      --voxels brainguide/roi/sub1_${roi}.npy \
      --tag ${cond}_${roi} --clip-scale 0.30 --seed "$seed" 2>&1 | tail -1
}

for roi in faces bodies words places earlyvisual; do
  for seed in 0 1; do
    echo ">>> roi=$roi seed=$seed  A/baseline"; run A ""            ""            "$roi" "$seed"
    echo ">>> roi=$roi seed=$seed  B/neg";      run B ""            "$QUALITY_NEG" "$roi" "$seed"
    echo ">>> roi=$roi seed=$seed  C/pos";      run C "$QUALITY_POS" ""           "$roi" "$seed"
    echo ">>> roi=$roi seed=$seed  D/both";     run D "$QUALITY_POS" "$QUALITY_NEG" "$roi" "$seed"
  done
done
echo "=================== DONE ==================="
ls -la "$OUT" | grep -c '\.png$' | xargs echo "png count:"
