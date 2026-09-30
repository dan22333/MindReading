#!/bin/bash
# One-run TEXT-repel test: build 1000-phrase text bank -> calibrate -> 1 config (z*=3,w=1) -> train -> score.
set -uo pipefail
BC=~/MindReading/brainconcept; cd "$BC"

echo "=== $(date) build text bank (1000 phrasings) ==="
( cd vendor_conceptlab && PYTHONPATH=. ~/braindive_venv/bin/python ../build_text_bank.py --n 1000 --out weights/text_bank.pt )

echo "=== $(date) calibrate text threshold (naturals vs text bank) ==="
~/braindive_venv/bin/python calib_cstar.py --anchors vendor_conceptlab/weights/text_bank.pt \
    --baseline_images data/s1_images.npy --baseline_split data/s1_split.npz --baseline_n 300 \
    --c_img 0.10 0.12 0.15 || true

echo "=== $(date) write config r2_txt_z3_w1 (c_img=0.10 text threshold) ==="
~/braindive_venv/bin/python - <<'PY'
cfg = '''# text-repel test — r2_txt_z3_w1
cache_root: "/tmp/kandinsky2"
output_dir: "outputs/r2_txt_z3_w1"
placeholder_token: "mybrainconcept"
initializer_token: "object"
num_tokens: 1
prompt_template: "A photo of a {token}"
img_size: 512
train_batch_size: 1
steps: 2000
log_image_frequency: 500
log_embed_frequency: 500
seed: 0
brain_objective: true
brain_encoder_weights: "weights/W_faces_s1.pt"
live_negatives: false
contrast_loss: true
floor_loss: false
comp_clamp: true
complement_weights: "weights/W_comp_s1.pt"
anchor_set: "weights/text_bank.pt"
classifier_repel: false
drive_floor: 3
c_img: 0.10
c_txt: 0.10
w_img: 1.0
w_txt: 0.0
use_max_negative: true
drift_monitor: true
roi_baseline_mean: -0.0476
roi_baseline_std: 0.2003
comp_baseline_mean: -0.0055
comp_baseline_std: 0.0296
'''
open("vendor_conceptlab/configs/r2_txt_z3_w1.yaml","w").write(cfg)
print("wrote config")
PY

echo "=== $(date) train ==="
( cd vendor_conceptlab && PYTHONPATH=. ~/braindive_venv/bin/python -m scripts.train --config_path configs/r2_txt_z3_w1.yaml > ~/contrast_r2_txt_z3_w1.log 2>&1 )
echo "train exit $?"

echo "=== $(date) score (ours+BLIP+drift) + DINOv2 ==="
~/braindive_venv/bin/python score_contrast_all.py --runs r2_txt_z3_w1 > ~/score_txt.log 2>&1
bash run_score_dino_all.sh r2_txt_z3_w1 > ~/dino_txt.log 2>&1
echo "=== $(date) TEXT_TEST_DONE ==="
