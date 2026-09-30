#!/bin/bash
# Round 2 master: banks -> probe(GATE) -> calibrate -> configs -> sweep(24) -> score+dino.
# One server-side process so nothing stalls between phases. Emits ROUND2_MASTER_ALL_DONE.
set -uo pipefail
BC=~/MindReading/brainconcept
cd "$BC"

echo "=== $(date) Phase0: build face/non-face banks ==="
( cd vendor_conceptlab && PYTHONPATH=. ~/braindive_venv/bin/python ../build_face_banks.py \
    --per_term 25 --out_face weights/face_bank.pt --out_nonface weights/nonface_bank.pt )

echo "=== $(date) Phase1: train + VALIDATE face probe (AUC gate >=0.97) ==="
( cd vendor_conceptlab && ~/braindive_venv/bin/python ../train_face_probe.py \
    --face weights/face_bank.pt --nonface weights/nonface_bank.pt --out weights/face_probe.pt \
    --min_auc 0.97 --baseline_images ../data/s1_images.npy --baseline_split ../data/s1_split.npz )
if [ $? -ne 0 ]; then echo "PROBE GATE FAILED — aborting round 2 (no sweep)"; exit 2; fi

echo "=== $(date) Phase2: calibrate E2 bank hinge ==="
~/braindive_venv/bin/python calib_cstar.py --anchors vendor_conceptlab/weights/face_bank.pt \
    --baseline_images data/s1_images.npy --baseline_split data/s1_split.npz --baseline_n 300 \
    --c_img 0.30 0.35 0.40 || true

echo "=== $(date) Phase3: generate configs + run 24-run sweep ==="
~/braindive_venv/bin/python gen_round2_configs.py --comp_mean -0.0055 --comp_std 0.0296 --c_img 0.30 --p_star 0.1
bash run_contrast.sh "$BC/round2_runs.txt" 2

echo "=== $(date) Phase4: score all 24 (ours+BLIP+drift+p_face) + DINOv2 ==="
ALL=$(cat round2_runs.txt | tr '\n' ' ')
~/braindive_venv/bin/python score_contrast_all.py --runs $ALL > ~/score_r2.log 2>&1
echo "contrast scoring exit $?"
bash run_score_dino_all.sh $ALL > ~/dino_r2.log 2>&1
echo "dino scoring exit $?"

echo "=== $(date) ROUND2_MASTER_ALL_DONE ==="
