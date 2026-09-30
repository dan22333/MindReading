#!/bin/bash
# Round 2 SETUP only: banks -> probe(train+validate) -> calibrate -> gen configs. No sweep.
set -uo pipefail
BC=~/MindReading/brainconcept; cd "$BC"
echo "=== $(date) Phase0: build face/non-face banks ==="
( cd vendor_conceptlab && PYTHONPATH=. ~/braindive_venv/bin/python ../build_face_banks.py \
    --per_term 25 --out_face weights/face_bank.pt --out_nonface weights/nonface_bank.pt )
echo "=== $(date) Phase1: train + validate face probe ==="
( cd vendor_conceptlab && ~/braindive_venv/bin/python ../train_face_probe.py \
    --face weights/face_bank.pt --nonface weights/nonface_bank.pt --out weights/face_probe.pt \
    --min_auc 0.97 --baseline_images ../data/s1_images.npy --baseline_split ../data/s1_split.npz )
PROBE_RC=$?
echo "PROBE_RC=$PROBE_RC"
echo "=== $(date) Phase2: calibrate E2 bank hinge ==="
~/braindive_venv/bin/python calib_cstar.py --anchors vendor_conceptlab/weights/face_bank.pt \
    --baseline_images data/s1_images.npy --baseline_split data/s1_split.npz --baseline_n 300 \
    --c_img 0.30 0.35 0.40 || true
echo "=== $(date) Phase3: generate configs ==="
~/braindive_venv/bin/python gen_round2_configs.py --comp_mean -0.0055 --comp_std 0.0296 --c_img 0.30 --p_star 0.1
echo "=== $(date) ROUND2_SETUP_DONE PROBE_RC=$PROBE_RC ==="
