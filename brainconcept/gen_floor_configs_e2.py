#!/usr/bin/env python
"""Generate the E2 floor-loss configs (10-phrase image anchor) + E2 driver.
Identical 6-config sweep as E1, but anchor_img_path -> anchor_face_e2.pt and
output dirs / config names prefixed floor_e2_. This is the second experiment:
same floor loss, anchor built from 10 face phrasings (broader face region).
"""
from pathlib import Path

CFG_DIR = Path.home() / "MindReading/brainconcept/vendor_conceptlab/configs"
DRIVER = Path.home() / "MindReading/brainconcept/run_floor_e2.sh"
BASE_MEAN = -0.047633
BASE_STD = 0.200336

TEMPLATE = '''# Floor-loss E2 (10-phrase image anchor): L = ReLU(drive_floor - z_norm) + w * cos(z, anchor_e2)
cache_root: "/tmp/kandinsky2"
output_dir: "outputs/NAME"
placeholder_token: "mybrainconcept"
initializer_token: "object"
num_tokens: 1
prompt_template: "A photo of a {token}"
live_negatives: false
extract_terms: false
neg_weight: 0.0
inference_seeds: [42, 500, 7, 13, 21, 99]
img_size: 512
train_batch_size: 1
steps: 2000
log_image_frequency: 250
log_embed_frequency: 250
early_stop_frac: 0.0
seed: 0
brain_objective: true
brain_encoder_weights: "weights/W_faces_s1.pt"
floor_loss: true
drive_floor: FLOOR
category_weight: WEIGHT
anchor_img_path: "weights/anchor_face_e2.pt"
roi_baseline_mean: MEAN
roi_baseline_std: STD
'''

RUNS = [
    ("floor_e2_faces_z1_w1", 1.0, 1.0),
    ("floor_e2_faces_z2_w1", 2.0, 1.0),
    ("floor_e2_faces_z3_w1", 3.0, 1.0),
    ("floor_e2_faces_z4_w1", 4.0, 1.0),
    ("floor_e2_faces_z2_w05", 2.0, 0.5),
    ("floor_e2_faces_z2_w2", 2.0, 2.0),
]

CFG_DIR.mkdir(parents=True, exist_ok=True)
for name, floor, w in RUNS:
    s = (TEMPLATE.replace("NAME", name).replace("FLOOR", str(floor))
         .replace("WEIGHT", str(w)).replace("MEAN", str(BASE_MEAN)).replace("STD", str(BASE_STD)))
    (CFG_DIR / f"{name}.yaml").write_text(s)
    print(f"wrote {CFG_DIR / (name + '.yaml')}")

names = " ".join(n for n, _, _ in RUNS)
DRIVER.write_text(f'''#!/bin/bash
cd ~/MindReading/brainconcept/vendor_conceptlab
for cfg in {names}; do
  echo "=== $(date) START $cfg ==="
  PYTHONPATH=. ~/braindive_venv/bin/python -m scripts.train --config_path configs/${{cfg}}.yaml
  echo "=== $(date) DONE $cfg ==="
done
echo "=== $(date) FLOOR_E2 SWEEP COMPLETE ==="
''')
DRIVER.chmod(0o755)
print(f"wrote {DRIVER}")
