#!/usr/bin/env python
"""Round 2 configs: E2 (large face-image bank cosine) + E3 (face classifier), with the drive fix.

Loss (all runs, comp_clamp=on):  L = ReLU(z* - (z_faces - max(z_comp,0))) + w * novelty
  E2 novelty = mean+max clamp(cos(e, face_bank), c_img)          (bank ~1000 faces)
  E3 novelty = ReLU(P_face(e) - p_star)                          (trained linear probe)

Sweep: z* in {2,2.5,3,3.5} x w in {1,1.5,2} x {E2,E3} = 24 runs. No E1, no text repel.

  python gen_round2_configs.py --comp_mean -0.0055 --comp_std 0.0296 --c_img 0.30 --p_star 0.1
"""
import argparse
from pathlib import Path

CFG = Path("vendor_conceptlab/configs")
FACES_MEAN, FACES_STD = -0.0476, 0.2003


def write_cfg(name, z, w, mode, c_img, p_star, cm, cs):
    cls = "true" if mode == "e3" else "false"
    out = f"""# round2 {mode} — {name}
cache_root: "/tmp/kandinsky2"
output_dir: "outputs/{name}"
placeholder_token: "mybrainconcept"
initializer_token: "object"
num_tokens: 1
prompt_template: "A photo of a {{token}}"
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
anchor_set: "weights/face_bank.pt"
classifier_repel: {cls}
face_probe_path: "weights/face_probe.pt"
p_star: {p_star}
drive_floor: {z}
c_img: {c_img}
c_txt: 0.1
w_img: {w}
w_txt: 0.0
use_max_negative: true
drift_monitor: true

roi_baseline_mean: {FACES_MEAN}
roi_baseline_std: {FACES_STD}
comp_baseline_mean: {cm}
comp_baseline_std: {cs}
"""
    (CFG / f"{name}.yaml").write_text(out)
    return name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--comp_mean", type=float, required=True)
    ap.add_argument("--comp_std", type=float, required=True)
    ap.add_argument("--c_img", type=float, default=0.30)
    ap.add_argument("--p_star", type=float, default=0.1)
    args = ap.parse_args()
    CFG.mkdir(parents=True, exist_ok=True)
    names = []
    for mode in ["e2", "e3"]:
        for z in [2, 2.5, 3, 3.5]:
            for w in [1, 1.5, 2]:
                zt = str(z).replace(".", "p"); wt = str(w).replace(".", "p")
                names.append(write_cfg(f"r2_{mode}_z{zt}_w{wt}", z, w, mode,
                                       args.c_img, args.p_star, args.comp_mean, args.comp_std))
    (CFG.parent.parent / "round2_runs.txt").write_text("\n".join(names) + "\n")
    print(f"[gen] wrote {len(names)} configs -> round2_runs.txt")
    for n in names:
        print("  ", n)


if __name__ == "__main__":
    main()
