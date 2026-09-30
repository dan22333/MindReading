#!/usr/bin/env python
"""Generate the contrast-loss (E1/E2 v2) sweep configs from floor_faces_base.yaml.

Loss per config:
  L = ReLU(z* - (z_faces - z_comp)) + w_img*(mean+max clamp(img_cos, c_img))
                                    + w_txt*(mean+max clamp(txt_cos, c_txt))

Two stages (run A, score, pick w_img*, then run B):
  Stage A — isolate the repel lever (E2 anchor, z*=1, c_img=0.30 = measured edge):
     image-only w_img in {0.5,1,2,2.5,3} (w_txt=0)  ->  5
     text-only  w_txt in {1,3} (w_img=0)            ->  2
     both       (w_img=1, w_txt=1)                  ->  1   = 8 runs
  Stage B — frontier at w_img* (w_txt=0):
     z* in {1,2,3,3.5} x {E1,E2}  at c_img=0.30     ->  8   (on-manifold edge)
     c_img=0.20 arm  z* in {1,2} x {E1,E2}          ->  4   (aggressive)  = 12 runs
  (c_img calibrated 2026-09-27: natural non-face images sit at image-cos ~0.29-0.46 to the
   face anchors, so 0.30 = manifold edge, 0.20 = aggressive/below-edge.)

Usage:
  python gen_contrast_configs.py --stage A --comp_mean <CM> --comp_std <CS>
  python gen_contrast_configs.py --stage B --comp_mean <CM> --comp_std <CS> --w_img_star 1.0
"""
import argparse
from pathlib import Path

CFG_DIR = Path("vendor_conceptlab/configs")
BASE = CFG_DIR / "floor_faces_base.yaml"

# faces ROI baseline (reconfirmed by slice_complement.py); complement stats come from CLI.
FACES_MEAN, FACES_STD = -0.0476, 0.2003
ANCHOR = {"e1": "weights/anchor_e1.pt", "e2": "weights/anchor_e2.pt"}


def wtag(w):
    return str(w).replace(".", "").replace("0", "0") if w == 0 else str(w).replace(".", "")


def write_cfg(name, exp, z, w_img, w_txt, c_img, comp_mean, comp_std):
    out = f"""# contrast-loss config (auto-gen) — {name}
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

# --- contrast loss ---
contrast_loss: true
floor_loss: false
complement_weights: "weights/W_comp_s1.pt"
anchor_set: "{ANCHOR[exp]}"
drive_floor: {z}
c_img: {c_img}
c_txt: 0.1
w_img: {w_img}
w_txt: {w_txt}
use_max_negative: true
drift_monitor: true

# baselines (faces reconfirmed; complement from slice_complement.py)
roi_baseline_mean: {FACES_MEAN}
roi_baseline_std: {FACES_STD}
comp_baseline_mean: {comp_mean}
comp_baseline_std: {comp_std}
"""
    (CFG_DIR / f"{name}.yaml").write_text(out)
    return name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["A", "B"], required=True)
    ap.add_argument("--comp_mean", type=float, required=True)
    ap.add_argument("--comp_std", type=float, required=True)
    ap.add_argument("--w_img_star", type=float, default=1.0, help="Stage B: winning w_img from A")
    ap.add_argument("--w_txt_star", type=float, default=1.0, help="Stage B: winning w_txt from A")
    args = ap.parse_args()
    CFG_DIR.mkdir(parents=True, exist_ok=True)

    names = []
    if args.stage == "A":
        # image-only (E2, z*=1, c_img=0.30 = measured edge)
        for w in [0.5, 1, 2, 2.5, 3]:
            names.append(write_cfg(f"cA_e2_z1_wi{wtag(w)}_wt0", "e2", 1, w, 0, 0.30,
                                   args.comp_mean, args.comp_std))
        # text-only
        for w in [1, 3]:
            names.append(write_cfg(f"cA_e2_z1_wi0_wt{wtag(w)}", "e2", 1, 0, w, 0.30,
                                   args.comp_mean, args.comp_std))
        # both
        names.append(write_cfg("cA_e2_z1_wi1_wt1", "e2", 1, 1, 1, 0.30,
                               args.comp_mean, args.comp_std))
    else:
        W = args.w_img_star
        WT = args.w_txt_star
        wt = wtag(W); wtt = wtag(WT)
        # frontier at c_img=0.30 (on-manifold edge), both anchors, z* sweep
        for exp in ["e1", "e2"]:
            for z in [1, 2, 3, 3.5]:
                zt = str(z).replace(".", "p")
                names.append(write_cfg(f"cB_{exp}_z{zt}_wi{wt}_wt{wtt}_ci030", exp, z, W, WT, 0.30,
                                       args.comp_mean, args.comp_std))
        # conservative c_img=0.40 arm (mild novelty, deep on-manifold) at low floors
        for exp in ["e1", "e2"]:
            for z in [1, 2]:
                zt = str(z).replace(".", "p")
                names.append(write_cfg(f"cB_{exp}_z{zt}_wi{wt}_wt{wtt}_ci040", exp, z, W, WT, 0.40,
                                       args.comp_mean, args.comp_std))

    print(f"[gen] stage {args.stage}: wrote {len(names)} configs")
    for n in names:
        print("  ", n)
    # emit a plain run-list for the driver
    (CFG_DIR.parent.parent / f"contrast_stage{args.stage}_runs.txt").write_text("\n".join(names) + "\n")
    print(f"[gen] run-list -> contrast_stage{args.stage}_runs.txt")


if __name__ == "__main__":
    main()
