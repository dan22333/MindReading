#!/usr/bin/env python
"""Write K generation configs for one loop round. Each run: distinct optimization seed,
classifier-repel against the CURRENT forbidden probe, E3 config, 1000 steps, 20 decode seeds
(final step only) for archive bulking. Prints the run-list to stdout + a runs file.

  python gen_round.py --round 1 --K 8 --probe weights/face_probe.pt --comp_mean -0.0055 --comp_std 0.0296
"""
import argparse
from pathlib import Path

CFG = Path("vendor_conceptlab/configs")
FACES_MEAN, FACES_STD = -0.0476, 0.2003
DECODE_SEEDS = list(range(20))   # 20 renderings of each found mode


def write_cfg(name, seed, probe, z, w, pstar, cm, cs, jitter):
    dec = "[" + ", ".join(str(s) for s in DECODE_SEEDS) + "]"
    out = f"""# loop round config (auto) — {name}
cache_root: "/tmp/kandinsky2"
output_dir: "outputs/{name}"
placeholder_token: "mybrainconcept"
initializer_token: "object"
num_tokens: 1
prompt_template: "A photo of a {{token}}"
img_size: 512
train_batch_size: 1
steps: 1000
log_image_frequency: 1000
log_embed_frequency: 1000
seed: {seed}
inference_seeds: {dec}

brain_objective: true
brain_encoder_weights: "weights/W_faces_s1.pt"
live_negatives: false

contrast_loss: true
floor_loss: false
comp_clamp: true
complement_weights: "weights/W_comp_s1.pt"
anchor_set: "weights/face_bank.pt"
classifier_repel: true
face_probe_path: "{probe}"
p_star: {pstar}
init_jitter: {jitter}
drive_floor: {z}
c_img: 0.30
c_txt: 0.1
w_img: {w}
w_txt: 0.0
use_max_negative: true
drift_monitor: false

roi_baseline_mean: {FACES_MEAN}
roi_baseline_std: {FACES_STD}
comp_baseline_mean: {cm}
comp_baseline_std: {cs}
"""
    (CFG / f"{name}.yaml").write_text(out)
    return name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", type=int, required=True)
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--probe", required=True)
    ap.add_argument("--comp_mean", type=float, required=True)
    ap.add_argument("--comp_std", type=float, required=True)
    ap.add_argument("--z", type=float, default=3.0)
    ap.add_argument("--w", type=float, default=1.0)
    ap.add_argument("--pstar", type=float, default=0.1)
    ap.add_argument("--jitter", type=float, default=0.0, help="per-token init perturbation (set >0 if seeds don't diversify)")
    args = ap.parse_args()
    CFG.mkdir(parents=True, exist_ok=True)
    names = []
    for k in range(args.K):
        seed = 1000 * args.round + k           # distinct per run, fresh per round
        names.append(write_cfg(f"loop_r{args.round}_s{seed}", seed, args.probe,
                               args.z, args.w, args.pstar, args.comp_mean, args.comp_std, args.jitter))
    runs_file = Path(f"loop_r{args.round}_runs.txt")
    runs_file.write_text("\n".join(names) + "\n")
    print(f"[gen_round] round {args.round}: wrote {len(names)} configs -> {runs_file}")
    for n in names:
        print("  ", n)


if __name__ == "__main__":
    main()
