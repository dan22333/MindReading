#!/usr/bin/env python
"""Load-once DINOv2 held-out scorer (runs inside the brainguide docker). Loads the
UniversalEncoder + baseline ONCE, iterates all runs -> /work/RESULTS/heldout/dino_<run>.json.
Reuses score_heldout.DinoBackend / baseline_stats / tiling."""
import sys, glob, json, os, argparse
sys.path.insert(0, "/work")
import score_heldout as SH


class Args:
    pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--roi", default="faces")
    ap.add_argument("--out", default="/work/RESULTS/heldout")
    a = ap.parse_args()
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args = Args()
    args.baseline_images = "/work/data/s1_images.npy"
    args.baseline_split = "/work/data/s1_split.npz"
    args.baseline_n = 300
    backend = SH.DinoBackend(a.roi, args, device)
    bmean, bstd, bn = SH.baseline_stats(backend, args)
    print(f"[baseline] dino/{a.roi}: mean={bmean:.4f} std={bstd:.4f} n={bn}", flush=True)
    os.makedirs(a.out, exist_ok=True)
    for run in a.runs:
        files = sorted(glob.glob(f"/work/vendor_conceptlab/outputs/{run}/images/*images.jpeg"), key=SH.step_key)
        if not files:
            print(f"skip {run} (no images)", flush=True); continue
        steps = {}
        for f in files:
            tiles = SH.load_grid_tiles(f); raw = backend.score(tiles); z = (raw - bmean) / bstd
            steps[str(SH.step_key(f))] = {"file": os.path.basename(f), "n_tiles": len(tiles),
                "raw_mean": float(raw.mean()), "z_mean": float(z.mean()), "z_max": float(z.max()),
                "z_tiles": [float(v) for v in z]}
        json.dump({"encoder": "dino", "roi": a.roi, "baseline": {"mean": bmean, "std": bstd, "n": bn},
                   "steps": steps}, open(f"{a.out}/dino_{run}.json", "w"), indent=2)
        fk = max(int(k) for k in steps)
        print(f"[{run}] dino final z={steps[str(fk)]['z_mean']:+.2f}", flush=True)
    print("DINO_ALL_DONE", flush=True)


if __name__ == "__main__":
    main()
