#!/usr/bin/env python
"""Stage 0 — assemble the subject-1 training arrays for the brainconcept encoder.

Produces, in the SAME fsaverage/Algonauts voxel order as brainit-fmri's
prepare_fmri.py ( concatenate([lh_masked, rh_masked]) → ~39,548 voxels ), so the
existing brainguide/roi/sub1_*.npy index arrays stay valid:

    data/s1_betas.npy    [N_img, V]  float16  repeat-averaged, per-session z-scored
    data/s1_images.npy   [N_img,256,256,3] uint8   the stimulus images S1 saw
    data/s1_split.npz    train_idx / val_idx / test_idx   (test = shared-1000)
    data/s1_ncsnr.npy    [V] float32   per-voxel ncsnr (for the noise ceiling)
    data/s1_global_idx.npy [N_img] int   0-based NSD id per row (for streaming imgs)

Runs on the L4 VM (needs aws + s3fs + h5py + nibabel).  Betas are downloaded one
session at a time, distilled to V voxels, then the ~469 MB raw .mgh is deleted —
so peak disk stays ~1 GB even though the full subj01 betas are 36.6 GB.

    python prep_s1.py --step betas     # download+distill fMRI (resumable per session)
    python prep_s1.py --step stimuli   # stream the ~10k S1 images from the remote hdf5
    python prep_s1.py --step all
"""
import argparse
import subprocess
from pathlib import Path

import numpy as np

SUBJ = 1
N_SESSIONS = 40
S3_BETAS = ("s3://natural-scenes-dataset/nsddata_betas/ppdata/subj01/"
            "fsaverage/betas_fithrf_GLMdenoise_RR")
S3_STIM = "natural-scenes-dataset/nsddata_stimuli/stimuli/nsd/nsd_stimuli.hdf5"

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
NSD = HERE.parent / "brainit-fmri" / "data" / "nsd_data"
MASKS = NSD / "roi_masks" / f"sub{SUBJ}"
EXPDESIGN = NSD / "experiment" / "nsd_expdesign.mat"
RAW = DATA / "raw_betas"           # transient .mgh downloads
SESS = DATA / "sess"               # distilled per-session [trials, V] float16


def sh(cmd):
    print("+", cmd, flush=True)
    subprocess.run(cmd, shell=True, check=True)


def load_masks():
    lh = np.load(MASKS / "lh.all-vertices_fsaverage_space.npy").astype(bool)
    rh = np.load(MASKS / "rh.all-vertices_fsaverage_space.npy").astype(bool)
    return lh, rh


def distill_session(ses, lh_mask, rh_mask):
    """Download lh/rh betas for one session, subset+concat+zscore, delete raw."""
    import nibabel as nib
    out = SESS / f"sess{ses:02d}.npy"
    if out.exists():
        print(f"  session {ses:02d}: skip (exists)")
        return
    RAW.mkdir(parents=True, exist_ok=True)
    cols = []
    for hem, mask in (("lh", lh_mask), ("rh", rh_mask)):
        fn = RAW / f"{hem}.betas_session{ses:02d}.mgh"
        if not fn.exists():
            sh(f"aws s3 cp --no-sign-request {S3_BETAS}/{hem}.betas_session{ses:02d}.mgh {fn}")
        arr = nib.load(str(fn)).get_fdata()[:, 0, 0, :].transpose()   # [trials, verts]
        cols.append(arr[:, mask])
        fn.unlink()                                                   # free ~469 MB now
    Y = np.concatenate(cols, axis=1)                                  # [trials, V]
    Y = (Y - Y.mean(0, keepdims=True)) / Y.std(0, keepdims=True)      # per-session z-score
    Y = np.nan_to_num(Y).astype(np.float16)
    SESS.mkdir(parents=True, exist_ok=True)
    np.save(out, Y)
    print(f"  session {ses:02d}: {Y.shape} -> {out.name}")


def distill_ncsnr(lh_mask, rh_mask):
    import nibabel as nib
    out = DATA / "s1_ncsnr.npy"
    if out.exists():
        return
    cols = []
    for hem, mask in (("lh", lh_mask), ("rh", rh_mask)):
        fn = RAW / f"{hem}.ncsnr.mgh"
        if not fn.exists():
            sh(f"aws s3 cp --no-sign-request {S3_BETAS}/{hem}.ncsnr.mgh {fn}")
        arr = np.asarray(nib.load(str(fn)).get_fdata()).squeeze()     # [verts]
        cols.append(arr[mask])
        fn.unlink()
    ncsnr = np.concatenate(cols).astype(np.float32)
    np.save(out, ncsnr)
    print(f"  ncsnr: {ncsnr.shape} -> {out.name}")


def build_betas():
    """Combine per-session trials in order, map to images, average repeats, split."""
    from scipy.io import loadmat
    mat = loadmat(str(EXPDESIGN))
    masterordering = mat["masterordering"][0, :] - 1     # [30000] trial -> 0-based session-image
    subjectim = mat["subjectim"] - 1                     # [8, 10000] (subj,img) -> 0-based NSD id
    sub_to_global = subjectim[SUBJ - 1]                  # [10000]

    # trials in session-concatenated order (session 1 first, ...) — matches masterordering
    trials = np.concatenate([np.load(SESS / f"sess{s:02d}.npy") for s in range(1, N_SESSIONS + 1)])
    n_trials, V = trials.shape
    print(f"[betas] {n_trials} trials x {V} voxels")

    global_of_trial = sub_to_global[masterordering[:n_trials]]        # [n_trials] 0-based NSD id
    uniq = np.unique(global_of_trial)
    # average the (up to 3) repeats of each image
    Y = np.zeros((len(uniq), V), dtype=np.float32)
    g2row = {g: i for i, g in enumerate(uniq)}
    counts = np.zeros(len(uniq))
    for t in range(n_trials):
        r = g2row[global_of_trial[t]]
        Y[r] += trials[t]
        counts[r] += 1
    Y /= counts[:, None]

    # test = the shared-1000 images (first 1000 of subjectim[0]) that S1 saw
    shared = set((subjectim[0][:1000]).tolist())
    is_test = np.array([g in shared for g in uniq])
    test_idx = np.where(is_test)[0]
    pool = np.where(~is_test)[0]
    rng = np.random.default_rng(0)
    val_idx = rng.choice(pool, size=min(2500, len(pool) // 5), replace=False)
    train_idx = np.setdiff1d(pool, val_idx)

    np.save(DATA / "s1_betas.npy", Y.astype(np.float16))
    np.save(DATA / "s1_global_idx.npy", uniq.astype(np.int64))
    np.savez(DATA / "s1_split.npz", train_idx=train_idx, val_idx=val_idx, test_idx=test_idx)
    print(f"[betas] images={len(uniq)}  train={len(train_idx)} val={len(val_idx)} "
          f"test={len(test_idx)} (shared-1000 seen by S1)")


def build_stimuli():
    """Stream only the S1 images out of the remote 37 GB hdf5 (ranged reads)."""
    import h5py, s3fs
    from PIL import Image
    global_idx = np.load(DATA / "s1_global_idx.npy")
    fs = s3fs.S3FileSystem(anon=True)
    with fs.open(S3_STIM, "rb") as f:
        h5 = h5py.File(f, "r")
        brick = h5["imgBrick"]           # [73000, H, W, 3] uint8
        print(f"[stimuli] imgBrick {brick.shape} {brick.dtype}; fetching {len(global_idx)} rows")
        out = np.zeros((len(global_idx), 256, 256, 3), dtype=np.uint8)
        for i, g in enumerate(global_idx):
            im = brick[int(g)]                                    # ranged read of one image
            if im.shape[:2] != (256, 256):
                im = np.asarray(Image.fromarray(im).resize((256, 256), Image.BICUBIC))
            out[i] = im
            if i % 500 == 0:
                print(f"  {i}/{len(global_idx)}", flush=True)
    np.save(DATA / "s1_images.npy", out)
    print(f"[stimuli] saved s1_images.npy {out.shape}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", choices=["betas", "stimuli", "all"], default="all")
    args = ap.parse_args()
    DATA.mkdir(parents=True, exist_ok=True)
    if args.step in ("betas", "all"):
        lh, rh = load_masks()
        for s in range(1, N_SESSIONS + 1):
            distill_session(s, lh, rh)
        distill_ncsnr(lh, rh)
        build_betas()
    if args.step in ("stimuli", "all"):
        build_stimuli()


if __name__ == "__main__":
    main()
