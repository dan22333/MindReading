#!/usr/bin/env python3
"""Regenerate the S1 functional-voxel set + per-ROI masks for original BrainDiVE
entirely from PUBLIC NSD files + the repo's rank_by_HCP logic. No gated betas, no
author-only files. Validates the rebuilt set against the released encoder dims.

Downloads (small, public NSD S3 bucket, HTTP 206, no creds) — see fetch_nsd.sh.
Needs: nsd/subj0X/roi/{floc-*,prf-visualrois,nsdgeneral,HCP_MMP1}.nii.gz,
       nsd/subj0X/floc_{faces,bodies,places,word}tval.nii.gz,
       nsd/subj0X/ncsnr.nii.gz  (X=1..8 for best_HCP; X=1 for the masks),
       food_voxels/food_s1.npy  (from issue #2 food_voxels.zip).

Expected result for S1: early=4657, higher=22961 (EXACT match to S1/00100.chkpt).
"""
import numpy as np, nibabel as nib, os

NSD = "nsd"
def L(p): return nib.load(p).get_fdata()

# --- 1. best_HCP.npy: rank HCP-MMP1 regions by mean ncsnr per subject, avg rank across 8 ---
def build_best_hcp():
    per_subj = {}
    for s in range(1, 9):
        ncsnr = L(f"{NSD}/subj0{s}/ncsnr.nii.gz")
        hcp = L(f"{NSD}/subj0{s}/roi/HCP_MMP1.nii.gz").astype(np.int64)
        uids = np.unique(hcp); uids = uids[uids > 0.5]
        ids, ncs = [], []
        for i in uids:
            ids.append(i); ncs.append(np.nanmean(ncsnr[hcp == i]))
        ids, ncs = np.array(ids), np.array(ncs)
        per_subj[s] = ids[np.argsort(ncs)[::-1]]
    roi_list = np.arange(1, 181)
    ranks = np.array([[per_subj[su].tolist().index(i) for su in range(1, 9)] for i in roi_list])
    best = roi_list[np.argsort(ranks.mean(axis=1))]
    np.save("best_HCP.npy", best)
    return best

# --- 2. functional mask (S1) = OR(prf, floc4, food, HCP-top45) ---
def build_functional(best_hcp, subj=1):
    d = f"{NSD}/subj0{subj}"
    prf = L(f"{d}/roi/prf-visualrois.nii.gz")
    sel = [prf >= 0.5]
    for r in ["bodies", "faces", "places", "words"]:
        sel.append(L(f"{d}/roi/floc-{r}.nii.gz") >= 0.5)
    sel.append(np.load(f"food_voxels/food_s{subj}.npy") >= 0.5)
    hcp = L(f"{d}/roi/HCP_MMP1.nii.gz").astype(np.int32)
    cont = np.zeros_like(hcp, float)
    for i in best_hcp[:45]:
        cont[hcp == i] += 1.0
    sel.append(cont >= 0.5)
    functional = np.logical_or.reduce(sel)
    np.save(f"functional_S{subj}.npy", functional)
    np.save(f"early_vis_mask_S{subj}.npy", (prf > 0.5)[functional])
    np.save(f"higher_vis_mask_S{subj}.npy", (prf < 0.5)[functional])
    return functional, prf

# --- 3. per-ROI masks over functional voxels ---
def build_roi_masks(functional, prf, subj=1):
    d = f"{NSD}/subj0{subj}"; os.makedirs("masks", exist_ok=True)
    tval_name = {"faces": "faces", "bodies": "bodies", "places": "places", "words": "word"}
    for r in ["faces", "bodies", "places", "words"]:
        roi = L(f"{d}/roi/floc-{r}.nii.gz")
        tval = L(f"{d}/floc_{tval_name[r]}tval.nii.gz")
        refined = (roi > 0.5) & (tval > 2)          # author's t>2 definition (issue #2)
        np.save(f"masks/{r}_S{subj}.npy", refined[functional])
    np.save(f"masks/earlyvisual_S{subj}.npy", (prf > 0.5)[functional])

if __name__ == "__main__":
    best = build_best_hcp()
    print("best_HCP top-45:", best[:45].tolist())
    functional, prf = build_functional(best)
    e = int((functional & (prf > 0.5)).sum()); h = int((functional & (prf < 0.5)).sum())
    print(f"S1 functional={int(functional.sum())} early={e} higher={h}")
    assert e == 4657 and h == 22961, "MISMATCH — reproduction not aligned to encoder!"
    build_roi_masks(functional, prf)
    print("OK — masks written to masks/, exact match to released S1 encoder.")
