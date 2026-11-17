#!/usr/bin/env bash
# Fetch the small PUBLIC NSD ROI/ncsnr files needed to rebuild the S1 masks, the
# released encoder weights, and the food masks. All public (NSD bucket = HTTP 206,
# no creds). NSD asks that you agree to their terms of use before using the data.
set -euo pipefail
NSD="https://natural-scenes-dataset.s3.amazonaws.com/nsddata/ppdata"
BETAS="https://natural-scenes-dataset.s3.amazonaws.com/nsddata_betas/ppdata"

# ncsnr + HCP_MMP1 for all 8 subjects (for best_HCP ranking)
for s in 1 2 3 4 5 6 7 8; do
  mkdir -p nsd/subj0$s/roi
  curl -s -o nsd/subj0$s/roi/HCP_MMP1.nii.gz "$NSD/subj0$s/func1pt8mm/roi/HCP_MMP1.nii.gz"
  curl -s -o nsd/subj0$s/ncsnr.nii.gz "$BETAS/subj0$s/func1pt8mm/betas_fithrf_GLMdenoise_RR/ncsnr.nii.gz"
done

# subject-1 ROI + tval files (for the functional + per-ROI masks)
d=nsd/subj01; mkdir -p $d/roi
for f in roi/floc-faces roi/floc-bodies roi/floc-places roi/floc-words \
         roi/prf-visualrois roi/nsdgeneral; do
  curl -s -o $d/$f.nii.gz "$NSD/subj01/func1pt8mm/$f.nii.gz"
done
for t in facestval bodiestval placestval wordtval; do
  curl -s -o $d/floc_$t.nii.gz "$NSD/subj01/func1pt8mm/floc_$t.nii.gz"
done

# food masks (author, issue #2 attachment)
curl -sL -o food_voxels.zip "https://github.com/user-attachments/files/20856843/food_voxels.zip"
unzip -o -q food_voxels.zip -d food_voxels

# released encoder weights (2.5 GB, pCloud; issue #2). Extract S1 (or all).
# Direct link host can rotate; if this 404s, re-resolve via:
#   curl -s "https://api.pcloud.com/getpublinkdownload?code=XZhLWj5Zd01s27dcrQ870TbFptYiNRUMWj57"
echo "Now download braindive_encoder_weights.zip from:"
echo "  https://u.pcloud.link/publink/show?code=XZhLWj5Zd01s27dcrQ870TbFptYiNRUMWj57"
echo "and: unzip -o braindive_encoder_weights.zip  (gives braindive_encoder_weights/S1/00100.chkpt)"
