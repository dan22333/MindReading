# Running brainconcept on the T4 VM

Everything heavy (model downloads, encoder training, the token loop) runs on the VM.
Code is authored on the host and mounted in — do **not** rebuild a stale Docker image
without mounting host code (see the brainguide VM notes).

## 0. Environment
```bash
cd brainconcept/vendor_conceptlab
conda env create -f environment/env.yaml      # ConceptLab's env (kandinsky2 + diffusers)
conda activate concept-lab
pip install git+https://github.com/openai/CLIP.git   # for train_encoder.py (ViT-L/14)
pip install scikit-learn                              # RidgeCV
```
BLIP-2 (`Salesforce/blip2-flan-t5-xl`) and Kandinsky 2.1 auto-download on first run
(Kandinsky into `cache_root=/tmp/kandinsky2`).

## 1. Get NSD data for S1  (only you can do this — data-access agreement)
Download NSD subject 1: the stimulus images the subject viewed + the fMRI betas +
the ROI masks (floc-faces/bodies/places/words, prf-visualrois). Produce four arrays:
```
data/s1_images.npy   # [N,H,W,3] uint8   stimulus images S1 saw
data/s1_betas.npy    # [N, V_all] float  z-scored, repeats averaged
data/s1_roi_faces.npy# [V] int           indices into V_all for the face ROI
data/s1_split.npz    # train_idx / val_idx / test_idx  (test = the ~903 shared images)
```
(Adapt `train_encoder.py::load_nsd_s1` if your download layout differs, e.g. brainit-fmri's fmri_v2.npz.)

## 2. Stage 1 — train the fMRI encoder W  (CLIP ViT-L/14 image emb -> fMRI)
```bash
cd brainconcept
python train_encoder.py \
  --images_npy data/s1_images.npy --betas_npy data/s1_betas.npy \
  --roi_npy data/s1_roi_faces.npy --split_npy data/s1_split.npz \
  --roi_name faces --out vendor_conceptlab/weights/W_faces_s1.pt
```
Gate: the printed **test median voxel Pearson r** must beat the noise floor before you trust W.

## 3. Stages 2–4 — the creative loop (token optimization + VLM negatives)
```bash
cd brainconcept/vendor_conceptlab
python -m scripts.train --config_path configs/brain_faces_s1.yaml
```
This runs ConceptLab's optimizer with the brain objective:
- every step: `z = prior(v*)`, `loss = -mean(W·z) + 0.5(mean+max) cos(z, negatives)`, step `v*`
- every 250 steps: generate images, BLIP-2 names them, the name is added as a negative
Outputs (images, embeddings, distance plots) land in `outputs/brain_faces_s1/`.

## 4. Other ROIs
Train a W per ROI (`--roi_npy data/s1_roi_places.npy --roi_name places`) and point a
copy of the config's `brain_encoder_weights` at it.

## Stop criteria (watch the logs / distance plot)
- `brain_roi_mean` stops rising / falls below ~0.7× its round-1 peak for 2 rounds, or
- BLIP-2 keeps repeating concepts already in the negative set (no new names), or
- a fixed cap (e.g. steps=2500 ≈ 10 negative rounds at freq 250).
