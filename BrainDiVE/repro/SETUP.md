# Original BrainDiVE — faithful S1 reproduction (to compare vs our brainguide panel)

Runs the **original** BrainDiVE (aluo-x/BrainDiVE, NeurIPS 2023) with *their* encoder
+ *their* NSD ROIs, on subject 1, for the same ROIs as our `brainguide/` panel
(faces, bodies, places, words, earlyvisual). Everything here is rebuilt from public
assets — released encoder weights + public NSD ROI files. See memory
`braindive-original-reproduction`.

## What's already in this folder (rebuilt + validated on 2026-08-20)
- `best_HCP.npy`, `functional_S1.npy`, `early_vis_mask_S1.npy`, `higher_vis_mask_S1.npy`
- `masks/{faces,bodies,places,words,earlyvisual}_S1.npy` — bool over the 27618 functional voxels
- `prep_masks.py` — regenerates all of the above from `nsd/` + validates early=4657/higher=22961
- `fetch_nsd.sh` — pulls the small public NSD files + food masks + points at the weights zip
- `run_repro_S1.py` — the generation runner (their exact recipe)

## Encoder recipe (theirs)
Linear layer on the unit-normalized OpenCLIP ViT-B/16 (`timm vit_base_patch16_clip_224.laion2b`)
last-layer token → per-voxel activation. Weights: `braindive_encoder_weights/S1/00100.chkpt`
(load with `weights_only=True`; dims read straight off the state_dict, so NO neural data needed).

## Generation recipe (theirs)
SD2.1-base fp16 + their `mypipelineSAG` (SAG) + `DPMSolverMultistepScheduler`:
`pipe("", sag_scale=0.75, guidance_scale=0.0, num_inference_steps=50, clip_guidance_scale=130.0)`.
Empty prompt = pure brain-discovery. loss = `-mean(pred_response[roi_mask])`.

## Environment (SEPARATE from our brainguide diffusers==0.27 stack)
Their `brain_guide_pipeline.py` uses pre-May-2023 diffusers APIs
(`diffusers.pipeline_utils`, `randn_tensor` in `diffusers.utils`). Use ~diffusers 0.15.x:
```bash
conda create -n braindive python=3.10 -y && conda activate braindive
pip install "torch==2.0.1" "diffusers==0.15.1" "transformers==4.27.4" \
            "accelerate" "timm" "nibabel" "numpy<2" "huggingface_hub==0.16.4" \
            "safetensors" "matplotlib" "pillow"
```
Needs a GPU (backprop through CLIP ViT every step) — use the T4 VM (see
`../../brainguide/RUN_ON_VM.md` for VM lifecycle; this is a NEW env, not the brainguide image).

## Run
```bash
cd BrainDiVE           # so `import encoder_model_vit, brain_guide_pipeline` resolve
python repro/run_repro_S1.py \
  --ckpt braindive_encoder_weights/S1/00100.chkpt \
  --repro repro --out repro/outputs \
  --rois faces bodies places words earlyvisual --seeds 0 1 2 3 4
```
~ a few s/image on a T4. Then compare `repro/outputs/*` against
`../brainguide/outputs/final/` (our universal-encoder panel) — same ROIs, same subject.
