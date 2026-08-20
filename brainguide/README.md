# brainguide

A clean, modular BrainDiVE-style guided-diffusion harness. The whole coupling
between a brain model and the Stable Diffusion sampler is a single function
`image -> scalar` assigned to `pipe.brain_tweak`. This project makes the *scorer*
side of that contract pluggable, so you can steer generation with any brain
encoder by changing one line of config.

```
encoder (BrainEncoder) --make_brain_tweak--> pipe.brain_tweak
--------------------------------------------------------------
braindive_clip   linear head on L2-normalized CLIP   (runnable today)
mock             braindive_clip with a random head   (smoke test)
universal        Paper 1 "Wisdom of a Crowd of Brains" (Stage B, drop-in)
```

## Layout

```
run.py                     CLI: config -> encoder -> pipe.brain_tweak -> generate
configs/                   one YAML per encoder; swap encoder.name, nothing else changes
brainguide/
  encoders/                base.py (contract) + braindive_clip / universal / registry
  pipeline/                the patched mypipelineSAG guided-diffusion pipeline
  external.py              resolves cross-project imports (universal encoder code)
outputs/                   generated images (gitignored)
```

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1. Validate the scorer + gradient contract (no model download):
python run.py --config configs/braindive_clip.yaml --smoke

# 2. Full guided generation (downloads SD v1.5 the first time):
python run.py --config configs/braindive_clip.yaml --prompt "a photograph of a face"
```

## Stage B — the universal encoder

`configs/universal.yaml` is a drop-in but needs the released model on disk:

1. Clone WeizmannVision's universal-brain-encoder, then symlink it so imports
   stay in-tree:
   ```bash
   ln -s /abs/path/to/universal-brain-encoder \
         "../universal-brain-encoder-code"
   ```
2. Set `encoder.checkpoint`, `encoder.subject`, and confirm `module` /
   `class_name` against the real repo.
3. Adapt `brainguide/encoders/universal.py::_forward_model` to the release's
   actual call signature (the only coupling point). Everything else — our
   `(img01, voxels) -> [B,V]` contract, preprocessing, gradient flow — is handled.

## Notes

- The sampler steps latents along `+grad(brain_tweak)`, so the scalar is the
  quantity to **maximize**; use `steer.sign: -1.0` to suppress a region.
- `braindive_clip` L2-normalizes the CLIP embedding to kill the radial gradient,
  so guidance can only rotate on the sphere, not inflate the norm.
