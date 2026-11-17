#!/usr/bin/env bash
# Runs ON the T4 VM. BrainDiVE/ (their code + repro/ with committed+validated masks)
# already scp'd to ~/BrainDiVE. Builds the pre-2023 diffusers venv, downloads the
# released encoder weights (only heavy asset needed — masks are already committed and
# run_repro_S1.py self-validates the 4657/22961 counts against the checkpoint), then
# runs BOTH the no-encoder control and the real brain-guided S1 generation.
set -euo pipefail
cd ~/BrainDiVE

# ---- 1. venv (pre-May-2023 diffusers for their brain_guide_pipeline; no conda on VM) ----
if [ ! -d ~/braindive_venv ]; then
  python3 -m venv ~/braindive_venv
  source ~/braindive_venv/bin/activate
  pip install --upgrade pip -q
  # torch 2.0.1 cu118 wheels run fine on the VM's newer driver (580, backward-compatible)
  pip install -q "torch==2.0.1" --index-url https://download.pytorch.org/whl/cu118
  pip install -q "diffusers==0.15.1" "transformers==4.27.4" \
              accelerate timm nibabel "numpy<2" "huggingface_hub==0.16.4" \
              safetensors matplotlib pillow open_clip_torch
else
  source ~/braindive_venv/bin/activate
fi
python -c "import torch;print('torch',torch.__version__,'cuda',torch.cuda.is_available(),torch.cuda.get_device_name(0))"

# ---- 2. released encoder weights (re-resolve pCloud direct link; ~2.5 GB) ----
cd repro
if [ ! -f braindive_encoder_weights/S1/00100.chkpt ]; then
  URL=$(curl -s "https://api.pcloud.com/getpublinkdownload?code=XZhLWj5Zd01s27dcrQ870TbFptYiNRUMWj57" \
        | python3 -c 'import sys,json;d=json.load(sys.stdin);print("https://"+d["hosts"][0]+d["path"])')
  echo "weights URL: $URL"
  curl -L -o braindive_encoder_weights.zip "$URL"
  unzip -o -q braindive_encoder_weights.zip
fi
ls -la braindive_encoder_weights/S1/00100.chkpt

# ---- 3a. CONTROL: plain SD, no encoder (confirms the diffusion pipeline works) ----
cd ~/BrainDiVE
python repro/run_control_plainSD.py --out repro/outputs_control --seeds 0 1 2 3 4

# ---- 3b. REAL: brain-guided S1, their exact recipe (self-validates voxel counts) ----
python repro/run_repro_S1.py \
  --ckpt repro/braindive_encoder_weights/S1/00100.chkpt \
  --repro repro --out repro/outputs \
  --rois faces bodies places words earlyvisual --seeds 0 1 2 3 4

echo "ALL_DONE control=repro/outputs_control guided=repro/outputs"
