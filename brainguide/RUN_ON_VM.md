# Running brainguide on a GCP GPU VM

SD v1.5 + the universal encoder (DINOv2 backprop every diffusion step) needs a
GPU. This is the mount-not-bake path: the image carries **code**; weights are
downloaded once on the VM and **mounted** into the container.

## 0. Prerequisite — GPU quota (one-time, may need approval)

GCP gates GPUs behind a project-wide quota `GPUS_ALL_REGIONS`. For this project
it is currently **0**, so VM creation will fail until it's raised.

Request it (Console is the reliable path):
1. https://console.cloud.google.com/iam-admin/quotas?project=capsule-487202
2. Filter for **"GPUs (all regions)"**, check it, **Edit Quotas** → request **1**.
3. Also ensure a regional GPU quota (e.g. `NVIDIA_T4_GPUS` in us-central1 = 1 — already true here).

Approval for a single GPU on an established billing account is often minutes to
a day. Until it clears, nothing below will provision.

## 1. Create the VM (T4, cheap; swap to V100 for ~3× speed)

```bash
ZONE=us-central1-b
gcloud compute instances create brainguide-gpu \
  --project=capsule-487202 --zone=$ZONE \
  --machine-type=n1-standard-8 \
  --accelerator=type=nvidia-tesla-t4,count=1 \
  --maintenance-policy=TERMINATE \
  --image-family=common-cu121-debian-11 \
  --image-project=deeplearning-platform-release \
  --boot-disk-size=100GB \
  --metadata=install-nvidia-driver=True
```
The Deep Learning image ships Docker + NVIDIA driver + nvidia-container-toolkit
(first boot installs the driver — give it ~2 min).

## 2. Get the code onto the VM

Push only the small `brainguide/` code; re-clone the encoder repo on the VM
(fast from GCP), so no multi-GB transfer from your laptop.

```bash
# from MindReading/ on your Mac:
gcloud compute scp --recurse --zone=$ZONE brainguide brainguide-gpu:~/MindReading/brainguide
gcloud compute scp --zone=$ZONE Dockerfile .dockerignore brainguide-gpu:~/MindReading/
gcloud compute ssh brainguide-gpu --zone=$ZONE --command \
  "cd ~/MindReading && git clone --depth 1 https://github.com/WeizmannVision/brainit-fmri.git"
```

## 3. Build the image on the VM (avoids pushing a 2 GB image)

```bash
gcloud compute ssh brainguide-gpu --zone=$ZONE
# --- on the VM ---
cd ~/MindReading
sudo docker build -t brainguide .
```

## 4. Download weights on the VM (into a host dir → mounted, not baked)

```bash
# --- on the VM, inside ~/MindReading ---
pip install huggingface_hub
python brainit-fmri/data/scripts/download/download_checkpoints.py   # encoder_ch128.pth
# (optional, for a real ROI) pip install gdown && bash brainit-fmri/data/scripts/download/download_roi_masks
```

## 5. Run — GPU + mounted weights/caches/outputs

```bash
mkdir -p ~/outputs
sudo docker run --gpus all --rm \
  -v ~/MindReading/brainit-fmri/results:/app/brainit-fmri/results \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -v ~/.cache/torch:/root/.cache/torch \
  -v ~/outputs:/app/brainguide/outputs \
  brainguide --config configs/universal.yaml --prompt "" \
    --voxels brainguide/roi/sub1_faces.npy --tag sub1_faces --clip-scale 0.30 --seed 0
```
`run.py` auto-selects `cuda` + fp16. First run also downloads SD v1.5 and DINOv2
into the mounted caches (once). Images land in `~/outputs`.

**Use `--prompt ""`** for BrainDiVE pure-discovery (the brain gradient alone
drives content — a text prompt would bias every ROI toward that text).
**`--clip-scale 0.30`** is the tuned dimensionless guidance strength for the
ROI-independent schedule; it is NOT the old raw-grad value (~100). Loop over ROIs
by swapping `--voxels brainguide/roi/sub1_<roi>.npy --tag sub1_<roi>` for
`<roi>` in {faces, bodies, words, places, earlyvisual}.

> **IMPORTANT — rebuild after any code change.** The image bakes a *copy* of the
> code (see the `COPY` lines in the Dockerfile). If you edit the pipeline/config
> and forget to `sudo docker build -t brainguide .`, the container silently runs
> the OLD code — e.g. the pre-fix `cond_fn` treats `clip_guidance_scale` as a raw
> ~100 scale, so `0.30` becomes ~300× too weak and every ROI collapses to the same
> unsteered image with no error. Rebuild, or mount live code with
> `-v ~/MindReading/brainguide:/app/brainguide` while iterating.

## 6. Pull results back + stop the VM (stop billing!)

```bash
gcloud compute scp --recurse --zone=$ZONE brainguide-gpu:~/outputs ./vm_outputs
gcloud compute instances stop brainguide-gpu --zone=$ZONE   # or `delete` to remove the disk too
```

## Alternative: push the image instead of building on the VM

If you prefer to build locally and ship it:
```bash
gcloud artifacts repositories create brainguide --repository-format=docker --location=us-central1
docker tag brainguide us-central1-docker.pkg.dev/capsule-487202/brainguide/brainguide:latest
docker push us-central1-docker.pkg.dev/capsule-487202/brainguide/brainguide:latest
# then on the VM: sudo docker pull <same path> && run as above
```
Build-on-VM (step 3) is usually faster since it avoids uploading ~2 GB from home.

## Cost note
T4 + n1-standard-8 ≈ $0.75/hr; V100 ≈ $2.85/hr. **Stop the instance** when idle.
