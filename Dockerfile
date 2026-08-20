# brainguide — SD v1.5 + universal brain encoder, guided diffusion.
# Build context is the MindReading repo root (needs brainguide/ + brainit-fmri/).
#   docker build -t brainguide .
# Weights are NOT baked (mount them at runtime) — see RUN_ON_VM.md.
FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime

# git: torch.hub fetches the DINOv2 repo code at runtime.
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

# --- deps layer (cached unless this list changes) ---------------------------
# torch / torchvision / numpy come from the base image; install the rest.
# Pinned to the era the patched pipeline was written for: transformers 4.40 still
# exposes CLIPFeatureExtractor, and diffusers 0.27 matches the pipeline's imports.
RUN pip install --no-cache-dir \
      "diffusers==0.27.2" \
      "transformers==4.40.2" \
      "huggingface_hub<0.26" \
      accelerate safetensors \
      pyyaml pillow matplotlib \
      gdown

# --- code layer (rebuilds only when source changes) -------------------------
# Layout mirrors the Mac: /app == MindReading root, so external.py resolves
# code_path "brainit-fmri" to /app/brainit-fmri exactly as it does locally.
WORKDIR /app
COPY brainguide/ /app/brainguide/
COPY brainit-fmri/ /app/brainit-fmri/
ENV PYTHONPATH=/app/brainit-fmri:${PYTHONPATH}

WORKDIR /app/brainguide
# Config selects the encoder; override args at `docker run`.
ENTRYPOINT ["python", "run.py"]
CMD ["--config", "configs/universal.yaml", "--prompt", "a photograph"]
