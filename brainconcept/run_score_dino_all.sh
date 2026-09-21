#!/bin/bash
# One docker pass: load DINOv2 once, score all runs. Args = run names.
sudo docker run --rm --gpus all \
  -v ~/MindReading/brainguide:/app/brainguide \
  -v ~/MindReading/brainit-fmri:/app/brainit-fmri \
  -v ~/MindReading/brainconcept:/work \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -v ~/.cache/torch:/root/.cache/torch \
  -w /app/brainguide -e PYTHONPATH=/app/brainguide:/app/brainit-fmri \
  --entrypoint python brainguide \
  /work/dino_all.py --roi faces --out /work/RESULTS/heldout --runs "$@"
