"""brainguide — modular BrainDiVE-style guided-diffusion harness.

The whole coupling between a brain model and the diffusion sampler is a single
function ``image -> scalar`` assigned to ``pipe.brain_tweak`` (see
brain_guide_pipeline_patched.py:506). This package makes the *scorer* side of
that contract pluggable:

    encoder (BrainEncoder)  --make_brain_tweak-->  pipe.brain_tweak
    ------------------------------------------------------------------
    braindive_clip   linear head on L2-normalized CLIP  (runnable today)
    universal        Paper 1 "Wisdom of a Crowd of Brains" (drop-in)
    mock             random-weight CLIP head (smoke test)

Swap encoders by name in a YAML config; nothing else in the loop changes.
"""

from .encoders import BrainEncoder, build_encoder, make_brain_tweak  # noqa: F401

__all__ = ["BrainEncoder", "build_encoder", "make_brain_tweak"]
