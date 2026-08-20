"""The BrainDiVE-style guided Stable Diffusion pipeline.

``mypipelineSAG`` is a StableDiffusion pipeline whose ``cond_fn`` steers latents
along the gradient of ``self.brain_tweak(image) -> scalar`` (see
brain_guide_pipeline.py:506). Assign that hook from a BrainEncoder via
``brainguide.make_brain_tweak`` and the sampler pushes samples toward images the
encoder scores highly.
"""
from .brain_guide_pipeline import mypipelineSAG  # noqa: F401

__all__ = ["mypipelineSAG"]
