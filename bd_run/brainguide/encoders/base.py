"""Encoder interface + the image->scalar factory for pipe.brain_tweak."""
from abc import ABC, abstractmethod

import torch

# OpenAI-CLIP normalization the diffusion pipeline applies to the image it hands
# to brain_tweak (brain_guide_pipeline_patched.py:497-505). We invert it so each
# encoder can apply its OWN preprocessing, keeping the graph differentiable.
_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
_CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def clip_denormalize(img_clipnorm: torch.Tensor) -> torch.Tensor:
    """CLIP-normalized image -> [0,1] image (differentiable)."""
    mean = img_clipnorm.new_tensor(_CLIP_MEAN)[None, :, None, None]
    std = img_clipnorm.new_tensor(_CLIP_STD)[None, :, None, None]
    return (img_clipnorm * std + mean).clamp(0.0, 1.0)


class BrainEncoder(torch.nn.Module, ABC):
    """Maps an image to per-voxel predicted activation.

    Contract:
        forward(img01: FloatTensor[B,3,H,W] in [0,1],
                voxels: LongTensor[V] | None) -> FloatTensor[B, V]

    Must be differentiable wrt ``img01`` when used to *steer*. Subclasses apply
    their own resize + normalization inside forward(). A voxel-selection is the
    target ROI whose activation we push up (or down).
    """

    #: steering encoders are True; a locked validation encoder sets this False
    is_differentiable: bool = True

    @abstractmethod
    def forward(self, img01: torch.Tensor, voxels=None) -> torch.Tensor:  # pragma: no cover
        ...

    @torch.no_grad()
    def predict_nograd(self, img01: torch.Tensor, voxels=None) -> torch.Tensor:
        """For the validation encoder: score without ever building a graph."""
        return self.forward(img01, voxels)


def make_brain_tweak(encoder: BrainEncoder, voxels=None, sign: float = 1.0,
                     reduce: str = "mean"):
    """Build the ``image -> scalar`` callable assigned to ``pipe.brain_tweak``.

    The sampler steps latents along +grad(scalar) (it computes
    ``grads = -grad(loss)`` then subtracts, brain_guide_pipeline_patched.py:507),
    so the returned scalar is the quantity to **maximize**. Use ``sign=-1.0`` to
    suppress a region instead of driving it.
    """
    if isinstance(voxels, (list, tuple)):
        voxels = torch.as_tensor(voxels, dtype=torch.long)
    try:
        reduce_fn = {"mean": torch.mean, "sum": torch.sum}[reduce]
    except KeyError:
        raise ValueError(f"reduce must be 'mean' or 'sum', got {reduce!r}")

    def brain_tweak(img_clipnorm: torch.Tensor) -> torch.Tensor:
        v = voxels.to(img_clipnorm.device) if voxels is not None else None
        img01 = clip_denormalize(img_clipnorm)
        acts = encoder(img01, v)  # [B, V]
        return sign * reduce_fn(acts)

    return brain_tweak
