"""Adapter for Paper 1 — "The Wisdom of a Crowd of Brains: A Universal Brain
Encoder".

The released model is a frozen DINOv2 backbone tapped at blocks 1/6/12/18/24, a
per-voxel 256-d embedding table used twice (cross-attention query + readout),
and a shared [256,640] readout dictionary. It maps (image, voxel_ids) -> per-
voxel scalar, which is exactly our BrainEncoder contract.

The real repo is NOT vendored in this tree. Point ``code_path`` at the cloned
repo (or symlink it into ``universal-brain-encoder-code/``) and set
``checkpoint``. The only coupling point you may need to touch is
``_forward_model`` — its call signature must match the actual release. Everything
else (preprocessing, our (img01, voxels) -> [B,V] contract, gradient flow) is
handled here.
"""
import importlib

import torch
import torch.nn.functional as F

from ..external import add_to_path, resolve
from .base import BrainEncoder


class UniversalEncoder(BrainEncoder):
    def __init__(self, code_path="universal-brain-encoder-code",
                 module="models.encoder_models", class_name="BrainEncodingModel",
                 model_kwargs=None, checkpoint=None, subject=None,
                 backbone_image_size=224, device="cpu", dtype=torch.float32):
        super().__init__()
        add_to_path(code_path, label="universal-brain-encoder-code")
        mod = importlib.import_module(module)
        Model = getattr(mod, class_name)

        self.model = Model(**(model_kwargs or {}))
        if checkpoint:
            sd = torch.load(str(resolve(checkpoint)), map_location="cpu")
            sd = sd.get("state_dict", sd) if isinstance(sd, dict) else sd
            missing, unexpected = self.model.load_state_dict(sd, strict=False)
            if missing or unexpected:
                print(f"[universal] load_state_dict: {len(missing)} missing, "
                      f"{len(unexpected)} unexpected keys (strict=False)")
        # Frozen weights; gradients still flow through to the image for steering.
        self.model.requires_grad_(False)
        self.model.eval()

        self.subject = subject
        self.image_size = backbone_image_size
        # DINOv2 expects ImageNet normalization.
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406])[None, :, None, None])
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225])[None, :, None, None])
        self.to(device=device, dtype=dtype)

    def _preprocess(self, img01: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(img01, size=self.image_size, mode="bicubic", align_corners=False)
        return (x - self.mean) / self.std

    def _forward_model(self, x: torch.Tensor, voxels: torch.Tensor) -> torch.Tensor:
        """ADAPT to the real release signature if arg names differ.
        Expected to return [B, len(voxels)] activations."""
        return self.model(x, voxels=voxels, subject=self.subject)

    def forward(self, img01: torch.Tensor, voxels=None) -> torch.Tensor:
        if voxels is None:
            raise ValueError(
                "UniversalEncoder needs an explicit voxel selection (the target "
                "ROI). Passing all ~316k voxels at once is almost never intended."
            )
        x = self._preprocess(img01.to(self.mean.dtype))
        return self._forward_model(x, voxels)  # [B, V]
