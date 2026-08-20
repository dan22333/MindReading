"""Adapter for Paper 1 — "The Wisdom of a Crowd of Brains: A Universal Brain
Encoder" (WeizmannVision/brainit-fmri).

The released encoder (``models/encoder_models.py::Encoder``) is a LoRA-adapted,
frozen DINOv2-reg backbone tapped at ``select_layers``, plus a per-voxel
embedding table used as the cross-attention query and the readout. It maps
(image, voxel_ind) -> per-voxel scalar, which is exactly our BrainEncoder
contract.

Construction mirrors ``train/train_encoder.py:155-183`` and the released
``encoder_ch128.pth`` (DINOv2 ViT-L/14-reg, in_channels=1024, inner_ch=128,
select_layers=[1,6,12,18,24]). ``num_voxels`` is read straight off the
checkpoint's ``voxel_embed`` so it can never drift from the weights.

Vendor the repo into ``brainit-fmri/`` (already cloned in this tree) and point
``checkpoint`` at ``results/saved_models/encoder_ch128.pth``.
"""
import os
import sys

import torch
import torch.nn.functional as F

from ..external import add_to_path, resolve
from .base import BrainEncoder


def _ensure_dinov2_importable():
    """The released checkpoint is a pickled Encoder whose backbone classes live
    in DINOv2's ``dinov2`` package. Put the torch.hub-cached repo on sys.path so
    unpickling can resolve them (downloads repo code once, no weights/model)."""
    try:
        import dinov2  # noqa: F401
        return
    except ModuleNotFoundError:
        pass
    torch.hub.list("facebookresearch/dinov2", trust_repo=True)  # fetch repo code
    repo = os.path.join(torch.hub.get_dir(), "facebookresearch_dinov2_main")
    if os.path.isdir(repo) and repo not in sys.path:
        sys.path.insert(0, repo)

#: torch.hub backbone name -> DINOv2 feature width, from train_encoder.py.
_BACKBONE_DIM = {
    "dinov2_vits14_reg": 384,
    "dinov2_vitb14_reg": 768,
    "dinov2_vitl14_reg": 1024,
    "dinov2_vitg14_reg": 1536,
}


def _clean_state_dict(sd):
    """Unwrap common checkpoint containers and strip a DataParallel prefix."""
    if isinstance(sd, dict):
        for key in ("state_dict", "model", "model_state_dict"):
            if key in sd and isinstance(sd[key], dict):
                sd = sd[key]
                break
    return {k[len("module."):] if k.startswith("module.") else k: v
            for k, v in sd.items()}


class UniversalEncoder(BrainEncoder):
    def __init__(self, code_path="brainit-fmri", checkpoint=None,
                 backbone="dinov2_vitl14_reg", in_channels=None, inner_ch=128,
                 select_layers=(1, 6, 12, 18, 24), r=3, alpha=1,
                 include_reg_tokens=False, num_reg_tokens=4, num_voxels=None,
                 backbone_image_size=224, device="cpu", dtype=torch.float32):
        super().__init__()
        if checkpoint is None:
            raise ValueError("UniversalEncoder needs `checkpoint` "
                             "(e.g. results/saved_models/encoder_ch128.pth)")
        add_to_path(code_path, label="brainit-fmri")
        from models.encoder_models import Encoder, encoder_param  # for unpickle + fallback

        # The released encoder_ch128.pth is a *pickled Encoder object* (whole
        # model, DINOv2 backbone included), not a state_dict. Load it directly;
        # fall back to the state_dict construction path if a dict is given.
        _ensure_dinov2_importable()
        obj = torch.load(str(resolve(checkpoint)), map_location="cpu",
                         weights_only=False)

        if isinstance(obj, torch.nn.Module):
            self.model = obj
        else:
            sd = _clean_state_dict(obj)
            if num_voxels is None:
                if "voxel_embed" not in sd:
                    raise KeyError("checkpoint has no `voxel_embed`; pass num_voxels")
                num_voxels = int(sd["voxel_embed"].shape[0])
            backbone_model = torch.hub.load("facebookresearch/dinov2", backbone)
            backbone_model.eval().requires_grad_(False)
            param = encoder_param(num_voxels)
            param.in_channels = in_channels or _BACKBONE_DIM[backbone]
            param.inner_ch = inner_ch
            self.model = Encoder(
                param, backbone_model, select_layers=list(select_layers),
                r=r, alpha=alpha, include_reg_tokens=include_reg_tokens,
                num_reg_tokens=num_reg_tokens,
            )
            missing, unexpected = self.model.load_state_dict(sd, strict=False)
            if missing or unexpected:
                print(f"[universal] load_state_dict(strict=False): "
                      f"{len(missing)} missing, {len(unexpected)} unexpected keys")

        # Frozen weights; gradients still flow through to the image for steering.
        self.model.requires_grad_(False)
        self.model.eval()

        self.num_voxels = int(self.model.voxel_embed.shape[0])
        self.image_size = backbone_image_size
        # DINOv2 expects ImageNet normalization (trans_imgs_shift in the repo).
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406])[None, :, None, None])
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225])[None, :, None, None])
        self.to(device=device, dtype=dtype)

    def _preprocess(self, img01: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(img01, size=self.image_size, mode="bicubic", align_corners=False)
        return (x - self.mean) / self.std

    def _forward_model(self, x: torch.Tensor, voxels: torch.Tensor) -> torch.Tensor:
        # The released Encoder takes voxel indices with a leading batch dim
        # ([1, V], per inference/encoder_inference.py::predict_fmri) and returns
        # [B, V] activations.
        voxel_ind = voxels.unsqueeze(0) if voxels.dim() == 1 else voxels
        return self.model(x, voxel_ind.to(x.device))

    def forward(self, img01: torch.Tensor, voxels=None) -> torch.Tensor:
        if voxels is None:
            raise ValueError(
                "UniversalEncoder needs an explicit voxel selection (the target "
                "ROI). Passing all voxels at once is almost never intended."
            )
        x = self._preprocess(img01.to(self.mean.dtype))
        return self._forward_model(x, voxels)  # [B, V]
