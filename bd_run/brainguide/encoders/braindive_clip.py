"""BrainDiVE-style scorer: linear readout on the L2-normalized CLIP embedding.

    act[v] = < normalize(clip_image(img)) , W[v] >

This is Paper 3's encoder shape and it is **runnable today** with the cached
``openai/clip-vit-base-patch32`` — useful for smoke-testing the whole
diffusion+guidance loop before the universal encoder lands. With
``weights_path=None`` the readout matrix is random (the "mock" alias).

L2-normalizing the embedding is defense (3) from the notes: it removes the
radial gradient so "make the embedding bigger" is unavailable to the sampler,
only rotation on the sphere.
"""
import torch
import torch.nn.functional as F

from .base import BrainEncoder


class BrainDiVECLIPEncoder(BrainEncoder):
    def __init__(self, clip_model_id="openai/clip-vit-base-patch32",
                 num_voxels=1000, weights_path=None, seed=0,
                 device="cpu", dtype=torch.float32):
        super().__init__()
        from transformers import CLIPImageProcessor, CLIPModel

        self.clip = CLIPModel.from_pretrained(clip_model_id)
        self.clip.requires_grad_(False)
        d = self.clip.config.projection_dim
        self.image_size = self.clip.config.vision_config.image_size

        if weights_path:
            W = torch.load(weights_path, map_location="cpu")
            W = W if torch.is_tensor(W) else torch.as_tensor(W)
        else:
            g = torch.Generator().manual_seed(int(seed))
            W = F.normalize(torch.randn(num_voxels, d, generator=g), dim=-1)
        self.register_buffer("W", W.to(torch.float32))

        proc = CLIPImageProcessor.from_pretrained(clip_model_id)
        self.register_buffer("mean", torch.tensor(proc.image_mean)[None, :, None, None])
        self.register_buffer("std", torch.tensor(proc.image_std)[None, :, None, None])
        self.to(device=device, dtype=dtype)

    def _embed(self, img01: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(img01, size=self.image_size, mode="bicubic", align_corners=False)
        x = (x - self.mean) / self.std
        emb = self.clip.get_image_features(pixel_values=x)
        return F.normalize(emb, dim=-1)  # defense (3): kill the radial gradient

    def forward(self, img01: torch.Tensor, voxels=None) -> torch.Tensor:
        emb = self._embed(img01.to(self.mean.dtype))          # [B, D]
        W = self.W if voxels is None else self.W[voxels]       # [V, D]
        return emb @ W.t().to(emb.dtype)                       # [B, V]
