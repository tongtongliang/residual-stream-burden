"""Colored patch noise from a precomputed patch second-moment square-root.

Noise is i.i.d. across spatial patches:
  eps_patch = scale * P_sqrt @ z,  z ~ N(0, I_{768})
then folded back to (B, 3, H, W).

Patch layout MUST match scripts/compute_imagenet_patch_second_moments.py:
  (B, C, H, W) -> (B, gh, gw, C, p, p) -> (B, N, C*p*p).
"""
from __future__ import annotations

from pathlib import Path

import torch


def patchify(x: torch.Tensor, patch_size: int) -> torch.Tensor:
    b, c, h, w = x.shape
    p = patch_size
    if h % p or w % p:
        raise ValueError(f"image size {(h, w)} not divisible by patch_size={p}")
    x = x.reshape(b, c, h // p, p, w // p, p)
    return x.permute(0, 2, 4, 1, 3, 5).reshape(b, (h // p) * (w // p), c * p * p)


def unpatchify(patches: torch.Tensor, *, channels: int, patch_size: int, image_size: int) -> torch.Tensor:
    b, n, d = patches.shape
    p = patch_size
    gh = gw = image_size // p
    if n != gh * gw:
        raise ValueError(f"expected {gh * gw} patches, got {n}")
    if d != channels * p * p:
        raise ValueError(f"expected patch dim {channels * p * p}, got {d}")
    x = patches.reshape(b, gh, gw, channels, p, p)
    return x.permute(0, 3, 1, 4, 2, 5).reshape(b, channels, image_size, image_size)


class PatchCovarianceNoise:
    def __init__(
        self,
        p_sqrt: torch.Tensor,
        *,
        patch_size: int = 16,
        image_size: int = 256,
        channels: int = 3,
    ):
        if p_sqrt.ndim != 2 or p_sqrt.shape[0] != p_sqrt.shape[1]:
            raise ValueError(f"p_sqrt must be square, got {tuple(p_sqrt.shape)}")
        expected = channels * patch_size * patch_size
        if p_sqrt.shape[0] != expected:
            raise ValueError(f"p_sqrt dim {p_sqrt.shape[0]} != expected {expected}")
        self.p_sqrt = p_sqrt.detach().contiguous()
        self.patch_size = int(patch_size)
        self.image_size = int(image_size)
        self.channels = int(channels)
        self.dim = int(p_sqrt.shape[0])
        self.num_patches = (self.image_size // self.patch_size) ** 2

    @classmethod
    def from_npz(cls, path, key: str = "p_sqrt_uncentered", map_location="cpu"):
        path = Path(path)
        data = np_load(path)
        if key not in data:
            raise KeyError(f"{path} missing key {key!r}; have {sorted(data.files)}")
        p_sqrt = torch.as_tensor(data[key], dtype=torch.float32)
        patch_size = int(data["patch_size"]) if "patch_size" in data.files else 16
        image_size = int(data["image_size"]) if "image_size" in data.files else 256
        return cls(p_sqrt, patch_size=patch_size, image_size=image_size)

    def to(self, device=None, dtype=None):
        self.p_sqrt = self.p_sqrt.to(device=device, dtype=dtype)
        return self

    def sample(self, batch_size: int, *, device=None, dtype=None, noise_scale: float = 1.0) -> torch.Tensor:
        device = device if device is not None else self.p_sqrt.device
        # Matmul in float32 for stability; cast output to requested dtype.
        p_sqrt = self.p_sqrt.to(device=device, dtype=torch.float32)
        z = torch.randn(batch_size, self.num_patches, self.dim, device=device, dtype=torch.float32)
        patches = torch.matmul(z, p_sqrt.transpose(0, 1))
        if noise_scale != 1.0:
            patches = patches * float(noise_scale)
        images = unpatchify(
            patches,
            channels=self.channels,
            patch_size=self.patch_size,
            image_size=self.image_size,
        )
        if dtype is not None and images.dtype != dtype:
            images = images.to(dtype=dtype)
        return images

    def sample_like(self, x: torch.Tensor, noise_scale: float = 1.0) -> torch.Tensor:
        return self.sample(x.shape[0], device=x.device, dtype=x.dtype, noise_scale=noise_scale)


def np_load(path):
    import numpy as np

    return np.load(path)


def maybe_load_patch_noise(path: str, key: str = "p_sqrt_uncentered"):
    if not path:
        return None
    return PatchCovarianceNoise.from_npz(path, key=key)
