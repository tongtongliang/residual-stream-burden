"""Neural-network building blocks used by SiHC."""

from __future__ import annotations

import math
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        normalized = x.float()
        normalized = normalized * torch.rsqrt(
            normalized.pow(2).mean(dim=-1, keepdim=True) + self.eps
        )
        return (normalized * self.weight).to(dtype)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x = x.reshape(*x.shape[:-1], -1, 2)
    x1, x2 = x.unbind(dim=-1)
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


class VisionRotaryEmbeddingFast(nn.Module):
    def __init__(
        self,
        dim: int,
        grid_size: int,
        num_prefix_tokens: int = 0,
        theta: float = 10000.0,
    ) -> None:
        super().__init__()
        freqs = 1.0 / (
            theta ** (torch.arange(0, dim, 2).float()[: dim // 2] / dim)
        )
        positions = torch.arange(grid_size).float()
        freqs = torch.einsum("i,j->ij", positions, freqs)
        freqs = torch.repeat_interleave(freqs, 2, dim=-1)
        freqs = torch.cat(
            [
                freqs[:, None, :].expand(grid_size, grid_size, -1),
                freqs[None, :, :].expand(grid_size, grid_size, -1),
            ],
            dim=-1,
        ).reshape(grid_size * grid_size, -1)
        cos = freqs.cos()
        sin = freqs.sin()
        if num_prefix_tokens:
            cos = torch.cat(
                [torch.ones(num_prefix_tokens, cos.shape[-1]), cos], dim=0
            )
            sin = torch.cat(
                [torch.zeros(num_prefix_tokens, sin.shape[-1]), sin], dim=0
            )
        self.register_buffer("freqs_cos", cos, persistent=False)
        self.register_buffer("freqs_sin", sin, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cos = self.freqs_cos.to(x.device, x.dtype)
        sin = self.freqs_sin.to(x.device, x.dtype)
        return x * cos + _rotate_half(x) * sin


class FactorizedPatchEmbedNHWC(nn.Module):
    """Two-factor linear patch map retained for historical checkpoints."""

    def __init__(
        self,
        img_size: int = 256,
        patch_size: int = 4,
        in_chans: int = 3,
        bottleneck_dim: int = 128,
        embed_dim: int = 768,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.img_size = (img_size, img_size)
        self.patch_size = (patch_size, patch_size)
        self.grid_size = (img_size // patch_size, img_size // patch_size)
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.proj1 = nn.Conv2d(
            in_chans,
            bottleneck_dim,
            kernel_size=patch_size,
            stride=patch_size,
            bias=False,
        )
        self.proj2 = nn.Conv2d(bottleneck_dim, embed_dim, kernel_size=1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-2:] != self.img_size:
            raise ValueError(f"expected image size {self.img_size}, got {tuple(x.shape[-2:])}")
        return self.proj2(self.proj1(x)).permute(0, 2, 3, 1).contiguous()


# Compatibility alias for the pre-audit class name.
BottleneckPatchEmbedNHWC = FactorizedPatchEmbedNHWC


class DirectPatchEmbedNHWC(nn.Module):
    """Direct image-to-high-resolution patch embedding."""

    def __init__(
        self,
        img_size: int = 256,
        patch_size: int = 4,
        in_chans: int = 3,
        bottleneck_dim: int = 128,
        embed_dim: int = 768,
        bias: bool = True,
    ) -> None:
        super().__init__()
        del bottleneck_dim
        self.img_size = (img_size, img_size)
        self.patch_size = (patch_size, patch_size)
        self.grid_size = (img_size // patch_size, img_size // patch_size)
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-2:] != self.img_size:
            raise ValueError(f"expected image size {self.img_size}, got {tuple(x.shape[-2:])}")
        return self.proj(x).permute(0, 2, 3, 1).contiguous()


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256) -> None:
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    @staticmethod
    def timestep_embedding(
        t: torch.Tensor, dim: int, max_period: int = 10000
    ) -> torch.Tensor:
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(half, dtype=torch.float32, device=t.device)
            / half
        )
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.timestep_embedding(t, self.frequency_embedding_size))


class LabelEmbedder(nn.Module):
    def __init__(self, num_classes: int, hidden_size: int) -> None:
        super().__init__()
        self.embedding_table = nn.Embedding(num_classes + 1, hidden_size)

    def forward(self, labels: torch.Tensor) -> torch.Tensor:
        return self.embedding_table(labels)


class WorkspaceAttention(nn.Module):
    """Self-attention over the low-resolution workspace."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        attn_backend: str = "flash",
    ) -> None:
        super().__init__()
        if dim % num_heads:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.attn_backend = attn_backend

    def _sdpa_context(self):
        if self.attn_backend == "flash":
            return sdpa_kernel(
                [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]
            )
        if self.attn_backend == "efficient":
            return sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH])
        if self.attn_backend == "math":
            return sdpa_kernel([SDPBackend.MATH])
        return nullcontext()

    def forward(
        self, x: torch.Tensor, rope: VisionRotaryEmbeddingFast
    ) -> torch.Tensor:
        batch, tokens, channels = x.shape
        qkv = self.qkv(x).reshape(
            batch, tokens, 3, self.num_heads, self.head_dim
        ).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(dim=0)
        q = rope(self.q_norm(q))
        k = rope(self.k_norm(k))
        dropout_p = self.attn_drop.p if self.training else 0.0
        with self._sdpa_context():
            output = F.scaled_dot_product_attention(
                q, k, v, dropout_p=dropout_p, is_causal=False
            )
        output = output.transpose(1, 2).reshape(batch, tokens, channels)
        return self.proj_drop(self.proj(output))


class SwiGLUFFN(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, drop: float = 0.0) -> None:
        super().__init__()
        hidden_dim = int(hidden_dim * 2 / 3)
        self.w12 = nn.Linear(dim, 2 * hidden_dim)
        self.w3 = nn.Linear(hidden_dim, dim)
        self.ffn_dropout = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = self.w12(x).chunk(2, dim=-1)
        return self.w3(self.ffn_dropout(F.silu(x1) * x2))


class WorkspaceJiTBranch(nn.Module):
    """Per-block AdaLN Transformer branch over workspace tokens."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        attn_backend: str = "flash",
    ) -> None:
        super().__init__()
        self.norm1 = RMSNorm(hidden_size)
        self.attn = WorkspaceAttention(
            hidden_size,
            num_heads,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            attn_backend=attn_backend,
        )
        self.norm2 = RMSNorm(hidden_size)
        self.mlp = SwiGLUFFN(
            hidden_size, int(hidden_size * mlp_ratio), drop=proj_drop
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size)
        )

    def conditioning(self, cond: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return self.adaLN_modulation(cond).chunk(6, dim=-1)

    def attention_update(
        self,
        z: torch.Tensor,
        modulation: tuple[torch.Tensor, ...],
        rope: VisionRotaryEmbeddingFast,
    ) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, _, _, _ = modulation
        return gate_msa.unsqueeze(1) * self.attn(
            modulate(self.norm1(z), shift_msa, scale_msa), rope
        )

    def mlp_update(
        self,
        z: torch.Tensor,
        modulation: tuple[torch.Tensor, ...],
    ) -> torch.Tensor:
        _, _, _, shift_mlp, scale_mlp, gate_mlp = modulation
        return gate_mlp.unsqueeze(1) * self.mlp(
            modulate(self.norm2(z), shift_mlp, scale_mlp)
        )

    def forward(
        self, z: torch.Tensor, cond: torch.Tensor, rope: VisionRotaryEmbeddingFast
    ) -> torch.Tensor:
        modulation = self.conditioning(cond)
        dz_attn = self.attention_update(z, modulation, rope)
        dz_mlp = self.mlp_update(z + dz_attn, modulation)
        return dz_attn + dz_mlp


class FinalLayer(nn.Module):
    def __init__(self, hidden_size: int, patch_size: int, out_channels: int) -> None:
        super().__init__()
        self.norm_final = RMSNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size)
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(cond).chunk(2, dim=1)
        return self.linear(modulate(self.norm_final(x), shift, scale))
