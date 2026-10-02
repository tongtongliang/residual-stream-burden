"""JiT-B/16 without class tokens or a patch bottleneck.

The transformer block and initialization follow the released JiT model. The
two intended changes are explicit: RGB patches use one full-rank 768 -> 768
projection, and class conditioning enters AdaLN without sequence prefix tokens.
"""

import math
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
try:
    from transformer_engine.pytorch.triton.mhc import (
        mhc_fused_aggregate,
        mhc_fused_expand_combine,
        mhc_fused_projection,
        mhc_fused_scale,
        mhc_fused_sinkhorn,
    )
except ImportError:  # Plain JiT and CPU tests do not require Transformer Engine.
    mhc_fused_aggregate = None
    mhc_fused_expand_combine = None
    mhc_fused_projection = None
    mhc_fused_scale = None
    mhc_fused_sinkhorn = None

try:
    from ..kernels.fraction_hc import (
        N_FRAGMENTS,
        N_MIXED,
        N_ROUTES,
        N_SLOTS,
        dghc_aggregate_m4n16,
        dghc_depth_m4n16,
        dghc_scale_m4n16,
    )
except ImportError:  # Plain JiT and mHC do not require the fraction-HC kernels.
    dghc_aggregate_m4n16 = dghc_depth_m4n16 = dghc_scale_m4n16 = None


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        y = x.float()
        y = y * torch.rsqrt(y.square().mean(dim=-1, keepdim=True) + self.eps)
        return (y * self.weight).to(dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x = x.reshape(*x.shape[:-1], -1, 2)
    x1, x2 = x.unbind(dim=-1)
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


class VisionRotaryEmbeddingFast(nn.Module):
    def __init__(self, dim: int, grid_size: int, theta: float = 10000.0):
        super().__init__()
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float()[: dim // 2] / dim))
        grid = torch.arange(grid_size).float()
        freqs = torch.einsum("i,j->ij", grid, freqs)
        freqs = torch.repeat_interleave(freqs, 2, dim=-1)
        freqs = torch.cat(
            [
                freqs[:, None, :].expand(grid_size, grid_size, -1),
                freqs[None, :, :].expand(grid_size, grid_size, -1),
            ],
            dim=-1,
        ).reshape(grid_size * grid_size, -1)
        self.register_buffer("freqs_cos", freqs.cos(), persistent=False)
        self.register_buffer("freqs_sin", freqs.sin(), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cos = self.freqs_cos.to(device=x.device, dtype=x.dtype)
        sin = self.freqs_sin.to(device=x.device, dtype=x.dtype)
        return x * cos + rotate_half(x) * sin


def _get_1d_sincos(embed_dim: int, pos: np.ndarray) -> np.ndarray:
    omega = np.arange(embed_dim // 2, dtype=np.float64) / (embed_dim / 2.0)
    omega = 1.0 / 10000**omega
    out = np.einsum("m,d->md", pos.reshape(-1), omega)
    return np.concatenate([np.sin(out), np.cos(out)], axis=1)


def get_2d_sincos_pos_embed(embed_dim: int, grid_size: int) -> np.ndarray:
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0).reshape(2, 1, grid_size, grid_size)
    emb_h = _get_1d_sincos(embed_dim // 2, grid[0])
    emb_w = _get_1d_sincos(embed_dim // 2, grid[1])
    return np.concatenate([emb_h, emb_w], axis=1)


class FullRankPatchEmbed(nn.Module):
    def __init__(self, image_size: int, patch_size: int, in_channels: int, hidden_size: int):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.proj = nn.Conv2d(
            in_channels,
            hidden_size,
            kernel_size=patch_size,
            stride=patch_size,
            bias=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-2:] != (self.image_size, self.image_size):
            raise ValueError(f"expected {self.image_size}x{self.image_size}, got {tuple(x.shape[-2:])}")
        return self.proj(x).flatten(2).transpose(1, 2)


class FactorizedPatchEmbed(nn.Module):
    """Linear patch stem with an explicit rank bottleneck and no activation."""

    def __init__(
        self,
        image_size: int,
        patch_size: int,
        in_channels: int,
        hidden_size: int,
        bottleneck_dim: int = 128,
    ):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.bottleneck_dim = bottleneck_dim
        self.num_patches = (image_size // patch_size) ** 2
        self.proj_in = nn.Conv2d(
            in_channels,
            bottleneck_dim,
            kernel_size=patch_size,
            stride=patch_size,
            bias=True,
        )
        self.proj_out = nn.Conv2d(bottleneck_dim, hidden_size, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-2:] != (self.image_size, self.image_size):
            raise ValueError(f"expected {self.image_size}x{self.image_size}, got {tuple(x.shape[-2:])}")
        return self.proj_out(self.proj_in(x)).flatten(2).transpose(1, 2)


class LocalSlotPatchEmbed(nn.Module):
    """Embed p4 pixels and regroup them as 16 named slots per p16 token."""

    def __init__(
        self,
        image_size: int,
        coarse_patch_size: int,
        local_patch_size: int,
        in_channels: int,
        slot_dim: int,
    ):
        super().__init__()
        if coarse_patch_size % local_patch_size:
            raise ValueError("coarse patch size must be divisible by local patch size")
        self.image_size = image_size
        self.coarse_patch_size = coarse_patch_size
        self.local_patch_size = local_patch_size
        self.cell_size = coarse_patch_size // local_patch_size
        self.num_slots = self.cell_size**2
        self.coarse_grid = image_size // coarse_patch_size
        self.local_grid = image_size // local_patch_size
        self.num_patches = self.coarse_grid**2
        self.proj = nn.Conv2d(
            in_channels,
            slot_dim,
            kernel_size=local_patch_size,
            stride=local_patch_size,
            bias=True,
        )

    def regroup(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, _, _ = x.shape
        g, q = self.coarse_grid, self.cell_size
        return (
            x.reshape(batch, channels, g, q, g, q)
            .permute(0, 2, 4, 3, 5, 1)
            .reshape(batch, g * g, q * q, channels)
            .contiguous()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-2:] != (self.image_size, self.image_size):
            raise ValueError(
                f"expected {self.image_size}x{self.image_size}, got {tuple(x.shape[-2:])}"
            )
        return self.regroup(self.proj(x))


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    @staticmethod
    def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(half, dtype=torch.float32, device=t.device) / half
        )
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.timestep_embedding(t, self.frequency_embedding_size))


class LabelEmbedder(nn.Module):
    def __init__(self, num_classes: int, hidden_size: int):
        super().__init__()
        self.embedding_table = nn.Embedding(num_classes + 1, hidden_size)

    def forward(self, labels: torch.Tensor) -> torch.Tensor:
        return self.embedding_table(labels)


class FlashAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, backend: str = "flash"):
        super().__init__()
        if dim % num_heads:
            raise ValueError(f"hidden size {dim} must be divisible by {num_heads} heads")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, 3 * dim, bias=True)
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)
        self.proj = nn.Linear(dim, dim)
        self.backend = backend

    def _context(self):
        if self.backend == "flash":
            # No fallback: an unsupported dtype/device should fail at startup.
            return sdpa_kernel([SDPBackend.FLASH_ATTENTION])
        if self.backend == "math":
            return sdpa_kernel([SDPBackend.MATH])
        return nullcontext()

    def forward(self, x: torch.Tensor, rope: VisionRotaryEmbeddingFast) -> torch.Tensor:
        batch, tokens, channels = x.shape
        qkv = self.qkv(x).reshape(batch, tokens, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        q = rope(self.q_norm(q))
        k = rope(self.k_norm(k))
        with self._context():
            x = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
        x = x.transpose(1, 2).reshape(batch, tokens, channels)
        return self.proj(x)


class SwiGLUFFN(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        hidden_dim = int(hidden_dim * 2 / 3)
        self.w12 = nn.Linear(dim, 2 * hidden_dim)
        self.w3 = nn.Linear(hidden_dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = self.w12(x).chunk(2, dim=-1)
        return self.w3(F.silu(x1) * x2)


class JiTBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float, attn_backend: str):
        super().__init__()
        self.norm1 = RMSNorm(hidden_size)
        self.attn = FlashAttention(hidden_size, num_heads, backend=attn_backend)
        self.norm2 = RMSNorm(hidden_size)
        self.mlp = SwiGLUFFN(hidden_size, int(hidden_size * mlp_ratio))
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size))

    def forward(self, x: torch.Tensor, c: torch.Tensor, rope: VisionRotaryEmbeddingFast) -> torch.Tensor:
        shift_attn, scale_attn, gate_attn, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=-1)
        )
        x = x + gate_attn.unsqueeze(1) * self.attn(
            modulate(self.norm1(x), shift_attn, scale_attn), rope
        )
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class MHCConnection(nn.Module):
    """One N=4 manifold-constrained hyper-connection around a sublayer.

    The persistent state uses NVIDIA's native (tokens, batch, channels, streams)
    layout. The FP32 routing projection is cast to the activation dtype because
    Transformer Engine 2.18's released Triton projection requires matching input
    dtypes; alpha and beta remain FP32.
    """

    def __init__(
        self,
        hidden_size: int,
        n: int = 4,
        backend: str = "nvidia_fused",
        init_residual_index: int = 0,
    ):
        super().__init__()
        if n != 4:
            raise ValueError("NVIDIA's fused mHC kernels currently support only N=4")
        if backend not in {"nvidia_fused", "reference"}:
            raise ValueError(f"unsupported mHC backend: {backend}")
        if backend == "nvidia_fused" and mhc_fused_projection is None:
            raise RuntimeError(
                "mHC topology requires Transformer Engine with PyTorch Triton mHC kernels"
            )
        self.hidden_size = hidden_size
        self.n = n
        self.backend = backend
        mix_size = 2 * n + n * n
        self.phi = nn.Parameter(torch.zeros(mix_size, n * hidden_size))
        self.alpha = nn.Parameter(torch.full((3,), 0.01))
        beta = torch.full((1, mix_size), -8.0)
        beta[:, n : 2 * n].zero_()
        beta[:, 2 * n :].reshape(1, n, n).diagonal(dim1=-2, dim2=-1).zero_()
        beta[:, init_residual_index % n] = 0.0
        self.beta = nn.Parameter(beta)

    def _reference_maps(self, x: torch.Tensor):
        tokens, batch, channels, n = x.shape
        flat = x.reshape(tokens * batch, channels * n).float()
        rms = flat.square().mean(dim=-1, keepdim=True).add(
            torch.finfo(torch.float32).eps
        ).sqrt()
        projected = flat @ self.phi.float().t()
        scaled = projected / rms
        pre = torch.sigmoid(scaled[:, :n] * self.alpha[0] + self.beta[:, :n])
        post = 2 * torch.sigmoid(
            scaled[:, n : 2 * n] * self.alpha[1] + self.beta[:, n : 2 * n]
        )
        logits = (
            scaled[:, 2 * n :] * self.alpha[2] + self.beta[:, 2 * n :]
        ).reshape(tokens, batch, n, n)
        left = torch.zeros_like(logits[..., 0])
        right = torch.zeros_like(logits[..., 0])
        for _ in range(20):
            left = -torch.logsumexp(logits + right.unsqueeze(-2), dim=-1)
            right = -torch.logsumexp(logits + left.unsqueeze(-1), dim=-2)
        residual = torch.exp(left.unsqueeze(-1) + logits + right.unsqueeze(-2))
        collapsed = torch.einsum(
            "tbcn,tbn->tbc", x.float(), pre.reshape(tokens, batch, n)
        ).to(x.dtype)
        return collapsed, post.reshape(tokens, batch, n), residual

    def mix_and_aggregate(self, x: torch.Tensor):
        if self.backend == "reference":
            return self._reference_maps(x)
        tokens, batch, channels, n = x.shape
        phi = self.phi.to(dtype=x.dtype)
        projected, mean_square = mhc_fused_projection(
            x.reshape(tokens * batch, channels * n), phi, use_tf32=True
        )
        pre, post, residual = mhc_fused_scale(
            projected, self.alpha, self.beta, mean_square, n
        )
        residual = mhc_fused_sinkhorn(
            residual.reshape(tokens, batch, n, n), n=n, recompute_hist=True, iters=20
        )
        collapsed = mhc_fused_aggregate(
            x, pre.reshape(tokens, batch, n), n=n, use_tf32=True
        )
        return collapsed, post.reshape(tokens, batch, n), residual

    def combine(
        self,
        branch_output: torch.Tensor,
        post: torch.Tensor,
        x: torch.Tensor,
        residual: torch.Tensor,
    ) -> torch.Tensor:
        if self.backend == "reference":
            return (
                torch.einsum("tbc,tbn->tbcn", branch_output.float(), post.float())
                + torch.einsum("tbcn,tbnm->tbcm", x.float(), residual.float())
            ).to(x.dtype)
        return mhc_fused_expand_combine(
            branch_output, None, post, x, residual, n=self.n, use_tf32=True
        )


class MHCJiTBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float,
        attn_backend: str,
        mhc_n: int,
        mhc_backend: str,
        layer_index: int,
    ):
        super().__init__()
        self.norm1 = RMSNorm(hidden_size)
        self.attn = FlashAttention(hidden_size, num_heads, backend=attn_backend)
        self.norm2 = RMSNorm(hidden_size)
        self.mlp = SwiGLUFFN(hidden_size, int(hidden_size * mlp_ratio))
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size)
        )
        self.attn_hc = MHCConnection(
            hidden_size,
            n=mhc_n,
            backend=mhc_backend,
            init_residual_index=2 * layer_index,
        )
        self.mlp_hc = MHCConnection(
            hidden_size,
            n=mhc_n,
            backend=mhc_backend,
            init_residual_index=2 * layer_index + 1,
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor, rope: VisionRotaryEmbeddingFast):
        shift_attn, scale_attn, gate_attn, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=-1)
        )
        collapsed, post, residual = self.attn_hc.mix_and_aggregate(x)
        branch = gate_attn.unsqueeze(1) * self.attn(
            modulate(self.norm1(collapsed.transpose(0, 1)), shift_attn, scale_attn), rope
        )
        x = self.attn_hc.combine(
            branch.transpose(0, 1).contiguous(), post, x, residual
        )

        collapsed, post, residual = self.mlp_hc.mix_and_aggregate(x)
        branch = gate_mlp.unsqueeze(1) * self.mlp(
            modulate(self.norm2(collapsed.transpose(0, 1)), shift_mlp, scale_mlp)
        )
        return self.mlp_hc.combine(
            branch.transpose(0, 1).contiguous(), post, x, residual
        )

class GHCConnection(nn.Module):
    """Dense dynamic m=4, n=16 GHC with a 192-wide slot state."""

    def __init__(self, hidden_size: int, backend: str = "triton"):
        super().__init__()
        if hidden_size % N_FRAGMENTS:
            raise ValueError("hidden size must be divisible by four GHC fragments")
        if backend not in {"triton", "reference"}:
            raise ValueError(f"unsupported GHC backend: {backend}")
        if backend == "triton" and (
            mhc_fused_projection is None
            or dghc_scale_m4n16 is None
            or dghc_aggregate_m4n16 is None
            or dghc_depth_m4n16 is None
        ):
            raise RuntimeError("GHC Triton backend requires Transformer Engine and repo kernels")
        self.hidden_size = hidden_size
        self.slot_dim = hidden_size // N_FRAGMENTS
        self.backend = backend
        self.factor = 1.0 / math.sqrt(self.slot_dim)
        self.dynamic_weight = nn.Parameter(torch.zeros(N_ROUTES, self.slot_dim))
        self.dynamic_scale = nn.Parameter(torch.ones(N_SLOTS, N_ROUTES))
        self.static_route = nn.Parameter(torch.empty(N_SLOTS, N_ROUTES))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.zeros_(self.dynamic_weight)
        nn.init.ones_(self.dynamic_scale)
        route = torch.zeros(N_SLOTS, N_ROUTES)
        # Four spatial 2x2 quadrants inside the local 4x4 p4 slot grid.
        groups = (0, 0, 1, 1, 0, 0, 1, 1, 2, 2, 3, 3, 2, 2, 3, 3)
        for slot, group in enumerate(groups):
            route[slot, group] = 0.25
            route[slot, N_FRAGMENTS + slot] = 1.0
            route[slot, N_MIXED + group] = 1.0
        with torch.no_grad():
            self.static_route.copy_(route)

    def _reference_width(self, h: torch.Tensor):
        flat = h.reshape(-1, self.slot_dim).float()
        ms = flat.square().mean(dim=-1, keepdim=True)
        projected = flat @ self.dynamic_weight.float().t()
        dynamic = torch.tanh(
            projected * torch.rsqrt(ms + torch.finfo(torch.float32).eps) * self.factor
        )
        routes = (
            self.static_route.float()[None]
            + self.dynamic_scale.float()[None] * dynamic.reshape(-1, N_SLOTS, N_ROUTES)
        )
        mixed = torch.einsum(
            "msc,msd->mdc", h.reshape(-1, N_SLOTS, self.slot_dim).float(), routes[..., :N_MIXED]
        )
        return mixed.to(h.dtype), F.pad(routes, (0, 32 - N_ROUTES)).to(h.dtype)

    def width(self, h: torch.Tensor):
        flat_h = h.reshape(-1, N_SLOTS, self.slot_dim).contiguous()
        if self.backend == "reference":
            return self._reference_width(flat_h)
        projected, ms = mhc_fused_projection(
            flat_h.reshape(-1, self.slot_dim),
            self.dynamic_weight.to(dtype=h.dtype),
            use_tf32=True,
        )
        routes = dghc_scale_m4n16(
            projected,
            ms,
            self.dynamic_scale,
            self.static_route,
            self.factor,
        ).reshape(-1, N_SLOTS, 32)
        return dghc_aggregate_m4n16(flat_h, routes), routes

    def depth(self, mixed: torch.Tensor, branch: torch.Tensor, routes: torch.Tensor):
        if self.backend == "reference":
            beta = routes[..., N_MIXED:N_ROUTES]
            write = torch.einsum("msf,mfc->msc", beta.float(), branch.float())
            return (write + mixed[:, N_FRAGMENTS:].float()).to(mixed.dtype)
        return dghc_depth_m4n16(mixed, branch, routes)


class GHCJiTBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float,
        attn_backend: str,
        ghc_backend: str,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.slot_dim = hidden_size // N_FRAGMENTS
        self.norm1 = RMSNorm(hidden_size)
        self.attn = FlashAttention(hidden_size, num_heads, backend=attn_backend)
        self.norm2 = RMSNorm(hidden_size)
        self.mlp = SwiGLUFFN(hidden_size, int(hidden_size * mlp_ratio))
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size)
        )
        self.attn_hc = GHCConnection(hidden_size, backend=ghc_backend)
        self.mlp_hc = GHCConnection(hidden_size, backend=ghc_backend)

    def _branch_input(self, mixed: torch.Tensor, batch: int, tokens: int) -> torch.Tensor:
        return mixed[:, :N_FRAGMENTS].reshape(batch, tokens, self.hidden_size)

    def _write(
        self,
        connection: GHCConnection,
        mixed: torch.Tensor,
        branch: torch.Tensor,
        routes: torch.Tensor,
        batch: int,
        tokens: int,
    ) -> torch.Tensor:
        branch = branch.reshape(batch * tokens, N_FRAGMENTS, self.slot_dim).contiguous()
        return connection.depth(mixed, branch, routes).reshape(
            batch, tokens, N_SLOTS, self.slot_dim
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor, rope: VisionRotaryEmbeddingFast):
        batch, tokens, _, _ = x.shape
        shift_attn, scale_attn, gate_attn, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=-1)
        )

        mixed, routes = self.attn_hc.width(x)
        branch = self.attn(
            modulate(self.norm1(self._branch_input(mixed, batch, tokens)), shift_attn, scale_attn),
            rope,
        )
        branch = gate_attn.unsqueeze(1) * branch
        x = self._write(self.attn_hc, mixed, branch, routes, batch, tokens)

        mixed, routes = self.mlp_hc.width(x)
        branch = self.mlp(
            modulate(self.norm2(self._branch_input(mixed, batch, tokens)), shift_mlp, scale_mlp)
        )
        branch = gate_mlp.unsqueeze(1) * branch
        return self._write(self.mlp_hc, mixed, branch, routes, batch, tokens)


class SlotFinalLayer(nn.Module):
    def __init__(self, slot_dim: int, conditioning_dim: int, out_channels: int, local_patch: int):
        super().__init__()
        self.norm_final = RMSNorm(slot_dim)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(conditioning_dim, 2 * slot_dim)
        )
        self.linear = nn.Linear(slot_dim, local_patch * local_patch * out_channels)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = self.norm_final(x)
        x = x * (1 + scale[:, None, None, :]) + shift[:, None, None, :]
        return self.linear(x)


class FinalLayer(nn.Module):
    def __init__(self, hidden_size: int, patch_size: int, out_channels: int):
        super().__init__()
        self.norm_final = RMSNorm(hidden_size)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size))
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        return self.linear(modulate(self.norm_final(x), shift, scale))


class PlainJiTB16(nn.Module):
    def __init__(
        self,
        image_size: int = 256,
        patch_size: int = 16,
        in_channels: int = 3,
        hidden_size: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        num_classes: int = 1000,
        attn_backend: str = "flash",
    ):
        super().__init__()
        self.input_size = image_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.hidden_size = hidden_size
        self.depth = depth
        self.num_heads = num_heads
        self.num_classes = num_classes

        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = LabelEmbedder(num_classes, hidden_size)
        self.x_embedder = FullRankPatchEmbed(image_size, patch_size, in_channels, hidden_size)
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.x_embedder.num_patches, hidden_size), requires_grad=False
        )
        head_rope_dim = hidden_size // num_heads // 2
        self.feat_rope = VisionRotaryEmbeddingFast(head_rope_dim, image_size // patch_size)
        self.blocks = nn.ModuleList(
            [JiTBlock(hidden_size, num_heads, mlp_ratio, attn_backend) for _ in range(depth)]
        )
        self.final_layer = FinalLayer(hidden_size, patch_size, in_channels)
        self.initialize_weights()

    def initialize_weights(self) -> None:
        def init_linear(module: nn.Module) -> None:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        self.apply(init_linear)
        pos = get_2d_sincos_pos_embed(
            self.hidden_size, int(self.x_embedder.num_patches**0.5)
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos).float().unsqueeze(0))
        if hasattr(self.x_embedder, "proj"):
            nn.init.xavier_uniform_(self.x_embedder.proj.weight.view(self.hidden_size, -1))
            nn.init.zeros_(self.x_embedder.proj.bias)
        elif isinstance(self.x_embedder, FactorizedPatchEmbed):
            nn.init.xavier_uniform_(
                self.x_embedder.proj_in.weight.view(self.x_embedder.bottleneck_dim, -1)
            )
            nn.init.zeros_(self.x_embedder.proj_in.bias)
            nn.init.xavier_uniform_(self.x_embedder.proj_out.weight.flatten(1))
            nn.init.zeros_(self.x_embedder.proj_out.bias)
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)
        for block in self.blocks:
            nn.init.zeros_(block.adaLN_modulation[-1].weight)
            nn.init.zeros_(block.adaLN_modulation[-1].bias)
        nn.init.zeros_(self.final_layer.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.final_layer.adaLN_modulation[-1].bias)
        nn.init.zeros_(self.final_layer.linear.weight)
        nn.init.zeros_(self.final_layer.linear.bias)

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        p = self.patch_size
        c = self.out_channels
        height = width = int(x.shape[1] ** 0.5)
        if height * width != x.shape[1]:
            raise ValueError(f"token count {x.shape[1]} is not square")
        x = x.reshape(x.shape[0], height, width, p, p, c)
        x = torch.einsum("nhwpqc->nchpwq", x)
        return x.reshape(x.shape[0], c, height * p, width * p)

    def _embed_tokens(self, x: torch.Tensor) -> torch.Tensor:
        x = self.x_embedder(x)
        return x + self.pos_embed.to(dtype=x.dtype)

    def forward_features(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor):
        c = self.t_embedder(t) + self.y_embedder(y)
        x = self._embed_tokens(x)
        fanins = []
        for block in self.blocks:
            fanins.append(x)
            x = block(x, c, self.feat_rope)
        return x, c, fanins

    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        c = self.t_embedder(t) + self.y_embedder(y)
        x = self._embed_tokens(x)
        for block in self.blocks:
            x = block(x, c, self.feat_rope)
        return self.unpatchify(self.final_layer(x, c))


class MHCJiTB16(PlainJiTB16):
    """JiT-B/16 with a four-stream mHC residual state and 768-wide sublayers."""

    def __init__(
        self,
        mhc_n: int = 4,
        mhc_backend: str = "nvidia_fused",
        mlp_ratio: float = 4.0,
        **kwargs,
    ):
        super().__init__(mlp_ratio=mlp_ratio, **kwargs)
        self.mhc_n = mhc_n
        self.mhc_backend = mhc_backend
        self.blocks = nn.ModuleList(
            [
                MHCJiTBlock(
                    self.hidden_size,
                    self.num_heads,
                    mlp_ratio,
                    kwargs.get("attn_backend", "flash"),
                    mhc_n,
                    mhc_backend,
                    layer_index,
                )
                for layer_index in range(self.depth)
            ]
        )
        self.initialize_weights()

    def _expand(self, x: torch.Tensor) -> torch.Tensor:
        return (
            x.transpose(0, 1)
            .unsqueeze(-1)
            .expand(-1, -1, -1, self.mhc_n)
            .contiguous()
        )

    def _collapse(self, x: torch.Tensor) -> torch.Tensor:
        return x.mean(dim=-1).transpose(0, 1)

    def forward_features(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor):
        c = self.t_embedder(t) + self.y_embedder(y)
        x = self._expand(self._embed_tokens(x))
        fanins = []
        for block in self.blocks:
            fanins.append(x.permute(1, 0, 3, 2).flatten(2))
            x = block(x, c, self.feat_rope)
        return self._collapse(x), c, fanins

    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        c = self.t_embedder(t) + self.y_embedder(y)
        x = self._expand(self._embed_tokens(x))
        for block in self.blocks:
            x = block(x, c, self.feat_rope)
        return self.unpatchify(self.final_layer(self._collapse(x), c))


class MHCSlotJiTB16(MHCJiTB16):
    """Full-width N=4 mHC whose four named streams are spatial p8 slots."""

    def __init__(
        self,
        local_patch_size: int = 8,
        mlp_ratio: float = 4.0,
        **kwargs,
    ):
        super().__init__(mlp_ratio=mlp_ratio, **kwargs)
        if self.patch_size != 16 or local_patch_size != 8:
            raise ValueError("the N=4 slot ablation is fixed to a p16 workspace and p8 slots")
        if self.mhc_n != 4:
            raise ValueError("the p8 slot ablation requires four full-width residual streams")
        self.local_patch_size = local_patch_size
        self.x_embedder = LocalSlotPatchEmbed(
            self.input_size,
            self.patch_size,
            local_patch_size,
            self.in_channels,
            self.hidden_size,
        )
        if self.x_embedder.num_slots != self.mhc_n:
            raise ValueError(
                f"expected {self.mhc_n} local slots, got {self.x_embedder.num_slots}"
            )

        dense_pos = torch.from_numpy(
            get_2d_sincos_pos_embed(self.hidden_size, self.x_embedder.local_grid)
        ).float()
        dense_pos = dense_pos.reshape(
            self.x_embedder.local_grid,
            self.x_embedder.local_grid,
            self.hidden_size,
        )
        dense_pos = dense_pos.permute(2, 0, 1).unsqueeze(0)
        self.pos_embed = nn.Parameter(
            self.x_embedder.regroup(dense_pos), requires_grad=False
        )
        self.final_layer = SlotFinalLayer(
            self.hidden_size,
            self.hidden_size,
            self.out_channels,
            self.local_patch_size,
        )
        self._initialize_slot_weights()

    def _initialize_slot_weights(self) -> None:
        def init_linear(module: nn.Module) -> None:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        self.apply(init_linear)
        nn.init.xavier_uniform_(self.x_embedder.proj.weight.view(self.hidden_size, -1))
        nn.init.zeros_(self.x_embedder.proj.bias)
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)
        for block in self.blocks:
            nn.init.zeros_(block.adaLN_modulation[-1].weight)
            nn.init.zeros_(block.adaLN_modulation[-1].bias)
        nn.init.zeros_(self.final_layer.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.final_layer.adaLN_modulation[-1].bias)
        nn.init.zeros_(self.final_layer.linear.weight)
        nn.init.zeros_(self.final_layer.linear.bias)

    def _slot_state(self, x: torch.Tensor) -> torch.Tensor:
        # LocalSlotPatchEmbed emits (batch, token, slot, channel); TE mHC uses
        # (token, batch, channel, stream).
        return self._embed_tokens(x).permute(1, 0, 3, 2).contiguous()

    def unpatchify_slots(self, x: torch.Tensor) -> torch.Tensor:
        batch, tokens, slots, raw = x.shape
        grid = int(tokens**0.5)
        q = self.patch_size // self.local_patch_size
        p = self.local_patch_size
        if grid * grid != tokens or slots != q * q:
            raise ValueError(f"invalid slot prediction shape: {tuple(x.shape)}")
        expected_raw = p * p * self.out_channels
        if raw != expected_raw:
            raise ValueError(f"expected raw slot width {expected_raw}, got {raw}")
        x = x.reshape(batch, grid, grid, q, q, p, p, self.out_channels)
        x = x.permute(0, 7, 1, 3, 5, 2, 4, 6)
        return x.reshape(batch, self.out_channels, grid * q * p, grid * q * p)

    def forward_features(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor):
        c = self.t_embedder(t) + self.y_embedder(y)
        x = self._slot_state(x)
        fanins = []
        for block in self.blocks:
            fanins.append(x.permute(1, 0, 3, 2).flatten(2))
            x = block(x, c, self.feat_rope)
        return x.permute(1, 0, 3, 2).flatten(2), c, fanins

    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        c = self.t_embedder(t) + self.y_embedder(y)
        x = self._slot_state(x)
        for block in self.blocks:
            x = block(x, c, self.feat_rope)
        x = x.permute(1, 0, 3, 2).contiguous()
        return self.unpatchify_slots(self.final_layer(x, c))

class GHCJiTB16(PlainJiTB16):
    """JiT-B/16 with sixteen positional p4 slots at the same 4d virtual width."""

    def __init__(
        self,
        ghc_backend: str = "triton",
        local_patch_size: int = 4,
        mlp_ratio: float = 4.0,
        **kwargs,
    ):
        super().__init__(mlp_ratio=mlp_ratio, **kwargs)
        if self.patch_size != 16 or local_patch_size != 4:
            raise ValueError("the initial GHC experiment is fixed to p16 workspace and p4 slots")
        if self.hidden_size % N_FRAGMENTS:
            raise ValueError("hidden size must be divisible by four")
        self.ghc_backend = ghc_backend
        self.local_patch_size = local_patch_size
        self.slot_dim = self.hidden_size // N_FRAGMENTS
        self.x_embedder = LocalSlotPatchEmbed(
            self.input_size,
            self.patch_size,
            local_patch_size,
            self.in_channels,
            self.slot_dim,
        )
        if self.x_embedder.num_slots != N_SLOTS:
            raise ValueError(f"expected 16 local slots, got {self.x_embedder.num_slots}")

        dense_pos = torch.from_numpy(
            get_2d_sincos_pos_embed(self.slot_dim, self.x_embedder.local_grid)
        ).float()
        dense_pos = dense_pos.reshape(
            self.x_embedder.local_grid, self.x_embedder.local_grid, self.slot_dim
        )
        dense_pos = dense_pos.permute(2, 0, 1).unsqueeze(0)
        self.pos_embed = nn.Parameter(
            self.x_embedder.regroup(dense_pos), requires_grad=False
        )
        self.blocks = nn.ModuleList(
            [
                GHCJiTBlock(
                    self.hidden_size,
                    self.num_heads,
                    mlp_ratio,
                    kwargs.get("attn_backend", "flash"),
                    ghc_backend,
                )
                for _ in range(self.depth)
            ]
        )
        self.final_layer = SlotFinalLayer(
            self.slot_dim,
            self.hidden_size,
            self.out_channels,
            self.local_patch_size,
        )
        self._initialize_ghc_weights()

    def _initialize_ghc_weights(self) -> None:
        def init_linear(module: nn.Module) -> None:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        self.apply(init_linear)
        nn.init.xavier_uniform_(self.x_embedder.proj.weight.view(self.slot_dim, -1))
        nn.init.zeros_(self.x_embedder.proj.bias)
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)
        for block in self.blocks:
            nn.init.zeros_(block.adaLN_modulation[-1].weight)
            nn.init.zeros_(block.adaLN_modulation[-1].bias)
            block.attn_hc.reset_parameters()
            block.mlp_hc.reset_parameters()
        nn.init.zeros_(self.final_layer.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.final_layer.adaLN_modulation[-1].bias)
        nn.init.zeros_(self.final_layer.linear.weight)
        nn.init.zeros_(self.final_layer.linear.bias)

    def unpatchify_slots(self, x: torch.Tensor) -> torch.Tensor:
        batch, tokens, slots, raw = x.shape
        grid = int(tokens**0.5)
        q = self.patch_size // self.local_patch_size
        p = self.local_patch_size
        if grid * grid != tokens or slots != q * q:
            raise ValueError(f"invalid slot prediction shape: {tuple(x.shape)}")
        expected_raw = p * p * self.out_channels
        if raw != expected_raw:
            raise ValueError(f"expected raw slot width {expected_raw}, got {raw}")
        x = x.reshape(batch, grid, grid, q, q, p, p, self.out_channels)
        x = x.permute(0, 7, 1, 3, 5, 2, 4, 6)
        return x.reshape(
            batch,
            self.out_channels,
            grid * q * p,
            grid * q * p,
        )

    def forward_features(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor):
        c = self.t_embedder(t) + self.y_embedder(y)
        x = self._embed_tokens(x)
        fanins = []
        for block in self.blocks:
            fanins.append(x.flatten(2))
            x = block(x, c, self.feat_rope)
        return x.flatten(2), c, fanins

    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        c = self.t_embedder(t) + self.y_embedder(y)
        x = self._embed_tokens(x)
        for block in self.blocks:
            x = block(x, c, self.feat_rope)
        return self.unpatchify_slots(self.final_layer(x, c))


class BottleneckJiTB16(PlainJiTB16):
    """JiT-B/16 control with a rank-128 linear patch bottleneck."""

    def __init__(self, bottleneck_dim: int = 128, **kwargs):
        super().__init__(**kwargs)
        self.bottleneck_dim = bottleneck_dim
        self.x_embedder = FactorizedPatchEmbed(
            self.input_size,
            self.patch_size,
            self.in_channels,
            self.hidden_size,
            bottleneck_dim,
        )
        self.initialize_weights()


class PlainJiTL16(PlainJiTB16):
    """Full-rank JiT-L/16 scale control: 24 layers at width 1024."""

    def __init__(self, **kwargs):
        kwargs.setdefault("hidden_size", 1024)
        kwargs.setdefault("depth", 24)
        kwargs.setdefault("num_heads", 16)
        super().__init__(**kwargs)


class PlainJiTXL16(PlainJiTB16):
    """Matched-capacity JiT-XL/16 control: 28 layers at width 1152.

    JiT does not publish an official XL preset. This constructor keeps the
    JiT block and uses the classic DiT/RAE XL depth and width so Group 2 can
    isolate capacity from representation and backbone-family changes.
    """

    def __init__(self, **kwargs):
        kwargs.setdefault("hidden_size", 1152)
        kwargs.setdefault("depth", 28)
        kwargs.setdefault("num_heads", 16)
        super().__init__(**kwargs)


def build_model(attn_backend: str = "flash", topology: str = "single", **kwargs):
    if topology == "single":
        kwargs.pop("mhc_n", None)
        kwargs.pop("mhc_backend", None)
        kwargs.pop("ghc_backend", None)
        kwargs.pop("local_patch_size", None)
        return PlainJiTB16(attn_backend=attn_backend, **kwargs)
    if topology == "mhc_n4":
        kwargs.pop("ghc_backend", None)
        kwargs.pop("local_patch_size", None)
        return MHCJiTB16(attn_backend=attn_backend, **kwargs)
    if topology == "mhc_n4_p8":
        kwargs.pop("ghc_backend", None)
        return MHCSlotJiTB16(attn_backend=attn_backend, **kwargs)
    if topology == "ghc_m4n16_p4":
        kwargs.pop("mhc_n", None)
        kwargs.pop("mhc_backend", None)
        return GHCJiTB16(attn_backend=attn_backend, **kwargs)
    raise ValueError(f"unsupported topology: {topology}")
