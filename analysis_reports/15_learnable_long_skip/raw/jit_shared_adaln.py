"""JiT-B style ImageNet model with time-only shared AdaLN and class-prefix tokens.

Timestep conditioning enters the AdaLN path, while class conditioning enters
only through in-context prefix tokens.
"""
import math
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (x * self.weight).to(dtype)


def rotate_half(x):
    x = x.reshape(*x.shape[:-1], -1, 2)
    x1, x2 = x.unbind(dim=-1)
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


class VisionRotaryEmbeddingFast(nn.Module):
    def __init__(self, dim, grid_size, num_prefix_tokens=0, theta=10000.0):
        super().__init__()
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float()[: dim // 2] / dim))
        t = torch.arange(grid_size).float()
        freqs = torch.einsum("i,j->ij", t, freqs)
        freqs = torch.repeat_interleave(freqs, 2, dim=-1)
        freqs = torch.cat([
            freqs[:, None, :].expand(grid_size, grid_size, -1),
            freqs[None, :, :].expand(grid_size, grid_size, -1),
        ], dim=-1).reshape(grid_size * grid_size, -1)
        cos = freqs.cos()
        sin = freqs.sin()
        if num_prefix_tokens:
            cos = torch.cat([torch.ones(num_prefix_tokens, cos.shape[-1]), cos], dim=0)
            sin = torch.cat([torch.zeros(num_prefix_tokens, sin.shape[-1]), sin], dim=0)
        self.register_buffer("freqs_cos", cos, persistent=False)
        self.register_buffer("freqs_sin", sin, persistent=False)

    def forward(self, x):
        return x * self.freqs_cos.to(x.device, x.dtype) + rotate_half(x) * self.freqs_sin.to(x.device, x.dtype)


def get_2d_sincos_pos_embed(embed_dim, grid_size):
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0).reshape(2, 1, grid_size, grid_size)
    return _get_2d_sincos_pos_embed_from_grid(embed_dim, grid)


def _get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    emb_h = _get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = _get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    return np.concatenate([emb_h, emb_w], axis=1)


def _get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega
    out = np.einsum("m,d->md", pos.reshape(-1), omega)
    return np.concatenate([np.sin(out), np.cos(out)], axis=1)


class BottleneckPatchEmbed(nn.Module):
    def __init__(self, img_size=256, patch_size=16, in_chans=3, bottleneck_dim=128, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj1 = nn.Conv2d(in_chans, bottleneck_dim, kernel_size=patch_size, stride=patch_size, bias=False)
        self.proj2 = nn.Conv2d(bottleneck_dim, embed_dim, kernel_size=1, bias=True)

    def forward(self, x):
        if x.shape[-2:] != (self.img_size, self.img_size):
            raise ValueError(f"expected {self.img_size}x{self.img_size}, got {tuple(x.shape[-2:])}")
        return self.proj2(self.proj1(x)).flatten(2).transpose(1, 2)


class TimestepEmbedder(nn.Module):
    def __init__(self, dim, frequency_embedding_size=256):
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        self.mlp = nn.Sequential(nn.Linear(frequency_embedding_size, dim), nn.SiLU(), nn.Linear(dim, dim))

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(0, half, dtype=torch.float32, device=t.device) / half)
        args = t[:, None].float() * freqs[None]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return emb

    def forward(self, t):
        return self.mlp(self.timestep_embedding(t, self.frequency_embedding_size))


class LabelEmbedder(nn.Module):
    def __init__(self, num_classes, dim):
        super().__init__()
        self.num_classes = num_classes
        self.embedding_table = nn.Embedding(num_classes + 1, dim)

    def forward(self, y):
        return self.embedding_table(y)


class FastAttention(nn.Module):
    def __init__(self, dim, num_heads=12, qkv_bias=True, qk_norm=True, attn_backend="flash"):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = RMSNorm(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = RMSNorm(self.head_dim) if qk_norm else nn.Identity()
        self.proj = nn.Linear(dim, dim)
        self.attn_backend = attn_backend

    def _sdpa_context(self):
        if self.attn_backend == "flash":
            return sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH])
        if self.attn_backend == "efficient":
            return sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH])
        if self.attn_backend == "math":
            return sdpa_kernel([SDPBackend.MATH])
        return nullcontext()

    def forward(self, x, rope):
        b, n, c = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = rope(self.q_norm(q))
        k = rope(self.k_norm(k))
        with self._sdpa_context():
            x = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
        x = x.transpose(1, 2).reshape(b, n, c)
        return self.proj(x)


class SwiGLUFFN(nn.Module):
    def __init__(self, dim, hidden_dim, bias=True):
        super().__init__()
        hidden_dim = int(hidden_dim * 2 / 3)
        self.w12 = nn.Linear(dim, 2 * hidden_dim, bias=bias)
        self.w3 = nn.Linear(hidden_dim, dim, bias=bias)

    def forward(self, x):
        x1, x2 = self.w12(x).chunk(2, dim=-1)
        return self.w3(F.silu(x1) * x2)


class SharedAdaLNBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, attn_backend="flash"):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = FastAttention(dim, num_heads=num_heads, qk_norm=True, attn_backend=attn_backend)
        self.norm2 = RMSNorm(dim)
        self.mlp = SwiGLUFFN(dim, int(dim * mlp_ratio))

    def forward(self, x, t6, rope):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = t6.chunk(6, dim=-1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa), rope)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FinalLayerSharedAdaLN(nn.Module):
    def __init__(self, dim, patch_size, out_channels):
        super().__init__()
        self.norm_final = RMSNorm(dim)
        self.linear = nn.Linear(dim, patch_size * patch_size * out_channels)

    def forward(self, x, t2):
        shift, scale = t2.chunk(2, dim=-1)
        return self.linear(modulate(self.norm_final(x), shift, scale))


class JiTSharedTimeAdaLN(nn.Module):
    def __init__(
        self,
        input_size=256,
        patch_size=16,
        in_channels=3,
        dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        num_classes=1000,
        bottleneck_dim=128,
        num_class_tokens=32,
        class_token_start=4,
        attn_backend="flash",
    ):
        super().__init__()
        self.input_size = input_size
        self.patch_size = patch_size
        self.out_channels = in_channels
        self.num_classes = num_classes
        self.num_class_tokens = num_class_tokens
        self.class_token_start = class_token_start

        self.t_embedder = TimestepEmbedder(dim)
        self.y_embedder = LabelEmbedder(num_classes, dim)
        self.x_embedder = BottleneckPatchEmbed(input_size, patch_size, in_channels, bottleneck_dim, dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.x_embedder.num_patches, dim), requires_grad=False)
        self.class_token_pos = nn.Parameter(torch.randn(1, num_class_tokens, dim) * 0.02)

        self.shared_adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))
        self.shared_final_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 2 * dim))
        self.blocks = nn.ModuleList([SharedAdaLNBlock(dim, num_heads, mlp_ratio, attn_backend) for _ in range(depth)])
        self.final_layer = FinalLayerSharedAdaLN(dim, patch_size, in_channels)

        grid = input_size // patch_size
        half_head_dim = dim // num_heads // 2
        self.feat_rope = VisionRotaryEmbeddingFast(half_head_dim, grid, num_prefix_tokens=0)
        self.feat_rope_incontext = VisionRotaryEmbeddingFast(half_head_dim, grid, num_prefix_tokens=num_class_tokens)
        self.initialize_weights()

    def initialize_weights(self):
        def init_linear(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        self.apply(init_linear)
        pos = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.x_embedder.num_patches**0.5))
        self.pos_embed.data.copy_(torch.from_numpy(pos).float().unsqueeze(0))
        nn.init.xavier_uniform_(self.x_embedder.proj1.weight.view(self.x_embedder.proj1.weight.shape[0], -1))
        nn.init.xavier_uniform_(self.x_embedder.proj2.weight.view(self.x_embedder.proj2.weight.shape[0], -1))
        nn.init.constant_(self.x_embedder.proj2.bias, 0)
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)
        nn.init.constant_(self.shared_adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.shared_adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.shared_final_modulation[-1].weight, 0)
        nn.init.constant_(self.shared_final_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x):
        c = self.out_channels
        p = self.patch_size
        h = w = int(x.shape[1] ** 0.5)
        x = x.reshape(x.shape[0], h, w, p, p, c)
        x = torch.einsum("nhwpqc->nchpwq", x)
        return x.reshape(x.shape[0], c, h * p, w * p)

    def forward(self, x, t, y):
        t_emb = self.t_embedder(t)
        y_emb = self.y_embedder(y)
        t6 = self.shared_adaLN_modulation(t_emb)
        t2 = self.shared_final_modulation(t_emb)

        x = self.x_embedder(x) + self.pos_embed
        for i, block in enumerate(self.blocks):
            if self.num_class_tokens > 0 and i == self.class_token_start:
                class_tokens = y_emb.unsqueeze(1).expand(-1, self.num_class_tokens, -1) + self.class_token_pos
                x = torch.cat([class_tokens, x], dim=1)
            rope = self.feat_rope if i < self.class_token_start else self.feat_rope_incontext
            x = block(x, t6, rope)
        if self.num_class_tokens > 0:
            x = x[:, self.num_class_tokens:]
        x = self.final_layer(x, t2)
        return self.unpatchify(x)


def build_jit_b16_shared_time(**kwargs):
    return JiTSharedTimeAdaLN(
        input_size=256,
        patch_size=16,
        dim=768,
        depth=12,
        num_heads=12,
        bottleneck_dim=128,
        num_class_tokens=32,
        class_token_start=4,
        **kwargs,
    )
