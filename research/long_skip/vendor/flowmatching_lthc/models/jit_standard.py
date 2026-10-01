"""Standard AdaLN JiT models wired into the FlowMatching training stack.

Adapted from the local external JiT reference so JiT-B/16 can share the same
train/eval scripts, checkpoint format, W&B handling, and clean/velocity targets
as the LTHC experiments.
"""
# --------------------------------------------------------
# References:
# SiT: https://github.com/willisma/SiT
# Lightning-DiT: https://github.com/hustvl/LightningDiT
# --------------------------------------------------------
import math
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from .jit_shared_adaln import VisionRotaryEmbeddingFast, get_2d_sincos_pos_embed, RMSNorm
from .repa import build_repa_projector, normalize_repa_z_dims


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class BottleneckPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, pca_dim=768, embed_dim=768, bias=True):
        super().__init__()
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj1 = nn.Conv2d(in_chans, pca_dim, kernel_size=patch_size, stride=patch_size, bias=False)
        self.proj2 = nn.Conv2d(pca_dim, embed_dim, kernel_size=1, stride=1, bias=bias)

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj2(self.proj1(x)).flatten(2).transpose(1, 2)
        return x



class PatchEmbed(nn.Module):
    """Full-rank image-to-patch embedding without the 128-d JiT bottleneck."""
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, bias=True):
        super().__init__()
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias)

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        return self.proj(x).flatten(2).transpose(1, 2)


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class TimeScalarSkip(nn.Module):
    """Time-conditioned scalar skip coefficients for velocity prediction."""

    def __init__(self, hidden_size=128, frequency_embedding_size=256):
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        self.net = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, 2, bias=True),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, t):
        emb = TimestepEmbedder.timestep_embedding(t, self.frequency_embedding_size)
        delta = self.net(emb.float())
        alpha = delta[:, 0]
        beta = 1.0 + delta[:, 1]
        return alpha, beta


class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, num_classes, hidden_size):
        super().__init__()
        self.embedding_table = nn.Embedding(num_classes + 1, hidden_size)
        self.num_classes = num_classes

    def forward(self, labels):
        embeddings = self.embedding_table(labels)
        return embeddings


def scaled_dot_product_attention(query, key, value, dropout_p=0.0) -> torch.Tensor:
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1))
    attn_bias = torch.zeros(query.size(0), 1, L, S, dtype=query.dtype).cuda()

    with torch.amp.autocast(query.device.type, enabled=False):
        attn_weight = query.float() @ key.float().transpose(-2, -1) * scale_factor
    attn_weight += attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1)
    attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
    return attn_weight @ value


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=True, qk_norm=True, attn_drop=0., proj_drop=0., attn_backend="flash"):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.q_norm = RMSNorm(head_dim) if qk_norm else nn.Identity()
        self.k_norm = RMSNorm(head_dim) if qk_norm else nn.Identity()

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
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
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        q = self.q_norm(q)
        k = self.k_norm(k)

        q = rope(q)
        k = rope(k)

        dropout_p = self.attn_drop.p if self.training else 0.
        with self._sdpa_context():
            x = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p, is_causal=False)

        x = x.transpose(1, 2).reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwiGLUFFN(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        drop=0.0,
        bias=True
    ) -> None:
        super().__init__()
        hidden_dim = int(hidden_dim * 2 / 3)
        self.w12 = nn.Linear(dim, 2 * hidden_dim, bias=bias)
        self.w3 = nn.Linear(hidden_dim, dim, bias=bias)
        self.ffn_dropout = nn.Dropout(drop)

    def forward(self, x):
        x12 = self.w12(x)
        x1, x2 = x12.chunk(2, dim=-1)
        hidden = F.silu(x1) * x2
        return self.w3(self.ffn_dropout(hidden))


class FinalLayer(nn.Module):
    """
    The final layer of JiT.
    """
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = RMSNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class JiTBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, attn_drop=0.0, proj_drop=0.0, attn_backend="flash"):
        super().__init__()
        self.norm1 = RMSNorm(hidden_size, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, qk_norm=True,
                              attn_drop=attn_drop, proj_drop=proj_drop, attn_backend=attn_backend)
        self.norm2 = RMSNorm(hidden_size, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = SwiGLUFFN(hidden_size, mlp_hidden_dim, drop=proj_drop)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x,  c, feat_rope=None):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=-1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa), rope=feat_rope)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class JiT(nn.Module):
    """
    Just image Transformer.
    """
    def __init__(
        self,
        input_size=256,
        patch_size=16,
        in_channels=3,
        hidden_size=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4.0,
        attn_drop=0.0,
        proj_drop=0.0,
        num_classes=1000,
        bottleneck_dim=128,
        full_patch_embed=False,
        in_context_len=32,
        in_context_start=8,
        attn_backend="flash",
        repa_depth=None,
        repa_z_dims=None,
        repa_projector_dim=2048,
        repa_source="semantic",
    ):
        super().__init__()
        repa_z_dims = normalize_repa_z_dims(repa_z_dims)
        if repa_depth is not None and not (1 <= repa_depth <= depth):
            raise ValueError(f"repa_depth must be in [1, {depth}], got {repa_depth}")
        if repa_z_dims and repa_depth is None:
            raise ValueError("repa_depth must be set when repa_z_dims is non-empty")
        if repa_z_dims and repa_source != "semantic":
            raise ValueError(f"JiT REPA source must be 'semantic', got {repa_source!r}")
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.depth = depth
        self.input_size = input_size
        self.in_context_len = in_context_len
        self.in_context_start = in_context_start
        self.num_classes = num_classes
        self.attn_backend = attn_backend
        self.repa_depth = repa_depth
        self.repa_source = repa_source
        self.repa_z_dims = repa_z_dims
        self.repa_projector_dim = int(repa_projector_dim)

        # time and class embed
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = LabelEmbedder(num_classes, hidden_size)

        # linear embed
        if full_patch_embed:
            self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
        else:
            self.x_embedder = BottleneckPatchEmbed(input_size, patch_size, in_channels, bottleneck_dim, hidden_size, bias=True)

        # use fixed sin-cos embedding
        num_patches = self.x_embedder.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)

        # in-context cls token
        if self.in_context_len > 0:
            self.in_context_posemb = nn.Parameter(torch.zeros(1, self.in_context_len, hidden_size), requires_grad=True)
            torch.nn.init.normal_(self.in_context_posemb, std=.02)

        # rope
        half_head_dim = hidden_size // num_heads // 2
        hw_seq_len = input_size // patch_size
        self.feat_rope = VisionRotaryEmbeddingFast(
            dim=half_head_dim,
            grid_size=hw_seq_len,
            num_prefix_tokens=0
        )
        self.feat_rope_incontext = VisionRotaryEmbeddingFast(
            dim=half_head_dim,
            grid_size=hw_seq_len,
            num_prefix_tokens=self.in_context_len
        )

        # transformer
        self.blocks = nn.ModuleList([
            JiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio,
                     attn_drop=attn_drop if (depth // 4 * 3 > i >= depth // 4) else 0.0,
                     proj_drop=proj_drop if (depth // 4 * 3 > i >= depth // 4) else 0.0,
                     attn_backend=attn_backend)
            for i in range(depth)
        ])

        # linear predict
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        self.repa_projectors = nn.ModuleList(
            [
                build_repa_projector(hidden_size, self.repa_projector_dim, z_dim)
                for z_dim in self.repa_z_dims
            ]
        )

        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        if isinstance(self.x_embedder, BottleneckPatchEmbed):
            w1 = self.x_embedder.proj1.weight.data
            nn.init.xavier_uniform_(w1.view([w1.shape[0], -1]))
            w2 = self.x_embedder.proj2.weight.data
            nn.init.xavier_uniform_(w2.view([w2.shape[0], -1]))
            nn.init.constant_(self.x_embedder.proj2.bias, 0)
        else:
            w = self.x_embedder.proj.weight.data
            nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
            nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize label embedding table:
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)

        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x, p):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    def _project_repa_tokens(self, x):
        batch, tokens, channels = x.shape
        flat = x.reshape(batch * tokens, channels)
        return tuple(
            projector(flat).reshape(batch, tokens, -1)
            for projector in self.repa_projectors
        )

    def forward(self, x, t, y, return_repa=False):
        """
        x: (N, C, H, W)
        t: (N,)
        y: (N,)
        """
        # class and time embeddings
        t_emb = self.t_embedder(t)
        y_emb = self.y_embedder(y)
        c = t_emb + y_emb

        # forward JiT
        x = self.x_embedder(x)
        x += self.pos_embed

        repa_outputs = ()
        for i, block in enumerate(self.blocks):
            # in-context
            if self.in_context_len > 0 and i == self.in_context_start:
                in_context_tokens = y_emb.unsqueeze(1).repeat(1, self.in_context_len, 1)
                in_context_tokens += self.in_context_posemb
                x = torch.cat([in_context_tokens, x], dim=1)
            x = block(x, c, self.feat_rope if i < self.in_context_start else self.feat_rope_incontext)
            if return_repa and self.repa_projectors and i + 1 == self.repa_depth:
                spatial_tokens = x[:, self.in_context_len:] if self.in_context_len > 0 else x
                repa_outputs = self._project_repa_tokens(spatial_tokens)

        x = x[:, self.in_context_len:]

        x = self.final_layer(x, c)
        output = self.unpatchify(x, self.patch_size)

        if return_repa:
            return output, repa_outputs
        return output


class LongSkipVelocityJiT(nn.Module):
    """JiT backbone with a learned scalar long skip for direct velocity output.

    The model predicts ``alpha(t) * x_t + beta(t) * backbone(x_t, t, y)``.
    Initialization is exactly the wrapped backbone path: alpha=0 and beta=1.
    """

    def __init__(self, *args, skip_hidden_size=128, **kwargs):
        super().__init__()
        self.backbone = JiT(*args, **kwargs)
        self.skip = TimeScalarSkip(hidden_size=skip_hidden_size)
        self.in_channels = self.backbone.in_channels
        self.out_channels = self.backbone.out_channels
        self.patch_size = self.backbone.patch_size
        self.num_heads = self.backbone.num_heads
        self.hidden_size = self.backbone.hidden_size
        self.depth = self.backbone.depth
        self.input_size = self.backbone.input_size
        self.in_context_len = self.backbone.in_context_len
        self.in_context_start = self.backbone.in_context_start
        self.num_classes = self.backbone.num_classes
        self.attn_backend = self.backbone.attn_backend

    def forward(self, x, t, y, return_repa=False):
        if return_repa:
            pred, repa_outputs = self.backbone(x, t, y, return_repa=True)
        else:
            pred = self.backbone(x, t, y)
            repa_outputs = ()
        alpha, beta = self.skip(t)
        alpha = alpha.view(-1, 1, 1, 1).to(dtype=x.dtype)
        beta = beta.view(-1, 1, 1, 1).to(dtype=pred.dtype)
        out = alpha * x + beta * pred
        if return_repa:
            return out, repa_outputs
        return out

    @torch.no_grad()
    def skip_effective_values(self, t):
        alpha, beta = self.skip(t)
        one_minus_t = 1.0 - t.float()
        return {
            "alpha": alpha.float(),
            "beta": beta.float(),
            "clean_alpha_effective": -one_minus_t * alpha.float(),
            "clean_beta_effective": one_minus_t * beta.float(),
        }


def _drop_flowmatching_kwargs(kwargs):
    return kwargs


def JiT_B_16(**kwargs):
    kwargs = _drop_flowmatching_kwargs(kwargs)
    return JiT(depth=12, hidden_size=768, num_heads=12,
               bottleneck_dim=128, in_context_len=0, in_context_start=0, patch_size=16, **kwargs)


def JiT_S_16_FullPatch(**kwargs):
    """DiT-S / JiT-S: hidden 384, no bottleneck, no context tokens."""
    kwargs = _drop_flowmatching_kwargs(kwargs)
    return JiT(depth=12, hidden_size=384, num_heads=6,
               full_patch_embed=True, in_context_len=0, in_context_start=0, patch_size=16, **kwargs)


def JiT_B_16_FullPatch(**kwargs):
    kwargs = _drop_flowmatching_kwargs(kwargs)
    return JiT(depth=12, hidden_size=768, num_heads=12,
               full_patch_embed=True, in_context_len=0, in_context_start=0, patch_size=16, **kwargs)


def JiT_B_16_FullPatch_LongSkip(**kwargs):
    kwargs = _drop_flowmatching_kwargs(kwargs)
    return LongSkipVelocityJiT(depth=12, hidden_size=768, num_heads=12,
                               full_patch_embed=True, in_context_len=0, in_context_start=0,
                               patch_size=16, **kwargs)


def JiT_B_16_CTX(**kwargs):
    kwargs = _drop_flowmatching_kwargs(kwargs)
    return JiT(depth=12, hidden_size=768, num_heads=12,
               bottleneck_dim=128, in_context_len=32, in_context_start=4, patch_size=16, **kwargs)


def JiT_B_16_FullPatch_CTX(**kwargs):
    """JiT-B/16: full patch embed (no bottleneck) + paper-style in-context tokens."""
    kwargs = _drop_flowmatching_kwargs(kwargs)
    return JiT(depth=12, hidden_size=768, num_heads=12,
               full_patch_embed=True, in_context_len=32, in_context_start=4, patch_size=16, **kwargs)


def JiT_B_32(**kwargs):
    kwargs = _drop_flowmatching_kwargs(kwargs)
    return JiT(depth=12, hidden_size=768, num_heads=12,
               bottleneck_dim=128, in_context_len=32, in_context_start=4, patch_size=32, **kwargs)

def JiT_L_16(**kwargs):
    kwargs = _drop_flowmatching_kwargs(kwargs)
    return JiT(depth=24, hidden_size=1024, num_heads=16,
               bottleneck_dim=128, in_context_len=32, in_context_start=8, patch_size=16, **kwargs)

def JiT_L_32(**kwargs):
    kwargs = _drop_flowmatching_kwargs(kwargs)
    return JiT(depth=24, hidden_size=1024, num_heads=16,
               bottleneck_dim=128, in_context_len=32, in_context_start=8, patch_size=32, **kwargs)

def JiT_H_16(**kwargs):
    kwargs = _drop_flowmatching_kwargs(kwargs)
    return JiT(depth=32, hidden_size=1280, num_heads=16,
               bottleneck_dim=256, in_context_len=32, in_context_start=10, patch_size=16, **kwargs)

def JiT_H_32(**kwargs):
    kwargs = _drop_flowmatching_kwargs(kwargs)
    return JiT(depth=32, hidden_size=1280, num_heads=16,
               bottleneck_dim=256, in_context_len=32, in_context_start=10, patch_size=32, **kwargs)


JiT_models = {
    'JiT-B/16': JiT_B_16,
    'JiT-S/16-FullPatch': JiT_S_16_FullPatch,
    'JiT-B/16-FullPatch': JiT_B_16_FullPatch,
    'JiT-B/16-FullPatch-LongSkip': JiT_B_16_FullPatch_LongSkip,
    'JiT-B/16-CTX': JiT_B_16_CTX,
    'JiT-B/16-FullPatch-CTX': JiT_B_16_FullPatch_CTX,
    'JiT-B/32': JiT_B_32,
    'JiT-L/16': JiT_L_16,
    'JiT-L/32': JiT_L_32,
    'JiT-H/16': JiT_H_16,
    'JiT-H/32': JiT_H_32,
}
