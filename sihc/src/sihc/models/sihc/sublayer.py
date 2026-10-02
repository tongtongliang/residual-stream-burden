"""Fused sublayer-wise SiHC: independent attention/MLP read and write maps."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .components import DirectPatchEmbedNHWC
from .model import SiHCModel, SiHCInContextModel
from .reference import SiHCSublayerReferenceBlock
from .sublayer_kernels import sublayer_read, sublayer_triangular, sublayer_write


class _DirectPatchEmbedGEMM(DirectPatchEmbedNHWC):
    """Same non-overlapping patch projection, without a dense Conv2d layout copy.

    Keep the existing proj.weight/proj.bias keys, shapes, parameter ordering and
    contiguous strides, including compatibility with old optimizer moments.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-2:] != self.img_size:
            raise ValueError(f"expected image size {self.img_size}, got {tuple(x.shape[-2:])}")
        ph, pw = self.patch_size
        patches = x.unfold(2, ph, ph).unfold(3, pw, pw)
        patches = patches.permute(0, 2, 3, 1, 4, 5).flatten(3)
        return F.linear(patches, self.proj.weight.flatten(1), self.proj.bias)


class SublayerSiHCModel(SiHCModel):
    """Stages count blocks; each block has two independent connection events.

    Attention writes before the MLP reads with its own alpha. Preserve the
    existing six-way AdaLN, gated branches, and reference parameter names.
    Block-wise checkpoints are not implicitly converted to this algorithm.
    """

    block_cls = SiHCSublayerReferenceBlock
    supports_tuple_backend = False
    x_embedder_cls = _DirectPatchEmbedGEMM
    uses_fused_b4_kernels = False
    allowed_stage_sizes = (8, 12)

    def __init__(self, **kwargs):
        if kwargs.get("patch_size", 4) not in (4, 8):
            raise ValueError("fused sublayer SiHC currently supports pixel P4/P8")
        if (kwargs.get("repa_z_dims") or kwargs.get("repa_depth") is not None) and not getattr(self, "supports_repa", False):
            raise ValueError("REPA tap semantics are not yet defined for sublayer SiHC")
        super().__init__(**kwargs)
        self.connection_frequency = "sublayer"

    def _run_fused_schedule_stage(
        self, x0, cond, blocks, stage_start, repa_outputs=None,
    ):
        connections = [connection for block in blocks for connection in
                       (block.attention_connection, block.mlp_connection)]
        alpha = torch.stack([c.alpha(dtype=x0.dtype) for c in connections]).contiguous()
        beta = torch.stack([c.beta(dtype=x0.dtype) for c in connections]).contiguous()
        base = sublayer_read(x0, alpha)
        gamma = (alpha[:, None] * beta[None, :]).sum(dim=-1)
        updates = []
        for i, block in enumerate(blocks):
            modulation = block.branch.conditioning(cond)
            z = base[2 * i]
            if updates:
                z = sublayer_triangular(z, updates, gamma[2 * i, :2 * i].contiguous())
            updates.append(block.branch.attention_update(
                z, modulation, self.workspace_rope).contiguous())
            z = sublayer_triangular(
                base[2 * i + 1], updates, gamma[2 * i + 1, :2 * i + 1].contiguous())
            updates.append(block.branch.mlp_update(z, modulation).contiguous())
        return sublayer_write(x0, updates, beta)


class SublayerSiHCInContextModel(SiHCInContextModel, SublayerSiHCModel):
    """Historical class prefixes, with independent spatial sublayer routing.

    Reuse context initialization, RoPE and forward from SiHCInContextModel.
    Prefixes have ordinary residual updates and never enter spatial routing.
    """

    supports_repa = True

    def _run_fused_schedule_stage_incontext(
        self, x0, cond, context_tokens, blocks, stage_start, repa_outputs=None,
    ):
        connections = [connection for block in blocks for connection in
                       (block.attention_connection, block.mlp_connection)]
        alpha = torch.stack([c.alpha(dtype=x0.dtype) for c in connections]).contiguous()
        beta = torch.stack([c.beta(dtype=x0.dtype) for c in connections]).contiguous()
        base = sublayer_read(x0, alpha)
        gamma = (alpha[:, None] * beta[None, :]).sum(dim=-1)
        updates = []
        for i, block in enumerate(blocks):
            modulation = block.branch.conditioning(cond)
            use_context = stage_start + i >= self.in_context_start
            for part in range(2):
                event = 2 * i + part
                z = base[event]
                if updates:
                    z = sublayer_triangular(z, updates, gamma[event, :event].contiguous())
                seq = torch.cat((context_tokens, z), dim=1) if use_context else z
                if part == 0:
                    rope = self.workspace_rope_incontext if use_context else self.workspace_rope
                    dseq = block.branch.attention_update(seq, modulation, rope)
                else:
                    dseq = block.branch.mlp_update(seq, modulation)
                if use_context:
                    context_tokens = context_tokens + dseq[:, :self.in_context_len]
                    # Removing prefixes produces a batch-strided view. All
                    # spatial Triton inputs MUST be packed, in both branches.
                    dz = dseq[:, self.in_context_len:].contiguous()
                else:
                    dz = dseq.contiguous()
                updates.append(dz)
            if (repa_outputs is not None and self.repa_projectors
                    and stage_start + i + 1 == self.repa_depth):
                # z is the spatial MLP read, dz its gated update. No prefix
                # enters the REPA interface, even when context is enabled.
                source = dz if self.repa_source == "dz" else z + dz
                repa_outputs.extend(self._project_repa_tokens(source))
        return sublayer_write(x0, updates, beta), context_tokens


def SiHCSublayer_40x1024_Context_REPA(**kwargs):
    kwargs.setdefault("patch_size", 4)
    kwargs.setdefault("hidden_size", 1024)
    kwargs.setdefault("num_heads", 16)
    kwargs.setdefault("depth", 40)
    kwargs.setdefault("fused_stage_size", 8)
    kwargs.setdefault("in_context_len", 32)
    kwargs.setdefault("in_context_start", 8)
    return SublayerSiHCInContextModel(**kwargs)


def SiHCSublayer_B4_Context(**kwargs):
    kwargs.setdefault("patch_size", 4)
    kwargs.setdefault("in_context_len", 32)
    kwargs.setdefault("in_context_start", 4)
    return SublayerSiHCInContextModel(**kwargs)


def SiHCSublayer_B4(**kwargs):
    kwargs.setdefault("patch_size", 4)
    return SublayerSiHCModel(**kwargs)


def SiHCSublayer_B8(**kwargs):
    kwargs.setdefault("patch_size", 8)
    return SublayerSiHCModel(**kwargs)


def SiHCSublayer_2x12_D1024_B4_CTX32S8(**kwargs):
    """L-wide sublayer P4: 24x1024, stages 12+12, ctx32 entering at block index 8."""
    kwargs.setdefault('patch_size', 4)
    kwargs.setdefault('hidden_size', 1024)
    kwargs.setdefault('depth', 24)
    kwargs.setdefault('num_heads', 16)
    kwargs.setdefault('workspace_grid', 16)
    kwargs.setdefault('stage_sizes', (12, 12))
    kwargs.setdefault('in_context_len', 32)
    kwargs.setdefault('in_context_start', 8)
    return SublayerSiHCInContextModel(**kwargs)


def SiHCSublayer_8x12x12_D1280_B4_CTX32S8(**kwargs):
    """H-wide sublayer P4: 32x1280, stages 8+12+12, ctx32 entering at block index 8."""
    kwargs.setdefault('patch_size', 4)
    kwargs.setdefault('hidden_size', 1280)
    kwargs.setdefault('depth', 32)
    kwargs.setdefault('num_heads', 16)
    kwargs.setdefault('workspace_grid', 16)
    kwargs.setdefault('stage_sizes', (8, 12, 12))
    kwargs.setdefault('in_context_len', 32)
    kwargs.setdefault('in_context_start', 8)
    return SublayerSiHCInContextModel(**kwargs)


SUBLAYER_SIHC_MODELS = {
    "sihc_sublayer_2x12_d1024_b4_ctx32s8": SiHCSublayer_2x12_D1024_B4_CTX32S8,
    "sihc_sublayer_8x12x12_d1280_b4_ctx32s8": SiHCSublayer_8x12x12_D1280_B4_CTX32S8,
    "sihc_sublayer_5x8_d1024_b4_ctx32s8": SiHCSublayer_40x1024_Context_REPA,
    "sihc_sublayer_1x12_d768_b4_ctx32s4": SiHCSublayer_B4_Context,
    "sihc_sublayer_1x12_d768_b4": SiHCSublayer_B4,
    "sihc_sublayer_1x12_d768_b8": SiHCSublayer_B8,
}
