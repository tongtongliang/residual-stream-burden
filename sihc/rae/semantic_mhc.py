"""Group 4: original RAEv2 DH with sublayer-wise semantic mHC only."""

import torch
from torch import nn
from torch.nn import functional as F

from stage2.models.DDT import DiTwDDTHeadIG
from sihc.models.baselines import MHCConnection


class SemanticMHCBlock(nn.Module):
    def __init__(self, original, width, index, backend):
        super().__init__()
        self.norm1, self.norm2 = original.norm1, original.norm2
        self.attn, self.mlp = original.attn, original.mlp
        self.attn_hc = MHCConnection(width, n=4, backend=backend, init_residual_index=2 * index)
        self.mlp_hc = MHCConnection(width, n=4, backend=backend, init_residual_index=2 * index + 1)

    def forward(self, streams, rope, attn_mask=None):
        u, post, residual = self.attn_hc.mix_and_aggregate(streams)
        delta = self.attn(self.norm1(u.transpose(0, 1)), rope=rope, attn_mask=attn_mask)
        streams = self.attn_hc.combine(delta.transpose(0, 1).contiguous(), post, streams, residual)
        u, post, residual = self.mlp_hc.mix_and_aggregate(streams)
        delta = self.mlp(self.norm2(u.transpose(0, 1))).transpose(0, 1).contiguous()
        # Auxiliary prediction observes the updated MLP workspace BEFORE write.
        workspace = (u + delta).transpose(0, 1).contiguous()
        streams = self.mlp_hc.combine(delta, post, streams, residual)
        return streams, workspace


class SemanticMHCDH(DiTwDDTHeadIG):
    def __init__(self, mhc_n=4, mhc_backend="nvidia_fused", **kwargs):
        if mhc_n != 4:
            raise ValueError("Group4 requires N=4")
        if kwargs.get("enable_repa", False):
            raise ValueError("Group4 uses the x-prediction base head, not a second REPA loss")
        super().__init__(**kwargs)
        if not 1 <= self.base_model_depth <= self.num_enc_blocks:
            raise ValueError("base_model_depth must lie inside the semantic backbone")
        for i in range(self.num_enc_blocks):
            self.blocks[i] = SemanticMHCBlock(self.blocks[i], self.enc_hidden_size, i, mhc_backend)

    def forward(self, x, t, return_intermediate=False, **condition_kwargs):
        if return_intermediate:
            raise ValueError("Group4 does not enable the separate legacy REPA projector")
        seq, t_emb_base = self._build_sequence(x, t, condition_kwargs)
        mask = self._build_attn_mask(seq, condition_kwargs)
        streams = seq.transpose(0, 1).unsqueeze(-1).expand(-1, -1, -1, 4).contiguous()
        base = None
        patches = self.s_embedder.num_patches
        for i in range(self.num_enc_blocks):
            streams, workspace = self.blocks[i](streams, self.enc_rope, mask)
            if i + 1 == self.base_model_depth:
                base = workspace[:, :patches].contiguous()
        semantic = streams.mean(dim=-1).transpose(0, 1)[:, :patches].contiguous()
        semantic = self.s_projector(F.silu(t_emb_base + semantic))
        output = self.x_embedder(x)
        for i in range(self.num_dec_blocks):
            output = self.blocks[self.num_enc_blocks + i](output, semantic, self.dec_rope)
        output = self.unpatchify(self.final_layer(output, semantic), self.x_patch_size)
        base = F.silu(t_emb_base + base)
        base = self.unpatchify(self.base_final_layer(base, base), self.s_patch_size)
        return output, base
