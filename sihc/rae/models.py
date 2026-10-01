"""Group 2 stage-2 models for the official RAE training pipeline.

The official RAE checkout remains an external dependency.  Put ``RAE/src`` on
``PYTHONPATH`` (the Group 2 launcher does this) before importing this module.
"""

from __future__ import annotations

import torch
from torch import nn

try:
    from stage2.models.lightningDiT import LightningDiT
    from stage2.models.model_utils import GaussianFourierEmbedding, LabelEmbedder, modulate
except ImportError as exc:  # pragma: no cover - exercised by collaborator preflight
    raise ImportError(
        "sihc.rae.models requires the official RAE/src directory on PYTHONPATH; "
        "run scripts/group2_prepare.py and use scripts/group2_launch.py"
    ) from exc

from sihc.models.baselines import (
    FactorizedPatchEmbed,
    MHCConnection,
    MHCJiTB16,
    PlainJiTB16,
    PlainJiTXL16,
)


class _TrainingAwareLabelEmbedder(LabelEmbedder):
    """Expose the RAE label-drop policy through JiT's one-argument interface."""

    def forward(self, labels: torch.Tensor) -> torch.Tensor:
        return super().forward(labels, self.training)


class _PredictionTarget:
    prediction_target: str
    clean_eps: float

    def _set_prediction_target(self, prediction_target: str, clean_eps: float) -> None:
        if prediction_target not in {"clean", "velocity"}:
            raise ValueError(f"unsupported prediction target: {prediction_target}")
        self.prediction_target = prediction_target
        self.clean_eps = float(clean_eps)

    def _to_velocity(
        self,
        noisy: torch.Tensor,
        timestep: torch.Tensor,
        prediction: torch.Tensor,
    ) -> torch.Tensor:
        if self.prediction_target == "velocity":
            return prediction
        # RAE uses x_t=(1-t)*x_data+t*x_noise, hence v=(x_t-x_data)/t.
        shape = (timestep.shape[0],) + (1,) * (noisy.ndim - 1)
        denominator = timestep.reshape(shape).clamp_min(self.clean_eps)
        return (noisy - prediction) / denominator


class RAELightningDiT(_PredictionTarget, LightningDiT):
    """The RAE paper's DiT baseline, explicitly without a DDT/DH head."""

    def __init__(
        self,
        prediction_target: str = "velocity",
        clean_eps: float = 0.05,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._set_prediction_target(prediction_target, clean_eps)

    def forward(self, x, t=None, y=None):
        prediction = super().forward(x, t=t, y=y)
        return self._to_velocity(x, t, prediction)


class _RAEMHCBlock(nn.Module):
    """One RAE LightningDiT block with N=4 mHC around both sublayers."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float,
        use_qknorm: bool,
        use_swiglu: bool,
        use_rmsnorm: bool,
        mhc_n: int,
        mhc_backend: str,
        layer_index: int,
    ):
        super().__init__()
        # Reuse the exact RAE block components instead of maintaining a fork.
        from stage2.models.lightningDiT import LightningDiTBlock

        block = LightningDiTBlock(
            hidden_size,
            num_heads,
            mlp_ratio=mlp_ratio,
            use_qknorm=use_qknorm,
            use_swiglu=use_swiglu,
            use_rmsnorm=use_rmsnorm,
            wo_shift=False,
        )
        self.norm1 = block.norm1
        self.attn = block.attn
        self.norm2 = block.norm2
        self.mlp = block.mlp
        self.adaLN_modulation = block.adaLN_modulation
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

    def forward(self, x, c, feat_rope=None):
        shift_a, scale_a, gate_a, shift_m, scale_m, gate_m = (
            self.adaLN_modulation(c).chunk(6, dim=-1)
        )
        collapsed, post, residual = self.attn_hc.mix_and_aggregate(x)
        branch = gate_a.unsqueeze(1) * self.attn(
            modulate(self.norm1(collapsed.transpose(0, 1)), shift_a, scale_a),
            rope=feat_rope,
        )
        x = self.attn_hc.combine(branch.transpose(0, 1).contiguous(), post, x, residual)

        collapsed, post, residual = self.mlp_hc.mix_and_aggregate(x)
        branch = gate_m.unsqueeze(1) * self.mlp(
            modulate(self.norm2(collapsed.transpose(0, 1)), shift_m, scale_m)
        )
        return self.mlp_hc.combine(
            branch.transpose(0, 1).contiguous(), post, x, residual
        )


class RAEMHCLightningDiT(RAELightningDiT):
    """RAE DiT-XL with sublayer-wise NVIDIA mHC, N=4, and no DDT head."""

    def __init__(
        self,
        mhc_n: int = 4,
        mhc_backend: str = "nvidia_fused",
        mlp_ratio: float = 4.0,
        use_qknorm: bool = False,
        use_swiglu: bool = True,
        use_rmsnorm: bool = True,
        **kwargs,
    ):
        super().__init__(
            mlp_ratio=mlp_ratio,
            use_qknorm=use_qknorm,
            use_swiglu=use_swiglu,
            use_rmsnorm=use_rmsnorm,
            **kwargs,
        )
        self.mhc_n = mhc_n
        self.mhc_backend = mhc_backend
        self.blocks = nn.ModuleList(
            _RAEMHCBlock(
                self.hidden_size,
                self.num_heads,
                mlp_ratio,
                use_qknorm,
                use_swiglu,
                use_rmsnorm,
                mhc_n,
                mhc_backend,
                index,
            )
            for index in range(self.depth)
        )
        self.initialize_weights()

    def forward(self, x, t=None, y=None):
        noisy = x
        x = self.x_embedder(x) + self.pos_embed.to(dtype=x.dtype)
        c = self.t_embedder(t) + self.y_embedder(y, self.training)
        x = x.transpose(0, 1).unsqueeze(-1).expand(
            -1, -1, -1, self.mhc_n
        ).contiguous()
        for block in self.blocks:
            x = block(x, c, feat_rope=self.feat_rope)
        x = x.mean(dim=-1).transpose(0, 1)
        prediction = self.unpatchify(self.final_layer(x, c))
        return self._to_velocity(noisy, t, prediction)


class _RAEJiTMixin(_PredictionTarget):
    def _configure_rae_conditioning(
        self,
        prediction_target: str,
        clean_eps: float,
        class_dropout_prob: float,
    ) -> None:
        self.t_embedder = GaussianFourierEmbedding(self.hidden_size)
        self.y_embedder = _TrainingAwareLabelEmbedder(
            self.num_classes, self.hidden_size, class_dropout_prob
        )
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)
        self._set_prediction_target(prediction_target, clean_eps)

    def _forward_as_velocity(self, x, t, y, parent_forward):
        prediction = parent_forward(x, t, y)
        return self._to_velocity(x, t, prediction)


class RAEJiTB(_RAEJiTMixin, PlainJiTB16):
    """JiT-B operating on the normalized 16x16x768 DINOv2-B RAE latent."""

    def __init__(
        self,
        prediction_target: str = "velocity",
        clean_eps: float = 0.05,
        class_dropout_prob: float = 0.1,
        **kwargs,
    ):
        kwargs.setdefault("image_size", 16)
        kwargs.setdefault("patch_size", 1)
        kwargs.setdefault("in_channels", 768)
        kwargs.setdefault("hidden_size", 768)
        kwargs.setdefault("depth", 12)
        kwargs.setdefault("num_heads", 12)
        super().__init__(**kwargs)
        self._configure_rae_conditioning(
            prediction_target, clean_eps, class_dropout_prob
        )

    def forward(self, x, t, y):
        return self._forward_as_velocity(x, t, y, super().forward)


class RAEJiTBottleneck(RAEJiTB):
    """RAE JiT-B with a linear 768 -> 128 -> 768 token bottleneck."""

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
        nn.init.xavier_uniform_(
            self.x_embedder.proj_in.weight.view(bottleneck_dim, -1)
        )
        nn.init.zeros_(self.x_embedder.proj_in.bias)
        nn.init.xavier_uniform_(self.x_embedder.proj_out.weight.flatten(1))
        nn.init.zeros_(self.x_embedder.proj_out.bias)


class RAEJiTXL(_RAEJiTMixin, PlainJiTXL16):
    """Matched 28x1152 JiT backbone on the 16x16x768 DINOv2-B latent."""

    def __init__(
        self,
        prediction_target: str = "velocity",
        clean_eps: float = 0.05,
        class_dropout_prob: float = 0.1,
        **kwargs,
    ):
        kwargs.setdefault("image_size", 16)
        kwargs.setdefault("patch_size", 1)
        kwargs.setdefault("in_channels", 768)
        super().__init__(**kwargs)
        self._configure_rae_conditioning(
            prediction_target, clean_eps, class_dropout_prob
        )

    def forward(self, x, t, y):
        return self._forward_as_velocity(x, t, y, super().forward)


class RAEMHCJiTB(_RAEJiTMixin, MHCJiTB16):
    """RAE JiT-B with sublayer-wise NVIDIA mHC, N=4."""

    def __init__(
        self,
        prediction_target: str = "velocity",
        clean_eps: float = 0.05,
        class_dropout_prob: float = 0.1,
        **kwargs,
    ):
        kwargs.setdefault("image_size", 16)
        kwargs.setdefault("patch_size", 1)
        kwargs.setdefault("in_channels", 768)
        kwargs.setdefault("hidden_size", 768)
        kwargs.setdefault("depth", 12)
        kwargs.setdefault("num_heads", 12)
        kwargs.setdefault("mhc_n", 4)
        super().__init__(**kwargs)
        self._configure_rae_conditioning(
            prediction_target, clean_eps, class_dropout_prob
        )

    def forward(self, x, t, y):
        return self._forward_as_velocity(x, t, y, super().forward)
