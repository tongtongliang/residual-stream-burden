"""Spatially Indexed Hyperconnection dense diffusion models.

This file keeps the image-token architecture dense: the high-resolution
residual stream is accumulated and passed to the normal patch prediction head.
The cross-scale residual interface is deliberately simple and linear:

    z_l[b, p, c] = sum_s x0[b, p, s, c] * alpha_l[c, s]
                 + sum_{j < l} gamma[l, j, c] * dz_j[b, p, c]
    x_L[b, p, s, c] = x0[b, p, s, c] + sum_j beta_j[c, s] * dz_j[b, p, c]

where gamma[l, j, c] = sum_s alpha_l[c, s] * beta_j[c, s].
There is no softmax, no eta, and no cross-layer sharing.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .components import (
    DirectPatchEmbedNHWC,
    FactorizedPatchEmbedNHWC,
    FinalLayer,
    LabelEmbedder,
    TimestepEmbedder,
    VisionRotaryEmbeddingFast,
    WorkspaceJiTBranch,
)
from .blockwise_kernels import blockwise_read, blockwise_triangular, blockwise_write
from .kernels import all_base_b4_8_split_traceable as triton_all_base_b4_8_split_traceable
from .kernels import all_base_b4_12_split_traceable as triton_all_base_b4_12_split_traceable
from .kernels import all_base_p8_12_split_traceable as triton_all_base_p8_12_split_traceable
from .kernels import final_accumulate_b4_8_traceable as triton_final_accumulate_b4_8_traceable
from .kernels import final_accumulate_b4_12_traceable as triton_final_accumulate_b4_12_traceable
from .kernels import final_accumulate_p8_12_traceable as triton_final_accumulate_p8_12_traceable
from .kernels import cell_mean_b4_traceable as triton_cell_mean_b4_traceable
from .kernels import (
    triangular_accumulate_b4_8_ptr_active_traceable_ops as triton_triangular_active_ops_8,
    triangular_accumulate_b4_12_ptr_active_traceable_ops as triton_triangular_active_ops_12,
)




def _build_repa_projector(hidden_size: int, projector_dim: int, z_dim: int) -> nn.Module:
    return nn.Sequential(
        nn.Linear(hidden_size, projector_dim),
        nn.SiLU(),
        nn.Linear(projector_dim, projector_dim),
        nn.SiLU(),
        nn.Linear(projector_dim, z_dim),
    )


def _normalize_repa_z_dims(repa_z_dims: tuple[int, ...] | list[int] | None) -> tuple[int, ...]:
    if repa_z_dims is None:
        return ()
    return tuple(int(dim) for dim in repa_z_dims)

class SpatiallyIndexedHyperConnection(nn.Module):
    """Feature-wise read/write maps over fixed, spatially named slots."""

    def __init__(
        self,
        high_grid: int,
        workspace_grid: int,
        hidden_size: int,
        read_init: float | None = None,
        write_init: float = 1.0,
    ) -> None:
        super().__init__()
        if high_grid <= 0 or workspace_grid <= 0 or high_grid % workspace_grid:
            raise ValueError(
                "SiHC requires a positive high-resolution grid divisible by "
                "the workspace grid"
            )
        self.high_grid = high_grid
        self.workspace_grid = workspace_grid
        self.hidden_size = hidden_size
        self.cell_size = high_grid // workspace_grid
        self.cell_tokens = self.cell_size * self.cell_size
        read_value = (
            (1.0 / self.cell_tokens) if read_init is None else float(read_init)
        )
        self.read_weight = nn.Parameter(
            torch.full((hidden_size, self.cell_tokens), read_value)
        )
        self.write_weight = nn.Parameter(
            torch.full((hidden_size, self.cell_tokens), float(write_init))
        )

    def alpha(self, dtype: torch.dtype | None = None) -> torch.Tensor:
        return self.read_weight if dtype is None else self.read_weight.to(dtype=dtype)

    def beta(self, dtype: torch.dtype | None = None) -> torch.Tensor:
        return self.write_weight if dtype is None else self.write_weight.to(dtype=dtype)


class SiHCWriteOnlyConnection(nn.Module):
    """Slot-wise write map; the stage read is fixed and parameter-free."""

    def __init__(
        self,
        high_grid: int,
        workspace_grid: int,
        hidden_size: int,
        write_init: float = 1.0,
    ) -> None:
        super().__init__()
        if high_grid != 64 or workspace_grid != 16:
            raise ValueError("write-only SiHC currently requires high_grid=64 and workspace_grid=16")
        self.high_grid = high_grid
        self.workspace_grid = workspace_grid
        self.hidden_size = hidden_size
        self.cell_tokens = 16
        self.write_weight = nn.Parameter(
            torch.full((hidden_size, self.cell_tokens), float(write_init))
        )

    def beta(self, dtype: torch.dtype | None = None) -> torch.Tensor:
        return self.write_weight if dtype is None else self.write_weight.to(dtype=dtype)


class SiHCWriteOnlyBlock(nn.Module):
    """Workspace branch whose update is written only at the stage boundary."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        high_grid: int,
        workspace_grid: int = 16,
        mlp_ratio: float = 4.0,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        attn_backend: str = "flash",
    ) -> None:
        super().__init__()
        self.connection = SiHCWriteOnlyConnection(
            high_grid, workspace_grid, hidden_size
        )
        self.branch = WorkspaceJiTBranch(
            hidden_size,
            num_heads,
            mlp_ratio=mlp_ratio,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            attn_backend=attn_backend,
        )

    def branch_update(
        self, z: torch.Tensor, c: torch.Tensor, rope: VisionRotaryEmbeddingFast
    ) -> torch.Tensor:
        return self.branch(z, c, rope)


class SiHCBlock(nn.Module):
    """SiHC block with standard per-block AdaLN modulation."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        high_grid: int,
        workspace_grid: int = 16,
        mlp_ratio: float = 4.0,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        attn_backend: str = "flash",
    ) -> None:
        super().__init__()
        self.connection = SpatiallyIndexedHyperConnection(high_grid, workspace_grid, hidden_size)
        self.branch = WorkspaceJiTBranch(
            hidden_size,
            num_heads,
            mlp_ratio=mlp_ratio,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            attn_backend=attn_backend,
        )

    def branch_update(self, z: torch.Tensor, c: torch.Tensor, rope: VisionRotaryEmbeddingFast) -> torch.Tensor:
        return self.branch(z, c, rope)


class _SiHCBase(nn.Module):
    """Common dense fused schedule for SiHC models.

    Direct patch projection is the default for registered SiHC models.
    Historical factorized-linear stems are reconstructed only when checkpoint
    metadata or state keys explicitly request them.
    """

    block_cls: type[nn.Module]
    final_cls: type[nn.Module]
    x_embedder_cls: type[nn.Module] = DirectPatchEmbedNHWC
    uses_fused_b4_kernels = True
    uses_fused_p8_kernels = False
    supports_tuple_backend = True
    allowed_stage_sizes: tuple[int, ...] | None = (8, 12)

    def __init__(
        self,
        input_size: int = 256,
        patch_size: int = 4,
        in_channels: int = 3,
        hidden_size: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        num_classes: int = 1000,
        bottleneck_dim: int = 128,
        patch_embed_type: str = "direct",
        workspace_grid: int = 16,
        connection_groups: int | None = None,
        fused_stage_size: int = 12,
        stage_sizes: tuple[int, ...] | list[int] | None = None,
        attn_backend: str = "flash",
        repa_depth: int | None = None,
        repa_z_dims: tuple[int, ...] | list[int] | None = None,
        repa_projector_dim: int = 2048,
        repa_source: str = "z_plus_dz",
        kernel_backend: str = "legacy",
    ) -> None:
        super().__init__()
        if kernel_backend not in ("legacy", "tuple"):
            raise ValueError("kernel_backend must be 'legacy' or 'tuple'")
        if kernel_backend == "tuple" and not self.supports_tuple_backend:
            raise ValueError("kernel_backend='tuple' selects block-wise fused routing only")
        self.kernel_backend = kernel_backend
        if input_size != 256 or workspace_grid != 16:
            raise ValueError("SiHC currently requires input=256 and workspace_grid=16")
        high_grid = input_size // patch_size
        if input_size % patch_size or high_grid % workspace_grid:
            raise ValueError(
                "SiHC patch size must produce a high-resolution grid divisible "
                "by the workspace grid"
            )
        if self.uses_fused_b4_kernels and patch_size != 4:
            raise ValueError("SiHC fused kernels currently require patch_size=4")
        if self.uses_fused_p8_kernels and patch_size != 8:
            raise ValueError("SiHC p8 fused kernels require patch_size=8")
        allowed_stage_sizes = self.allowed_stage_sizes
        if stage_sizes is None:
            if fused_stage_size <= 0 or depth % fused_stage_size != 0:
                raise ValueError(
                    "SiHC requires depth to be divisible by fused_stage_size"
                )
            if (
                allowed_stage_sizes is not None
                and fused_stage_size not in allowed_stage_sizes
            ):
                raise ValueError(
                    f"SiHC fused_stage_size must be one of {allowed_stage_sizes}"
                )
            stage_sizes_tuple = (int(fused_stage_size),) * (depth // fused_stage_size)
        else:
            stage_sizes_tuple = tuple(int(size) for size in stage_sizes)
            if not stage_sizes_tuple or any(size <= 0 for size in stage_sizes_tuple):
                raise ValueError("SiHC stage_sizes must contain positive integers")
            if allowed_stage_sizes is not None and any(
                size not in allowed_stage_sizes for size in stage_sizes_tuple
            ):
                raise ValueError(
                    f"SiHC stage_sizes must contain only {allowed_stage_sizes}"
                )
            if sum(stage_sizes_tuple) != depth:
                raise ValueError(
                    f"SiHC stage_sizes must sum to depth {depth}, got {stage_sizes_tuple}"
                )
        if connection_groups not in (None, hidden_size):
            raise ValueError(
                "SiHC uses per-channel connections; "
                "connection_groups must be None or hidden_size"
            )
        repa_z_dims_tuple = _normalize_repa_z_dims(repa_z_dims)
        if repa_depth is not None and not (1 <= repa_depth <= depth):
            raise ValueError(f"repa_depth must be in [1, {depth}], got {repa_depth}")
        if repa_z_dims_tuple and repa_depth is None:
            raise ValueError("repa_depth must be set when repa_z_dims is non-empty")
        if repa_source not in {"dz", "z_plus_dz"}:
            raise ValueError(f"repa_source must be 'dz' or 'z_plus_dz', got {repa_source!r}")
        self.input_size = input_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.hidden_size = hidden_size
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = float(mlp_ratio)
        self.attn_drop = float(attn_drop)
        self.proj_drop = float(proj_drop)
        self.num_classes = num_classes
        self.workspace_grid = workspace_grid
        self.high_grid = input_size // patch_size
        self.num_high_tokens = self.high_grid * self.high_grid
        self.num_workspace_tokens = workspace_grid * workspace_grid
        self.connection_groups = hidden_size
        self.stage_sizes = stage_sizes_tuple
        self.fused_stage_size = (
            stage_sizes_tuple[0]
            if all(size == stage_sizes_tuple[0] for size in stage_sizes_tuple)
            else None
        )
        self.repa_depth = repa_depth
        self.repa_source = repa_source
        self.repa_z_dims = repa_z_dims_tuple
        self.repa_projector_dim = int(repa_projector_dim)

        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = LabelEmbedder(num_classes, hidden_size)
        if patch_embed_type == "direct":
            x_embedder_cls = self.x_embedder_cls
        elif patch_embed_type == "factorized_linear":
            x_embedder_cls = FactorizedPatchEmbedNHWC
        else:
            raise ValueError(
                "patch_embed_type must be 'direct' or 'factorized_linear', "
                f"got {patch_embed_type!r}"
            )
        self.patch_embed_type = patch_embed_type
        self.bottleneck_dim = int(bottleneck_dim)
        self.x_embedder = x_embedder_cls(
            input_size, patch_size, in_channels, bottleneck_dim, hidden_size
        )
        half_head_dim = hidden_size // num_heads // 2
        self.workspace_rope = VisionRotaryEmbeddingFast(half_head_dim, workspace_grid, num_prefix_tokens=0)
        self.blocks = nn.ModuleList(
            [
                self.block_cls(
                    hidden_size=hidden_size,
                    num_heads=num_heads,
                    high_grid=self.high_grid,
                    workspace_grid=workspace_grid,
                    mlp_ratio=mlp_ratio,
                    attn_drop=attn_drop if (depth // 4 * 3 > i >= depth // 4) else 0.0,
                    proj_drop=proj_drop if (depth // 4 * 3 > i >= depth // 4) else 0.0,
                    attn_backend=attn_backend,
                )
                for i in range(depth)
            ]
        )
        self.final_layer = self.final_cls(hidden_size, patch_size, self.out_channels)
        self.repa_projectors = nn.ModuleList(
            [_build_repa_projector(hidden_size, self.repa_projector_dim, z_dim) for z_dim in self.repa_z_dims]
        )
        self.initialize_weights()

    def initialize_weights(self) -> None:
        def basic_init(module: nn.Module) -> None:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(basic_init)
        if hasattr(self.x_embedder, "proj"):
            nn.init.xavier_uniform_(self.x_embedder.proj.weight.data.view(self.x_embedder.proj.weight.shape[0], -1))
            if self.x_embedder.proj.bias is not None:
                nn.init.constant_(self.x_embedder.proj.bias, 0)
        else:
            nn.init.xavier_uniform_(self.x_embedder.proj1.weight.data.view(self.x_embedder.proj1.weight.shape[0], -1))
            nn.init.xavier_uniform_(self.x_embedder.proj2.weight.data.view(self.x_embedder.proj2.weight.shape[0], -1))
            nn.init.constant_(self.x_embedder.proj2.bias, 0)
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        self._init_adaln()
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def _init_adaln(self) -> None:
        for block in self.blocks:
            if hasattr(block.branch, "adaLN_modulation"):
                nn.init.constant_(block.branch.adaLN_modulation[-1].weight, 0)
                nn.init.constant_(block.branch.adaLN_modulation[-1].bias, 0)
        if hasattr(self.final_layer, "adaLN_modulation"):
            nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        c = self.out_channels
        p = self.patch_size
        h = w = int(x.shape[1] ** 0.5)
        if h * w != x.shape[1]:
            raise ValueError(f"token count {x.shape[1]} is not square")
        x = x.reshape(x.shape[0], h, w, p, p, c)
        x = x.permute(0, 5, 1, 3, 2, 4)
        return x.reshape(x.shape[0], c, h * p, w * p)

    def _alpha_stack(self, blocks: list[nn.Module], dtype: torch.dtype) -> torch.Tensor:
        return torch.stack([block.connection.alpha(dtype=dtype) for block in blocks], dim=0).contiguous()

    def _beta_stack(self, blocks: list[nn.Module], dtype: torch.dtype) -> torch.Tensor:
        return torch.stack([block.connection.beta(dtype=dtype) for block in blocks], dim=0).contiguous()

    def _final_accumulate(self, x0: torch.Tensor, dz_list: list[torch.Tensor], blocks: list[nn.Module]) -> torch.Tensor:
        # Keep these casts distinct from the gamma beta stack: merging the two
        # paths would sum their BF16 gradients before promotion to FP32 parameters.
        betas = [block.connection.beta(dtype=dz_list[0].dtype).contiguous() for block in blocks]
        if self.kernel_backend == "tuple":
            return blockwise_write(x0, dz_list, torch.stack(betas).contiguous())
        if self.patch_size == 8:
            if len(dz_list) == 12:
                return triton_final_accumulate_p8_12_traceable(
                    x0, *dz_list, *betas
                )
            raise ValueError(
                "SiHC p8 fused final accumulation requires 12 dz tensors, "
                f"got {len(dz_list)}"
            )
        if len(dz_list) == 12:
            return triton_final_accumulate_b4_12_traceable(x0, *dz_list, *betas)
        if len(dz_list) == 8:
            return triton_final_accumulate_b4_8_traceable(x0, *dz_list, *betas)
        raise ValueError(f"SiHC fused final accumulation requires 8 or 12 dz tensors, got {len(dz_list)}")

    def _dense_output(self, x_hi: torch.Tensor, final_cond: torch.Tensor) -> torch.Tensor:
        b, hh, wh, c_dim = x_hi.shape
        x_out = self.final_layer(x_hi.reshape(b, hh * wh, c_dim), final_cond)
        return self.unpatchify(x_out)

    def _project_repa_tokens(self, z: torch.Tensor) -> tuple[torch.Tensor, ...]:
        b, n, c_dim = z.shape
        flat = z.reshape(b * n, c_dim)
        return tuple(projector(flat).reshape(b, n, -1) for projector in self.repa_projectors)

    def _run_fused_schedule_stage(
        self,
        x0: torch.Tensor,
        cond: torch.Tensor,
        blocks: list[nn.Module],
        stage_start: int,
        repa_outputs: list[torch.Tensor] | None = None,
    ) -> torch.Tensor:
        stage_size = len(blocks)
        alpha_stack = self._alpha_stack(blocks, dtype=x0.dtype)
        if self.kernel_backend == "tuple":
            base_reads = blockwise_read(x0, alpha_stack)
            active_ops = None
        elif self.patch_size == 8 and stage_size == 12:
            base_reads = triton_all_base_p8_12_split_traceable(x0, alpha_stack)
            active_ops = triton_triangular_active_ops_12
        elif stage_size == 12:
            base_reads = triton_all_base_b4_12_split_traceable(x0, alpha_stack)
            active_ops = triton_triangular_active_ops_12
        elif stage_size == 8:
            base_reads = triton_all_base_b4_8_split_traceable(x0, alpha_stack)
            active_ops = triton_triangular_active_ops_8
        else:
            raise ValueError(f"SiHC fused schedule stage requires 8 or 12 blocks, got {stage_size}")
        beta_stack = self._beta_stack(blocks, dtype=x0.dtype)
        gamma = (alpha_stack[:, None, :, :] * beta_stack[None, :, :, :]).sum(dim=-1)
        zero_dz = torch.zeros_like(base_reads[0]) if active_ops is not None else None

        dz_list: list[torch.Tensor] = []
        for layer_idx, block in enumerate(blocks):
            z = base_reads[layer_idx]
            if dz_list:
                if self.kernel_backend == "tuple":
                    z = blockwise_triangular(z, dz_list, gamma[layer_idx, :layer_idx].contiguous())
                else:
                    dz_args = tuple(dz_list[j] if j < layer_idx else zero_dz for j in range(stage_size))
                    z = active_ops[layer_idx - 1](z, *dz_args, gamma[layer_idx].contiguous())
            dz = self._branch_update(block, z, cond)
            layer_number = stage_start + layer_idx + 1
            if repa_outputs is not None and self.repa_projectors and layer_number == self.repa_depth:
                repa_source = dz if self.repa_source == "dz" else z + dz
                repa_outputs.extend(self._project_repa_tokens(repa_source))
            dz_list.append(dz)
        return self._final_accumulate(x0, dz_list, blocks)

    def _run_fused_schedule(
        self, x0: torch.Tensor, cond: torch.Tensor, return_repa: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        x_hi = x0
        blocks = list(self.blocks)
        repa_outputs: list[torch.Tensor] | None = [] if return_repa else None
        start = 0
        for stage_size in self.stage_sizes:
            x_hi = self._run_fused_schedule_stage(
                x_hi, cond, blocks[start : start + stage_size], start, repa_outputs
            )
            start += stage_size
        if return_repa:
            return x_hi, tuple(repa_outputs or ())
        return x_hi

    def _branch_update(self, block: nn.Module, z: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class SiHCModel(_SiHCBase):
    """Dense SiHC with standard per-block AdaLN modulation."""

    block_cls = SiHCBlock
    final_cls = FinalLayer

    def _branch_update(self, block: nn.Module, z: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        return block.branch_update(z, cond, self.workspace_rope)

    def forward(
        self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor, return_repa: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        c = self.t_embedder(t) + self.y_embedder(y)
        x0 = self.x_embedder(x)
        schedule_out = self._run_fused_schedule(x0, c, return_repa=return_repa)
        if return_repa:
            x_hi, repa_outputs = schedule_out
            return self._dense_output(x_hi, c), repa_outputs
        return self._dense_output(schedule_out, c)


class FusedP8SiHCModel(SiHCModel):
    """Block-wise SiHC with four spatially indexed slots and fused stage boundaries."""

    uses_fused_b4_kernels = False
    uses_fused_p8_kernels = True
    allowed_stage_sizes = (12,)


class SiHCWriteOnlyModel(SiHCModel):
    """SiHC ablation with fixed stage reads and learned slot-wise writes only."""

    block_cls = SiHCWriteOnlyBlock
    supports_tuple_backend = False

    def _run_fused_schedule_stage(
        self,
        x0: torch.Tensor,
        cond: torch.Tensor,
        blocks: list[nn.Module],
        stage_start: int,
        repa_outputs: list[torch.Tensor] | None = None,
    ) -> torch.Tensor:
        if len(blocks) not in (8, 12):
            raise ValueError(
                f"write-only SiHC stage requires 8 or 12 blocks, got {len(blocks)}"
            )

        z = triton_cell_mean_b4_traceable(x0)
        dz_list: list[torch.Tensor] = []
        for layer_idx, block in enumerate(blocks):
            dz = self._branch_update(block, z, cond)
            z_next = z + dz
            layer_number = stage_start + layer_idx + 1
            if (
                repa_outputs is not None
                and self.repa_projectors
                and layer_number == self.repa_depth
            ):
                repa_source = dz if self.repa_source == "dz" else z_next
                repa_outputs.extend(self._project_repa_tokens(repa_source))
            dz_list.append(dz)
            z = z_next

        return self._final_accumulate(x0, dz_list, blocks)


class SiHCInContextModel(SiHCModel):
    """Standard-AdaLN SiHC with persistent JiT class-prefix tokens."""

    def __init__(
        self,
        *args,
        in_context_len: int = 32,
        in_context_start: int = 4,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        if in_context_len <= 0:
            raise ValueError(f"in_context_len must be positive, got {in_context_len}")
        if not (0 <= in_context_start < self.depth):
            raise ValueError(f"in_context_start must be in [0, {self.depth}), got {in_context_start}")
        self.in_context_len = int(in_context_len)
        self.in_context_start = int(in_context_start)
        self.in_context_posemb = nn.Parameter(torch.empty(1, self.in_context_len, self.hidden_size))
        nn.init.normal_(self.in_context_posemb, std=0.02)
        half_head_dim = self.hidden_size // self.num_heads // 2
        self.workspace_rope_incontext = VisionRotaryEmbeddingFast(
            half_head_dim,
            self.workspace_grid,
            num_prefix_tokens=self.in_context_len,
        )

    def _run_fused_schedule_stage_incontext(
        self,
        x0: torch.Tensor,
        cond: torch.Tensor,
        context_tokens: torch.Tensor,
        blocks: list[nn.Module],
        stage_start: int,
        repa_outputs: list[torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        stage_size = len(blocks)
        alpha_stack = self._alpha_stack(blocks, dtype=x0.dtype)
        if self.kernel_backend == "tuple":
            base_reads = blockwise_read(x0, alpha_stack)
            active_ops = None
        elif stage_size == 12:
            base_reads = triton_all_base_b4_12_split_traceable(x0, alpha_stack)
            active_ops = triton_triangular_active_ops_12
        elif stage_size == 8:
            base_reads = triton_all_base_b4_8_split_traceable(x0, alpha_stack)
            active_ops = triton_triangular_active_ops_8
        else:
            raise ValueError(f"SiHC fused schedule stage requires 8 or 12 blocks, got {stage_size}")
        beta_stack = self._beta_stack(blocks, dtype=x0.dtype)
        gamma = (alpha_stack[:, None, :, :] * beta_stack[None, :, :, :]).sum(dim=-1)
        zero_dz = torch.zeros_like(base_reads[0]) if active_ops is not None else None

        dz_list: list[torch.Tensor] = []
        for layer_idx, block in enumerate(blocks):
            z = base_reads[layer_idx]
            if dz_list:
                if self.kernel_backend == "tuple":
                    z = blockwise_triangular(z, dz_list, gamma[layer_idx, :layer_idx].contiguous())
                else:
                    dz_args = tuple(dz_list[j] if j < layer_idx else zero_dz for j in range(stage_size))
                    z = active_ops[layer_idx - 1](z, *dz_args, gamma[layer_idx].contiguous())

            block_index = stage_start + layer_idx
            if block_index >= self.in_context_start:
                seq = torch.cat((context_tokens, z), dim=1)
                dseq = block.branch_update(seq, cond, self.workspace_rope_incontext)
                context_tokens = context_tokens + dseq[:, : self.in_context_len]
                dz = dseq[:, self.in_context_len :].contiguous()
            else:
                dz = block.branch_update(z, cond, self.workspace_rope)

            layer_number = block_index + 1
            if repa_outputs is not None and self.repa_projectors and layer_number == self.repa_depth:
                repa_source = dz if self.repa_source == "dz" else z + dz
                repa_outputs.extend(self._project_repa_tokens(repa_source))
            dz_list.append(dz)

        return self._final_accumulate(x0, dz_list, blocks), context_tokens

    def _run_fused_schedule_incontext(
        self,
        x0: torch.Tensor,
        cond: torch.Tensor,
        context_tokens: torch.Tensor,
        return_repa: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        x_hi = x0
        blocks = list(self.blocks)
        repa_outputs: list[torch.Tensor] | None = [] if return_repa else None
        start = 0
        for stage_size in self.stage_sizes:
            x_hi, context_tokens = self._run_fused_schedule_stage_incontext(
                x_hi,
                cond,
                context_tokens,
                blocks[start : start + stage_size],
                start,
                repa_outputs,
            )
            start += stage_size
        if return_repa:
            return x_hi, tuple(repa_outputs or ())
        return x_hi

    def forward(
        self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor, return_repa: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        t_emb = self.t_embedder(t)
        y_emb = self.y_embedder(y)
        c = t_emb + y_emb
        x0 = self.x_embedder(x)
        context_tokens = (
            y_emb.unsqueeze(1).expand(-1, self.in_context_len, -1) + self.in_context_posemb
        ).to(dtype=x0.dtype)
        schedule_out = self._run_fused_schedule_incontext(x0, c, context_tokens, return_repa=return_repa)
        if return_repa:
            x_hi, repa_outputs = schedule_out
            return self._dense_output(x_hi, c), repa_outputs
        return self._dense_output(schedule_out, c)


class DirectSiHCModel(SiHCModel):
    """Backward-compatible explicit name for the now-default direct stem."""


def SiHC_1x12_D768_B4(**kwargs) -> SiHCModel:
    kwargs.setdefault('patch_size', 4)
    kwargs.setdefault('hidden_size', 768)
    kwargs.setdefault('depth', 12)
    kwargs.setdefault('num_heads', 12)
    kwargs.setdefault('workspace_grid', 16)
    kwargs.setdefault('fused_stage_size', 12)
    return SiHCModel(**kwargs)


def SiHC_1x12_D768_B8(**kwargs) -> FusedP8SiHCModel:
    """JiT-B-sized p8 model with four fused spatial residual slots."""
    kwargs.setdefault('patch_size', 8)
    kwargs.setdefault('hidden_size', 768)
    kwargs.setdefault('depth', 12)
    kwargs.setdefault('num_heads', 12)
    kwargs.setdefault('workspace_grid', 16)
    kwargs.setdefault('fused_stage_size', 12)
    return FusedP8SiHCModel(**kwargs)


def SiHC_2x12_D768_B4(**kwargs) -> SiHCModel:
    """D=768 B/4 with 24 blocks executed as two 12-layer stages."""
    kwargs.setdefault('patch_size', 4)
    kwargs.setdefault('hidden_size', 768)
    kwargs.setdefault('depth', 24)
    kwargs.setdefault('num_heads', 12)
    kwargs.setdefault('workspace_grid', 16)
    kwargs.setdefault('fused_stage_size', 12)
    return SiHCModel(**kwargs)


def SiHC_Direct_1x12_D768_B4(**kwargs) -> DirectSiHCModel:
    kwargs.setdefault('patch_size', 4)
    kwargs.setdefault('hidden_size', 768)
    kwargs.setdefault('depth', 12)
    kwargs.setdefault('num_heads', 12)
    kwargs.setdefault('workspace_grid', 16)
    kwargs.setdefault('fused_stage_size', 12)
    return DirectSiHCModel(**kwargs)


def SiHC_2x12_D1024_B4(**kwargs) -> SiHCModel:
    """D=1024 B/4 with 24 blocks executed as two 12-layer stages."""
    kwargs.setdefault('patch_size', 4)
    kwargs.setdefault('hidden_size', 1024)
    kwargs.setdefault('depth', 24)
    kwargs.setdefault('num_heads', 16)
    kwargs.setdefault('workspace_grid', 16)
    kwargs.setdefault('fused_stage_size', 12)
    return SiHCModel(**kwargs)


def SiHC_3x8_D768_B4(**kwargs) -> SiHCModel:
    """D=768 B/4 with 24 blocks executed as three 8-layer stages."""
    kwargs.setdefault('patch_size', 4)
    kwargs.setdefault('hidden_size', 768)
    kwargs.setdefault('depth', 24)
    kwargs.setdefault('num_heads', 12)
    kwargs.setdefault('workspace_grid', 16)
    kwargs.setdefault('fused_stage_size', 8)
    return SiHCModel(**kwargs)


def SiHC_3x8_D1024_B4(**kwargs) -> SiHCModel:
    """D=1024 B/4 with 24 blocks executed as three 8-layer stages."""
    kwargs.setdefault('patch_size', 4)
    kwargs.setdefault('hidden_size', 1024)
    kwargs.setdefault('depth', 24)
    kwargs.setdefault('num_heads', 16)
    kwargs.setdefault('workspace_grid', 16)
    kwargs.setdefault('fused_stage_size', 8)
    return SiHCModel(**kwargs)


def SiHC_5x8_D768_B4(**kwargs) -> SiHCModel:
    """D=768 B/4 with 40 blocks executed as five 8-layer stages."""
    kwargs.setdefault('patch_size', 4)
    kwargs.setdefault('hidden_size', 768)
    kwargs.setdefault('depth', 40)
    kwargs.setdefault('num_heads', 12)
    kwargs.setdefault('workspace_grid', 16)
    kwargs.setdefault('fused_stage_size', 8)
    return SiHCModel(**kwargs)


def SiHC_5x8_D1024_B4(**kwargs) -> SiHCModel:
    """D=1024 B/4 with 40 blocks executed as five 8-layer stages."""
    kwargs.setdefault('patch_size', 4)
    kwargs.setdefault('hidden_size', 1024)
    kwargs.setdefault('depth', 40)
    kwargs.setdefault('num_heads', 16)
    kwargs.setdefault('workspace_grid', 16)
    kwargs.setdefault('fused_stage_size', 8)
    return SiHCModel(**kwargs)


def SiHC_8x12x8_D1152_B4(**kwargs) -> SiHCModel:
    """SiT-XL-scale model with REPA-friendly mixed fused stages."""
    kwargs.setdefault('patch_size', 4)
    kwargs.setdefault('hidden_size', 1152)
    kwargs.setdefault('depth', 28)
    kwargs.setdefault('num_heads', 16)
    kwargs.setdefault('workspace_grid', 16)
    kwargs.setdefault('stage_sizes', (8, 12, 8))
    return SiHCModel(**kwargs)


def SiHCWriteOnly_1x12_D768_B4(**kwargs) -> SiHCWriteOnlyModel:
    """Small write-only preset for correctness and throughput comparisons."""
    kwargs.setdefault('patch_size', 4)
    kwargs.setdefault('hidden_size', 768)
    kwargs.setdefault('depth', 12)
    kwargs.setdefault('num_heads', 12)
    kwargs.setdefault('workspace_grid', 16)
    kwargs.setdefault('fused_stage_size', 12)
    return SiHCWriteOnlyModel(**kwargs)


def SiHCWriteOnly_8x12x8_D1152_B4(**kwargs) -> SiHCWriteOnlyModel:
    """SiT-XL-scale write-only ablation with the same 8+12+8 boundaries."""
    kwargs.setdefault('patch_size', 4)
    kwargs.setdefault('hidden_size', 1152)
    kwargs.setdefault('depth', 28)
    kwargs.setdefault('num_heads', 16)
    kwargs.setdefault('workspace_grid', 16)
    kwargs.setdefault('stage_sizes', (8, 12, 8))
    return SiHCWriteOnlyModel(**kwargs)


def SiHC_5x8_D1024_B4_CTX32S4(
    **kwargs,
) -> SiHCInContextModel:
    """D=1024, depth-40 model with 32 class-prefix tokens entering block five."""
    kwargs.setdefault('patch_size', 4)
    kwargs.setdefault('hidden_size', 1024)
    kwargs.setdefault('depth', 40)
    kwargs.setdefault('num_heads', 16)
    kwargs.setdefault('workspace_grid', 16)
    kwargs.setdefault('fused_stage_size', 8)
    kwargs.setdefault('in_context_len', 32)
    kwargs.setdefault('in_context_start', 4)
    return SiHCInContextModel(**kwargs)


SIHC_MODELS = {
    "sihc_1x12_d768_b4": SiHC_1x12_D768_B4,
    "sihc_1x12_d768_b8": SiHC_1x12_D768_B8,
    "sihc_2x12_d768_b4": SiHC_2x12_D768_B4,
    "sihc_2x12_d1024_b4": SiHC_2x12_D1024_B4,
    "sihc_3x8_d768_b4": SiHC_3x8_D768_B4,
    "sihc_3x8_d1024_b4": SiHC_3x8_D1024_B4,
    "sihc_5x8_d768_b4": SiHC_5x8_D768_B4,
    "sihc_5x8_d1024_b4": SiHC_5x8_D1024_B4,
    "sihc_8x12x8_d1152_b4": SiHC_8x12x8_D1152_B4,
    "sihc_write_only_1x12_d768_b4": SiHCWriteOnly_1x12_D768_B4,
    "sihc_write_only_8x12x8_d1152_b4": SiHCWriteOnly_8x12x8_D1152_B4,
    "sihc_5x8_d1024_b4_ctx32s4": SiHC_5x8_D1024_B4_CTX32S4,
    "sihc_direct_1x12_d768_b4": SiHC_Direct_1x12_D768_B4,
}

