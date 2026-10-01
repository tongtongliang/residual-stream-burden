"""Unfused research references for SiHC geometry and connection frequency."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from types import MethodType

import torch
import torch.nn as nn

from .components import FinalLayer, WorkspaceJiTBranch
from .model import SiHCModel, SpatiallyIndexedHyperConnection


class _ReferenceScheduleMixin:
    """Explicit PyTorch recurrence used before committing to new Triton kernels."""

    uses_fused_b4_kernels = False
    supports_tuple_backend = False
    allowed_stage_sizes = None

    def _high_to_cells(self, x_hi: torch.Tensor) -> torch.Tensor:
        batch, height, width, channels = x_hi.shape
        grid = self.workspace_grid
        if height != self.high_grid or width != self.high_grid:
            raise ValueError(
                f"expected high-resolution grid {self.high_grid}x{self.high_grid}, "
                f"got {height}x{width}"
            )
        cell_size = self.high_grid // grid
        return (
            x_hi.reshape(batch, grid, cell_size, grid, cell_size, channels)
            .permute(0, 1, 3, 2, 4, 5)
            .reshape(batch, grid * grid, cell_size * cell_size, channels)
        )

    def _cells_to_high(self, cells: torch.Tensor) -> torch.Tensor:
        batch, tokens, slots, channels = cells.shape
        grid = self.workspace_grid
        cell_size = self.high_grid // grid
        expected_slots = cell_size * cell_size
        if tokens != grid * grid or slots != expected_slots:
            raise ValueError(
                f"expected cells [B,{grid * grid},{expected_slots},C], "
                f"got {tuple(cells.shape)}"
            )
        return (
            cells.reshape(batch, grid, grid, cell_size, cell_size, channels)
            .permute(0, 1, 3, 2, 4, 5)
            .reshape(batch, self.high_grid, self.high_grid, channels)
        )

    @staticmethod
    def _route_update(
        base_read: torch.Tensor,
        previous_updates: list[torch.Tensor],
        gamma_row: torch.Tensor,
    ) -> torch.Tensor:
        if not previous_updates:
            return base_read
        stacked = torch.stack(previous_updates, dim=0)
        return base_read + torch.einsum("jbnc,jc->bnc", stacked, gamma_row)

    def _reference_stage(
        self,
        x_hi: torch.Tensor,
        alpha_stack: torch.Tensor,
        beta_stack: torch.Tensor,
        update_functions: Sequence[Callable[[torch.Tensor], torch.Tensor]],
    ) -> tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor]]:
        cells = self._high_to_cells(x_hi)
        base_reads = torch.einsum("bnsc,lcs->lbnc", cells, alpha_stack)
        gamma = torch.einsum("lcs,jcs->ljc", alpha_stack, beta_stack)
        reads: list[torch.Tensor] = []
        updates: list[torch.Tensor] = []
        for index, update_function in enumerate(update_functions):
            read = self._route_update(
                base_reads[index], updates, gamma[index, :index]
            )
            reads.append(read)
            updates.append(update_function(read))
        update_stack = torch.stack(updates, dim=0)
        cells = cells + torch.einsum("lbnc,lcs->bnsc", update_stack, beta_stack)
        return self._cells_to_high(cells), reads, updates


class ReferenceSiHCModel(_ReferenceScheduleMixin, SiHCModel):
    """Block-wise SiHC expressed without fixed B/4 Triton kernels."""

    def _run_fused_schedule_stage(
        self,
        x0: torch.Tensor,
        cond: torch.Tensor,
        blocks: list[nn.Module],
        stage_start: int,
        repa_outputs: list[torch.Tensor] | None = None,
    ) -> torch.Tensor:
        alpha_stack = self._alpha_stack(blocks, dtype=x0.dtype)
        beta_stack = self._beta_stack(blocks, dtype=x0.dtype)
        update_functions = [
            lambda z, block=block: block.branch_update(z, cond, self.workspace_rope)
            for block in blocks
        ]
        x_hi, reads, updates = self._reference_stage(
            x0, alpha_stack, beta_stack, update_functions
        )
        if repa_outputs is not None and self.repa_projectors:
            local_index = self.repa_depth - stage_start - 1
            if 0 <= local_index < len(blocks):
                source = (
                    updates[local_index]
                    if self.repa_source == "dz"
                    else reads[local_index] + updates[local_index]
                )
                repa_outputs.extend(self._project_repa_tokens(source))
        return x_hi


class SiHCSublayerReferenceBlock(nn.Module):
    """One JiT block with independent attention and MLP connections."""

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
        self.attention_connection = SpatiallyIndexedHyperConnection(
            high_grid, workspace_grid, hidden_size
        )
        self.mlp_connection = SpatiallyIndexedHyperConnection(
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


class SublayerReferenceSiHCModel(_ReferenceScheduleMixin, SiHCModel):
    """SiHC with separate residual connections around attention and MLP."""

    block_cls = SiHCSublayerReferenceBlock
    final_cls = FinalLayer

    def _run_fused_schedule_stage(
        self,
        x0: torch.Tensor,
        cond: torch.Tensor,
        blocks: list[nn.Module],
        stage_start: int,
        repa_outputs: list[torch.Tensor] | None = None,
    ) -> torch.Tensor:
        if repa_outputs is not None and self.repa_projectors:
            raise NotImplementedError(
                "REPA semantics are intentionally undefined for the "
                "sublayer reference"
            )
        connections = []
        update_functions = []
        for block in blocks:
            modulation = block.branch.conditioning(cond)
            connections.extend(
                (block.attention_connection, block.mlp_connection)
            )
            update_functions.extend(
                (
                    lambda z, block=block, modulation=modulation: (
                        block.branch.attention_update(
                            z, modulation, self.workspace_rope
                        )
                    ),
                    lambda z, block=block, modulation=modulation: (
                        block.branch.mlp_update(z, modulation)
                    ),
                )
            )
        alpha_stack = torch.stack(
            [connection.alpha(dtype=x0.dtype) for connection in connections],
            dim=0,
        ).contiguous()
        beta_stack = torch.stack(
            [connection.beta(dtype=x0.dtype) for connection in connections],
            dim=0,
        ).contiguous()
        x_hi, _, _ = self._reference_stage(
            x0, alpha_stack, beta_stack, update_functions
        )
        return x_hi


def SiHCReference_1x12_D768_B8(**kwargs) -> ReferenceSiHCModel:
    """JiT-B-sized p8 model with four spatially indexed residual slots."""
    kwargs.setdefault('patch_size', 8)
    kwargs.setdefault('hidden_size', 768)
    kwargs.setdefault('depth', 12)
    kwargs.setdefault('num_heads', 12)
    kwargs.setdefault('workspace_grid', 16)
    kwargs.setdefault('fused_stage_size', 12)
    return ReferenceSiHCModel(**kwargs)


def SiHCSublayerReference_1x12_D768_B8(
    **kwargs,
) -> SublayerReferenceSiHCModel:
    """JiT-B-sized p8 model with 24 residual-sublayer connection events."""
    kwargs.setdefault('patch_size', 8)
    kwargs.setdefault('hidden_size', 768)
    kwargs.setdefault('depth', 12)
    kwargs.setdefault('num_heads', 12)
    kwargs.setdefault('workspace_grid', 16)
    kwargs.setdefault('fused_stage_size', 12)
    return SublayerReferenceSiHCModel(**kwargs)


REFERENCE_SIHC_MODELS = {
    "sihc_reference_1x12_d768_b8": SiHCReference_1x12_D768_B8,
    "sihc_sublayer_reference_1x12_d768_b8": (
        SiHCSublayerReference_1x12_D768_B8
    ),
}


def enable_reference_schedule(model: SiHCModel) -> SiHCModel:
    """Replace the fused stage recurrence with its explicit PyTorch reference."""
    from .model import SiHCInContextModel, SiHCWriteOnlyModel

    if not isinstance(model, SiHCModel):
        raise TypeError("the reference schedule is only available for SiHC models")
    if isinstance(model, (SiHCInContextModel, SiHCWriteOnlyModel)):
        raise TypeError(
            f"the reference schedule is not defined for {type(model).__name__}"
        )
    model._high_to_cells = MethodType(
        _ReferenceScheduleMixin._high_to_cells, model
    )
    model._cells_to_high = MethodType(
        _ReferenceScheduleMixin._cells_to_high, model
    )
    model._route_update = _ReferenceScheduleMixin._route_update
    model._reference_stage = MethodType(
        _ReferenceScheduleMixin._reference_stage, model
    )
    stage = (
        SublayerReferenceSiHCModel._run_fused_schedule_stage
        if isinstance(model.blocks[0], SiHCSublayerReferenceBlock)
        else ReferenceSiHCModel._run_fused_schedule_stage
    )
    model._run_fused_schedule_stage = MethodType(stage, model)
    return model
