"""Block-wise adapters for the shared tuple-pointer SiHC fusion engine.

These operators change only the routing implementation. Each event remains
an attention/MLP block, with the same learned connection parameters.
Custom-op, fake-tensor and autograd registrations live in the shared module.
"""

from __future__ import annotations

from torch import Tensor

from .sublayer_kernels import sublayer_read, sublayer_triangular, sublayer_write


def _require_block_events(weights: Tensor) -> None:
    if weights.ndim != 3 or weights.shape[0] not in (8, 12):
        raise ValueError("block-wise weights must have shape [8 or 12, channels, slots]")


def blockwise_read(x: Tensor, alpha: Tensor) -> list[Tensor]:
    """Read one separate workspace tensor per block in an 8/12-block stage."""
    _require_block_events(alpha)
    return sublayer_read(x, alpha)


def blockwise_write(x: Tensor, updates: list[Tensor], beta: Tensor) -> Tensor:
    """Accumulate the stage's 8/12 workspace writes and identity carry."""
    _require_block_events(beta)
    return sublayer_write(x, updates, beta)


def blockwise_triangular(
    base: Tensor, updates: list[Tensor], gamma: Tensor
) -> Tensor:
    """Accumulate an active prefix without padded tensors or unused edges."""
    if not 1 <= len(updates) <= 11:
        raise ValueError("block-wise triangular accumulation requires 1..11 active updates")
    return sublayer_triangular(base, updates, gamma)


__all__ = ["blockwise_read", "blockwise_write", "blockwise_triangular"]
