"""Activation-checkpoint helpers that preserve SiHC checkpoint keys."""

from __future__ import annotations

import torch.nn as nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl,
    checkpoint_wrapper,
)


ACTIVATION_CHECKPOINT_MODES = ("none", "mlp", "branch")


def configure_activation_checkpointing(model: nn.Module, mode: str) -> int:
    """Wrap SiHC submodules before ``torch.compile``.

    ``mlp`` recomputes only each workspace SwiGLU. ``branch`` recomputes the
    complete workspace Transformer branch while deliberately leaving the
    SiHC Triton read/accumulate/write schedule outside the checkpointed
    region. The non-reentrant implementation composes with DDP and compile.

    Returns the number of wrapped modules. Parameter objects and ``state_dict``
    keys are preserved by PyTorch's checkpoint wrapper.
    """

    if mode not in ACTIVATION_CHECKPOINT_MODES:
        raise ValueError(
            f"activation checkpoint mode must be one of {ACTIVATION_CHECKPOINT_MODES}, got {mode!r}"
        )
    if mode == "none":
        return 0
    if mode == "branch" and getattr(model, "connection_frequency", None) == "sublayer":
        raise ValueError(
            "sublayer SiHC calls attention/MLP separately; use 'mlp' checkpointing, "
            "not the full-block 'branch' wrapper"
        )

    blocks = getattr(model, "blocks", None)
    if blocks is None:
        raise ValueError(
            f"activation checkpoint mode {mode!r} requires a model with workspace blocks"
        )

    wrapped = 0
    for block_index, block in enumerate(blocks):
        branch = getattr(block, "branch", None)
        if branch is None:
            raise ValueError(f"block {block_index} has no checkpointable branch module")
        if mode == "mlp":
            mlp = getattr(branch, "mlp", None)
            if mlp is None:
                raise ValueError(f"block {block_index} branch has no checkpointable MLP module")
            branch.mlp = checkpoint_wrapper(
                mlp,
                checkpoint_impl=CheckpointImpl.NO_REENTRANT,
            )
        else:
            block.branch = checkpoint_wrapper(
                branch,
                checkpoint_impl=CheckpointImpl.NO_REENTRANT,
            )
        wrapped += 1

    return wrapped
