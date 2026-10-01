"""Checkpoint loading helpers for current public parameter names."""

from __future__ import annotations

from collections.abc import Mapping

import torch


def checkpoint_state(state: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Return checkpoint tensors without legacy read/write key migration."""
    return dict(state)


def load_model_state(model: torch.nn.Module, ckpt: Mapping, key: str = "ema") -> None:
    """Load model or EMA weights using the repo's current parameter names.

    EMA checkpoints store parameters only, while ``model.state_dict()`` also
    contains buffers. Fill buffers from the freshly constructed model, but
    require every trainable parameter to be present in the checkpoint.
    """
    if key == "model":
        model.load_state_dict(checkpoint_state(ckpt["model"]))
        return

    if key not in {"ema", "ema2"}:
        raise ValueError(f"unsupported checkpoint state key: {key}")
    if key not in ckpt:
        raise KeyError(f"checkpoint does not contain {key} weights")

    ema = checkpoint_state(ckpt[key])
    parameter_names = {name for name, _ in model.named_parameters()}
    missing_params = sorted(name for name in parameter_names if name not in ema)
    if missing_params:
        preview = ", ".join(missing_params[:8])
        suffix = "" if len(missing_params) <= 8 else f", ... ({len(missing_params)} total)"
        raise KeyError(f"EMA checkpoint missing parameter(s): {preview}{suffix}")

    state = model.state_dict()
    unexpected = sorted(name for name in ema if name not in state)
    if unexpected:
        preview = ", ".join(unexpected[:8])
        suffix = "" if len(unexpected) <= 8 else f", ... ({len(unexpected)} total)"
        raise KeyError(f"EMA checkpoint has unexpected parameter(s): {preview}{suffix}")

    for name, value in ema.items():
        state[name] = value
    model.load_state_dict(state)


# Backward-compatible function name for training resume code. It intentionally
# no longer performs read/write migration.
def migrate_checkpoint_state(state: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return checkpoint_state(state)
