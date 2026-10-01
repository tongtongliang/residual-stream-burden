"""Small shared helpers for optional REPA projection heads."""

import torch.nn as nn


def build_repa_projector(hidden_size: int, projector_dim: int, z_dim: int) -> nn.Module:
    return nn.Sequential(
        nn.Linear(hidden_size, projector_dim),
        nn.SiLU(),
        nn.Linear(projector_dim, projector_dim),
        nn.SiLU(),
        nn.Linear(projector_dim, z_dim),
    )


def normalize_repa_z_dims(repa_z_dims) -> tuple[int, ...]:
    if repa_z_dims is None:
        return ()
    return tuple(int(dim) for dim in repa_z_dims)
