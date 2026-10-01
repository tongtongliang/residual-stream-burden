"""AdaLN-zero FCN and patch SiHC, with explicit diagnostic taps."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class ModelConfig:
    family: str = "fcn"
    ambient_dim: int = 512
    width: int = 256
    depth: int = 5
    time_embed_dim: int = 256
    streams: int = 4

    def __post_init__(self):
        if self.family not in {"fcn", "sihc"}:
            raise ValueError("family must be fcn or sihc")
        if min(self.ambient_dim, self.width, self.depth, self.streams) < 1:
            raise ValueError("model dimensions must be positive")
        if self.time_embed_dim < 4 or self.time_embed_dim % 2:
            raise ValueError("time_embed_dim must be even and at least 4")
        if self.family == "sihc" and self.ambient_dim % self.streams:
            raise ValueError("ambient_dim must be divisible by streams")


def sinusoidal_embedding(tau: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    frequency = torch.exp(
        torch.arange(half, device=tau.device, dtype=tau.dtype)
        * (-math.log(10000.0) / (half - 1))
    )
    phase = tau[:, None] * frequency[None, :]
    return torch.cat((phase.cos(), phase.sin()), dim=-1)


class Branch(nn.Module):
    """Return an update, not an already-updated residual stream."""

    def __init__(self, width: int):
        super().__init__()
        self.norm = nn.LayerNorm(width, elementwise_affine=False)
        self.ada = nn.Linear(width, 3 * width)
        self.fc0 = nn.Linear(width, width)
        self.fc1 = nn.Linear(width, width)
        nn.init.zeros_(self.ada.weight)
        nn.init.zeros_(self.ada.bias)

    def forward(self, h, condition):
        scale, shift, gate = self.ada(condition).chunk(3, dim=-1)
        fanin = self.norm(h) * (1 + scale) + shift
        update = gate * self.fc1(F.relu(self.fc0(fanin)))
        return update, fanin


class ToyModel(nn.Module):
    """Two alternatives with matching depth, width, and time conditioning.

    ``fcn`` keeps one full-dimensional hidden residual stream. ``sihc`` keeps
    patch-local carriers and reads/writes a shared workspace feature by feature.
    All SiHC routing weights are unconstrained: there is no routing softmax.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        c = config
        self.time = nn.Sequential(
            nn.Linear(c.time_embed_dim, c.width), nn.SiLU(), nn.Linear(c.width, c.width)
        )
        self.branches = nn.ModuleList(Branch(c.width) for _ in range(c.depth))
        if c.family == "fcn":
            self.encode = nn.Linear(c.ambient_dim, c.width)
            self.decode = nn.Linear(c.width, c.ambient_dim)
            # The FCN baseline uses Gaussian Xavier weights and zero biases.
            for module in self.modules():
                if isinstance(module, nn.Linear):
                    nn.init.xavier_normal_(module.weight)
                    nn.init.zeros_(module.bias)
            for branch in self.branches:
                nn.init.zeros_(branch.ada.weight)
            nn.init.zeros_(self.decode.weight)
        else:
            patch = c.ambient_dim // c.streams
            self.encode = nn.ModuleList(nn.Linear(patch, c.width) for _ in range(c.streams))
            self.decode = nn.ModuleList(nn.Linear(c.width, patch) for _ in range(c.streams))
            self.read = nn.Parameter(torch.full((c.depth, c.width, c.streams), 1 / c.streams))
            self.write = nn.Parameter(torch.ones(c.depth, c.width, c.streams))
            for layer in self.decode:
                nn.init.zeros_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, z, tau, *, return_taps=False, schedule="triangular"):
        """Use noise time ``tau``: clean=0, Gaussian=1.

        ``schedule='explicit'`` materializes every carrier update and provides
        a readable correctness reference for the triangular SiHC schedule.
        Taps are attached tensors so callers can inspect their gradients.
        """
        if schedule not in {"triangular", "explicit"}:
            raise ValueError("schedule must be triangular or explicit")
        if z.ndim != 2 or z.shape[1] != self.config.ambient_dim or tau.shape != z.shape[:1]:
            raise ValueError("expected z [batch, ambient_dim], tau [batch]")
        condition = self.time(sinusoidal_embedding(tau, self.config.time_embed_dim))
        taps = {}
        if self.config.family == "fcn":
            h = self.encode(z)
            for index, branch in enumerate(self.branches):
                update, fanin = branch(h, condition)
                if return_taps:
                    taps[f"block{index}.residual"] = h
                    taps[f"block{index}.fanin"] = fanin
                h = h + update
            output = self.decode(h)
        else:
            patches = z.reshape(len(z), self.config.streams, -1)
            carrier = torch.stack([layer(patches[:, i]) for i, layer in enumerate(self.encode)], 1)
            initial = carrier
            base = torch.einsum("bsc,lcs->lbc", initial, self.read)
            gamma = torch.einsum("lcs,jcs->ljc", self.read, self.write)
            updates = []
            for index, branch in enumerate(self.branches):
                if schedule == "explicit":
                    workspace = torch.einsum("bsc,cs->bc", carrier, self.read[index])
                else:
                    workspace = base[index]
                    if updates:
                        workspace = workspace + torch.einsum(
                            "jbc,jc->bc", torch.stack(updates), gamma[index, :index]
                        )
                update, fanin = branch(workspace, condition)
                if return_taps:
                    # The carrier and workspace are different diagnostic objects.
                    taps[f"block{index}.carrier"] = carrier
                    taps[f"block{index}.workspace"] = workspace
                    taps[f"block{index}.fanin"] = fanin
                updates.append(update)
                if schedule == "explicit" or return_taps:
                    carrier = carrier + torch.einsum("bc,cs->bsc", update, self.write[index])
            if schedule == "triangular":
                carrier = initial + torch.einsum("lbc,lcs->bsc", torch.stack(updates), self.write)
            output = torch.cat([layer(carrier[:, i]) for i, layer in enumerate(self.decode)], -1)
        return (output, taps) if return_taps else output
