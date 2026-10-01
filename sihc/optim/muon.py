"""Single-device Muon optimizer with AdamW auxiliary groups.

This is a small local implementation for experiments on PyTorch versions that do
not yet ship ``torch.optim.Muon``. Muon is applied to hidden matrix/conv weights;
all auxiliary parameters are handled by fused AdamW in the training script.
"""

from __future__ import annotations

import math
from typing import Iterable

import torch


@torch.compile
def _zeropower_via_newtonschulz5(g: torch.Tensor, steps: int) -> torch.Tensor:
    """Approximate the zeroth power / orthogonalization of a 2D update matrix.

    The iteration uses the quintic Newton-Schulz coefficients popularized in
    Muon implementations. It intentionally runs in bf16 for speed; the result is
    cast back to the input dtype by the caller.
    """
    if g.ndim != 2:
        raise RuntimeError("Muon Newton-Schulz input must be 2D")
    a, b, c = (3.4445, -4.7750, 2.0315)
    x = g.bfloat16()
    transposed = x.size(0) > x.size(1)
    if transposed:
        x = x.T
    x = x / (x.norm() + 1e-7)
    for _ in range(steps):
        aa = x @ x.T
        bb = b * aa + c * (aa @ aa)
        x = a * x + bb @ x
    if transposed:
        x = x.T
    return x


class Muon(torch.optim.Optimizer):
    """Muon for one process/GPU.

    Parameters are expected to be matrix-like hidden weights. For tensors with
    more than two dimensions, the first dimension is treated as rows and the
    remaining dimensions are flattened before the Newton-Schulz step.
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-3,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        weight_decay: float = 0.0,
    ) -> None:
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            nesterov = group["nesterov"]
            ns_steps = group["ns_steps"]
            weight_decay = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                if weight_decay:
                    p.mul_(1 - lr * weight_decay)
                g = p.grad
                if g.ndim < 2:
                    raise RuntimeError("Muon should only receive tensors with ndim >= 2")
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(g, alpha=1 - momentum)
                update = g.add(buf, alpha=momentum) if nesterov else buf

                original_shape = update.shape
                update_2d = update.reshape(update.shape[0], -1)
                update_2d = _zeropower_via_newtonschulz5(update_2d, ns_steps).to(dtype=p.dtype)

                # Original Muon shape scaling: wider matrices keep the base lr;
                # tall matrices get sqrt(rows / cols) scaling.
                rows, cols = update_2d.shape
                adjusted_lr = lr * math.sqrt(max(1.0, rows / cols))
                p.add_(update_2d.reshape(original_shape), alpha=-adjusted_lr)
        return loss


def split_muon_params(named_params):
    """Split parameters into Muon and AdamW auxiliary groups.

    Keep local read/write operators and output/embedding-like parameters on
    AdamW. Applying Muon to read/write templates would change the algorithmic
    object we are trying to diagnose.
    """
    muon_params = []
    aux_params = []
    muon_names = []
    aux_names = []
    aux_markers = (
        "embedding_table",
        "final_layer.linear",
        "write_weight",
        "read_weight",
    )
    for name, p in named_params:
        if not p.requires_grad:
            continue
        use_muon = p.ndim >= 2 and not any(marker in name for marker in aux_markers)
        if use_muon:
            muon_params.append(p)
            muon_names.append(name)
        else:
            aux_params.append(p)
            aux_names.append(name)
    return muon_params, aux_params, muon_names, aux_names
