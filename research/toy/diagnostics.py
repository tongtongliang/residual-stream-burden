"""Spectra, noise sensitivity, and linear-layer gradient checks."""

from __future__ import annotations

import numpy as np
import torch


def spectrum(matrix, *, center=True):
    """FP64 SVD of flattened samples; return metrics and absolute energy."""
    if isinstance(matrix, torch.Tensor):
        matrix = matrix.detach().cpu().numpy()
    a = np.asarray(matrix, dtype=np.float64)
    if a.ndim < 2 or not a.shape[0]:
        raise ValueError("expected a nonempty [sample, ...] array")
    a = a.reshape(len(a), -1)
    if center:
        a = a - a.mean(0, keepdims=True)
    energy = np.linalg.svd(a, compute_uv=False) ** 2
    total = float(energy.sum())
    if total == 0:
        return {"stable_rank": 0.0, "rank90": 0, "rank95": 0,
                "entropy_rank": 0.0, "frobenius_squared": 0.0,
                "largest_singular_squared": 0.0}, energy
    p = energy / total
    positive = p[p > 0]
    cumulative = p.cumsum()
    metrics = {
        "stable_rank": total / float(energy[0]),
        "rank90": int(np.searchsorted(cumulative, .90) + 1),
        "rank95": int(np.searchsorted(cumulative, .95) + 1),
        "entropy_rank": float(np.exp(-(positive * np.log(positive)).sum())),
        "frobenius_squared": total,
        "largest_singular_squared": float(energy[0]),
    }
    return metrics, energy


def normalized_noise_variance(representations):
    """E_x sum Var_noise(h|x) / E_{x,noise} ||h||²; population variance."""
    h = np.asarray(representations, dtype=np.float64)
    if h.ndim < 3 or min(h.shape[:2]) < 1:
        raise ValueError("expected [clean sample, noise draw, ...]")
    h = h.reshape(*h.shape[:2], -1)
    numerator = float(h.var(axis=1).sum(-1).mean())
    denominator = float(np.square(h).sum(-1).mean())
    return numerator / (denominator + 1e-20)


def linear_weight_gradient(inputs, output_gradient):
    """Exact linear-layer identity dL/dW = (dL/dY)^T X, including loss scale."""
    x = inputs.reshape(-1, inputs.shape[-1])
    residual = output_gradient.reshape(-1, output_gradient.shape[-1])
    if x.shape[0] != residual.shape[0]:
        raise ValueError("input/output-gradient sample dimensions differ")
    return residual.T @ x


def gradient_snapshot(model, batch, mode):
    """One diagnostic backward; report uncentered weight-gradient spectra.

    Hooks are removed even on error. No optimizer step is performed. Parameters'
    existing gradients are cleared, so use a separate loaded model for analysis.
    """
    from .data import velocity_loss

    captured = {}
    handles = []

    def hook(name):
        def record(module, inputs, output):
            output.retain_grad()
            captured[name] = (module, inputs[0].detach(), output)
        return record

    try:
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                handles.append(module.register_forward_hook(hook(name)))
        model.zero_grad(set_to_none=True)
        _, _, tau, z = batch
        velocity_loss(model(z, tau), batch, mode).backward()
        rows = []
        for name, (layer, inputs, output) in captured.items():
            exact = linear_weight_gradient(inputs, output.grad)
            grad = layer.weight.grad
            metrics, _ = spectrum(grad, center=False)
            error = float((grad - exact).double().norm().item())
            scale = float(grad.double().norm().item())
            rows.append({"layer": name, "factorization_absolute_error": error,
                         "factorization_relative_error": error / max(scale, 1e-20), **metrics})
        return rows
    finally:
        for handle in handles:
            handle.remove()
        model.zero_grad(set_to_none=True)
