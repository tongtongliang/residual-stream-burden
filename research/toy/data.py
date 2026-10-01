"""Seeded Swiss-roll curve in a normalized high-dimensional ambient plane."""

from __future__ import annotations

import numpy as np
import torch


def make_dataset(n_samples=8192, ambient_dim=512, seed=42, noise=0.0):
    if n_samples < 2 or ambient_dim < 2 or noise < 0:
        raise ValueError("need at least 2 samples/dimensions and nonnegative noise")
    # Draw the complete 3D roll before selecting its x/z coordinates. Keeping
    # this draw order also specifies the additive-noise experiment precisely.
    rng = np.random.RandomState(seed)
    angle = 1.5 * np.pi * (1 + 2 * rng.uniform(size=n_samples))
    height = 21 * rng.uniform(size=n_samples)
    roll = np.vstack((angle * np.cos(angle), height, angle * np.sin(angle)))
    roll += noise * rng.standard_normal((3, n_samples))
    data_2d = roll[[0, 2]].T.astype(np.float32)
    raw = np.random.default_rng(seed).normal(size=(ambient_dim, 2))
    projection = np.linalg.qr(raw)[0][:, :2].astype(np.float32)
    ambient = data_2d @ projection.T
    mean = ambient.mean(0, keepdims=True).astype(np.float32)
    std = ambient.std(0, keepdims=True).astype(np.float32) + 1e-6
    x0 = ((ambient - mean) / std).astype(np.float32)
    return {"x0": x0, "data_2d": data_2d, "projection": projection,
            "mean": mean.squeeze(0), "std": std.squeeze(0)}


def project_to_2d(normalized, dataset):
    return (normalized * dataset["std"] + dataset["mean"]) @ dataset["projection"]


def normalized_data_plane(dataset):
    """Orthonormal basis after per-coordinate standardization (not raw P)."""
    return np.linalg.qr(dataset["projection"] / dataset["std"][:, None])[0][:, :2]


def make_batch(x0_all, batch_size, generator, tau_min=1e-3):
    device = x0_all.device
    indices = torch.randint(len(x0_all), (batch_size,), generator=generator, device=device)
    x0 = x0_all[indices]
    tau = torch.randn(batch_size, generator=generator, device=device).sigmoid()
    tau = tau.clamp(tau_min, 1 - tau_min)
    noise = torch.randn(x0.shape, generator=generator, device=device, dtype=x0.dtype)
    z = (1 - tau[:, None]) * x0 + tau[:, None] * noise
    return x0, noise, tau, z


def to_velocity(prediction, z, tau, mode, tau_min=1e-3):
    """Convert x/v/eps predictions to dz/dtau = noise - clean."""
    time = tau[:, None].clamp(tau_min, 1 - tau_min)
    if mode == "v":
        return prediction
    if mode == "x":
        return (z - prediction) / time
    if mode == "eps":
        return (prediction - z) / (1 - time)
    raise ValueError("mode must be x, v, or eps")


def velocity_loss(prediction, batch, mode, tau_min=1e-3):
    x0, noise, tau, z = batch
    return (to_velocity(prediction, z, tau, mode, tau_min) - (noise - x0)).square().mean()
