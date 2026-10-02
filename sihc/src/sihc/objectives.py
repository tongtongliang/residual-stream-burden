"""Matched flow path and output parameterizations."""

from dataclasses import dataclass

import torch


@dataclass
class FlowBatch:
    clean: torch.Tensor
    noise: torch.Tensor
    noisy: torch.Tensor
    timestep: torch.Tensor
    noise_fraction: torch.Tensor
    velocity_target: torch.Tensor


def make_flow_batch(
    clean: torch.Tensor,
    p_mean: float,
    p_std: float,
    noise_scale: float,
    t_eps: float,
) -> FlowBatch:
    batch = clean.shape[0]
    timestep = torch.sigmoid(
        torch.randn(batch, device=clean.device) * p_std + p_mean
    ).view(-1, 1, 1, 1)
    noise = torch.randn_like(clean) * noise_scale
    noisy = timestep * clean + (1 - timestep) * noise
    velocity = (clean - noisy) / (1 - timestep).clamp_min(t_eps)
    return FlowBatch(
        clean=clean,
        noise=noise,
        noisy=noisy,
        timestep=timestep,
        noise_fraction=1 - timestep,
        velocity_target=velocity,
    )


def prediction_to_velocity(
    prediction: torch.Tensor,
    noisy: torch.Tensor,
    timestep: torch.Tensor,
    parameterization: str,
    t_eps: float,
) -> torch.Tensor:
    if parameterization == "clean":
        return (prediction - noisy) / (1 - timestep).clamp_min(t_eps)
    if parameterization == "velocity":
        return prediction
    raise ValueError(f"unknown parameterization: {parameterization}")


def flow_matching_loss(
    prediction: torch.Tensor,
    flow: FlowBatch,
    parameterization: str,
    t_eps: float,
):
    velocity_prediction = prediction_to_velocity(
        prediction, flow.noisy, flow.timestep, parameterization, t_eps
    )
    per_sample = (velocity_prediction - flow.velocity_target).square().flatten(1).mean(1)
    return per_sample.mean(), per_sample, velocity_prediction

