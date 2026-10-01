"""Low-frequency diagnostics shared with residual-stream dynamics runs."""

import torch
import torch.distributed as dist
import torch.nn.functional as F


@torch.no_grad()
def tensor_l2_norm(tensors) -> torch.Tensor:
    total = None
    for tensor in tensors:
        if tensor is None:
            continue
        value = tensor.detach().float().square().sum()
        total = value if total is None else total + value
    if total is None:
        return torch.zeros((), device="cuda" if torch.cuda.is_available() else "cpu")
    return total.sqrt()


def parameter_groups(model) -> dict[str, list[torch.nn.Parameter]]:
    groups = {
        "patch_embed": list(model.x_embedder.parameters()),
        "conditioning": list(model.t_embedder.parameters()) + list(model.y_embedder.parameters()),
        "readout": list(model.final_layer.parameters()),
    }
    for index, block in enumerate(model.blocks, start=1):
        groups[f"block{index:02d}"] = list(block.parameters())
    return groups


@torch.no_grad()
def norm_metrics(model, ema_params, include_groups: bool) -> dict[str, float]:
    parameters = list(model.parameters())
    weight_norm = tensor_l2_norm(parameters)
    grad_norm = tensor_l2_norm([parameter.grad for parameter in parameters])
    output = {
        "train/weight_norm": weight_norm.item(),
        "train/grad_norm": grad_norm.item(),
        "train/grad_weight_ratio": (grad_norm / weight_norm.clamp_min(1e-12)).item(),
    }
    if include_groups:
        for name, params in parameter_groups(model).items():
            output[f"norm/weight_{name}"] = tensor_l2_norm(params).item()
            output[f"norm/grad_{name}"] = tensor_l2_norm(
                [parameter.grad for parameter in params]
            ).item()
        ema_distance = tensor_l2_norm(
            [parameter.detach() - ema for parameter, ema in zip(parameters, ema_params)]
        )
        output["train/ema_parameter_distance"] = ema_distance.item()
        output["train/ema_relative_distance"] = (
            ema_distance / weight_norm.clamp_min(1e-12)
        ).item()
    return output


@torch.no_grad()
def batch_geometry_metrics(noisy, target, prediction, noise_fraction) -> dict[str, float]:
    target_flat = target.float().flatten(1)
    prediction_flat = prediction.float().flatten(1)
    cosine = F.cosine_similarity(prediction_flat, target_flat, dim=1).mean()
    values = torch.stack(
        [
            noisy.float().square().mean(),
            target_flat.square().mean(),
            prediction_flat.square().mean(),
            cosine,
            noise_fraction.mean(),
        ]
    )
    if dist.is_initialized():
        dist.all_reduce(values)
        values /= dist.get_world_size()
    return {
        "batch/input_rms": values[0].sqrt().item(),
        "batch/target_rms": values[1].sqrt().item(),
        "batch/prediction_rms": values[2].sqrt().item(),
        "batch/prediction_target_cosine": values[3].item(),
        "batch/noise_fraction_mean": values[4].item(),
    }


@torch.no_grad()
def timestep_bin_metrics(per_sample_loss, noise_fraction, bins: int = 10):
    q = noise_fraction.flatten()
    indexes = torch.clamp((q * bins).long(), 0, bins - 1)
    sums = torch.zeros(bins, device=q.device, dtype=torch.float64)
    counts = torch.zeros(bins, device=q.device, dtype=torch.float64)
    sums.scatter_add_(0, indexes, per_sample_loss.detach().double())
    counts.scatter_add_(0, indexes, torch.ones_like(q, dtype=torch.float64))
    if dist.is_initialized():
        dist.all_reduce(sums)
        dist.all_reduce(counts)
    output = {}
    for index in range(bins):
        if counts[index] > 0:
            output[f"loss_by_noise/q{index / bins:.1f}_{(index + 1) / bins:.1f}"] = (
                sums[index] / counts[index]
            ).item()
    return output


@torch.no_grad()
def reduce_mean(value: torch.Tensor) -> torch.Tensor:
    value = value.detach().clone()
    if dist.is_initialized():
        dist.all_reduce(value, op=dist.ReduceOp.SUM)
        value /= dist.get_world_size()
    return value
