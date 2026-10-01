"""Euler sampling and geometry summaries for a saved toy model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .data import normalized_data_plane, project_to_2d, to_velocity
from .diagnostics import spectrum
from .models import ModelConfig, ToyModel
from .train import load_checkpoint


@torch.no_grad()
def euler_sample(model, mode, count, steps, seed=12345, tau_min=1e-3):
    if count < 1 or steps < 1 or not 0 < tau_min < .5:
        raise ValueError("positive count/steps and 0 < tau_min < .5 required")
    device = next(model.parameters()).device
    generator = torch.Generator(device=device).manual_seed(seed)
    z = torch.randn(count, model.config.ambient_dim, generator=generator, device=device)
    times = torch.linspace(1 - tau_min, tau_min, steps + 1, device=device)
    for index in range(steps):
        tau = times[index].expand(count)
        velocity = to_velocity(model(z, tau), z, tau, mode, tau_min)
        z = z + (times[index + 1] - times[index]) * velocity
    return z


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=2048)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args(argv)
    if args.threads < 1:
        raise ValueError("threads must be positive")
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("sample output directory must be empty")
    torch.set_num_threads(args.threads)
    state = load_checkpoint(args.checkpoint)
    config = state["config"]
    model = ToyModel(ModelConfig(**{k: config[k] for k in ModelConfig.__dataclass_fields__}))
    model.load_state_dict(state["model"])
    model.eval()
    samples = euler_sample(model, config["mode"], args.count, args.steps, args.seed).numpy()
    if not np.isfinite(samples).all():
        raise ValueError("nonfinite samples; inspect the checkpoint and solver settings")
    dataset = {k: v.numpy() for k, v in state["dataset"].items()}
    basis = normalized_data_plane(dataset)
    outside = samples - (samples @ basis) @ basis.T
    metrics, energy = spectrum(samples, center=True)
    metrics.update({"offplane_mean_squared_norm": float(np.square(outside).sum(1).mean()),
                    "sample_mean_squared_norm": float(np.square(samples).sum(1).mean()),
                    "solver": "euler", "steps": args.steps, "count": args.count,
                    "seed": args.seed, "tau_start": .999, "tau_end": .001,
                    "mode": config["mode"], "checkpoint_step": state["step"]})
    args.output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output / "samples.npz", normalized=samples,
                        projected_2d=project_to_2d(samples, dataset), squared_singular_values=energy)
    (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
