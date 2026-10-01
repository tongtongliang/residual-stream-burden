"""Read a toy checkpoint and write paired representation diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .data import make_batch
from .diagnostics import gradient_snapshot, normalized_noise_variance, spectrum
from .models import ModelConfig, ToyModel
from .train import load_checkpoint, write_csv


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--clean-times", type=float, nargs="+", default=[.1, .3, .5, .7, .9])
    parser.add_argument("--n-clean", type=int, default=128)
    parser.add_argument("--n-noise", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--gradients", action="store_true")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args(argv)
    if min(args.n_clean, args.n_noise, args.batch_size, args.threads) < 1:
        raise ValueError("sample counts, batch size, and threads must be positive")
    if any(not 0 <= t <= 1 for t in args.clean_times):
        raise ValueError("clean-times must be in [0, 1]")
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("diagnostic output directory must be empty")
    torch.set_num_threads(args.threads)
    state = load_checkpoint(args.checkpoint)
    config = state["config"]
    model = ToyModel(ModelConfig(**{k: config[k] for k in ModelConfig.__dataclass_fields__}))
    model.load_state_dict(state["model"])
    model.eval()
    dataset = state["dataset"]["x0"]
    if args.n_clean > len(dataset):
        raise ValueError("n-clean exceeds the saved dataset size")
    generator = torch.Generator().manual_seed(args.seed)
    # Fixed sample prefix and fixed noise give paired comparisons across runs
    # that share the same dataset; no train batch generator state is consumed.
    clean = dataset[:args.n_clean].repeat_interleave(args.n_noise, dim=0)
    noise = torch.randn(clean.shape, generator=generator)
    rows, spectra = [], {}
    for clean_time in args.clean_times:
        tau = 1 - clean_time
        z = clean_time * clean + tau * noise
        collected = {}
        with torch.no_grad():
            for start in range(0, len(z), args.batch_size):
                batch = z[start:start + args.batch_size]
                _, taps = model(batch, torch.full((len(batch),), tau), return_taps=True)
                for name, tensor in taps.items():
                    collected.setdefault(name, []).append(tensor.numpy())
        for name, batches in collected.items():
            feature = np.concatenate(batches)
            metrics, energy = spectrum(feature, center=True)
            nsv = normalized_noise_variance(feature.reshape(args.n_clean, args.n_noise, -1))
            key = f"t{clean_time:g}_{name}"
            spectra[key] = energy
            rows.append({"clean_time": clean_time, "noise_time": tau, "tap": name,
                         "samples": len(feature), "dimension": int(np.prod(feature.shape[1:])),
                         "normalized_noise_variance": nsv, **metrics})
    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / "representations.csv", rows)
    np.savez_compressed(args.output / "spectra.npz", **spectra)
    if args.gradients:
        batch = make_batch(dataset, min(args.batch_size, len(dataset)), generator)
        write_csv(args.output / "gradients.csv", gradient_snapshot(model, batch, config["mode"]))
    protocol = {"mode": config["mode"], "checkpoint_step": state["step"],
                "model_config": {k: config[k] for k in ModelConfig.__dataclass_fields__},
                "seed": args.seed, "n_clean": args.n_clean, "n_noise": args.n_noise,
                "clean_times": args.clean_times, "batch_size": args.batch_size,
                "time_convention": "clean t=1-tau; z=t*x0+(1-t)*epsilon; model receives tau",
                "spectrum": "centered FP64 features; squared singular values, not normalized",
                "sampling": "saved dataset prefix; paired Gaussian noise at every time",
                "noise_variance": "population variance within each clean sample",
                "forward_dtype": "float32", "device": "cpu",
                "gradient_diagnostics": args.gradients}
    (args.output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
