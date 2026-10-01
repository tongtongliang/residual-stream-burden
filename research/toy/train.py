"""Train or resume a small synthetic experiment; no external services."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path

import torch

from .data import make_batch, make_dataset, velocity_loss
from .models import ModelConfig, ToyModel


DEFAULTS = {"family": "fcn", "mode": "x", "ambient_dim": 512, "width": 256,
            "depth": 5, "time_embed_dim": 256, "streams": 4, "n_samples": 8192,
            "data_noise": 0.0, "seed": 42, "batch_size": 256, "lr": 1e-4}


def write_csv(path, rows):
    if not rows:
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def save_checkpoint(path, model, optimizer, generator, dataset, config, step, history):
    payload = {"format_version": 1, "config": config, "model": model.state_dict(),
               "optimizer": optimizer.state_dict(), "rng": generator.get_state(),
               "dataset": dataset, "step": step, "history": history,
               "device_type": next(model.parameters()).device.type}
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_checkpoint(path):
    state = torch.load(path, map_location="cpu", weights_only=True)
    if state.get("format_version") != 1:
        raise ValueError("unsupported toy checkpoint format")
    return state


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--steps", type=int, default=100000, help="total target optimizer steps")
    parser.add_argument("--device", choices=("cpu", "cuda"), default=None)
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--threads", type=int, default=4)
    for name, default in DEFAULTS.items():
        kwargs = {"type": type(default), "default": None}
        if name == "family":
            kwargs["choices"] = ("fcn", "sihc")
        if name == "mode":
            kwargs["choices"] = ("x", "v", "eps")
        parser.add_argument("--" + name.replace("_", "-"), **kwargs)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if min(args.steps, args.save_every, args.log_every, args.threads) < 1:
        raise ValueError("steps, intervals, and threads must be positive")
    torch.set_num_threads(args.threads)
    state = load_checkpoint(args.resume) if args.resume else None
    if state:
        config = state["config"]
        for key in DEFAULTS:
            value = getattr(args, key)
            if value is not None and value != config[key]:
                raise ValueError(f"cannot change {key} when resuming")
        output = args.output or args.resume.parent
    else:
        if args.output is None:
            raise ValueError("new runs require --output")
        config = {k: getattr(args, k) if getattr(args, k) is not None else v
                  for k, v in DEFAULTS.items()}
        output = args.output
    if config["batch_size"] < 1 or config["lr"] <= 0:
        raise ValueError("batch_size and lr must be positive")
    device = torch.device(args.device or (state["device_type"] if state else "cpu"))
    if state and device.type != state["device_type"]:
        raise ValueError("resume requires the same device type to restore its batch RNG")
    if output.exists() and any(output.iterdir()):
        if not args.resume or output.resolve() != args.resume.parent.resolve():
            raise ValueError("output must be empty for a new or relocated run")
        latest = output / "last.pt"
        if latest.exists() and latest.resolve() != args.resume.resolve():
            raise ValueError("resume from last.pt or choose a new empty output directory")
    output.mkdir(parents=True, exist_ok=True)
    model_config = ModelConfig(**{k: config[k] for k in ModelConfig.__dataclass_fields__})
    torch.manual_seed(config["seed"] + 123)
    model = ToyModel(model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"],
                                  betas=(.9, .999), eps=1e-8, weight_decay=0)
    generator = torch.Generator(device=device).manual_seed(config["seed"] + 1000)
    if state:
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        generator.set_state(state["rng"])
        dataset, start, history = state["dataset"], state["step"], state["history"]
    else:
        arrays = make_dataset(config["n_samples"], config["ambient_dim"],
                              config["seed"], config["data_noise"])
        dataset = {k: torch.from_numpy(v) for k, v in arrays.items()}
        start, history = 0, []
    if args.steps <= start:
        raise ValueError("target steps must exceed the checkpoint step")
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    x0 = dataset["x0"].to(device)
    model.train()
    for step in range(start + 1, args.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        batch = make_batch(x0, config["batch_size"], generator)
        loss = velocity_loss(model(batch[3], batch[2]), batch, config["mode"])
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        if step == 1 or step % args.log_every == 0 or step == args.steps:
            row = {"step": step, "loss": loss.item(), "gradient_norm": grad_norm.item()}
            history.append(row)
            print(json.dumps(row), flush=True)
        if step % args.save_every == 0 or step == args.steps:
            save_checkpoint(output / "last.pt", model, optimizer, generator,
                            dataset, config, step, history)
            write_csv(output / "loss.csv", history)


if __name__ == "__main__":
    main()
