#!/usr/bin/env python3
"""Verify that a checkpoint can reconstruct and load its exact student model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from sihc.checkpoint import (
    build_model_from_checkpoint,
    checkpoint_model_kwargs,
    checkpoint_model_name,
    checkpoint_prediction,
    load_checkpoint,
    load_model_state,
    validate_checkpoint_model,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--state_key",
        default="ema",
        choices=("model", "ema", "ema2"),
    )
    parser.add_argument("--model", default="")
    parser.add_argument(
        "--allow_unsafe_checkpoint",
        action="store_true",
        help="Allow executable pickle for a trusted legacy checkpoint.",
    )
    parser.add_argument("--attn_backend", default="math")
    parser.add_argument(
        "--no_mmap",
        action="store_true",
        help="Materialize the checkpoint instead of memory-mapping it.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    path = Path(args.checkpoint).expanduser().resolve()
    ckpt = load_checkpoint(
        path,
        mmap=not args.no_mmap,
        allow_unsafe=args.allow_unsafe_checkpoint,
    )
    model_name = args.model or checkpoint_model_name(ckpt)
    model_kwargs = checkpoint_model_kwargs(
        ckpt,
        attn_backend=args.attn_backend,
    )

    # Meta construction checks the full parameter tree without first allocating
    # several GB for large models. assign=True then performs a real state load
    # backed by the mmap tensors.
    with torch.device("meta"):
        model = build_model_from_checkpoint(
            ckpt,
            model_name=model_name,
            attn_backend=args.attn_backend,
        )
    validation = validate_checkpoint_model(model, ckpt, args.state_key)
    load_model_state(model, ckpt, args.state_key, assign=True)

    parameter_devices = sorted({parameter.device.type for parameter in model.parameters()})
    summary = {
        "checkpoint": str(path),
        "size_bytes": path.stat().st_size,
        "step": int(ckpt.get("step", -1)),
        "model": model_name,
        "prediction": checkpoint_prediction(ckpt),
        "model_kwargs": model_kwargs,
        "state_key": args.state_key,
        "state_tensors": validation["state_tensors"],
        "parameters": validation["parameters"],
        "parameter_devices_after_assign": parameter_devices,
        "available_states": [
            key for key in ("model", "ema", "ema2", "optimizer") if key in ckpt
        ],
        "status": "ok",
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
