#!/usr/bin/env python3
"""Sample from a FlowMatching SiHC checkpoint.

Non-context models can use ``--naive`` for CPU diagnostics. Context models
require the fused CUDA path; the reference schedule does not support them.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image

from sihc.checkpoint import (
    build_model_from_checkpoint,
    checkpoint_model_name,
    checkpoint_prediction,
    load_checkpoint,
    load_model_state,
)
from sihc.models.sihc import enable_reference_schedule
from sihc.sampling import sample_heun, to_uint8


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", default="outputs/sample_grid.png")
    p.add_argument("--model", default="auto")
    p.add_argument("--state_key", default="ema", choices=["ema", "ema2", "model"])
    p.add_argument("--prediction", default="auto", choices=["auto", "clean", "velocity"])
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--class_id", type=int, default=207, help="ImageNet class id used for all samples unless --balanced_labels is set.")
    p.add_argument("--balanced_labels", action="store_true")
    p.add_argument("--nrow", type=int, default=4)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--cfg", type=float, default=2.4)
    p.add_argument("--interval_min", type=float, default=0.1)
    p.add_argument("--interval_max", type=float, default=0.9)
    p.add_argument("--noise_scale", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--compile", action="store_true")
    p.add_argument("--compile_mode", default="reduce-overhead")
    p.add_argument("--naive", action="store_true", help="Use the PyTorch reference for non-context models; context and write-only models are unsupported.")
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb_project", default="")
    p.add_argument("--wandb_entity", default="")
    p.add_argument("--wandb_run_id", default="")
    p.add_argument("--wandb_run_name", default="")
    p.add_argument("--wandb_epoch", type=float, default=None)
    p.add_argument(
        "--allow_unsafe_checkpoint",
        action="store_true",
        help="Allow executable pickle for a trusted legacy checkpoint.",
    )
    p.add_argument('--kernel_backend', default=None, choices=['legacy', 'tuple'],
                   help='Override block-wise routing backend; default follows checkpoint.')
    return p.parse_args()


def infer_model_name(args, ckpt):
    if args.model != "auto":
        return args.model
    return checkpoint_model_name(ckpt)


def infer_prediction(args, ckpt):
    if args.prediction != "auto":
        return args.prediction
    return checkpoint_prediction(ckpt)


def save_grid(images, path, nrow=4):
    images = images.cpu().numpy().transpose(0, 2, 3, 1)
    h, w = images.shape[1], images.shape[2]
    nrow = min(nrow, len(images))
    ncol = (len(images) + nrow - 1) // nrow
    canvas = Image.new("RGB", (nrow * w, ncol * h))
    for i, img in enumerate(images):
        canvas.paste(Image.fromarray(img), ((i % nrow) * w, (i // nrow) * h))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def main():
    args = parse_args()
    device = torch.device(args.device)
    ckpt = load_checkpoint(
        args.checkpoint,
        mmap=True,
        allow_unsafe=args.allow_unsafe_checkpoint,
    )
    model_name = infer_model_name(args, ckpt)
    prediction = infer_prediction(args, ckpt)
    model = build_model_from_checkpoint(
        ckpt,
        model_name=model_name,
        overrides={'kernel_backend': args.kernel_backend} if args.kernel_backend else None,
        attn_backend="flash" if device.type == "cuda" else "math",
    )
    load_model_state(model, ckpt, args.state_key)
    if args.naive:
        enable_reference_schedule(model)
    model.to(device).eval()
    if args.compile:
        model = torch.compile(model, mode=args.compile_mode)

    # Model initialization consumes RNG in a size-dependent way. Reset here so
    # scaling runs use exactly the same labels and initial sampling noise.
    torch.manual_seed(args.seed)
    if args.balanced_labels:
        labels = torch.arange(args.batch_size, device=device, dtype=torch.long) % 1000
    else:
        labels = torch.full((args.batch_size,), args.class_id, device=device, dtype=torch.long)
    with torch.no_grad():
        images = sample_heun(
            model,
            labels,
            image_size=256,
            noise_scale=args.noise_scale,
            steps=args.steps,
            cfg_scale=args.cfg,
            interval=(args.interval_min, args.interval_max),
            num_classes=1000,
            prediction=prediction,
        )
    out = to_uint8(images)
    output_path = Path(args.output)
    save_grid(out, output_path, nrow=args.nrow)
    if args.wandb:
        if not args.wandb_project or not args.wandb_run_id:
            raise ValueError('--wandb requires --wandb_project and --wandb_run_id')
        import wandb

        run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity or None,
            id=args.wandb_run_id,
            name=args.wandb_run_name or None,
            resume='allow',
            config={'sample_script': 'evaluation/sample.py'},
        )
        wandb.define_metric('sample/global_step')
        wandb.define_metric('sample/*', step_metric='sample/global_step')
        global_step = int(ckpt.get('step', ckpt.get('global_step', 0)))
        payload = {
            'sample/global_step': global_step,
            'sample/grid_5x5': wandb.Image(
                str(output_path),
                caption=(
                    f'{args.state_key} Heun-{args.steps} CFG={args.cfg} '
                    f'interval=({args.interval_min},{args.interval_max}) seed={args.seed}'
                ),
            ),
        }
        if args.wandb_epoch is not None:
            payload['sample/epoch'] = args.wandb_epoch
        history_step = global_step + 1
        payload['sample/wandb_history_step'] = history_step
        wandb.log(payload, step=history_step)
        run.finish()
    print(f"saved {args.output} model={model_name} state={args.state_key} prediction={prediction} labels={labels[:8].tolist()}")


if __name__ == "__main__":
    main()
