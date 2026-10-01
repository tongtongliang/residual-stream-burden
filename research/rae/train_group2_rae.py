#!/usr/bin/env python3
"""Run a Group 2 model through the official RAE trainer.

This wrapper keeps the upstream RAE checkout external while adding parquet
ImageNet loading, full W&B metadata, rank-zero noise-bin monitoring, and strict
checks that Group 2 stays unguided and does not instantiate a DDT/DH head.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import torch


def _bootstrap_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", required=True)
    args, _ = parser.parse_known_args()
    return args


def main() -> None:
    bootstrap = _bootstrap_args()
    rae_root = Path(os.environ.get("RAE_ROOT", "")).expanduser().resolve()
    rae_src = rae_root / "src"
    if not (rae_src / "train.py").is_file():
        raise FileNotFoundError(
            f"RAE_ROOT does not contain the official checkout: {rae_root}"
        )
    sys.path.insert(0, str(rae_src))

    from omegaconf import OmegaConf

    config_path = Path(bootstrap.config).resolve()
    cfg = OmegaConf.load(config_path)
    base_name = cfg.get("_base_", None)
    if base_name:
        base_path = (config_path.parent / str(base_name)).resolve()
        overrides = cfg
        del overrides["_base_"]
        cfg = OmegaConf.merge(OmegaConf.load(base_path), overrides)
        generated_root = Path(os.environ["GROUP2_ROOT"]) / "generated_configs"
        generated_root.mkdir(parents=True, exist_ok=True)
        config_path = generated_root / config_path.name
        OmegaConf.save(cfg, config_path)
    target = str(cfg.stage_2.target)
    if "DDT" in target or "DiTwDDTHead" in target or "DH" in target:
        raise ValueError(f"Group 2 excludes the DDT/DH head, got target={target}")
    if float(cfg.guidance.scale) != 1.0:
        raise ValueError("Group 2 evaluation must be class-conditional and unguided")

    import train as rae_train
    from stage2.transport.transport import Transport
    from utils import resume_utils, wandb_utils

    from sihc.imagenet import (
        CachedTensorImageNet256Float,
        ParquetImageNet,
        build_loader,
    )

    def prepare_parquet_dataloader(
        data_path, batch_size, workers, rank, world_size, transform=None
    ):
        data_path = Path(data_path)
        if (data_path / "manifest.json").is_file():
            # The cache already contains the exact ADM 256 crop. Keep the
            # train-only flip online and match torchvision.ToTensor's [0, 1].
            dataset = CachedTensorImageNet256Float(data_path, train=True)
        else:
            dataset = ParquetImageNet(data_path, split="train", transform=transform)
        loader = build_loader(
            dataset,
            batch_size=batch_size,
            num_workers=workers,
            distributed=True,
            rank=rank,
            world_size=world_size,
            seed=int(cfg.training.global_seed),
            drop_last=True,
        )
        return loader, loader.sampler

    def parquet_eval_dataset(path, transform=None):
        path = Path(path)
        split = path.name if path.name in {"val", "validation"} else "validation"
        root = path.parent if path.name in {"val", "validation"} else path
        return ParquetImageNet(root, split=split, transform=transform)

    rae_train.prepare_dataloader = prepare_parquet_dataloader
    rae_train.ImageFolder = parquet_eval_dataset

    original_eval = rae_train.evaluate_generation_distributed

    def skip_step_zero_eval(*args, **kwargs):
        if int(kwargs.get("global_step", -1)) == 0:
            return {}
        return original_eval(*args, **kwargs)

    rae_train.evaluate_generation_distributed = skip_step_zero_eval

    bins = 10
    bin_loss = torch.zeros(bins, dtype=torch.float64)
    bin_count = torch.zeros(bins, dtype=torch.float64)
    original_sample = Transport.sample
    original_losses = Transport.training_losses

    def tracked_sample(self, x1):
        sample = original_sample(self, x1)
        self._group2_timestep = sample[0].detach()
        return sample

    def tracked_losses(self, model, x1, model_kwargs=None):
        terms = original_losses(self, model, x1, model_kwargs)
        if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
            t = self._group2_timestep.detach().float().cpu()
            loss = terms["loss"].detach().float().cpu()
            indices = torch.clamp((t * bins).long(), max=bins - 1)
            bin_loss.scatter_add_(0, indices, loss.double())
            bin_count.scatter_add_(0, indices, torch.ones_like(loss, dtype=torch.float64))
        return terms

    Transport.sample = tracked_sample
    Transport.training_losses = tracked_losses

    def initialize_wandb(args, entity, exp_name, project_name):
        import wandb

        if os.environ.get("WANDB_KEY"):
            wandb.login(key=os.environ["WANDB_KEY"])
        config = {
            "launcher": vars(args),
            "experiment": OmegaConf.to_container(cfg, resolve=True),
            "world_size": int(os.environ.get("WORLD_SIZE", "1")),
            "hardware": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        }
        wandb.init(
            entity=entity or None,
            project=project_name,
            group=os.environ.get("WANDB_GROUP", "group2-rae"),
            name=exp_name,
            id=exp_name,
            resume="allow",
            config=config,
        )
        wandb.define_metric("train/global_step")
        wandb.define_metric("train/*", step_metric="train/global_step")
        wandb.define_metric("loss_by_noise/*", step_metric="train/global_step")
        wandb.define_metric("eval_*/*", step_metric="train/global_step")

    resume_utils.initialize = initialize_wandb

    original_log = wandb_utils.log
    last_log_step = 0
    last_log_time = time.monotonic()

    def enriched_log(stats, step=None):
        nonlocal last_log_step, last_log_time
        payload = dict(stats)
        if step is not None:
            payload["train/global_step"] = int(step)
        if "train/loss" in payload and step is not None:
            now = time.monotonic()
            delta_steps = max(int(step) - last_log_step, 1)
            delta_time = max(now - last_log_time, 1e-9)
            payload["train/step_time_sec"] = delta_time / delta_steps
            payload["train/samples_per_sec"] = (
                delta_steps * int(cfg.training.global_batch_size) / delta_time
            )
            if torch.cuda.is_available():
                payload["train/peak_memory_allocated_gib"] = (
                    torch.cuda.max_memory_allocated() / 2**30
                )
                payload["train/peak_memory_reserved_gib"] = (
                    torch.cuda.max_memory_reserved() / 2**30
                )
            valid = bin_count > 0
            for index in valid.nonzero().flatten().tolist():
                payload[f"loss_by_noise/bin_{index:02d}"] = float(
                    bin_loss[index] / bin_count[index]
                )
            bin_loss.zero_()
            bin_count.zero_()
            last_log_step = int(step)
            last_log_time = now
        original_log(payload, step=step)

    wandb_utils.log = enriched_log
    rae_train.wandb_utils.log = enriched_log

    os.environ.setdefault("ENTITY", os.environ.get("WANDB_ENTITY", ""))
    os.environ.setdefault(
        "PROJECT", os.environ.get("WANDB_PROJECT", "residual-stream-training-dynamics")
    )
    if os.environ.get("WANDB_API_KEY"):
        os.environ.setdefault("WANDB_KEY", os.environ["WANDB_API_KEY"])

    # Upstream saves its source snapshot relative to cwd.
    os.chdir(rae_root)
    sys.argv[sys.argv.index("--config") + 1] = str(config_path)
    rae_train.main()


if __name__ == "__main__":
    main()
