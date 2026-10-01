#!/usr/bin/env python3
"""Evaluate any Group 2 pixel run with the Group 1/JiT protocol; optionally bridge to ADM."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
GROUP2_DIR = REPO_ROOT / "research" / "transport_burden" / "group2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--sample-dir", required=True)
    parser.add_argument("--nproc", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=32, help="Per-rank generation batch.")
    parser.add_argument("--output-npz", default="")
    parser.add_argument("--adm-bridge", action="store_true", help="Also pack and ADM-score the PNGs.")
    parser.add_argument("--keep-samples", action="store_true")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_pixel_spec(experiment: str) -> tuple[dict, dict]:
    registry = json.loads((GROUP2_DIR / "experiments.json").read_text())
    if experiment not in registry:
        raise ValueError(f"unknown experiment {experiment}; choices={sorted(registry)}")
    spec = registry[experiment]
    if spec["engine"] != "pixel":
        raise ValueError(f"{experiment} is not a pixel experiment")
    config = json.loads((GROUP2_DIR / spec["config"]).read_text())
    required = {
        "eval_sampler", "eval_steps", "eval_num_samples", "eval_state_key",
        "eval_seed", "eval_cfg", "eval_cfg_interval", "fid_stats",
    }
    missing = sorted(required - config.keys())
    if missing:
        raise RuntimeError(f"pixel config is missing final-eval metadata: {missing}")
    return spec, config


def pack_pngs(sample_dir: Path, output_npz: Path, expected: int) -> None:
    files = sorted(sample_dir.glob("*.png"))
    if len(files) != expected:
        raise RuntimeError(f"expected {expected} PNGs in {sample_dir}, found {len(files)}")
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    if output_npz.exists():
        raise FileExistsError(f"refusing to overwrite final archive: {output_npz}")
    staging = output_npz.with_suffix(output_npz.suffix + ".arr_0.tmp.npy")
    if staging.exists():
        raise FileExistsError(f"remove stale temporary archive first: {staging}")

    array = np.lib.format.open_memmap(
        staging, mode="w+", dtype=np.uint8, shape=(expected, 256, 256, 3)
    )
    try:
        for index, path in enumerate(files):
            expected_name = f"{index:08d}.png"
            if path.name != expected_name:
                raise RuntimeError(
                    f"non-contiguous sample sequence: expected {expected_name}, got {path.name}"
                )
            with Image.open(path) as image:
                image = image.convert("RGB")
                if image.size != (256, 256):
                    raise RuntimeError(f"unexpected sample size {image.size}: {path}")
                array[index] = np.asarray(image, dtype=np.uint8)
        array.flush()
        del array
        np.savez(output_npz, arr_0=np.load(staging, mmap_mode="r"))
    finally:
        if "array" in locals():
            del array
        staging.unlink(missing_ok=True)
    print(f"ADM_SAMPLE_NPZ={output_npz}", flush=True)


def main() -> None:
    args = parse_args()
    _, config = load_pixel_spec(args.experiment)
    if args.nproc < 1:
        raise ValueError("--nproc must be positive")
    checkpoint = Path(args.checkpoint).resolve()
    if not args.dry_run and not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    sample_dir = Path(args.sample_dir).resolve()
    existing = list(sample_dir.glob("*.png")) if sample_dir.exists() else []
    if existing and not args.dry_run:
        raise RuntimeError(f"sample directory already contains {len(existing)} PNGs; use an empty directory")
    output_npz = Path(args.output_npz).resolve() if args.output_npz else sample_dir.with_suffix(".npz")
    interval = config["eval_cfg_interval"]
    output_dir = sample_dir.parent / f"{sample_dir.name}_metrics"
    fid_stats = os.environ.get("FID_STATS")
    if not fid_stats:
        raise RuntimeError("FID_STATS must point to the Group 1/JiT ImageNet-256 statistics")
    command = [
        "torchrun", "--standalone", f"--nproc_per_node={args.nproc}",
        str(REPO_ROOT / "scripts" / "evaluate.py"),
        "--checkpoint", str(checkpoint), "--output_dir", str(output_dir),
        "--sample_dir", str(sample_dir), "--fid_stats", str(Path(fid_stats).resolve()),
        "--state_key", str(config["eval_state_key"]),
        "--num_samples", str(config["eval_num_samples"]),
        "--batch_size", str(args.batch_size), "--sampler", str(config["eval_sampler"]),
        "--steps", str(config["eval_steps"]), "--cfg", str(config["eval_cfg"]),
        "--interval_min", str(interval[0]), "--interval_max", str(interval[1]),
        "--noise_scale", str(config.get("noise_scale", 1.0)), "--seed", str(config["eval_seed"]),
        "--compile", "--compile_mode", "reduce-overhead",
    ]
    if args.keep_samples or args.adm_bridge:
        command.append("--keep_samples")
    if args.wandb:
        command.extend([
            "--wandb", "--wandb_project", os.environ.get("WANDB_PROJECT", ""),
            "--wandb_entity", os.environ.get("WANDB_ENTITY", ""),
            "--wandb_run_id", str(config.get("wandb_id", args.experiment)),
            "--wandb_run_name", str(config.get("run_name", args.experiment)),
        ])
    print("command:", " ".join(command), flush=True)
    if args.dry_run:
        return
    subprocess.run(command, cwd=REPO_ROOT, env=os.environ.copy(), check=True)
    if args.adm_bridge:
        pack_pngs(sample_dir, output_npz, int(config["eval_num_samples"]))
        subprocess.run(
            [sys.executable, str(REPO_ROOT / "scripts" / "group2_score_adm.py"), str(output_npz)],
            cwd=REPO_ROOT, env=os.environ.copy(), check=True,
        )


if __name__ == "__main__":
    main()
