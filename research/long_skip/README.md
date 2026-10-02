# Learned input-to-output long skip

`vendor/` preserves the Group 8 experiment's original Python module names. The launcher changes only import-path setup, keeping the archived model and training implementation.

```bash
python -m research.long_skip.run train --help
torchrun --standalone --nproc_per_node=8 -m research.long_skip.run train --model jit_b16_fullpatch_longskip --prediction velocity --data_path /path/to/imagenet --run_dir runs/longskip --max_steps 250200 --lr 0.0002 --ema_decay 0.9999 --ema_decay2 0.9996
```

Time follows `x_t = t*x + (1-t)*epsilon`: 0 is noise, 1 is clean. The analytic clean-to-velocity long-skip coefficient is `-1/(1-t)`. Group 8 [weights](https://huggingface.co/TongtongLiang/sihc-group8-longskip) are separate from the paper's earlier coefficient-dynamics measurements in `research/long_skip`; do not merge their run identities.

For the **Group 8** checkpoint, use its bundled evaluator, not the SiHC checkpoint loader. This explicit example uses the paper's B-size guidance settings; it does not reproduce the separate earlier coefficient-dynamics run:

```bash
python sihc/tools/download_checkpoint.py --id imagenet256-fid-stats --output checkpoints
torchrun --standalone --nproc_per_node=8 -m research.long_skip.run evaluate \
  --checkpoint /path/to/group8-checkpoint.pt --output_dir outputs/longskip-eval \
  --fid_stats checkpoints/assets/fid_stats/jit_in256_stats.npz \
  --state_key ema --prediction velocity --num_samples 50000 --batch_size 64 \
  --steps 50 --cfg 2.9 --interval_min 0.1 --interval_max 1.0 \
  --noise_scale 1.0 --seed 12345 --precision bf16
```

The archived evaluator implements Heun sampling; its default CFG is 3.0, so specify the desired CFG explicitly. No weights or dataset images are bundled.

## Measurements

`data/`, `model/` and `meta/` retain the earlier paper coefficient-dynamics experiment; [measurement notes](MEASUREMENTS.md) identify its run. `vendor/` and `run.py` contain the separate Group 8 training pipeline. `earlier_parameterization/` retains the earlier convention explicitly. Plot the paper measurements with `python research/plotting/render_learned_long_skip.py`.
