# Learned input-to-output long skip

`vendor/` preserves the Group 8 experiment's original Python module names. The launcher changes only import-path setup, keeping the archived model and training implementation.

```bash
python -m research.long_skip.run train --help
torchrun --standalone --nproc_per_node=8 -m research.long_skip.run train --model jit_b16_fullpatch_longskip --prediction velocity --data_path /path/to/imagenet --run_dir runs/longskip --max_steps 250200 --lr 0.0002 --ema_decay 0.9999 --ema_decay2 0.9996
```

Time follows `x_t = t*x + (1-t)*epsilon`: 0 is noise, 1 is clean. The analytic clean-to-velocity long-skip coefficient is `-1/(1-t)`. Group 8 [weights](https://huggingface.co/TongtongLiang/sihc-group8-longskip) are separate from the paper's earlier coefficient-dynamics measurements in `research/long_skip`; do not merge their run identities.

The original evaluator accepts JiT statistics supplied separately from the HF research collection. No weights or dataset images are bundled.

## Measurements

`data/`, `model/` and `meta/` retain the earlier paper coefficient-dynamics experiment; [measurement notes](MEASUREMENTS.md) identify its run. `vendor/` and `run.py` contain the separate Group 8 training pipeline. `earlier_parameterization/` retains the earlier convention explicitly. Plot the paper measurements with `python research/plotting/render_learned_long_skip.py`.
