# Training image models

Run from the repository root after installation. `training.train` is a
standalone AdamW trainer; it imports no toy/research code. The public configs
are portable recipes with no fixed account, run identity or artifact path.

For the complete **XL + DINOv3-L REPA** experiment, use the
[flagship workflow](FLAGSHIP.md), including a save/resume smoke and periodic
sampling/evaluation. The implementation below is shared by all sizes.

## Data and distributed execution

Set `IMAGENET256_ROOT` or pass `--data_path`. See [supported inputs](../data/README.md).
Training uses 256px ADM-style center crops, random horizontal flips, and pixels
scaled to [-1,1]. Labels are ImageNet indices 0–999; index 1000 is the null label.
The class ordering must be consistent across training and evaluation.

The configs specify batch 128 per rank and one accumulation step. With eight
ranks this is global batch 1024. To preserve that batch with four GPUs, use
`--grad_accum 2`. Adjust the total step budget when changing the dataset/global
batch. Treat GPU memory as a measured property: run a short disposable job on
your allocated GPUs before a long run, with a fresh output directory.

```bash
torchrun --standalone --nproc_per_node=8 -m training.train \
  --config configs/sublayer_b.json --data_path /path/to/imagenet \
  --run_dir outputs/disposable_check --max_steps 3 --save_every 3 --log_every 1
```

This command really trains and writes a checkpoint. It does not use W&B by default.

## Objective and optimizer

Let `t=0` be noise and `t=1` clean. Training constructs
`z=t*x+(1-t)*epsilon`, using logit-normal time sampling with mean -0.8,
standard deviation 0.8 and unit noise scale. The velocity target is
`(x-z)/max(1-t,0.05)`. Velocity prediction minimizes its squared error directly;
`--prediction clean` first converts the network output to velocity using the
same denominator. This is a time-weighted clean-prediction objective.

The shipped configs use AdamW LR 2e-4, betas (0.9,0.95), zero weight decay, 6250 warmup
steps followed by a constant LR, label dropout 0.1, and BF16 autocast on CUDA.
Model parameters and optimizer moments remain FP32. Two EMAs use 0.9999 and
0.9996. Gradient clipping is optional (`--grad_clip`); zero disables it.

`torch.compile` uses `default` in the configs. Recompute is disabled;
`--activation_checkpoint mlp` is an explicit opt-in and trades
workspace activation storage for recomputation. Block-wise models also support
`branch`; sublayer-wise models do not. It does not change the fused
carrier routing schedule. See the [fusion notes](../docs/kernel_fusion/README.md).

REPA is optional and requires a locally supplied compatible DINOv3 teacher
and the `repa` installation extra. B/L/H recipes explicitly disable it;
`configs/sublayer_xl_repa.json` enables DINOv3-L/16 at block 8 with coefficient
0.5 and output dimension 1024. No recipe downloads a teacher. See the
[flagship recipe](FLAGSHIP.md). The package retains projector metadata for loading models
that used this auxiliary loss.

## Checkpoints and resume

Use `--resume` with a checkpoint from a matching recipe and an absolute target
`--max_steps`. Checkpoints contain raw model, AdamW moments, primary/secondary
EMA, training step and reconstruction metadata. Files are written through a
temporary file followed by an atomic rename. A run directory containing
checkpoints requires explicit resume, and orphan CSV rows beyond the resumed
checkpoint must be archived/reconciled explicitly before proceeding.

Resume restores model/optimizer/EMAs and the sampler's epoch/batch offset.
The image trainer does not save per-rank RNG states, so resume is not promised
to be bitwise identical to uninterrupted stochastic training. Keep the same
model, objective, dataset, distributed layout and effective batch for a
continuation. Inference-only exports cannot resume training. Optimizer formats
other than this trainer's AdamW are outside this compact release's resume contract.

Logs are local CSV by default. Optional W&B requires `pip install -e '.[tracking]'`
and explicit `--wandb --wandb_project ... --wandb_entity ... --wandb_id ...`.
Choose your own identity; no project account is bundled. Raw training logs and
checkpoints can contain paths/identity, and are not part of an anonymous source release.

For fused block-wise models, `--kernel_backend tuple` selects the shared
tuple kernels;
`--kernel_backend legacy` selects the original kernels.
Without an override, checkpoint metadata determines the backend and older
checkpoints default to legacy. See [kernel fusion notes](../docs/kernel_fusion/README.md)
for implementation details. This option does not
change block-wise connections to sublayer connections.

On resume, an explicit `--attn_backend` or `--kernel_backend` overrides the
checkpoint's execution choice. Omitted flags follow its metadata; a fresh
model defaults to Flash attention and legacy block-wise routing. Checkpoints
record the instantiated architecture, including custom widths, stages, CTX,
MLP/dropout settings and class/channel counts, so saving after a resume does
not discard constructor overrides.
