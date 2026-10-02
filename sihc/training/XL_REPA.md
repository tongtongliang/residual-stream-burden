# Flagship XL with online REPA

This is the complete sublayer-wise **40 × 1024** training recipe: direct P4,
five 8-block stages, 16 attention heads and 32 context tokens. It uses the
same `training.train` implementation as B/L/H, with REPA enabled. No private
repository, tracking account or unpublished training module is required.

## Teacher and loss

Supply the official `facebook/dinov3-vitl16-pretrain-lvd1689m` weights in a
local Hugging Face directory, including its config and safetensors. Obtain
access from the upstream provider and download with your own credentials;
teacher weights are not redistributed in this source archive.

The frozen teacher sees the same clean, randomly flipped 256px image as the
student, with ImageNet mean/std normalization. Do not apply a 224px teacher
crop or precompute unflipped teacher features. Remove its CLS and four
register tokens to obtain 256 × 1024 targets.

After block 8's MLP, the student projects `z_mlp + dz_mlp` using
1024 → 2048 → 2048 → 1024 linears with SiLU between them. CTX32 starts at
block 9, after this tap. The objective is velocity MSE plus 0.5 times negative
mean cosine similarity. A negative REPA loss is expected. The teacher is
frozen; the projector is optimized and included in raw and both EMA states.
The executable recipe is [sublayer_xl_repa.json](../configs/sublayer_xl_repa.json).

## Prepare and smoke-test

From the source root:

```bash
python -m pip install -e '.[repa,eval]'
export IMAGENET256_ROOT=/path/to/imagenet256
export DINOV3_PATH=/path/to/dinov3-vitl16-pretrain-lvd1689m
export FID_STATS=/path/to/jit_in256_stats.npz

# Real teacher, loss, backward, checkpoint save and optimizer/EMA resume.
# Select allocated GPUs; use a fresh directory for every smoke.
CUDA_VISIBLE_DEVICES=0 python -m training.xl_repa \
  --run-dir ../outputs/xl_repa_smoke --nproc 1 --batch-size 2 --smoke --execute
```

The smoke trains four steps, saves a full checkpoint and resumes to step six.
Repeat with eight allocated GPUs and batch 128 before the full run.
Recompute is off by default.

Use ImageNet's 1,281,167 training images with the [supported loader/cache](../data/README.md).
The 600-epoch budget assumes 8 × 128 = 1024 images per optimizer step,
1,251 steps per epoch, and 750,600 total steps. AdamW uses LR 2e-4,
6,250-step warmup then constant LR, betas (0.9, 0.95), zero weight decay,
FP32 parameters, BF16 autocast, and EMAs 0.9999/0.9996.

## Train, evaluate and resume

```bash
# Print the entire plan without launching jobs.
python -m training.xl_repa --run-dir ../outputs/xl_repa
# Execute in your job scheduler or persistent terminal session.
python -m training.xl_repa --run-dir ../outputs/xl_repa --execute

# Continue an existing run to the absolute 600-epoch target.
python -m training.xl_repa --run-dir ../outputs/xl_repa \
  --resume ../outputs/xl_repa/checkpoints/step_00650520.pt --epochs 600 --execute
```

The driver trains in 20-epoch segments and saves a primary-EMA 5×5 grid at
each boundary. Every 40 epochs it runs balanced 50K Heun-50 evaluation with
**CFG 2.4, interval [0.1, 0.9], seed 12345**, then continues only if evaluation
succeeds. Training and historical sampling/evaluation use default compilation.
No W&B session is created. Use the individual CLIs if a different schedule is
needed; `training.train --config sihc/configs/sublayer_xl_repa.json` trains directly
without automatic grids/evaluation.

Resume requires a full training checkpoint, not an inference export. Existing
CSV rows newer than the resumed checkpoint must be reconciled explicitly;
see [resume semantics](README.md#checkpoints-and-resume). At an exact 20-epoch
resume boundary the driver repeats the grid and, when due, evaluation before
continuing. It never treats an empty done marker as proof of successful eval.
Use a fresh smoke directory; formal runs require explicit `--resume` once a
run directory exists. Child failures stop the sequence.

The recommended released XL checkpoint is epoch 520 (`step_00650520.pt`,
primary EMA), not automatically the final epoch-600 checkpoint. Sampling from
it reconstructs the projector but does not require the teacher weights.
