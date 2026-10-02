# Spatially Indexed Hyperconnections

SiHC keeps a high-resolution residual stream while attention and MLPs operate
on a compact workspace. Learned spatial read/write maps connect the two.
The default is **sublayer-wise SiHC-B with 32 context tokens**. Block-wise
models and both fused kernel implementations are also included.

This source release provides models, sihc/training/resume, evaluation
and independent toy diagnostics. Supply your own image data and checkpoints.
No account or tracking service is required.

## Install

Use Linux and Python 3.10+. Install a CUDA-matched PyTorch/torchvision pair,
then install this checkout:

```bash
python -m pip install -e '.[eval,dev]'
```

Fused image models require an NVIDIA GPU and Triton. The tested stack is
PyTorch 2.9.1+cu128 / Triton 3.5.1; see `requirements-tested.txt`. Optional
extras are `parquet`, `repa`, `tracking` and `bench` (the JiT comparison).

## Use a model

```python
import torch
from sihc import build_model

model = build_model().cuda().eval()  # initialized sublayer SiHC-B + CTX32
x = torch.randn(1, 3, 256, 256, device="cuda")
t = torch.tensor([0.5], device="cuda")  # 0 = noise, 1 = clean
y = torch.tensor([0], device="cuda")
with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
    prediction = model(x, t, y)  # [1, 3, 256, 256]
```

`build_preset("B")` constructs a named recipe with initialized weights;
`L`, `H` and `XL` are also available. Use the checkpoint loader or sampling CLI for trained weights.

| Config | Blocks × width | Context | REPA |
|---|---:|---:|---|
| [sublayer_b.json](../configs/sublayer_b.json) | 12×768 | 32 | off |
| [sublayer_l.json](../configs/sublayer_l.json) | 24×1024 | 32 | off |
| [sublayer_h.json](../configs/sublayer_h.json) | 32×1280 | 32 | off |
| [sublayer_xl_repa.json](../configs/sublayer_xl_repa.json) | 40×1024 | 32 | DINOv3-L/16 |
| [blockwise_b.json](../configs/blockwise_b.json) | 12×768 | none | off |

All use 256px images, a direct P4 stem and a 16×16 workspace.
**Recompute is off by default**; use `--activation_checkpoint mlp` when needed.
See the [model and checkpoint guide](REPRODUCIBILITY.md).

## Train and resume

Provide `train/<class>/*.JPEG` under the dataset root, or use the supported
[parquet/cache inputs](../data/README.md).

```bash
export IMAGENET256_ROOT=/path/to/imagenet
torchrun --standalone --nproc_per_node=8 -m training.train \
  --config sihc/configs/sublayer_b.json --run_dir outputs/sublayer_b

# Resume the same run; max_steps is an absolute target.
torchrun --standalone --nproc_per_node=8 -m training.train \
  --config sihc/configs/sublayer_b.json --run_dir outputs/sublayer_b \
  --resume outputs/sublayer_b/checkpoints/step_00012510.pt
```

Configs use AdamW, BF16 and two EMAs, targeting 600 epochs at global batch
1024. Adjust the step budget for another dataset or global batch. The
[training guide](../training/README.md) covers loss conventions and resume limits.

### Flagship XL + REPA

The [flagship guide](../training/XL_REPA.md) covers the frozen DINOv3-L teacher,
REPA loss, smoke/resume check, and the complete 600-epoch train/grid/eval schedule.
After setting the data, teacher and FID-stat paths:

```bash
python -m training.xl_repa --run-dir ../outputs/xl_repa          # preview
python -m training.xl_repa --run-dir ../outputs/xl_repa --execute
```

## Sample and evaluate

```bash
python -m evaluation.sample --checkpoint /path/to/checkpoint.pt \
  --state_key ema --output outputs/samples.png --steps 50 \
  --cfg 2.9 --interval_min 0.1 --interval_max 0.9

export FID_STATS=/path/to/inception_reference_stats.npz
torchrun --standalone --nproc_per_node=8 -m evaluation.evaluate \
  --checkpoint /path/to/checkpoint.pt --output_dir outputs/evaluation \
  --state_key ema --num_samples 50000 --batch_size 32 --sampler heun --steps 50 \
  --cfg 2.9 --interval_min 0.1 --interval_max 0.9 --compile --no_save_fake_features
```

The example uses B's CFG 2.9; L/H use 2.4/2.1, and XL + REPA uses 2.4. FID statistics must match
preprocessing and Inception. See the [evaluation guide](../evaluation/README.md),
including anonymous weight export.

## Code and checks

[SiHC directory guide](../README.md) · [Research experiments](../../research/README.md) · [Kernel fusion](kernel_fusion/README.md)

```bash
python -m pytest -q -p no:cacheprovider
# On an allocated GPU:
SIHC_RUN_CUDA_TESTS=1 python -m pytest -q -p no:cacheprovider
```

MIT-licensed, with [upstream attribution](../../THIRD_PARTY_NOTICES.md).
