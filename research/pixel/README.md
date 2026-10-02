# Controlled JiT and mHC experiments

The B-size controls use full-rank 16×16 RGB patch embeddings, 12 Transformer blocks, width 768, 12 heads, and no bottleneck or in-context class tokens. Clean and velocity objectives share this backbone.

## mHC configuration

The paper mHC controls use **dynamic four-stream manifold-constrained Hyper-Connections**, implemented with **Transformer Engine's PyTorch Triton mHC kernels** (`mhc_backend=nvidia_fused`). A separate connection surrounds each attention and each MLP sublayer. Each stream has width 768; shared attention/MLP computation still operates at width 768. Residual mixing uses 20 Sinkhorn iterations. Read, write and residual-mixing maps depend on the current state.

| Setting | Value |
|---|---|
| Streams | 4 |
| Connection placement | Sublayerwise: attention and MLP each have a connection |
| Backend | `transformer_engine.pytorch.triton.mhc` / `nvidia_fused` |
| Sinkhorn iterations | 20 |
| Optimizer | AdamW, lr 2e-4, betas (0.9, 0.95), weight decay 0 |
| Global batch | 1024 |
| Main-table boundary | Epoch 200 / step 250200 |
| Primary EMA | 0.9999 |

SiHC instead uses static feature-wise maps and identity carry; it does **not** require Transformer Engine. The mHC reference backend is a separate PyTorch implementation for debugging and correctness comparisons, not the backend used for the reported training runs.

## Epoch-200 results

| Pixel-space target | Plain JiT FID | Four-stream mHC FID |
|---|---:|---:|
| Clean | 10.192 | 9.775 |
| Velocity | 139.829 | 25.360 |

Results come from [the archived evaluation table](../reference_tables/group1_epoch200.csv): primary EMA, balanced 50K ImageNet-256 samples, JiT statistics, Heun-50, CFG 2.9, clean-time interval [0.1, 1.0]. DINOv2-B controls have their own [RAE setup](../rae/README.md).

## Reproduce with the original backend

```bash
pip install -e '.[mhc,eval,research]'
python -c 'from transformer_engine.pytorch.triton.mhc import mhc_fused_projection, mhc_fused_aggregate, mhc_fused_expand_combine, mhc_fused_scale, mhc_fused_sinkhorn'
torchrun --standalone --nproc_per_node=8 -m research.pixel.train \
  --config research/pixel/configs/mhc_n4_velocity.json \
  --mhc_backend nvidia_fused --max_steps 250200 \
  --data_path /path/to/imagenet --run_dir runs/mhc-v
```

Transformer Engine must provide the five imported mHC functions; the installed TE version/commit was not pinned in the recovered source environment. The import check makes this dependency explicit. Existing configs retain archived training extensions, so `--max_steps 250200` selects the main-table epoch-200 boundary.

For debugging without TE, explicitly choose `--mhc_backend reference`. The trainer records this backend in checkpoints and fails if the requested fused implementation is unavailable; it does not silently substitute reference kernels.
