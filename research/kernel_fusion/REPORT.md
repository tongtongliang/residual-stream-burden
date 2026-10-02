# Kernel fusion benchmark

A matched comparison of **ordinary Torch + compile** against **fused routing + compile**, within each SiHC topology. Block-wise fusion uses the tuple backend; sublayer-wise fusion uses the native kernels. All models include CTX32. B/L/XL/H are 12×768, 24×1024, 40×1024 and 32×1280, respectively.

## Setup

- H100 80GB HBM3, 256×256 images, batch 128 per GPU; FP32 parameters, BF16 autocast and Flash SDPA.
- Training: compile **default**, then DDP with world size 1 (`gradient_as_bucket_view=False`); forward/loss/backward, fused AdamW and two FP32 EMAs.
- AdamW: LR 2e-4, betas (0.9, 0.95), decay 0; EMA decays 0.9999/0.9996.
- Inference: compile **reduce-overhead**, one denoiser forward, no DDP or sampling/CFG loop.
- Three fresh processes per cell, eight warmups and 50 measured steps. Tables show medians; CSVs retain ranges and throughput.
- PyTorch 2.9.1+cu128 / Triton 3.5.1; `dynamic=False`, no fullgraph argument, `expandable_segments:True`.

Six GPUs ran independent trials. These timings exclude inter-GPU communication, data loading, compilation, REPA and checkpoint I/O. XL is the backbone without its REPA teacher/projector. Training allocated memory is the steady peak; inference includes CUDA Graph setup. GiB means 2^30 bytes; driver/library allocations are excluded.

## Training

B/L use no recompute; XL/H use MLP recompute in this table. Both implementations use the same setting. The separate recompute table below shows the alternatives.

| Model | Recompute | Torch ms | Fused ms | Speedup | Torch / fused GiB |
|---|---|---:|---:|---:|---:|
| B block | none | 123.42 | 97.05 | 1.27× | 26.82 / 18.90 |
| L block | none | 365.24 | 292.77 | 1.25× | 70.83 / 50.33 |
| XL block | mlp | OOM | 527.67 | — | OOM / 59.04 |
| H block | mlp | OOM | 602.97 | — | OOM / 63.06 |
| B sublayer | none | 210.38 | 113.73 | 1.85× | 30.18 / 19.82 |
| L sublayer | none | OOM | 330.64 | — | OOM / 52.07 |
| XL sublayer | mlp | OOM | 573.15 | — | OOM / 62.07 |
| H sublayer | mlp | OOM | 682.42 | — | OOM / 67.18 |

## Inference

| Model | Torch ms | Fused ms | Speedup | Torch / fused GiB |
|---|---:|---:|---:|---:|
| B block | 35.90 | 26.54 | 1.35× | 3.05 / 3.01 |
| L block | 102.17 | 76.76 | 1.33× | 5.05 / 4.99 |
| XL block | 171.47 | 129.73 | 1.32× | 6.19 / 6.13 |
| H block | 191.16 | 149.61 | 1.28× | 7.71 / 7.60 |
| B sublayer | 52.27 | 28.82 | 1.81× | 3.68 / 3.25 |
| L sublayer | 146.50 | 83.48 | 1.75× | 5.93 / 5.44 |
| XL sublayer | 247.14 | 137.36 | 1.80× | 7.07 / 6.08 |
| H sublayer | 266.41 | 157.72 | 1.69× | 8.86 / 8.19 |

## Recompute: fused training

| Model | None: ms / GiB | MLP: ms / GiB |
|---|---:|---:|
| B block | 97.05 / 18.90 | 104.78 / 13.57 |
| L block | 292.77 / 50.33 | 313.66 / 35.65 |
| XL block | OOM / OOM | 527.67 / 59.04 |
| H block | OOM / OOM | 602.97 / 63.06 |
| B sublayer | 113.73 / 19.82 | 119.65 / 15.11 |
| L sublayer | 330.64 / 52.07 | 348.28 / 37.91 |
| XL sublayer | OOM / OOM | 573.15 / 62.07 |
| H sublayer | OOM / OOM | 682.42 / 67.18 |

## Interpretation and reproduction

Fused routing improves the measured latency and makes several larger training cases fit. OOM keeps batch 128 and has no finite speedup. Recompute remains off in the released configs; MLP is an explicit choice for the memory-constrained H/XL setting.

The earlier H OOM was sensitive to the DDP wrapper: removing DDP still OOMs with fullgraph disabled, whereas one-rank DDP fits. It cannot be attributed to the fullgraph flag alone. Actual six-rank H and XL trainer smokes also completed five steps each with MLP recompute; these are compatibility checks, not a throughput or convergence study.

The full CSV additionally contains a factored-Torch control and JiT references. Factored Torch is sometimes slower than literal Torch, so algebraic reordering alone is not claimed to always improve speed. JiT has a different P16 architecture. These auxiliary results are kept separate from the direct comparison above.

Run each implementation in a fresh process on the same idle GPU:

```bash
CUDA_VISIBLE_DEVICES=0 PYTORCH_ALLOC_CONF=expandable_segments:True \
  python -m bench.fusion_profile --size H --frequency sublayer \
  --implementation fused --phase train --checkpoint-mode mlp \
  --batch 128 --repeat 0 --output-dir ../fusion-results
```

Repeat with `--implementation literal` and repeats 1/2. Use `--phase inference` for the inference table; `--frequency block` selects block-wise routing. Checkpointing is ignored during inference.

[Aggregate CSV](data/aggregate.csv) · [Per-process CSV](data/replicates.csv) · [Paper figures/table](paper_assets/) · [Supplement notes](SUPPLEMENT.md)
