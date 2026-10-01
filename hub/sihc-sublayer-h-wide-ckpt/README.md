---
tags:
- image-generation
- sihc
- research-checkpoints
---
# SiHC-H: sublayer scaling checkpoint

32 blocks, width 1280, stages 8+12+12, direct 4×4 subpatch embedding, 32 context tokens from zero-based block index 8. Velocity prediction, no REPA.

The available checkpoint is epoch 80. Evaluation records in this repository must be associated with their recorded steps; an epoch-40 metric is not an epoch-80 measurement. This snapshot is not the paper's epoch-200 H scaling checkpoint.

## Files and configuration

| Checkpoint | Epoch | Format | Architecture / status |
|---|---:|---|---|
| [checkpoints/step_00100080.pt](checkpoints/step_00100080.pt) | 80 | resume | sihc_sublayer_8x12x12_d1280_b4_ctx32s8 |

[Machine-readable checkpoint catalog](catalog.json) records revisions, file sizes, configuration and architecture classification.

[Code and reproduction guide](https://github.com/tongtongliang/residual-stream-burden) · [Research archive](https://huggingface.co/TongtongLiang/sihc-research-checkpoints) · [Paper flagship XL](https://huggingface.co/xiziqiao/sihc-group5-flagship-ckpt)

The DINO teacher is needed only for REPA training. Follow the model-specific loader; blockwise and sublayerwise weights are different architectures.

## Paper flagship

The selected SiHC-XL + REPA checkpoint is hosted by our coauthor: [step_00650520.pt](https://huggingface.co/xiziqiao/sihc-group5-flagship-ckpt/blob/73ad9f390370a1b626ac299089e085c374d361e7/checkpoints/step_00650520.pt). Its recorded primary-EMA Heun-50 evaluation is FID **1.71491**, IS **299.5535**, CFG 2.4, interval [0.1, 0.9], 50,000 samples. Historical checkpoints above retain their own identities and metrics.
