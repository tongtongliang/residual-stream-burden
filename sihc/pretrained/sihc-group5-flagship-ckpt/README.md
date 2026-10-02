---
tags:
- image-generation
- sihc
- research-checkpoints
---
# SiHC-XL: sublayer flagship with REPA

40 blocks, width 1024, five stages of eight blocks, direct 4×4 subpatch embedding, 16×16 workspace, 32 context tokens from zero-based block index 8. Velocity prediction with DINOv3-L/16 REPA at block 8.

Sampling: primary EMA, Heun-50, CFG 2.4 in clean-time interval [0.1, 0.9], balanced 50K ImageNet-256 samples and JiT reference statistics.

This collection retains earlier epoch-400/480 copies. The selected paper checkpoint is linked in the Paper flagship section below.

`checkpoints/` contains full resume files. `evaluation_50k/` contains existing evaluation records. `HANDOFF.md` and orphan logs are historical operational records.

## Files and configuration

| Checkpoint | Epoch | Format | Architecture / status |
|---|---:|---|---|
| [checkpoints/step_00500400.pt](https://huggingface.co/TongtongLiang/sihc-group5-flagship-ckpt/resolve/f9cc4a4dd97cb4599e2470b3a7f383f97a1e67ee/checkpoints/step_00500400.pt) | 400 | resume | sihc_sublayer_5x8_d1024_b4_ctx32s8 |
| [checkpoints/step_00600480.pt](https://huggingface.co/TongtongLiang/sihc-group5-flagship-ckpt/resolve/f9cc4a4dd97cb4599e2470b3a7f383f97a1e67ee/checkpoints/step_00600480.pt) | 480 | resume | sihc_sublayer_5x8_d1024_b4_ctx32s8 |

[Machine-readable checkpoint catalog](catalog.json) records revisions, file sizes, configuration and architecture classification.

[Code and reproduction guide](https://github.com/tongtongliang/residual-stream-burden) · [Research archive](https://huggingface.co/TongtongLiang/sihc-research-checkpoints) · [Paper XL + REPA](https://huggingface.co/xiziqiao/sihc-group5-flagship-ckpt)

The DINO teacher is needed only for REPA training. Follow the model-specific loader; blockwise and sublayerwise weights are different architectures.

## Paper flagship

The selected SiHC-XL + REPA checkpoint is hosted by our coauthor: [step_00650520.pt](https://huggingface.co/xiziqiao/sihc-group5-flagship-ckpt/blob/73ad9f390370a1b626ac299089e085c374d361e7/checkpoints/step_00650520.pt). Its recorded primary-EMA Heun-50 evaluation is FID **1.71491**, IS **299.5535**, CFG 2.4, interval [0.1, 0.9], 50,000 samples. Historical checkpoints above retain their own identities and metrics.
