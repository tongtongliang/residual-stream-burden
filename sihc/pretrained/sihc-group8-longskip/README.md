---
tags:
- image-generation
- sihc
- research-checkpoints
---
# JiT-B/16 with a learned long skip

Full-patch embedding, velocity prediction, a learned timestep-dependent input-to-output skip. Clean time: `x_t = t*x + (1-t)*epsilon`, with t=0 noise and t=1 clean. The analytic clean-to-velocity skip coefficient is `-1/(1-t)`.

This Group 8 run is a separate archive from the earlier long-skip coefficient experiment used in Appendix B.8. Match run identity before combining records.

Sampling: EMA, Heun-50, CFG 2.9, clean-time interval [0.1, 1.0].

## Files and configuration

| Checkpoint | Epoch | Format | Architecture / status |
|---|---:|---|---|
| [checkpoints/step_00050040.pt](https://huggingface.co/TongtongLiang/sihc-group8-longskip/resolve/245c92906b898267ca26059dc83dcd15344742dd/checkpoints/step_00050040.pt) | 40 | resume | jit_b16_fullpatch_longskip |
| [checkpoints/step_00100080.pt](https://huggingface.co/TongtongLiang/sihc-group8-longskip/resolve/245c92906b898267ca26059dc83dcd15344742dd/checkpoints/step_00100080.pt) | 80 | resume | jit_b16_fullpatch_longskip |
| [checkpoints/step_00150120.pt](https://huggingface.co/TongtongLiang/sihc-group8-longskip/resolve/245c92906b898267ca26059dc83dcd15344742dd/checkpoints/step_00150120.pt) | 120 | resume | jit_b16_fullpatch_longskip |
| [checkpoints/step_00200160.pt](https://huggingface.co/TongtongLiang/sihc-group8-longskip/resolve/245c92906b898267ca26059dc83dcd15344742dd/checkpoints/step_00200160.pt) | 160 | resume | jit_b16_fullpatch_longskip |
| [checkpoints/step_00250200.pt](https://huggingface.co/TongtongLiang/sihc-group8-longskip/resolve/245c92906b898267ca26059dc83dcd15344742dd/checkpoints/step_00250200.pt) | 200 | resume | jit_b16_fullpatch_longskip |

[Machine-readable checkpoint catalog](catalog.json) records revisions, file sizes, configuration and architecture classification.

[Code and reproduction guide](https://github.com/tongtongliang/residual-stream-burden) · [Research archive](https://huggingface.co/TongtongLiang/sihc-research-checkpoints) · [Paper XL + REPA](https://huggingface.co/xiziqiao/sihc-group5-flagship-ckpt)

The DINO teacher is needed only for REPA training. Follow the model-specific loader; blockwise and sublayerwise weights are different architectures.

## Paper flagship

The selected SiHC-XL + REPA checkpoint is hosted by our coauthor: [step_00650520.pt](https://huggingface.co/xiziqiao/sihc-group5-flagship-ckpt/blob/73ad9f390370a1b626ac299089e085c374d361e7/checkpoints/step_00650520.pt). Its recorded primary-EMA Heun-50 evaluation is FID **1.71491**, IS **299.5535**, CFG 2.4, interval [0.1, 0.9], 50,000 samples. Historical checkpoints above retain their own identities and metrics.
