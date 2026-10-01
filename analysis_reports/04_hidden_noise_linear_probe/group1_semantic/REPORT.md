# Group1 semantic-flow GAP probes

Completed 12/12 model-noise jobs. Only completed and validated results appear below.

Four raw epoch200 backbones: DiP, PixelDiT, DeCo (12 semantic blocks), HyperDiT (8 semantic blocks). Spatial256 tokens only, patch RMS eps1e-6 then GAP. HyperDiT registers are excluded. Boundaries after blocks1-11 or1-7 respectively, before the next semantic block; no decoder/fine-stream output.

Protocol: frozen BF16 backbone, compile default, FP32 zero-initialized heads, AdamW .01 to .0001 cosine, wd0, global4096/micro128, 40 epochs. Same ImageNet no-flip crop protocol. Input noise25/50/75%; pixel [-1,1], t=1-alpha. Null class1000; target labels never condition the backbone.

Validation: 50k images, 3 matched-noise views averaged; auxiliary clean-input separate. Two GPUs per job; rank-offset RNG follows the previous Group1 distributed implementation. Noise draws are paired across these four models, not bitwise identical to older runs with different world sizes. Noise-seed SD is not independent training-run uncertainty.

| Model | Noise | Last measured block | Top1 | Top5 | Peak Top1 | Peak block |
|---|---:|---:|---:|---:|---:|---:|
| deco_b_velocity | 0.25 | 11 | 22.0580 | 41.7640 | 32.9307 | 7 |
| deco_b_velocity | 0.5 | 11 | 23.3780 | 44.1540 | 35.6587 | 7 |
| deco_b_velocity | 0.75 | 11 | 19.4467 | 39.0120 | 27.3473 | 7 |
| dip_b_velocity | 0.25 | 11 | 19.6580 | 38.4240 | 32.3700 | 7 |
| dip_b_velocity | 0.5 | 11 | 20.6960 | 40.4660 | 34.2613 | 7 |
| dip_b_velocity | 0.75 | 11 | 17.6013 | 36.3060 | 26.2200 | 8 |
| hyperdit_b_velocity | 0.25 | 7 | 45.9687 | 69.9893 | 45.9687 | 7 |
| hyperdit_b_velocity | 0.5 | 7 | 45.0847 | 69.4953 | 45.0847 | 7 |
| hyperdit_b_velocity | 0.75 | 7 | 33.0893 | 56.4060 | 33.0893 | 7 |
| pixeldit_b_velocity | 0.25 | 11 | 34.5973 | 57.8273 | 39.5320 | 9 |
| pixeldit_b_velocity | 0.5 | 11 | 35.9193 | 59.9713 | 41.1840 | 9 |
| pixeldit_b_velocity | 0.75 | 11 | 28.2033 | 50.6753 | 31.1433 | 9 |

![top1](figures/top1.png)

![top5](figures/top5.png)

Raw protocols, per-view values, training histories and boundary/compile checks are in data/. No claims of statistically significant model differences are made from a single probe training seed.
