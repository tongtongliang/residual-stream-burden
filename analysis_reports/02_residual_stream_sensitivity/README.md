# Part 2: Residual-Stream Noise Sensitivity

This folder packages the raw epoch-200, null-class sensitivity analysis for JiT-B clean/velocity and the DeCO-B, PixelDiT-B, DiP-B, and HyperDiT-B semantic streams.

Protocol: 8,192 ImageNet training images balanced over 32 classes; 128 deterministic Gaussian noises per image; full `256 x 768` spatial residual; identical samples and seeds across models; fan-in taps immediately before each attention/MLP AdaLN. HyperDiT has eight semantic blocks and therefore 16 points; its 256 register tokens are excluded. The other models have 12 blocks and 24 points.

- `METRICS.md`: exact estimators and interpretation limits.
- `RESULTS.md`: paper-facing absolute and ratio tables.
- `residual_stream_sensitivity.ipynb`: one plot per metric and noise level.
- `data/`: complete depthwise, bootstrap, paired, and depth-mean CSVs.

