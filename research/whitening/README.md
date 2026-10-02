# Whitening: clean and velocity prediction

Clean 16×16 RGB patches are whitened by a full-rank invertible linear map fitted on 100,000 ImageNet training images. All 768 coordinates are retained. Training uses isotropic corruption in whitened coordinates, and generated images are mapped back to RGB for evaluation.

## Generation results

| Target | Raw-pixel FID | Whitened-patch FID | Whitened-patch IS |
|---|---:|---:|---:|
| Clean | 10.19 | 132.95 | 9.54 |
| Velocity | 139.83 | 212.54 | 3.35 |

These are epoch-200 results. Evaluation uses primary EMA (0.9999), 50,000 balanced ImageNet-256 samples, JiT reference statistics, Heun-50, CFG 2.9 in clean-time interval [0.1, 1.0], and seed 12345. Training uses JiT-B/16 with a full-rank embedding, no bottleneck or context tokens, AdamW at 2e-4 and global batch 1024. Time follows `x_t = t*x + (1-t)*epsilon`, with t=1 clean.

## Learned embedding results

In raw pixel space, the clean-prediction embedding assigns 61.48% of its gain to the leading 64 clean-PCA directions, versus 9.99% for velocity. After whitening, the corresponding shares are **10.36%** and **9.09%**, measured in native whitened coordinates ordered by the original clean PCA. Raw-weight Gram effective ranks are **738.07** and **747.69**. The strong preference for leading clean directions observed in raw clean prediction disappears.

Target covariance, embedding Gram energy and directional gain are distinct measurements; [REPORT.md](REPORT.md) records the definitions and coordinate transformations.

## Data and reproduction

- [evaluation_history.csv](data/evaluation_history.csv): recorded FID/IS at epochs 40, 80, 120, 160 and 200.
- [run_metadata.json](data/run_metadata.json): checkpoint configuration and evaluation protocol.
- [embedding_matrices.npz](data/embedding_matrices.npz): raw/EMA embeddings and the checkpoint-owned whitening basis, scales and means.
- [spectra.npz](data/spectra.npz) and [spectrum_metrics.csv](data/spectrum_metrics.csv): singular spectra, gain shares and rank statistics.
- `provenance/`: original measurement and PCA-estimator provenance.

```bash
python research/whitening/render.py
python research/plotting/render_geometry.py
python research/plotting/render_target_alignment_gain.py
```

These commands render cached results without training or checkpoint inference. `render.py` writes local `figures/`; the shared renderers write under `research/figures/`.
