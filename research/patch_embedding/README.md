# Patch PCA, embedding alignment and directional gain

This experiment compares the covariance spectrum of clean 16×16 RGB patches with learned patch-embedding matrices from raw epoch-200 B-size checkpoints. ImageNet-256 PCA uses 100,000 training images and all 25.6 million patches, giving 768 coordinates per patch.

## Results

Eight principal directions explain 90% of clean-patch variance. The velocity target requires 668 directions for the same fraction. These are **target covariance** statistics. The embedding analysis separately measures the input map `W`, its Gram matrix `WᵀW`, alignment with clean PCA, and directional gain `g_i = ||W v_i||²`.

| Model | Gain assigned to leading 64 clean-PCA directions |
|---|---:|
| JiT-B clean | 61.48% |
| JiT-B velocity | 9.99% |
| DeCo-B semantic stream | 66.44% |
| PixelDiT-B semantic stream | 75.52% |
| DiP-B semantic stream | 61.05% |

All embedding measurements above use **raw weights**, not EMA. See [metric definitions](METRICS.md) and [full result tables](RESULTS.md). Whitened-coordinate comparisons are in [the whitening experiment](../whitening/README.md).

## Data and figures

| File | Contents |
|---|---|
| [raw_pixel_patch_pca.npz](figure_cache/data/raw_pixel_patch_pca.npz) | Clean-patch covariance, eigenvalues and PCA basis |
| [patch_embedding_matrices.npz](figure_cache/data/patch_embedding_matrices.npz) | Raw embedding matrices and Gram matrices |
| [rankwise_embedding_metrics.csv](figure_cache/data/rankwise_embedding_metrics.csv) | Direction-by-direction gains and spectrum statistics |
| [gain_rank_bins.csv](figure_cache/data/gain_rank_bins.csv) | Summed gain shares in specified PCA rank ranges |
| [embedding_topk_pca_alignment.csv](figure_cache/data/embedding_topk_pca_alignment.csv) | Leading eigenspace overlaps |

From the repository root:

```bash
python research/plotting/render_embedding_filtering.py
python research/plotting/render_target_alignment_gain.py
```

The first renders raw clean/velocity and decoupled embeddings; the second compares raw and whitened clean/velocity filtering. Both read cached measurements and write under `research/figures/`.
