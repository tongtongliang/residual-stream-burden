# Part 1: Patch-Embedding Matrix Analysis

This folder is the self-contained paper-analysis package for clean RGB p16 PCA and six raw epoch-200 B-size patch embeddings.

- `METRICS.md`: estimator and metric definitions.
- `RESULTS.md`: paper-facing numerical tables and bounded interpretation.
- `patch_embedding_matrix_analysis.ipynb`: one-figure-at-a-time visual report.
- `data/raw_pixel_patch_pca.npz`: registered 100k-image RGB patch PCA.
- `data/patch_embedding_matrices.npz`: six raw `W` matrices and six `W^T W` matrices.
- `data/*.csv`: complete scalar, top-k, and rankwise values.
- `figures/directional_overlap_all_models.png`: shared-scale six-panel log-overlap heatmap.
- `data/gain_rank_bins.csv`: top/bottom rank-bin gain shares used by the grouped bar chart.
- `data/raw_patch_noise_spectra.csv`: all 768 analytic eigenvalues for the five displayed noise levels.
- `HYPERDIT_SMALL_EMBEDDER.md`: separate P8 fine-stream spectrum and P8-PCA alignment analysis.

No inference was run for this organization step; values reproduce the previously audited analysis.
