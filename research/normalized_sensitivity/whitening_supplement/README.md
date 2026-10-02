# Whitening supplement: normalized representation sensitivity

Completed on 2026-09-19. This extends [experiment 12](../README.md) with the epoch-200 whitening JiT-B clean and velocity checkpoints. All six tasks completed: two models × three clean fractions, with 24 observed residual events per task. This is forward-only analysis; no training or checkpoint changes.

## Paper-facing entry points

- [REPORT.md](REPORT.md): exact protocol, absolute B/D and W/D, pointwise-ratio summaries, interpretation and limitations.
- [summary.csv](summary.csv): six depth summaries. `mean_point_ratio` is the mean of 24 corrected B/W ratios; it is **not** `ratio_of_mean_traces`.
- [per_event.csv](per_event.csv): all 144 records, retaining B_raw, B_corrected, W, dimensions and both ratios without rounding.
- [figures/](figures/): PNG and vector PDF depth curves for corrected B/W and separate B/D, W/D.
- [historical_plain_comparison.csv](historical_plain_comparison.csv): references to the original unwhitened results, not new runs. The old environment differs; do not treat this as a same-environment causal ablation.
- [data/](data/): each completed task's manifest, metrics CSV/JSON, completion record and compiled/eager reduction check.
- [audit.json](audit.json): six-task/144-record integrity and estimator checks.
- [samples.npz](samples.npz): the exact 2048 ordered sample IDs and original labels, identical to the parent experiment. Conditioning uses null class 1000, not those labels.
- [geometry_and_source_validation.json](geometry_and_source_validation.json): clean/velocity checkpoint whitening transforms are tensor-equal; original streaming-statistic source was copied without changes.
- [provenance/](provenance/): collection source, own analysis logs, package metadata and original source-path/size inventory. CSV line endings are normalized to LF without changing any numeric fields. No file hashes were computed.

## Protocol

Raw `model` weights at step250200, 2048 ImageNet training images ×128 deterministic CUDA Gaussian draws, `t_clean = 0.25, 0.50, 0.75`, no flip, null class1000. Uint8 inputs become [-1,1], then the checkpoint-owned FP32 whitening transform is applied. Noise is isotropic **in whitened coordinates**: `z=t_clean*whiten(x)+(1-t_clean)*epsilon`. The two models use the same image IDs and per-image noise seeds. Whitening is never refitted.

Observe each block's `norm1`/`norm2` input: 24 full256×768 spatial residuals, no GAP, no head. Apply non-affine per-patch RMS with eps1e-6. Use the archived N−1/K−1 variance estimators and `B_corrected=B_raw-W/128`, without clipping negative estimates. Preserve raw traces and dimension-normalized traces. B is between-image variation, not pure semantics; low W alone does not establish useful representations. No confidence intervals or significance claims were added.

The source checkpoints remain on Hugging Face in [TongtongLiang/sihc-research-checkpoints](https://huggingface.co/TongtongLiang/sihc-research-checkpoints/tree/d4601d0fc44f7f552fd446376707c7f733149d3d/resume/group6), under `g6_jit_b16_whiten_{clean,velocity}_seed0/step_00250200.pt`. They are not included here.

## Reproduce figures from cached data

From the repository root, after installing `requirements-viz.txt`:

```bash
python research/normalized_sensitivity/whitening_supplement/render.py
```

The script finds CSVs relative to itself, works from any working directory, and requires only matplotlib for cached rendering. It uses no GPU, checkpoint, dataset download or network. `--output-dir /tmp/whitening-sensitivity-figures` writes elsewhere.

`provenance/code/` and `provenance/run.sh` are immutable server execution snapshots, **not portable launchers**. Their absolute paths document the original run. Use `render.py` for paper figures; do not execute the archived collector merely to inspect this package. The manuscript and its existing scientific claims are not modified by this supplement.
