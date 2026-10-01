# Completed semantic-flow probe analysis

All 12 model/noise jobs completed on 2026-09-15. See [REPORT.md](REPORT.md)
for final-boundary Top1/Top5 and depth curves, and [provenance/sources.json](provenance/sources.json)
for checkpoint and source-run identifiers.

## Main results

Peak matched-noise validation Top1 (%):

| Model | Noise 25% | Noise 50% | Noise 75% |
|---|---:|---:|---:|
| DiP | 32.3700 | 34.2613 | 26.2200 |
| DeCo | 32.9307 | 35.6587 | 27.3473 |
| PixelDiT | 39.5320 | 41.1840 | 31.1433 |
| HyperDiT | 45.9687 | 45.0847 | 33.0893 |

HyperDiT > PixelDiT > DeCo > DiP at all three noise levels. HyperDiT peaks
at measured block7; its backbone has eight semantic blocks, whereas the
other models have twelve and are measured through block11. These are not
measurements of the final decoder interface. HyperDiT register tokens are
excluded. No claim of significant between-model differences follows from
one probe-training seed.

## Generation quality association

Exploratory comparison at input noise alpha=0.5, using epoch200 checkpoints:

| Model | EMA gFID | Raw probe peak Top1 | Raw probe block11 Top1 |
|---|---:|---:|---:|
| JiT-velocity | 139.82911691692595 | 6.804 | 4.559 |
| DiP | 13.305966600360875 | 34.2613 | 20.696 |
| JiT-clean | 10.192036253189201 | 35.053 | 27.539 |
| DeCo | 8.069581201761252 | 35.6587 | 23.378 |
| SiHC-P4 velocity | 8.066986277211925 | 36.183 | 27.808 |
| PixelDiT | 6.330378998614265 | 41.184 | 35.9193 |

Generation: 50k samples, EMA, Heun50, CFG2.9, guidance interval [0.1,1.0].
Probes: raw weights, null class conditioning, matched-noise validation.
SiHC probes measure the workspace, not its high-resolution residual carrier.
HyperDiT is not included in this six-model correlation calculation.

| Subset | Probe | Spearman rho | Exact two-sided p | Kendall tau | Pearson r (FID) |
|---|---|---:|---:|---:|---:|
| All six | Peak | -1.000000 | 0.002778 | -1.000000 | -0.986165 |
| All six | Block11 | -0.942857 | 0.016667 | -0.866667 | -0.887877 |
| Exclude JiT-velocity | Peak | -1.000000 | 0.016667 | -1.000000 | -0.778440 |
| Exclude JiT-velocity | Block11 | -0.900000 | 0.083333 | -0.800000 | -0.772934 |
| DiP, DeCo, PixelDiT | Peak | -1.000000 | 0.333333 | -1.000000 | -0.817748 |
| DiP, DeCo, PixelDiT | Block11 | -1.000000 | 0.333333 | -1.000000 | -0.802350 |

Exact p counts all n! permutations with abs(rho_perm) >= abs(rho_observed).
The tests are exploratory, uncorrected for multiple comparisons, and use the
displayed probe values. SiHC and DeCo differ by only 0.0026 FID: their strict
ordering is not evidence of a reproducible generation-quality difference.

Peak semantic readability strongly tracks generation ranking here, even
without the failed plain-velocity model. However, later-layer GAP readability
need not determine decoder performance: spatial and nonlinear conditioning
information can be useful without being linearly classifiable after GAP.
Architecture, decoder capacity, raw versus EMA weights, and conditioning differ
between these measurements. This is an association, not a causal test.

## Metadata and reproducibility

- `data/raw/<model>/noiseXX/protocol.json`: model, checkpoint, extraction boundaries and run protocol.
- `distributed_resume.json`: distributed execution and RNG provenance.
- `extraction_verified.json`, `compile.json`: native-boundary and compiled extraction checks.
- `validation_view0.json`: auxiliary clean-input validation, not the primary result.
- `validation_view1.json` through `validation_view3.json`: matched-noise validation views.
- `accuracy.json`, `complete.json`: aggregate metrics and completion markers.
- `data/accuracy.csv`, `accuracy_by_seed.csv`, `training.csv`: portable consolidated data.
- `provenance/*.py`: execution source snapshots; local dataset/model dependencies are not bundled.

Training uses 1,281,167 images, 40 probe epochs, global batch4096, microbatch128,
two GPUs per model/noise job, frozen BF16 compiled backbone and FP32 heads.
Validation uses 50,000 images and three matched-noise views. Input noise is
x_alpha=(1-alpha)*x_clean+alpha*epsilon, model time t=1-alpha; pixels are [-1,1].
Patch RMS normalization precedes spatial GAP. Probe heads use zero initialization,
AdamW with cosine LR0.01 to0.0001 and zero weight decay. Labels never condition
the frozen backbone. Seeds and rank-offset implementation are preserved in source
and per-job metadata. Different world sizes in older Group1 runs imply different
noise draws; test-view standard deviations are not training-run uncertainty.

The archive contains no model checkpoints, feature caches, credentials or full logs.
