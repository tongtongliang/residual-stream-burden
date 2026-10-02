# DINOv2-B patch spectrum and SNR

## Scope

ImageNet train-100k balanced; every spatial patch token; globally centered
bag-of-patches covariance. DINOv2-B/14 tokens are reshaped to a 16×16 grid
(256 patches/image → 25,600,000 vectors). Pixel RGB `16×16×3` and SD-VAE
`2×2×4` rows are retained in the shared metrics table for contrast.

This is **input / latent patch geometry**, not a DiT residual-stream
measurement.

## Clean spectrum summary

| Variant | dim | ER | Stable | r90 | r95 | r99 | PCA-1 | top-128 share | Tr | Tr/d |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| DINOv2-B final LN | 768 | 493.0 | 37.7 | 510 | 597 | 698 | 2.65% | 47.0% | 605.0 | 0.79 |
| DINO + RAE position norm | 768 | **579.6** | **66.8** | **563** | 649 | 735 | **1.50%** | **41.8%** | 793.6 | **1.03** |
| DINO + RAE norm + RMS | 768 | 578.7 | 64.4 | 564 | 649 | 735 | 1.55% | 41.9% | 610.4 | 0.79 |
| RGB p16 raw (contrast) | 768 | 4.77 | 1.40 | 8 | 26 | 151 | 71.2% | 98.7% | 236.3 | 0.31 |

RAE’s released per-position variance normalization raises effective rank from
493 → 580 and sets mean per-coordinate variance ≈ 1, matching unit Gaussian
training noise. Extra per-token RMS barely changes the directional spectrum.

## Per-mode SNR (RAE-normalized DINO)

Noisy input \(z_q=(1-q)x+q\varepsilon\) in the clean eigenbasis:

\[
\mathrm{SNR}_i(q)=\frac{(1-q)^2\lambda_i}{q^2}.
\]

Clear directions with \(\mathrm{SNR}_i\ge 1\) (from `reports/PER_MODE_SNR.md`):

| q | Pixel clear | VAE clear | **RAE-DINO clear** |
|---:|---:|---:|---:|
| 0.10 | 214 / 768 | 16 / 16 | **767 / 768** |
| 0.20 | 79 | 16 | **767** |
| 0.30 | 34 | 13 | **763** |
| 0.50 | 11 | 3 | **265** |

At training-relevant mid noise, nearly the full ambient dimension of the RAE
latent remains clear, while pixel patches retain only a thin high-SNR ridge.
Full tables, scalar SNR, and critical-\(q\) curves are in
`reports/PER_MODE_SNR.md` and `data/snr/`.

## Direct noise sweep (input patches)

Under the same linear mixing on stored latents / pixels, DINO+RAE tokens stay
broad while RGB patches inflate from a low-rank clean state. See
`reports/input_noise_sweep_RESULTS.md`, `data/noise_sweep/`, and figures
`direct_noise_rank_metrics.png`, `dino_direct_vs_noisy_image.png`.

## Velocity-target identity

For independent \(\varepsilon\sim\mathcal N(0,I)\),

\[
\mathrm{Cov}(\varepsilon-x)=\mathrm{Cov}(x)+I,
\]

so RAE-DINO velocity eigenvalues are \(\lambda_i+1\). Measured clean → velocity
ranks at low noise remain mild (~1.2× ER), unlike pixel velocity (~56×). Target-
geometry rows live in `data/snr/dino_rae_patch_eigenvalues_*.csv` and the parent
target-geometry package.

## Files

| Path | Contents |
|---|---|
| `data/spectra/dino_*.npz` | Eigenvalues, explained-variance ratios, mean, count |
| `data/spectra/dinov2b_patch_eigenvalues.csv` | Same spectra as a single CSV |
| `data/spectra/dinov2b_patch_covariance_moments.npz` | Centered `mean` + `m2` for three DINO variants (eigenvector recovery) |
| `data/metrics/input_patch_spectrum_metrics.csv` | Rank / variance table including pixel & VAE contrasts |
| `data/snr/` | Per-mode SNR counts/modes; clean and noisy-grid DINO eigenvalues |
| `data/noise_sweep/` | Direct input noise-sweep metrics (DINO-filtered + full) |
| `reports/` | Copied RESULTS / PER_MODE_SNR markdown sources |
| `figures/` | Cumulative spectra, SNR panels, noise-sweep plots |
| `provenance/` | Original resolved config, provenance, spectrum summary JSON |

## Provenance

- Population: ImageNet train-100k balanced, 100k images × 256 patches.
- Encoder path / RAE stats: see `provenance/input_patch_spectrum_resolved_config.json`.
- Local source tree: `residual_stream_research/pca/input_patch_spectrum/` and
  `pca/target_geometry/` (per-mode SNR).
