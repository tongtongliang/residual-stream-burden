# Noisy Input Patch Spectrum Results

## Scope

This experiment measures source-input patches rather than diffusion-transformer
hidden tokens. It uses the fixed class-balanced ImageNet train-100k subset and
all 256 patches per image, giving 25,600,000 vectors per condition.

The input path is

`x_q = (1 - q) x + q epsilon`, where `epsilon ~ N(0, I)` and
`q = 0.0, 0.1, ..., 0.9`.

`q` is the interpolation coefficient, not the noise fraction of total
variance. The latter depends on the clean representation scale.

The three direct input spaces are:

- pixel-space DiT input: `16 x 16 x 3 = 768` RGB patches in `[-1, 1]`;
- SiT-XL/2 input: `2 x 2 x 4 = 16` SD-VAE latent patches, including the
  training-time `0.18215` latent scale;
- RAE input: 768-dimensional released position-normalized DINOv2-B tokens.

An additional test corrupts RGB before the DINO encoder, then measures both the
affine-free final-LayerNorm tokens and RAE's released position-wise variance
normalization. RGB noise is not clipped before DINO preprocessing.

## Direct input noise

| Source | Metric | q=0.0 | q=0.3 | q=0.6 | q=0.9 |
|---|---|---:|---:|---:|---:|
| RGB patch | effective rank | 4.77 | 55.24 | 565.03 | 766.58 |
| RGB patch | stable rank | 1.40 | 2.24 | 11.52 | 250.47 |
| RGB patch | r90 | 8 | 563 | 681 | 691 |
| RGB patch | PCA-1 share | 71.24% | 44.66% | 8.68% | 0.40% |
| VAE patch | effective rank | 9.09 | 11.07 | 15.34 | 16.00 |
| VAE patch | stable rank | 2.60 | 3.16 | 7.24 | 15.34 |
| VAE patch | r90 | 10 | 12 | 15 | 15 |
| VAE patch | PCA-1 share | 38.42% | 31.67% | 13.82% | 6.52% |
| DINO+RAE token | effective rank | 579.61 | 624.27 | 742.64 | 767.94 |
| DINO+RAE token | stable rank | 66.84 | 77.52 | 178.57 | 678.52 |
| DINO+RAE token | r90 | 563 | 598 | 670 | 691 |
| DINO+RAE token | PCA-1 share | 1.50% | 1.29% | 0.56% | 0.15% |

These increases are exactly the expected white-noise effect. For independent
noise,

`Sigma_q = (1 - q)^2 Sigma_0 + q^2 I`.

The measured and predicted curves agree to within 0.014 effective-rank
components, 0.19 stable-rank components, `1` r90 component, `9e-6` absolute
PCA-1 share, and `1.5e-4` relative total variance. Direct noise fills every
coordinate and flattens the spectrum; it does not create information.

Total variance is U-shaped. Signal variance shrinks as `(1-q)^2`, while noise
variance grows as `q^2 d`. Its minimum is at `q=0.2` for RGB, `q=0.4` for the
VAE patch, and `q=0.5` for the RAE-normalized DINO token.

## Noisy image through DINOv2

| DINO output | Metric | q=0.0 | q=0.3 | q=0.6 | q=0.9 |
|---|---|---:|---:|---:|---:|
| final LN | effective rank | 493.04 | 450.53 | 220.42 | 45.68 |
| final LN | stable rank | 37.75 | 30.58 | 7.98 | 3.93 |
| final LN | r90 | 510 | 502 | 419 | 115 |
| final LN | PCA-1 share | 2.65% | 3.27% | 12.54% | 25.46% |
| DINO+RAE norm | effective rank | 579.61 | 540.16 | 257.93 | 48.42 |
| DINO+RAE norm | stable rank | 66.84 | 42.61 | 8.16 | 4.57 |
| DINO+RAE norm | r90 | 563 | 557 | 456 | 106 |
| DINO+RAE norm | PCA-1 share | 1.50% | 2.35% | 12.26% | 21.89% |

This trend is the opposite of direct token noise. Severe RGB corruption makes
DINO patch outputs more concentrated rather than more isotropic. From clean to
`q=0.9`, final-LN effective rank falls 90.7%, stable rank falls 89.6%, and
centered total variance falls 80.7%.

Affine-free final LayerNorm keeps the per-token second moment at 768. The
centered-variance reduction is instead explained by a growing shared mean:
the squared global mean rises from 21.2% to 84.8% of the token second moment.
Tokens retain their normalized radius but increasingly point toward a common
feature profile. RAE's fixed position-wise rescaling broadens clean DINO
features, but it does not prevent this out-of-distribution collapse.

LayerNorm also places every final-LN token in a channel-sum-zero hyperplane.
The corresponding zero covariance eigenvalue appears as a float32 numerical
residual as low as `-2.51e-8`; it is clipped to zero for spectrum metrics.

## Interpretation

The noise location changes the scientific meaning:

- direct VAE or RAE-token noise matches the actual denoiser input path and
  mechanically raises PCA rank;
- encoding a noisy RGB image is a DINO robustness experiment and strongly
  lowers rank;
- neither high rank from white noise nor low rank from encoder collapse alone
  measures semantic or spatial information.

This is a useful control for residual-stream PCA: any rank claim under input
noise must be compared with the analytic `q^2 I` baseline and must state
whether corruption occurs before or after a representation encoder.

## Artifacts

- Executed notebook: `notebooks/noisy_input_patch_spectrum.ipynb`
- Measured metrics: `outputs/imagenet_train100k_input_patch_noise_sweep/metrics.csv`
- Full spectra: `outputs/imagenet_train100k_input_patch_noise_sweep/spectra/`
- Provenance: `outputs/imagenet_train100k_input_patch_noise_sweep/provenance.json`
- Analytic baseline: `analytic_direct_noise_metrics.csv`
- DINO mean-direction diagnostic: `dino_mean_direction_metrics.csv`
- Reproducible configuration: `config.json`

