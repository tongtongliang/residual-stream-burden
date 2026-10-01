# Input Patch Spectrum Results

## Setup

The fixed ImageNet train-100k subset contributes all 256 patches per image,
giving 25,600,000 observations per representation. RGB uses `16 x 16 x 3`
patches, SD-VAE latents use `2 x 2 x 4` patches, and RAE uses individual
768-dimensional DINOv2-B tokens. Covariances are globally centered across the
bag of patches. Per-token RMS measurements audit patch-radius effects.

## Main measurements

These rows describe **patch-embedder fan-in**, before learned projection and
positional embedding. They are not transformer block hidden states.

| Input | Dimension | Effective rank | Eff./dim | Stable rank | r90 | PCA-1 share |
|---|---:|---:|---:|---:|---:|---:|
| RGB 16x16 raw | 768 | 4.77 | 0.62% | 1.40 | 8 | 71.24% |
| RGB 16x16 RMS | 768 | 8.43 | 1.10% | 1.73 | 19 | 57.78% |
| SD-VAE 2x2 raw | 16 | 9.09 | 56.81% | 2.60 | 10 | 38.42% |
| SD-VAE 2x2 RMS | 16 | 10.97 | 68.59% | 3.48 | 11 | 28.74% |
| DINOv2-B final LN | 768 | 493.04 | 64.20% | 37.75 | 510 | 2.65% |
| DINO + RAE position norm | 768 | 579.61 | 75.47% | 66.84 | 563 | 1.50% |
| DINO + RAE norm + RMS | 768 | 578.66 | 75.35% | 64.43 | 564 | 1.55% |

The EMA and MSE SD-VAE checkpoints share the same encoder; their cached latent
patch spectra and direct sample encodings are identical. Their differing model
parameters are confined to decoder and post-quantization modules.

## Exact SiT patch definition

SiT-XL/2 receives a `4 x 32 x 32` VAE latent. Its `PatchEmbed` has kernel and
stride 2, so each learned projection reads exactly `2 x 2 x 4 = 16` scalars and
produces one 1152-dimensional token. The resulting grid is `16 x 16`, or 256
tokens. The `/2` refers to latent-space patch size, corresponding spatially to
roughly `16 x 16` RGB pixels after the VAE's factor-eight downsampling.

The first transformer block sees the projection plus fixed positional embedding,
not the 16-dimensional source vector:

| Model | Patch-embedder source | Source eff. rank | Block-1 width | Block-1 raw eff. rank | Block-1 RMS eff. rank | Block-1 raw stable | Block-1 raw r90 |
|---|---|---:|---:|---:|---:|---:|---:|
| SiT-XL/2 | VAE `2x2x4` | 9.09 / 16 | 1152 | 3.44 | 6.84 | 1.37 | 5 |
| REPA-XL/2 | VAE `2x2x4` | 9.09 / 16 | 1152 | 7.90 | 11.66 | 1.99 | 12 |
| PixelDiT | RGB `16x16x3` | 4.77 / 768 | 1152 | 6.04 | 6.89 | 2.00 | 6 |
| RAE DiT-DH-XL | DINO token | 579.61 / 768 | 1152 | 362.13 | 390.29 | 21.08 | 435 |

Therefore `9.09/16` answers how the local VAE source vectors occupy their native
space. The residual-stream claim that SiT starts highly anisotropic is instead
the `block01_fanin` result, `3.44/1152`. The main residual-stream experiment has
always measured the latter.

## Interpretation

RGB patches are extremely anisotropic because shared DC, color, and smooth
low-frequency directions dominate natural-image patch covariance. This remains
true after per-token RMS control.

Pre-embedding VAE patches are not exceptionally anisotropic relative to their native 16
dimensions: raw effective rank occupies 56.8% of the available space. They are
nevertheless an intrinsically narrow carrier for a width-1152 transformer. The
very low SiT and REPA block-1 ranks should therefore be described as the geometry
of the embedded residual input, not as a 768-dimensional VAE patch collapse.

DINO tokens are broad. RAE's released per-position variance normalization
further raises effective rank from 493.04 to 579.61 and halves PCA-1 share from
2.65% to 1.50%. Additional per-token RMS has negligible effect, confirming that
the flat spectrum is directional rather than a token-norm artifact.

## RAE and REPA

RAE directly supplies normalized DINO tokens at depth zero. The token effective
rank is 579.61/768; after patch embedding and positional coordinates, block-1
fan-in remains broad at 362.13/1152.

REPA does not inject teacher activations during inference. A training-only
projector after block 8 receives a normalized patch-alignment loss. The block
receiving that gradient transforms its residual from effective rank 42.35 at
block-8 fan-in to 469.86 at block-9 fan-in. Total variance falls from 13,551 to
7,505 while stable rank rises from 3.56 to 23.69. The per-token RMS transition
is 60.34 to 475.41, so this is learned angular flattening rather than energy
growth.

This supports a functional analogy: RAE starts from a representation-rich
workspace; REPA trains its early backbone to construct one. It does not prove
that isotropy causes fast convergence. Representation content, spatial
relations, auxiliary-gradient quality, latent dimension, decoder, and network
architecture remain confounded.

## Causal follow-up

The direct test is a matched spectral-tempering sweep of the same DINO tokens or
REPA teacher targets while preserving eigenspaces and changing only eigenvalue
flatness. Early loss, FID, gradient conditioning, and depthwise spectra should be
tracked across checkpoints. This would distinguish covariance conditioning from
the semantic and spatial information contained in the teacher directions.

## Artifacts

- Metrics: `outputs/imagenet_train100k_input_patch_spectrum_v2/metrics.csv`
- Spectra: `outputs/imagenet_train100k_input_patch_spectrum_v2/spectra/`
- Notebook: `notebooks/input_patch_spectrum_and_representation_alignment.ipynb`
- Reproducible configuration: `config.json`

## Noise sweep follow-up

The controlled `q=0.0, 0.1, ..., 0.9` source-noise sweep and the additional
noisy-image-through-DINO experiment are in `noise_sweep/`. Direct RGB, VAE, and
DINO-token noise follows the analytic white-noise covariance prediction and
flattens the spectrum. In contrast, DINO outputs from noisy RGB images become
strongly concentrated. See `noise_sweep/RESULTS.md` and
`noise_sweep/notebooks/noisy_input_patch_spectrum.ipynb`.
