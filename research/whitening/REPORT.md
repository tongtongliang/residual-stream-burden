# Whitening patch embedding spectra — epoch 200

## Sources and definitions

Group 6 g6_jit_b16_whiten_clean_seed0 and g6_jit_b16_whiten_velocity_seed0,
checkpoints/step_00250200.pt (verified step250200), raw model and primary ema.
Absolute checkpoint paths and model kwargs are in provenance/spectrum_sources.json. Weight key:
x_embedder.proj.weight, flattened Conv2d [768,3,16,16] to [768,768], CHW order.
CPU FP64 SVD; no model inference, new FID evaluation, training or W&B changes.

Historical pixel controls reuse the existing epoch200 raw-state compact_arrays.npz
and embedding_alignment_metrics.csv, not newly unpickled legacy checkpoints.
Their original checkpoint provenance is in the historical report. Bias does not
enter this linear-weight spectrum. Saved PCA rows/scales come from each checkpoint.

Spectrum means singular values sigma_i; Gram energy is sigma_i squared.
ER=exp(-sum p_i log p_i), p_i=sigma_i^2/sum sigma^2.
r90 is the minimum number of singular directions retaining90% Gram energy.
This is a weight spectrum, NOT residual activation PCA or semantic information.

## Native input-coordinate comparison (raw weights)

| Model | Gram ER | Stable rank | r90 | Top64 Gram energy | Frobenius norm |
|---|---:|---:|---:|---:|---:|
| Historical pixel clean |175.06|28.51|192|62.41%|57.15|
| Historical pixel velocity |696.24|195.77|616|17.12%|99.26|
| Whitening clean |738.07|270.75|647|13.12%|92.05|
| Whitening velocity |747.69|428.37|656|11.31%|90.95|

Confirmed: whitening clean no longer has the highly concentrated spectrum of
pixel clean. Both whitening models distribute most weight energy broadly across
input directions. Clean retains a larger leading singular value (5.59 vs4.39),
so they are similar but not identical. Small tail singular values remain;
"broad spectrum" does not mean perfectly orthogonal/well-conditioned weights.
Primary EMA gives ER740.16/747.82 and r90=649/657, corroborating the raw result.

## Directional gains and RGB-effective map

Let B be saved PCA rows and D=diag(sqrt(lambda)). Whitened column input is
z=D^-1 B(x-mu). Effective RGB weight is A=W D^-1 B.
This A has Gram ER327.27 (clean) /319.03 (velocity), but is NOT directly comparable
to W as a claim of learned selectivity: D^-1 itself strongly amplifies low-variance
RGB directions. Bottom100 RGB PCA directions receive54.19%/55.34% of A's gain.

In whitening coordinates, weight gain assigned to the first64 PCA coordinates
is10.36% (clean) /9.09% (velocity); equal allocation would be8.33%.
Bottom100 receive10.44%/11.74%; equal allocation would be13.02%.
The extreme top-PC preference of pixel clean (top64 gain61.48%) is absent.
Directional gain shares and singular-energy shares are different statistics.

## Interpretation and limits

Supported: the concentrated clean stem observed in pixel space is not invariant
to invertible whitening. Clean targets in whitened coordinates have identity
covariance under the fitted estimator; the learned stem broadly retains these
coordinates, looking much closer to the velocity stem than pixel clean did.

This supports a representation-dependent target-geometry interpretation, not
"clean is intrinsically easy regardless of coordinates." It does NOT establish
that this stem spectrum causes the poor FID, prove a causal transport burden
mechanism, or demonstrate how later blocks represent noise. Whitening also
changes the loss metric and RGB noise covariance, so it is not a pure target-only
intervention. Native-coordinate spectra are descriptive, not coordinate-invariant.

## Files

- figures/spectrum.png: native singular spectrum (raw/EMA), cumulative Gram energy with
  historical controls, and RGB-effective spectrum.
- data/spectrum_metrics.csv: full raw/EMA metrics in both coordinate systems.
- data/spectra.npz: all singular values and directional gains.
- provenance/spectrum_sources.json: exact new checkpoint and historical artifact provenance.
- render.py: portable cached-data renderer; original weights and whitening transform
  are in data/embedding_matrices.npz for independent numerical analysis.
