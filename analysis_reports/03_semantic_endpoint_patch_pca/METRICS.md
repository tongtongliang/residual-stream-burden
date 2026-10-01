# Metrics

## Population covariance

For each model and clean fraction `t`, the endpoint observations are all
`100000 * 256 = 25.6M` semantic patch vectors `h_t` in `R^768`. The globally
centered covariance is

`C_t = (1 / (N-1)) sum_n (h_t,n - mean(h_t))(h_t,n - mean(h_t))^T`.

The batch cross-products are FP32. Per-shard and cross-shard Chan merges and
the final eigendecomposition are FP64. All reported values use the raw
epoch-200 `model` state and true ImageNet class labels.

## Absolute spectral statistics

- `total_variance`: `tr(C_t) = sum_i lambda_i`.
- `largest_eigenvalue`: `lambda_1`.
- `top-k absolute variance`: `sum_{i=1}^k lambda_i`.
- `top-k variance share`: `(sum_{i=1}^k lambda_i) / tr(C_t)`.
- `effective_rank`: `exp(-sum_i p_i log p_i)`, where `p_i=lambda_i/tr(C_t)`.
- `participation_rank`: `(sum_i lambda_i)^2 / sum_i lambda_i^2`.
- `stable_rank`: `tr(C_t) / lambda_1`.
- `r90`, `r95`, `r99`: minimum ranks whose cumulative variance reaches the
  corresponding threshold.

## Input reference and ratios

The raw noisy-patch spectrum is analytic:

`lambda_i(input,t) = t^2 lambda_i(clean patch) + (1-t)^2`.

Every ratio table retains both the endpoint numerator and raw-input
denominator. Rank ratios compare spectral dimensionality. The total-variance
ratio is scale-sensitive and must not be compared across architectures as an
intrinsic information measure.
