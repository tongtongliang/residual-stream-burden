# Metrics

At fan-in point `l`, for image `i`, noise draw `epsilon`, and full residual coordinate dimension `D = 256 x 768`:

```
mu_il = E_epsilon h_l(i, epsilon)
W_l = E_i E_epsilon ||h_l(i,epsilon) - mu_il||^2 / D
B_raw_l = E_i ||mu_il - E_i mu_il||^2 / D
bias_l = (1 - 1/N) W_l / K
B_l = max(B_raw_l - bias_l, 0)
T_l = W_l + B_l
F_l = W_l / T_l
```

Here `N=8192` images and `K=128` noises per image.

- `W` is the absolute, per-coordinate noise-conditioned variance.
- `B_raw` is the uncorrected absolute between-image variance of finite-`K` sample means.
- `W/K` is the finite-noise-sample contribution to `B_raw`; `B` is the corrected absolute image variance.
- `T=W+B` is the retained absolute marginal variance under this decomposition.
- `F=W/(W+B)` is scale invariant. It is a noise-conditioned variance fraction, not semantic SNR and not mutual information.
- Per-image fractions and their q25/median/q75/q90 summarize image heterogeneity.
- Bootstrap files retain confidence intervals; paired JiT clean/velocity differences use identical images and noise seeds.

The report gives both absolute `W`, `B`, `T` and ratio `F`; the ratio must not be interpreted without its absolute components. Cross-architecture absolute values remain sensitive to residual-stream scaling.
