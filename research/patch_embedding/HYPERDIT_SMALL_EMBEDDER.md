# HyperDiT-B P8 Small-Embedder Spectrum

## Setup

- Weight: raw checkpoint `model['small_embedder.proj.weight']`, shape `768 x 3 x 8 x 8`, flattened to `768 x 192`.
- Input patch size: `8 x 8`; input dimension `3 x 8 x 8 = 192`; output dimension `768`.
- Raw P8 covariance: exactly pooled from the four `C x 8 x 8` quadrant second moments of the registered ImageNet-train-100k P16 covariance. This represents 102.4 million aligned P8 patches with the same preprocessing and image selection.
- Baselines: three `768 x 192` Gaussian matrices rescaled to the trained weight's Frobenius norm.

## Raw P8 patch spectrum

The raw P8 covariance is extremely concentrated: total variance `59.053831`, effective rank `3.055970`, top-1 share `0.779198`, top-10 share `0.956283`, `r90=4`, and `r99=41`.

## Small-embedder Gram spectrum

| matrix | top1_share | top10_share | top64_share | effective_rank | participation_rank | stable_rank | rank90 | normalized_effective_rank |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| HyperDiT-B P8 small embedder | 0.04239 | 0.18119 | 0.50890 | 163.30585 | 119.50429 | 23.58852 | 161 | 0.85055 |
| Gaussian baseline 0 | 0.01150 | 0.10711 | 0.53550 | 169.04820 | 153.03375 | 86.93182 | 146 | 0.88046 |
| Gaussian baseline 1 | 0.01105 | 0.10516 | 0.53344 | 169.56838 | 153.87316 | 90.51146 | 147 | 0.88317 |
| Gaussian baseline 2 | 0.01177 | 0.10791 | 0.53601 | 168.94590 | 152.82093 | 84.92612 | 146 | 0.87993 |

The trained Gram is **spiked but broad**, not globally low rank. Its top-1 and top-10 shares (`4.24%` and `18.12%`) exceed the Gaussian baselines (about `1.1%` and `10.7%`), and its participation/stable ranks are much lower. However, its entropy effective rank remains `163.31/192 = 85.1%` and `r90=161`. Thus a few strong singular directions coexist with a substantial full-rank tail.

## Alignment with raw P8 PCA

| matrix | top10_gain_share | top32_gain_share | top64_gain_share | bottom32_gain_share | lambda_amplification_vs_isotropic | gain_effective_rank | log_lambda_log_gain_spearman | gram_raw_covariance_frobenius_cosine | pca_basis_offdiagonal_ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| HyperDiT-B P8 small embedder | 0.17886 | 0.31905 | 0.47417 | 0.12848 | 5.41073 | 166.33212 | 0.88182 | 0.39283 | 0.15605 |
| Gaussian baseline 0 | 0.05246 | 0.16867 | 0.33605 | 0.16401 | 1.07976 | 191.71178 | 0.12908 | 0.08871 | 0.44783 |
| Gaussian baseline 1 | 0.05133 | 0.16601 | 0.33391 | 0.16584 | 1.04243 | 191.77759 | 0.04348 | 0.08588 | 0.44353 |
| Gaussian baseline 2 | 0.05294 | 0.17057 | 0.33629 | 0.16510 | 1.01864 | 191.72142 | 0.19065 | 0.08363 | 0.44915 |

The trained map is directionally selective even though it is not low rank. It assigns `17.89%` of gain to the top-10 P8 PCA directions and `47.42%` to the top 64, versus roughly `5.2%` and `33.5%` for Gaussian matrices. Lambda amplification is `5.41x`, versus approximately `1.05x` at random. The bottom-32 share falls to `12.85%`, compared with about `16.5%` at random.

## Principal-subspace alignment

| matrix | top8_principal_mean_cos2 | top16_principal_mean_cos2 | top32_principal_mean_cos2 | top64_principal_mean_cos2 | top128_principal_mean_cos2 |
|---|---:|---:|---:|---:|---:|
| HyperDiT-B P8 small embedder | 0.94347 | 0.83733 | 0.70889 | 0.65424 | 0.75615 |
| Gaussian baseline 0 | 0.05491 | 0.09262 | 0.17214 | 0.34217 | 0.66958 |
| Gaussian baseline 1 | 0.03751 | 0.07537 | 0.16902 | 0.33492 | 0.66461 |
| Gaussian baseline 2 | 0.05022 | 0.09524 | 0.18231 | 0.34100 | 0.67071 |

Top-8 mean cosine squared is `0.9435` versus random expectation `0.0417`; top-32 is `0.7089` versus `0.1667`; top-64 is `0.6542` versus `0.3333`. The leading learned directions therefore align strongly with the leading P8 data directions, while the remaining spectrum stays broad.

## Conclusion

HyperDiT's P8 fine-stream embedder is less aggressively compressed than its P16 semantic stem. The P16 semantic stem has gain ER `132.51/768` and top-64 gain share `70.57%`; the P8 fine stem has gain ER `166.33/192` and top-64 gain share `47.42%`. The clean conclusion is: **the P8 embedder learns strong leading-direction alignment and several spectral spikes, but it does not discard most low-energy input dimensions or become a low-rank map.** This is consistent with its fine stream retaining detailed spatial information.
