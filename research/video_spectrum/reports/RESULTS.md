# UCF101 Tubelet versus ImageNet Patch PCA

The comparison is matched at 1024 tokens per sample and 3072 raw RGB
coordinates per token.

| Source | Effective rank | Stable rank | 90% rank | PC1 share | Total variance |
|---|---:|---:|---:|---:|---:|
| UCF101 `16x256x256`, tubelet `4x16x16` | 4.446 | 1.338 | 9 | 0.747 | 948.930 |
| ImageNet `1024x1024`, patch `32x32` | 3.532 | 1.303 | 5 | 0.768 | 955.949 |

The video tubelet has 1.259x the effective rank and needs four additional
components to explain 90% of variance, but its stable rank is only 1.027x larger.
Its covariance trace is 0.993x the ImageNet trace. Thus temporal tubelets spread
nearly the same total energy across a somewhat broader spectral tail while
remaining strongly dominated by the first principal component.

This comparison does not isolate temporal motion: UCF101 and ImageNet differ in
content, compression, and resizing. It establishes the requested raw-data
baseline at matched token and feature dimensions.

## 128-frame, 128-resolution profile

The second UCF101 configuration uses a `128x128x128` clip and a `16x8x8` RGB
tubelet. Its tubelet dimension remains 3072, while its token count is 2048.

| UCF101 configuration | Effective rank | Stable rank | 90% rank | PC1 share | Total variance |
|---|---:|---:|---:|---:|---:|
| `16x256x256`, tubelet `4x16x16` | 4.446 | 1.338 | 9 | 0.747 | 948.930 |
| `128x128x128`, tubelet `16x8x8` | 4.568 | 1.331 | 9 | 0.751 | 947.092 |

The local patch spectra are almost unchanged: the second configuration has
1.028x the effective rank, 0.995x the stable rank, and 0.998x the total
variance. The sequence-level token count nevertheless doubles, which patch PCA
does not capture.
