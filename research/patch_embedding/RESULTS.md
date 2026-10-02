# Patch-Embedding Matrix Results

## Main statistics

| model | ||W||F | tr(W^T W) | Gram ER | Gram SR | Top-10 gain share | Top-64 gain share | Bottom-100 gain share | Lambda amplification | Gain ER | Log Spearman | Gram/cov cosine | PCA offdiag ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| JiT-B clean | 57.153427 | 3266.514177 | 175.061827 | 28.512222 | 0.245411 | 0.614758 | 0.000702 | 22.438163 | 181.255359 | 0.937711 | 0.400350 | 0.152317 |
| JiT-B velocity | 99.264285 | 9853.398207 | 696.241766 | 195.766542 | 0.018923 | 0.099867 | 0.109365 | 1.136703 | 749.699851 | 0.837855 | 0.051992 | 0.361736 |
| DeCO-B | 44.417488 | 1972.913252 | 153.470761 | 28.072070 | 0.263255 | 0.664399 | 0.000941 | 23.259669 | 156.789817 | 0.919965 | 0.391319 | 0.107974 |
| PixelDiT-B | 56.146218 | 3152.397827 | 104.434508 | 15.953032 | 0.386069 | 0.755200 | 0.001223 | 40.948429 | 107.047588 | 0.902712 | 0.515779 | 0.111348 |
| DiP-B | 43.018916 | 1850.627173 | 171.384841 | 16.597637 | 0.315192 | 0.610528 | 0.046190 | 38.832265 | 180.161090 | 0.625511 | 0.576953 | 0.120166 |
| HyperDiT-B | 49.716594 | 2471.739716 | 126.888400 | 23.157883 | 0.307971 | 0.705737 | 0.001278 | 24.732589 | 132.513342 | 0.940270 | 0.371535 | 0.132259 |

## Top-k principal alignment (mean cosine squared)

| Model | k=16 | k=32 | k=64 | k=128 | k=256 |
|---|---:|---:|---:|---:|---:|
| DeCO-B | 0.948353 | 0.945844 | 0.929748 | 0.927354 | 0.928924 |
| DiP-B | 0.924395 | 0.929075 | 0.884244 | 0.725604 | 0.770342 |
| HyperDiT-B | 0.924990 | 0.867401 | 0.873919 | 0.890563 | 0.912051 |
| JiT-B clean | 0.902932 | 0.924592 | 0.921005 | 0.909150 | 0.949196 |
| JiT-B velocity | 0.075533 | 0.095295 | 0.268173 | 0.529529 | 0.651862 |
| PixelDiT-B | 0.919932 | 0.877023 | 0.884562 | 0.891467 | 0.935154 |

## Head/tail gain shares

| model | Top 64 | Top 128 | Top 256 | Bottom 256 | Bottom 128 | Bottom 64 |
|---|---:|---:|---:|---:|---:|---:|
| JiT-B velocity | 0.099867 | 0.220397 | 0.417203 | 0.280186 | 0.140044 | 0.069883 |
| JiT-B clean | 0.614758 | 0.789435 | 0.952762 | 0.001819 | 0.000901 | 0.000449 |
| DeCO-B | 0.664399 | 0.834736 | 0.968715 | 0.002437 | 0.001207 | 0.000600 |
| PixelDiT-B | 0.755200 | 0.877000 | 0.979723 | 0.003160 | 0.001573 | 0.000786 |
| DiP-B | 0.610528 | 0.716006 | 0.864617 | 0.053967 | 0.048504 | 0.039362 |
| HyperDiT-B | 0.705737 | 0.863007 | 0.978815 | 0.003264 | 0.001626 | 0.000816 |

## Overall interpretation

JiT-B velocity is the clear isotropic-tail case: it assigns only 9.99% of gain
to the leading 64 clean-patch directions and 10.94% to the bottom 100. JiT-B
clean and all four decoupled semantic stems concentrate much more gain in the
high-variance clean-patch subspace. DiP-B has strong leading-subspace alignment
but a thicker low-variance tail than DeCO-B, PixelDiT-B, or HyperDiT-B.

## HyperDiT-B semantic stem

The measured matrix is the P16 semantic input projection
`large_embedder.proj.weight`. HyperDiT-B assigns `30.80%` of total embedding
gain to the top 10 clean-patch PCA directions and `70.57%` to the top 64, while
only `0.128%` reaches the bottom 100 directions. JiT-B velocity assigns
`1.89%`, `9.99%`, and `10.94%`, respectively. HyperDiT therefore performs a
strong spectral selection at the semantic-stream entrance rather than
transporting raw-patch directions nearly isotropically.

Its lambda amplification is `24.73x`, close to JiT-B clean (`22.44x`) and
DeCO-B (`23.26x`), but below PixelDiT-B (`40.95x`) and DiP-B (`38.83x`). Its
gain effective rank is `132.51`, lower than JiT-B clean (`181.26`) and DeCO-B
(`156.79`), so HyperDiT concentrates its input gain into fewer clean-patch
directions.

The leading subspace is also strongly aligned: mean cosine squared is `0.9250`
at k=16 and remains `0.8739` at k=64, compared with random expectations
`0.0208` and `0.0833`. The PCA-basis off-diagonal ratio is only `0.1323`, far
below JiT-B velocity (`0.3617`), which means the learned Gram is relatively
close to diagonal in the clean-patch PCA basis.

The Gram/covariance cosine (`0.3715`) is not the largest despite the strong
subspace scores. This is not a contradiction: principal alignment tests
whether the leading subspaces coincide, whereas Frobenius cosine also demands
matching eigenvalue magnitudes inside those directions and is dominated by
the highly anisotropic leading raw-patch eigenvalues.

Overall, HyperDiT-B's P16 semantic stem supports the same input-side pattern as
the other decoupled models: it resembles JiT-B clean much more than JiT-B
velocity. These are properties of the linear input map; they do not alone
establish downstream semantic quality or causal optimization benefit. The P8
fine stem is a separate stream and is intentionally not mixed into this
semantic-stem comparison.
