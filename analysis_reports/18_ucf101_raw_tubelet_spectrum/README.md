# UCF101 raw tubelet spectrum

Start with [REPORT.md](REPORT.md).

This package archives the centered PCA spectra of raw RGB **spatiotemporal
tubelets** from UCF101, compared with dimension- and token-matched ImageNet
spatial patches. Spectra store the leading 512 eigenvalues plus the unresolved
trace (top-512 captures ≥99.7% of total variance in every stored config).

No video files are included. Sample paths and tubelet indices are in
`data/manifests/`.
