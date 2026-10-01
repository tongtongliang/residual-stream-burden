# DINOv2-B input patch spectrum

Start with [REPORT.md](REPORT.md).

This package archives the **measured DINOv2-B /14 patch-token covariance
spectra** on the fixed ImageNet train-100k balanced subset (25.6M patches),
plus the linked noise-sweep and per-mode SNR analyses used in the transport-
burden / RAE discussion.

Three clean variants are stored:

1. affine-free **final LayerNorm** patch tokens;
2. **RAE position-wise variance normalization** (the actual RAE diffusion
   coordinate);
3. RAE-normalized tokens after an extra **per-token RMS** audit.

No GPU is required to read these artifacts. Eigenvectors are not stored as
separate arrays; they can be recovered from
`data/spectra/dinov2b_patch_covariance_moments.npz` via `eigh` on the `m2`
matrices if needed.
