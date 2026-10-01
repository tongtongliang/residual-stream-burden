# UCF101 Raw Tubelet PCA

This experiment compares raw spatiotemporal video tubelets with raw spatial
ImageNet patches under a matched feature dimension and token budget.

| Source | Input | Patch / tubelet | RGB feature dimension | Tokens per sample |
|---|---:|---:|---:|---:|
| UCF101 | `16 x 256 x 256` | `4 x 16 x 16` | 3072 | 1024 |
| UCF101 | `128 x 128 x 128` | `16 x 8 x 8` | 3072 | 2048 |
| ImageNet | `1024 x 1024` | `32 x 32` | 3072 | 1024 |

## Protocol

- UCF101: 50 videos per class, selected deterministically with seed 2021.
- Temporal sampling: centered 16-frame clip with frame stride 4. Videos shorter
  than the 61-frame span use 16 uniformly spaced frames.
- Spatial preprocessing: bicubic resize of the short side to 256, then a
  `256 x 256` center crop.
- Coordinates: RGB values are mapped from `uint8` to `[-1, 1]`.
- Population sampling: one deterministic random tubelet per video, matching the
  previous ImageNet protocol of one random patch per image.
- PCA: subtract one global mean vector across sampled tubelets, then compute the
  leading 512 eigenvalues and exact total covariance trace. Effective-rank
  bounds account for the unresolved tail.

Run with the environment that provides Decord, OpenCV, PyTorch, pandas, and
NumPy:

```bash
/home/pengrun/anaconda3/envs/GRPO-new/bin/python \
  experiments/raw_tubelet_profile/run_profile.py
```

The executed notebook is generated after the measurement completes:

```text
notebooks/ucf101_tubelet_vs_imagenet_patch_pca.ipynb
notebooks/ucf101_tubelet_scale_comparison.ipynb
```
