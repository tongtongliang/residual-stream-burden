# UCF101 raw tubelet versus ImageNet patch PCA

## 1. Question

At matched raw RGB feature dimension and (for the primary pair) matched token
budget, how anisotropic are natural **video tubelets** relative to natural
**image patches**?

This is a raw-data geometry baseline. It is not a video DiT residual-stream
measurement and does not isolate pure temporal motion (UCF101 and ImageNet also
differ in content and compression).

## 2. Main numbers

Official ranks use randomized top-512 PCA with unresolved-tail entropy bounds
(see protocol). Values below are from `data/metrics/`.

| Source | Input | Tubelet / patch | Tokens / sample | dim | ER | Stable | r90 | PC1 | Tr | top-512 capture |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| UCF101 tubelet | `16×256×256` | `4×16×16` | 1024 | 3072 | **4.446** | 1.338 | **9** | 74.7% | 948.9 | 99.92% |
| ImageNet patch | `1024×1024` | `32×32` | 1024 | 3072 | 3.532 | 1.303 | 5 | 76.8% | 955.9 | 99.98% |
| UCF101 tubelet | `128×128×128` | `16×8×8` | 2048 | 3072 | 4.568 | 1.331 | 9 | 75.1% | 947.1 | 99.71% |

Primary comparison (row 1 vs 2): video ER is **1.26×** image ER; r90 needs four
extra components (9 vs 5); stable rank is only **1.03×**; traces match within 1%.
Both remain extremely anisotropic (PC1 ≈ 75%).

The second UCF config keeps dim=3072 but doubles tokens. Local spectra barely
change (ER 4.45 → 4.57): trading spatial side for temporal extent does not
broaden the raw tubelet cloud much.

Energy coverage of the stored leading spectrum (of total trace):

| Spectrum | top-8 | top-32 | top-128 | top-512 |
|---|---:|---:|---:|---:|
| UCF `4×16×16` | 89.6% | 95.4% | 98.8% | **99.92%** |
| ImageNet `32×32` | 93.0% | 97.0% | 99.4% | **99.98%** |
| UCF `16×8×8` | 89.5% | 94.6% | 97.9% | **99.71%** |

## 3. Protocol (metadata)

Authoritative machine-written copies:
`provenance/t16_r256_metadata.json`,
`provenance/t128_r128_metadata.json`.

### 3.1 Population

| Item | Primary (`t16_r256`) | Scale variant (`t128_r128`) |
|---|---|---|
| Dataset | UCF101, 101 action classes | same |
| Video selection | 50 videos / class, deterministic, seed **2021** | same |
| Videos requested / analyzed | 5050 / 5050 | 5050 / 5050 |
| Decode failures | 0 | 0 |
| Sampling unit | **one** deterministic random tubelet per video | same |
| PCA centering | global mean over sampled tubelet vectors | same |
| Pixel scaling | `uint8 / 127.5 - 1` → RGB in `[-1, 1]` | same |
| Per-tubelet norm | none | none |

ImageNet reference (shared): validation **5 images/class** (5000 images), one
random `32×32` patch per image on `1024×1024` ADM-style crops, from
`raw_pixel_patch_profile` (`imagenet_val_r1024_p32`).

### 3.2 Clip construction

**Primary — `16×256×256` clip, tubelet `4×16×16`**

1. Decode a **centered 16-frame** clip with **source-frame stride 4**
   (span = 61 source frames). Videos shorter than that span use 16 indices
   uniformly spanning the video (`short_video_fallback` in metadata).
2. Spatial: bicubic resize **short side → 256**, then **256×256 center crop**.
3. Conceptually partition the clip into a non-overlapping tubelet grid
   `4 × 16 × 16` in `(time, height, width)` → **1024** tubelets/clip.
4. Draw one tubelet via deterministic unit-cube coordinates (seed 2021).
5. Flatten order: **channel, time, height, width** → dimension
   `3×4×16×16 = 3072`.

**Scale variant — `128×128×128` clip, tubelet `16×8×8`**

1. Centered **128 consecutive** frames (`frame_stride=1`). Videos shorter than
   128 frames use 128 indices spanning the video **with repeated indices
   allowed**.
2. Spatial: short side → **128**, center crop **128×128**.
3. Non-overlapping `16×8×8` grid → **2048** tubelets/clip; same flatten order →
   still dim 3072.

### 3.3 PCA method

- Matrix: `N × 3072` sampled tubelets/patches (`N=5050` video, `N=5000` ImageNet).
- Subtract the global sample mean; no whitening.
- Randomized PCA via `torch.pca_lowrank`, **q=512**, `niter=5`.
- Store leading 512 eigenvalues and
  `unresolved_variance = total_trace − sum(stored eigenvalues)`.
- Reported effective rank uses geometric-mean entropy bounds over min/max
  entropy completions of the unresolved tail (same convention as the image
  raw-pixel profile). Stable rank = `total_variance / λ₁`. r90 uses the total
  trace as denominator.

### 3.4 What is intentionally not claimed

- Temporal motion isolation (no repeated-frame null in this package).
- Full 3072-eigenvalue spectrum or eigenvectors (recoverable only by re-running
  on the raw videos).
- Equivalence to any specific video tokenizer recipe beyond “non-overlapping
  RGB tubelets.”

## 4. Files

| Path | Contents |
|---|---|
| `REPORT.md` / `README.md` | This report / entrypoint |
| `data/spectra/*.npz` | Eigenvalues, unresolved/total variance, sample count, dim |
| `data/spectra/tubelet_patch_eigenvalues.csv` | Same spectra as CSV |
| `data/metrics/*.csv` | Per-config comparison metrics (official ER / r90 / …) |
| `data/manifests/*.csv` | Per-video paths, frame ranges, tubelet indices |
| `provenance/*_metadata.json` | Exact protocol knobs written at run time |
| `provenance/spectrum_summary.json` | Compact energy-coverage summary |
| `provenance/ucf101_MANIFEST.md` | Local UCF101 download/extract audit |
| `reports/` | Copied source RESULTS / README text |

Local source tree (not in git):
`residual_stream_research/video_patch_geometry/experiments/raw_tubelet_profile/`.

## 5. Citation

Khurram Soomro, Amir Roshan Zamir, and Mubarak Shah. *UCF101: A Dataset of 101
Human Actions Classes From Videos in the Wild.* CRCV-TR-12-01, 2012.
