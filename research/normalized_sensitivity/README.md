# Normalized Representation Sensitivity

## Protocol

11 historical raw checkpoints, registered in models.json; 3 clean fractions
(0.25, 0.50, 0.75), 2048 uniformly sampled ImageNet training IDs without
replacement, 128 independent Gaussian noises per image. All use null class 1000.
No flips. Pixel inputs are in [-1,1]; RAE uses the verified normalized DINOv2-B
FP32 no-flip cache. Pixel and RAE sample IDs and labels are aligned. Gaussian
seeds depend only on image ID and base seed; same-shape models/times share draws.
RAE receives time 1-t_clean; pixel models receive t_clean.

Models: pixel JiT clean/velocity; SiHC sublayer P4 velocity; RAE JiT and mHC,
both clean/velocity; PixelDiT, DiP, DeCO, HyperDiT semantic backbones.
Native model depths are retained (HyperDiT has 8 semantic blocks, others 12).
No decoder/head features, no context/register tokens, no GAP.

## Observation

Workspace: input to each norm1/norm2 before modulation, followed by a separate
non-affine per-patch RMS normalization with eps 1e-6. Concatenate all spatial
patches conceptually; no need to reshape for trace reductions.
SiHC: native fused read/triangular workspace computation. Additionally observe
the carrier before each event read, reconstructed via FP32 incremental native
write from the actual workspace updates and actual connection coefficients.
This diagnostic carrier is never fed back to the native workspace computation.
4096 carrier patches and 256 workspace patches are normalized individually.

## Streaming Algorithm

One forward batch contains all 128 noises of one image. Each hook immediately
reduces normalized features into an image-mean tensor and an unbiased within-image
covariance trace (K-1 divisor). Raw feature tensors are not retained across hooks.
Online Welford accumulation of the image means uses FP64 vectors/scalars:

delta = image_mean - running_mean
M2 += ||delta||^2 * old_count / new_count
running_mean += delta / new_count

B_raw = M2/(N-1); W = sum(within_trace)/N.
B_corrected = B_raw - W/K; R_raw=B_raw/W; R_corrected=B_corrected/W.
No clipping of negative corrected estimates. W=0 gives an undefined ratio.
Save B and W both raw and divided by feature dimension. Dividing the carrier
traces by 16 is equivalent to token-count alignment at equal hidden width.

The only large persistent states are one FP64 running mean per observation.
There is no feature cache, full covariance matrix, or second forward pass.
CPU validation compares against explicit covariance, reversed sample order,
production FP32 moments, negative correction, and zero-noise cases.

## Operation

Eight one-GPU workers consume 33 tasks. No W&B, no optimizer, no training changes.
Logs: logs/<model>_tXXX.log. Progress/partial metrics update every 64 images;
partial metrics are not final results. Completed tasks have complete.json and
metrics.csv/json. A failed task is recorded and is never marked complete.
Launch service: normalized-representation-sensitivity.service.

Interpretation: B measures between-image variation, not exclusively semantics.
W measures noise sensitivity after patch RMS, not absolute amplitude sensitivity.
Low W alone can reflect collapse: always inspect B and the ratio together.
