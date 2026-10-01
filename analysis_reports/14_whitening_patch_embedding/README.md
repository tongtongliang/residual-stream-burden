# Whitening clean vs velocity: writing-agent handoff

## Read first

Both Group6 whitening runs completed200 epochs (250200 optimizer steps).
Use REPORT.md for matrix interpretation, data/evaluation_history.csv for ALL
40/80/120/160/200-epoch primary-EMA FID/IS, and data/run_metadata.json for the
exact configuration. These are full768-dimensional invertible whitening runs,
NOT PCA64/128 truncation and NOT the later top16-noise or shift8 experiments.

All FID values use torch-fidelity against JiT ImageNet256 FID_STATS, balanced50K,
Heun50, CFG2.9 on clean-time (0.1,1), seed12345, primary EMA .9999.
Training: JiT-B/16 direct (no bottleneck/context), AdamW2e-4, global1024=4x256,
FP32 parameters/BF16 model autocast, default compile. t=1 means clean.

## Results and claim boundaries

At epoch200: clean FID132.94993521097024 / IS9.540327976212733;
velocity FID212.535562934228 / IS3.3529748983653236.
These are poor generation results, not successful whitening baselines.

Raw embedding Gram effective ranks are738.07/747.69, versus historical pixel
clean175.06/velocity696.24. The low-rank tendency of pixel-clean disappears in
whitened coordinates. This is descriptive evidence of coordinate-dependent
geometry, not proof that the embedding spectrum causes poor FID. Whitening
also changes the loss metric and the RGB-equivalent noise covariance.
Do not confuse weight Gram spectrum with activation/residual covariance.

## Portable data layout

- data/evaluation_history.csv: full original evaluation rows with model labels.
- data/run_metadata.json: verified final checkpoint metadata and protocols.
- data/embedding_matrices.npz: clean/velocity × raw(model)/primaryEMA weights
  [768,768] and biases [768], plus per-model PCA basis rows, scales, mean.
- data/spectra.npz: singular values and directional gains in native whitening
  and effective RGB coordinates; historical pixel raw controls included.
- data/spectrum_metrics.csv: absolute norms and energy/rank/gain statistics.
- data/pixel_reference_metrics.csv: original historical comparison table.
- provenance/: original spectrum and PCA-estimator provenance; absolute paths
  are audit records only, never required for rendering.
- figures/: ready-to-use spectrum and FID/IS PNG/PDF figures.

Matrix keys: whitening_clean_model_weight, whitening_clean_ema_weight, likewise
velocity; suffix bias for biases. clean_basis, clean_scales, clean_mean and
velocity equivalents describe checkpoint-owned preprocessing.

For column RGB patch x flattened C,H,W, z=D^-1 B(x-mu); model output is Wz+b.
RGB-effective A=W D^-1 B and effective bias b-A mu. Singular values ignore bias.
Whitening was fitted on100k ImageNet train images (100/class), all25.6M p16
patches, with uint8/127.5-1 normalization. No directions were truncated.

## Replot (no torch, checkpoints, GPU, network or W&B needed)

From repository root:

    python analysis_reports/14_whitening_patch_embedding/render.py

Requires NumPy and Matplotlib (requirements-viz.txt). Reads only this folder;
produces figures/spectrum.{png,pdf} and figures/fid_is.{png,pdf}.
Training checkpoints, optimizer states, credentials and image datasets excluded.
No new model inference or FID computation was performed during packaging.
