# Sampling, evaluation and inference weights

Use a local checkpoint from training, or an inference export. The loader
reconstructs the registered architecture from metadata, validates state keys
and shapes, and uses restricted `torch.load(weights_only=True)` by default.
There is no implicit account-specific checkpoint download.

```bash
python -m evaluation.verify_checkpoint --checkpoint /path/to/checkpoint.pt --state_key ema
python -m evaluation.sample --checkpoint /path/to/checkpoint.pt \
  --output outputs/grid.png --state_key ema --batch_size 16 \
  --steps 50 --cfg 2.9 --interval_min 0.1 --interval_max 1.0
```

`ema` is the primary EMA; `ema2` is the second EMA; `model` is raw training
weights. The grid sampler uses FP32 model execution by default. Evaluation uses
BF16 autocast, so do not interpret their outputs as bitwise-equivalent. `--naive`
in the grid sampler is available only for non-context, non-write-only SiHC
reference models; the CTX presets require the CUDA fused path.

## Reproduce the paper's SiHC-XL + REPA evaluation

Run from the repository root in the CUDA environment after `pip install -e '.[eval,research]'`. Download the pinned primary-EMA checkpoint and the paper's JiT reference statistics:

```bash
python sihc/tools/download_checkpoint.py --id sihc-xl-repa --output checkpoints
python sihc/tools/download_checkpoint.py --id imagenet256-fid-stats --output checkpoints
python -m evaluation.verify_checkpoint --checkpoint checkpoints/checkpoints/step_00650520.pt --state_key ema

torchrun --standalone --nproc_per_node=8 -m evaluation.evaluate \
  --checkpoint checkpoints/checkpoints/step_00650520.pt --output_dir outputs/xl-repa-heun50 \
  --fid_stats checkpoints/assets/fid_stats/jit_in256_stats.npz \
  --state_key ema --prediction velocity --num_samples 50000 --batch_size 64 \
  --sampler heun --steps 50 --cfg 2.4 --interval_min 0.1 --interval_max 0.9 \
  --noise_scale 1.0 --seed 12345 --attn_backend flash \
  --compile --compile_mode default --no_save_fake_features
```

The repeated `checkpoints/checkpoints/` follows the file's path inside the HF repository. This is a full training checkpoint (about 15.4 GB); allow additional RAM for loading it. The recorded primary-EMA result is **FID 1.71491 / IS 299.5535**. The checkpoint and revision are pinned in the [catalog](../pretrained/checkpoints.json). REPA teacher weights and ImageNet training images are not needed for this evaluation.

The command preserves the recorded eight-rank, 64-samples-per-rank generation setup. Reducing these values can fit smaller hardware, but changes the random sample stream. Hardware and software changes can also affect the result. A preview grid is not a 50K FID evaluation.

## Paper protocols

All rows use 50,000 samples, primary EMA (decay 0.9999), Heun-50 and clean-time intervals. ImageNet sampling uses 50 labels per class, noise scale 1 and seed 12345.

| Experiment | CFG | Interval | Evaluation entry |
|---|---:|---|---|
| Pixel B-size / ablations | 2.9 | [0.1, 1] | `evaluation.evaluate` for supported RGB models |
| SiHC-L | 2.4 | [0.1, 1] | `evaluation.evaluate` |
| SiHC-H | 2.1 | [0.1, 0.9] | `evaluation.evaluate` |
| SiHC-XL + REPA | 2.4 | [0.1, 0.9] | Command above |
| Whitened clean / velocity | 2.9 | [0.1, 1] | Final whitening sampler still to be recovered |
| DINOv2-B plain / mHC | 2.9 | [0.1, 1] | Final guided RAE sampler still to be recovered |

For JiT clean checkpoints, select `--prediction clean`; velocity models use `--prediction velocity`. The generic RGB evaluator does **not** implement whitening/inverse whitening or RAE latent decoding. Do not use it to report those rows. See [reproduction coverage](../../research/docs/REPRODUCTION_COVERAGE.md) for available weights and experiment pipelines. Representation diagnostics use raw weights, whereas generation uses EMA.

## FID and Inception Score

Install the `eval` extra and supply reference statistics with `mu[2048]` and
`sigma[2048,2048]`, computed using the same torch-fidelity Inception-compat
features and matching dataset/preprocessing. ImageNet data/statistics are not
bundled; the pinned reference-statistics download above supplies the paper protocol. File presence, array shapes and finite values are checked before sampling.
torch-fidelity may download its public Inception weights on first use;
prepopulate its `TORCH_HOME` cache for offline execution.

```bash
torchrun --standalone --nproc_per_node=8 -m evaluation.evaluate \
  --checkpoint /path/to/checkpoint.pt --output_dir outputs/eval \
  --fid_stats /path/to/reference_stats.npz --num_samples 50000 --batch_size 32 \
  --state_key ema --sampler heun --steps 50 --cfg 2.9 \
  --interval_min 0.1 --interval_max 1.0 --seed 12345 \
  --compile --no_save_fake_features
```

The evaluator generates balanced labels, writes a preview grid, saves fake
feature statistics, and appends a CSV row recording sampler, CFG, interval,
checkpoint state, sample count, timing, FID and IS. Heun uses an Euler final
step; Euler is also available. Intervals act in **clean time**; guidance is
strictly inside their bounds (except the lower-bound-zero special case).
Sample counts divisible by 1000 give exactly equal class counts.

Use a fresh output directory for an independent evaluation. Internally created
temporary PNGs are removed after successful metrics unless `--keep_samples`
is selected. Explicit sample directories are never deleted automatically.
Metric failures are recorded and cause a nonzero exit. Distributed workers wait
for rank-zero metrics with a two-hour process-group timeout.

FID uses torch-fidelity 0.4.0 Inception-compat 2048-dimensional features on uint8 RGB images. IS uses its `logits_unbiased`, ten shuffled splits and split seed 2020.

FID is not comparable across different feature implementations or preprocessing.
No performance/quality result is implied by the example arguments. B/L/H CFG
examples are 2.9/2.4/2.1; use the value appropriate for the checkpoint being studied.

## Export inference weights

Training checkpoints may contain run names, local paths, optimizer state and
account settings. Do not publish them verbatim when anonymity matters.

```bash
python -m evaluation.export_checkpoint --checkpoint /path/to/training.pt \
  --state_key ema --output outputs/model_inference.pt
python -m evaluation.verify_checkpoint --checkpoint outputs/model_inference.pt
```

The exporter keeps one validated parameter state and a strict allowlist of
architecture fields. It discards the source args, optimizer, training step,
run identity and arbitrary provenance. The selected weights are exposed as
both `model` and `ema` for compatible inference defaults; the tensor storage
is shared in the serialized file. This file is for inference, not training resume.
No exported weights are included in this source package.

For fused block-wise models, `--kernel_backend tuple` selects the shared
tuple kernels; `--kernel_backend legacy` selects the original kernels.
Without an override, checkpoint metadata determines the backend and older
checkpoints default to legacy. See [kernel fusion notes](../docs/kernel_fusion/README.md)
for implementation details. This option does not
change block-wise connections to sublayer connections.
