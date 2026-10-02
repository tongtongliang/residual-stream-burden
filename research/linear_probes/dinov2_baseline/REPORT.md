# DINOv2-B normalized latent noisy GAP linear probe baseline

This is the ground-representation control for the layerwise JiT/SiHC/mHC probes. It measures the class information already linearly readable from the normalized DINOv2-B latent supplied to RAE training, before any generative backbone is applied.

## Results

All values are ImageNet-1k validation accuracy at the fixed epoch-40 probe checkpoint. The matched-noise mean and standard deviation use three independently sampled validation-noise instances; they are not uncertainty over independently trained probes.

| Probe training noise | Validation input | Top-1 (%) | Top-5 (%) |
|---:|---|---:|---:|
| 0% | clean | 79.372 | 95.036 |
| 25% | clean | 79.972 | 95.288 |
| 25% | matched 25% noise | 79.872 +/- 0.037 | 95.267 +/- 0.011 |
| 50% | clean | 80.850 | 95.710 |
| 50% | matched 50% noise | 80.485 +/- 0.072 | 95.545 +/- 0.011 |
| 75% | clean | 82.274 | 96.392 |
| 75% | matched 75% noise | 80.820 +/- 0.088 | 95.847 +/- 0.042 |

## Setup

- Images use the same ImageNet train/validation split, ADM center crop, and no-flip protocol as the RAE hidden-state probes.
- The frozen official RAE encoder resizes to 224, applies its image normalization, runs DINOv2-with-registers base, removes the CLS and four register tokens, applies a non-affine final LayerNorm, and then applies the official RAE latent mean/variance normalization. The resulting tensor has shape `768 x 16 x 16`.
- Each noise level trains its own affine `768 -> 1000` classifier. The 0%, 25%, 50%, and 75% classifiers are not one fixed probe evaluated across noise levels.
- The feature is GAP over 256 normalized patch tokens. For efficiency, matched noise is sampled after GAP as `(1-alpha) GAP(x) + (alpha/16) epsilon`. Because patch noise is independent standard Gaussian, this has exactly the same distribution as adding `(1-alpha)x + alpha epsilon_patch` before GAP.
- Probes train for 40 epochs on all 1,281,167 ImageNet training images with global batch 4096, AdamW (`wd=0`), and cosine learning rate `0.01 -> 0.0001`. Encoding and probe optimization are FP32 with TF32 disabled.
- Validation uses all 50,000 images. The fixed epoch-40 checkpoint is evaluated without validation-based model selection. Matched-noise results average three noise seeds.

## Interpretation

The normalized DINOv2-B input already exposes approximately 80% ImageNet Top-1 accuracy under every tested matched-noise condition. Therefore the 79-81% accuracies observed inside RAE JiT/mHC cannot be interpreted as semantic information created from scratch by those backbones. The meaningful RAE comparison is the change with depth and relative to this input baseline.

The slight increase from 79.87% to 80.82% as the probe's training noise increases is compatible with noise acting as regularization for this linear classifier. It does not mean that a fixed representation becomes more informative when noise is added, because a different probe is trained at each noise level. The auxiliary clean-input rows show transfer of each noise-trained probe and are not part of the matched-noise headline comparison.

## Archived data

- `data/accuracy.csv`: summarized clean and matched-noise results.
- `data/accuracy_by_seed.csv`: every validation-noise instance and cross-entropy.
- `data/training.csv`: all 40 training epochs and per-noise training losses.
- `protocol.json`: reproducible protocol without machine-specific checkpoint paths or file hashes.
- `provenance/train.py`: source snapshot used for the experiment. Checkpoints and environments are intentionally excluded.
