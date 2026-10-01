# Learnable long-skip JiT velocity experiment

Retrieved 2026-09-23 (local time) from the user-authorized private W&B project.

Run: https://wandb.ai/tol011-uc-san-diego/long_skip_jit/runs/long_skip_jit_b16_fullpatch_velocity_20260830_1747

## Verified results

| Item | Value |
|---|---:|
| Model | jit_b16_fullpatch_longskip / LongSkipVelocityJiT |
| Prediction | velocity |
| Layers / hidden width / heads | 12 / 768 / 12 |
| Parameters | 131,722,114 |
| In-context tokens | 0 |
| REPA | disabled |
| Training/evaluation checkpoint | 500,400 |
| Evaluation samples | 50,000 |
| Sampling steps | 50 |
| CFG | 2.9 |
| Guidance interval | 0.1–1.0 |
| Evaluation state | ema |
| FID | 6.7031203188908535 |
| Inception score | 187.46496173383034 ± 2.021082092672404 |

Evaluation arguments are recorded under `_wandb` in the raw config. The sampler algorithm is not specified by these arguments; do not label this Heun without checking the evaluation implementation. Config batch size is 256; it is not labeled here as global batch size.

## Downloaded data

- `raw/config.yaml`: original W&B file, 8,255 bytes, including evaluation invocation and source provenance.
- `raw/wandb-summary.json`: original final summary, 3,215 bytes.
- `raw/output.log`: original available log, 50,497 bytes. Contains 500 training records from step 450,400 through 500,300. This is the final resumed segment, not the full training history.
- `raw/history_manifest_v6.json`: original manifest identifying the latest history artifact, containing `0000.parquet` (1,729,809 bytes).
- `raw/jit_shared_adaln.py`: model file from the Git commit recorded by W&B, `29130b4a8017e8f31c47e6de662cd912e4fbf179`. It does not contain `LongSkipVelocityJiT`; it is provenance context, not the verified implementation of this experiment.
- `config.json`, `summary.csv`: convenient extracted config and summary.
- `coefficients_final.csv`: 20 paired final coefficient values, preserving the rounded time labels exactly as logged.
- `training_log_tail.csv`: parsed 500-record log segment.
- `overview.png` / `overview.pdf`: final coefficients and available training-loss segment.

The complete history Parquet has **not** been downloaded. Its W&B artifact is visible, but the direct download request returns 404 without browser authentication, and the in-app browser download did not yield a local file. Do not describe the extracted tail log or final coefficient table as full history. No checkpoint was downloaded or evaluated.

History artifact: https://wandb.ai/tol011-uc-san-diego/long_skip_jit/artifacts/wandb-history/run-long_skip_jit_b16_fullpatch_velocity_20260830_1747-history/v6/files/0000.parquet

## Coefficient interpretation

The user observes that the learned velocity skip coefficient approaches `-1/(1-t)`. For the paper convention `x_t=t*x+(1-t)*epsilon`, the identity `v=(x-x_t)/(1-t)` supplies exactly this input coefficient when the other branch supplies a clean prediction divided by `1-t`.

The downloaded metrics are named `skip/clean_alpha_effective_t_*` and `skip/clean_beta_effective_t_*`, not the raw velocity skip coefficient. Across the 18 logged labels from 0.05 through 0.86, alpha differs from 1 by at most 0.015073 (1.51%). It equals 0.904490 at label 0.90 and 0.608922 at label 0.95. Beta decreases from 0.688639 to 0.124501. The exact time grid may be rounded in the metric names.

These observations should not be silently converted into a raw velocity coefficient: the forward equation, coefficient normalization/sign, and training time convention remain to be verified in the actual `LongSkipVelocityJiT` implementation. Neither the recorded commit's model file nor the current repository's filename inventory identifies that implementation. Once the mapping is verified, compare the normalized ratio `-(1-t)*c_skip(t)` with 1 to avoid the singular scale near t=1.

If that ratio approaches 1 from a freely learned initialization, the experiment would support the interpretation that training learns an explicit noisy-input bypass resembling the analytic clean-to-velocity conversion. Near-unity effective alpha by itself does not establish that interpretation or prove what the backbone represents.

No manuscript, Overleaf, or GitHub files were updated as part of this retrieval.
