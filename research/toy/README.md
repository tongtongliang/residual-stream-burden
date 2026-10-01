# Toy residual-stream diagnostics

This self-contained PyTorch example studies how clean-data, velocity, and noise
prediction affect a residual stream. It keeps two small architectures and the
diagnostics needed to inspect them. It requires only NumPy and PyTorch; it does
not import the image-model library, use network services, or require CUDA.

## Models and data

| Model | Persistent state | Learned update |
| --- | --- | --- |
| `fcn` | One hidden residual vector | AdaLN-zero, Linear–ReLU–Linear branch |
| `sihc` | One hidden vector per 1D input patch | Featurewise linear read, shared AdaLN-zero branch, featurewise linear write |

Defaults are ambient dimension 512, width 256, depth 5, and time-embedding
dimension 256. SiHC uses four streams, hence 128 coordinates per patch.
Both models start with zero output projections. The FCN uses Gaussian Xavier
initialization and zero biases; the patch model uses PyTorch Linear defaults
except for zero AdaLN/output projections. This compact port preserves the
model equations; it is not a loader for other toy checkpoint formats or a claim
of bitwise reproduction across different frameworks.

The synthetic data are the x/z coordinates of a Swiss roll, embedded through
a seeded orthonormal matrix into the ambient space, then centered and scaled
per ambient coordinate. The noiseless curve lies in a two-dimensional linear
plane. Per-coordinate scaling changes the plane's orthonormal basis, so geometry
diagnostics use `QR(projection / std)`, not the original projection matrix.
All data and normalization tensors are saved in each checkpoint.

For SiHC, with carrier `X[b,s,c]`, read `a[l,c,s]`, and write `b[l,c,s]`,

```text
workspace_l = sum_s a[l,c,s] * X_l[b,s,c]
update_l    = branch_l(workspace_l, time_condition)
X_(l+1)     = X_l + b[l,c,s] * update_l[b,c]
```

Reads start at `1 / streams`; writes start at one. They are unconstrained
parameters. The default implementation precomputes initial reads and the
triangular coefficient `gamma[l,j,c] = sum_s a[l,c,s] * b[j,c,s]`, then
accumulates updates once. `schedule="explicit"` exposes the materialized
recurrence for correctness checks. These are readable tensor contractions;
the image-model kernel implementations and their benchmarks are separate.

## Time and loss conventions

The model receives **noise time** `tau`:

```text
z_tau = (1 - tau) * x0 + tau * epsilon
velocity target = epsilon - x0
```

The reporting coordinate is **clean time** `t = 1 - tau`, so `t=1` is clean.
All three prediction modes minimize the same velocity-space MSE:

| Mode | Raw network output | Conversion to velocity |
| --- | --- | --- |
| `x` | Clean data | `(z - prediction) / tau` |
| `v` | Velocity | `prediction` |
| `eps` | Noise | `(prediction - z) / (1 - tau)` |

Training draws `tau = sigmoid(N(0,1))`, clamped to `[0.001, 0.999]`.
Direct unweighted clean-data MSE would be a different objective. Optimizer
defaults are AdamW, learning rate `1e-4`, betas `(0.9, 0.999)`, zero decay,
and global gradient clipping at one. There is no EMA or LR schedule here.

## Run a comparison

Run these commands from the repository root after installing its Python
dependencies. CPU is the default. The commands below train actual models;
use a smaller `--steps` value for a quick functionality check.

```bash
python -m research.toy.train --family fcn --mode x --steps 200000 --output outputs/toy/fcn_x
python -m research.toy.train --family fcn --mode v --steps 200000 --output outputs/toy/fcn_v
python -m research.toy.train --family sihc --mode x --steps 200000 --output outputs/toy/sihc_x
python -m research.toy.train --family sihc --mode v --steps 200000 --output outputs/toy/sihc_v
```

Use `--device cuda` only when GPU execution is intended. These runners are
single-process and do not modify image-model training. Within each family,
the same seed gives the same initialization and training batch/noise sequence
across prediction modes. The seeds also define a common dataset across families.

Each directory contains `config.json`, `loss.csv`, and `last.pt`. Checkpoints
include model, AdamW state, exact data, batch RNG state, committed log history,
and completed step. Files are replaced atomically. Resume with a new total step
target; scientific settings cannot change on resume:

```bash
python -m research.toy.train --resume outputs/toy/fcn_x/last.pt --steps 300000
```

Resume requires the same device type for its random-number generator. An
optional empty `--output` directory creates a separate continuation. Checkpoints
contain tensors and plain metadata and load with `weights_only=True`.

## Diagnose a checkpoint

```bash
python -m research.toy.analyze outputs/toy/fcn_x/last.pt --output outputs/toy/fcn_x/diagnostics --gradients
python -m research.toy.sample outputs/toy/fcn_x/last.pt --output outputs/toy/fcn_x/samples
```

Analysis uses CPU, the saved dataset prefix, and a fixed set of Gaussian noise
draws paired across all clean times. Matching seeds and data across checkpoints
give paired model comparisons. Outputs are:

- `representations.csv`: stable rank, 90%/95% energy rank, entropy rank, absolute
  energy, and conditional normalized noise variance for each tap and time.
- `spectra.npz`: full squared singular values of globally centered feature
  matrices, computed in FP64 after FP32 model evaluation.
- `gradients.csv` with `--gradients`: uncentered gradient spectra and the
  identity check `grad_W = output_gradient.T @ layer_input` for each Linear.
  These are gradients before clipping or any optimizer update.
- `protocol.json`: model/configuration, time convention, sample counts, and seed.
- Sampling produces `samples.npz` and `metrics.json`: Euler integration from
  noise time 0.999 to 0.001, two-dimensional projections, centered spectra, and
  absolute off-plane energy. Solver endpoints are finite; this is not an exact
  endpoint solution.

FCN taps separate the persistent residual from the post-AdaLN branch input.
SiHC additionally separates the patch carrier from its read workspace. Carrier
spectra flatten all streams per sample, so their dimensionality differs from
workspace spectra. Stable rank is `||H||_F² / ||H||_op²`, where `H` is the
centered feature matrix; it is not the stable rank of its covariance matrix.
Conditional noise variance uses population variance within each clean sample,
divided by the mean squared feature norm. A single noise draw yields zero
conditional variance by construction; use several draws for this diagnostic.

Rank concentration and off-plane energy measure geometry. They alone do not
establish semantic quality, necessity of a direction, or causation. This release
does not embed historical result tables or pretrained toy checkpoints.

## Checks

```bash
python -m pytest tests/test_toy.py
```

The tests cover time/loss conversions, normalized data geometry, explicit versus
triangular routing (including gradients), spectrum definitions, gradient
factorization, and exact next-step checkpoint continuation on CPU.
