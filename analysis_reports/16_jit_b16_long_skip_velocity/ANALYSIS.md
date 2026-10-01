# Analysis: long-skip velocity learns a clean-prediction skip

## Result summary

| Epoch | FID (50k EMA) | IS |
|------:|-------------:|---:|
| 40 | 79.32 | 20.1 |
| 80 | 24.88 | 72.9 |
| 120 | 15.47 | 109.7 |
| 160 | 11.84 | 133.8 |
| 200 | 9.99 | 149.3 |
| 240 | 9.02 | 158.1 |
| 280 | 8.09 | 169.4 |
| 320 | 7.50 | 176.0 |
| 360 | 7.14 | 180.6 |
| 400 | **6.70** | **187.5** |

Eval: Heun-50, CFG 2.9, interval `[0.1, 1.0]`, JiT ImageNet-256 stats.
Same recipe as other B-size velocity controls in `flowmatching_lthc`.

Relative to historical **plain / REPA-only JiT-B velocity** collapses (FID ≫ 100 through e80),
the scalar long skip alone makes velocity training steadily improve into the single-digit FID
regime. It is still above clean+context (~4.9 @ e200) and far from LTHC+REPA D1024 (~1.72 @ e400);
the claim is about **inductive bias for velocity**, not parity with larger aligned models.

## Coefficient dynamics

From reconstructed surfaces in `data/`:

1. **`-α(t)` rises toward `1/(1-t)`** over training for most of the interior `t` grid.
   By late training, `clean_alpha_effective ≈ 1` from `t≈0.05` through `t≈0.86`, with the usual
   shortfall near the **clean** endpoint (`t=0.95`), where the analytic \(|α|=1/(1-t)\) diverges.
2. **`clean_beta_effective = (1-t)β(t)` shrinks** in both `t` and epoch: after α peels the
   identity / \(x_t\) piece out of the velocity, the backbone only needs a scaled residual.

This is the intended residual-stream / transport reading: identity information takes the
cheap skip; expensive backbone compute handles a smaller target.

Time convention matches the paper / training code: \(x_t=t x+(1-t)\epsilon\)
(**\(t=0\) noise, \(t=1\) clean**). See `meta/train_xt_snippet.py`.

## Intended 3D figure (data ready; not rendered here)

Axes:

- \(x\): diffusion time `t` (**\(t=0\) noise → \(t=1\) clean**)
- \(y\): training `epoch`
- \(z\): `neg_alpha(t, epoch) = -α(t)`

Reference wire / translucent surface: \(z = 1/(1-t)\) (independent of epoch).

Primary arrays: `data/alpha_surface.npz` or `data/alpha_surface_long.csv.gz`.
A lighter mesh is `data/neg_alpha_grid_2ep.csv` (~193 epoch samples × 20 `t` points).
Eval-aligned slices only: `data/neg_alpha_grid_eval_epochs.csv`.

Suggested matplotlib sketch (for later figure code, not executed in packaging):

```python
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np

d = np.load("data/alpha_surface.npz", allow_pickle=True)
T, E = np.meshgrid(d["t"], d["epoch"])
Z = d["neg_alpha"]
Zt = np.broadcast_to(d["neg_alpha_analytic"], Z.shape)

fig = plt.figure(figsize=(7, 5))
ax = fig.add_subplot(111, projection="3d")
ax.plot_surface(T, E, Z, linewidth=0, antialiased=True, alpha=0.9)
ax.plot_surface(T, E, Zt, linewidth=0, antialiased=True, alpha=0.25)
ax.set_xlabel("t"); ax.set_ylabel("epoch"); ax.set_zlabel(r"$-\alpha(t)$")
```

## Provenance

- Skip curves: W&B history for run `long_skip_jit_b16_fullpatch_velocity_20260830_1747`
  (`run.history(..., samples=10000)` → 5002 non-null rows, ~every 100 train steps).
- FID/IS: W&B `eval/fid` at checkpoint steps 50040…500400 (also mirrored in local
  `wandb-summary.json` shards on the training host).
- Local run directory / checkpoints were deleted for disk; this package is the durable
  scientific record without weights.
