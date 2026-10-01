# 16 — JiT-B/16 full-patch long-skip velocity (ImageNet-256)

Minimal architectural intervention on a **no-context JiT-B/16** velocity model:

\[
\hat v(x_t,t,y)=\alpha(t)\,x_t+\beta(t)\,\mathrm{nnet}(x_t,t,y)
\]

with \(\alpha,\beta\) from a zero-init time MLP (`TimeScalarSkip`; init \(\alpha=0\), \(\beta=1\)).
No REPA, no in-context tokens. Trained 400 epochs; primary-EMA 50k FID **6.70** at e400.

W&B: [long_skip_jit_b16_fullpatch_velocity_20260830_1747](https://wandb.ai/tol011-uc-san-diego/long_skip_jit/runs/long_skip_jit_b16_fullpatch_velocity_20260830_1747)

## Layout

| Path | Contents |
|---|---|
| `meta/run_metadata.json` | Run identity, architecture, train/eval protocol |
| `meta/train_xt_snippet.py` | Verified `x_t = t x + (1-t)ε` / velocity target lines |
| `meta/*.sh` | Original launchers (e0–200 and e200–400 resume) |
| `model/jit_standard.py` | Archived backbone + `TimeScalarSkip` + `LongSkipVelocityJiT` |
| `data/fid_is_50k.csv` | Epoch FID / IS table |
| `data/alpha_surface_long.csv.gz` | Long-form rows for 3D plot: `t`, `epoch`, `alpha`, `neg_alpha`, … |
| `data/alpha_surface.npz` | Mesh-friendly arrays (`epoch`, `t`, `neg_alpha`, analytic target) |
| `data/neg_alpha_grid_*.csv` | Wide grids (full / ~2-epoch / eval-epoch) |
| `data/analytic_targets.csv` | \(1/(1-t)\) and \(-1/(1-t)\) on the logged \(t\)-grid |
| `data/wandb_skip_history_sampled.csv.gz` | Raw W&B skip keys used to reconstruct \(\alpha\) |
| `notes/` | Session notes from the original training machine |
| `ANALYSIS.md` | Interpretation + how to draw the 3D surface |

Checkpoints and ImageNet data are **not** included (repo packaging policy).

## Sign convention (important for the 3D figure)

Training logs only the clean-effective transforms:

- `clean_alpha_effective = -(1-t)·α(t)` → **1** when \(α(t)=-1/(1-t)\)
- Therefore **`-α(t) → +1/(1-t)`**

For the requested surface (shape approaching \(1/(1-t)\)):

- **x** = `t`
- **y** = `epoch` (= `step / 1251`)
- **z** = `neg_alpha` (= `-alpha` = `clean_alpha_effective / (1-t)`)
- overlay analytic curve / surface: `neg_alpha_analytic = 1/(1-t)`

Raw `alpha` (negative, → `-1/(1-t)`) is also in `alpha_surface_long.csv.gz`.

**Time convention (verified against training code):** same as the paper,

\[
x_t = t\,x + (1-t)\,\epsilon,
\qquad
v^\star = \frac{x-x_t}{1-t}.
\]

So **\(t=0\) is noise, \(t=1\) is clean**. The analytic input-skip coefficient on
\(x_t\) is therefore **\(-1/(1-t)\)**, matching the logged
`clean_alpha_effective=-(1-t)α → 1` construction. Source line archived in
`meta/train_xt_snippet.py` (`z = t * x + (1 - t) * e`).

## Quick load

```python
import numpy as np
d = np.load("data/alpha_surface.npz", allow_pickle=True)
epoch, t, Z = d["epoch"], d["t"], d["neg_alpha"]  # Z shape [n_epoch, n_t]
Z_target = d["neg_alpha_analytic"]                 # shape [n_t]
# meshgrid for plot_surface: T, E = np.meshgrid(t, epoch)
```
