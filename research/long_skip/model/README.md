# Long-skip velocity head (ImageNet-256 JiT-B/16)

Archived from `flowmatching_lthc` run
`long_skip_jit_b16_fullpatch_velocity_20260830_1747`.
Documentation/reference only — not a training entrypoint.

## Forward form

```text
v_hat = alpha(t) * x_t + beta(t) * backbone(x_t, t, y)
```

`alpha`, `beta` from `TimeScalarSkip` (init `alpha=0`, `beta=1`).

## Time / skip convention (matches the paper)

Training constructs

```text
x_t = t * x + (1 - t) * eps      # t=0 noise, t=1 clean
v*  = (x - x_t) / (1 - t)
```

so the analytic coefficient on the `x_t` skip is **`-1/(1-t)`**. Logged:

```text
clean_alpha_effective = -(1-t) * alpha  → 1 when alpha = -1/(1-t)
```

Hence `-alpha(t) → +1/(1-t)` for the 3D surface. See
`../meta/train_xt_snippet.py` for the exact training lines.

Full backbone: `jit_standard.py` (`TimeScalarSkip`, `LongSkipVelocityJiT`).
