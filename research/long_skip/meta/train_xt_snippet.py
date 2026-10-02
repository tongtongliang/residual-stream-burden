"""Exact noisy-input / velocity-target lines from the long-skip training script.

Source: flowmatching_lthc/scripts/train_imagenet256.py
(run long_skip_jit_b16_fullpatch_velocity_20260830_1747).

This is the ground-truth time convention for analysis_reports/16_*:
  t = 0 → pure noise,  t = 1 → clean image.
"""

# --- excerpt (variable names as in training) ---
# x  : clean image in [-1, 1]
# e  : eps ~ N(0, I) * noise_scale  (or colored patch noise)
# t  : sigmoid-logit schedule in (0, 1), shape [B,1,1,1]
#
# z = t * x + (1 - t) * e
# v = (x - z) / (1 - t).clamp_min(t_eps)
#
# For prediction == "velocity": model outputs v_pred directly (long-skip wraps
#   v_hat = alpha(t) * z + beta(t) * backbone(...)).
# For prediction == "clean":
#   v_pred = (pred - z) / (1 - t).clamp_min(t_eps)

SNIPPET = """
t = torch.sigmoid(torch.randn(y.shape[0], device=device) * args.P_std + args.P_mean).view(-1,1,1,1)
e = torch.randn_like(x) * args.noise_scale
z = t * x + (1 - t) * e
v = (x - z) / (1 - t).clamp_min(args.t_eps)
"""
