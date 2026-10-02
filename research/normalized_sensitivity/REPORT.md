# Normalized Representation Sensitivity

See [depth analysis](INTERPRETATION.md), [all per-event statistics](per_event.csv),
and [archive/metadata guide](ARCHIVE.md). Exact task metadata are under `data/`;
sample IDs, noise seeds, config snapshots and environment metadata are under `metadata/`.

Completed tasks only. N=2048, K=128, null class, raw historical weights. Time is CLEAN fraction.
B is finite-K corrected between-image covariance trace; W is within-image noise covariance trace.
B and W below are divided by feature dimension then averaged over observed events. R is mean of per-event ratios, not ratio of these displayed mean traces.
HyperDiT has 16 events; others have 24. Last event is the last MLP input, not the final model output.

| Model | t clean | Scope | B / D | W / D | Mean event R |
|---|---:|---|---:|---:|---:|
| deco_b_velocity | 0.25 | workspace | 0.097954 | 0.090100 | 1.4128 |
| deco_b_velocity | 0.5 | workspace | 0.149100 | 0.034830 | 4.4371 |
| deco_b_velocity | 0.75 | workspace | 0.196431 | 0.010290 | 19.0668 |
| dip_b_velocity | 0.25 | workspace | 0.120647 | 0.083473 | 1.7164 |
| dip_b_velocity | 0.5 | workspace | 0.161638 | 0.048505 | 3.5570 |
| dip_b_velocity | 0.75 | workspace | 0.182066 | 0.035720 | 11.2448 |
| hyperdit_b_velocity | 0.25 | workspace | 0.216336 | 0.188934 | 1.1945 |
| hyperdit_b_velocity | 0.5 | workspace | 0.293143 | 0.099360 | 3.0729 |
| hyperdit_b_velocity | 0.75 | workspace | 0.357153 | 0.043822 | 12.2524 |
| pixeldit_b_velocity | 0.25 | workspace | 0.196986 | 0.085974 | 2.7824 |
| pixeldit_b_velocity | 0.5 | workspace | 0.223950 | 0.031887 | 7.8725 |
| pixeldit_b_velocity | 0.75 | workspace | 0.241170 | 0.009661 | 28.8316 |
| plain_jit_clean | 0.25 | workspace | 0.137170 | 0.115637 | 1.5008 |
| plain_jit_clean | 0.5 | workspace | 0.191948 | 0.044613 | 4.3929 |
| plain_jit_clean | 0.75 | workspace | 0.234187 | 0.013603 | 17.6873 |
| plain_jit_velocity | 0.25 | workspace | 0.081101 | 0.164150 | 1.0429 |
| plain_jit_velocity | 0.5 | workspace | 0.142343 | 0.132686 | 1.8945 |
| plain_jit_velocity | 0.75 | workspace | 0.220509 | 0.071034 | 3.9031 |
| raev1-g2a-01-jit-b-clean-epoch200 | 0.25 | workspace | 0.286036 | 0.169684 | 4.4577 |
| raev1-g2a-01-jit-b-clean-epoch200 | 0.5 | workspace | 0.255974 | 0.053630 | 8.5576 |
| raev1-g2a-01-jit-b-clean-epoch200 | 0.75 | workspace | 0.174852 | 0.008943 | 20.7074 |
| raev1-g2a-02-jit-b-velocity-epoch200 | 0.25 | workspace | 0.359792 | 0.252211 | 2.0529 |
| raev1-g2a-02-jit-b-velocity-epoch200 | 0.5 | workspace | 0.418072 | 0.131015 | 3.7538 |
| raev1-g2a-02-jit-b-velocity-epoch200 | 0.75 | workspace | 0.443211 | 0.033262 | 13.6240 |
| raev1-g2b-01-jit-b-mhc4-clean-epoch200 | 0.25 | workspace | 0.226960 | 0.206250 | 4.4523 |
| raev1-g2b-01-jit-b-mhc4-clean-epoch200 | 0.5 | workspace | 0.288052 | 0.087512 | 9.8774 |
| raev1-g2b-01-jit-b-mhc4-clean-epoch200 | 0.75 | workspace | 0.286123 | 0.015619 | 30.4468 |
| raev1-g2b-02-jit-b-mhc4-velocity-epoch200 | 0.25 | workspace | 0.250104 | 0.191887 | 4.8366 |
| raev1-g2b-02-jit-b-mhc4-velocity-epoch200 | 0.5 | workspace | 0.279124 | 0.082416 | 11.3291 |
| raev1-g2b-02-jit-b-mhc4-velocity-epoch200 | 0.75 | workspace | 0.276101 | 0.017399 | 39.4604 |
| sihc_sublayer_p4 | 0.25 | carrier | 0.216370 | 0.233801 | 1.2065 |
| sihc_sublayer_p4 | 0.25 | workspace | 0.093628 | 0.084268 | 1.8769 |
| sihc_sublayer_p4 | 0.5 | carrier | 0.289482 | 0.128052 | 2.6326 |
| sihc_sublayer_p4 | 0.5 | workspace | 0.120902 | 0.047565 | 4.1889 |
| sihc_sublayer_p4 | 0.75 | carrier | 0.338039 | 0.046603 | 7.9234 |
| sihc_sublayer_p4 | 0.75 | workspace | 0.140329 | 0.017342 | 12.9766 |

Per-depth raw traces and both corrected/uncorrected ratios are in data/*/metrics.csv. See README.md and validation JSONs for protocol and estimator checks.
