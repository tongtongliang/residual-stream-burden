# Group1 semantic-flow probes

DiP, PixelDiT, HyperDiT, DeCo: raw epoch200 semantic representations under
25%, 50%, 75% input noise. Same patch RMS/GAP linear-probe protocol as the
preceding Group1 experiments; four GPU pairs run the four models concurrently.

Measure after semantic blocks1-11 for DiP/PixelDiT/DeCo, and1-7 for HyperDiT,
whose semantic backbone has only8 blocks. Exclude HyperDiT's256 register tokens
from pooling, retaining256 spatial patches. Decoder/fine representations are
not probed. Probe heads are independently trained from zero for each noise level.

Each completed job is automatically validated and added to REPORT.md and data/.
Runtime status: `/data/tongtong/sihc_artifacts/hidden_noise_linear_probe/semantic_runs/status.json`.
No W&B or automatic git push. No older experiment results are overwritten.
