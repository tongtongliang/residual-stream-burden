# Research experiments

Training code, measurement scripts and cached results are grouped by experiment topic. Directory names describe the experiment rather than the order in which reports were collected.

| Topic | What it contains |
|---|---|
| [pixel/](pixel/README.md) | Controlled JiT and dynamic mHC training, with matching configurations |
| [rae/](rae/README.md) | DINOv2-B RAE controls and upstream RAE integration |
| [long_skip/](long_skip/README.md) | Learned long-skip trainer, model, coefficient measurements and FID dynamics |
| [patch_embedding/](patch_embedding/) | Embedding matrices, PCA alignment and gain analysis |
| [whitening/](whitening/) | Whitened-patch geometry, learned embeddings and evaluation records |
| [semantic_endpoints/](semantic_endpoints/) | Decoder endpoint spectra |
| [linear_probes/](linear_probes/) | Pixel/DINO linear-probe measurements and source scripts |
| [residual_sensitivity/](residual_sensitivity/), [normalized_sensitivity/](normalized_sensitivity/) | Residual-stream sensitivity measurements |
| [dino_spectrum/](dino_spectrum/) | DINOv2-B patch spectra |
| [video_spectrum/](video_spectrum/) | UCF101 tubelet spectra |
| [decoder_comparison/](decoder_comparison/) | Controlled B-size decoder FID records |
| [scaling/](scaling/) | Scaling measurements and compute counts |
| [kernel_fusion/](kernel_fusion/) | Recorded kernel benchmark results |
| [toy/](toy/README.md) | Independent toy training and diagnostics |
| `plotting/`, `reference_tables/` | Shared figure renderers and comparison tables; each experiment owns its `figure_cache/` |

See [reproduction coverage](docs/REPRODUCTION_COVERAGE.md) for the distinction between runnable training/evaluation and cached analysis.

## Recreate figures from cached measurements

No checkpoint download or GPU is needed:

```bash
pip install numpy matplotlib pandas scipy
python research/plotting/render_target_alignment_gain.py
python research/plotting/render_dino_target_spectra.py
python research/plotting/render_video_tubelet_spectra.py
python research/plotting/render_learned_long_skip.py
```

Outputs go to `research/figures/`. Measurement values are unchanged; plotting reads existing arrays and tables.

## Train the controls

Install the repository first. JiT-B clean/velocity controls use full-rank patch embeddings, without bottleneck or context tokens.

```bash
torchrun --standalone --nproc_per_node=4 -m research.pixel.train --config research/pixel/configs/jit_b16_velocity.json --data_path /path/to/imagenet --run_dir runs/jit-v
torchrun --standalone --nproc_per_node=8 -m research.pixel.train --config research/pixel/configs/mhc_n4_velocity.json --mhc_backend nvidia_fused --max_steps 250200 --data_path /path/to/imagenet --run_dir runs/mhc-v
```

The mHC command uses the original Transformer Engine backend; see [its setup and results](pixel/README.md). Keep the archived effective batch size at 1024. For other pipelines see [long skip](long_skip/README.md), [RAE](rae/README.md) and [toy experiments](toy/README.md).

Original run identifiers and checkpoint steps are preserved for traceability. Measurement scripts imported from reports are source snapshots and may require their original dataset caches or external model repositories. The static scalar-access control factory and external B-size decoder training builder have not been recovered; their available records are indexed without advertising runnable replacements. Some paper scaling/DINO control weights are also absent from the currently inventoried HF collections.

Shared model and experiment correctness checks are under `tests/`: run `python -m pytest` from the repository root after installation.
