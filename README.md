# Residual-Stream Burden Shapes Representation Learning in Diffusion Transformers

SiHC models, training code, controlled experiments, and cached analyses accompanying our study of residual-stream burden. SiHC connects spatially assigned residual streams to a shared Transformer workspace through learned feature-wise linear read/write maps.

## Start here

- [Model installation and API](docs/MODEL_QUICKSTART.md)
- [Training and model sizes](docs/REPRODUCIBILITY.md): B / L / H without REPA; XL with DINOv3-L REPA.
- [Paper experiments and figure reproduction](docs/PAPER_EXPERIMENTS.md)
- [Checkpoint catalog](hub/checkpoints.json) and [checkpoint guide](docs/CHECKPOINTS.md)
- [Kernel benchmarks](docs/kernel_fusion/README.md)

## Flagship model

[SiHC-XL + REPA weights](https://huggingface.co/xiziqiao/sihc-group5-flagship-ckpt) are hosted by our coauthor. The selected file is `checkpoints/step_00650520.pt`, using its **primary EMA**. Its recorded evaluation is FID **1.71491**, IS **299.5535**: ImageNet 256×256, 50,000 balanced samples, JiT reference statistics, Heun-50, CFG 2.4 over clean-time interval [0.1, 0.9].

```bash
pip install huggingface_hub
python tools/download_checkpoint.py --id flagship-xl --output checkpoints
```

The original file is a full training checkpoint (approximately 15.4 GB), not an EMA-only export. See [XL training and evaluation](training/FLAGSHIP.md) for the matching architecture and teacher setup. The teacher is needed for REPA training, not sampling.

## Repository layout

| Directory | Contents |
|---|---|
| `sihc/` | Model definitions, fused kernels, checkpoint and sampling utilities |
| `training/`, `evaluation/` | SiHC training, resume, sampling, FID/IS and exports |
| `research/pixel/` | JiT and dynamic mHC control training |
| `research/rae/` | DINOv2-B RAE controls and integration with upstream RAE |
| `research/long_skip/` | Archived learned long-skip training and evaluation |
| `research/toy/` | Independent toy experiments |
| `analysis_reports/` | Cached measurements and their source provenance |
| `paper/fig_table/code/` | Figure renderers reading cached measurements |
| `hub/` | Checkpoint inventories, architecture identities and existing metrics |
| `bench/`, `tests/` | Fusion benchmarks and correctness checks |

The default model is sublayerwise. Historical blockwise checkpoint families have separate identifiers in the catalog. Representation diagnostics use raw weights; generation metrics use the state specified in each evaluation record.

GPU model execution requires Linux, NVIDIA CUDA and Triton. Cached analysis and toy tests run on CPU. This release is organized from the previously tested supplement; the current preparation checks are listed in [release validation](docs/RELEASE_VALIDATION.md).

Code is distributed under [MIT](LICENSE); [third-party notices](THIRD_PARTY_NOTICES.md) preserve upstream attribution. Images, teacher weights and large checkpoints are supplied separately.
