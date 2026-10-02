# SiHC: training, inference and evaluation

This directory contains the new model and its runnable workflows.

| Directory | Contents |
|---|---|
| `src/sihc/` | Model definitions, spatial read/write kernels, checkpoint helpers and sampling utilities |
| `training/` | Training, resume and REPA integration |
| `evaluation/` | Sampling, FID/IS evaluation and weight export |
| `configs/` | B / L / H / XL model recipes |
| `data/` | ImageNet dataset and preprocessing instructions |
| `pretrained/` | HF catalogs and existing evaluation metadata; large weights remain on HF |
| `bench/` | Fusion benchmark scripts |
| `tests/`, `tools/` | Correctness checks, auditing and checkpoint download |

Start with [the model API](docs/MODEL_QUICKSTART.md), [model sizes and recipes](docs/REPRODUCIBILITY.md), [training](training/README.md), and [inference/evaluation](evaluation/README.md). The [XL + REPA recipe](training/XL_REPA.md) provides teacher setup and full commands. Research controls such as JiT/mHC are explained under [research](../research/README.md).

Run commands from the repository root after `pip install -e .`. Existing Python module and checkpoint names are unchanged.
