# Residual-Stream Burden in Diffusion Transformers

Code and experiments accompanying our study of residual-stream burden and Spatially Indexed Hyper-Connections (SiHC).

The repository has two parts:

| Directory | Purpose |
|---|---|
| **[sihc/](sihc/README.md)** | SiHC model implementation, training, inference, evaluation, configurations and checkpoints |
| **[research/](research/README.md)** | Research experiments: controlled model training, learned long skips, matrix/spectrum analysis, linear probes and their cached measurements |

## Install

Use Linux with a CUDA-matched PyTorch installation for image-model training and inference:

```bash
pip install -e '.[eval,repa,research,dev]'
```

Fused SiHC requires Triton; see [the tested environment](requirements-tested.txt). Model import names remain `sihc`, `training` and `evaluation` after installation. Toy experiments and cached plots can run on CPU.

## Pretrained SiHC

[Hugging Face collection](https://huggingface.co/collections/TongtongLiang/residual-stream-burden-sihc-6abe6b2f99f62128cf8c14a7) brings together model and research checkpoints. The selected SiHC-XL + REPA checkpoint is [hosted by our coauthor](https://huggingface.co/xiziqiao/sihc-group5-flagship-ckpt): `step_00650520.pt`, primary EMA, Heun-50 FID **1.71491**, IS **299.5535**.

```bash
python sihc/tools/download_checkpoint.py --id sihc-xl-repa --output checkpoints
```

Checkpoint architectures, states and sampling settings are recorded in the [catalog](sihc/pretrained/checkpoints.json). Historical blockwise models retain separate identities from paper sublayerwise models.

[MIT license](LICENSE) · [Third-party notices](THIRD_PARTY_NOTICES.md) · [Preparation checks](research/docs/RELEASE_VALIDATION.md)
