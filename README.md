# Residual-Stream Burden in Diffusion Transformers

Code and experiments accompanying our study of residual-stream burden and Spatially Indexed Hyper-Connections (SiHC).

## Main findings

**Residual-stream burden** is the information-preservation requirement imposed on the backbone by its prediction target and output paths. Our experiments connect patch geometry, learned filtering and residual-stream bandwidth:

- **Clean-patch variance is spectrally concentrated.** Eight of the 768 directions in a 16×16 RGB patch explain 90% of clean variance, versus 668 for the velocity target. Invertible whitening flattens this spectrum; clean-prediction FID rises from **10.19 to 132.95**, supporting the role of spectral concentration in clean prediction. [PCA and gain results](research/patch_embedding/README.md) · [Whitening results](research/whitening/README.md)
- **Prediction targets and output paths shape representations.** Clean prediction learns selective input filtering, while direct velocity prediction retains broader responses. Decoupled pixel paths recover clean-like filtering in their semantic streams; the learned-long-skip experiment connects this behavior to an automatically learned input-to-output path. [Embedding analysis](research/patch_embedding/README.md) · [Linear probes](research/linear_probes/) · [Long skip](research/long_skip/README.md)
- **Persistent bandwidth is a key resource.** Four-stream dynamic mHC keeps each attention/MLP workspace at width 768 but lowers velocity FID from **139.83 to 25.36**; clean FID changes from **10.19 to 9.77**. [Control setup and results](research/pixel/README.md)
- **SiHC turns this understanding into a model design.** Spatially assigned prediction states and static feature-wise linear read/write maps connect expanded residual bandwidth to shared Transformer computation, without a dedicated decoder or cross-attention interface. [Model training and inference](sihc/README.md)

## Repository layout

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
