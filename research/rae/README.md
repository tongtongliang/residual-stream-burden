# DINOv2-B RAE controls

These wrappers come from the controlled RAE experiment source. Install the official [RAE repository](https://github.com/bytetriper/RAE), its dependencies, DINOv2-B encoder, decoder and normalization statistics separately. Set `RAE_ROOT`, `RAE_MODELS`, `DINOV2_B_ROOT`, `IMAGENET_PARQUET_ROOT`, `FID_STATS` and `GROUP2_ROOT` (a writable output directory). Install OmegaConf for configuration loading.

The YAMLs contain the original training-era unguided evaluation settings. The final paper comparison uses **Heun-50, CFG 2.9 and the same interval as pixel controls**; preserve this distinction when reporting metrics. Cached paper FIDs are clean: 5.496 / 3.926, velocity: 16.559 / 3.982 (plain / mHC). DINOv2-B uses 16×16 tokens of width 768.

Run `torchrun --standalone --nproc_per_node=8 -m research.rae.train_group2_rae --config research/rae/configs/2a_rae_baselines/g2a_01_rae_jit_b_clean.yaml` in the configured RAE environment. The wrapper passes remaining arguments to the upstream trainer. Checkpoints and encoder assets are not included here.
