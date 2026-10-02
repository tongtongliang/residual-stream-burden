---
tags:
- image-generation
- sihc
- research-checkpoints
---
# SiHC research checkpoints

Weights and resume checkpoints for the residual-stream burden experiments. Use the catalog below to distinguish architectures, training objectives and weight formats.

## Choose a checkpoint

- `group1/`: weights-only pixel-space controls and SiHC variants.
- `resume/group1/`: corresponding full training checkpoints.
- `resume/group6/`: clean/velocity models trained in whitened patch coordinates.
- `resume/external_group1/`: controlled B-size DeCo, DiP and PixelDiT checkpoints.
- `flagship/` and `scaling_law/`: historical **blockwise** runs. The paper sublayer XL model is linked separately below.
- `raev2/`: additional research runs.

The withdrawn write-only control is excluded from the paper result index. Its original archive files remain in place for provenance.

Weights-only derivatives omit the optimizer. Resume archives carry additional training state. Use raw `model` weights for representation measurements and the documented EMA state for generation.

## Files and configuration

| Checkpoint | Epoch | Format | Architecture / status |
|---|---:|---|---|
| [flagship/sihc_5x8_d1024_velocity_repa_dinov3l/step_00550440.pt](flagship/sihc_5x8_d1024_velocity_repa_dinov3l/step_00550440.pt) | 440 | weights_only | local_thc_v2_fused_5x8_adaln_d1024_b4 |
| [flagship/sihc_8x12x8_d1152_velocity_repa_dinov3l/step_00100080.pt](flagship/sihc_8x12x8_d1152_velocity_repa_dinov3l/step_00100080.pt) | 80 | weights_only | shc_8x12x8_d1152_b4 |
| [group1/g1_fraction_hc_b16_velocity_p4_seed0/epoch_0200.pt](group1/g1_fraction_hc_b16_velocity_p4_seed0/epoch_0200.pt) | 200 | weights_only | available |
| [group1/g1_hyperdit_b_p16p8_velocity_seed0/step_00250200.pt](group1/g1_hyperdit_b_p16p8_velocity_seed0/step_00250200.pt) | 200 | weights_only | hyperdit_b |
| [group1/g1_jit_b16_clean_p16_seed0/epoch_0200.pt](group1/g1_jit_b16_clean_p16_seed0/epoch_0200.pt) | 200 | weights_only | jit_b16 |
| [group1/g1_jit_b16_velocity_p16_seed0/epoch_0200.pt](group1/g1_jit_b16_velocity_p16_seed0/epoch_0200.pt) | 200 | weights_only | jit_b16 |
| [group1/g1_mhc4_b16_clean_p16_seed0/step_00250200.pt](group1/g1_mhc4_b16_clean_p16_seed0/step_00250200.pt) | 200 | weights_only | mhc_jit_b16_n4 |
| [group1/g1_mhc4_b16_velocity_p16_seed0/epoch_0200.pt](group1/g1_mhc4_b16_velocity_p16_seed0/epoch_0200.pt) | 200 | weights_only | mhc_jit_b16_n4 |
| [group1/g1_mhc4_b16_velocity_p8_seed0/epoch_0200.pt](group1/g1_mhc4_b16_velocity_p8_seed0/epoch_0200.pt) | 200 | weights_only | mhc_jit_b16_n4_p8 |
| [group1/g1_sihc_b16_velocity_p4_ctx32s4_seed0/step_00250200.pt](group1/g1_sihc_b16_velocity_p4_ctx32s4_seed0/step_00250200.pt) | 200 | weights_only | sihc_1x12_d768_b4_ctx32s4 |
| [group1/g1_sihc_b16_velocity_p4_factorized_seed0/step_00250200.pt](group1/g1_sihc_b16_velocity_p4_factorized_seed0/step_00250200.pt) | 200 | weights_only | shc_1x12_d768_b4 |
| [group1/g1_sihc_b16_velocity_p4_seed0/step_00250200.pt](group1/g1_sihc_b16_velocity_p4_seed0/step_00250200.pt) | 200 | weights_only | shc_1x12_d768_b4 |
| [group1/g1_sihc_b16_velocity_p8_seed0/step_00250200.pt](group1/g1_sihc_b16_velocity_p8_seed0/step_00250200.pt) | 200 | weights_only | shc_1x12_d768_b8 |
| [group1/g1_sihc_sublayer_b16_velocity_p4_ctx32s4_gmuon_seed0/step_00250200.pt](group1/g1_sihc_sublayer_b16_velocity_p4_ctx32s4_gmuon_seed0/step_00250200.pt) | 200 | weights_only | sihc_sublayer_1x12_d768_b4_ctx32s4 |
| [group1/g1_sihc_sublayer_b16_velocity_p4_ctx32s4_seed0/step_00250200.pt](group1/g1_sihc_sublayer_b16_velocity_p4_ctx32s4_seed0/step_00250200.pt) | 200 | weights_only | sihc_sublayer_1x12_d768_b4_ctx32s4 |
| [group1/g1_sihc_sublayer_b16_velocity_p4_seed0/step_00250200.pt](group1/g1_sihc_sublayer_b16_velocity_p4_seed0/step_00250200.pt) | 200 | weights_only | sihc_sublayer_1x12_d768_b4 |
| [group1/g1_sihc_sublayer_b16_velocity_p8_seed0/step_00250200.pt](group1/g1_sihc_sublayer_b16_velocity_p8_seed0/step_00250200.pt) | 200 | weights_only | sihc_sublayer_1x12_d768_b8 |
| [group1/g1_sihc_sublayer_b_n16_rank48_meanout_velocity_seed0/step_00250200.pt](group1/g1_sihc_sublayer_b_n16_rank48_meanout_velocity_seed0/step_00250200.pt) | 200 | weights_only | sihc_sublayer_b_n16_rank48_meanout |
| [group1/g1_static_hc4_b16_velocity_p8_seed0/step_00250200.pt](group1/g1_static_hc4_b16_velocity_p8_seed0/step_00250200.pt) | 200 | weights_only | static_hc_jit_b16_n4_p8 |
| [raev2/imagenet_dinov3l_k7_latent_sihc_p8_d32_h1024_n4_seed42/ep-0000080.pt](raev2/imagenet_dinov3l_k7_latent_sihc_p8_d32_h1024_n4_seed42/ep-0000080.pt) | 80 | weights_only | additional_research |
| [resume/external_group1/bsize_deco_b16_velocity_seed0/step_00250200.pt](resume/external_group1/bsize_deco_b16_velocity_seed0/step_00250200.pt) | 200 | resume | available |
| [resume/external_group1/bsize_dip_b16_velocity_seed0/step_00250200.pt](resume/external_group1/bsize_dip_b16_velocity_seed0/step_00250200.pt) | 200 | resume | available |
| [resume/external_group1/bsize_pixeldit_b16_velocity_seed0/step_00250200.pt](resume/external_group1/bsize_pixeldit_b16_velocity_seed0/step_00250200.pt) | 200 | resume | available |
| [resume/flagship/sihc_5x8_d1024_velocity_repa_dinov3l/step_00550440.pt](resume/flagship/sihc_5x8_d1024_velocity_repa_dinov3l/step_00550440.pt) | 440 | resume | local_thc_v2_fused_5x8_adaln_d1024_b4 |
| [resume/flagship/sihc_8x12x8_d1152_velocity_repa_dinov3l/step_00100080.pt](resume/flagship/sihc_8x12x8_d1152_velocity_repa_dinov3l/step_00100080.pt) | 80 | resume | shc_8x12x8_d1152_b4 |
| [resume/group1/g1_fraction_hc_b16_velocity_p4_seed0/epoch_0200.pt](resume/group1/g1_fraction_hc_b16_velocity_p4_seed0/epoch_0200.pt) | 200 | resume | available |
| [resume/group1/g1_hyperdit_b_p16p8_velocity_seed0/step_00250200.pt](resume/group1/g1_hyperdit_b_p16p8_velocity_seed0/step_00250200.pt) | 200 | resume | hyperdit_b |
| [resume/group1/g1_jit_b16_clean_p16_seed0/epoch_0200.pt](resume/group1/g1_jit_b16_clean_p16_seed0/epoch_0200.pt) | 200 | resume | jit_b16 |
| [resume/group1/g1_jit_b16_velocity_p16_seed0/epoch_0200.pt](resume/group1/g1_jit_b16_velocity_p16_seed0/epoch_0200.pt) | 200 | resume | jit_b16 |
| [resume/group1/g1_mhc4_b16_clean_p16_seed0/step_00250200.pt](resume/group1/g1_mhc4_b16_clean_p16_seed0/step_00250200.pt) | 200 | resume | mhc_jit_b16_n4 |
| [resume/group1/g1_mhc4_b16_velocity_p16_seed0/epoch_0200.pt](resume/group1/g1_mhc4_b16_velocity_p16_seed0/epoch_0200.pt) | 200 | resume | mhc_jit_b16_n4 |
| [resume/group1/g1_mhc4_b16_velocity_p8_seed0/epoch_0200.pt](resume/group1/g1_mhc4_b16_velocity_p8_seed0/epoch_0200.pt) | 200 | resume | mhc_jit_b16_n4_p8 |
| [resume/group1/g1_sihc_b16_velocity_p4_ctx32s4_seed0/step_00250200.pt](resume/group1/g1_sihc_b16_velocity_p4_ctx32s4_seed0/step_00250200.pt) | 200 | resume | sihc_1x12_d768_b4_ctx32s4 |
| [resume/group1/g1_sihc_b16_velocity_p4_factorized_seed0/step_00250200.pt](resume/group1/g1_sihc_b16_velocity_p4_factorized_seed0/step_00250200.pt) | 200 | resume | shc_1x12_d768_b4 |
| [resume/group1/g1_sihc_b16_velocity_p4_seed0/step_00250200.pt](resume/group1/g1_sihc_b16_velocity_p4_seed0/step_00250200.pt) | 200 | resume | shc_1x12_d768_b4 |
| [resume/group1/g1_sihc_b16_velocity_p8_seed0/step_00250200.pt](resume/group1/g1_sihc_b16_velocity_p8_seed0/step_00250200.pt) | 200 | resume | shc_1x12_d768_b8 |
| [resume/group1/g1_sihc_sublayer_b16_velocity_p4_ctx32s4_gmuon_seed0/step_00250200.pt](resume/group1/g1_sihc_sublayer_b16_velocity_p4_ctx32s4_gmuon_seed0/step_00250200.pt) | 200 | resume | sihc_sublayer_1x12_d768_b4_ctx32s4 |
| [resume/group1/g1_sihc_sublayer_b16_velocity_p4_ctx32s4_seed0/step_00250200.pt](resume/group1/g1_sihc_sublayer_b16_velocity_p4_ctx32s4_seed0/step_00250200.pt) | 200 | resume | sihc_sublayer_1x12_d768_b4_ctx32s4 |
| [resume/group1/g1_sihc_sublayer_b16_velocity_p4_seed0/step_00250200.pt](resume/group1/g1_sihc_sublayer_b16_velocity_p4_seed0/step_00250200.pt) | 200 | resume | sihc_sublayer_1x12_d768_b4 |
| [resume/group1/g1_sihc_sublayer_b16_velocity_p8_seed0/step_00250200.pt](resume/group1/g1_sihc_sublayer_b16_velocity_p8_seed0/step_00250200.pt) | 200 | resume | sihc_sublayer_1x12_d768_b8 |
| [resume/group1/g1_sihc_sublayer_b_n16_rank48_meanout_velocity_seed0/step_00250200.pt](resume/group1/g1_sihc_sublayer_b_n16_rank48_meanout_velocity_seed0/step_00250200.pt) | 200 | resume | sihc_sublayer_b_n16_rank48_meanout |
| [resume/group1/g1_static_hc4_b16_velocity_p8_seed0/step_00250200.pt](resume/group1/g1_static_hc4_b16_velocity_p8_seed0/step_00250200.pt) | 200 | resume | static_hc_jit_b16_n4_p8 |
| [resume/group6/g6_jit_b16_whiten_clean_seed0/step_00250200.pt](resume/group6/g6_jit_b16_whiten_clean_seed0/step_00250200.pt) | 200 | resume | jit_b16 |
| [resume/group6/g6_jit_b16_whiten_velocity_seed0/step_00250200.pt](resume/group6/g6_jit_b16_whiten_velocity_seed0/step_00250200.pt) | 200 | resume | jit_b16 |
| [resume/raev2/imagenet_dinov3l_k7_latent_sihc_p8_d32_h1024_n4_ep25_decay1e4hold_seed42/ep-0000080.pt](resume/raev2/imagenet_dinov3l_k7_latent_sihc_p8_d32_h1024_n4_ep25_decay1e4hold_seed42/ep-0000080.pt) | 80 | resume | additional_research |
| [resume/raev2/imagenet_dinov3l_k7_latent_sihc_p8_d32_h1024_n4_seed42/ep-0000080.pt](resume/raev2/imagenet_dinov3l_k7_latent_sihc_p8_d32_h1024_n4_seed42/ep-0000080.pt) | 80 | resume | additional_research |
| [resume/raev2/imagenet_dinov3l_k7_latent_sihc_p8_d32_h1024_n4_slotmod_fused8_origlr_bf16_unguided_e20_seed42/ep-0000080.pt](resume/raev2/imagenet_dinov3l_k7_latent_sihc_p8_d32_h1024_n4_slotmod_fused8_origlr_bf16_unguided_e20_seed42/ep-0000080.pt) | 80 | resume | additional_research |
| [resume/raev2/sihc_d32_direct4_ctx32s8_concatrms_e80_seed42/ep-0000080.pt](resume/raev2/sihc_d32_direct4_ctx32s8_concatrms_e80_seed42/ep-0000080.pt) | 80 | resume | additional_research |
| [resume/scaling_law/g1_sihc_h_ctx32s8_velocity_norepa_mlp_recompute_projdrop02_seed0/step_00100080.pt](resume/scaling_law/g1_sihc_h_ctx32s8_velocity_norepa_mlp_recompute_projdrop02_seed0/step_00100080.pt) | 80 | resume | sihc_8x12x12_d1280_b4_ctx32s8 |
| [resume/scaling_law/g1_sihc_h_ctx32s8_velocity_norepa_mlp_recompute_seed0/step_00250200.pt](resume/scaling_law/g1_sihc_h_ctx32s8_velocity_norepa_mlp_recompute_seed0/step_00250200.pt) | 200 | resume | sihc_8x12x12_d1280_b4_ctx32s8 |
| [resume/scaling_law/g1_sihc_h_ctx32s8_velocity_norepa_mlp_recompute_seed0/step_00350280.pt](resume/scaling_law/g1_sihc_h_ctx32s8_velocity_norepa_mlp_recompute_seed0/step_00350280.pt) | 280 | resume | sihc_8x12x12_d1280_b4_ctx32s8 |
| [resume/scaling_law/g1_sihc_l_deep_ctx32s12_velocity_norepa_seed0/step_00250200.pt](resume/scaling_law/g1_sihc_l_deep_ctx32s12_velocity_norepa_seed0/step_00250200.pt) | 200 | resume | sihc_4x12_d768_b4_ctx32s12 |
| [resume/scaling_law/g1_sihc_l_wide_ctx32s8_velocity_norepa_seed0/step_00250200.pt](resume/scaling_law/g1_sihc_l_wide_ctx32s8_velocity_norepa_seed0/step_00250200.pt) | 200 | resume | sihc_2x12_d1024_b4_ctx32s8 |
| [scaling_law/g1_sihc_h_ctx32s8_velocity_norepa_mlp_recompute_projdrop02_seed0/step_00100080.pt](scaling_law/g1_sihc_h_ctx32s8_velocity_norepa_mlp_recompute_projdrop02_seed0/step_00100080.pt) | 80 | weights_only | sihc_8x12x12_d1280_b4_ctx32s8 |
| [scaling_law/g1_sihc_h_ctx32s8_velocity_norepa_mlp_recompute_seed0/step_00250200.pt](scaling_law/g1_sihc_h_ctx32s8_velocity_norepa_mlp_recompute_seed0/step_00250200.pt) | 200 | weights_only | sihc_8x12x12_d1280_b4_ctx32s8 |
| [scaling_law/g1_sihc_h_ctx32s8_velocity_norepa_mlp_recompute_seed0/step_00350280.pt](scaling_law/g1_sihc_h_ctx32s8_velocity_norepa_mlp_recompute_seed0/step_00350280.pt) | 280 | weights_only | sihc_8x12x12_d1280_b4_ctx32s8 |
| [scaling_law/g1_sihc_l_deep_ctx32s12_velocity_norepa_seed0/step_00250200.pt](scaling_law/g1_sihc_l_deep_ctx32s12_velocity_norepa_seed0/step_00250200.pt) | 200 | weights_only | sihc_4x12_d768_b4_ctx32s12 |
| [scaling_law/g1_sihc_l_wide_ctx32s8_velocity_norepa_seed0/step_00250200.pt](scaling_law/g1_sihc_l_wide_ctx32s8_velocity_norepa_seed0/step_00250200.pt) | 200 | weights_only | sihc_2x12_d1024_b4_ctx32s8 |

[Machine-readable checkpoint catalog](catalog.json) records revisions, file sizes, configuration and architecture classification.

[Code and reproduction guide](https://github.com/tongtongliang/residual-stream-burden) · [Research archive](https://huggingface.co/TongtongLiang/sihc-research-checkpoints) · [Paper XL + REPA](https://huggingface.co/xiziqiao/sihc-group5-flagship-ckpt)

The DINO teacher is needed only for REPA training. Follow the model-specific loader; blockwise and sublayerwise weights are different architectures.

## Paper flagship

The selected SiHC-XL + REPA checkpoint is hosted by our coauthor: [step_00650520.pt](https://huggingface.co/xiziqiao/sihc-group5-flagship-ckpt/blob/73ad9f390370a1b626ac299089e085c374d361e7/checkpoints/step_00650520.pt). Its recorded primary-EMA Heun-50 evaluation is FID **1.71491**, IS **299.5535**, CFG 2.4, interval [0.1, 0.9], 50,000 samples. Historical checkpoints above retain their own identities and metrics.
