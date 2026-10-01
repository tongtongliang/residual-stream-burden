# B-size decoupled-model FID histories

Retrieved on 2026-09-24 from the authenticated W&B project
[residual-stream-training-dynamics](https://wandb.ai/tol011-uc-san-diego/residual-stream-training-dynamics).

The three runs are `bsize_deco_b16_velocity_seed0`, `bsize_dip_b16_velocity_seed0`, and `bsize_pixeldit_b16_velocity_seed0`. These are our controlled B-size experiments, not the published XL results.

## Files and extraction

- `*_wandb_export.csv`: exact column names and numerical cells transcribed from each `eval/fid_50k` chart's CSV Export Preview, with x-axis `eval/global_step`. These are local reconstructions of the visible export tables, not downloaded history Parquet files.
- `fid_history.csv`: tidy combined records, retaining full FID precision and W&B history-step metadata.
- `run_metadata.json`: selected configuration and evaluation fields from the rendered W&B run files and the existing evaluator source.

Each export contains five distinct evaluation checkpoints. Epochs are derived as `eval/global_step / 1251`, consistent with final `eval/epoch = 200` at step 250200. DeCo's first checkpoint was re-logged: W&B reports internal step mean 25362 with min 684 and max 50040. Its FID min/max coincide exactly. Align by checkpoint `eval/global_step`, never by internal `_step`. MIN/MAX columns describe the chart export, not uncertainty across training seeds.

The final run summaries independently match all three final FIDs: DeCo 8.069581201761252, DiP 13.305966600360875, PixelDiT 6.330378998614265.

## Protocol evidence

All configs specify a 12-block, width-768 semantic backbone, 250200 training steps, learning rate 0.0002, zero weight decay, 6255 warmup steps and primary EMA decay 0.9999. All final summaries record zero REPA loss, consistent with the controlled no-REPA protocol documented in the manuscript.

DiP and PixelDiT W&B configs explicitly record evaluator arguments: `--state-key ema --num-samples 50000 --steps 50 --cfg 2.9 --interval-min 0.1 --interval-max 1.0`, using JiT `jit_in256_stats.npz`. The archived evaluator at `paper/fig_table/result_statistic/reproducibility/r52/source/sihc_artifacts/external_semantic_flow/code/evaluate_bsize_decoupled.py` calls `sample_heun`. DeCo's latest config writer is a manual metric re-log, so it does not independently expose the sampling command; its protocol follows the shared controlled-pixel experiment configuration already documented in the manuscript.

The appendix table is `tab:bsize-decoupled-fid` in `paper/main.tex`. It reports EMA generation results; the corresponding representation diagnostics use raw weights. No training, inference or FID recomputation was performed for this extraction.
