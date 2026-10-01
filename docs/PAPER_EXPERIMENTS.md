# Paper experiments and cached figures

The analysis cache preserves measurement values. Original measurement scripts inside `analysis_reports` are provenance snapshots; several depend on the original dataset caches or external model repositories. The commands below render existing measurements without those resources or a GPU.

| Evidence | Cached measurements | Renderer |
|---|---|---|
| Clean-patch spectrum and whitening | `paper/fig_table/result_statistic/section03_analysis`, `analysis_reports/14_whitening_patch_embedding` | `render_geometry.py`, `render_target_alignment_gain.py` |
| Input filtering | `analysis_reports/01_patch_embedding_matrix`, `14_whitening_patch_embedding` | `render_embedding_filtering.py` |
| Semantic endpoints | `analysis_reports/03_semantic_endpoint_patch_pca` | `render_endpoint_spectra.py` |
| Linear probes and noise sensitivity | `analysis_reports/04_hidden_noise_linear_probe`, `12_normalized_representation_sensitivity` | `render_plain_probes.py`, `render_normalized_sensitivity.py` |
| DINOv2-B spectrum | `analysis_reports/16_dinov2b_input_patch_spectrum` | `render_dino_target_spectra.py` |
| Learned long skip and FID dynamics | `analysis_reports/16_jit_b16_long_skip_velocity` | `render_learned_long_skip.py` |
| UCF101 tubelet spectrum | `analysis_reports/18_ucf101_raw_tubelet_spectrum` | `render_video_tubelet_spectra.py` |
| Controlled decoder FID records | `analysis_reports/17_bsize_decoupled_fid` | CSV/JSON evaluation records |
| Kernel fusion | `analysis_reports/18_kernel_fusion` | [Benchmark commands](kernel_fusion/README.md) |

```bash
pip install numpy matplotlib pandas scipy
python paper/fig_table/code/render_dino_target_spectra.py
python paper/fig_table/code/render_video_tubelet_spectra.py
python paper/fig_table/code/render_learned_long_skip.py
```

Figures are written under `paper/figures/`. No training, checkpoint inference or new FID evaluation is performed by these commands. [Source manifest](source_manifest.json) records the imported files and the training-source revision. The manuscript, private notes and research Git history are not included.

## Training the controls

JiT-B clean/velocity controls use full-rank patch embeddings, without bottleneck or context tokens. The research trainer supports the original dynamic mHC controls and an explicit reference backend implementing the same routing:

```bash
torchrun --standalone --nproc_per_node=4 -m research.pixel.train --config research/pixel/configs/jit_b16_velocity.json --data_path /path/to/imagenet --run_dir runs/jit-v
torchrun --standalone --nproc_per_node=8 -m research.pixel.train --config research/pixel/configs/mhc_n4_velocity.json --mhc_backend reference --data_path /path/to/imagenet --run_dir runs/mhc-v
```

The baseline recipes preserve the archived per-device batch and world size; keep their product (including gradient accumulation) at global batch 1024.

[SiHC training](../training/README.md) uses the maintained supplement trainer. [RAE integration](../research/rae/README.md) and [long-skip training](../research/long_skip/README.md) use their own original pipelines.

The static scalar-access control factory and the external B-size decoder training builder are not recovered in this source snapshot. Their configurations, checkpoints and measurement records are indexed, but they are not advertised as runnable factories. DeCo, DiP and PixelDiT retain their upstream implementations; no replacement architecture is inferred from result files.
