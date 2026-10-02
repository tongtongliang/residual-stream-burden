# Release scope and main-text evidence

The release has two goals: provide SiHC training, inference and evaluation, and make the main paper's research findings inspectable through experiment settings, measurements and analysis code. Research coverage does not require retraining every historical model or rerunning every appendix experiment.

## Follow the main results

| Main-text result | Evidence and entry point |
|---|---|
| Clean patch variance is spectrally concentrated; whitening removes the clean-prediction advantage | [Patch PCA, embeddings and gain](../patch_embedding/README.md), [whitening spectra and FID/IS](../whitening/README.md); `research/plotting/render_geometry.py` |
| Prediction targets shape input filtering and learned representations | Raw/whitened alignment and gain via `research/plotting/render_target_alignment_gain.py`; [linear-probe measurements](../linear_probes/) and `research/plotting/render_plain_probes.py` |
| Decoupled architectures recover selective filtering in their main streams | [Embedding comparisons](../patch_embedding/README.md), [endpoint measurements](../semantic_endpoints/), `research/plotting/render_embedding_filtering.py` |
| Added residual bandwidth helps velocity prediction most | [JiT/mHC settings and results](../pixel/README.md), [DINOv2-B comparison](../rae/README.md), [recorded pixel FIDs](../reference_tables/group1_epoch200.csv) |
| Spatial assignment and feature-wise access yield SiHC | [Control/ablation results](../reference_tables/group1_epoch200.csv), [SiHC model and workflows](../../sihc/README.md), `research/plotting/render_sihc_probes.py` |
| SiHC reaches competitive generation quality | [Published checkpoint and 50K FID/IS evaluation](../../sihc/evaluation/README.md) |

Run plotting commands from the repository root. Cached analysis runs on CPU; training and model evaluation need their documented GPU environment. Distinguish raw-weight representation diagnostics from EMA generation metrics, and preserve run identities when comparing historical blockwise and paper sublayerwise models.

## Additional available material

Toy experiments, learned long skips, sensitivity measurements, video/DINO spectra, scaling and fusion benchmarks remain available as supporting material. Their completeness is not a release prerequisite. The workflow inventory below records what is available for users who want to run new experiments; it is not a list of release blockers.

| Paper component | Included material | Remaining requirement for a new run |
|---|---|---|
| SiHC-XL + REPA, FID 1.71491 / IS 299.5535 | Model, training/resume, 50K FID/IS evaluator, exact checkpoint/revision, reference-statistics download and evaluation command | Linux CUDA/Triton; GPU runtime validation of this reorganized release. Training additionally needs ImageNet and DINOv3-L weights |
| SiHC B/L/H scaling | Sublayer CTX32 models, training configs, evaluation code and cached scaling results | Exact paper L/H 200-epoch checkpoints are not in the inventoried collections; historical blockwise weights are different models |
| Plain JiT and dynamic mHC pixel controls | Training configs, model implementations, RGB evaluator, cached metrics and checkpoint catalog | ImageNet for retraining; CUDA/Transformer Engine for the original mHC backend. The exact historical TE revision is not recovered |
| Static scalar-access ablation | Recorded configuration, metrics and checkpoint catalog | Original `static_hc_jit_b16_n4_p8` factory is not recovered |
| Whitened clean / velocity | Target and embedding spectra, gain measurements, evaluation records, figure renderer | Original whitening-aware training and final sampling pipeline; generic RGB evaluation does not apply |
| DINOv2-B plain / mHC | RAE training wrapper/configs, spectra, probes and paper FID records | External RAE/encoder/decoder assets, the four final control checkpoints and final guided Heun-50 sampling entry |
| B-size DeCo / DiP / PixelDiT | FID trajectories, representation measurements and analysis caches | Original external B-size training/model builder is not recovered |
| Learned long skip | Paper coefficient/FID measurements and renderers; separate Group 8 training source | Original dataset/checkpoint dependencies for retraining. Group 8 and the paper coefficient experiment are separate runs; see [long-skip provenance](../long_skip/README.md) |
| PCA, gain, endpoint spectra, probes and sensitivity | Topic-local arrays/tables, figure renderers and recovered measurement scripts | Cached figures run on CPU. Remeasurement needs the corresponding models, weights and datasets; source snapshots can reference external caches |
| Toy experiments | Standalone training/diagnostics and original FCN caches | CPU Python dependencies; choose the intended toy configuration |
| Kernel fusion | Implementation, benchmark scripts and recorded measurements | Supported NVIDIA GPU and matching CUDA environment for new performance measurements |

## Validation

The SiHC model/kernel implementation comes from the previously checked anonymous package. This release's CPU, installation, plotting and protocol checks are recorded in [release validation](RELEASE_VALIDATION.md). This packaging session did not run new GPU evaluations.
