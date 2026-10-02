# Reproduction coverage

This index separates rerunning a model from rerendering its recorded measurements. Paths are relative to the repository root. The paper's sampling settings are centralized in the [evaluation guide](../../sihc/evaluation/README.md).

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

## What has been checked

The previously checked anonymous package supplies the SiHC model/kernel implementation. Release preparation checks installation/layout, CPU tests, figure rendering and protocol consistency; it does not substitute for a new GPU run. The current checks and their exact limits are recorded in [release validation](RELEASE_VALIDATION.md).

## Before calling this a complete paper-reproduction release

Recover the missing scalar-access and external-decoder factories, the whitening and guided RAE evaluators, and the missing final scaling/DINO weights. Pin the historical Transformer Engine build when available. On a GPU machine, strictly load the published EMA checkpoint, run a small sampling smoke test, and then rerun the documented 50K evaluation. Those checks are separate from CPU packaging validation.
