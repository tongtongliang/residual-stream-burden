# Release preparation checks

Prepared on macOS ARM64 with Python 3.12, using CPU only. The model/kernel implementation is based on the previously checked anonymous supplement. This packaging session did not run GPU training, inference, FID evaluation or download multi-GB checkpoint tensors.

- **25 tests passed:** independent toy experiments and release audit tests.
- Tiny plain JiT and dynamic mHC reference models passed CPU forward/backward and finite-gradient checks.
- Cached DINOv2-B, UCF101 tubelet and learned-long-skip figures rendered successfully. The tubelet ranks reproduce 9 / 29 / 151; the long-skip plot uses 246 recorded epochs.
- Python source syntax, guide links and catalog identity were checked. The archived registry insertion fragment is stored as `.txt`, not as an executable module.
- No HF/GitHub credential strings were found in the staged text files. Checkpoints and dataset images are excluded from Git.

Full model/kernel tests require the Linux CUDA/Triton environment documented in `requirements-tested.txt`. On this Mac, full-suite collection stops at the missing Triton dependency; GPU correctness was not re-verified during packaging. CPU-only toy and cached plotting checks are independent of Triton.

The catalog binds paper flagship metrics to the coauthor's exact step-650520 file and revision. Historical blockwise runs, early checkpoints and the separate Group 8 long-skip archive remain distinguishable. Missing static-scalar/external-decoder factories and missing paper scaling/DINO control weights are listed in the experiment/checkpoint guides.

## Repository reorganization

The release now has `sihc/` for model training/inference/evaluation and `research/` for experiments and measurements. Topic names replace historical report numbers. Package building, guide links, CPU tests and cached renderers were rechecked after moving files; import names and checkpoint model identities are preserved.

Post-reorganization checks: **30 CPU tests passed**; the package wheel builds and contains model, training, evaluation and archived long-skip modules. Cached gain, whitening, embedding, endpoint, DINO, video, long-skip and scaling renderers run at their new locations. All 10 model implementation files are byte-identical to the previous release.

Editable installation was checked from the repository root: `sihc`, `training`, `evaluation` and `research` resolve to their intended modules.
