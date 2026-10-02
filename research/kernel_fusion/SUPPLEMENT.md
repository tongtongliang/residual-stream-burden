# Appendix use

Use the sihc/training/inference tables in [REPORT.md](REPORT.md), or the figures and
LaTeX table in `paper_assets/`. `render_assets.py` regenerates them from local
CSV files without running a model. The full numeric data are in `data/`.

The existing manuscript appendix contains an older P8 four-GPU table.
Replace it with these matched P4 results rather than retaining its 3.33× claim.
The manuscript itself has not been edited. In the equations, one routing
event is a whole block for block-wise SiHC and an attention/MLP branch for
sublayer-wise SiHC. Exact reordering is an algebraic statement; BF16 can round
differently. Avoid interpreting logical single materialization as a measured
count of physical memory transactions.

Suggested caption:

> Compiled SiHC with and without fused routing on H100 80GB at batch 128,
> with FP32 parameters and BF16 autocast. Training uses default compilation
> followed by one-rank DDP, AdamW and two EMAs; inference uses reduce-overhead
> compilation for one denoiser forward. B/L use no recompute and XL/H use MLP
> recompute in the training table. Results are medians of three fresh processes;
> error bars span the minimum and maximum process medians.

Submit the separately prepared anonymous source ZIP, not the private research
repository. It defaults to sublayer SiHC-B and includes runnable training,
evaluation and benchmark instructions. Weights, data and raw logs are separate.
The [ICLR author guidelines](https://iclr.cc/Conferences/2027/AuthorGuidelines)
apply to supplementary anonymity and the manuscript's AI-use statement.
