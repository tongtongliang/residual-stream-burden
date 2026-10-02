# Checkpoints

The [catalog](../pretrained/checkpoints.json) pins each file to its source revision and records architecture, prediction target and format. Existing archives retain their file paths.

| Collection | Purpose |
|---|---|
| [Coauthor flagship](https://huggingface.co/xiziqiao/sihc-group5-flagship-ckpt) | Paper SiHC-XL + REPA, selected step 650520, primary EMA |
| [Research checkpoints](https://huggingface.co/TongtongLiang/sihc-research-checkpoints) | JiT/mHC/SiHC controls, whitening, historical scaling, external controls |
| [Earlier flagship copies](https://huggingface.co/TongtongLiang/sihc-group5-flagship-ckpt) | Same flagship architecture at earlier training steps |
| [H sublayer archive](https://huggingface.co/TongtongLiang/sihc-sublayer-h-wide-ckpt) | Available intermediate H checkpoint; metric steps are indexed separately |
| [Group 8 long skip](https://huggingface.co/TongtongLiang/sihc-group8-longskip) | Separate long-skip training archive |

```bash
python sihc/tools/download_checkpoint.py --id sihc-xl-repa --output checkpoints
python sihc/tools/download_checkpoint.py --repo TongtongLiang/sihc-research-checkpoints --file <catalog-file-path> --output checkpoints
```

Full resume files and weights-only files serve different purposes. Use the matching model and explicit `model`, `ema` or `ema2` state. The catalog uses existing manifest descriptions where available; large checkpoint tensors were not downloaded and re-inspected during this packaging session.

The original paper coefficient-dynamics run is in `research/long_skip`; Group 8 is a separate experiment. The withdrawn write-only control is excluded from the paper evidence index. Some paper L/H scaling and DINOv2-B control weights are not present in the currently inventoried HF collections; their results remain available as cached measurements.
