# Reproducing the registered sublayer models

Use `get_preset("B"|"L"|"H"|"XL")` to inspect the recipe and
`build_preset(...)` to construct the matching initialized model. These names
refer to sublayer SiHC with CTX32, direct P4 and a 16×16 workspace. Long
architecture identifiers remain the checkpoint identities.

| Size | Long model identifier | Blocks × width | Heads | Fused stages | CTX start index | REPA |
|---|---|---|---|---|---|---|
| B | `sihc_sublayer_1x12_d768_b4_ctx32s4` | 12×768 | 12 | 12 | 4 | off |
| L | `sihc_sublayer_2x12_d1024_b4_ctx32s8` | 24×1024 | 16 | 12+12 | 8 | off |
| H | `sihc_sublayer_8x12x12_d1280_b4_ctx32s8` | 32×1280 | 16 | 8+12+12 | 8 | off |
| XL | `sihc_sublayer_5x8_d1024_b4_ctx32s8` | 40×1024 | 16 | 8+8+8+8+8 | 8 | DINOv3-L/16 |

## Training and REPA

The configs use AdamW, global batch 1024 and 750600 steps (600 epochs at
1251 steps/epoch). See [training](../training/README.md) for data, objective,
optimizer and resume settings. The [XL flagship guide](../training/FLAGSHIP.md)
provides the DINOv3-L teacher setup and full train/sample/eval commands.
B/L/H disable REPA; XL enables its block-8 projector before CTX insertion.

## Checkpoints

```bash
python -m evaluation.verify_checkpoint --checkpoint /path/to/checkpoint.pt --state_key model
python -m evaluation.verify_checkpoint --checkpoint /path/to/checkpoint.pt --state_key ema
python -m evaluation.verify_checkpoint --checkpoint /path/to/checkpoint.pt --state_key ema2
```

Training resume requires raw model, optimizer, both EMAs and reconstruction
metadata. Inference exports cannot resume training. Load REPA checkpoints
with their saved projector even when sampling; teacher weights are needed
only for training. Block-wise and sublayer-wise checkpoints are distinct
architectures and cannot be interchanged.

The selected flagship checkpoint is epoch 520, step 650520, primary EMA.
Weights are supplied separately. See [evaluation](../evaluation/README.md)
for sampling, FID/IS and portable inference-weight export.
