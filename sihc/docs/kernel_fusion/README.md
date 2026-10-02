# Kernel fusion notes

SiHC preserves a high-resolution spatial carrier while attention and MLPs run
on a smaller workspace. The fused kernels implement the carrier read/write
recurrence for both **block-wise** and **sublayer-wise** connections. Attention,
RMSNorm, AdaLN and SwiGLU remain ordinary PyTorch operations; `torch.compile`
can optimize the surrounding model.

The production models take 256 × 256 images and use a 16 × 16 workspace.
P4 has a 64 × 64 carrier with 16 spatial slots per workspace cell; P8 has a
32 × 32 carrier with four slots. Slot identities remain fixed. Read/write
weights are static, learned and feature-wise; there is no cross-slot mixing.

## Two connection topologies

| Topology | One connection event | Fused stage sizes | Connection events per stage |
|---|---|---|---|
| Block-wise P4 | Complete attention + MLP block | 8 or 12 blocks | 8 or 12 |
| Block-wise P8 | Complete attention + MLP block | 12 blocks | 12 |
| Sublayer-wise P4/P8 | Attention or MLP separately | 8 or 12 blocks | 16 or 24 |

In the block-wise model, the MLP consumes the attention-updated workspace
inside the block. The combined block update is then written to the carrier.
In the sublayer-wise model, attention writes to the carrier first, and the MLP
reads that updated carrier using its own read weights. This changes the
model's residual topology. Checkpoints cannot be interchanged between the two
topologies. `depth` and `stage_sizes` always count Transformer blocks.

## Equivalent stage recurrence

Let `X_i[n,s,c]` be the carrier before connection event `i`, where `n` combines
batch and workspace position, `s` is a spatial slot, and `c` is a feature.
`alpha_i[c,s]` and `beta_i[c,s]` are that event's read and write weights.
The literal dense recurrence is:

```text
z_i[n,c]       = sum_s alpha_i[c,s] * X_i[n,s,c]
dz_i          = F_i(z_i)
X_(i+1)[n,s,c] = X_i[n,s,c] + beta_i[c,s] * dz_i[n,c]
```

`F_i` is the gated block update or a gated attention/MLP update, depending on
topology. Identity carry lets us express the entire stage in terms of its
input `X_0` and workspace updates:

```text
base_i[n,c]  = sum_s alpha_i[c,s] * X_0[n,s,c]
gamma[i,j,c] = sum_s alpha_i[c,s] * beta_j[c,s]
z_i[n,c]     = base_i[n,c] + sum_(j<i) gamma[i,j,c] * dz_j[n,c]
X_out[n,s,c] = X_0[n,s,c] + sum_i beta_i[c,s] * dz_i[n,c]
```

The fast path follows these equations:

1. **Split base read:** read the stage carrier and produce one separate
   workspace tensor per event. Separate outputs avoid a forward stack/slice
   boundary in the compiled consumer.
2. **Active triangular accumulation:** combine each base read with only the
   already-computed workspace updates. Legacy block-wise operators specialize
   each active prefix length; tuple operators receive only the active update list.
3. **Final write:** accumulate all event writes for a carrier tile, add the
   stage input, and store the dense stage output once.

Equivalence is algebraic. BF16 casts and reassociation change rounding sites,
so numerical tests use tolerances instead of requiring bitwise equality.

## Backward and memory

Every fused forward operator has a registered autograd rule. For fixed
connection weights, the carrier read and write are adjoints. For example,
with upstream workspace gradient `g_i` and carrier gradient `h`:

```text
read:   dX_0[n,s,c]   = sum_i alpha_i[c,s] * g_i[n,c]
        dalpha_i[c,s] = sum_n X_0[n,s,c] * g_i[n,c]
write:  ddz_i[n,c]   = sum_s beta_i[c,s] * h[n,s,c]
        dbeta_i[c,s] = sum_n dz_i[n,c] * h[n,s,c]
```

The final write also passes `h` directly to `X_0` through identity carry.
Triangular accumulation passes its upstream gradient to the base read,
multiplies it by each active `gamma` for update gradients, and reduces its
product with each update for `dgamma`. PyTorch differentiates `gamma` into
both read and write weights, adding those terms to the gradients above.

Legacy block-wise backward uses specialized Triton adjoints and some stacked
workspace-gradient buffers. The tuple backend shared by sublayer and optional
block-wise routing reuses read/write kernels; its small weight reductions
accumulate in FP32. Tuple triangular update
gradients remain visible to AOTAutograd/Inductor, allowing multiplication to
fuse with gradient accumulation. The gamma reduction stays in Triton.
Floating-point atomic reductions can produce small run-to-run differences.

Fusion removes dense intermediate carriers at every connection event. It
still retains workspace base reads, updates, branch activations, saved tensor
references and backward buffers. For `E` events, the base/update storage
scales with `E × batch × workspace_tokens × channels`; gamma has
`E × E × channels` elements. Doubling events approximately quadruples the
triangular pair count. The complete carrier does not stay on chip throughout
a stage, and the kernels do not remove attention/MLP activation memory.

**Activation recomputation is off by default.** Optional MLP checkpointing
recomputes workspace MLPs and leaves the routing schedule outside the
checkpoint region. Full-branch checkpointing applies only to block-wise
models; sublayer models reject it. Compare actual peak allocation and step
time before enabling recomputation for a workload.

## Layout, context tokens and implementation boundaries

The carrier uses packed contiguous NHWC storage. A workspace update uses
packed `[batch, workspace_tokens, channels]` storage. Connection parameters
use `[channels, slots]`, stacked as `[events, channels, slots]`. The tuple
read temporarily transposes this small stack to `[events, slots, channels]`
for coalesced loads; checkpoint parameter layouts stay unchanged.

Context variants create class-prefix workspace tokens with a learned
`[1, context_length, channels]` position embedding. Prefixes share the
Transformer projections with spatial tokens and receive ordinary residual
updates. They never enter spatial read/write. After removing prefixes,
the spatial update must be made `.contiguous()` before a fused kernel call;
the slice otherwise retains a larger batch stride. Both topologies preserve
this boundary.

The tuple BF16 P4 read/write paths use per-channel Tensor Core products
with FP32 accumulation on NVIDIA compute capability 8.0 or newer, when
sufficient shared memory is available. Other dtypes, P8 routing, and devices
that do not meet these requirements use SIMT paths. These choices
are fixed implementation choices, not a promise of equal performance across
GPU architectures. The sublayer direct patch stem uses patchify plus GEMM
with the same stored Conv2d projection parameters and optimizer-state layout.

| Source | Responsibility |
|---|---|
| [`kernels.py`](../../models/sihc/kernels.py) | Legacy block-wise fused operators and backward |
| [`blockwise_kernels.py`](../../models/sihc/blockwise_kernels.py) | 8/12-event block-wise interface to the shared tuple kernels |
| [`sublayer_kernels.py`](../../models/sihc/sublayer_kernels.py) | Shared 8/12/16/24-event tuple kernels, fake implementations and autograd |
| [`model.py`](../../models/sihc/model.py) | Block-wise schedule, carrier geometry and context handling |
| [`sublayer.py`](../../models/sihc/sublayer.py) | Independent attention/MLP routing and direct GEMM patch stem |
| [`reference.py`](../../models/sihc/reference.py) | Readable PyTorch recurrence and non-context reference models |

The relative source links above resolve from the repository's `docs` tree.
Production fused execution requires CUDA and Triton. Non-context models have
a PyTorch reference schedule for CPU diagnostics. `enable_reference_schedule`
does not support context models or the write-only control. It is a diagnostic
path, not an automatic CPU fallback for every registered model.

Legacy block-wise kernels use `torch.library.triton_op`; tuple kernels use
`torch.library.custom_op` with fake implementations and registered autograd.
This preserves tuple-pointer launches under fullgraph compilation. Keep the
kernel registrations and their backward/fake implementations together when
modifying or extracting the implementation.

## Block-wise tuple backend

The optional `kernel_backend="tuple"` ports the sublayer routing implementation
to the same block-wise equations. It preserves a complete attention/MLP block
as one event, CTX prefix handling, parameter names/shapes, gamma construction,
and separate beta casts for gamma and final write. Raw/EMA weights and optimizer
state keep their existing schema. This is an implementation choice within the
block-wise topology; it does not convert a checkpoint to sublayer routing.

The port removes the stacked workspace gradients in read backward, shares
coalesced read-weight storage and FP32 reductions, uses the BF16 P4 MMA path,
and passes active update lists without zero placeholders. Block-wise MMA uses
16 event lanes for 8/12 events; sublayer retains 32 lanes for 16/24 events.
The shared-memory guard requires at least 72 KiB per block for the 16-lane
path and 136 KiB for the 32-lane path; otherwise routing falls back to SIMT.
These are kernel launch requirements, not complete-model memory measurements.
The tensor-core final write accumulates updates before adding the carrier,
so its BF16 rounding can differ from the legacy ordered accumulation.

```python
from sihc import build_model
model = build_model("sihc_1x12_d768_b4", kernel_backend="tuple")
```

Training, sampling and evaluation accept `--kernel_backend tuple` or `legacy`.
Training checkpoints record the resolved choice. With no override, loading
follows checkpoint metadata; older checkpoints and fresh models use `legacy`.
The anonymous inference exporter retains this validated choice. Explicit
`tuple` selection is rejected for sublayer, reference and write-only presets,
which keep their own schedules. Activation recomputation remains off.

## Validation

Run from the repository root after installation. The CPU checks exercise
reference equations, geometry, parameter layouts and checkpoint round trips:

```bash
python -m pytest -q tests/test_reference_variants.py tests/test_sublayer_model.py
```

On an allocated CUDA GPU with sufficient free memory, enable the GPU tests:

```bash
SIHC_RUN_CUDA_TESTS=1 python -m pytest -q \
  tests/test_blockwise_kernels.py tests/test_blockwise_tuple_kernels.py \
  tests/test_p8_kernels.py \
  tests/test_sublayer_kernels.py tests/test_sublayer_context.py \
  tests/test_runtime_compatibility.py
```

The selection checks FP32/BF16 forward and backward, channel/tile tails,
packed-layout rejection, unused sublayer read outputs, dense-carrier
references, full-model compile, context handling, and checkpoint forward
passes. Individual tests cover different subsets; passing CPU tests alone
does not validate CUDA execution. Before changing numerical kernels, verify
all affected inputs and parameter gradients against a literal reference with
nonzero gates and output weights. Zero-initialized branches can mask errors.

## Run a benchmark

Use a fresh process for each implementation on the same allocated GPU:

```bash
CUDA_VISIBLE_DEVICES=0 PYTORCH_ALLOC_CONF=expandable_segments:True \
  python -m bench.fusion_profile --size H --frequency sublayer \
  --implementation fused --phase train --checkpoint-mode mlp \
  --batch 128 --repeat 0 --output-dir ../fusion-results
```

Select `literal`, `factored` or `fused`; `--frequency block` selects the
block-wise tuple backend. Training uses default compilation, one-rank DDP,
AdamW and two EMAs. Inference (`--phase inference`) uses reduce-overhead
compilation and measures one model forward. Select training recompute with
`--checkpoint-mode none|mlp`. The benchmark excludes the REPA teacher and
projector, data loading, compilation and sampling loops. Output includes
latency and allocated/reserved CUDA memory; keep generated files outside
the source package. Use `--help` for all options.
