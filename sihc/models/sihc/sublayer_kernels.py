"""Shared tuple-pointer SiHC kernels for 8/12/16/24 connection events.

The read and write are adjoints, so their backward passes reuse the same
kernels. Workspace updates are passed by pointer, never stacked in forward.
Only the small connection-weight gradients use FP32 atomic reductions.
The historical sublayer operator names are retained for compatibility;
block-wise schedules use the validated adapters in ``blockwise_kernels``.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch import Tensor
from torch.library import custom_op, register_autograd

from .kernels import _require_packed


@triton.jit
def _high_offsets(m, c, s, C: tl.constexpr, G: tl.constexpr, K: tl.constexpr):
    n = m % (G * G)
    return (((m // (G * G) * G * K + n // G * K + s // K)
             * G * K + n % G * K + s % K) * C + c)


@triton.jit
def _read(X, A, OUT, M: tl.constexpr, C: tl.constexpr, G: tl.constexpr,
          K: tl.constexpr, BM: tl.constexpr, BC: tl.constexpr):
    # A is a transient [event, slot, channel] copy: channel loads coalesce.
    # Keep the slot loop rolled. Fully unrolling 24 * 16 coefficient loads
    # causes expensive scalar loads/hoisting and spills with larger tiles.
    m = tl.program_id(0) * BM + tl.arange(0, BM)
    c = tl.program_id(1) * BC + tl.arange(0, BC)
    mask = (m[:, None] < M) & (c[None, :] < C)
    acc = ()
    for l in tl.static_range(len(OUT)):
        acc += (tl.full((BM, BC), 0, tl.float32),)
    for s in range(K * K):
        x = tl.load(X + _high_offsets(m[:, None], c[None, :], s, C, G, K),
                    mask, other=0).to(tl.float32)
        updated = ()
        for l in tl.static_range(len(OUT)):
            a = tl.load(A + (l * K * K + s) * C + c, c < C, other=0).to(tl.float32)
            updated += (acc[l] + x * a[None, :],)
        acc = updated
    for l in tl.static_range(len(OUT)):
        tl.store(OUT[l] + m[:, None] * C + c[None, :], acc[l], mask)


@triton.jit
def _read_bf16_mma(X, A, OUT, M: tl.constexpr, C: tl.constexpr,
                   G: tl.constexpr, K: tl.constexpr, BE: tl.constexpr = 32):
    # Each channel is an independent tiny [token, slot] @ [slot, event]
    # matrix product. Batch 64 adjacent channels so carrier loads coalesce.
    # BF16 operands are exact inputs; MMA accumulation remains FP32.
    m = tl.program_id(0) * 16 + tl.arange(0, 16)
    c = tl.program_id(1) * 64 + tl.arange(0, 64)
    s = tl.arange(0, 16)
    event = tl.arange(0, BE)
    hi = _high_offsets(m[None, :, None], c[:, None, None],
                       s[None, None, :], C, G, K)
    x = tl.load(X + hi, (m[None, :, None] < M) &
                (c[:, None, None] < C) & (s[None, None, :] < K * K), other=0)
    alpha = tl.load(A + (event[None, None, :] * K * K + s[None, :, None]) * C
                    + c[:, None, None], (c[:, None, None] < C) &
                    (s[None, :, None] < K * K) &
                    (event[None, None, :] < len(OUT)), other=0)
    value = tl.dot(x, alpha)
    # Store directly into the separate base-read tensors: no stacked output
    # or host-created pointer table is materialized.
    ptr = tl.full((BE,), 0, tl.uint64)
    for i in tl.static_range(len(OUT)):
        ptr = tl.where(event == i, OUT[i].to(tl.uint64), ptr)
    ptr = ptr.to(tl.pointer_type(OUT[0].dtype.element_ty))
    tl.store(ptr[None, None, :] + m[None, :, None] * C + c[:, None, None],
             value, (m[None, :, None] < M) & (c[:, None, None] < C) &
             (event[None, None, :] < len(OUT)))


@triton.jit
def _write(X, UPDATES, BETA, OUT, C: tl.constexpr, G: tl.constexpr,
           K: tl.constexpr, ADD_X: tl.constexpr, BC: tl.constexpr):
    m = tl.program_id(0)
    c = tl.program_id(1) * BC + tl.arange(0, BC)
    s = tl.arange(0, K * K)
    hi = _high_offsets(m, c[None, :], s[:, None], C, G, K)
    acc = tl.full((K * K, BC), 0, tl.float32)
    if ADD_X:
        acc = tl.load(X + hi, c[None, :] < C, other=0).to(tl.float32)
    for l in tl.static_range(len(UPDATES)):
        dz = tl.load(UPDATES[l] + m * C + c, c < C, other=0).to(tl.float32)
        beta = tl.load(BETA + (l * C + c[None, :]) * K * K + s[:, None],
                       c[None, :] < C, other=0).to(tl.float32)
        acc += dz[None, :] * beta
    tl.store(OUT + hi, acc, c[None, :] < C)


@triton.jit
def _write_bf16_mma(X, UPDATES, BETA, OUT, M: tl.constexpr, C: tl.constexpr,
                    G: tl.constexpr, K: tl.constexpr, ADD_X: tl.constexpr,
                    BE: tl.constexpr = 32):
    m = tl.program_id(0) * 16 + tl.arange(0, 16)
    c = tl.program_id(1) * 64 + tl.arange(0, 64)
    event = tl.arange(0, BE)
    s = tl.arange(0, 16)
    ptr = tl.full((BE,), 0, tl.uint64)
    for i in tl.static_range(len(UPDATES)):
        ptr = tl.where(event == i, UPDATES[i].to(tl.uint64), ptr)
    ptr = ptr.to(tl.pointer_type(UPDATES[0].dtype.element_ty))
    dz = tl.load(ptr[None, None, :] + m[None, :, None] * C + c[:, None, None],
                 (m[None, :, None] < M) & (c[:, None, None] < C) &
                 (event[None, None, :] < len(UPDATES)), other=0)
    beta = tl.load(BETA + (event[None, :, None] * C + c[:, None, None]) * K * K
                   + s[None, None, :], (c[:, None, None] < C) &
                   (event[None, :, None] < len(UPDATES)) &
                   (s[None, None, :] < K * K), other=0)
    acc = tl.dot(dz, beta)
    hi = _high_offsets(m[None, :, None], c[:, None, None],
                       s[None, None, :], C, G, K)
    mask = (m[None, :, None] < M) & (c[:, None, None] < C) & (s[None, None, :] < K * K)
    if ADD_X:
        # Reassociation relative to the SIMT path: add the carrier after the
        # FP32 dot, still before the one BF16 output cast.
        acc += tl.load(X + hi, mask, other=0).to(tl.float32)
    tl.store(OUT + hi, acc, mask)


@triton.jit
def _weight_grad(X, Z, DA, M: tl.constexpr, C: tl.constexpr,
                 G: tl.constexpr, K: tl.constexpr, BM: tl.constexpr, BC: tl.constexpr):
    # Adjacent CTAs handle different slots of the same token/channel tile,
    # improving reuse of the workspace inputs across connection events. The reduction output is
    # [event, slot, channel] so each atomic instruction is channel-coalesced.
    m = (tl.program_id(0) // (K * K)) * BM + tl.arange(0, BM)
    c = tl.program_id(1) * BC + tl.arange(0, BC)
    s = tl.program_id(0) % (K * K)
    mask = (m[:, None] < M) & (c[None, :] < C)
    x = tl.load(X + _high_offsets(m[:, None], c[None, :], s, C, G, K),
                mask, other=0).to(tl.float32)
    for l in tl.static_range(len(Z)):
        z = tl.load(Z[l] + m[:, None] * C + c[None, :], mask, other=0).to(tl.float32)
        value = tl.sum(x * z, 0)
        tl.atomic_add(DA + (l * K * K + s) * C + c, value, c < C, sem="relaxed")


@triton.jit
def _weight_grad_bf16_mma(X, Z, DA, M: tl.constexpr, C: tl.constexpr,
                         G: tl.constexpr, K: tl.constexpr,
                         BE: tl.constexpr = 32):
    # Per-channel [slot, token] @ [token, event]. Accumulate 16 token tiles
    # before touching the global reduction output; operands/accumulator stay
    # BF16/FP32 respectively. No full workspace or carrier stack is created.
    m = tl.program_id(0) * 256 + tl.arange(0, 16)
    c = tl.program_id(1) * 64 + tl.arange(0, 64)
    s = tl.arange(0, 16)
    event = tl.arange(0, BE)
    ptr = tl.full((BE,), 0, tl.uint64)
    for i in tl.static_range(len(Z)):
        ptr = tl.where(event == i, Z[i].to(tl.uint64), ptr)
    ptr = ptr.to(tl.pointer_type(Z[0].dtype.element_ty))
    acc = tl.full((64, 16, BE), 0, tl.float32)
    for chunk in range(16):
        mi = m + chunk * 16
        hi = _high_offsets(mi[None, None, :], c[:, None, None],
                           s[None, :, None], C, G, K)
        x = tl.load(X + hi, (mi[None, None, :] < M) &
                    (c[:, None, None] < C) & (s[None, :, None] < K * K), other=0)
        z = tl.load(ptr[None, None, :] + mi[None, :, None] * C + c[:, None, None],
                    (mi[None, :, None] < M) & (c[:, None, None] < C) &
                    (event[None, None, :] < len(Z)), other=0)
        acc = tl.dot(x, z, acc)
    tl.atomic_add(DA + (event[None, None, :] * K * K + s[None, :, None]) * C
                   + c[:, None, None], acc,
                   (c[:, None, None] < C) & (s[None, :, None] < K * K) &
                   (event[None, None, :] < len(Z)), sem="relaxed")


def _geometry(x: Tensor, weights: Tensor) -> tuple[int, int, int, int]:
    _require_packed("sublayer read/write", x=x, weights=weights)
    if x.ndim != 4 or weights.ndim != 3:
        raise ValueError("expected NHWC carrier and [events, channels, slots] weights")
    b, h, w, c = x.shape
    events, wc, slots = weights.shape
    if events not in (8, 12, 16, 24) or wc != c or slots not in (4, 16):
        raise ValueError("fused stages require 8/12/16/24 events and 4/16 spatial slots")
    k = 2 if slots == 4 else 4
    if h != w or h % k:
        raise ValueError("carrier grid must be square and divisible by the slot grid")
    if x.dtype != weights.dtype or x.device != weights.device or not x.is_cuda:
        raise ValueError("carrier and weights must share CUDA device and dtype")
    return b, c, h // k, k


def _use_bf16_mma(x: Tensor, k: int, events: int) -> bool:
    # Called only inside real custom-op execution. Fake implementations do
    # not inspect a physical device, keeping fullgraph tracing device-free.
    if x.dtype != torch.bfloat16 or k != 4 or torch.version.hip is not None:
        return False
    properties = torch.cuda.get_device_properties(x.device)
    if properties.major < 8:
        return False
    # Offline sm80/sm86/sm90 builds of these fixed launch configurations need
    # at most 72 KiB (16 event lanes) or 136 KiB (32 event lanes), including
    # the final write. The latter exceeds some Ampere GPUs' opt-in limit.
    required = 72 * 1024 if events in (8, 12) else 136 * 1024
    available = max(properties.shared_memory_per_block,
                    getattr(properties, "shared_memory_per_block_optin", 0))
    return available >= required


def _event_tile(events: int) -> int:
    # BF16 dot reductions need at least 16 lanes. Preserve the original
    # 32-lane specialization for both existing sublayer stage sizes.
    return 16 if events in (8, 12) else 32


@custom_op("sihc::sublayer_read", mutates_args=())
def sublayer_read(x: Tensor, alpha: Tensor) -> list[Tensor]:
    b, c, g, k = _geometry(x, alpha)
    out = [torch.empty((b, g * g, c), device=x.device, dtype=x.dtype)
           for _ in range(alpha.shape[0])]
    alpha_read = alpha.transpose(1, 2).contiguous()
    if _use_bf16_mma(x, k, alpha.shape[0]):
        _read_bf16_mma[(triton.cdiv(b * g * g, 16), triton.cdiv(c, 64))](
            x, alpha_read, tuple(out), b * g * g, c, g, k,
            BE=_event_tile(alpha.shape[0]), num_warps=8)
    else:
        _read[(triton.cdiv(b * g * g, 4), triton.cdiv(c, 64))](
            x, alpha_read, tuple(out), b * g * g, c, g, k, 4, 64, num_warps=4)
    return out


def _validate_updates(x: Tensor, weights: Tensor, updates: list[Tensor]):
    b, c, g, k = _geometry(x, weights)
    if len(updates) != weights.shape[0]:
        raise ValueError("one workspace update is required per connection event")
    for dz in updates:
        _require_packed("sublayer write", update=dz)
        if dz.shape != (b, g * g, c) or dz.dtype != x.dtype or dz.device != x.device:
            raise ValueError("update geometry, dtype, or device does not match carrier")
    return b, c, g, k


@custom_op("sihc::sublayer_write_impl", mutates_args=())
def _write_impl(x: Tensor, updates: list[Tensor], beta: Tensor, add_x: bool) -> Tensor:
    b, c, g, k = _validate_updates(x, beta, updates)
    out = torch.empty_like(x)
    if _use_bf16_mma(x, k, len(updates)):
        _write_bf16_mma[(triton.cdiv(b * g * g, 16), triton.cdiv(c, 64))](
            x, tuple(updates), beta, out, b * g * g, c, g, k, add_x,
            BE=_event_tile(len(updates)), num_warps=8)
    else:
        _write[(b * g * g, triton.cdiv(c, 64))](
            x, tuple(updates), beta, out, c, g, k, add_x, 64,
            num_warps=1 if k == 2 else 4)
    return out


@custom_op("sihc::sublayer_weight_grad", mutates_args=())
def _weights_backward(x: Tensor, updates: list[Tensor], weights: Tensor) -> Tensor:
    b, c, g, k = _validate_updates(x, weights, updates)
    out = torch.zeros((weights.shape[0], k * k, c), device=x.device, dtype=torch.float32)
    if _use_bf16_mma(x, k, len(updates)):
        _weight_grad_bf16_mma[(triton.cdiv(b * g * g, 256), triton.cdiv(c, 64))](
            x, tuple(updates), out, b * g * g, c, g, k,
            BE=_event_tile(len(updates)), num_warps=8, num_stages=1)
    else:
        # One warp reduces synchronization overhead at event reductions.
        _weight_grad[(triton.cdiv(b * g * g, 32) * k * k, triton.cdiv(c, 64))](
            x, tuple(updates), out, b * g * g, c, g, k, 32, 64, num_warps=1)
    return out.transpose(1, 2).to(weights.dtype).contiguous()


def _read_setup(ctx, inputs, output):
    ctx.save_for_backward(*inputs)


def _read_backward(ctx, grads):
    x, alpha = ctx.saved_tensors
    shape = (x.shape[0], x.shape[1] * x.shape[2] // alpha.shape[2], x.shape[3])
    grads = [g.contiguous() if g is not None else x.new_zeros(shape) for g in grads]
    return _write_impl(x, grads, alpha, False), _weights_backward(x, grads, alpha)


register_autograd(sublayer_read, _read_backward, setup_context=_read_setup)


@custom_op("sihc::sublayer_write", mutates_args=())
def sublayer_write(x: Tensor, updates: list[Tensor], beta: Tensor) -> Tensor:
    return _write_impl(x, updates, beta, True)


def _write_setup(ctx, inputs, output):
    _, updates, beta = inputs
    ctx.save_for_backward(beta, *updates)


def _write_backward(ctx, grad):
    beta, *updates = ctx.saved_tensors
    grad = grad.contiguous()
    return grad, sublayer_read(grad, beta), _weights_backward(grad, updates, beta)


register_autograd(sublayer_write, _write_backward, setup_context=_write_setup)


@triton.jit
def _triangular(BASE, DZ, GAMMA, OUT, M: tl.constexpr, C: tl.constexpr,
                BM: tl.constexpr, BC: tl.constexpr):
    m = tl.program_id(0) * BM + tl.arange(0, BM)
    c = tl.program_id(1) * BC + tl.arange(0, BC)
    off = m[:, None] * C + c[None, :]
    mask = (m[:, None] < M) & (c[None, :] < C)
    acc = tl.load(BASE + off, mask, other=0).to(tl.float32)
    for l in tl.static_range(len(DZ)):
        dz = tl.load(DZ[l] + off, mask, other=0).to(tl.float32)
        gamma = tl.load(GAMMA + l * C + c, c < C, other=0).to(tl.float32)
        acc += dz * gamma[None, :]
    tl.store(OUT + off, acc, mask)


@triton.jit
def _triangular_gamma_grad(DOUT, DZ, DGAMMA, M: tl.constexpr,
                          C: tl.constexpr, BM: tl.constexpr, BC: tl.constexpr):
    m = tl.program_id(0) * BM + tl.arange(0, BM)
    c = tl.program_id(1) * BC + tl.arange(0, BC)
    off = m[:, None] * C + c[None, :]
    mask = (m[:, None] < M) & (c[None, :] < C)
    go = tl.load(DOUT + off, mask, other=0).to(tl.float32)
    for l in tl.static_range(len(DZ)):
        dz = tl.load(DZ[l] + off, mask, other=0).to(tl.float32)
        tl.atomic_add(DGAMMA + l * C + c, tl.sum(go * dz, 0), c < C, sem="relaxed")


@custom_op("sihc::sublayer_triangular", mutates_args=())
def sublayer_triangular(base: Tensor, updates: list[Tensor], gamma: Tensor) -> Tensor:
    _require_packed("sublayer triangular", base=base, gamma=gamma)
    if not 1 <= len(updates) <= 23 or gamma.shape != (len(updates), base.shape[-1]):
        raise ValueError("gamma must match the active prefix (1..23) and channel count")
    for dz in updates:
        _require_packed("sublayer triangular", update=dz)
        if dz.shape != base.shape or dz.dtype != base.dtype or dz.device != base.device:
            raise ValueError("active updates must match the workspace tensor")
    if gamma.device != base.device or gamma.dtype not in (base.dtype, torch.float32):
        raise ValueError("gamma must share workspace device and use workspace dtype or fp32")
    b, n, c = base.shape
    out = torch.empty_like(base)
    _triangular[(triton.cdiv(b * n, 32), triton.cdiv(c, 128))](
        base, tuple(updates), gamma, out, b * n, c, 32, 128)
    return out


@custom_op("sihc::sublayer_triangular_gamma_backward", mutates_args=())
def _tri_gamma_backward(grad: Tensor, updates: list[Tensor], gamma: Tensor) -> Tensor:
    b, n, c = grad.shape
    dg = torch.zeros(gamma.shape, device=grad.device, dtype=torch.float32)
    _triangular_gamma_grad[(triton.cdiv(b * n, 64), triton.cdiv(c, 64))](
        grad, tuple(updates), dg, b * n, c, 64, 64, num_warps=2)
    return dg.to(gamma.dtype)


def _tri_setup(ctx, inputs, output):
    _, updates, gamma = inputs
    ctx.save_for_backward(gamma, *updates)


def _tri_autograd(ctx, grad):
    gamma, *updates = ctx.saved_tensors
    grad = grad.contiguous()
    dg = _tri_gamma_backward(grad, updates, gamma)
    # Leave elementwise update gradients visible to AOTAutograd/Inductor.
    # They can fuse with accumulation from later events instead of writing
    # one full workspace tensor per triangular edge inside an opaque op.
    # Preserve the previous FP32 product followed by the workspace dtype cast.
    ddz = [(grad.float() * gamma[i].float()).to(grad.dtype)
           for i in range(len(updates))]
    return grad, ddz, dg


register_autograd(sublayer_triangular, _tri_autograd, setup_context=_tri_setup)


# PyTorch 2.9 wrap_triton cannot functionalize tuples of tensor pointers.
# Opaque custom ops preserve the fused kernels in fullgraph compile.
@sublayer_read.register_fake
def _read_fake(x, alpha):
    b, c, g, _ = _geometry(x, alpha)
    return [x.new_empty((b, g * g, c)) for _ in range(alpha.shape[0])]


@_write_impl.register_fake
def _write_impl_fake(x, updates, beta, add_x):
    _validate_updates(x, beta, updates)
    return torch.empty_like(x)


@sublayer_write.register_fake
def _write_fake(x, updates, beta):
    return _write_impl_fake(x, updates, beta, True)


@_weights_backward.register_fake
def _weight_fake(x, updates, weights):
    return torch.empty_like(weights)


@sublayer_triangular.register_fake
def _tri_fake(base, updates, gamma):
    return torch.empty_like(base)


@_tri_gamma_backward.register_fake
def _tri_gamma_backward_fake(grad, updates, gamma):
    return torch.empty_like(gamma)
