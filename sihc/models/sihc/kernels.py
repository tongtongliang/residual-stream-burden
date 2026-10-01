"""Triton kernels for fixed-shape SiHC fused schedules.

The block-wise recurrence always runs on a 16x16 workspace.  The p4 path uses
16 spatial slots per workspace token; the p8 path uses four.  Only the 8- and
12-layer operations used by SiHC are defined here.
"""

from __future__ import annotations

import os

import torch
import triton
import triton.language as tl
from torch import Tensor
from torch.library import register_autograd, triton_op, wrap_triton


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if not value:
        return default
    return int(value)


def _require_packed(op_name: str, **tensors: Tensor) -> None:
    """Reject views incompatible with the kernels' linear indexing."""
    for tensor_name, tensor in tensors.items():
        if not tensor.is_contiguous():
            raise ValueError(
                f"{op_name} requires contiguous {tensor_name}, got stride {tensor.stride()}"
            )


@triton.jit
def _cell_mean_b4_fwd_kernel(
    x_ptr,
    out_ptr,
    total_low: tl.constexpr,
    channels: tl.constexpr,
    block_m: tl.constexpr,
    block_c: tl.constexpr,
):
    """Reduce each non-overlapping 4x4 high-resolution cell to one token."""
    offs_m = tl.program_id(0) * block_m + tl.arange(0, block_m)
    offs_c = tl.program_id(1) * block_c + tl.arange(0, block_c)
    mask = (offs_m[:, None] < total_low) & (offs_c[None, :] < channels)

    b = offs_m // 256
    n = offs_m - b * 256
    wy = n // 16
    wx = n - wy * 16
    acc = tl.zeros((block_m, block_c), tl.float32)
    for pos in tl.static_range(0, 16):
        uy = pos // 4
        ux = pos - uy * 4
        h = wy * 4 + uy
        col = wx * 4 + ux
        x_offsets = (
            ((b[:, None] * 64 + h[:, None]) * 64 + col[:, None]) * channels
            + offs_c[None, :]
        )
        acc += tl.load(x_ptr + x_offsets, mask=mask, other=0.0).to(tl.float32)

    out_offsets = offs_m[:, None] * channels + offs_c[None, :]
    tl.store(out_ptr + out_offsets, acc * (1.0 / 16.0), mask=mask)


@triton.jit
def _cell_mean_b4_bwd_kernel(
    grad_out_ptr,
    grad_x_ptr,
    total_high: tl.constexpr,
    channels: tl.constexpr,
    block_m: tl.constexpr,
    block_c: tl.constexpr,
):
    """Scatter the cell-mean gradient without atomics."""
    offs_m = tl.program_id(0) * block_m + tl.arange(0, block_m)
    offs_c = tl.program_id(1) * block_c + tl.arange(0, block_c)
    mask = (offs_m[:, None] < total_high) & (offs_c[None, :] < channels)

    b = offs_m // 4096
    high_n = offs_m - b * 4096
    h = high_n // 64
    col = high_n - h * 64
    low_n = (h // 4) * 16 + col // 4
    grad_offsets = (b[:, None] * 256 + low_n[:, None]) * channels + offs_c[None, :]
    grad = tl.load(grad_out_ptr + grad_offsets, mask=mask, other=0.0).to(tl.float32)
    x_offsets = offs_m[:, None] * channels + offs_c[None, :]
    tl.store(grad_x_ptr + x_offsets, grad * (1.0 / 16.0), mask=mask)


@triton.jit
def _final_accumulate_12_fwd_kernel(
    x0_ptr,
    dz0_ptr,
    dz1_ptr,
    dz2_ptr,
    dz3_ptr,
    dz4_ptr,
    dz5_ptr,
    dz6_ptr,
    dz7_ptr,
    dz8_ptr,
    dz9_ptr,
    dz10_ptr,
    dz11_ptr,
    beta0_ptr,
    beta1_ptr,
    beta2_ptr,
    beta3_ptr,
    beta4_ptr,
    beta5_ptr,
    beta6_ptr,
    beta7_ptr,
    beta8_ptr,
    beta9_ptr,
    beta10_ptr,
    beta11_ptr,
    out_ptr,
    total_low: tl.constexpr,
    channels: tl.constexpr,
    high_grid: tl.constexpr,
    cell_size: tl.constexpr,
    slots: tl.constexpr,
    block_c: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_c = tl.program_id(1)
    offs_c = pid_c * block_c + tl.arange(0, block_c)
    offs_p = tl.arange(0, slots)
    mask_c = offs_c < channels

    b = pid_m // 256
    n = pid_m - b * 256
    wy = n // 16
    wx = n - wy * 16
    uy = offs_p // cell_size
    ux = offs_p - uy * cell_size
    h = wy * cell_size + uy
    col = wx * cell_size + ux
    x_offsets = (
        ((b * high_grid + h[:, None]) * high_grid + col[:, None]) * channels
        + offs_c[None, :]
    )
    dz_offsets = pid_m * channels + offs_c
    beta_offsets = offs_c[None, :] * slots + offs_p[:, None]

    acc = tl.load(x0_ptr + x_offsets, mask=mask_c[None, :], other=0.0).to(tl.float32)
    dz0 = tl.load(dz0_ptr + dz_offsets, mask=mask_c, other=0.0).to(tl.float32)
    beta0 = tl.load(beta0_ptr + beta_offsets, mask=mask_c[None, :], other=0.0).to(tl.float32)
    acc += beta0 * dz0[None, :]
    dz1 = tl.load(dz1_ptr + dz_offsets, mask=mask_c, other=0.0).to(tl.float32)
    beta1 = tl.load(beta1_ptr + beta_offsets, mask=mask_c[None, :], other=0.0).to(tl.float32)
    acc += beta1 * dz1[None, :]
    dz2 = tl.load(dz2_ptr + dz_offsets, mask=mask_c, other=0.0).to(tl.float32)
    beta2 = tl.load(beta2_ptr + beta_offsets, mask=mask_c[None, :], other=0.0).to(tl.float32)
    acc += beta2 * dz2[None, :]
    dz3 = tl.load(dz3_ptr + dz_offsets, mask=mask_c, other=0.0).to(tl.float32)
    beta3 = tl.load(beta3_ptr + beta_offsets, mask=mask_c[None, :], other=0.0).to(tl.float32)
    acc += beta3 * dz3[None, :]
    dz4 = tl.load(dz4_ptr + dz_offsets, mask=mask_c, other=0.0).to(tl.float32)
    beta4 = tl.load(beta4_ptr + beta_offsets, mask=mask_c[None, :], other=0.0).to(tl.float32)
    acc += beta4 * dz4[None, :]
    dz5 = tl.load(dz5_ptr + dz_offsets, mask=mask_c, other=0.0).to(tl.float32)
    beta5 = tl.load(beta5_ptr + beta_offsets, mask=mask_c[None, :], other=0.0).to(tl.float32)
    acc += beta5 * dz5[None, :]
    dz6 = tl.load(dz6_ptr + dz_offsets, mask=mask_c, other=0.0).to(tl.float32)
    beta6 = tl.load(beta6_ptr + beta_offsets, mask=mask_c[None, :], other=0.0).to(tl.float32)
    acc += beta6 * dz6[None, :]
    dz7 = tl.load(dz7_ptr + dz_offsets, mask=mask_c, other=0.0).to(tl.float32)
    beta7 = tl.load(beta7_ptr + beta_offsets, mask=mask_c[None, :], other=0.0).to(tl.float32)
    acc += beta7 * dz7[None, :]
    dz8 = tl.load(dz8_ptr + dz_offsets, mask=mask_c, other=0.0).to(tl.float32)
    beta8 = tl.load(beta8_ptr + beta_offsets, mask=mask_c[None, :], other=0.0).to(tl.float32)
    acc += beta8 * dz8[None, :]
    dz9 = tl.load(dz9_ptr + dz_offsets, mask=mask_c, other=0.0).to(tl.float32)
    beta9 = tl.load(beta9_ptr + beta_offsets, mask=mask_c[None, :], other=0.0).to(tl.float32)
    acc += beta9 * dz9[None, :]
    dz10 = tl.load(dz10_ptr + dz_offsets, mask=mask_c, other=0.0).to(tl.float32)
    beta10 = tl.load(beta10_ptr + beta_offsets, mask=mask_c[None, :], other=0.0).to(tl.float32)
    acc += beta10 * dz10[None, :]
    dz11 = tl.load(dz11_ptr + dz_offsets, mask=mask_c, other=0.0).to(tl.float32)
    beta11 = tl.load(beta11_ptr + beta_offsets, mask=mask_c[None, :], other=0.0).to(tl.float32)
    acc += beta11 * dz11[None, :]

    tl.store(out_ptr + x_offsets, acc, mask=mask_c[None, :])




@triton.jit
def _final_accumulate_b4_8_fwd_kernel(
    x0_ptr,
    dz0_ptr,
    dz1_ptr,
    dz2_ptr,
    dz3_ptr,
    dz4_ptr,
    dz5_ptr,
    dz6_ptr,
    dz7_ptr,
    beta0_ptr,
    beta1_ptr,
    beta2_ptr,
    beta3_ptr,
    beta4_ptr,
    beta5_ptr,
    beta6_ptr,
    beta7_ptr,
    out_ptr,
    total_low: tl.constexpr,
    channels: tl.constexpr,
    block_c: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_c = tl.program_id(1)
    offs_c = pid_c * block_c + tl.arange(0, block_c)
    offs_p = tl.arange(0, 16)
    mask_c = offs_c < channels

    b = pid_m // 256
    n = pid_m - b * 256
    wy = n // 16
    wx = n - wy * 16
    uy = offs_p // 4
    ux = offs_p - uy * 4
    h = wy * 4 + uy
    col = wx * 4 + ux
    x_offsets = (((b * 64 + h[:, None]) * 64 + col[:, None]) * channels) + offs_c[None, :]
    dz_offsets = pid_m * channels + offs_c
    beta_offsets = offs_c[None, :] * 16 + offs_p[:, None]

    acc = tl.load(x0_ptr + x_offsets, mask=mask_c[None, :], other=0.0).to(tl.float32)
    dz0 = tl.load(dz0_ptr + dz_offsets, mask=mask_c, other=0.0).to(tl.float32)
    beta0 = tl.load(beta0_ptr + beta_offsets, mask=mask_c[None, :], other=0.0).to(tl.float32)
    acc += beta0 * dz0[None, :]
    dz1 = tl.load(dz1_ptr + dz_offsets, mask=mask_c, other=0.0).to(tl.float32)
    beta1 = tl.load(beta1_ptr + beta_offsets, mask=mask_c[None, :], other=0.0).to(tl.float32)
    acc += beta1 * dz1[None, :]
    dz2 = tl.load(dz2_ptr + dz_offsets, mask=mask_c, other=0.0).to(tl.float32)
    beta2 = tl.load(beta2_ptr + beta_offsets, mask=mask_c[None, :], other=0.0).to(tl.float32)
    acc += beta2 * dz2[None, :]
    dz3 = tl.load(dz3_ptr + dz_offsets, mask=mask_c, other=0.0).to(tl.float32)
    beta3 = tl.load(beta3_ptr + beta_offsets, mask=mask_c[None, :], other=0.0).to(tl.float32)
    acc += beta3 * dz3[None, :]
    dz4 = tl.load(dz4_ptr + dz_offsets, mask=mask_c, other=0.0).to(tl.float32)
    beta4 = tl.load(beta4_ptr + beta_offsets, mask=mask_c[None, :], other=0.0).to(tl.float32)
    acc += beta4 * dz4[None, :]
    dz5 = tl.load(dz5_ptr + dz_offsets, mask=mask_c, other=0.0).to(tl.float32)
    beta5 = tl.load(beta5_ptr + beta_offsets, mask=mask_c[None, :], other=0.0).to(tl.float32)
    acc += beta5 * dz5[None, :]
    dz6 = tl.load(dz6_ptr + dz_offsets, mask=mask_c, other=0.0).to(tl.float32)
    beta6 = tl.load(beta6_ptr + beta_offsets, mask=mask_c[None, :], other=0.0).to(tl.float32)
    acc += beta6 * dz6[None, :]
    dz7 = tl.load(dz7_ptr + dz_offsets, mask=mask_c, other=0.0).to(tl.float32)
    beta7 = tl.load(beta7_ptr + beta_offsets, mask=mask_c[None, :], other=0.0).to(tl.float32)
    acc += beta7 * dz7[None, :]

    tl.store(out_ptr + x_offsets, acc, mask=mask_c[None, :])


@triton.jit
def _final_accumulate_b4_8_grad_dz_ptr_kernel(
    grad_out_ptr,
    beta0_ptr,
    beta1_ptr,
    beta2_ptr,
    beta3_ptr,
    beta4_ptr,
    beta5_ptr,
    beta6_ptr,
    beta7_ptr,
    grad_dz_ptr,
    total_low: tl.constexpr,
    channels: tl.constexpr,
    block_m: tl.constexpr,
    block_c: tl.constexpr,
):
    pid_l = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_c = tl.program_id(2)
    offs_m = pid_m * block_m + tl.arange(0, block_m)
    offs_c = pid_c * block_c + tl.arange(0, block_c)
    mask = (offs_m[:, None] < total_low) & (offs_c[None, :] < channels)

    beta_ptr = tl.where(pid_l == 0, beta0_ptr, tl.where(pid_l == 1, beta1_ptr, tl.where(pid_l == 2, beta2_ptr, tl.where(pid_l == 3, beta3_ptr, tl.where(pid_l == 4, beta4_ptr, tl.where(pid_l == 5, beta5_ptr, tl.where(pid_l == 6, beta6_ptr, beta7_ptr)))))))

    b = offs_m // 256
    n = offs_m - b * 256
    wy = n // 16
    wx = n - wy * 16
    acc = tl.zeros((block_m, block_c), tl.float32)
    for pos in tl.static_range(0, 16):
        uy = pos // 4
        ux = pos - uy * 4
        h = wy * 4 + uy
        col = wx * 4 + ux
        go_offsets = (((b[:, None] * 64 + h[:, None]) * 64 + col[:, None]) * channels) + offs_c[None, :]
        beta_offsets = offs_c * 16 + pos
        go_vals = tl.load(grad_out_ptr + go_offsets, mask=mask, other=0.0).to(tl.float32)
        beta_vals = tl.load(beta_ptr + beta_offsets, mask=offs_c < channels, other=0.0).to(tl.float32)
        acc += go_vals * beta_vals[None, :]

    gd_offsets = ((pid_l * total_low + offs_m[:, None]) * channels) + offs_c[None, :]
    tl.store(grad_dz_ptr + gd_offsets, acc, mask=mask)


@triton.jit
def _final_accumulate_b4_8_grad_beta_ptr_kernel(
    grad_out_ptr,
    dz0_ptr,
    dz1_ptr,
    dz2_ptr,
    dz3_ptr,
    dz4_ptr,
    dz5_ptr,
    dz6_ptr,
    dz7_ptr,
    grad_beta_ptr,
    total_low: tl.constexpr,
    channels: tl.constexpr,
    block_m: tl.constexpr,
    block_c: tl.constexpr,
):
    pid_lp = tl.program_id(0)
    pid_l = pid_lp // 16
    pid_m = tl.program_id(1)
    pid_c = tl.program_id(2)
    pos = pid_lp - pid_l * 16
    offs_m = pid_m * block_m + tl.arange(0, block_m)
    offs_c = pid_c * block_c + tl.arange(0, block_c)
    mask = (offs_m[:, None] < total_low) & (offs_c[None, :] < channels)

    dz_ptr = tl.where(pid_l == 0, dz0_ptr, tl.where(pid_l == 1, dz1_ptr, tl.where(pid_l == 2, dz2_ptr, tl.where(pid_l == 3, dz3_ptr, tl.where(pid_l == 4, dz4_ptr, tl.where(pid_l == 5, dz5_ptr, tl.where(pid_l == 6, dz6_ptr, dz7_ptr)))))))

    b = offs_m // 256
    n = offs_m - b * 256
    wy = n // 16
    wx = n - wy * 16
    uy = pos // 4
    ux = pos - uy * 4
    h = wy * 4 + uy
    col = wx * 4 + ux
    go_offsets = (((b[:, None] * 64 + h[:, None]) * 64 + col[:, None]) * channels) + offs_c[None, :]
    dz_offsets = offs_m[:, None] * channels + offs_c[None, :]
    go_vals = tl.load(grad_out_ptr + go_offsets, mask=mask, other=0.0).to(tl.float32)
    dz_vals = tl.load(dz_ptr + dz_offsets, mask=mask, other=0.0).to(tl.float32)
    grad = tl.sum(go_vals * dz_vals, axis=0)
    beta_offsets = (pid_l * channels + offs_c) * 16 + pos
    tl.atomic_add(grad_beta_ptr + beta_offsets, grad, sem="relaxed", mask=offs_c < channels)


@triton.jit
def _final_accumulate_12_grad_dz_ptr_kernel(
    grad_out_ptr,
    beta0_ptr,
    beta1_ptr,
    beta2_ptr,
    beta3_ptr,
    beta4_ptr,
    beta5_ptr,
    beta6_ptr,
    beta7_ptr,
    beta8_ptr,
    beta9_ptr,
    beta10_ptr,
    beta11_ptr,
    grad_dz_ptr,
    total_low: tl.constexpr,
    channels: tl.constexpr,
    high_grid: tl.constexpr,
    cell_size: tl.constexpr,
    slots: tl.constexpr,
    block_m: tl.constexpr,
    block_c: tl.constexpr,
):
    pid_l = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_c = tl.program_id(2)
    offs_m = pid_m * block_m + tl.arange(0, block_m)
    offs_c = pid_c * block_c + tl.arange(0, block_c)
    mask = (offs_m[:, None] < total_low) & (offs_c[None, :] < channels)

    beta_ptr = tl.where(
        pid_l == 0,
        beta0_ptr,
        tl.where(
            pid_l == 1,
            beta1_ptr,
            tl.where(
                pid_l == 2,
                beta2_ptr,
                tl.where(
                    pid_l == 3,
                    beta3_ptr,
                    tl.where(
                        pid_l == 4,
                        beta4_ptr,
                        tl.where(
                            pid_l == 5,
                            beta5_ptr,
                            tl.where(
                                pid_l == 6,
                                beta6_ptr,
                                tl.where(
                                    pid_l == 7,
                                    beta7_ptr,
                                    tl.where(
                                        pid_l == 8,
                                        beta8_ptr,
                                        tl.where(pid_l == 9, beta9_ptr, tl.where(pid_l == 10, beta10_ptr, beta11_ptr)),
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        ),
    )

    b = offs_m // 256
    n = offs_m - b * 256
    wy = n // 16
    wx = n - wy * 16
    acc = tl.zeros((block_m, block_c), tl.float32)
    for pos in tl.static_range(0, slots):
        uy = pos // cell_size
        ux = pos - uy * cell_size
        h = wy * cell_size + uy
        col = wx * cell_size + ux
        go_offsets = (
            ((b[:, None] * high_grid + h[:, None]) * high_grid + col[:, None])
            * channels
            + offs_c[None, :]
        )
        beta_offsets = offs_c * slots + pos
        go_vals = tl.load(grad_out_ptr + go_offsets, mask=mask, other=0.0).to(tl.float32)
        beta_vals = tl.load(beta_ptr + beta_offsets, mask=offs_c < channels, other=0.0).to(tl.float32)
        acc += go_vals * beta_vals[None, :]

    gd_offsets = ((pid_l * total_low + offs_m[:, None]) * channels) + offs_c[None, :]
    tl.store(grad_dz_ptr + gd_offsets, acc, mask=mask)


@triton.jit
def _final_accumulate_12_grad_beta_ptr_kernel(
    grad_out_ptr,
    dz0_ptr,
    dz1_ptr,
    dz2_ptr,
    dz3_ptr,
    dz4_ptr,
    dz5_ptr,
    dz6_ptr,
    dz7_ptr,
    dz8_ptr,
    dz9_ptr,
    dz10_ptr,
    dz11_ptr,
    grad_beta_ptr,
    total_low: tl.constexpr,
    channels: tl.constexpr,
    high_grid: tl.constexpr,
    cell_size: tl.constexpr,
    slots: tl.constexpr,
    block_m: tl.constexpr,
    block_c: tl.constexpr,
):
    pid_lp = tl.program_id(0)
    pid_l = pid_lp // slots
    pid_m = tl.program_id(1)
    pid_c = tl.program_id(2)
    pos = pid_lp - pid_l * slots
    offs_m = pid_m * block_m + tl.arange(0, block_m)
    offs_c = pid_c * block_c + tl.arange(0, block_c)
    mask = (offs_m[:, None] < total_low) & (offs_c[None, :] < channels)

    dz_ptr = tl.where(
        pid_l == 0,
        dz0_ptr,
        tl.where(
            pid_l == 1,
            dz1_ptr,
            tl.where(
                pid_l == 2,
                dz2_ptr,
                tl.where(
                    pid_l == 3,
                    dz3_ptr,
                    tl.where(
                        pid_l == 4,
                        dz4_ptr,
                        tl.where(
                            pid_l == 5,
                            dz5_ptr,
                            tl.where(
                                pid_l == 6,
                                dz6_ptr,
                                tl.where(
                                    pid_l == 7,
                                    dz7_ptr,
                                    tl.where(
                                        pid_l == 8,
                                        dz8_ptr,
                                        tl.where(pid_l == 9, dz9_ptr, tl.where(pid_l == 10, dz10_ptr, dz11_ptr)),
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        ),
    )

    b = offs_m // 256
    n = offs_m - b * 256
    wy = n // 16
    wx = n - wy * 16
    uy = pos // cell_size
    ux = pos - uy * cell_size
    h = wy * cell_size + uy
    col = wx * cell_size + ux
    go_offsets = (
        ((b[:, None] * high_grid + h[:, None]) * high_grid + col[:, None])
        * channels
        + offs_c[None, :]
    )
    dz_offsets = offs_m[:, None] * channels + offs_c[None, :]
    go_vals = tl.load(grad_out_ptr + go_offsets, mask=mask, other=0.0).to(tl.float32)
    dz_vals = tl.load(dz_ptr + dz_offsets, mask=mask, other=0.0).to(tl.float32)
    grad = tl.sum(go_vals * dz_vals, axis=0)
    beta_offsets = (pid_l * channels + offs_c) * slots + pos
    tl.atomic_add(grad_beta_ptr + beta_offsets, grad, sem="relaxed", mask=offs_c < channels)


@triton.jit
def _all_base_12_split_fwd_kernel(
    x_ptr,
    alpha_ptr,
    z0_ptr,
    z1_ptr,
    z2_ptr,
    z3_ptr,
    z4_ptr,
    z5_ptr,
    z6_ptr,
    z7_ptr,
    z8_ptr,
    z9_ptr,
    z10_ptr,
    z11_ptr,
    total_low: tl.constexpr,
    channels: tl.constexpr,
    high_grid: tl.constexpr,
    cell_size: tl.constexpr,
    slots: tl.constexpr,
    block_m: tl.constexpr,
    block_c: tl.constexpr,
):
    """B/4 all-base read with one x0 tile load and 12 separate outputs."""
    pid_m = tl.program_id(0)
    pid_c = tl.program_id(1)
    offs_m = pid_m * block_m + tl.arange(0, block_m)
    offs_c = pid_c * block_c + tl.arange(0, block_c)
    mask = (offs_m[:, None] < total_low) & (offs_c[None, :] < channels)

    b = offs_m // 256
    n = offs_m - b * 256
    wy = n // 16
    wx = n - wy * 16

    acc0 = tl.zeros((block_m, block_c), tl.float32)
    acc1 = tl.zeros((block_m, block_c), tl.float32)
    acc2 = tl.zeros((block_m, block_c), tl.float32)
    acc3 = tl.zeros((block_m, block_c), tl.float32)
    acc4 = tl.zeros((block_m, block_c), tl.float32)
    acc5 = tl.zeros((block_m, block_c), tl.float32)
    acc6 = tl.zeros((block_m, block_c), tl.float32)
    acc7 = tl.zeros((block_m, block_c), tl.float32)
    acc8 = tl.zeros((block_m, block_c), tl.float32)
    acc9 = tl.zeros((block_m, block_c), tl.float32)
    acc10 = tl.zeros((block_m, block_c), tl.float32)
    acc11 = tl.zeros((block_m, block_c), tl.float32)

    for pos in tl.static_range(0, slots):
        uy = pos // cell_size
        ux = pos - uy * cell_size
        h = wy * cell_size + uy
        w = wx * cell_size + ux
        x_offsets = (
            ((b[:, None] * high_grid + h[:, None]) * high_grid + w[:, None])
            * channels
            + offs_c[None, :]
        )
        x_vals = tl.load(x_ptr + x_offsets, mask=mask, other=0.0).to(tl.float32)

        alpha_offsets = (offs_c * slots) + pos
        a0 = tl.load(alpha_ptr + alpha_offsets, mask=offs_c < channels, other=0.0).to(tl.float32)
        a1 = tl.load(alpha_ptr + alpha_offsets + channels * slots, mask=offs_c < channels, other=0.0).to(tl.float32)
        a2 = tl.load(alpha_ptr + alpha_offsets + 2 * channels * slots, mask=offs_c < channels, other=0.0).to(tl.float32)
        a3 = tl.load(alpha_ptr + alpha_offsets + 3 * channels * slots, mask=offs_c < channels, other=0.0).to(tl.float32)
        a4 = tl.load(alpha_ptr + alpha_offsets + 4 * channels * slots, mask=offs_c < channels, other=0.0).to(tl.float32)
        a5 = tl.load(alpha_ptr + alpha_offsets + 5 * channels * slots, mask=offs_c < channels, other=0.0).to(tl.float32)
        a6 = tl.load(alpha_ptr + alpha_offsets + 6 * channels * slots, mask=offs_c < channels, other=0.0).to(tl.float32)
        a7 = tl.load(alpha_ptr + alpha_offsets + 7 * channels * slots, mask=offs_c < channels, other=0.0).to(tl.float32)
        a8 = tl.load(alpha_ptr + alpha_offsets + 8 * channels * slots, mask=offs_c < channels, other=0.0).to(tl.float32)
        a9 = tl.load(alpha_ptr + alpha_offsets + 9 * channels * slots, mask=offs_c < channels, other=0.0).to(tl.float32)
        a10 = tl.load(alpha_ptr + alpha_offsets + 10 * channels * slots, mask=offs_c < channels, other=0.0).to(tl.float32)
        a11 = tl.load(alpha_ptr + alpha_offsets + 11 * channels * slots, mask=offs_c < channels, other=0.0).to(tl.float32)
        acc0 += x_vals * a0[None, :]
        acc1 += x_vals * a1[None, :]
        acc2 += x_vals * a2[None, :]
        acc3 += x_vals * a3[None, :]
        acc4 += x_vals * a4[None, :]
        acc5 += x_vals * a5[None, :]
        acc6 += x_vals * a6[None, :]
        acc7 += x_vals * a7[None, :]
        acc8 += x_vals * a8[None, :]
        acc9 += x_vals * a9[None, :]
        acc10 += x_vals * a10[None, :]
        acc11 += x_vals * a11[None, :]

    out_offsets = offs_m[:, None] * channels + offs_c[None, :]
    tl.store(z0_ptr + out_offsets, acc0, mask=mask)
    tl.store(z1_ptr + out_offsets, acc1, mask=mask)
    tl.store(z2_ptr + out_offsets, acc2, mask=mask)
    tl.store(z3_ptr + out_offsets, acc3, mask=mask)
    tl.store(z4_ptr + out_offsets, acc4, mask=mask)
    tl.store(z5_ptr + out_offsets, acc5, mask=mask)
    tl.store(z6_ptr + out_offsets, acc6, mask=mask)
    tl.store(z7_ptr + out_offsets, acc7, mask=mask)
    tl.store(z8_ptr + out_offsets, acc8, mask=mask)
    tl.store(z9_ptr + out_offsets, acc9, mask=mask)
    tl.store(z10_ptr + out_offsets, acc10, mask=mask)
    tl.store(z11_ptr + out_offsets, acc11, mask=mask)


@triton.jit
def _all_base_12_grad_x_kernel(
    grad_base_ptr,
    alpha_ptr,
    grad_x_ptr,
    total_high: tl.constexpr,
    channels: tl.constexpr,
    high_grid: tl.constexpr,
    high_tokens: tl.constexpr,
    cell_size: tl.constexpr,
    slots: tl.constexpr,
    block_m: tl.constexpr,
    block_c: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_c = tl.program_id(1)
    offs_m = pid_m * block_m + tl.arange(0, block_m)
    offs_c = pid_c * block_c + tl.arange(0, block_c)
    mask = (offs_m[:, None] < total_high) & (offs_c[None, :] < channels)

    b = offs_m // high_tokens
    r = offs_m - b * high_tokens
    h = r // high_grid
    col = r - h * high_grid
    wy = h // cell_size
    wx = col // cell_size
    pos = (h - wy * cell_size) * cell_size + (col - wx * cell_size)
    n = wy * 16 + wx
    total_low = total_high // slots
    layer_stride = total_low * channels

    acc = tl.zeros((block_m, block_c), tl.float32)
    for layer in tl.static_range(0, 12):
        gb_offsets = layer * layer_stride + (b[:, None] * 256 + n[:, None]) * channels + offs_c[None, :]
        alpha_offsets = (layer * channels + offs_c[None, :]) * slots + pos[:, None]
        gb_vals = tl.load(grad_base_ptr + gb_offsets, mask=mask, other=0.0).to(tl.float32)
        alpha_vals = tl.load(alpha_ptr + alpha_offsets, mask=mask, other=0.0).to(tl.float32)
        acc += gb_vals * alpha_vals

    x_offsets = offs_m[:, None] * channels + offs_c[None, :]
    tl.store(grad_x_ptr + x_offsets, acc, mask=mask)


@triton.jit
def _all_base_12_split_grad_alpha_kernel(
    x_ptr,
    grad_base_ptr,
    grad_alpha_ptr,
    total_low: tl.constexpr,
    channels: tl.constexpr,
    high_grid: tl.constexpr,
    cell_size: tl.constexpr,
    slots: tl.constexpr,
    rows: tl.constexpr,
    block_c: tl.constexpr,
):
    """Single-pass grad_alpha for B/4 all-base read.

    grad_alpha[l, c, pos] = sum_{b,n} grad_base[l,b,n,c] * x0[b, cell(n,pos), c].
    Each program streams one token chunk and channel chunk, keeping the full
    [12,16,block_c] partial sum in registers.
    """
    pid_m = tl.program_id(0)
    pid_c = tl.program_id(1)
    offs_c = pid_c * block_c + tl.arange(0, block_c)
    c_mask = offs_c < channels
    lidx = tl.arange(0, 16)
    l_mask = lidx < 12
    pidx = tl.arange(0, slots)

    acc = tl.zeros((16, slots, block_c), tl.float32)
    m0 = pid_m * rows
    for r in tl.range(0, rows):
        m = m0 + r
        inb = m < total_low
        b = m // 256
        n = m - b * 256
        wy = n // 16
        wx = n - wy * 16
        uy = pidx // cell_size
        ux = pidx - uy * cell_size
        h = wy * cell_size + uy
        w = wx * cell_size + ux
        x_offsets = (
            ((b * high_grid + h[:, None]) * high_grid + w[:, None]) * channels
            + offs_c[None, :]
        )
        x_vals = tl.load(x_ptr + x_offsets, mask=c_mask[None, :] & inb, other=0.0).to(tl.float32)
        gb_vals = tl.load(
            grad_base_ptr + (lidx[:, None] * total_low + m) * channels + offs_c[None, :],
            mask=l_mask[:, None] & c_mask[None, :] & inb,
            other=0.0,
        ).to(tl.float32)
        acc += gb_vals[:, None, :] * x_vals[None, :, :]

    out_offsets = (
        (lidx[:, None, None] * channels + offs_c[None, None, :]) * slots
        + pidx[None, :, None]
    )
    tl.atomic_add(
        grad_alpha_ptr + out_offsets,
        acc,
        mask=l_mask[:, None, None] & c_mask[None, None, :],
        sem="relaxed",
    )




@triton.jit
def _all_base_b4_8_split_fwd_kernel(
    x_ptr,
    alpha_ptr,
    z0_ptr,
    z1_ptr,
    z2_ptr,
    z3_ptr,
    z4_ptr,
    z5_ptr,
    z6_ptr,
    z7_ptr,
    total_low: tl.constexpr,
    channels: tl.constexpr,
    block_m: tl.constexpr,
    block_c: tl.constexpr,
):
    """B/4 all-base read with one x0 tile load and 8 separate outputs."""
    pid_m = tl.program_id(0)
    pid_c = tl.program_id(1)
    offs_m = pid_m * block_m + tl.arange(0, block_m)
    offs_c = pid_c * block_c + tl.arange(0, block_c)
    mask = (offs_m[:, None] < total_low) & (offs_c[None, :] < channels)

    b = offs_m // 256
    n = offs_m - b * 256
    wy = n // 16
    wx = n - wy * 16
    acc0 = tl.zeros((block_m, block_c), tl.float32)
    acc1 = tl.zeros((block_m, block_c), tl.float32)
    acc2 = tl.zeros((block_m, block_c), tl.float32)
    acc3 = tl.zeros((block_m, block_c), tl.float32)
    acc4 = tl.zeros((block_m, block_c), tl.float32)
    acc5 = tl.zeros((block_m, block_c), tl.float32)
    acc6 = tl.zeros((block_m, block_c), tl.float32)
    acc7 = tl.zeros((block_m, block_c), tl.float32)

    for pos in tl.static_range(0, 16):
        uy = pos // 4
        ux = pos - uy * 4
        h = wy * 4 + uy
        w = wx * 4 + ux
        x_offsets = (((b[:, None] * 64 + h[:, None]) * 64 + w[:, None]) * channels) + offs_c[None, :]
        x_vals = tl.load(x_ptr + x_offsets, mask=mask, other=0.0).to(tl.float32)

        alpha_offsets = (offs_c * 16) + pos
        a0 = tl.load(alpha_ptr + alpha_offsets, mask=offs_c < channels, other=0.0).to(tl.float32)
        a1 = tl.load(alpha_ptr + alpha_offsets + 1 * channels * 16, mask=offs_c < channels, other=0.0).to(tl.float32)
        a2 = tl.load(alpha_ptr + alpha_offsets + 2 * channels * 16, mask=offs_c < channels, other=0.0).to(tl.float32)
        a3 = tl.load(alpha_ptr + alpha_offsets + 3 * channels * 16, mask=offs_c < channels, other=0.0).to(tl.float32)
        a4 = tl.load(alpha_ptr + alpha_offsets + 4 * channels * 16, mask=offs_c < channels, other=0.0).to(tl.float32)
        a5 = tl.load(alpha_ptr + alpha_offsets + 5 * channels * 16, mask=offs_c < channels, other=0.0).to(tl.float32)
        a6 = tl.load(alpha_ptr + alpha_offsets + 6 * channels * 16, mask=offs_c < channels, other=0.0).to(tl.float32)
        a7 = tl.load(alpha_ptr + alpha_offsets + 7 * channels * 16, mask=offs_c < channels, other=0.0).to(tl.float32)
        acc0 += x_vals * a0[None, :]
        acc1 += x_vals * a1[None, :]
        acc2 += x_vals * a2[None, :]
        acc3 += x_vals * a3[None, :]
        acc4 += x_vals * a4[None, :]
        acc5 += x_vals * a5[None, :]
        acc6 += x_vals * a6[None, :]
        acc7 += x_vals * a7[None, :]

    out_offsets = offs_m[:, None] * channels + offs_c[None, :]
    tl.store(z0_ptr + out_offsets, acc0, mask=mask)
    tl.store(z1_ptr + out_offsets, acc1, mask=mask)
    tl.store(z2_ptr + out_offsets, acc2, mask=mask)
    tl.store(z3_ptr + out_offsets, acc3, mask=mask)
    tl.store(z4_ptr + out_offsets, acc4, mask=mask)
    tl.store(z5_ptr + out_offsets, acc5, mask=mask)
    tl.store(z6_ptr + out_offsets, acc6, mask=mask)
    tl.store(z7_ptr + out_offsets, acc7, mask=mask)



@triton.jit
def _all_base_b4_8_grad_x_kernel(
    grad_base_ptr,
    alpha_ptr,
    grad_x_ptr,
    total_high: tl.constexpr,
    channels: tl.constexpr,
    block_m: tl.constexpr,
    block_c: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_c = tl.program_id(1)
    offs_m = pid_m * block_m + tl.arange(0, block_m)
    offs_c = pid_c * block_c + tl.arange(0, block_c)
    mask = (offs_m[:, None] < total_high) & (offs_c[None, :] < channels)

    b = offs_m // 4096
    r = offs_m - b * 4096
    h = r // 64
    col = r - h * 64
    wy = h // 4
    wx = col // 4
    pos = (h - wy * 4) * 4 + (col - wx * 4)
    n = wy * 16 + wx
    total_low = total_high // 16
    layer_stride = total_low * channels

    acc = tl.zeros((block_m, block_c), tl.float32)
    for layer in tl.static_range(0, 8):
        gb_offsets = layer * layer_stride + (b[:, None] * 256 + n[:, None]) * channels + offs_c[None, :]
        alpha_offsets = (layer * channels + offs_c[None, :]) * 16 + pos[:, None]
        gb_vals = tl.load(grad_base_ptr + gb_offsets, mask=mask, other=0.0).to(tl.float32)
        alpha_vals = tl.load(alpha_ptr + alpha_offsets, mask=mask, other=0.0).to(tl.float32)
        acc += gb_vals * alpha_vals

    x_offsets = offs_m[:, None] * channels + offs_c[None, :]
    tl.store(grad_x_ptr + x_offsets, acc, mask=mask)


@triton.jit
def _all_base_b4_8_split_grad_alpha_kernel(
    x_ptr,
    grad_base_ptr,
    grad_alpha_ptr,
    total_low: tl.constexpr,
    channels: tl.constexpr,
    rows: tl.constexpr,
    block_c: tl.constexpr,
):
    """Single-pass grad_alpha for B/4 8-layer all-base read."""
    pid_m = tl.program_id(0)
    pid_c = tl.program_id(1)
    offs_c = pid_c * block_c + tl.arange(0, block_c)
    c_mask = offs_c < channels
    lidx = tl.arange(0, 16)
    l_mask = lidx < 8
    pidx = tl.arange(0, 16)

    acc = tl.zeros((16, 16, block_c), tl.float32)
    m0 = pid_m * rows
    for r in tl.range(0, rows):
        m = m0 + r
        inb = m < total_low
        b = m // 256
        n = m - b * 256
        wy = n // 16
        wx = n - wy * 16
        uy = pidx // 4
        ux = pidx - uy * 4
        h = wy * 4 + uy
        w = wx * 4 + ux
        x_offsets = (((b * 64 + h[:, None]) * 64 + w[:, None]) * channels) + offs_c[None, :]
        x_vals = tl.load(x_ptr + x_offsets, mask=c_mask[None, :] & inb, other=0.0).to(tl.float32)
        gb_vals = tl.load(
            grad_base_ptr + (lidx[:, None] * total_low + m) * channels + offs_c[None, :],
            mask=l_mask[:, None] & c_mask[None, :] & inb,
            other=0.0,
        ).to(tl.float32)
        acc += gb_vals[:, None, :] * x_vals[None, :, :]

    out_offsets = (lidx[:, None, None] * channels + offs_c[None, None, :]) * 16 + pidx[None, :, None]
    tl.atomic_add(
        grad_alpha_ptr + out_offsets,
        acc,
        mask=l_mask[:, None, None] & c_mask[None, None, :],
        sem="relaxed",
    )


@triton.jit
def _triangular_accumulate_b4_12_ptr_active_fwd_kernel(
    base_ptr,
    dz0_ptr,
    dz1_ptr,
    dz2_ptr,
    dz3_ptr,
    dz4_ptr,
    dz5_ptr,
    dz6_ptr,
    dz7_ptr,
    dz8_ptr,
    dz9_ptr,
    dz10_ptr,
    dz11_ptr,
    gamma_ptr,
    out_ptr,
    total_low: tl.constexpr,
    channels: tl.constexpr,
    block_m: tl.constexpr,
    block_c: tl.constexpr,
    active: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_c = tl.program_id(1)
    offs_m = pid_m * block_m + tl.arange(0, block_m)
    offs_c = pid_c * block_c + tl.arange(0, block_c)
    mask = (offs_m[:, None] < total_low) & (offs_c[None, :] < channels)
    offsets = offs_m[:, None] * channels + offs_c[None, :]
    gamma_offsets = offs_c
    acc = tl.load(base_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    if active >= 1:
        g0 = tl.load(gamma_ptr + gamma_offsets, mask=offs_c < channels, other=0.0).to(tl.float32)
        d0 = tl.load(dz0_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        acc += d0 * g0[None, :]
    if active >= 2:
        g1 = tl.load(gamma_ptr + channels + gamma_offsets, mask=offs_c < channels, other=0.0).to(tl.float32)
        d1 = tl.load(dz1_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        acc += d1 * g1[None, :]
    if active >= 3:
        g2 = tl.load(gamma_ptr + 2 * channels + gamma_offsets, mask=offs_c < channels, other=0.0).to(tl.float32)
        d2 = tl.load(dz2_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        acc += d2 * g2[None, :]
    if active >= 4:
        g3 = tl.load(gamma_ptr + 3 * channels + gamma_offsets, mask=offs_c < channels, other=0.0).to(tl.float32)
        d3 = tl.load(dz3_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        acc += d3 * g3[None, :]
    if active >= 5:
        g4 = tl.load(gamma_ptr + 4 * channels + gamma_offsets, mask=offs_c < channels, other=0.0).to(tl.float32)
        d4 = tl.load(dz4_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        acc += d4 * g4[None, :]
    if active >= 6:
        g5 = tl.load(gamma_ptr + 5 * channels + gamma_offsets, mask=offs_c < channels, other=0.0).to(tl.float32)
        d5 = tl.load(dz5_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        acc += d5 * g5[None, :]
    if active >= 7:
        g6 = tl.load(gamma_ptr + 6 * channels + gamma_offsets, mask=offs_c < channels, other=0.0).to(tl.float32)
        d6 = tl.load(dz6_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        acc += d6 * g6[None, :]
    if active >= 8:
        g7 = tl.load(gamma_ptr + 7 * channels + gamma_offsets, mask=offs_c < channels, other=0.0).to(tl.float32)
        d7 = tl.load(dz7_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        acc += d7 * g7[None, :]
    if active >= 9:
        g8 = tl.load(gamma_ptr + 8 * channels + gamma_offsets, mask=offs_c < channels, other=0.0).to(tl.float32)
        d8 = tl.load(dz8_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        acc += d8 * g8[None, :]
    if active >= 10:
        g9 = tl.load(gamma_ptr + 9 * channels + gamma_offsets, mask=offs_c < channels, other=0.0).to(tl.float32)
        d9 = tl.load(dz9_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        acc += d9 * g9[None, :]
    if active >= 11:
        g10 = tl.load(gamma_ptr + 10 * channels + gamma_offsets, mask=offs_c < channels, other=0.0).to(tl.float32)
        d10 = tl.load(dz10_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        acc += d10 * g10[None, :]
    if active >= 12:
        g11 = tl.load(gamma_ptr + 11 * channels + gamma_offsets, mask=offs_c < channels, other=0.0).to(tl.float32)
        d11 = tl.load(dz11_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        acc += d11 * g11[None, :]

    tl.store(out_ptr + offsets, acc, mask=mask)


@triton.jit
def _triangular_accumulate_b4_12_ptr_grad_dz_kernel(
    grad_out_ptr,
    gamma_ptr,
    grad_dz_ptr,
    total_low: tl.constexpr,
    channels: tl.constexpr,
    block_m: tl.constexpr,
    block_c: tl.constexpr,
):
    pid_l = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_c = tl.program_id(2)
    offs_m = pid_m * block_m + tl.arange(0, block_m)
    offs_c = pid_c * block_c + tl.arange(0, block_c)
    mask = (offs_m[:, None] < total_low) & (offs_c[None, :] < channels)
    offsets = offs_m[:, None] * channels + offs_c[None, :]
    go_vals = tl.load(grad_out_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    gamma_vals = tl.load(gamma_ptr + pid_l * channels + offs_c, mask=offs_c < channels, other=0.0).to(tl.float32)
    out = go_vals * gamma_vals[None, :]
    layer_stride = total_low * channels
    tl.store(grad_dz_ptr + pid_l * layer_stride + offsets, out, mask=mask)


@triton.jit
def _triangular_accumulate_b4_12_ptr_grad_gamma_kernel(
    grad_out_ptr,
    dz0_ptr,
    dz1_ptr,
    dz2_ptr,
    dz3_ptr,
    dz4_ptr,
    dz5_ptr,
    dz6_ptr,
    dz7_ptr,
    dz8_ptr,
    dz9_ptr,
    dz10_ptr,
    dz11_ptr,
    grad_gamma_ptr,
    total_low: tl.constexpr,
    channels: tl.constexpr,
    block_m: tl.constexpr,
    block_c: tl.constexpr,
):
    pid_l = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_c = tl.program_id(2)
    offs_m = pid_m * block_m + tl.arange(0, block_m)
    offs_c = pid_c * block_c + tl.arange(0, block_c)
    mask = (offs_m[:, None] < total_low) & (offs_c[None, :] < channels)
    offsets = offs_m[:, None] * channels + offs_c[None, :]
    dz_ptr = tl.where(
        pid_l == 0,
        dz0_ptr,
        tl.where(
            pid_l == 1,
            dz1_ptr,
            tl.where(
                pid_l == 2,
                dz2_ptr,
                tl.where(
                    pid_l == 3,
                    dz3_ptr,
                    tl.where(
                        pid_l == 4,
                        dz4_ptr,
                        tl.where(
                            pid_l == 5,
                            dz5_ptr,
                            tl.where(
                                pid_l == 6,
                                dz6_ptr,
                                tl.where(
                                    pid_l == 7,
                                    dz7_ptr,
                                    tl.where(
                                        pid_l == 8,
                                        dz8_ptr,
                                        tl.where(pid_l == 9, dz9_ptr, tl.where(pid_l == 10, dz10_ptr, dz11_ptr)),
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        ),
    )
    go_vals = tl.load(grad_out_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    dz_vals = tl.load(dz_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    grad = tl.sum(go_vals * dz_vals, axis=0)
    tl.atomic_add(grad_gamma_ptr + pid_l * channels + offs_c, grad, sem="relaxed", mask=offs_c < channels)


@triton.jit
def _triangular_accumulate_b4_12_ptr_grad_fused_kernel(
    grad_out_ptr,
    dz0_ptr,
    dz1_ptr,
    dz2_ptr,
    dz3_ptr,
    dz4_ptr,
    dz5_ptr,
    dz6_ptr,
    dz7_ptr,
    dz8_ptr,
    dz9_ptr,
    dz10_ptr,
    dz11_ptr,
    gamma_ptr,
    grad_dz_ptr,
    grad_gamma_ptr,
    total_low: tl.constexpr,
    channels: tl.constexpr,
    block_m: tl.constexpr,
    block_c: tl.constexpr,
):
    """Compute grad_dz and grad_gamma while loading grad_out once."""
    pid_l = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_c = tl.program_id(2)
    offs_m = pid_m * block_m + tl.arange(0, block_m)
    offs_c = pid_c * block_c + tl.arange(0, block_c)
    mask = (offs_m[:, None] < total_low) & (offs_c[None, :] < channels)
    offsets = offs_m[:, None] * channels + offs_c[None, :]
    dz_ptr = tl.where(
        pid_l == 0,
        dz0_ptr,
        tl.where(
            pid_l == 1,
            dz1_ptr,
            tl.where(
                pid_l == 2,
                dz2_ptr,
                tl.where(
                    pid_l == 3,
                    dz3_ptr,
                    tl.where(
                        pid_l == 4,
                        dz4_ptr,
                        tl.where(
                            pid_l == 5,
                            dz5_ptr,
                            tl.where(
                                pid_l == 6,
                                dz6_ptr,
                                tl.where(
                                    pid_l == 7,
                                    dz7_ptr,
                                    tl.where(
                                        pid_l == 8,
                                        dz8_ptr,
                                        tl.where(
                                            pid_l == 9,
                                            dz9_ptr,
                                            tl.where(pid_l == 10, dz10_ptr, dz11_ptr),
                                        ),
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        ),
    )
    go_vals = tl.load(grad_out_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    dz_vals = tl.load(dz_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    gamma_vals = tl.load(
        gamma_ptr + pid_l * channels + offs_c,
        mask=offs_c < channels,
        other=0.0,
    ).to(tl.float32)

    layer_stride = total_low * channels
    tl.store(
        grad_dz_ptr + pid_l * layer_stride + offsets,
        go_vals * gamma_vals[None, :],
        mask=mask,
    )
    grad_gamma = tl.sum(go_vals * dz_vals, axis=0)
    tl.atomic_add(
        grad_gamma_ptr + pid_l * channels + offs_c,
        grad_gamma,
        sem="relaxed",
        mask=offs_c < channels,
    )


@triton_op("sihc::final_accumulate_b4_12_traceable", mutates_args=())
def final_accumulate_b4_12_traceable(
    x0: Tensor,
    dz0: Tensor,
    dz1: Tensor,
    dz2: Tensor,
    dz3: Tensor,
    dz4: Tensor,
    dz5: Tensor,
    dz6: Tensor,
    dz7: Tensor,
    dz8: Tensor,
    dz9: Tensor,
    dz10: Tensor,
    dz11: Tensor,
    beta0: Tensor,
    beta1: Tensor,
    beta2: Tensor,
    beta3: Tensor,
    beta4: Tensor,
    beta5: Tensor,
    beta6: Tensor,
    beta7: Tensor,
    beta8: Tensor,
    beta9: Tensor,
    beta10: Tensor,
    beta11: Tensor,
) -> Tensor:
    b = x0.shape[0]
    c = x0.shape[3]
    out = torch.empty_like(x0)
    block_c = _env_int("SIHC_FF12_FWD_BLOCK_C", 64)

    def grid(meta):
        return (b * 256, triton.cdiv(c, meta["block_c"]))

    wrap_triton(_final_accumulate_12_fwd_kernel)[grid](
        x0,
        dz0,
        dz1,
        dz2,
        dz3,
        dz4,
        dz5,
        dz6,
        dz7,
        dz8,
        dz9,
        dz10,
        dz11,
        beta0,
        beta1,
        beta2,
        beta3,
        beta4,
        beta5,
        beta6,
        beta7,
        beta8,
        beta9,
        beta10,
        beta11,
        out,
        b * 256,
        c,
        high_grid=64,
        cell_size=4,
        slots=16,
        block_c=block_c,
    )
    return out


@triton_op("sihc::final_accumulate_b4_12_grad_dz_ptr", mutates_args=())
def final_accumulate_b4_12_grad_dz_ptr_traceable(
    grad_out: Tensor,
    beta0: Tensor,
    beta1: Tensor,
    beta2: Tensor,
    beta3: Tensor,
    beta4: Tensor,
    beta5: Tensor,
    beta6: Tensor,
    beta7: Tensor,
    beta8: Tensor,
    beta9: Tensor,
    beta10: Tensor,
    beta11: Tensor,
) -> Tensor:
    b = grad_out.shape[0]
    c = grad_out.shape[3]
    grad_dz = torch.empty((12, b, 256, c), device=grad_out.device, dtype=grad_out.dtype)
    block_m = _env_int("SIHC_FF12_GDZ_BLOCK_M", 16)
    block_c = _env_int("SIHC_FF12_GDZ_BLOCK_C", 64)

    def grid(meta):
        return (12, triton.cdiv(b * 256, meta["block_m"]), triton.cdiv(c, meta["block_c"]))

    wrap_triton(_final_accumulate_12_grad_dz_ptr_kernel)[grid](
        grad_out,
        beta0,
        beta1,
        beta2,
        beta3,
        beta4,
        beta5,
        beta6,
        beta7,
        beta8,
        beta9,
        beta10,
        beta11,
        grad_dz,
        b * 256,
        c,
        high_grid=64,
        cell_size=4,
        slots=16,
        block_m=block_m,
        block_c=block_c,
    )
    return grad_dz


@triton_op("sihc::final_accumulate_b4_12_grad_beta_ptr", mutates_args=())
def final_accumulate_b4_12_grad_beta_ptr_traceable(
    grad_out: Tensor,
    dz0: Tensor,
    dz1: Tensor,
    dz2: Tensor,
    dz3: Tensor,
    dz4: Tensor,
    dz5: Tensor,
    dz6: Tensor,
    dz7: Tensor,
    dz8: Tensor,
    dz9: Tensor,
    dz10: Tensor,
    dz11: Tensor,
) -> Tensor:
    b = grad_out.shape[0]
    c = grad_out.shape[3]
    grad_beta = torch.empty((12, c, 16), device=grad_out.device, dtype=torch.float32)
    grad_beta.zero_()
    block_m = _env_int("SIHC_FF12_GBETA_BLOCK_M", 64)
    block_c = _env_int("SIHC_FF12_GBETA_BLOCK_C", 64)

    def grid(meta):
        return (12 * 16, triton.cdiv(b * 256, meta["block_m"]), triton.cdiv(c, meta["block_c"]))

    wrap_triton(_final_accumulate_12_grad_beta_ptr_kernel)[grid](
        grad_out,
        dz0,
        dz1,
        dz2,
        dz3,
        dz4,
        dz5,
        dz6,
        dz7,
        dz8,
        dz9,
        dz10,
        dz11,
        grad_beta,
        b * 256,
        c,
        high_grid=64,
        cell_size=4,
        slots=16,
        block_m=block_m,
        block_c=block_c,
    )
    return grad_beta


def _setup_final_accumulate_12_ctx(ctx, inputs, output):
    ctx.save_for_backward(*inputs[1:])


def _backward_final_accumulate_12(ctx, grad_out):
    saved = ctx.saved_tensors
    dzs = saved[:12]
    betas = saved[12:]
    if not grad_out.is_contiguous():
        grad_out = grad_out.contiguous()

    grad_dz_stack = final_accumulate_b4_12_grad_dz_ptr_traceable(grad_out, *betas)
    grad_beta_stack = final_accumulate_b4_12_grad_beta_ptr_traceable(grad_out, *dzs)
    grad_dzs = tuple(grad_dz_stack.unbind(dim=0))
    grad_betas = tuple(g.to(dtype=b.dtype) for g, b in zip(grad_beta_stack.unbind(dim=0), betas, strict=True))
    return (grad_out, *grad_dzs, *grad_betas)


register_autograd(
    final_accumulate_b4_12_traceable,
    _backward_final_accumulate_12,
    setup_context=_setup_final_accumulate_12_ctx,
)


@triton_op("sihc::final_accumulate_p8_12_traceable", mutates_args=())
def final_accumulate_p8_12_traceable(
    x0: Tensor,
    dz0: Tensor,
    dz1: Tensor,
    dz2: Tensor,
    dz3: Tensor,
    dz4: Tensor,
    dz5: Tensor,
    dz6: Tensor,
    dz7: Tensor,
    dz8: Tensor,
    dz9: Tensor,
    dz10: Tensor,
    dz11: Tensor,
    beta0: Tensor,
    beta1: Tensor,
    beta2: Tensor,
    beta3: Tensor,
    beta4: Tensor,
    beta5: Tensor,
    beta6: Tensor,
    beta7: Tensor,
    beta8: Tensor,
    beta9: Tensor,
    beta10: Tensor,
    beta11: Tensor,
) -> Tensor:
    b = x0.shape[0]
    c = x0.shape[3]
    if x0.shape[1] != 32 or x0.shape[2] != 32:
        raise ValueError(f"final_accumulate_p8_12 expects x0 [B,32,32,C], got {tuple(x0.shape)}")
    out = torch.empty_like(x0)
    block_c = _env_int("SIHC_P8_FF12_FWD_BLOCK_C", 64)

    def grid(meta):
        return (b * 256, triton.cdiv(c, meta["block_c"]))

    wrap_triton(_final_accumulate_12_fwd_kernel)[grid](
        x0,
        dz0,
        dz1,
        dz2,
        dz3,
        dz4,
        dz5,
        dz6,
        dz7,
        dz8,
        dz9,
        dz10,
        dz11,
        beta0,
        beta1,
        beta2,
        beta3,
        beta4,
        beta5,
        beta6,
        beta7,
        beta8,
        beta9,
        beta10,
        beta11,
        out,
        b * 256,
        c,
        high_grid=32,
        cell_size=2,
        slots=4,
        block_c=block_c,
    )
    return out


@triton_op("sihc::final_accumulate_p8_12_grad_dz_ptr", mutates_args=())
def final_accumulate_p8_12_grad_dz_ptr_traceable(
    grad_out: Tensor,
    beta0: Tensor,
    beta1: Tensor,
    beta2: Tensor,
    beta3: Tensor,
    beta4: Tensor,
    beta5: Tensor,
    beta6: Tensor,
    beta7: Tensor,
    beta8: Tensor,
    beta9: Tensor,
    beta10: Tensor,
    beta11: Tensor,
) -> Tensor:
    b = grad_out.shape[0]
    c = grad_out.shape[3]
    grad_dz = torch.empty(
        (12, b, 256, c), device=grad_out.device, dtype=grad_out.dtype
    )
    block_m = _env_int("SIHC_P8_FF12_GDZ_BLOCK_M", 16)
    block_c = _env_int("SIHC_P8_FF12_GDZ_BLOCK_C", 64)

    def grid(meta):
        return (
            12,
            triton.cdiv(b * 256, meta["block_m"]),
            triton.cdiv(c, meta["block_c"]),
        )

    wrap_triton(_final_accumulate_12_grad_dz_ptr_kernel)[grid](
        grad_out,
        beta0,
        beta1,
        beta2,
        beta3,
        beta4,
        beta5,
        beta6,
        beta7,
        beta8,
        beta9,
        beta10,
        beta11,
        grad_dz,
        b * 256,
        c,
        high_grid=32,
        cell_size=2,
        slots=4,
        block_m=block_m,
        block_c=block_c,
    )
    return grad_dz


@triton_op("sihc::final_accumulate_p8_12_grad_beta_ptr", mutates_args=())
def final_accumulate_p8_12_grad_beta_ptr_traceable(
    grad_out: Tensor,
    dz0: Tensor,
    dz1: Tensor,
    dz2: Tensor,
    dz3: Tensor,
    dz4: Tensor,
    dz5: Tensor,
    dz6: Tensor,
    dz7: Tensor,
    dz8: Tensor,
    dz9: Tensor,
    dz10: Tensor,
    dz11: Tensor,
) -> Tensor:
    b = grad_out.shape[0]
    c = grad_out.shape[3]
    grad_beta = torch.zeros((12, c, 4), device=grad_out.device, dtype=torch.float32)
    block_m = _env_int("SIHC_P8_FF12_GBETA_BLOCK_M", 64)
    block_c = _env_int("SIHC_P8_FF12_GBETA_BLOCK_C", 64)

    def grid(meta):
        return (
            12 * 4,
            triton.cdiv(b * 256, meta["block_m"]),
            triton.cdiv(c, meta["block_c"]),
        )

    wrap_triton(_final_accumulate_12_grad_beta_ptr_kernel)[grid](
        grad_out,
        dz0,
        dz1,
        dz2,
        dz3,
        dz4,
        dz5,
        dz6,
        dz7,
        dz8,
        dz9,
        dz10,
        dz11,
        grad_beta,
        b * 256,
        c,
        high_grid=32,
        cell_size=2,
        slots=4,
        block_m=block_m,
        block_c=block_c,
    )
    return grad_beta


def _backward_final_accumulate_p8_12(ctx, grad_out):
    saved = ctx.saved_tensors
    dzs = saved[:12]
    betas = saved[12:]
    if not grad_out.is_contiguous():
        grad_out = grad_out.contiguous()
    grad_dz_stack = final_accumulate_p8_12_grad_dz_ptr_traceable(
        grad_out, *betas
    )
    grad_beta_stack = final_accumulate_p8_12_grad_beta_ptr_traceable(
        grad_out, *dzs
    )
    grad_dzs = tuple(grad_dz_stack.unbind(dim=0))
    grad_betas = tuple(
        grad.to(dtype=beta.dtype)
        for grad, beta in zip(
            grad_beta_stack.unbind(dim=0), betas, strict=True
        )
    )
    return (grad_out, *grad_dzs, *grad_betas)


register_autograd(
    final_accumulate_p8_12_traceable,
    _backward_final_accumulate_p8_12,
    setup_context=_setup_final_accumulate_12_ctx,
)




@triton_op("sihc::final_accumulate_b4_8_traceable", mutates_args=())
def final_accumulate_b4_8_traceable(
    x0: Tensor,
    dz0: Tensor,
    dz1: Tensor,
    dz2: Tensor,
    dz3: Tensor,
    dz4: Tensor,
    dz5: Tensor,
    dz6: Tensor,
    dz7: Tensor,
    beta0: Tensor,
    beta1: Tensor,
    beta2: Tensor,
    beta3: Tensor,
    beta4: Tensor,
    beta5: Tensor,
    beta6: Tensor,
    beta7: Tensor,
) -> Tensor:
    b = x0.shape[0]
    c = x0.shape[3]
    out = torch.empty_like(x0)
    block_c = _env_int("SIHC_FF8_FWD_BLOCK_C", 64)

    def grid(meta):
        return (b * 256, triton.cdiv(c, meta["block_c"]))

    wrap_triton(_final_accumulate_b4_8_fwd_kernel)[grid](
        x0,
        dz0,
        dz1,
        dz2,
        dz3,
        dz4,
        dz5,
        dz6,
        dz7,
        beta0,
        beta1,
        beta2,
        beta3,
        beta4,
        beta5,
        beta6,
        beta7,
        out,
        b * 256,
        c,
        block_c=block_c,
    )
    return out


@triton_op("sihc::final_accumulate_b4_8_grad_dz_ptr", mutates_args=())
def final_accumulate_b4_8_grad_dz_ptr_traceable(
    grad_out: Tensor,
    beta0: Tensor,
    beta1: Tensor,
    beta2: Tensor,
    beta3: Tensor,
    beta4: Tensor,
    beta5: Tensor,
    beta6: Tensor,
    beta7: Tensor,
) -> Tensor:
    b = grad_out.shape[0]
    c = grad_out.shape[3]
    grad_dz = torch.empty((8, b, 256, c), device=grad_out.device, dtype=grad_out.dtype)
    block_m = _env_int("SIHC_FF8_GDZ_BLOCK_M", 16)
    block_c = _env_int("SIHC_FF8_GDZ_BLOCK_C", 64)

    def grid(meta):
        return (8, triton.cdiv(b * 256, meta["block_m"]), triton.cdiv(c, meta["block_c"]))

    wrap_triton(_final_accumulate_b4_8_grad_dz_ptr_kernel)[grid](
        grad_out,
        beta0,
        beta1,
        beta2,
        beta3,
        beta4,
        beta5,
        beta6,
        beta7,
        grad_dz,
        b * 256,
        c,
        block_m=block_m,
        block_c=block_c,
    )
    return grad_dz


@triton_op("sihc::final_accumulate_b4_8_grad_beta_ptr", mutates_args=())
def final_accumulate_b4_8_grad_beta_ptr_traceable(
    grad_out: Tensor,
    dz0: Tensor,
    dz1: Tensor,
    dz2: Tensor,
    dz3: Tensor,
    dz4: Tensor,
    dz5: Tensor,
    dz6: Tensor,
    dz7: Tensor,
) -> Tensor:
    b = grad_out.shape[0]
    c = grad_out.shape[3]
    grad_beta = torch.empty((8, c, 16), device=grad_out.device, dtype=torch.float32)
    grad_beta.zero_()
    block_m = _env_int("SIHC_FF8_GBETA_BLOCK_M", 64)
    block_c = _env_int("SIHC_FF8_GBETA_BLOCK_C", 64)

    def grid(meta):
        return (8 * 16, triton.cdiv(b * 256, meta["block_m"]), triton.cdiv(c, meta["block_c"]))

    wrap_triton(_final_accumulate_b4_8_grad_beta_ptr_kernel)[grid](
        grad_out,
        dz0,
        dz1,
        dz2,
        dz3,
        dz4,
        dz5,
        dz6,
        dz7,
        grad_beta,
        b * 256,
        c,
        block_m=block_m,
        block_c=block_c,
    )
    return grad_beta


def _setup_final_accumulate_8_ctx(ctx, inputs, output):
    ctx.save_for_backward(*inputs[1:])


def _backward_final_accumulate_8(ctx, grad_out):
    saved = ctx.saved_tensors
    dzs = saved[:8]
    betas = saved[8:]
    if not grad_out.is_contiguous():
        grad_out = grad_out.contiguous()

    grad_dz_stack = final_accumulate_b4_8_grad_dz_ptr_traceable(grad_out, *betas)
    grad_beta_stack = final_accumulate_b4_8_grad_beta_ptr_traceable(grad_out, *dzs)
    grad_dzs = tuple(grad_dz_stack.unbind(dim=0))
    grad_betas = tuple(g.to(dtype=b.dtype) for g, b in zip(grad_beta_stack.unbind(dim=0), betas, strict=True))
    return (grad_out, *grad_dzs, *grad_betas)


register_autograd(
    final_accumulate_b4_8_traceable,
    _backward_final_accumulate_8,
    setup_context=_setup_final_accumulate_8_ctx,
)


def _make_triangular_accumulate_b4_12_ptr_active_traceable(active: int):
    if not 1 <= active <= 11:
        raise ValueError(f"active must be in [1, 11], got {active}")

    @triton_op(f"sihc::triangular_accumulate_b4_12_ptr_active{active}", mutates_args=())
    def triangular_accumulate_b4_12_ptr_active_traceable(
        base: Tensor,
        dz0: Tensor,
        dz1: Tensor,
        dz2: Tensor,
        dz3: Tensor,
        dz4: Tensor,
        dz5: Tensor,
        dz6: Tensor,
        dz7: Tensor,
        dz8: Tensor,
        dz9: Tensor,
        dz10: Tensor,
        dz11: Tensor,
        gamma: Tensor,
    ) -> Tensor:
        b = base.shape[0]
        c = base.shape[2]
        if base.shape[1] != 256 or gamma.shape != (12, c):
            raise ValueError(
                f"triangular_accumulate_b4_12_active expects base [B,256,C] and gamma [12,C], "
                f"got {tuple(base.shape)} and {tuple(gamma.shape)}"
            )
        for idx, dz in enumerate((dz0, dz1, dz2, dz3, dz4, dz5, dz6, dz7, dz8, dz9, dz10, dz11)):
            if dz.shape != base.shape:
                raise ValueError(
                    f"triangular_accumulate_b4_12_active dz{idx} shape {tuple(dz.shape)} "
                    f"does not match base {tuple(base.shape)}"
                )
        out = torch.empty_like(base)

        def grid(meta):
            return (triton.cdiv(b * 256, meta["block_m"]), triton.cdiv(c, meta["block_c"]))

        wrap_triton(_triangular_accumulate_b4_12_ptr_active_fwd_kernel)[grid](
            base,
            dz0,
            dz1,
            dz2,
            dz3,
            dz4,
            dz5,
            dz6,
            dz7,
            dz8,
            dz9,
            dz10,
            dz11,
            gamma,
            out,
            b * 256,
            c,
            block_m=16,
            block_c=64,
            active=active,
        )
        return out

    @triton_op(f"sihc::triangular_accumulate_b4_12_ptr_active{active}_grad_dz", mutates_args=())
    def triangular_accumulate_b4_12_ptr_active_grad_dz_traceable(grad_out: Tensor, gamma: Tensor) -> Tensor:
        b = grad_out.shape[0]
        c = grad_out.shape[2]
        grad_dz = torch.empty((active, b, 256, c), device=grad_out.device, dtype=grad_out.dtype)

        def grid(meta):
            return (active, triton.cdiv(b * 256, meta["block_m"]), triton.cdiv(c, meta["block_c"]))

        wrap_triton(_triangular_accumulate_b4_12_ptr_grad_dz_kernel)[grid](
            grad_out,
            gamma,
            grad_dz,
            b * 256,
            c,
            block_m=16,
            block_c=64,
        )
        return grad_dz

    @triton_op(f"sihc::triangular_accumulate_b4_12_ptr_active{active}_grad_gamma", mutates_args=())
    def triangular_accumulate_b4_12_ptr_active_grad_gamma_traceable(
        grad_out: Tensor,
        dz0: Tensor,
        dz1: Tensor,
        dz2: Tensor,
        dz3: Tensor,
        dz4: Tensor,
        dz5: Tensor,
        dz6: Tensor,
        dz7: Tensor,
        dz8: Tensor,
        dz9: Tensor,
        dz10: Tensor,
        dz11: Tensor,
        gamma: Tensor,
    ) -> Tensor:
        b = grad_out.shape[0]
        c = grad_out.shape[2]
        grad_gamma = torch.empty((12, c), device=grad_out.device, dtype=torch.float32)
        grad_gamma.zero_()

        def grid(meta):
            return (active, triton.cdiv(b * 256, meta["block_m"]), triton.cdiv(c, meta["block_c"]))

        wrap_triton(_triangular_accumulate_b4_12_ptr_grad_gamma_kernel)[grid](
            grad_out,
            dz0,
            dz1,
            dz2,
            dz3,
            dz4,
            dz5,
            dz6,
            dz7,
            dz8,
            dz9,
            dz10,
            dz11,
            grad_gamma,
            b * 256,
            c,
            block_m=64,
            block_c=64,
        )
        return grad_gamma

    @triton_op(f"sihc::triangular_accumulate_b4_12_ptr_active{active}_grad_fused", mutates_args=())
    def triangular_accumulate_b4_12_ptr_active_grad_fused_traceable(
        grad_out: Tensor,
        dz0: Tensor,
        dz1: Tensor,
        dz2: Tensor,
        dz3: Tensor,
        dz4: Tensor,
        dz5: Tensor,
        dz6: Tensor,
        dz7: Tensor,
        dz8: Tensor,
        dz9: Tensor,
        dz10: Tensor,
        dz11: Tensor,
        gamma: Tensor,
    ) -> tuple[Tensor, Tensor]:
        b = grad_out.shape[0]
        c = grad_out.shape[2]
        grad_dz = torch.empty((active, b, 256, c), device=grad_out.device, dtype=grad_out.dtype)
        grad_gamma = torch.zeros((12, c), device=grad_out.device, dtype=torch.float32)
        block_m = _env_int("SIHC_TRI12_BWD_BLOCK_M", 64)
        block_c = _env_int("SIHC_TRI12_BWD_BLOCK_C", 64)

        def grid(meta):
            return (
                active,
                triton.cdiv(b * 256, meta["block_m"]),
                triton.cdiv(c, meta["block_c"]),
            )

        wrap_triton(_triangular_accumulate_b4_12_ptr_grad_fused_kernel)[grid](
            grad_out,
            dz0,
            dz1,
            dz2,
            dz3,
            dz4,
            dz5,
            dz6,
            dz7,
            dz8,
            dz9,
            dz10,
            dz11,
            gamma,
            grad_dz,
            grad_gamma,
            b * 256,
            c,
            block_m=block_m,
            block_c=block_c,
        )
        return grad_dz, grad_gamma

    def _setup_active_ctx(ctx, inputs, output):
        ctx.save_for_backward(*inputs[1:])

    def _backward_active(ctx, grad_out):
        saved = ctx.saved_tensors
        dzs = saved[:12]
        gamma = saved[12]
        if not grad_out.is_contiguous():
            grad_out = grad_out.contiguous()
        grad_base = grad_out
        grad_dz_stack, grad_gamma = triangular_accumulate_b4_12_ptr_active_grad_fused_traceable(
            grad_out, *dzs, gamma
        )
        active_grads = tuple(grad_dz_stack.unbind(dim=0))
        grad_dzs = active_grads + (None,) * (12 - active)
        return (grad_base, *grad_dzs, grad_gamma.to(dtype=gamma.dtype))

    register_autograd(
        triangular_accumulate_b4_12_ptr_active_traceable,
        _backward_active,
        setup_context=_setup_active_ctx,
    )

    return triangular_accumulate_b4_12_ptr_active_traceable


# The 11 active-prefix ops are deliberately used by the fast SiHC path.
# A separate op per prefix length lets torch.compile see the active prefix as a
# constant and avoids a runtime branch inside the triangular accumulation path.
triangular_accumulate_b4_12_ptr_active_traceable_ops = tuple(
    _make_triangular_accumulate_b4_12_ptr_active_traceable(active) for active in range(1, 12)
)





def _make_triangular_accumulate_b4_8_ptr_active_traceable(active: int):
    if not 1 <= active <= 7:
        raise ValueError(f"active must be in [1, 7], got {active}")

    @triton_op(f"sihc::triangular_accumulate_b4_8_ptr_active{active}", mutates_args=())
    def triangular_accumulate_b4_8_ptr_active_traceable(
        base: Tensor,
    dz0: Tensor,
    dz1: Tensor,
    dz2: Tensor,
    dz3: Tensor,
    dz4: Tensor,
    dz5: Tensor,
    dz6: Tensor,
    dz7: Tensor,
        gamma: Tensor,
    ) -> Tensor:
        b = base.shape[0]
        c = base.shape[2]
        if base.shape[1] != 256 or gamma.shape != (8, c):
            raise ValueError(
                f"triangular_accumulate_b4_8_active expects base [B,256,C] and gamma [8,C], "
                f"got {tuple(base.shape)} and {tuple(gamma.shape)}"
            )
        for idx, dz in enumerate((dz0, dz1, dz2, dz3, dz4, dz5, dz6, dz7)):
            if dz.shape != base.shape:
                raise ValueError(
                    f"triangular_accumulate_b4_8_active dz{idx} shape {tuple(dz.shape)} "
                    f"does not match base {tuple(base.shape)}"
                )
        out = torch.empty_like(base)

        def grid(meta):
            return (triton.cdiv(b * 256, meta["block_m"]), triton.cdiv(c, meta["block_c"]))

        wrap_triton(_triangular_accumulate_b4_12_ptr_active_fwd_kernel)[grid](
            base,
        dz0,
        dz1,
        dz2,
        dz3,
        dz4,
        dz5,
        dz6,
        dz7,
            dz7,
            dz7,
            dz7,
            dz7,
            gamma,
            out,
            b * 256,
            c,
            block_m=16,
            block_c=64,
            active=active,
        )
        return out

    @triton_op(f"sihc::triangular_accumulate_b4_8_ptr_active{active}_grad_dz", mutates_args=())
    def triangular_accumulate_b4_8_ptr_active_grad_dz_traceable(grad_out: Tensor, gamma: Tensor) -> Tensor:
        b = grad_out.shape[0]
        c = grad_out.shape[2]
        grad_dz = torch.empty((active, b, 256, c), device=grad_out.device, dtype=grad_out.dtype)

        def grid(meta):
            return (active, triton.cdiv(b * 256, meta["block_m"]), triton.cdiv(c, meta["block_c"]))

        wrap_triton(_triangular_accumulate_b4_12_ptr_grad_dz_kernel)[grid](
            grad_out,
            gamma,
            grad_dz,
            b * 256,
            c,
            block_m=16,
            block_c=64,
        )
        return grad_dz

    @triton_op(f"sihc::triangular_accumulate_b4_8_ptr_active{active}_grad_gamma", mutates_args=())
    def triangular_accumulate_b4_8_ptr_active_grad_gamma_traceable(
        grad_out: Tensor,
    dz0: Tensor,
    dz1: Tensor,
    dz2: Tensor,
    dz3: Tensor,
    dz4: Tensor,
    dz5: Tensor,
    dz6: Tensor,
    dz7: Tensor,
        gamma: Tensor,
    ) -> Tensor:
        b = grad_out.shape[0]
        c = grad_out.shape[2]
        grad_gamma = torch.empty((8, c), device=grad_out.device, dtype=torch.float32)
        grad_gamma.zero_()

        def grid(meta):
            return (active, triton.cdiv(b * 256, meta["block_m"]), triton.cdiv(c, meta["block_c"]))

        wrap_triton(_triangular_accumulate_b4_12_ptr_grad_gamma_kernel)[grid](
            grad_out,
        dz0,
        dz1,
        dz2,
        dz3,
        dz4,
        dz5,
        dz6,
        dz7,
            dz7,
            dz7,
            dz7,
            dz7,
            grad_gamma,
            b * 256,
            c,
            block_m=64,
            block_c=64,
        )
        return grad_gamma

    @triton_op(f"sihc::triangular_accumulate_b4_8_ptr_active{active}_grad_fused", mutates_args=())
    def triangular_accumulate_b4_8_ptr_active_grad_fused_traceable(
        grad_out: Tensor,
        dz0: Tensor,
        dz1: Tensor,
        dz2: Tensor,
        dz3: Tensor,
        dz4: Tensor,
        dz5: Tensor,
        dz6: Tensor,
        dz7: Tensor,
        gamma: Tensor,
    ) -> tuple[Tensor, Tensor]:
        b = grad_out.shape[0]
        c = grad_out.shape[2]
        grad_dz = torch.empty((active, b, 256, c), device=grad_out.device, dtype=grad_out.dtype)
        grad_gamma = torch.zeros((8, c), device=grad_out.device, dtype=torch.float32)
        block_m = _env_int("SIHC_TRI8_BWD_BLOCK_M", 64)
        block_c = _env_int("SIHC_TRI8_BWD_BLOCK_C", 64)

        def grid(meta):
            return (
                active,
                triton.cdiv(b * 256, meta["block_m"]),
                triton.cdiv(c, meta["block_c"]),
            )

        wrap_triton(_triangular_accumulate_b4_12_ptr_grad_fused_kernel)[grid](
            grad_out,
            dz0,
            dz1,
            dz2,
            dz3,
            dz4,
            dz5,
            dz6,
            dz7,
            dz7,
            dz7,
            dz7,
            dz7,
            gamma,
            grad_dz,
            grad_gamma,
            b * 256,
            c,
            block_m=block_m,
            block_c=block_c,
        )
        return grad_dz, grad_gamma

    def _setup_active_ctx(ctx, inputs, output):
        ctx.save_for_backward(*inputs[1:])

    def _backward_active(ctx, grad_out):
        saved = ctx.saved_tensors
        dzs = saved[:8]
        gamma = saved[8]
        if not grad_out.is_contiguous():
            grad_out = grad_out.contiguous()
        grad_base = grad_out
        grad_dz_stack, grad_gamma = triangular_accumulate_b4_8_ptr_active_grad_fused_traceable(
            grad_out, *dzs, gamma
        )
        active_grads = tuple(grad_dz_stack.unbind(dim=0))
        grad_dzs = active_grads + (None,) * (8 - active)
        return (grad_base, *grad_dzs, grad_gamma.to(dtype=gamma.dtype))

    register_autograd(
        triangular_accumulate_b4_8_ptr_active_traceable,
        _backward_active,
        setup_context=_setup_active_ctx,
    )

    return triangular_accumulate_b4_8_ptr_active_traceable


triangular_accumulate_b4_8_ptr_active_traceable_ops = tuple(
    _make_triangular_accumulate_b4_8_ptr_active_traceable(active) for active in range(1, 8)
)


@triton_op("sihc::all_base_b4_12_grad_x", mutates_args=())
def all_base_b4_12_grad_x_traceable(grad_base: Tensor, alpha_stack: Tensor) -> Tensor:
    b = grad_base.shape[1]
    c = grad_base.shape[3]
    grad_x = torch.empty((b, 64, 64, c), device=grad_base.device, dtype=grad_base.dtype)

    def grid(meta):
        return (triton.cdiv(b * 4096, meta["block_m"]), triton.cdiv(c, meta["block_c"]))

    wrap_triton(_all_base_12_grad_x_kernel)[grid](
        grad_base,
        alpha_stack,
        grad_x,
        b * 4096,
        c,
        high_grid=64,
        high_tokens=4096,
        cell_size=4,
        slots=16,
        block_m=16,
        block_c=64,
    )
    return grad_x


@triton_op("sihc::all_base_b4_12_split_traceable", mutates_args=())
def all_base_b4_12_split_traceable(x0: Tensor, alpha_stack: Tensor) -> list[Tensor]:
    b = x0.shape[0]
    c = x0.shape[3]
    if x0.shape[1] != 64 or x0.shape[2] != 64 or alpha_stack.shape != (12, c, 16):
        raise ValueError(
            f"all_base_b4_12_split expects x0 [B,64,64,C] and alpha [12,C,16], "
            f"got {tuple(x0.shape)} and {tuple(alpha_stack.shape)}"
        )
    outs = [torch.empty((b, 256, c), device=x0.device, dtype=x0.dtype) for _ in range(12)]
    block_m = _env_int("SIHC_READ12_SPLIT_BLOCK_M", 2)
    block_c = _env_int("SIHC_READ12_SPLIT_BLOCK_C", 64)
    num_warps = _env_int("SIHC_READ12_SPLIT_WARPS", 2)

    def grid(meta):
        return (triton.cdiv(b * 256, meta["block_m"]), triton.cdiv(c, meta["block_c"]))

    wrap_triton(_all_base_12_split_fwd_kernel)[grid](
        x0,
        alpha_stack,
        *outs,
        b * 256,
        c,
        high_grid=64,
        cell_size=4,
        slots=16,
        block_m=block_m,
        block_c=block_c,
        num_warps=num_warps,
    )
    return outs


@triton_op("sihc::all_base_b4_12_split_grad_alpha", mutates_args=())
def all_base_b4_12_split_grad_alpha_traceable(x0: Tensor, grad_base: Tensor) -> Tensor:
    b = x0.shape[0]
    c = x0.shape[3]
    grad_alpha = torch.zeros((12, c, 16), device=x0.device, dtype=torch.float32)
    rows = _env_int("SIHC_READ12_SPLIT_GRAD_ALPHA_ROWS", 64)
    block_c = _env_int("SIHC_READ12_SPLIT_GRAD_ALPHA_BLOCK_C", 64)
    num_warps = _env_int("SIHC_READ12_SPLIT_GRAD_ALPHA_WARPS", 8)

    def grid(meta):
        return (triton.cdiv(b * 256, meta["rows"]), triton.cdiv(c, meta["block_c"]))

    wrap_triton(_all_base_12_split_grad_alpha_kernel)[grid](
        x0,
        grad_base,
        grad_alpha,
        b * 256,
        c,
        high_grid=64,
        cell_size=4,
        slots=16,
        rows=rows,
        block_c=block_c,
        num_warps=num_warps,
    )
    return grad_alpha


def _setup_all_base_split_ctx(ctx, inputs, output):
    x0, alpha_stack = inputs
    ctx.save_for_backward(x0, alpha_stack)


def _backward_all_base_split(ctx, *grads):
    x0, alpha_stack = ctx.saved_tensors
    if len(grads) == 1 and isinstance(grads[0], (list, tuple)):
        grads = tuple(grads[0])
    grad_base = torch.stack([g.contiguous() for g in grads], dim=0)
    grad_x = all_base_b4_12_grad_x_traceable(grad_base, alpha_stack)
    grad_alpha = all_base_b4_12_split_grad_alpha_traceable(x0, grad_base)
    return grad_x, grad_alpha.to(dtype=alpha_stack.dtype)


register_autograd(
    all_base_b4_12_split_traceable,
    _backward_all_base_split,
    setup_context=_setup_all_base_split_ctx,
)


@triton_op("sihc::all_base_p8_12_grad_x", mutates_args=())
def all_base_p8_12_grad_x_traceable(
    grad_base: Tensor, alpha_stack: Tensor
) -> Tensor:
    b = grad_base.shape[1]
    c = grad_base.shape[3]
    grad_x = torch.empty(
        (b, 32, 32, c), device=grad_base.device, dtype=grad_base.dtype
    )

    def grid(meta):
        return (
            triton.cdiv(b * 1024, meta["block_m"]),
            triton.cdiv(c, meta["block_c"]),
        )

    wrap_triton(_all_base_12_grad_x_kernel)[grid](
        grad_base,
        alpha_stack,
        grad_x,
        b * 1024,
        c,
        high_grid=32,
        high_tokens=1024,
        cell_size=2,
        slots=4,
        block_m=16,
        block_c=64,
    )
    return grad_x


@triton_op("sihc::all_base_p8_12_split_traceable", mutates_args=())
def all_base_p8_12_split_traceable(
    x0: Tensor, alpha_stack: Tensor
) -> list[Tensor]:
    b = x0.shape[0]
    c = x0.shape[3]
    if x0.shape[1] != 32 or x0.shape[2] != 32 or alpha_stack.shape != (
        12,
        c,
        4,
    ):
        raise ValueError(
            "all_base_p8_12_split expects x0 [B,32,32,C] and "
            f"alpha [12,C,4], got {tuple(x0.shape)} and "
            f"{tuple(alpha_stack.shape)}"
        )
    outs = [
        torch.empty((b, 256, c), device=x0.device, dtype=x0.dtype)
        for _ in range(12)
    ]
    block_m = _env_int("SIHC_P8_READ12_SPLIT_BLOCK_M", 4)
    block_c = _env_int("SIHC_P8_READ12_SPLIT_BLOCK_C", 64)
    num_warps = _env_int("SIHC_P8_READ12_SPLIT_WARPS", 2)

    def grid(meta):
        return (
            triton.cdiv(b * 256, meta["block_m"]),
            triton.cdiv(c, meta["block_c"]),
        )

    wrap_triton(_all_base_12_split_fwd_kernel)[grid](
        x0,
        alpha_stack,
        *outs,
        b * 256,
        c,
        high_grid=32,
        cell_size=2,
        slots=4,
        block_m=block_m,
        block_c=block_c,
        num_warps=num_warps,
    )
    return outs


@triton_op("sihc::all_base_p8_12_split_grad_alpha", mutates_args=())
def all_base_p8_12_split_grad_alpha_traceable(
    x0: Tensor, grad_base: Tensor
) -> Tensor:
    b = x0.shape[0]
    c = x0.shape[3]
    grad_alpha = torch.zeros(
        (12, c, 4), device=x0.device, dtype=torch.float32
    )
    rows = _env_int("SIHC_P8_READ12_SPLIT_GRAD_ALPHA_ROWS", 64)
    block_c = _env_int("SIHC_P8_READ12_SPLIT_GRAD_ALPHA_BLOCK_C", 64)
    num_warps = _env_int("SIHC_P8_READ12_SPLIT_GRAD_ALPHA_WARPS", 4)

    def grid(meta):
        return (
            triton.cdiv(b * 256, meta["rows"]),
            triton.cdiv(c, meta["block_c"]),
        )

    wrap_triton(_all_base_12_split_grad_alpha_kernel)[grid](
        x0,
        grad_base,
        grad_alpha,
        b * 256,
        c,
        high_grid=32,
        cell_size=2,
        slots=4,
        rows=rows,
        block_c=block_c,
        num_warps=num_warps,
    )
    return grad_alpha


def _backward_all_base_p8_12_split(ctx, *grads):
    x0, alpha_stack = ctx.saved_tensors
    if len(grads) == 1 and isinstance(grads[0], (list, tuple)):
        grads = tuple(grads[0])
    grad_base = torch.stack([grad.contiguous() for grad in grads], dim=0)
    grad_x = all_base_p8_12_grad_x_traceable(grad_base, alpha_stack)
    grad_alpha = all_base_p8_12_split_grad_alpha_traceable(x0, grad_base)
    return grad_x, grad_alpha.to(dtype=alpha_stack.dtype)


register_autograd(
    all_base_p8_12_split_traceable,
    _backward_all_base_p8_12_split,
    setup_context=_setup_all_base_split_ctx,
)




@triton_op("sihc::all_base_b4_8_split_traceable", mutates_args=())
def all_base_b4_8_split_traceable(x0: Tensor, alpha_stack: Tensor) -> list[Tensor]:
    b = x0.shape[0]
    c = x0.shape[3]
    if x0.shape[1] != 64 or x0.shape[2] != 64 or alpha_stack.shape != (8, c, 16):
        raise ValueError(
            f"all_base_b4_8_split expects x0 [B,64,64,C] and alpha [8,C,16], "
            f"got {tuple(x0.shape)} and {tuple(alpha_stack.shape)}"
        )
    outs = [torch.empty((b, 256, c), device=x0.device, dtype=x0.dtype) for _ in range(8)]
    block_m = _env_int("SIHC_READ8_SPLIT_BLOCK_M", 4)
    block_c = _env_int("SIHC_READ8_SPLIT_BLOCK_C", 64)
    num_warps = _env_int("SIHC_READ8_SPLIT_WARPS", 2)

    def grid(meta):
        return (triton.cdiv(b * 256, meta["block_m"]), triton.cdiv(c, meta["block_c"]))

    wrap_triton(_all_base_b4_8_split_fwd_kernel)[grid](
        x0,
        alpha_stack,
        *outs,
        b * 256,
        c,
        block_m=block_m,
        block_c=block_c,
        num_warps=num_warps,
    )
    return outs


@triton_op("sihc::all_base_b4_8_grad_x", mutates_args=())
def all_base_b4_8_grad_x_traceable(grad_base: Tensor, alpha_stack: Tensor) -> Tensor:
    b = grad_base.shape[1]
    c = grad_base.shape[3]
    grad_x = torch.empty((b, 64, 64, c), device=grad_base.device, dtype=grad_base.dtype)

    def grid(meta):
        return (triton.cdiv(b * 4096, meta["block_m"]), triton.cdiv(c, meta["block_c"]))

    wrap_triton(_all_base_b4_8_grad_x_kernel)[grid](
        grad_base,
        alpha_stack,
        grad_x,
        b * 4096,
        c,
        block_m=16,
        block_c=64,
    )
    return grad_x


@triton_op("sihc::all_base_b4_8_split_grad_alpha", mutates_args=())
def all_base_b4_8_split_grad_alpha_traceable(x0: Tensor, grad_base: Tensor) -> Tensor:
    b = x0.shape[0]
    c = x0.shape[3]
    grad_alpha = torch.zeros((8, c, 16), device=x0.device, dtype=torch.float32)
    rows = _env_int("SIHC_READ8_SPLIT_GRAD_ALPHA_ROWS", 64)
    block_c = _env_int("SIHC_READ8_SPLIT_GRAD_ALPHA_BLOCK_C", 64)
    num_warps = _env_int("SIHC_READ8_SPLIT_GRAD_ALPHA_WARPS", 8)

    def grid(meta):
        return (triton.cdiv(b * 256, meta["rows"]), triton.cdiv(c, meta["block_c"]))

    wrap_triton(_all_base_b4_8_split_grad_alpha_kernel)[grid](
        x0,
        grad_base,
        grad_alpha,
        b * 256,
        c,
        rows=rows,
        block_c=block_c,
        num_warps=num_warps,
    )
    return grad_alpha


def _setup_all_base_8_split_ctx(ctx, inputs, output):
    x0, alpha_stack = inputs
    ctx.save_for_backward(x0, alpha_stack)


def _backward_all_base_8_split(ctx, *grads):
    x0, alpha_stack = ctx.saved_tensors
    if len(grads) == 1 and isinstance(grads[0], (list, tuple)):
        grads = tuple(grads[0])
    grad_base = torch.stack([g.contiguous() for g in grads], dim=0)
    grad_x = all_base_b4_8_grad_x_traceable(grad_base, alpha_stack)
    grad_alpha = all_base_b4_8_split_grad_alpha_traceable(x0, grad_base)
    return grad_x, grad_alpha.to(dtype=alpha_stack.dtype)


register_autograd(
    all_base_b4_8_split_traceable,
    _backward_all_base_8_split,
    setup_context=_setup_all_base_8_split_ctx,
)


@triton_op("sihc::cell_mean_b4_traceable", mutates_args=())
def cell_mean_b4_traceable(x: Tensor) -> Tensor:
    """Fixed, parameter-free stage seed for write-only SiHC."""
    if x.ndim != 4 or x.shape[1] != 64 or x.shape[2] != 64:
        raise ValueError(f"cell_mean_b4 expects x [B,64,64,C], got {tuple(x.shape)}")
    b, _, _, c = x.shape
    out = torch.empty((b, 256, c), device=x.device, dtype=x.dtype)
    block_m = _env_int("SIHC_MEAN_READ_BLOCK_M", 4)
    block_c = _env_int("SIHC_MEAN_READ_BLOCK_C", 64)

    def grid(meta):
        return (
            triton.cdiv(b * 256, meta["block_m"]),
            triton.cdiv(c, meta["block_c"]),
        )

    wrap_triton(_cell_mean_b4_fwd_kernel)[grid](
        x,
        out,
        b * 256,
        c,
        block_m=block_m,
        block_c=block_c,
    )
    return out


@triton_op("sihc::cell_mean_b4_backward_traceable", mutates_args=())
def cell_mean_b4_backward_traceable(grad_out: Tensor) -> Tensor:
    b, _, c = grad_out.shape
    grad_x = torch.empty((b, 64, 64, c), device=grad_out.device, dtype=grad_out.dtype)
    block_m = _env_int("SIHC_MEAN_READ_BWD_BLOCK_M", 16)
    block_c = _env_int("SIHC_MEAN_READ_BWD_BLOCK_C", 64)

    def grid(meta):
        return (
            triton.cdiv(b * 4096, meta["block_m"]),
            triton.cdiv(c, meta["block_c"]),
        )

    wrap_triton(_cell_mean_b4_bwd_kernel)[grid](
        grad_out,
        grad_x,
        b * 4096,
        c,
        block_m=block_m,
        block_c=block_c,
    )
    return grad_x


def _backward_cell_mean_b4(ctx, grad_out):
    del ctx
    return cell_mean_b4_backward_traceable(grad_out.contiguous())

register_autograd(cell_mean_b4_traceable, _backward_cell_mean_b4)
