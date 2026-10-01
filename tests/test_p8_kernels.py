from __future__ import annotations

import os

import pytest
import torch

from sihc.models.sihc.kernels import (
    all_base_p8_12_split_traceable,
    final_accumulate_p8_12_traceable,
)


pytestmark = pytest.mark.skipif(
    os.environ.get("SIHC_RUN_CUDA_TESTS") != "1" or not torch.cuda.is_available(),
    reason="set SIHC_RUN_CUDA_TESTS=1 on an idle CUDA GPU",
)


def _high_to_cells(x: torch.Tensor) -> torch.Tensor:
    batch, _, _, channels = x.shape
    return (
        x.reshape(batch, 16, 2, 16, 2, channels)
        .permute(0, 1, 3, 2, 4, 5)
        .reshape(batch, 256, 4, channels)
    )


def _cells_to_high(x: torch.Tensor) -> torch.Tensor:
    batch, _, _, channels = x.shape
    return (
        x.reshape(batch, 16, 16, 2, 2, channels)
        .permute(0, 1, 3, 2, 4, 5)
        .reshape(batch, 32, 32, channels)
    )


def _tolerances(dtype: torch.dtype) -> tuple[float, float]:
    return (0.25, 0.03) if dtype == torch.bfloat16 else (1e-3, 1e-4)


@pytest.mark.parametrize(
    "dtype", (torch.float32, torch.bfloat16), ids=("fp32", "bf16")
)
def test_all_base_p8_forward_and_backward_match_reference(
    dtype: torch.dtype,
) -> None:
    torch.manual_seed(810)
    x_actual = torch.randn(
        2, 32, 32, 37, device="cuda", dtype=dtype, requires_grad=True
    )
    alpha_actual = torch.randn(
        12, 37, 4, device="cuda", dtype=dtype, requires_grad=True
    )
    x_reference = x_actual.detach().clone().requires_grad_(True)
    alpha_reference = alpha_actual.detach().clone().requires_grad_(True)

    actual = all_base_p8_12_split_traceable(x_actual, alpha_actual)
    reference = torch.einsum(
        "bnsc,lcs->lbnc", _high_to_cells(x_reference), alpha_reference
    )
    upstream = torch.randn_like(reference)
    actual_loss = sum(
        (value * upstream[index]).sum()
        for index, value in enumerate(actual)
    )
    reference_loss = (reference * upstream).sum()
    actual_grads = torch.autograd.grad(
        actual_loss, (x_actual, alpha_actual)
    )
    reference_grads = torch.autograd.grad(
        reference_loss, (x_reference, alpha_reference)
    )
    atol, rtol = _tolerances(dtype)
    torch.testing.assert_close(
        torch.stack(actual), reference, atol=atol, rtol=rtol
    )
    for actual_grad, reference_grad in zip(
        actual_grads, reference_grads, strict=True
    ):
        torch.testing.assert_close(
            actual_grad, reference_grad, atol=atol, rtol=rtol
        )


@pytest.mark.parametrize(
    "dtype", (torch.float32, torch.bfloat16), ids=("fp32", "bf16")
)
def test_final_accumulate_p8_forward_and_backward_match_reference(
    dtype: torch.dtype,
) -> None:
    torch.manual_seed(812)
    x_actual = torch.randn(
        2, 32, 32, 37, device="cuda", dtype=dtype, requires_grad=True
    )
    dz_actual = [
        torch.randn(
            2, 256, 37, device="cuda", dtype=dtype, requires_grad=True
        )
        for _ in range(12)
    ]
    beta_actual = [
        torch.randn(37, 4, device="cuda", dtype=dtype, requires_grad=True)
        for _ in range(12)
    ]
    x_reference = x_actual.detach().clone().requires_grad_(True)
    dz_reference = [
        value.detach().clone().requires_grad_(True) for value in dz_actual
    ]
    beta_reference = [
        value.detach().clone().requires_grad_(True) for value in beta_actual
    ]

    actual = final_accumulate_p8_12_traceable(
        x_actual, *dz_actual, *beta_actual
    )
    reference_cells = _high_to_cells(x_reference) + torch.einsum(
        "lbnc,lcs->bnsc",
        torch.stack(dz_reference),
        torch.stack(beta_reference),
    )
    reference = _cells_to_high(reference_cells)
    upstream = torch.randn_like(reference)
    actual_grads = torch.autograd.grad(
        actual, (x_actual, *dz_actual, *beta_actual), upstream
    )
    reference_grads = torch.autograd.grad(
        reference,
        (x_reference, *dz_reference, *beta_reference),
        upstream,
    )
    atol, rtol = _tolerances(dtype)
    torch.testing.assert_close(actual, reference, atol=atol, rtol=rtol)
    for actual_grad, reference_grad in zip(
        actual_grads, reference_grads, strict=True
    ):
        torch.testing.assert_close(
            actual_grad, reference_grad, atol=atol, rtol=rtol
        )
