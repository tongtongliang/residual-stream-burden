from __future__ import annotations

import os

import pytest
import torch

from sihc.models.sihc.kernels import (
    triangular_accumulate_b4_8_ptr_active_traceable_ops,
    triangular_accumulate_b4_12_ptr_active_traceable_ops,
)


pytestmark = pytest.mark.skipif(
    os.environ.get("SIHC_RUN_CUDA_TESTS") != "1" or not torch.cuda.is_available(),
    reason="set SIHC_RUN_CUDA_TESTS=1 on an idle CUDA GPU",
)


@pytest.mark.parametrize(
    "dtype", (torch.float32, torch.bfloat16), ids=("fp32", "bf16")
)
@pytest.mark.parametrize(
    ("stage_size", "active"),
    ((8, 1), (8, 4), (8, 7), (12, 1), (12, 6), (12, 11)),
)
def test_triangular_accumulate_fused_backward_matches_reference(
    stage_size: int, active: int, dtype: torch.dtype
) -> None:
    torch.manual_seed(1000 + stage_size * 10 + active)
    shape = (1, 256, 97)
    base_actual = torch.randn(
        shape, device="cuda", dtype=dtype, requires_grad=True
    )
    dz_actual = [
        torch.randn(shape, device="cuda", dtype=dtype, requires_grad=True)
        for _ in range(stage_size)
    ]
    gamma_actual = torch.randn(
        stage_size,
        shape[-1],
        device="cuda",
        dtype=dtype,
        requires_grad=True,
    )

    base_reference = base_actual.detach().clone().requires_grad_(True)
    dz_reference = [
        value.detach().clone().requires_grad_(True) for value in dz_actual
    ]
    gamma_reference = gamma_actual.detach().clone().requires_grad_(True)
    upstream = torch.randn(shape, device="cuda", dtype=dtype)

    ops = (
        triangular_accumulate_b4_8_ptr_active_traceable_ops
        if stage_size == 8
        else triangular_accumulate_b4_12_ptr_active_traceable_ops
    )
    actual = ops[active - 1](
        base_actual, *dz_actual, gamma_actual.contiguous()
    )
    reference = base_reference
    for index in range(active):
        reference = reference + dz_reference[index] * gamma_reference[index]

    actual_inputs = (base_actual, *dz_actual, gamma_actual)
    reference_inputs = (base_reference, *dz_reference, gamma_reference)
    actual_grads = torch.autograd.grad(
        actual, actual_inputs, upstream, allow_unused=True
    )
    reference_grads = torch.autograd.grad(
        reference, reference_inputs, upstream, allow_unused=True
    )

    atol = 0.25 if dtype == torch.bfloat16 else 2e-5
    rtol = 0.02 if dtype == torch.bfloat16 else 2e-5
    torch.testing.assert_close(actual, reference, atol=atol, rtol=rtol)
    for index, (actual_grad, reference_grad) in enumerate(
        zip(actual_grads, reference_grads, strict=True)
    ):
        if reference_grad is None:
            assert actual_grad is None, f"unexpected gradient for input {index}"
        else:
            assert actual_grad is not None
            torch.testing.assert_close(
                actual_grad, reference_grad, atol=atol, rtol=rtol
            )
