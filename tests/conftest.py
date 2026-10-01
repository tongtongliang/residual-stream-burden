"""Use full FP32 precision for reference comparisons, independent of host defaults."""
import pytest
import torch


@pytest.fixture(scope="session", autouse=True)
def full_precision_reference_math():
    # Convolution may otherwise use TF32 while the equivalent explicit GEMM
    # uses FP32, invalidating tight oracle tolerances. BF16 tests stay BF16.
    matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
    cudnn_tf32 = torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = matmul_tf32
        torch.backends.cudnn.allow_tf32 = cudnn_tf32
