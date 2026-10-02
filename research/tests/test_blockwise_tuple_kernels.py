"""Block-wise tuple kernels: dense oracles, compatibility and CUDA contracts.

CUDA tests require an explicitly allocated device. Nonzero branch gates and
output weights make forward/gradient checks sensitive to the routing path.
"""
from __future__ import annotations

import copy
import os

import pytest
import torch

from sihc.checkpoint import build_model_from_checkpoint, load_checkpoint, load_model_state
from sihc.models.sihc.model import FusedP8SiHCModel, SiHCInContextModel, SiHCModel
from sihc.models.sihc.blockwise_kernels import (
    blockwise_read, blockwise_triangular, blockwise_write,
)

CUDA = pytest.mark.skipif(
    os.environ.get("SIHC_RUN_CUDA_TESTS") != "1" or not torch.cuda.is_available(),
    reason="set SIHC_RUN_CUDA_TESTS=1 on an allocated idle CUDA device",
)


def cells(x, slots):
    batch, height, _, channels = x.shape
    k = int(slots ** 0.5)
    grid = height // k
    return x.reshape(batch, grid, k, grid, k, channels).permute(0, 1, 3, 2, 4, 5).reshape(
        batch, grid * grid, slots, channels)


def high(x):
    batch, tokens, slots, channels = x.shape
    grid, k = int(tokens ** 0.5), int(slots ** 0.5)
    return x.reshape(batch, grid, grid, k, k, channels).permute(0, 1, 3, 2, 4, 5).reshape(
        batch, grid * k, grid * k, channels)


def model_for(case, backend):
    kwargs = dict(hidden_size=32, num_heads=4, attn_backend="math", kernel_backend=backend)
    if case == "p8":
        return FusedP8SiHCModel(patch_size=8, depth=12, fused_stage_size=12, **kwargs)
    if case == "ctx":
        return SiHCInContextModel(depth=20, stage_sizes=(8, 12),
                                  in_context_start=4, in_context_len=3, **kwargs)
    if case == "mixed":
        return SiHCModel(depth=20, stage_sizes=(8, 12), **kwargs)
    return SiHCModel(depth=8, fused_stage_size=8, **kwargs)


def nonzero_weights(model):
    with torch.no_grad():
        for block in model.blocks:
            block.branch.adaLN_modulation[-1].weight.normal_(std=0.01)
            block.branch.adaLN_modulation[-1].bias.normal_(std=0.03)
            block.connection.read_weight.add_(torch.randn_like(block.connection.read_weight) * 0.01)
            block.connection.write_weight.add_(torch.randn_like(block.connection.write_weight) * 0.02)
        model.final_layer.linear.weight.normal_(std=0.03)
    return model


@pytest.mark.parametrize("case", ["p4", "p8", "mixed", "ctx"])
def test_backend_preserves_parameter_schema_and_blockwise_topology(case):
    old, new = model_for(case, "legacy"), model_for(case, "tuple")
    new.load_state_dict(old.state_dict(), strict=True)
    assert tuple(old.state_dict()) == tuple(new.state_dict())
    for (name, p), (new_name, q) in zip(old.named_parameters(), new.named_parameters(), strict=True):
        assert name == new_name and p.shape == q.shape and p.stride() == q.stride()
        torch.testing.assert_close(p, q, atol=0, rtol=0)
    assert len([name for name, _ in new.named_parameters() if name.endswith("connection.read_weight")]) == new.depth
    assert not any("attention_connection" in name or "mlp_connection" in name for name, _ in new.named_parameters())
    assert new.stage_sizes == old.stage_sizes


@pytest.mark.parametrize("backend", ["legacy", "tuple"])
def test_checkpoint_reconstruction_and_backend_override(tmp_path, backend):
    model = model_for("p4", backend)
    ckpt = {"model": model.state_dict(), "ema": dict(model.named_parameters()),
            "meta": {"model": "sihc_1x12_d768_b4", "model_kwargs": {
                "hidden_size": 32, "num_heads": 4, "depth": 8,
                "fused_stage_size": 8, "kernel_backend": backend}}}
    path = tmp_path / "model.pt"
    torch.save(ckpt, path)
    loaded = load_checkpoint(path)
    for selected in ("legacy", "tuple"):
        restored = build_model_from_checkpoint(loaded, attn_backend="math",
                                               overrides={"kernel_backend": selected})
        assert restored.kernel_backend == selected
        for key in ("model", "ema"):
            load_model_state(restored, loaded, key)
            for p, q in zip(model.parameters(), restored.parameters(), strict=True):
                torch.testing.assert_close(p, q, atol=0, rtol=0)


def test_invalid_backend_fails_at_construction():
    with pytest.raises(ValueError, match="kernel_backend"):
        model_for("p4", "misspelled")


@pytest.mark.parametrize("backend", ["legacy", "tuple"])
@pytest.mark.parametrize("state_key", ["model", "ema", "ema2"])
def test_anonymous_export_preserves_validated_backend(backend, state_key):
    from evaluation.export_checkpoint import export_state

    model = model_for("p4", backend)
    source = {"model": model.state_dict(), "ema": dict(model.named_parameters()),
              "ema2": dict(model.named_parameters()),
              "args": {"model": "sihc_1x12_d768_b4", "prediction": "velocity"},
              "meta": {"model_kwargs": {"hidden_size": 32, "num_heads": 4, "depth": 8,
                                         "fused_stage_size": 8, "kernel_backend": backend}}}
    exported = export_state(source, state_key)
    assert exported["meta"]["model_kwargs"]["kernel_backend"] == backend
    restored = build_model_from_checkpoint(exported, attn_backend="math")
    assert restored.kernel_backend == backend
    load_model_state(restored, exported, "model")
    source["meta"]["model_kwargs"]["kernel_backend"] = "unrecognized"
    with pytest.raises(ValueError, match="kernel_backend"):
        export_state(source, state_key)


def test_old_checkpoint_without_backend_keeps_legacy_default():
    model = model_for("p4", "legacy")
    source = {"model": model.state_dict(), "meta": {"model": "sihc_1x12_d768_b4", "model_kwargs": {
        "hidden_size": 32, "num_heads": 4, "depth": 8, "fused_stage_size": 8}}}
    restored = build_model_from_checkpoint(source, attn_backend="math")
    assert restored.kernel_backend == "legacy"
    load_model_state(restored, source, "model")


@pytest.mark.parametrize("name", ["sihc_sublayer_1x12_d768_b4_ctx32s4",
                                   "sihc_reference_1x12_d768_b8"])
def test_explicit_tuple_rejects_other_routing_schedules(name):
    from sihc.models import build_model

    with torch.device("meta"), pytest.raises(ValueError, match="block-wise"):
        build_model(name, kernel_backend="tuple")


@pytest.mark.parametrize("case", ["p4", "ctx"])
def test_cpu_schedule_keeps_one_event_per_block(monkeypatch, case):
    # PyTorch substitutes validate scheduling and CTX packing only. Device
    # kernels themselves remain covered by the separately gated CUDA tests.
    import sihc.models.sihc.model as module
    calls = {"reads": [], "writes": [], "prefixes": []}

    def read(x, alpha):
        assert x.is_contiguous() and alpha.is_contiguous()
        calls["reads"].append(alpha.shape[0])
        return list(torch.einsum("bnsc,lcs->lbnc", cells(x, alpha.shape[-1]), alpha).unbind(0))

    def triangular(base, updates, gamma):
        assert gamma.is_contiguous() and all(z.is_contiguous() for z in updates)
        assert gamma.shape[0] == len(updates)
        calls["prefixes"].append(len(updates))
        return base + sum(z * g for z, g in zip(updates, gamma))

    def write(x, updates, beta):
        assert beta.is_contiguous() and all(z.is_contiguous() for z in updates)
        assert len(updates) == beta.shape[0]
        calls["writes"].append(len(updates))
        delta = torch.einsum("lbnc,lcs->bnsc", torch.stack(updates), beta)
        return high(cells(x, beta.shape[-1]) + delta)

    monkeypatch.setattr(module, "blockwise_read", read)
    monkeypatch.setattr(module, "blockwise_triangular", triangular)
    monkeypatch.setattr(module, "blockwise_write", write)
    torch.manual_seed(977)
    model = nonzero_weights(model_for(case, "tuple"))
    reference = copy.deepcopy(model)
    x, t, labels = torch.randn(2, 3, 256, 256), torch.rand(2), torch.tensor([1, 2])
    actual, expected = model(x, t, labels), literal_dense_forward(reference, x, t, labels)
    torch.testing.assert_close(actual, expected, atol=3e-5, rtol=3e-4)
    actual.square().mean().backward()
    expected.square().mean().backward()
    compare_model_gradients(model, reference, torch.float32, dense=True)
    assert calls["reads"] == calls["writes"] == list(model.stage_sizes)
    assert calls["prefixes"] == [i for size in model.stage_sizes for i in range(1, size)]


@CUDA
@pytest.mark.parametrize("events,slots", [(8, 16), (12, 16), (12, 4)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_read_write_adjoint_and_reduction_tails(events, slots, dtype):
    # M=363 crosses a 256-token weight reduction tile; C=67 crosses a
    # 64-channel tile. All 8/12 event tiles are partially occupied.
    torch.manual_seed(153)
    k = int(slots ** 0.5)
    x = torch.randn(3, 11 * k, 11 * k, 67, device="cuda", dtype=dtype).requires_grad_()
    a = (torch.randn(events, 67, slots, device="cuda", dtype=dtype) * 0.05).requires_grad_()
    xr, ar = [v.detach().clone().requires_grad_() for v in (x, a)]
    out = blockwise_read(x, a)
    ref = torch.einsum("bnsc,lcs->lbnc", cells(xr, slots).float(), ar.float()).to(dtype)
    upstream = torch.randn_like(ref) * 0.03
    actual_grads = torch.autograd.grad(sum((z * u).sum() for z, u in zip(out, upstream)), (x, a))
    reference_grads = torch.autograd.grad(ref, (xr, ar), upstream)
    tolerance = dict(atol=2e-4, rtol=2e-4) if dtype == torch.float32 else dict(atol=0.003, rtol=0.03)
    torch.testing.assert_close(torch.stack(out), ref, **tolerance)
    for actual, reference in zip(actual_grads, reference_grads):
        torch.testing.assert_close(actual, reference, **tolerance)

    z = [torch.randn(3, 121, 67, device="cuda", dtype=dtype).requires_grad_() for _ in range(events)]
    zr = [v.detach().clone().requires_grad_() for v in z]
    actual = blockwise_write(x, z, a)
    delta = (torch.stack(zr).float().unsqueeze(3) * ar.transpose(1, 2).float()[:, None, None]).sum(0)
    reference = high((cells(xr, slots).float() + delta).to(dtype))
    go = torch.randn_like(actual) * 0.03
    ga = torch.autograd.grad(actual, (x, *z, a), go)
    gr = torch.autograd.grad(reference, (xr, *zr, ar), go)
    torch.testing.assert_close(actual, reference, **tolerance)
    for actual_grad, reference_grad in zip(ga, gr):
        torch.testing.assert_close(actual_grad, reference_grad, **tolerance)


@CUDA
@pytest.mark.parametrize("events", [8, 12])
def test_unused_read_outputs_and_packed_prefix_contract(events):
    x = torch.randn(2, 16, 16, 37, device="cuda", requires_grad=True)
    alpha = torch.randn(events, 37, 16, device="cuda", requires_grad=True)
    blockwise_read(x, alpha)[0].sum().backward()
    assert torch.count_nonzero(alpha.grad[1:]) == 0
    with pytest.raises(ValueError, match="contiguous|packed"):
        blockwise_read(x.transpose(1, 2), alpha)
    padded = torch.randn(2, 19, 37, device="cuda")
    prefix_slice = padded[:, 3:]
    assert not prefix_slice.is_contiguous()
    with pytest.raises(ValueError, match="contiguous|packed"):
        blockwise_triangular(prefix_slice, [prefix_slice], torch.ones(1, 37, device="cuda"))


@CUDA
@pytest.mark.parametrize("active", [1, 7, 11])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_triangular_active_prefix_and_multiple_gradient_consumers(active, dtype):
    torch.manual_seed(711)
    base = torch.randn(3, 49, 67, device="cuda", dtype=dtype).requires_grad_()
    updates = [torch.randn_like(base).requires_grad_() for _ in range(active)]
    gamma = (torch.randn(active, 67, device="cuda") * 0.05).requires_grad_()
    inputs = (base, *updates, gamma)
    actual = blockwise_triangular(base, updates, gamma)
    expected = (base.float() + sum(z.float() * g for z, g in zip(updates, gamma))).to(dtype)
    extra = sum(z.float().square().mean() for z in updates)
    ga = torch.autograd.grad(actual.float().square().mean() + extra, inputs, retain_graph=True)
    gr = torch.autograd.grad(expected.float().square().mean() + extra, inputs)
    tolerance = dict(atol=2e-5, rtol=3e-4) if dtype == torch.float32 else dict(atol=0.016, rtol=0.03)
    torch.testing.assert_close(actual, expected, **tolerance)
    for actual_grad, reference_grad in zip(ga, gr):
        torch.testing.assert_close(actual_grad, reference_grad,
                                   atol=1e-7 if dtype == torch.float32 else 2e-5,
                                   rtol=3e-4 if dtype == torch.float32 else 0.035)


def literal_dense_forward(model, x, t, labels):
    """One read and write per *whole block*, independent of fused schedules."""
    label_embedding = model.y_embedder(labels)
    condition = model.t_embedder(t) + label_embedding
    carrier = cells(model.x_embedder(x), (model.high_grid // 16) ** 2)
    context = None
    if isinstance(model, SiHCInContextModel):
        context = (label_embedding[:, None] + model.in_context_posemb).to(carrier.dtype)
    for index, block in enumerate(model.blocks):
        alpha = block.connection.alpha(carrier.dtype)
        z = (carrier.float() * alpha.T.float()).sum(2).to(carrier.dtype)
        if context is not None and index >= model.in_context_start:
            sequence = torch.cat((context, z), dim=1)
            delta = block.branch_update(sequence, condition, model.workspace_rope_incontext)
            context = context + delta[:, :model.in_context_len]
            dz = delta[:, model.in_context_len:]
        else:
            dz = block.branch_update(z, condition, model.workspace_rope)
        beta = block.connection.beta(carrier.dtype)
        carrier = (carrier.float() + dz.float().unsqueeze(2) * beta.T.float()).to(carrier.dtype)
    return model._dense_output(high(carrier), condition)


def compare_model_gradients(actual, expected, dtype, *, dense=False):
    tolerance = (dict(atol=3e-5, rtol=0.005) if dtype == torch.float32
                 else dict(atol=5e-4, rtol=0.15 if dense else 0.08))
    error, norm = 0.0, 0.0
    for (name, p), (other_name, q) in zip(actual.named_parameters(), expected.named_parameters(), strict=True):
        assert name.replace("_checkpoint_wrapped_module.", "") == other_name.replace("_checkpoint_wrapped_module.", ""), name
        assert p.grad is not None and q.grad is not None, name
        assert torch.isfinite(p.grad).all() and torch.isfinite(q.grad).all(), name
        torch.testing.assert_close(p.grad, q.grad, msg=name, **tolerance)
        error += (p.grad.float() - q.grad.float()).square().sum().item()
        norm += q.grad.float().square().sum().item()
    assert (error / max(norm, 1e-20)) ** 0.5 < (0.06 if dtype == torch.bfloat16 else 2e-4)


@CUDA
@pytest.mark.parametrize("case", ["p4", "p8", "mixed", "ctx"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_full_model_dense_blockwise_oracle_and_legacy(case, dtype):
    torch.manual_seed(522)
    model = nonzero_weights(model_for(case, "tuple")).cuda()
    legacy, dense = copy.deepcopy(model), copy.deepcopy(model)
    legacy.kernel_backend = "legacy"
    # Batch > 1 makes spatial slices of CTX-prefixed output non-packed.
    x = torch.randn(2, 3, 256, 256, device="cuda")
    t = torch.rand(2, device="cuda")
    labels = torch.tensor([1, 2], device="cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dtype == torch.bfloat16):
        actual = model(x, t, labels)
        old = legacy(x, t, labels)
        expected = literal_dense_forward(dense, x, t, labels)
    for output in (actual, old, expected):
        output.float().square().mean().backward()
    tolerance = dict(atol=3e-5, rtol=3e-4) if dtype == torch.float32 else dict(atol=0.04, rtol=0.05)
    torch.testing.assert_close(actual, old, **tolerance)
    torch.testing.assert_close(actual, expected, **tolerance)
    compare_model_gradients(model, legacy, dtype)
    compare_model_gradients(model, dense, dtype, dense=True)


@CUDA
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_fullgraph_compile_forward_backward_with_context(dtype):
    torch.manual_seed(103)
    model = nonzero_weights(SiHCInContextModel(
        hidden_size=32, num_heads=4, depth=8, fused_stage_size=8,
        in_context_start=4, in_context_len=3, attn_backend="math", kernel_backend="tuple")).cuda()
    eager = copy.deepcopy(model)
    compiled = torch.compile(model, mode="default", fullgraph=True)
    x = torch.randn(2, 3, 256, 256, device="cuda")
    t, labels = torch.rand(2, device="cuda"), torch.tensor([1, 2], device="cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dtype == torch.bfloat16):
        expected, actual = eager(x, t, labels), compiled(x, t, labels)
    expected.float().square().mean().backward()
    actual.float().square().mean().backward()
    torch.testing.assert_close(actual, expected, atol=3e-5 if dtype == torch.float32 else 0.03,
                               rtol=3e-4 if dtype == torch.float32 else 0.03)
    compare_model_gradients(model, eager, dtype)


@CUDA
@pytest.mark.parametrize("recompute", ["none", "mlp"])
def test_optimizer_resume_and_optional_mlp_recompute(tmp_path, recompute):
    from sihc.activation_checkpointing import configure_activation_checkpointing

    torch.manual_seed(230)
    model = nonzero_weights(model_for("p4", "tuple")).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, betas=(0.9, 0.95))
    x = torch.randn(2, 3, 256, 256, device="cuda")
    t, labels = torch.rand(2, device="cuda"), torch.tensor([1, 2], device="cuda")

    def step(module, opt):
        opt.zero_grad(set_to_none=True)
        output = module(x, t, labels)
        output.square().mean().backward()
        opt.step()
        return output

    step(model, optimizer)
    path = tmp_path / "resume.pt"
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict()}, path)
    restored = model_for("p4", "tuple").cuda()
    loaded = load_checkpoint(path)
    restored.load_state_dict(loaded["model"], strict=True)
    new_optimizer = torch.optim.AdamW(restored.parameters(), lr=2e-4, betas=(0.9, 0.95))
    new_optimizer.load_state_dict(loaded["optimizer"])
    configure_activation_checkpointing(restored, recompute)
    expected, actual = step(model, optimizer), step(restored, new_optimizer)
    torch.testing.assert_close(actual, expected, atol=3e-5, rtol=3e-4)
    compare_model_gradients(restored, model, torch.float32)
    for p, q in zip(model.parameters(), restored.parameters(), strict=True):
        torch.testing.assert_close(p, q, atol=2e-6, rtol=3e-4)
