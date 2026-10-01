"""Literal dense-carrier oracles; nonzero gates prevent vacuous zero-head tests."""

import copy
import os

import pytest
import torch

from sihc.models.sihc.sublayer import SublayerSiHCModel
from sihc.models.sihc.sublayer_kernels import (
    sublayer_read, sublayer_triangular, sublayer_write,
)

pytestmark = pytest.mark.skipif(
    os.environ.get("SIHC_RUN_CUDA_TESTS") != "1" or not torch.cuda.is_available(),
    reason="set SIHC_RUN_CUDA_TESTS=1 on an idle GPU",
)


def cells(x, slots):
    b, h, _, c = x.shape
    k = int(slots ** 0.5)
    g = h // k
    return x.reshape(b, g, k, g, k, c).permute(0, 1, 3, 2, 4, 5).reshape(b, g * g, slots, c)


def high(x):
    b, n, s, c = x.shape
    g, k = int(n ** 0.5), int(s ** 0.5)
    return x.reshape(b, g, g, k, k, c).permute(0, 1, 3, 2, 4, 5).reshape(b, g * k, g * k, c)


def assert_numerics(a, b, dtype):
    # FP32 accumulation followed by boundary casting is the kernel contract.
    atol, rtol = (2e-4, 2e-4) if dtype == torch.float32 else (0.025, 0.03)
    torch.testing.assert_close(a, b, atol=atol, rtol=rtol)


@pytest.mark.parametrize("patch", [4, 8])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_gemm_patch_stem_against_conv_forward_backward(patch, dtype):
    from sihc.models.sihc.components import DirectPatchEmbedNHWC
    from sihc.models.sihc.sublayer import _DirectPatchEmbedGEMM

    torch.manual_seed(233)
    ref = DirectPatchEmbedNHWC(img_size=32, patch_size=patch, embed_dim=32).cuda()
    stem = _DirectPatchEmbedGEMM(img_size=32, patch_size=patch, embed_dim=32).cuda()
    old_opt = torch.optim.AdamW(ref.parameters(), lr=2e-4, fused=True)
    new_opt = torch.optim.AdamW(stem.parameters(), lr=2e-4, fused=True)
    warm = torch.randn(2, 3, 32, 32, device="cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dtype == torch.bfloat16):
        warm_out = ref(warm)
    warm_out.float().square().mean().backward()
    old_opt.step()
    old_opt.zero_grad(set_to_none=True)
    stem.load_state_dict(ref.state_dict(), strict=True)
    new_opt.load_state_dict(copy.deepcopy(old_opt.state_dict()))
    x = torch.randn(2, 3, 32, 32, device="cuda", requires_grad=True)
    xr = x.detach().clone().requires_grad_()
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dtype == torch.bfloat16):
        actual, expected = stem(x), ref(xr)
    actual.float().square().mean().backward()
    expected.float().square().mean().backward()
    assert_numerics(actual, expected, dtype)
    for a, b in [(x.grad, xr.grad), *[(a.grad, b.grad) for a, b in zip(stem.parameters(), ref.parameters())]]:
        torch.testing.assert_close(a, b, atol=1e-6 if dtype == torch.float32 else 2e-4,
                                   rtol=1e-4 if dtype == torch.float32 else 0.03)
    old_opt.step()
    new_opt.step()
    for a, b in zip(stem.parameters(), ref.parameters()):
        assert a.stride() == b.stride()
        torch.testing.assert_close(a, b, atol=2e-6 if dtype == torch.float32 else 2e-5,
                                   rtol=2e-5 if dtype == torch.float32 else 0.001)


@pytest.mark.parametrize("events", [16, 24])
@pytest.mark.parametrize("slots", [4, 16])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_dense_recurrence_and_all_gradients(events, slots, dtype):
    torch.manual_seed(92)
    k = int(slots ** 0.5)
    x = torch.randn(2, 4 * k, 4 * k, 37, device="cuda", dtype=dtype).requires_grad_()
    alpha = (torch.randn(events, 37, slots, device="cuda", dtype=dtype) * 0.07).requires_grad_()
    beta = (torch.randn_like(alpha) * 0.07).requires_grad_()
    xr, ar, br = [t.detach().clone().requires_grad_() for t in (x, alpha, beta)]
    base = sublayer_read(x, alpha)
    gamma = (alpha[:, None] * beta[None, :]).sum(-1)
    updates = []
    # Dense recurrence uses fp32 routing weights/products before each boundary
    # cast. BF16 triangular reassociation has expected rounding differences.
    carrier = cells(xr, slots)
    for i in range(events):
        z = base[i] if i == 0 else sublayer_triangular(base[i], updates, gamma[i, :i].contiguous())
        updates.append((0.05 * z.tanh()).contiguous())
        zr = (carrier.float() * ar[i].T.float()[None, None]).sum(2).to(dtype)
        dzr = 0.05 * zr.tanh()
        carrier = (carrier.float() + dzr.float().unsqueeze(2) * br[i].T.float()).to(dtype)
    out = sublayer_write(x, updates, beta)
    ref = high(carrier)
    upstream = torch.randn_like(out) * 0.1
    ga = torch.autograd.grad(out, (x, alpha, beta), upstream)
    gr = torch.autograd.grad(ref, (xr, ar, br), upstream)
    # Dense BF16 rounds the carrier at every event; compare its accumulated
    # error with relative L2 in addition to a bounded elementwise tolerance.
    assert_numerics(out, ref, dtype)
    for a, b in zip(ga, gr):
        assert_numerics(a, b, dtype)
        assert (a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-8) < (0.035 if dtype == torch.bfloat16 else 1e-4)


def test_packed_layout_contract_and_unused_reads():
    x = torch.randn(1, 16, 16, 37, device="cuda", requires_grad=True)
    a = torch.randn(16, 37, 16, device="cuda", requires_grad=True)
    with pytest.raises(ValueError, match="contiguous|packed"):
        sublayer_read(x.transpose(1, 2), a)
    base = sublayer_read(x, a)
    # Context-prefix slicing must be packed, even when the shape is correct.
    padded = torch.randn(2, 18, 37, device="cuda")
    bad = padded[:, 2:]
    with pytest.raises(ValueError, match="contiguous|packed"):
        sublayer_triangular(bad, [bad], torch.ones(1, 37, device="cuda"))
    base[0].sum().backward()
    assert torch.count_nonzero(a.grad[1:]) == 0


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("slots", [4, 16])
def test_read_adjoint_with_token_and_channel_tile_tails(dtype, slots):
    # M=147 crosses the 32-token reduction boundary; C=67 crosses the
    # 64-channel boundary. Exercise both transposes and slot-fast indexing.
    torch.manual_seed(317)
    k = int(slots ** 0.5)
    x = torch.randn(3, 7 * k, 7 * k, 67, device="cuda", dtype=dtype).requires_grad_()
    alpha = (torch.randn(24, 67, slots, device="cuda", dtype=dtype) * 0.05).requires_grad_()
    xr, ar = (v.detach().clone().requires_grad_() for v in (x, alpha))
    actual = sublayer_read(x, alpha)
    reference = torch.einsum("bnsc,lcs->lbnc", cells(xr, slots).float(), ar.float()).to(dtype)
    upstream = torch.randn_like(reference) * 0.1
    ga = torch.autograd.grad(sum((v * u).sum() for v, u in zip(actual, upstream)), (x, alpha))
    gr = torch.autograd.grad(reference, (xr, ar), upstream)
    assert_numerics(torch.stack(actual), reference, dtype)
    for a, b in zip(ga, gr):
        assert_numerics(a, b, dtype)


@pytest.mark.parametrize("events", [16, 24])
def test_weight_mma_fp32_accumulator_and_chunk_tails(events):
    from sihc.models.sihc.sublayer_kernels import _weight_grad_bf16_mma

    # M=363 covers a full 256-token reduction CTA plus a partial one. C=67
    # exercises masked channels; verify FP32 accumulators before BF16 casting.
    torch.manual_seed(671)
    x = torch.randn(3, 44, 44, 67, device="cuda", dtype=torch.bfloat16)
    updates = [torch.randn(3, 121, 67, device="cuda", dtype=torch.bfloat16) * 0.1
               for _ in range(events)]
    result = torch.zeros(events, 16, 67, device="cuda", dtype=torch.float32)
    _weight_grad_bf16_mma[(2, 2)](
        x, tuple(updates), result, 363, 67, 11, 4, num_warps=8, num_stages=1)
    expected = (cells(x, 16).float().unsqueeze(0)
                * torch.stack(updates).float().unsqueeze(3)).sum((1, 2))
    torch.testing.assert_close(result, expected, atol=2e-4, rtol=2e-4)


@pytest.mark.parametrize("events", [16, 24])
def test_write_mma_forward_and_adjoint_with_tails(events):
    torch.manual_seed(291)
    x = torch.randn(3, 44, 44, 67, device="cuda", dtype=torch.bfloat16).requires_grad_()
    updates = [torch.randn(3, 121, 67, device="cuda", dtype=x.dtype).requires_grad_()
               for _ in range(events)]
    beta = (torch.randn(events, 67, 16, device="cuda", dtype=x.dtype) * 0.05).requires_grad_()
    xr, *rest = [v.detach().clone().requires_grad_() for v in (x, *updates, beta)]
    zr, br = rest[:-1], rest[-1]
    actual = sublayer_write(x, updates, beta)
    delta = (torch.stack(zr).float().unsqueeze(3)
             * br.transpose(1, 2).float()[:, None, None]).sum(0)
    expected = high((cells(xr, 16).float() + delta).to(x.dtype))
    upstream = torch.randn_like(x) * 0.03
    ga = torch.autograd.grad(actual, (x, *updates, beta), upstream)
    gr = torch.autograd.grad(expected, (xr, *zr, br), upstream)
    assert_numerics(actual, expected, x.dtype)
    for a, b in zip(ga, gr):
        torch.testing.assert_close(a, b, atol=0.002, rtol=0.015)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("active", [3, 23])
def test_triangular_compiled_gradient_accumulation_and_tails(dtype, active):
    torch.manual_seed(583)
    base = torch.randn(3, 49, 67, device="cuda", dtype=dtype).requires_grad_()
    updates = [torch.randn_like(base).requires_grad_() for _ in range(active)]
    # Mixed FP32 gamma/BF16 workspace is an explicitly supported contract.
    gamma = (torch.randn(active, 67, device="cuda") * 0.05).requires_grad_()
    inputs = (base, *updates, gamma)

    def fn(base, updates, gamma):
        out = sublayer_triangular(base, updates, gamma)
        # Each update receives both the triangular and a separate gradient.
        return out.float().square().mean() + sum(u.float().square().mean() for u in updates)

    loss = fn(base, updates, gamma)
    eager = torch.autograd.grad(loss, inputs)
    ref = (base.float() + sum(u.float() * g for u, g in zip(updates, gamma))).to(dtype)
    ref_loss = ref.float().square().mean() + sum(u.float().square().mean() for u in updates)
    expected = torch.autograd.grad(ref_loss, inputs)
    compiled = torch.compile(fn, mode="default", fullgraph=True)
    actual = compiled(base, updates, gamma)
    compiled_grads = torch.autograd.grad(actual, inputs)
    assert_numerics(actual, loss, dtype)
    for a, b, c in zip(eager, expected, compiled_grads):
        torch.testing.assert_close(a, b, atol=1e-7 if dtype == torch.float32 else 2e-5,
                                   rtol=3e-4 if dtype == torch.float32 else 0.035)
        torch.testing.assert_close(c, a, atol=1e-7 if dtype == torch.float32 else 2e-5,
                                   rtol=3e-4 if dtype == torch.float32 else 0.035)


def literal_forward(model, x, t, y):
    cond = model.t_embedder(t) + model.y_embedder(y)
    carrier = cells(model.x_embedder(x), (model.high_grid // 16) ** 2)
    for block in model.blocks:
        modulation = block.branch.conditioning(cond)
        for connection, kind in ((block.attention_connection, "attn"), (block.mlp_connection, "mlp")):
            z = (carrier.float() * connection.alpha(carrier.dtype).T.float()).sum(2).to(carrier.dtype)
            dz = (block.branch.attention_update(z, modulation, model.workspace_rope)
                  if kind == "attn" else block.branch.mlp_update(z, modulation))
            carrier = (carrier.float() + dz.float().unsqueeze(2) * connection.beta(carrier.dtype).T.float()).to(carrier.dtype)
    return model._dense_output(high(carrier), cond)


def small_model(depth, patch):
    model = SublayerSiHCModel(hidden_size=32, num_heads=4, depth=depth,
                             fused_stage_size=depth,
                             stage_sizes=(8, 12) if depth == 20 else None,
                             patch_size=patch, attn_backend="math").cuda()
    with torch.no_grad():
        for block in model.blocks:
            block.branch.adaLN_modulation[-1].weight.normal_(std=0.01)
            block.branch.adaLN_modulation[-1].bias.normal_(std=0.03)
            for connection in (block.attention_connection, block.mlp_connection):
                connection.read_weight.add_(torch.randn_like(connection.read_weight) * 0.01)
                connection.write_weight.add_(torch.randn_like(connection.write_weight) * 0.02)
        model.final_layer.linear.weight.normal_(std=0.03)
    return model


@pytest.mark.parametrize("depth,patch", [(8, 4), (12, 8), (20, 4)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_full_model_against_literal_carrier(depth, patch, dtype):
    torch.manual_seed(19)
    model = small_model(depth, patch)
    ref = copy.deepcopy(model)
    x = torch.randn(1, 3, 256, 256, device="cuda")
    t, y = torch.rand(1, device="cuda"), torch.zeros(1, device="cuda", dtype=torch.long)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dtype == torch.bfloat16):
        actual = model(x, t, y)
        expected = literal_forward(ref, x, t, y)
    actual.float().square().mean().backward()
    expected.float().square().mean().backward()
    assert_numerics(actual, expected, dtype)
    for (name, a), (_, b) in zip(model.named_parameters(), ref.named_parameters()):
        assert a.grad is not None and b.grad is not None, name
        assert torch.isfinite(a.grad).all(), name
        torch.testing.assert_close(a.grad, b.grad, atol=2e-5 if dtype == torch.float32 else 5e-4,
                                   rtol=0.003 if dtype == torch.float32 else 0.15, msg=name)


@pytest.mark.parametrize("depth", [8, 12])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_fullgraph_compile_forward_backward(depth, dtype):
    torch.manual_seed(42)
    model = small_model(depth, 4 if dtype == torch.bfloat16 else 8)
    x = torch.randn(1, 3, 256, 256, device="cuda")
    t, y = torch.rand(1, device="cuda"), torch.zeros(1, device="cuda", dtype=torch.long)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dtype == torch.bfloat16):
        eager = model(x, t, y)
    eager.float().square().mean().backward()
    grads = {name: p.grad.clone() for name, p in model.named_parameters()}
    model.zero_grad(set_to_none=True)
    compiled = torch.compile(model, mode="default", fullgraph=True)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dtype == torch.bfloat16):
        result = compiled(x, t, y)
    result.float().square().mean().backward()
    torch.testing.assert_close(result, eager, atol=3e-5 if dtype == torch.float32 else 0.025,
                               rtol=3e-4 if dtype == torch.float32 else 0.03)
    for name, p in model.named_parameters():
        torch.testing.assert_close(p.grad, grads[name], atol=3e-5 if dtype == torch.float32 else 5e-4,
                                   rtol=0.003 if dtype == torch.float32 else 0.15, msg=name)


def test_checkpoint_forward_and_mlp_recompute(tmp_path):
    from sihc.activation_checkpointing import configure_activation_checkpointing
    from sihc.checkpoint import build_model_from_checkpoint, load_checkpoint, load_model_state

    torch.manual_seed(128)
    model = small_model(8, 8)
    x = torch.randn(1, 3, 256, 256, device="cuda")
    t, y = torch.rand(1, device="cuda"), torch.zeros(1, device="cuda", dtype=torch.long)
    expected = model(x, t, y)
    expected.square().mean().backward()
    ckpt = {
        "model": model.state_dict(),
        "ema": dict(model.named_parameters()),
        "ema2": dict(model.named_parameters()),
        "meta": {"model": "sihc_sublayer_1x12_d768_b8", "model_kwargs": {
            "hidden_size": 32, "num_heads": 4, "depth": 8, "fused_stage_size": 8}},
    }
    path = tmp_path / "sublayer.pt"
    torch.save(ckpt, path)
    ckpt = load_checkpoint(path)
    for key in ("model", "ema", "ema2"):
        restored = build_model_from_checkpoint(ckpt, attn_backend="math").cuda()
        load_model_state(restored, ckpt, key)
        with torch.no_grad():
            torch.testing.assert_close(restored(x, t, y), expected, atol=0, rtol=0)
    configure_activation_checkpointing(restored, "mlp")
    actual = restored(x, t, y)
    actual.square().mean().backward()
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    for (name, p), (_, q) in zip(model.named_parameters(), restored.named_parameters()):
        torch.testing.assert_close(q.grad, p.grad, atol=3e-5, rtol=0.003, msg=name)
