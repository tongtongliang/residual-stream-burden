import copy
import os

import pytest
import torch

from sihc.models import build_model
from sihc.models.sihc.reference import _ReferenceScheduleMixin
from sihc.checkpoint import build_model_from_checkpoint, load_model_state

NAME = 'sihc_sublayer_1x12_d768_b4_ctx32s4'
CUDA = pytest.mark.skipif(
    os.environ.get('SIHC_RUN_CUDA_TESTS') != '1' or not torch.cuda.is_available(),
    reason='requires an allocated CUDA GPU',
)


def make_model(depth=8):
    return build_model(NAME, hidden_size=32, num_heads=4, depth=depth,
                       stage_sizes=(8, 12) if depth == 20 else (depth,),
                       attn_backend='math')


def test_context_registry_and_initialization():
    model = make_model()
    assert model.in_context_len == 32 and model.in_context_start == 4
    assert model.connection_frequency == 'sublayer'
    assert model.patch_embed_type == 'direct'
    for block in model.blocks:
        a, m = block.attention_connection, block.mlp_connection
        assert a.read_weight.data_ptr() != m.read_weight.data_ptr()
        assert a.write_weight.data_ptr() != m.write_weight.data_ptr()
        for c in (a, m):
            assert c.read_weight.shape == (32, 16)
            assert torch.all(c.read_weight == 1/16)
            assert torch.all(c.write_weight == 1)
        assert torch.count_nonzero(block.branch.adaLN_modulation[-1].weight) == 0


def literal(model, x, t, y):
    yemb = model.y_embedder(y)
    cond = model.t_embedder(t) + yemb
    x0 = model.x_embedder(x)
    carrier = _ReferenceScheduleMixin._high_to_cells(model, x0)
    ctx = (yemb[:, None].expand(-1, model.in_context_len, -1)
           + model.in_context_posemb).to(x0.dtype)
    for i, block in enumerate(model.blocks):
        mod = block.branch.conditioning(cond)
        for part, connection in enumerate((block.attention_connection, block.mlp_connection)):
            z = (carrier.float() * connection.alpha(carrier.dtype).T.float()).sum(2).to(carrier.dtype)
            active = i >= model.in_context_start
            seq = torch.cat((ctx, z), 1) if active else z
            if part == 0:
                rope = model.workspace_rope_incontext if active else model.workspace_rope
                ds = block.branch.attention_update(seq, mod, rope)
            else:
                ds = block.branch.mlp_update(seq, mod)
            if active:
                ctx = ctx + ds[:, :model.in_context_len]
                dz = ds[:, model.in_context_len:]
            else:
                dz = ds
            carrier = (carrier.float() + dz.float()[:, :, None]
                       * connection.beta(carrier.dtype).T.float()).to(carrier.dtype)
    return model._dense_output(_ReferenceScheduleMixin._cells_to_high(model, carrier), cond)


@CUDA
@pytest.mark.parametrize('depth', [8, 12, 20])
@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
def test_context_literal_forward_backward(depth, dtype):
    torch.manual_seed(913)
    model = make_model(depth).cuda()
    with torch.no_grad():
        for block in model.blocks:
            block.branch.adaLN_modulation[-1].weight.normal_(std=.01)
            block.branch.adaLN_modulation[-1].bias.normal_(std=.03)
            for c in (block.attention_connection, block.mlp_connection):
                c.read_weight.add_(torch.randn_like(c.read_weight)*.01)
                c.write_weight.add_(torch.randn_like(c.write_weight)*.02)
        model.final_layer.linear.weight.normal_(std=.03)
    ref = copy.deepcopy(model)
    x = torch.randn(2, 3, 256, 256, device='cuda')
    t = torch.rand(2, device='cuda')
    y = torch.tensor([12, 1000], device='cuda')
    with torch.autocast('cuda', dtype=dtype, enabled=dtype != torch.float32):
        out, expected = model(x, t, y), literal(ref, x, t, y)
    torch.testing.assert_close(out, expected, atol=3e-5 if dtype == torch.float32 else .015,
                               rtol=.003 if dtype == torch.float32 else .06)
    out.float().square().mean().backward()
    expected.float().square().mean().backward()
    for (name, p), (_, q) in zip(model.named_parameters(), ref.named_parameters()):
        assert p.grad is not None and torch.isfinite(p.grad).all(), name
        torch.testing.assert_close(p.grad, q.grad, atol=3e-5 if dtype == torch.float32 else 5e-4,
                                   rtol=.003 if dtype == torch.float32 else .15, msg=name)
    assert model.in_context_posemb.grad.abs().sum() > 0


@CUDA
def test_context_compile_and_checkpoint():
    torch.manual_seed(15)
    model = make_model().cuda()
    x = torch.randn(2, 3, 256, 256, device='cuda')
    t, y = torch.rand(2, device='cuda'), torch.zeros(2, device='cuda', dtype=torch.long)
    with torch.no_grad():
        model.final_layer.linear.weight.normal_(std=.03)
    compiled = torch.compile(model, mode='default', fullgraph=True)
    with torch.autocast('cuda', dtype=torch.bfloat16):
        actual = compiled(x, t, y)
        expected = model(x, t, y)
    torch.testing.assert_close(actual, expected, atol=.01, rtol=.04)
    actual.float().square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    ckpt = {'model': model.state_dict(), 'ema': dict(model.named_parameters()),
            'ema2': dict(model.named_parameters()), 'meta': {'model': NAME,
            'model_kwargs': {'hidden_size': 32, 'num_heads': 4, 'depth': 8, 'stage_sizes': [8]}}}
    for key in ('model', 'ema', 'ema2'):
        loaded = build_model_from_checkpoint(ckpt, attn_backend='math').cuda()
        load_model_state(loaded, ckpt, key)
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            torch.testing.assert_close(loaded(x,t,y), expected, atol=.01, rtol=.04)
