"""Portable trainer checkpoints preserve actual model overrides across resumes."""
import os
from types import SimpleNamespace

import pytest
import torch

from evaluation.export_checkpoint import export_state
from sihc.activation_checkpointing import configure_activation_checkpointing
from sihc.checkpoint import build_model_from_checkpoint, load_checkpoint, load_model_state
from sihc.models import build_model
from training.train import build_adamw, save_ckpt


@pytest.mark.parametrize('case', ['block_p4', 'block_p8', 'block_ctx_mixed', 'sublayer_ctx'])
@pytest.mark.parametrize('recompute', ['none', 'mlp'])
def test_trainer_save_preserves_custom_architecture_and_next_resume(tmp_path, case, recompute):
    kwargs = dict(hidden_size=16, num_heads=4, mlp_ratio=3.5, attn_drop=0.1,
                  proj_drop=0.2, in_channels=1, num_classes=7, attn_backend='math')
    if case == 'block_p4':
        name = 'sihc_1x12_d768_b4'
        kwargs.update(depth=8, stage_sizes=(8,), kernel_backend='tuple')
    elif case == 'block_p8':
        name = 'sihc_1x12_d768_b8'
        kwargs.update(depth=12, stage_sizes=(12,), kernel_backend='tuple')
    elif case == 'block_ctx_mixed':
        name = 'sihc_5x8_d1024_b4_ctx32s4'
        kwargs.update(depth=20, stage_sizes=(8, 12), in_context_len=3,
                      in_context_start=5, kernel_backend='tuple')
    else:
        name = 'sihc_sublayer_1x12_d768_b4_ctx32s4'
        kwargs.update(depth=8, stage_sizes=(8,), in_context_len=3,
                      in_context_start=2)
    model = build_model(name, **kwargs)
    names = [name for name, _ in model.named_parameters()]
    parameters = list(model.parameters())
    ema = [parameter.detach().clone() for parameter in parameters]
    ema2 = [parameter.detach().clone().mul_(0.5) for parameter in parameters]
    optimizer = build_adamw(parameters, lr=2e-4, weight_decay=0,
                            device=torch.device('cpu'))
    # Populate optimizer moments without calling any CUDA-only routing kernels.
    for parameter in parameters:
        parameter.grad = torch.full_like(parameter, 0.01)
    optimizer.step()
    configure_activation_checkpointing(model, recompute)
    args = SimpleNamespace(model=name, attn_backend='flash', repa=False,
                           prediction='velocity', ema_decay=0.9999, ema_decay2=0.9996)
    first = tmp_path / 'first.pt'
    save_ckpt(first, model, optimizer, names, ema, ema2, 1, args)
    checkpoint = load_checkpoint(first)
    metadata = checkpoint['meta']['model_kwargs']
    assert metadata['attn_backend'] == 'math'  # actual model, not stale CLI input
    assert metadata['mlp_ratio'] == 3.5
    assert metadata['attn_drop'] == 0.1 and metadata['proj_drop'] == 0.2
    assert metadata['in_channels'] == 1 and metadata['num_classes'] == 7
    assert checkpoint['meta']['num_classes'] == 7

    for key in ('model', 'ema', 'ema2'):
        restored = build_model_from_checkpoint(checkpoint)
        load_model_state(restored, checkpoint, key)
        assert restored.hidden_size == 16 and restored.stage_sizes == model.stage_sizes
        assert restored.mlp_ratio == 3.5
        assert restored.blocks[restored.depth // 2].branch.attn.attn_drop.p == 0.1
        assert restored.blocks[restored.depth // 2].branch.attn.proj_drop.p == 0.2
        if hasattr(model, 'in_context_len'):
            assert restored.in_context_len == model.in_context_len
            assert restored.in_context_start == model.in_context_start
        for parameter_name, parameter in restored.named_parameters():
            torch.testing.assert_close(parameter, checkpoint[key][parameter_name], atol=0, rtol=0)

    restored = build_model_from_checkpoint(checkpoint)
    load_model_state(restored, checkpoint, 'model')
    restored_optimizer = build_adamw(restored.parameters(), lr=2e-4, weight_decay=0,
                                     device=torch.device('cpu'))
    restored_optimizer.load_state_dict(checkpoint['optimizer'])
    for original, resumed in zip(parameters, restored.parameters(), strict=True):
        for field in ('exp_avg', 'exp_avg_sq', 'step'):
            torch.testing.assert_close(optimizer.state[original][field],
                                       restored_optimizer.state[resumed][field], atol=0, rtol=0)
    for module, opt in ((model, optimizer), (restored, restored_optimizer)):
        opt.zero_grad(set_to_none=True)
        for parameter in module.parameters():
            parameter.grad = torch.full_like(parameter, 0.02)
        opt.step()
    for original, resumed in zip(parameters, restored.parameters(), strict=True):
        torch.testing.assert_close(original, resumed, atol=0, rtol=0)
    second = tmp_path / 'second.pt'
    save_ckpt(second, restored, restored_optimizer, names, ema, ema2, 2, args)
    second_checkpoint = load_checkpoint(second)
    assert second_checkpoint['meta']['model_kwargs'] == metadata
    exported = export_state(second_checkpoint, 'ema')
    inference = build_model_from_checkpoint(exported)
    load_model_state(inference, exported, 'model')
    assert inference.hidden_size == 16 and inference.stage_sizes == model.stage_sizes
    assert inference.in_channels == 1 and inference.mlp_ratio == 3.5


@pytest.mark.skipif(
    os.environ.get('SIHC_RUN_CUDA_TESTS') != '1' or not torch.cuda.is_available(),
    reason='requires explicitly allocated CUDA execution',
)
@pytest.mark.parametrize('backend', ['legacy', 'tuple'])
def test_heun_cfg_preserves_cudagraph_outputs_with_context(backend):
    """Conditional outputs must survive the following unconditional graph replay."""
    from sihc.models.sihc.model import SiHCInContextModel
    from sihc.sampling import sample_heun

    torch.manual_seed(480)
    model = SiHCInContextModel(
        hidden_size=32, num_heads=4, depth=8, fused_stage_size=8,
        in_context_len=3, in_context_start=4, num_classes=16,
        attn_backend='math', kernel_backend=backend,
    ).cuda().eval()
    with torch.no_grad():
        model.y_embedder.embedding_table.weight.normal_(std=0.3)
        for block in model.blocks:
            block.branch.adaLN_modulation[-1].weight.normal_(std=0.03)
            block.branch.adaLN_modulation[-1].bias.normal_(std=0.03)
        model.final_layer.adaLN_modulation[-1].weight.normal_(std=0.05)
        model.final_layer.linear.weight.normal_(std=0.05)
    compiled = torch.compile(model, mode='reduce-overhead', fullgraph=True)
    labels = torch.arange(8, device='cuda', dtype=torch.long)
    sampling = dict(steps=2, cfg_scale=2.9, interval=(0.1, 0.9),
                    num_classes=model.num_classes, prediction='velocity')

    def unconditional(z, t, ignored_labels):
        return model(z, t, torch.full_like(ignored_labels, model.num_classes))

    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        # Repeated complete trajectories reach graph capture/replay, beyond
        # the first invocation's compiler warmup.
        for seed in (481, 482):
            torch.manual_seed(seed)
            sample_heun(compiled, labels, **sampling)
        torch.manual_seed(483)
        expected = sample_heun(model, labels, **sampling)
        torch.manual_seed(483)
        actual = sample_heun(compiled, labels, **sampling)
        torch.manual_seed(483)
        overwritten_reference = sample_heun(unconditional, labels, **sampling)

    assert actual.shape == (8, 3, 256, 256)
    assert torch.isfinite(actual).all() and torch.isfinite(expected).all()
    torch.testing.assert_close(actual, expected, atol=0.02, rtol=0.02)
    # If conditional graph outputs alias later unconditional outputs, CFG
    # collapses towards unconditional sampling. Ensure this fixture detects
    # that failure instead of passing because zero gates erased conditioning.
    conditioning_signal = (expected.float() - overwritten_reference.float()).square().mean().sqrt()
    compiled_error = (actual.float() - expected.float()).square().mean().sqrt()
    assert conditioning_signal.item() > 1e-3
    assert compiled_error.item() < 0.25 * conditioning_signal.item()


@pytest.mark.parametrize(
    'checkpoint_backend, requested, expected',
    [(None, None, 'flash'), ('math', None, 'math'),
     ('math', 'flash', 'flash'), ('flash', 'math', 'math')],
)
def test_training_attention_backend_override_precedence(
    monkeypatch, checkpoint_backend, requested, expected,
):
    from training.train import model_kwargs_for_training, parse_args

    argv = ['train', '--data_path', 'unused-dataset', '--model', 'sihc_1x12_d768_b4']
    if requested is not None:
        argv += ['--attn_backend', requested]
    monkeypatch.setattr('sys.argv', argv)
    args = parse_args()
    checkpoint = None if checkpoint_backend is None else {
        'model': {}, 'meta': {'model_kwargs': {'attn_backend': checkpoint_backend}},
    }
    kwargs = model_kwargs_for_training(args, checkpoint)
    assert kwargs['attn_backend'] == expected
    with torch.device('meta'):
        model = build_model(args.model, **kwargs)
    assert model.blocks[0].branch.attn.attn_backend == expected


def test_attention_config_choice_is_overridden_by_explicit_cli(tmp_path, monkeypatch):
    import json
    from training.train import model_kwargs_for_training, parse_args

    config = tmp_path / 'config.json'
    config.write_text(json.dumps({'attn_backend': 'math'}))
    checkpoint = {'model': {}, 'meta': {'model_kwargs': {'attn_backend': 'efficient'}}}
    for cli, expected in (([], 'math'), (['--attn_backend', 'flash'], 'flash')):
        monkeypatch.setattr('sys.argv', [
            'train', '--data_path', 'unused-dataset', '--config', str(config),
        ] + cli)
        args = parse_args()
        assert model_kwargs_for_training(args, checkpoint)['attn_backend'] == expected


def test_trainer_saves_actual_attention_backend_with_branch_recompute(tmp_path):
    name = 'sihc_1x12_d768_b4'
    model = build_model(name, hidden_size=16, num_heads=4, depth=8, stage_sizes=(8,),
                        attn_backend='math', kernel_backend='tuple')
    names = [name for name, _ in model.named_parameters()]
    parameters = list(model.parameters())
    ema = [parameter.detach().clone() for parameter in parameters]
    optimizer = build_adamw(parameters, lr=2e-4, weight_decay=0,
                            device=torch.device('cpu'))
    configure_activation_checkpointing(model, 'branch')
    args = SimpleNamespace(model=name, attn_backend='flash', repa=False,
                           prediction='velocity', ema_decay=0.9999, ema_decay2=0.0)
    path = tmp_path / 'branch.pt'
    save_ckpt(path, model, optimizer, names, ema, None, 1, args)
    checkpoint = load_checkpoint(path)
    assert checkpoint['meta']['model_kwargs']['attn_backend'] == 'math'
    restored = build_model_from_checkpoint(checkpoint)
    for key in ('model', 'ema'):
        load_model_state(restored, checkpoint, key)
    assert restored.blocks[0].branch.attn.attn_backend == 'math'
    assert restored.hidden_size == 16 and restored.depth == 8
