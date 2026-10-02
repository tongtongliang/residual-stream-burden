"""Public B/L/H/XL architecture and explicit REPA recipe contracts."""
import json
from pathlib import Path
import sys

import pytest
import torch

from sihc import PRESET_NAMES, build_preset, get_preset
from sihc.checkpoint import build_model_from_checkpoint, validate_checkpoint_model
from training.train import model_kwargs_for_training, parse_args
from sihc.models import build_model
from sihc.config import validate_model_architecture

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize('size,count', [('B',131000112), ('L',459532304),
                                       ('H',953750640), ('XL',771234128)])
def test_preset_config_and_checkpoint_agree(size, count, monkeypatch):
    spec = get_preset(size)
    config = json.loads((ROOT/spec['config']).read_text())
    monkeypatch.setattr(sys, 'argv', ['train', '--config', str(ROOT/spec['config']),
                                      '--data_path', 'test-data'])
    args = parse_args()
    assert args.repa == spec['repa']
    assert args.activation_checkpoint == 'none' and args.grad_clip == 0
    if args.repa:
        assert args.repa_encoder == spec['teacher'] == 'dinov3_vitl16'
        assert args.repa_z_dim == 1024 and args.repa_coeff == .5
    with torch.device('meta'):
        model = build_preset(size)
        trained = build_model(args.model, **model_kwargs_for_training(args))
        state = {'model': model.state_dict(), 'args': vars(args),
                 'meta': {'model_kwargs': spec['model_kwargs']}}
        restored = build_model_from_checkpoint(state)
    assert sum(p.numel() for p in model.parameters()) == count
    assert model.depth == spec['depth'] and model.hidden_size == spec['hidden_size']
    assert model.stage_sizes == spec['stage_sizes']
    assert model.in_context_start == spec['in_context_start'] and model.in_context_len == 32
    assert bool(model.repa_projectors) == spec['repa']
    for candidate in (trained, restored):
        validate_model_architecture(candidate, args.config_architecture)
        validate_checkpoint_model(candidate, state, 'model')
    assert config['max_steps'] == 750600


def test_preset_lookup_returns_independent_metadata():
    assert PRESET_NAMES == ('B','L','H','XL')
    spec = get_preset('xl'); spec['model_kwargs']['repa_depth'] = 1
    assert get_preset('XL')['model_kwargs']['repa_depth'] == 8
    assert get_preset('XL')['recommended_checkpoint'] == dict(
        epoch=520, step=650520, filename='step_00650520.pt', state_key='ema')
    with pytest.raises(ValueError): get_preset('unknown')


def test_repa_teacher_guard_and_prefix_removal():
    from types import SimpleNamespace
    from training.train import validate_repa_teacher_config, dinov3_patch_tokens
    config = SimpleNamespace(model_type='dinov3_vit', hidden_size=1024,
                             patch_size=16, num_register_tokens=4, num_channels=3)
    validate_repa_teacher_config(config, 'dinov3_vitl16', 1024)
    with pytest.raises(ValueError, match='repa_z_dim'):
        validate_repa_teacher_config(config, 'dinov3_vitl16', 768)
    with pytest.raises(ValueError, match='hidden_size'):
        validate_repa_teacher_config(config, 'dinov3_vitb16', 768)
    config.num_register_tokens = 0
    with pytest.raises(ValueError, match='num_register_tokens'):
        validate_repa_teacher_config(config, 'dinov3_vitl16', 1024)
    # Prefix entries must not leak into the spatial teacher targets.
    tokens = torch.arange(261).view(1,261,1).expand(2,261,1024)
    patches = dinov3_patch_tokens(SimpleNamespace(last_hidden_state=tokens))
    assert patches.shape == (2,256,1024)
    assert torch.equal(patches, tokens[:,5:])
