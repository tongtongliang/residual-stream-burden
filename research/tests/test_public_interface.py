"""CPU contracts for the public package; no dataset, service or GPU required."""
import json
import subprocess
import sys
from pathlib import Path
from types import MethodType

import pytest
import torch

from sihc.models import build_model
from sihc.models.sihc.model import SiHCInContextModel
from sihc.checkpoint import build_model_from_checkpoint, load_model_state
from evaluation.export_checkpoint import export_state

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize('state_key', ['model', 'ema', 'ema2'])
def test_export_is_allowlisted_and_loadable(state_key):
    name = 'sihc_sublayer_1x12_d768_b4_ctx32s4'
    kwargs = dict(hidden_size=16, num_heads=4, depth=8, stage_sizes=(8,), attn_backend='math')
    model = build_model(name, **kwargs)
    source = dict(model=model.state_dict(), ema=model.state_dict(), ema2=model.state_dict(),
                  args=dict(model=name, prediction='velocity', run_name='not-for-publication'),
                  optimizer={'private_state': 123}, step=100,
                  meta={'model_kwargs': kwargs, 'owner': 'not-for-publication'})
    exported = export_state(source, state_key)
    assert set(exported) == {'format_version', 'args', 'meta', 'model', 'ema'}
    assert 'not-for-publication' not in str(exported['meta']) + str(exported['args'])
    restored = build_model_from_checkpoint(exported, attn_backend='math')
    for key in ('model', 'ema'):
        load_model_state(restored, exported, key)
        for p, q in zip(model.parameters(), restored.parameters()):
            torch.testing.assert_close(p, q, atol=0, rtol=0)


def test_blockwise_context_slices_are_packed(monkeypatch):
    import sihc.models.sihc.model as module
    model = SiHCInContextModel(hidden_size=16, num_heads=4, depth=8, fused_stage_size=8,
                               in_context_start=4, attn_backend='math')
    x = torch.randn(2, 64, 64, 16)
    monkeypatch.setattr(module, 'triton_all_base_b4_8_split_traceable',
                        lambda x, a: torch.zeros(8, 2, 256, 16))
    def active(z, *updates_and_gamma):
        assert all(t.is_contiguous() for t in updates_and_gamma)
        return z
    monkeypatch.setattr(module, 'triton_triangular_active_ops_8', [active] * 7)
    for block in model.blocks:
        block.branch_update = MethodType(lambda self, z, cond, rope: z + 1, block)
    def final(self, x0, updates, blocks):
        assert len(updates) == 8 and all(u.is_contiguous() for u in updates)
        return x0
    model._final_accumulate = MethodType(final, model)
    _, context = model._run_fused_schedule_stage_incontext(
        x, torch.zeros(2,16), torch.zeros(2,32,16), list(model.blocks), 0)
    assert context.shape == (2, 32, 16)


@pytest.mark.parametrize('module', ['training.train', 'evaluation.sample', 'evaluation.evaluate',
                                  'evaluation.export_checkpoint', 'evaluation.verify_checkpoint',
                                  'data.prepare_cache'])
def test_cli_help(module):
    subprocess.run([sys.executable, '-m', module, '--help'], cwd=ROOT,
                   capture_output=True, text=True, check=True)


def test_public_config_defaults():
    for path in (ROOT/'sihc/configs').glob('*.json'):
        config = json.loads(path.read_text())
        assert config['activation_checkpoint'] == 'none'
        assert config['patch_embed_type'] == 'direct'
        assert not any(k.startswith('wandb') for k in config)
        with torch.device('meta'):
            model = build_model(config['model'])
        assert model.patch_size == 4


def test_resume_objective_and_recipe_guard():
    from argparse import Namespace
    from training.train import validate_resume_settings
    args = Namespace(prediction='velocity', repa=False, lr=2e-4)
    valid = {'args': {'prediction': 'velocity', 'lr': 2e-4}, 'optimizer': {}}
    validate_resume_settings(valid, args)
    with pytest.raises(ValueError, match='objective'):
        validate_resume_settings({'args': {'prediction': 'clean'}, 'optimizer': {}}, args)
    with pytest.raises(ValueError, match='lr'):
        validate_resume_settings({'args': {'prediction': 'velocity', 'lr': 3e-4}, 'optimizer': {}}, args)
    with pytest.raises(ValueError, match='Inference-only'):
        validate_resume_settings({'args': {'prediction': 'velocity'}}, args)


@pytest.mark.parametrize('name,width,depth,stages', [
    ('sihc_sublayer_2x12_d1024_b4_ctx32s8', 1024, 24, (12,12)),
    ('sihc_sublayer_8x12x12_d1280_b4_ctx32s8', 1280, 32, (8,12,12)),
])
def test_explicit_preset_metadata_reconstructs(name,width,depth,stages):
    with torch.device('meta'):
        model = build_model(name, hidden_size=width, depth=depth, stage_sizes=stages, patch_size=4)
    assert model.hidden_size == width and model.stage_sizes == stages
