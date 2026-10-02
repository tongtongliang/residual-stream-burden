"""Paper protocol metadata can be checked without importing CUDA models."""
import importlib.util
import json
from pathlib import Path
import runpy
import sys
import types

ROOT = Path(__file__).resolve().parents[2]


def test_paper_size_guidance_intervals():
    spec = importlib.util.spec_from_file_location('release_presets', ROOT / 'sihc/presets.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    expected = {'B': (2.9, (0.1, 1.0)), 'L': (2.4, (0.1, 1.0)),
                'H': (2.1, (0.1, 0.9)), 'XL': (2.4, (0.1, 0.9))}
    for size, (cfg, interval) in expected.items():
        actual = module.get_preset(size)
        assert (actual['cfg'], actual['eval_interval']) == (cfg, interval)


def test_stats_download_uses_pinned_asset(monkeypatch, tmp_path):
    calls = []
    stub = types.ModuleType('huggingface_hub')
    stub.hf_hub_download = lambda **kwargs: calls.append(kwargs) or 'downloaded.npz'
    monkeypatch.setitem(sys.modules, 'huggingface_hub', stub)
    monkeypatch.setattr(sys, 'argv', ['download_checkpoint.py', '--id', 'imagenet256-fid-stats', '--output', str(tmp_path)])
    runpy.run_path(str(ROOT / 'sihc/tools/download_checkpoint.py'), run_name='__main__')
    asset = json.loads((ROOT / 'sihc/pretrained/assets.json').read_text())['imagenet256-fid-stats']
    assert calls == [dict(repo_id=asset['repo_id'], filename=asset['file'], revision=asset['revision'], local_dir=str(tmp_path))]
    assert len(asset['revision']) == 40
