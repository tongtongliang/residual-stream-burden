"""Anonymous ZIP boundaries and the advertised default model."""
import importlib.util
from pathlib import Path
import stat
import sys
import zipfile

import pytest
import torch

TOOLS = Path(__file__).resolve().parents[1]/'tools'
sys.path.insert(0, str(TOOLS))
from build_supplement import build, DIRECTORIES, ROOT_FILES
sys.path.pop(0)


def source_tree(tmp_path):
    root = tmp_path/'source'
    root.mkdir()
    for name in ROOT_FILES:
        (root/name).write_text('')
    for name in DIRECTORIES:
        (root/name).mkdir()
    (root/'sihc'/'example.py').write_text('value = 1\n')
    return root


def test_archive_excludes_history_and_normalizes_metadata(tmp_path):
    root = source_tree(tmp_path)
    (root/'.git').mkdir()
    (root/'.git'/'config').write_text('private history')
    (root/'sihc'/'__pycache__').mkdir()
    (root/'sihc'/'__pycache__'/'local.pyc').write_bytes(b'private cache')
    output = tmp_path/'supplement.zip'
    result = build(root, output)
    assert result['audit'] == 'passed'
    with zipfile.ZipFile(output) as archive:
        assert all('.git/' not in n and '__pycache__' not in n for n in archive.namelist())
        assert archive.read('sihc-supplement/sihc/example.py') == b'value = 1\n'
        for entry in archive.infolist():
            assert entry.date_time == (1980,1,1,0,0,0)
            assert not entry.extra and not entry.comment
            assert stat.S_IMODE(entry.external_attr >> 16) == 0o644


def test_private_identity_blocks_archive_without_echoing_value(tmp_path):
    root = source_tree(tmp_path)
    identifier = 'example'+'privateowner'
    (root/'docs'/'note.md').write_text(identifier)
    output = tmp_path/'supplement.zip'
    with pytest.raises(ValueError, match='audit failed') as error:
        build(root, output, (identifier,))
    assert identifier not in str(error.value)
    assert not output.exists()


def test_linked_or_binary_source_is_rejected(tmp_path):
    root = source_tree(tmp_path)
    linked = root/'docs'/'outside'
    linked.symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError, match='Symlink'):
        build(root, tmp_path/'supplement.zip')
    linked.unlink()
    (root/'docs'/'weights.pt').write_bytes(b'not source')
    with pytest.raises(ValueError, match='Unreviewed artifact'):
        build(root, tmp_path/'supplement.zip')


def test_api_and_training_default_are_sublayer_context(monkeypatch):
    from sihc import DEFAULT_MODEL, build_model
    from training.train import parse_args
    assert DEFAULT_MODEL == 'sihc_sublayer_1x12_d768_b4_ctx32s4'
    with torch.device('meta'):
        model = build_model()
    assert model.connection_frequency == 'sublayer'
    assert model.in_context_len == 32 and model.depth == 12 and model.hidden_size == 768
    monkeypatch.setattr(sys, 'argv', ['train', '--data_path', 'example-data'])
    assert parse_args().model == DEFAULT_MODEL
