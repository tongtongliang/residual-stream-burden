"""Evaluation reports failed metrics to every worker and the calling shell."""

import pytest
import torch

from evaluation.evaluate import metric_failure_status


@pytest.mark.parametrize('status, expected', [('ok', False), ('metric_failed', True)])
def test_local_metric_status_needs_no_process_group(monkeypatch, status, expected):
    def unexpected_collective(*args, **kwargs):
        pytest.fail('single-process evaluation must not call a collective')

    monkeypatch.setattr('evaluation.evaluate.dist.broadcast', unexpected_collective)
    assert metric_failure_status(
        status, distributed=False, rank=0, device=torch.device('cpu'),
    ) is expected


@pytest.mark.parametrize('root_failed', [False, True])
def test_nonzero_rank_receives_metric_outcome(monkeypatch, root_failed):
    def receive_root_status(flag, src):
        assert src == 0
        flag.fill_(int(root_failed))

    monkeypatch.setattr('evaluation.evaluate.dist.broadcast', receive_root_status)
    # Nonzero ranks have not computed metrics and still hold the initial 'ok'.
    assert metric_failure_status(
        'ok', distributed=True, rank=1, device=torch.device('cpu'),
    ) is root_failed


def test_rank_zero_broadcasts_failure(monkeypatch):
    observed = []

    def send_root_status(flag, src):
        observed.append((flag.item(), src))

    monkeypatch.setattr('evaluation.evaluate.dist.broadcast', send_root_status)
    assert metric_failure_status(
        'metric_failed', distributed=True, rank=0, device=torch.device('cpu'),
    ) is True
    assert observed == [(1, 0)]


def test_implicit_sample_directories_are_unique(tmp_path):
    from evaluation.evaluate import prepare_sample_directory

    first = prepare_sample_directory(tmp_path, '', 10)
    marker = first / 'keep.txt'
    marker.write_text('previous evaluation')
    second = prepare_sample_directory(tmp_path, '', 10)
    assert first != second
    assert first.parent == second.parent == tmp_path
    assert marker.read_text() == 'previous evaluation'
    assert second.is_dir() and not any(second.iterdir())


def test_explicit_sample_directory_accepts_only_new_or_empty(tmp_path):
    from evaluation.evaluate import prepare_sample_directory

    explicit = tmp_path / 'samples'
    assert prepare_sample_directory(tmp_path, str(explicit), 0) == explicit
    assert prepare_sample_directory(tmp_path, str(explicit), 0) == explicit
    marker = explicit / 'existing.txt'
    marker.write_text('must survive')
    with pytest.raises(ValueError, match='new or empty'):
        prepare_sample_directory(tmp_path, str(explicit), 0)
    assert marker.read_text() == 'must survive'
    with pytest.raises(ValueError, match='new or empty'):
        prepare_sample_directory(tmp_path, str(marker), 0)
    assert marker.read_text() == 'must survive'


@pytest.mark.parametrize(
    'explicit, keep_samples, metrics_ok, deleted',
    [(False, False, True, True), (True, False, True, False),
     (False, True, True, False), (False, False, False, False)],
)
def test_cleanup_preserves_user_directories_and_failed_samples(
    tmp_path, explicit, keep_samples, metrics_ok, deleted,
):
    from evaluation.evaluate import cleanup_sample_directory

    path = tmp_path / 'samples'
    path.mkdir()
    marker = path / 'image.png'
    marker.write_bytes(b'generated sample')
    cleanup_sample_directory(
        path, explicit=explicit, keep_samples=keep_samples, metrics_ok=metrics_ok,
    )
    assert path.exists() is not deleted
    if not deleted:
        assert marker.read_bytes() == b'generated sample'


@pytest.mark.parametrize('num_samples', [10000, 50000])
def test_tracking_metrics_carry_actual_sample_count(monkeypatch, num_samples):
    import sys
    from types import SimpleNamespace
    from evaluation.evaluate import log_wandb

    logged = []
    fake_wandb = SimpleNamespace(
        init=lambda **kwargs: SimpleNamespace(finish=lambda: None),
        define_metric=lambda *args, **kwargs: None,
        log=lambda payload: logged.append(payload),
    )
    monkeypatch.setitem(sys.modules, 'wandb', fake_wandb)
    args = SimpleNamespace(
        wandb=True, wandb_run_id='test', wandb_project='test',
        wandb_entity='', wandb_run_name='', wandb_epoch=None,
    )
    log_wandb(args, dict(
        checkpoint_step=10, fid=3.0, inception_score_mean=100.0,
        inception_score_std=1.0, num_samples=num_samples,
        sampling_sec=2.0, metric_sec=1.0,
    ))
    assert logged[0]['eval/num_samples'] == num_samples
    assert logged[0]['eval/fid'] == 3.0
    assert not any('50k' in key for key in logged[0])


def test_metric_preflight_requires_existing_statistics(tmp_path):
    from types import SimpleNamespace
    from evaluation.evaluate import preflight_metrics

    args = SimpleNamespace(sample_only=False, fid_stats=tmp_path / 'missing.npz')
    with pytest.raises(FileNotFoundError, match='FID statistics file does not exist'):
        preflight_metrics(args)
    args.sample_only = True
    preflight_metrics(args)


def test_metric_preflight_accepts_inception_statistics(tmp_path):
    import numpy as np
    from evaluation.evaluate import validate_fid_statistics

    path = tmp_path / 'stats.npz'
    np.savez_compressed(path, mu=np.zeros(2048, dtype=np.float32),
                        sigma=np.eye(2048, dtype=np.float32))
    validate_fid_statistics(path)


@pytest.mark.parametrize('kind', ['invalid_file', 'missing_keys', 'wrong_shape', 'nonfinite_mu', 'nonfinite_sigma'])
def test_metric_preflight_rejects_malformed_statistics(tmp_path, kind):
    import numpy as np
    from evaluation.evaluate import validate_fid_statistics

    path = tmp_path / 'stats.npz'
    if kind == 'invalid_file':
        path.write_bytes(b'not an npz file')
    elif kind == 'missing_keys':
        np.savez(path, unrelated=np.zeros(1))
    elif kind == 'wrong_shape':
        np.savez(path, mu=np.zeros(3), sigma=np.eye(3))
    else:
        mu = np.zeros(2048, dtype=np.float32)
        sigma = np.zeros((2048, 2048), dtype=np.float32)
        if kind == 'nonfinite_mu':
            mu[0] = np.nan
        else:
            sigma[0, 0] = np.inf
        np.savez_compressed(path, mu=mu, sigma=sigma)
    with pytest.raises(ValueError, match='Invalid FID statistics'):
        validate_fid_statistics(path)
