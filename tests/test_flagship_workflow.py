"""Flagship scheduling, true resume wiring, and stop-on-failure contract."""
import subprocess
import sys

import pytest

from training import flagship


def args(tmp_path, *extra):
    return flagship.parse_args(['--run-dir', str(tmp_path/'run'),
        '--data-path', str(tmp_path/'data'), '--teacher-path', str(tmp_path/'teacher'),
        '--fid-stats', str(tmp_path/'stats.npz'), *extra])


def test_full_schedule(tmp_path):
    commands = list(flagship.plan(args(tmp_path)))
    trains = [c for c in commands if 'training.train' in c]
    grids = [c for c in commands if 'evaluation.sample' in c]
    evals = [c for c in commands if 'evaluation.evaluate' in c]
    assert (len(trains), len(grids), len(evals)) == (30, 30, 15)
    assert trains[-1][trains[-1].index('--max_steps')+1] == '750600'
    assert '--resume' not in trains[0] and '--resume' in trains[1]
    assert all('--wandb' not in c for c in commands)
    for c in grids + evals:
        assert c[c.index('--cfg')+1] == '2.4'
        assert c[c.index('--interval_max')+1] == '0.9'
    assert commands[4] == evals[0]  # train20, grid20, train40, grid40, eval40


def test_resume_evaluates_boundary_before_training(tmp_path):
    a = args(tmp_path, '--resume', str(tmp_path/'step_00650520.pt'))
    commands = list(flagship.plan(a, 650520))
    assert 'evaluation.sample' in commands[0]
    assert 'evaluation.evaluate' in commands[1]
    assert 'training.train' in commands[2]
    assert commands[2][commands[2].index('--max_steps')+1] == str(540*1251)
    assert commands[2][commands[2].index('--resume')+1] == str(a.resume)


def test_smoke_saves_then_resumes_optimizer(tmp_path):
    commands = list(flagship.plan(args(tmp_path, '--smoke', '--nproc', '1', '--batch-size', '2')))
    assert len(commands) == 2
    assert commands[0][commands[0].index('--max_steps')+1] == '4'
    assert commands[1][commands[1].index('--max_steps')+1] == '6'
    assert commands[1][commands[1].index('--resume')+1].endswith('step_00000004.pt')


def test_bad_production_batch_rejected(tmp_path):
    with pytest.raises(SystemExit):
        args(tmp_path, '--nproc', '1')


def test_child_failure_stops_sequence(tmp_path, monkeypatch):
    (tmp_path/'data').mkdir()
    (tmp_path/'teacher').mkdir()
    (tmp_path/'teacher/config.json').write_text('{}')
    a = args(tmp_path, '--smoke', '--nproc', '1', '--batch-size', '2', '--execute')
    monkeypatch.setattr(flagship, 'parse_args', lambda: a)
    calls = []
    def fail(command, **kwargs):
        calls.append(command)
        raise subprocess.CalledProcessError(1, command)
    monkeypatch.setattr(subprocess, 'run', fail)
    with pytest.raises(subprocess.CalledProcessError):
        flagship.main()
    assert len(calls) == 1
