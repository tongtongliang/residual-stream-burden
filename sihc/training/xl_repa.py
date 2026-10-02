"""XL + online REPA: preview the train/grid/eval sequence; --execute runs it."""
import argparse
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / 'sihc/configs/sublayer_xl_repa.json'
STEPS_PER_EPOCH = 1251


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--data-path', default=os.environ.get('IMAGENET256_ROOT', ''))
    p.add_argument('--teacher-path', default=os.environ.get('DINOV3_PATH', ''))
    p.add_argument('--fid-stats', default=os.environ.get('FID_STATS', ''))
    p.add_argument('--epochs', type=int, default=600)
    p.add_argument('--nproc', type=int, default=8)
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--resume', type=Path)
    p.add_argument('--smoke', action='store_true', help='Train 4 steps, then resume to 6; no sampling/eval.')
    p.add_argument('--execute', action='store_true')
    a = p.parse_args(argv)
    if a.nproc < 1 or a.batch_size < 1:
        p.error('nproc and batch-size must be positive')
    if not a.data_path or not a.teacher_path:
        p.error('Supply data-path and teacher-path (or their environment variables)')
    if a.smoke:
        if a.resume:
            p.error('Smoke requires a fresh run, without --resume')
    elif a.nproc * a.batch_size != 1024 or a.epochs < 40 or a.epochs % 40:
        p.error('Production requires global batch 1024 and epochs divisible by 40')
    elif not a.fid_stats:
        p.error('Supply fid-stats (or FID_STATS) for production evaluation')
    return a


def plan(a, start_step=0):
    """Commands share the existing trainer/evaluators; no second loss implementation."""
    run = a.run_dir.resolve()
    distributed = [sys.executable, '-m', 'torch.distributed.run', '--standalone',
                   f'--nproc_per_node={a.nproc}']
    train = distributed + ['-m', 'training.train', '--config', str(CONFIG),
        '--run_dir', str(run), '--data_path', str(Path(a.data_path).resolve()),
        '--repa_dinov3_path', str(Path(a.teacher_path).resolve()),
        '--batch_size', str(a.batch_size)]
    def checkpoint(step):
        return str(run / 'checkpoints' / f'step_{step:08d}.pt')
    if a.smoke:
        yield train + ['--max_steps', '4', '--save_every', '4', '--log_every', '1']
        yield train + ['--max_steps', '6', '--save_every', '2', '--log_every', '1',
                       '--resume', checkpoint(4)]
        return
    target = a.epochs * STEPS_PER_EPOCH
    if start_step < 0 or start_step > target:
        raise ValueError('Resume step must be within the requested training budget')
    resume = str(a.resume.resolve()) if a.resume else None
    interval = 20 * STEPS_PER_EPOCH
    first = max(interval, ((start_step + interval - 1) // interval) * interval)
    guidance = ['--state_key', 'ema', '--steps', '50', '--cfg', '2.4',
                '--interval_min', '0.1', '--interval_max', '0.9', '--seed', '12345',
                '--compile', '--compile_mode', 'default']
    for step in range(first, target + 1, interval):
        ckpt = resume if step == start_step else checkpoint(step)
        if step > start_step:
            yield train + ['--max_steps', str(step)] + (['--resume', resume] if resume else [])
        yield [sys.executable, '-m', 'evaluation.sample', '--checkpoint', ckpt,
               '--output', str(run / 'samples' / f'step_{step:08d}.png'),
               '--batch_size', '25', '--balanced_labels', '--nrow', '5', *guidance]
        if step % (40 * STEPS_PER_EPOCH) == 0:
            yield distributed + ['-m', 'evaluation.evaluate', '--checkpoint', ckpt,
                '--output_dir', str(run / 'evaluation_50k' / f'step_{step:08d}'),
                '--fid_stats', str(Path(a.fid_stats).resolve()), '--num_samples', '50000',
                '--batch_size', '64', '--sampler', 'heun', '--prediction', 'velocity',
                '--attn_backend', 'flash', '--no_save_fake_features', *guidance]
        resume = ckpt


def main():
    a = parse_args()
    start = 0
    if a.resume:
        from sihc.checkpoint import load_checkpoint, checkpoint_model_name
        ckpt = load_checkpoint(a.resume, mmap=True)
        if checkpoint_model_name(ckpt) != 'sihc_sublayer_5x8_d1024_b4_ctx32s8':
            raise ValueError('Resume requires the sublayer XL flagship checkpoint')
        if not all(key in ckpt for key in ('optimizer', 'ema', 'ema2', 'step')):
            raise ValueError('Resume requires a full training checkpoint with both EMAs')
        start = int(ckpt['step'])
        del ckpt
    if a.execute:
        if not a.resume and a.run_dir.exists():
            raise FileExistsError('Use a fresh run directory or explicit --resume')
        if not Path(a.data_path).is_dir() or not (Path(a.teacher_path) / 'config.json').is_file():
            raise FileNotFoundError('Check the data directory and local teacher config')
        if not a.smoke and not Path(a.fid_stats).is_file():
            raise FileNotFoundError('Missing FID reference statistics')
    env = dict(os.environ)
    env.setdefault('OMP_NUM_THREADS', '1')
    env.setdefault('PYTORCH_ALLOC_CONF', 'expandable_segments:True')
    env.setdefault('IMAGENET256_TENSOR_CACHE_GROUPS', 'cache_shard')
    for command in plan(a, start):
        print(shlex.join(command), flush=True)
        if a.execute:
            subprocess.run(command, cwd=ROOT, env=env, check=True)


if __name__ == '__main__':
    main()
