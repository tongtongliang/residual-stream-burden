import argparse
import csv
import json
import os
import shutil
import sys
import time
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import torch
import torch.distributed as dist
from PIL import Image

from sihc.checkpoint import (
    build_model_from_checkpoint,
    checkpoint_model_name,
    checkpoint_prediction,
    load_checkpoint,
    load_model_state,
)
from sihc.models import MODEL_CHOICES
from sihc.sampling import sample_euler, sample_heun, to_uint8


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--output_dir', required=True)
    p.add_argument('--state_key', default='ema', choices=['ema', 'ema2', 'model'])
    p.add_argument('--model', default='auto', choices=('auto',) + MODEL_CHOICES)
    p.add_argument('--num_samples', type=int, default=50000)
    p.add_argument('--batch_size', type=int, default=256, help='Per-rank generation batch size.')
    p.add_argument('--steps', type=int, default=50)
    p.add_argument('--sampler', default='heun', choices=['heun', 'euler'])
    p.add_argument('--cfg', type=float, default=2.4)
    p.add_argument('--interval_min', type=float, default=0.1)
    p.add_argument('--interval_max', type=float, default=0.9)
    p.add_argument('--noise_scale', type=float, default=1.0)
    p.add_argument('--prediction', default='auto', choices=['auto', 'clean', 'velocity'])
    p.add_argument('--num_classes', type=int, default=1000)
    p.add_argument(
        '--fid_stats',
        default=os.environ.get('FID_STATS', 'fid_stats/imagenet256_stats.npz'),
    )
    p.add_argument('--device', default='cuda')
    p.add_argument('--attn_backend', default='flash', choices=['flash', 'efficient', 'math', 'default'])
    p.add_argument('--compile', action='store_true')
    p.add_argument(
        '--compile_mode',
        default='reduce-overhead',
        choices=['default', 'reduce-overhead', 'max-autotune', 'max-autotune-no-cudagraphs'],
    )
    p.add_argument('--seed', type=int, default=12345)
    p.add_argument('--csv_file', default='metrics_history.csv')
    p.add_argument('--sample_dir', default='', help='New or empty sample directory, preserved after evaluation. Default: a unique temporary directory under output_dir.')
    p.add_argument('--keep_samples', action='store_true', help='Keep the automatically created sample directory after successful metrics; explicit --sample_dir is always preserved.')
    p.add_argument('--fake_stats_dir', default='', help='Directory for generated-sample Inception statistics/features. Defaults to output_dir/fake_stats.')
    p.add_argument('--no_save_fake_features', action='store_true', help='Only save fake mu/sigma, not full generated-sample Inception features/logits.')
    p.add_argument('--sample_only', action='store_true', help='Only generate PNG samples; do not compute FID/IS or write the metric CSV.')
    p.add_argument('--wandb', action='store_true')
    p.add_argument('--wandb_project', default='sihc')
    p.add_argument('--wandb_entity', default='')
    p.add_argument('--wandb_run_id', default='')
    p.add_argument('--wandb_run_name', default='')
    p.add_argument('--wandb_epoch', type=float, default=None)
    p.add_argument(
        '--allow_unsafe_checkpoint',
        action='store_true',
        help='Allow executable pickle for a trusted legacy checkpoint.',
    )
    p.add_argument('--kernel_backend', default=None, choices=['legacy', 'tuple'],
                   help='Override block-wise routing backend; default follows checkpoint.')
    return p.parse_args()


def validate_fid_statistics(path):
    """Reject missing or incompatible Inception statistics before sampling."""
    import numpy as np

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f'FID statistics file does not exist: {path}')
    try:
        with np.load(path, allow_pickle=False) as statistics:
            mu, sigma = statistics['mu'], statistics['sigma']
            if mu.shape != (2048,) or sigma.shape != (2048, 2048):
                raise ValueError('expected mu shape (2048,) and sigma shape (2048, 2048)')
            if not np.isfinite(mu).all() or not np.isfinite(sigma).all():
                raise ValueError('mu and sigma must contain only finite values')
    except (OSError, ValueError, TypeError, KeyError) as exc:
        raise ValueError(f'Invalid FID statistics in {path}: {exc}') from exc


def preflight_metrics(args):
    if not args.sample_only:
        validate_fid_statistics(args.fid_stats)


def init_dist():
    if 'RANK' not in os.environ:
        return False, 0, 1, 0
    dist.init_process_group('nccl', timeout=timedelta(hours=2))
    rank = dist.get_rank()
    world = dist.get_world_size()
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    torch.cuda.set_device(local_rank)
    return True, rank, world, local_rank


def infer_model_name(args, ckpt):
    if args.model != 'auto':
        return args.model
    return checkpoint_model_name(ckpt)


def infer_prediction(args, ckpt):
    if args.prediction != 'auto':
        return args.prediction
    return checkpoint_prediction(ckpt)


def save_grid(images, path, nrow=10):
    images = images[: nrow * nrow].cpu().numpy().transpose(0, 2, 3, 1)
    h, w = images.shape[1], images.shape[2]
    canvas = Image.new('RGB', (nrow * w, nrow * h))
    for i, img in enumerate(images):
        canvas.paste(Image.fromarray(img), ((i % nrow) * w, (i // nrow) * h))
    canvas.save(path)


def balanced_labels(num_samples, num_classes):
    labels = torch.arange(num_classes, dtype=torch.long).repeat_interleave(num_samples // num_classes)
    rem = num_samples - labels.numel()
    if rem > 0:
        labels = torch.cat([labels, torch.arange(rem, dtype=torch.long)])
    return labels


def append_csv(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open('a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def prepare_sample_directory(output_dir, explicit_sample_dir, step):
    """Create owned temporary storage or accept only an empty explicit directory."""
    if explicit_sample_dir:
        path = Path(explicit_sample_dir)
        if path.exists() and (not path.is_dir() or any(path.iterdir())):
            raise ValueError('--sample_dir must be a new or empty directory')
        path.mkdir(parents=True, exist_ok=True)
        return path
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix=f'eval_samples_step_{step:08d}_', dir=output_dir))


def cleanup_sample_directory(path, *, explicit, keep_samples, metrics_ok):
    """Delete only an evaluator-owned temporary directory after successful metrics."""
    if not explicit and not keep_samples and metrics_ok:
        shutil.rmtree(path)


def metric_failure_status(status, *, distributed, rank, device):
    """Share rank zero's metric outcome before every worker exits."""
    failed = rank == 0 and status != 'ok'
    if not distributed:
        return failed
    flag = torch.tensor(int(failed), device=device, dtype=torch.int32)
    dist.broadcast(flag, src=0)
    return bool(flag.item())


def log_wandb(args, row):
    if not args.wandb:
        return
    if not args.wandb_run_id:
        raise ValueError('--wandb requires --wandb_run_id so eval logs attach to the training run')
    import wandb

    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity or None,
        id=args.wandb_run_id,
        name=args.wandb_run_name or None,
        resume='allow',
        config={'eval_script': 'evaluation/evaluate.py'},
    )
    wandb.define_metric('eval/global_step')
    wandb.define_metric('eval/*', step_metric='eval/global_step')
    payload = {
        'eval/global_step': int(row['checkpoint_step']),
        'eval/epoch': args.wandb_epoch,
        'eval/fid': row['fid'],
        'eval/inception_score_mean': row['inception_score_mean'],
        'eval/inception_score_std': row['inception_score_std'],
        'eval/num_samples': row['num_samples'],
        'eval/sampling_sec': row['sampling_sec'],
        'eval/metric_sec': row['metric_sec'],
    }
    if args.wandb_epoch is None:
        payload.pop('eval/epoch')
    wandb.log(payload)
    run.finish()


def _as_numpy(array):
    if hasattr(array, 'detach'):
        return array.detach().cpu().numpy()
    return array


def compute_fid_is(sample_dir, fid_stats, cuda, fake_stats_path=None, fake_features_path=None):
    torch_home = Path(os.environ.get('TORCH_HOME', '.cache/torch_home'))
    torch_home.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault('TORCH_HOME', str(torch_home))
    vendor_tf = os.environ.get('TORCH_FIDELITY_ROOT', '')
    if vendor_tf and Path(vendor_tf).is_dir() and vendor_tf not in sys.path:
        sys.path.insert(0, vendor_tf)

    import numpy as np
    from torch_fidelity.datasets import ImagesPathDataset
    from torch_fidelity.metric_fid import fid_features_to_statistics, fid_statistics_to_metric
    from torch_fidelity.metric_isc import isc_features_to_metric
    from torch_fidelity.utils import create_feature_extractor, get_featuresdict_from_dataset, glob_samples_paths

    files = glob_samples_paths(str(sample_dir), False, 'png,jpg,jpeg', 'jpg,jpeg', verbose=False)
    dataset = ImagesPathDataset(files)
    extractor = create_feature_extractor(
        'inception-v3-compat',
        ['2048', 'logits_unbiased'],
        cuda=cuda,
        verbose=False,
    )
    features = get_featuresdict_from_dataset(
        dataset,
        extractor,
        batch_size=int(os.environ.get('TORCH_FIDELITY_BATCH_SIZE', '64')),
        cuda=cuda,
        save_cpu_ram=os.environ.get('TORCH_FIDELITY_SAVE_CPU_RAM', '0') == '1',
        verbose=os.environ.get('TORCH_FIDELITY_VERBOSE', '0') == '1',
    )

    real = np.load(fid_stats)
    fake_stats = fid_features_to_statistics(features['2048'])
    if fake_stats_path is not None:
        fake_stats_path = Path(fake_stats_path)
        fake_stats_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            fake_stats_path,
            mu=_as_numpy(fake_stats['mu']),
            sigma=_as_numpy(fake_stats['sigma']),
            num_samples=len(files),
            sample_dir=str(sample_dir),
        )
    if fake_features_path is not None:
        fake_features_path = Path(fake_features_path)
        fake_features_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            fake_features_path,
            features_2048=_as_numpy(features['2048']),
            logits_unbiased=_as_numpy(features['logits_unbiased']),
            num_samples=len(files),
            sample_dir=str(sample_dir),
        )
    out = {}
    out.update(fid_statistics_to_metric(fake_stats, {'mu': real['mu'], 'sigma': real['sigma']}, verbose=False))
    out.update(isc_features_to_metric(features['logits_unbiased'], splits=10, shuffle=True, rng_seed=2020))
    real.close()
    return out


def main():
    args = parse_args()
    preflight_metrics(args)
    distributed, rank, world, local_rank = init_dist()
    device = torch.device(f'cuda:{local_rank}' if distributed and torch.cuda.is_available() else args.device)
    out = Path(args.output_dir)
    if rank == 0:
        out.mkdir(parents=True, exist_ok=True)

    ckpt = load_checkpoint(
        args.checkpoint,
        mmap=True,
        allow_unsafe=args.allow_unsafe_checkpoint,
    )
    step = int(ckpt.get('step', -1))
    directory_info = [None, None]
    if rank == 0:
        try:
            directory_info[0] = str(prepare_sample_directory(out, args.sample_dir, step))
        except (OSError, ValueError) as exc:
            directory_info[1] = str(exc)
    if distributed:
        dist.broadcast_object_list(directory_info, src=0, device=device)
    if directory_info[1] is not None:
        raise RuntimeError(directory_info[1])
    sample_dir = Path(directory_info[0])

    model_name = infer_model_name(args, ckpt)
    prediction = infer_prediction(args, ckpt)
    model = build_model_from_checkpoint(
        ckpt,
        model_name=model_name,
        overrides={'kernel_backend': args.kernel_backend} if args.kernel_backend else None,
        attn_backend=args.attn_backend,
    ).to(device).eval()
    load_model_state(model, ckpt, args.state_key)
    if args.compile:
        torch._dynamo.config.cache_size_limit = 128
        print(f'torch_compile_mode={args.compile_mode}', flush=True)
        model = torch.compile(model, mode=args.compile_mode)

    labels_all = balanced_labels(args.num_samples, args.num_classes)
    torch.manual_seed(args.seed + rank * 1000003)

    generated_local = 0
    first_grid = None
    sample_start = time.time()
    write_sec = 0.0
    global_batch = args.batch_size * world

    with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=device.type == 'cuda'):
        for start in range(rank * args.batch_size, args.num_samples, global_batch):
            end = min(start + args.batch_size, args.num_samples)
            if start >= end:
                continue
            labels = labels_all[start:end].to(device, non_blocking=True)
            sampler_fn = {
                'heun': sample_heun,
                'euler': sample_euler,
            }[args.sampler]
            x = sampler_fn(
                model,
                labels,
                image_size=256,
                noise_scale=args.noise_scale,
                steps=args.steps,
                cfg_scale=args.cfg,
                interval=(args.interval_min, args.interval_max),
                num_classes=args.num_classes,
                prediction=prediction,
            )
            u8 = to_uint8(x)
            if rank == 0 and first_grid is None:
                first_grid = u8[:100].cpu()

            t_write = time.time()
            arr = u8.cpu().numpy().transpose(0, 2, 3, 1)
            for local_i, img in enumerate(arr):
                Image.fromarray(img).save(sample_dir / f'{start + local_i:08d}.png')
            write_sec += time.time() - t_write
            generated_local += end - start
            print(f'[rank {rank}] sampled {generated_local} local images; last_global={end}/{args.num_samples}', flush=True)

    sampling_sec = time.time() - sample_start
    if rank == 0 and first_grid is not None:
        save_grid(first_grid, out / f'sample_grid_step_{step:08d}.png')

    if distributed:
        dist.barrier(device_ids=[local_rank] if torch.cuda.is_available() else None)

    if args.sample_only:
        if rank == 0:
            print('SAMPLE_ONLY_DONE=' + json.dumps({
                'checkpoint': str(args.checkpoint),
                'checkpoint_step': step,
                'sample_dir': str(sample_dir),
                'num_samples': args.num_samples,
                'sampling_sec': sampling_sec,
            }), flush=True)
        if distributed:
            dist.destroy_process_group()
        return

    row = {
        'timestamp_utc': datetime.utcnow().isoformat(),
        'checkpoint': str(args.checkpoint),
        'checkpoint_step': step,
        'state_key': args.state_key,
        'prediction': prediction,
        'num_samples': args.num_samples,
        'sampler': args.sampler,
        'sample_steps': args.steps,
        'cfg': args.cfg,
        'interval_min': args.interval_min,
        'interval_max': args.interval_max,
        'noise_scale': args.noise_scale,
        'batch_size_per_rank': args.batch_size,
        'world_size': world,
        'sampling_sec': sampling_sec,
        'png_write_sec_rank0': write_sec if rank == 0 else '',
        'metric_sec': '',
        'total_sec': '',
        'fid': '',
        'inception_score_mean': '',
        'inception_score_std': '',
        'fid_stats': str(args.fid_stats),
        'fake_stats_npz': '',
        'fake_features_npz': '',
        'sample_dir': str(sample_dir) if args.keep_samples or args.sample_dir else '',
        'status': 'ok',
        'error': '',
    }

    total_start = sample_start
    if rank == 0:
        metric_start = time.time()
        try:
            actual_samples = sum(1 for _ in sample_dir.glob('*.png'))
            if actual_samples != args.num_samples:
                raise RuntimeError(
                    f'generated sample count mismatch: expected {args.num_samples}, found {actual_samples} in {sample_dir}'
                )
            fake_stats_dir = Path(args.fake_stats_dir) if args.fake_stats_dir else out / 'fake_stats'
            sampler_stem = '' if args.sampler == 'heun' else f'_{args.sampler}'
            stem = f'fake{sampler_stem}_step_{step:08d}_n{args.num_samples}_cfg{args.cfg:g}_s{args.steps}_{prediction}_{args.state_key}'
            fake_stats_path = fake_stats_dir / f'{stem}_stats.npz'
            fake_features_path = None if args.no_save_fake_features else fake_stats_dir / f'{stem}_features.npz'
            metrics = compute_fid_is(
                sample_dir,
                args.fid_stats,
                cuda=(device.type == 'cuda'),
                fake_stats_path=fake_stats_path,
                fake_features_path=fake_features_path,
            )
            row['metric_sec'] = time.time() - metric_start
            row['total_sec'] = time.time() - total_start
            row['fid'] = float(metrics['frechet_inception_distance'])
            row['inception_score_mean'] = float(metrics['inception_score_mean'])
            row['inception_score_std'] = float(metrics.get('inception_score_std', 0.0))
            row['fake_stats_npz'] = str(fake_stats_path)
            row['fake_features_npz'] = '' if fake_features_path is None else str(fake_features_path)
        except Exception as exc:
            row['metric_sec'] = time.time() - metric_start
            row['total_sec'] = time.time() - total_start
            row['status'] = 'metric_failed'
            row['error'] = repr(exc)
            row['sample_dir'] = str(sample_dir)
            print(f'[eval] metric failed: {exc!r}', flush=True)

        if row['status'] == 'ok':
            log_wandb(args, row)
        append_csv(out / args.csv_file, row)
        print('EVAL_RESULT_JSON=' + json.dumps(row, ensure_ascii=True), flush=True)
        cleanup_sample_directory(
            sample_dir, explicit=bool(args.sample_dir), keep_samples=args.keep_samples,
            metrics_ok=row['status'] == 'ok',
        )

    metric_failed = metric_failure_status(
        row['status'], distributed=distributed, rank=rank, device=device,
    )
    if distributed:
        dist.destroy_process_group()
    if metric_failed:
        raise RuntimeError('Evaluation metrics failed; see EVAL_RESULT_JSON and the metric CSV for details.')


if __name__ == '__main__':
    main()
