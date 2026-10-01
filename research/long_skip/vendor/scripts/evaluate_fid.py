import argparse
import csv
import json
import os
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.distributed as dist
from PIL import Image

from flowmatching_lthc.checkpoint import load_model_state
from flowmatching_lthc.models import MODEL_NAMES, build_model
from flowmatching_lthc.patch_noise import maybe_load_patch_noise
from flowmatching_lthc.sampling import sample_heun, to_uint8


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--output_dir', required=True)
    p.add_argument('--state_key', default='ema', choices=['ema', 'ema2', 'model'])
    p.add_argument('--model', default='auto', choices=('auto',) + MODEL_NAMES)
    p.add_argument('--num_samples', type=int, default=50000)
    p.add_argument('--batch_size', type=int, default=256, help='Per-rank generation batch size.')
    p.add_argument('--steps', type=int, default=50)
    p.add_argument('--cfg', type=float, default=3.0)
    p.add_argument('--interval_min', type=float, default=0.1)
    p.add_argument('--interval_max', type=float, default=1.0)
    p.add_argument('--noise_scale', type=float, default=1.0)
    p.add_argument('--patch_noise_cov', default='', help='Override; default inherits from checkpoint args if present.')
    p.add_argument('--patch_noise_key', default='')
    p.add_argument('--prediction', default='auto', choices=['auto', 'clean', 'velocity'])
    p.add_argument('--precision', default='bf16', choices=['fp32', 'bf16'], help='Eval autocast dtype. Match training precision.')
    p.add_argument('--num_classes', type=int, default=1000)
    p.add_argument('--fid_stats', default='fid_stats/jit_in256_stats.npz')
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
    p.add_argument('--sample_dir', default='', help='Optional explicit sample directory. Defaults to output_dir/eval_samples_<step>_<timestamp>.')
    p.add_argument('--keep_samples', action='store_true', help='Keep generated PNGs instead of deleting them after metrics.')
    p.add_argument('--fake_stats_dir', default='', help='Directory for generated-sample Inception statistics/features. Defaults to output_dir/fake_stats.')
    p.add_argument('--no_save_fake_features', action='store_true', help='Only save fake mu/sigma, not full generated-sample Inception features/logits.')
    p.add_argument('--sample_only', action='store_true', help='Only generate PNG samples; do not compute FID/IS or write the metric CSV.')
    p.add_argument('--wandb', action='store_true')
    p.add_argument('--wandb_project', default='flowmatching-lthc')
    p.add_argument('--wandb_entity', default='')
    p.add_argument('--wandb_run_id', default='')
    p.add_argument('--wandb_run_name', default='')
    return p.parse_args()


def resolve_patch_noise_args(args, ckpt):
    ckpt_args = ckpt.get('args') or {}
    if not isinstance(ckpt_args, dict):
        ckpt_args = {}
    cov = args.patch_noise_cov or ckpt_args.get('patch_noise_cov') or ''
    key = args.patch_noise_key or ckpt_args.get('patch_noise_key') or 'p_sqrt_uncentered'
    return cov, key


def apply_eval_overrides(args):
    """Allow a running chunk driver to pick up corrected eval settings safely.

    The train/eval chunk driver is a long-running shell process. Editing its
    bash defaults does not affect already-exported shell variables, but the
    driver loads this Python file fresh for every eval boundary. A small JSON
    file in the eval output directory lets us correct non-training settings
    without interrupting the active training process.
    """
    override_path = Path(args.output_dir) / 'eval_overrides.json'
    if not override_path.exists():
        return args

    with override_path.open() as f:
        overrides = json.load(f)

    allowed = {'cfg', 'interval_min', 'interval_max'}
    for key, value in overrides.items():
        if key not in allowed:
            raise ValueError(f'Unsupported eval override key: {key}')
        setattr(args, key, value)
    print(f'[eval] applied overrides from {override_path}: {overrides}', flush=True)
    return args


def init_dist():
    if 'RANK' not in os.environ:
        return False, 0, 1, 0
    dist.init_process_group('nccl')
    rank = dist.get_rank()
    world = dist.get_world_size()
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    torch.cuda.set_device(local_rank)
    return True, rank, world, local_rank


def infer_model_name(args, ckpt):
    if args.model != 'auto':
        return args.model
    ckpt_args = ckpt.get('args') or {}
    if isinstance(ckpt_args, dict) and ckpt_args.get('model') in MODEL_NAMES:
        return ckpt_args['model']
    meta = ckpt.get('meta') or {}
    if isinstance(meta, dict) and meta.get('model') in MODEL_NAMES:
        return meta['model']
    return 'lthc_b4_velocity'


def infer_prediction(args, ckpt):
    if args.prediction != 'auto':
        return args.prediction
    ckpt_args = ckpt.get('args') or {}
    if isinstance(ckpt_args, dict):
        pred = ckpt_args.get('prediction')
        if pred in {'clean', 'velocity'}:
            return pred
    meta = ckpt.get('meta') or {}
    if isinstance(meta, dict):
        pred = meta.get('prediction')
        if pred in {'clean', 'velocity'}:
            return pred
    return 'clean'


def model_kwargs_from_checkpoint(args, ckpt):
    kwargs = {'attn_backend': args.attn_backend}
    ckpt_args = ckpt.get('args') or {}
    if isinstance(ckpt_args, dict) and ckpt_args.get('repa'):
        # The REPA projector is unused during sampling, but it must exist so
        # strict checkpoint loading sees the same parameter tree as training.
        kwargs.update(
            repa_depth=int(ckpt_args.get('repa_depth', 8)),
            repa_z_dims=(int(ckpt_args.get('repa_z_dim', 768)),),
            repa_projector_dim=int(ckpt_args.get('repa_projector_dim', 2048)),
            repa_source=ckpt_args.get('repa_source', 'z_plus_dz'),
        )
    return kwargs


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
        config={'eval_script': 'fid_eval/evaluate.py'},
    )
    wandb.define_metric('eval/checkpoint_step')
    wandb.define_metric('eval/*', step_metric='eval/checkpoint_step')
    wandb.define_metric('eval_50k/checkpoint_step')
    wandb.define_metric('eval_50k/*', step_metric='eval_50k/checkpoint_step')
    payload = {
        'eval/checkpoint_step': int(row['checkpoint_step']),
        'eval/fid': row['fid'],
        'eval/inception_score_mean': row['inception_score_mean'],
        'eval/inception_score_std': row['inception_score_std'],
        'eval/num_samples': row['num_samples'],
        'eval/sample_steps': row['sample_steps'],
        'eval/cfg': row['cfg'],
        'eval/sampling_sec': row['sampling_sec'],
        'eval/metric_sec': row['metric_sec'],
        'eval/total_sec': row['total_sec'],
    }
    if int(row['num_samples']) >= 50000:
        payload.update({
            'eval_50k/checkpoint_step': int(row['checkpoint_step']),
            'eval_50k/fid': row['fid'],
            'eval_50k/inception_score_mean': row['inception_score_mean'],
            'eval_50k/inception_score_std': row['inception_score_std'],
            'eval_50k/cfg': row['cfg'],
            'eval_50k/sample_steps': row['sample_steps'],
            'eval_50k/total_sec': row['total_sec'],
        })
    wandb.log(payload, step=int(row['checkpoint_step']))
    run.finish()


def _as_numpy(array):
    if hasattr(array, 'detach'):
        return array.detach().cpu().numpy()
    return array


def compute_fid_is(sample_dir, fid_stats, cuda, fake_stats_path=None, fake_features_path=None):
    torch_home = Path(os.environ.get('TORCH_HOME', '.cache/torch_home'))
    torch_home.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault('TORCH_HOME', str(torch_home))
    vendor_tf = os.environ.get('TORCH_FIDELITY_PATH', '')
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
    save_cpu_ram = os.environ.get('TORCH_FIDELITY_SAVE_CPU_RAM', '0') == '1'
    metric_verbose = os.environ.get('TORCH_FIDELITY_VERBOSE', '0') == '1'
    features = get_featuresdict_from_dataset(
        dataset,
        extractor,
        batch_size=int(os.environ.get('TORCH_FIDELITY_BATCH_SIZE', '64')),
        cuda=cuda,
        save_cpu_ram=save_cpu_ram,
        verbose=metric_verbose,
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
    real_stats = {'mu': real['mu'], 'sigma': real['sigma']}
    out = {}
    out.update(fid_statistics_to_metric(fake_stats, real_stats, verbose=False))
    out.update(isc_features_to_metric(features['logits_unbiased'], splits=10, shuffle=True, rng_seed=2020))
    real.close()
    return out


def main():
    args = parse_args()
    args = apply_eval_overrides(args)
    distributed, rank, world, local_rank = init_dist()
    device = torch.device(f'cuda:{local_rank}' if distributed and torch.cuda.is_available() else args.device)
    out = Path(args.output_dir)
    if rank == 0:
        out.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(args.checkpoint, map_location='cpu')
    step = int(ckpt.get('step', -1))
    sample_dir = Path(args.sample_dir) if args.sample_dir else out / f'eval_samples_step_{step:08d}_tmp'
    if rank == 0:
        if not args.sample_dir and sample_dir.exists():
            shutil.rmtree(sample_dir)
        sample_dir.mkdir(parents=True, exist_ok=True)

    if distributed:
        dist.barrier(device_ids=[local_rank] if torch.cuda.is_available() else None)
    if rank != 0:
        sample_dir.mkdir(parents=True, exist_ok=True)

    model_name = infer_model_name(args, ckpt)
    prediction = infer_prediction(args, ckpt)
    model_kwargs = model_kwargs_from_checkpoint(args, ckpt)
    patch_noise_cov, patch_noise_key = resolve_patch_noise_args(args, ckpt)
    patch_noise = maybe_load_patch_noise(patch_noise_cov, key=patch_noise_key)
    if patch_noise is not None:
        patch_noise = patch_noise.to(device=device, dtype=torch.float32)
    if rank == 0:
        print(f'[eval] model={model_name} prediction={prediction} model_kwargs={model_kwargs}', flush=True)
        if patch_noise is not None:
            print(f'[eval] patch_noise cov={patch_noise_cov} key={patch_noise_key}', flush=True)
        else:
            print('[eval] patch_noise=isotropic Gaussian', flush=True)
    model = build_model(model_name, **model_kwargs).to(device).eval()
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

    with torch.no_grad(), torch.amp.autocast(
        'cuda',
        dtype=torch.bfloat16,
        enabled=(device.type == 'cuda' and args.precision != 'fp32'),
    ):
        for start in range(rank * args.batch_size, args.num_samples, global_batch):
            end = min(start + args.batch_size, args.num_samples)
            if start >= end:
                continue
            labels = labels_all[start:end].to(device, non_blocking=True)
            x = sample_heun(
                model,
                labels,
                image_size=256,
                noise_scale=args.noise_scale,
                steps=args.steps,
                cfg_scale=args.cfg,
                interval=(args.interval_min, args.interval_max),
                num_classes=args.num_classes,
                prediction=prediction,
                patch_noise=patch_noise,
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
        'sampler': 'heun',
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
        'sample_dir': '' if not args.keep_samples else str(sample_dir),
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
            stem = f'fake_step_{step:08d}_n{args.num_samples}_cfg{args.cfg:g}_s{args.steps}_{prediction}_{args.state_key}'
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
            print(f'[eval] metric failed: {exc!r}', flush=True)

        if row['status'] == 'ok':
            log_wandb(args, row)
        append_csv(out / args.csv_file, row)
        print('EVAL_RESULT_JSON=' + json.dumps(row, ensure_ascii=True), flush=True)
        if not args.keep_samples:
            shutil.rmtree(sample_dir, ignore_errors=True)

    if distributed:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
