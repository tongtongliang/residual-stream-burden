import argparse
import csv
import json
import os
import time
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from sihc.config import (
    architecture_expectations,
    load_config,
    validate_model_architecture,
)
from sihc.activation_checkpointing import (
    ACTIVATION_CHECKPOINT_MODES,
    configure_activation_checkpointing,
)
from sihc.checkpoint import (
    checkpoint_model_kwargs,
    checkpoint_model_name,
    load_checkpoint,
    load_model_state,
    resolve_model_state,
)
from sihc.imagenet import build_dataset, build_loader
from sihc.metrics import (
    batch_geometry_metrics,
    norm_metrics,
    reduce_mean,
    timestep_bin_metrics,
)
from sihc.models import (
    DEFAULT_MODEL,
    MODEL_CHOICES,
    build_model,
    canonical_model_name,
)
from sihc.optim.muon import Muon as LocalMuon, split_muon_params


def parse_args():
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument('--config', default='')
    pre_args, _ = pre.parse_known_args()

    p = argparse.ArgumentParser(parents=[pre])
    p.add_argument('--data_path', default=os.environ.get('IMAGENET256_ROOT', ''))
    p.add_argument('--run_dir', default='runs/sihc_experiment')
    p.add_argument('--batch_size', type=int, default=128)
    p.add_argument('--grad_accum', type=int, default=1)
    p.add_argument('--num_workers', type=int, default=12)
    p.add_argument('--max_steps', type=int, default=750000)
    p.add_argument('--save_every', type=int, default=10000)
    p.add_argument('--no_save', action='store_true', help='Disable checkpoint writes for disposable smoke tests.')
    p.add_argument('--log_every', type=int, default=100)
    p.add_argument('--lr', type=float, default=None)
    p.add_argument('--blr', type=float, default=5e-5, help='JiT-style base LR: absolute lr = blr * effective_batch_size / 256 when --lr is omitted.')
    p.add_argument('--warmup_steps', type=int, default=6250)
    p.add_argument('--weight_decay', type=float, default=0.0)
    p.add_argument('--optimizer', default='adamw', choices=['adamw', 'muon', 'gmuon_chunked'])
    p.add_argument('--grad_clip', type=float, default=0.0, help='Global gradient norm limit; zero disables clipping.')
    p.add_argument('--gmuon_use_kernels', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--muon_aux_lr', type=float, default=0.0)
    p.add_argument('--muon_momentum', type=float, default=0.95)
    p.add_argument('--muon_ns_steps', type=int, default=5)
    p.add_argument('--muon_adjust_lr_fn', default='original', choices=['none', 'original', 'match_rms_adamw'])
    p.add_argument('--ema_decay', type=float, default=0.9999)
    p.add_argument('--ema_decay2', type=float, default=0.0, help='Optional second EMA decay; use 0.9996 to match JiT.')
    p.add_argument('--P_mean', type=float, default=-0.8)
    p.add_argument('--P_std', type=float, default=0.8)
    p.add_argument('--noise_scale', type=float, default=1.0)
    p.add_argument('--t_eps', type=float, default=5e-2)
    p.add_argument('--label_drop_prob', type=float, default=0.1)
    p.add_argument('--repa', action='store_true', help='Enable online DINOv3 REPA auxiliary loss.')
    p.add_argument('--repa_coeff', type=float, default=0.5, help='REPA projection-loss coefficient; 0.5 matches the official REPA default.')
    p.add_argument('--repa_depth', type=int, default=8, help='1-indexed block depth where the model exposes the REPA feature.')
    p.add_argument('--repa_projector_dim', type=int, default=2048)
    p.add_argument('--repa_z_dim', type=int, default=768)
    p.add_argument('--repa_source', default='z_plus_dz', choices=['z_plus_dz', 'dz'])
    p.add_argument('--repa_encoder', default='dinov3_vitb16', choices=['dinov3_vitb16', 'dinov3_vitl16'])
    p.add_argument(
        '--repa_dinov3_path',
        default=os.environ.get('DINOV3_PATH', ''),
        help='Local Hugging Face DINOv3 directory. Can also be set with DINOV3_PATH.',
    )
    p.add_argument('--prediction', default='velocity', choices=['clean', 'velocity'])
    p.add_argument('--compile', action='store_true')
    p.add_argument(
        '--activation_checkpoint',
        default='none',
        choices=ACTIVATION_CHECKPOINT_MODES,
        help='Non-reentrant checkpoint boundary: workspace MLP or full branch.',
    )
    p.add_argument(
        '--log_cuda_memory',
        action='store_true',
        help='Append per-rank-zero peak allocated/reserved CUDA memory to console logs.',
    )
    p.add_argument(
        '--ddp_gradient_as_bucket_view',
        action='store_true',
        help='Make DDP gradients alias all-reduce buckets to remove a duplicate gradient-sized buffer.',
    )
    p.add_argument(
        '--compile_mode',
        default='auto',
        choices=['auto', 'default', 'reduce-overhead', 'max-autotune', 'max-autotune-no-cudagraphs'],
        help='torch.compile mode. auto keeps the historical train.py behavior.',
    )
    p.add_argument('--attn_backend', default='flash', choices=['flash','efficient','math','default'])
    p.add_argument('--mhc_backend', default='nvidia_fused', choices=['nvidia_fused', 'reference'],
                   help='Backend for dynamic mHC controls; reference implements the same routing without the optional NVIDIA package.')
    p.add_argument(
        '--patch_embed_type',
        default='auto',
        choices=['auto', 'direct', 'factorized_linear'],
        help='SiHC patch stem; auto uses the model preset or checkpoint schema.',
    )
    p.add_argument('--bottleneck_dim', type=int, default=128)
    p.add_argument('--model', default=DEFAULT_MODEL, choices=MODEL_CHOICES)
    p.add_argument('--resume', default='')
    p.add_argument(
        '--allow_unsafe_checkpoint',
        action='store_true',
        help='Allow executable pickle for a trusted legacy checkpoint.',
    )
    p.add_argument('--wandb', action='store_true')
    p.add_argument('--wandb_project', default='residual-stream-training-dynamics')
    p.add_argument('--wandb_entity', default='')
    p.add_argument('--wandb_group', default='residual-stream-sihc')
    p.add_argument('--wandb_id', default='')
    p.add_argument('--run_name', default='sihc')
    config_architecture = {}
    if pre_args.config:
        config = load_config(pre_args.config)
        config_architecture = architecture_expectations(config)
        aliases = {
            'batch_size_per_gpu': 'batch_size',
            'train_compile_mode': 'compile_mode',
            'save_every_steps': 'save_every',
        }
        defaults = {aliases.get(key, key): value for key, value in config.items()}
        if 'max_steps' not in defaults and {
            'epochs', 'steps_per_epoch'
        }.issubset(defaults):
            defaults['max_steps'] = int(defaults['epochs']) * int(defaults['steps_per_epoch'])
        valid = {action.dest for action in p._actions}
        p.set_defaults(**{key: value for key, value in defaults.items() if key in valid})
    args = p.parse_args()
    args.config_architecture = config_architecture
    if not args.data_path:
        p.error('--data_path or IMAGENET256_ROOT is required')
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


@torch.no_grad()
def update_ema(ema_params, model_params, decay):
    torch._foreach_mul_(ema_params, decay)
    torch._foreach_add_(ema_params, model_params, alpha=1 - decay)


def save_ckpt(path, model, optimizer, ema_param_names, ema_params, ema_params2, step, args):
    model_kwargs = {'attn_backend': args.attn_backend}
    if args.model.startswith('mhc_'):
        model_kwargs['mhc_backend'] = args.mhc_backend
    if hasattr(model, "patch_embed_type"):
        model_kwargs.update(
            patch_embed_type=model.patch_embed_type,
            bottleneck_dim=getattr(model, "bottleneck_dim", 128),
        )
    if args.repa:
        model_kwargs.update(
            repa_depth=args.repa_depth,
            repa_z_dims=(args.repa_z_dim,),
            repa_projector_dim=args.repa_projector_dim,
            repa_source=args.repa_source,
        )
    state = {
        'format_version': 2,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'ema': {name: ema.detach().cpu() for name, ema in zip(ema_param_names, ema_params)},
        'step': step,
        'args': vars(args),
        'meta': {
            'model': args.model,
            'model_kwargs': model_kwargs,
            'prediction': args.prediction,
            'num_classes': 1000,
            'image_size': getattr(model, 'input_size', 256),
            'patch_size': getattr(model, 'patch_size', 16),
            'num_parameters': sum(param.numel() for param in model.parameters()),
            'depth': getattr(model, 'depth', None),
            'stage_sizes': getattr(model, 'stage_sizes', None),
            'hidden_size': getattr(model, 'hidden_size', None),
            'num_heads': getattr(model, 'num_heads', None),
            'in_context_len': getattr(model, 'in_context_len', 0),
            'in_context_start': getattr(model, 'in_context_start', 0),
            'ema_decay': args.ema_decay,
            'ema_decay2': args.ema_decay2,
        },
    }
    if ema_params2 is not None:
        state['ema2'] = {
            name: ema.detach().cpu() for name, ema in zip(ema_param_names, ema_params2)
        }
    torch.save(state, path)


def read_max_logged_step(csv_path):
    if not csv_path.exists():
        return -1
    max_step = -1
    with csv_path.open(newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or row[0] == 'step':
                continue
            try:
                max_step = max(max_step, int(row[0]))
            except ValueError:
                continue
    return max_step


def lr_for_step(base_lr, step, warmup_steps):
    if warmup_steps <= 0:
        return base_lr
    return base_lr * min(1.0, step / warmup_steps)


def data_position_for_step(completed_steps, batches_per_epoch, grad_accum=1):
    """Return sampler epoch and batch offset after completed optimizer steps."""
    if batches_per_epoch <= 0:
        raise ValueError("batches_per_epoch must be positive")
    if grad_accum <= 0:
        raise ValueError("grad_accum must be positive")
    completed_batches = completed_steps * grad_accum
    return divmod(completed_batches, batches_per_epoch)


def data_epoch_for_step(completed_steps, steps_per_epoch, grad_accum=1):
    return data_position_for_step(
        completed_steps, steps_per_epoch, grad_accum
    )[0]


def build_adamw(params, *, lr, weight_decay, device):
    """Construct AdamW with the fused CUDA path only when supported."""
    return torch.optim.AdamW(
        params,
        lr=lr,
        betas=(0.9, 0.95),
        weight_decay=weight_decay,
        fused=device.type == "cuda",
    )


def reduce_accumulated_losses(losses, grad_accum):
    """Average microbatches, then average the same values across ranks."""
    values = torch.stack(tuple(losses))
    if values.shape[0] != grad_accum:
        raise ValueError(
            f"expected {grad_accum} microbatch losses, got {values.shape[0]}"
        )
    return reduce_mean(values.mean(dim=0))


def set_optimizer_lr(optimizer, lr):
    for group in optimizer.param_groups:
        group['lr'] = lr * group.get('lr_scale', 1.0)


def mark_compile_step():
    if hasattr(torch, 'compiler') and hasattr(torch.compiler, 'cudagraph_mark_step_begin'):
        torch.compiler.cudagraph_mark_step_begin()


def build_repa_teacher(args, device):
    if not args.repa:
        return None
    if args.repa_encoder not in {'dinov3_vitb16', 'dinov3_vitl16'}:
        raise ValueError(f'unsupported REPA encoder: {args.repa_encoder}')
    if not args.repa_dinov3_path:
        raise ValueError(
            '--repa requires --repa_dinov3_path or the DINOV3_PATH environment variable'
        )
    from transformers import AutoModel

    teacher = AutoModel.from_pretrained(
        args.repa_dinov3_path,
        local_files_only=True,
        trust_remote_code=False,
    )
    teacher.eval().to(device)
    for param in teacher.parameters():
        param.requires_grad_(False)
    return teacher


def repa_cosine_loss(pred, target):
    pred = F.normalize(pred.float(), dim=-1)
    target = F.normalize(target.float(), dim=-1)
    return -(pred * target).sum(dim=-1).mean()


def dinov3_patch_tokens(output):
    tokens = output.last_hidden_state
    # DINOv3 ViT checkpoints expose 1 CLS token + 4 register tokens.
    return tokens[:, 5:, :]


def main():
    args = parse_args()
    args.model = canonical_model_name(args.model)
    distributed, rank, world, local_rank = init_dist()
    effective_batch = args.batch_size * args.grad_accum * world
    if args.lr is None:
        args.lr = args.blr * effective_batch / 256
    device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')
    run_dir = Path(args.run_dir)
    if rank == 0:
        (run_dir / 'checkpoints').mkdir(parents=True, exist_ok=True)
        (run_dir / 'logs').mkdir(parents=True, exist_ok=True)
        (run_dir / 'config.json').write_text(json.dumps(vars(args), indent=2) + '\n')
    torch.manual_seed(0 + rank)
    torch.backends.cudnn.benchmark = True
    if rank == 0:
        print(f"Base lr: {args.lr * 256 / effective_batch:.2e}", flush=True)
        print(f"Actual lr: {args.lr:.2e}", flush=True)
        print(f"Effective batch size: {effective_batch}", flush=True)

    dataset = build_dataset(args.data_path, split='train', image_size=256)
    loader = build_loader(dataset, args.batch_size, args.num_workers, distributed, rank, world, seed=0)
    if rank == 0:
        group_kind = getattr(dataset, 'sample_group_kind', '')
        group_suffix = f' sample_group_kind={group_kind}' if group_kind else ''
        print(f'Dataset: {type(dataset).__name__} n={len(dataset)}{group_suffix}', flush=True)

    resume_ckpt = (
        load_checkpoint(
            args.resume,
            mmap=True,
            allow_unsafe=args.allow_unsafe_checkpoint,
        )
        if args.resume
        else None
    )
    if resume_ckpt is not None:
        checkpoint_name = checkpoint_model_name(resume_ckpt)
        if checkpoint_name != args.model:
            raise ValueError(
                f"resume checkpoint model {checkpoint_name!r} does not match "
                f"requested model {args.model!r}"
            )

    model_kwargs = {'attn_backend': args.attn_backend}
    if args.patch_embed_type != 'auto':
        model_kwargs.update(
            patch_embed_type=args.patch_embed_type,
            bottleneck_dim=args.bottleneck_dim,
        )
    if args.repa:
        model_kwargs.update(
            repa_depth=args.repa_depth,
            repa_z_dims=(args.repa_z_dim,),
            repa_projector_dim=args.repa_projector_dim,
            repa_source=args.repa_source,
        )
    if resume_ckpt is not None:
        recovered_kwargs = checkpoint_model_kwargs(
            resume_ckpt,
            model_name=args.model,
        )
        model_kwargs.update(recovered_kwargs)
    if args.model.startswith('mhc_'):
        model_kwargs['mhc_backend'] = args.mhc_backend
    raw_model = build_model(args.model, **model_kwargs).to(device)
    validate_model_architecture(raw_model, args.config_architecture)
    if (
        args.patch_embed_type != "auto"
        and getattr(raw_model, "patch_embed_type", None) != args.patch_embed_type
    ):
        raise ValueError(
            "configured patch_embed_type does not match the selected model: "
            f"config={args.patch_embed_type!r}, "
            f"checkpoint={getattr(raw_model, 'patch_embed_type', None)!r}"
        )
    repa_teacher = build_repa_teacher(args, device)
    repa_mean = repa_std = None
    if args.repa:
        repa_mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        repa_std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
        if rank == 0:
            print(
                f'REPA enabled: coeff={args.repa_coeff} depth={args.repa_depth} '
                f'z_dim={args.repa_z_dim} projector_dim={args.repa_projector_dim} '
                f'source={args.repa_source} teacher={args.repa_dinov3_path}',
                flush=True,
            )
    named_params = list(raw_model.named_parameters())
    ema_param_names = [name for name, _ in named_params]
    model_params = [p for _, p in named_params]
    ema_params = [p.detach().clone() for p in model_params]
    ema_params2 = [p.detach().clone() for p in model_params] if args.ema_decay2 > 0 else None
    if rank == 0:
        secondary = f'{args.ema_decay2:.6f}' if ema_params2 is not None else 'disabled'
        print(f'EMA decays: primary={args.ema_decay:.6f} secondary={secondary}', flush=True)
    if args.optimizer == 'adamw':
        optimizer = build_adamw(
            model_params,
            lr=args.lr,
            weight_decay=args.weight_decay,
            device=device,
        )
    elif args.optimizer == 'gmuon_chunked':
        from sihc.optim.gmuon_chunked import ChunkedGMuon
        optimizer = ChunkedGMuon(
            named_params, lr=args.lr, ns_steps=args.muon_ns_steps,
            momentum=args.muon_momentum, weight_decay=args.weight_decay,
            use_kernels=args.gmuon_use_kernels and device.type == 'cuda',
        )
        if rank == 0:
            print(f'optimizer=gmuon_chunked partition={optimizer.summary()}', flush=True)
    else:
        muon_params, aux_params, muon_names, aux_names = split_muon_params(named_params)
        if rank == 0:
            print(f"optimizer=muon muon_params={len(muon_params)} aux_params={len(aux_params)}", flush=True)
            print(f"muon_examples={muon_names[:8]}", flush=True)
            print(f"aux_examples={aux_names[:8]}", flush=True)
        aux_lr = args.muon_aux_lr if args.muon_aux_lr > 0 else args.lr
        optimizer = torch.optim.Optimizer(model_params, defaults={})
        optimizer.param_groups = []
        optimizer.state = {}
        # Wrap two optimizers behind a minimal composite object while preserving
        # train.py's state_dict/load_state_dict/save_ckpt expectations.
        class _CompositeOptimizer:
            def __init__(self):
                muon_cls = getattr(torch.optim, 'Muon', None) or LocalMuon
                muon_kwargs = dict(
                    lr=args.lr,
                    momentum=args.muon_momentum,
                    nesterov=True,
                    ns_steps=args.muon_ns_steps,
                    weight_decay=args.weight_decay,
                )
                if muon_cls is not LocalMuon:
                    muon_kwargs['adjust_lr_fn'] = None if args.muon_adjust_lr_fn == 'none' else args.muon_adjust_lr_fn
                self.muon = muon_cls(muon_params, **muon_kwargs)
                self.adamw = build_adamw(
                    aux_params,
                    lr=aux_lr,
                    weight_decay=args.weight_decay,
                    device=device,
                )
                for group in self.muon.param_groups:
                    group['lr_scale'] = 1.0
                aux_lr_scale = aux_lr / args.lr if args.lr > 0 else 1.0
                for group in self.adamw.param_groups:
                    group['lr_scale'] = aux_lr_scale
                self.param_groups = self.muon.param_groups + self.adamw.param_groups

            def zero_grad(self, set_to_none=True):
                self.muon.zero_grad(set_to_none=set_to_none)
                self.adamw.zero_grad(set_to_none=set_to_none)

            def step(self):
                self.muon.step()
                self.adamw.step()

            def state_dict(self):
                return {'muon': self.muon.state_dict(), 'adamw': self.adamw.state_dict()}

            def load_state_dict(self, state):
                if 'muon' in state and 'adamw' in state:
                    self.muon.load_state_dict(state['muon'])
                    self.adamw.load_state_dict(state['adamw'])
                else:
                    raise ValueError('cannot load non-Muon optimizer state into composite Muon optimizer')

        optimizer = _CompositeOptimizer()

    start_step = 0
    if resume_ckpt is not None:
        load_model_state(raw_model, resume_ckpt, 'model')
        optimizer.load_state_dict(resume_ckpt['optimizer'])
        start_step = int(resume_ckpt.get('step', resume_ckpt.get('global_step', 0)))
        if 'ema' in resume_ckpt:
            ema_state = resolve_model_state(raw_model, resume_ckpt['ema'], parameters_only=True)
            for name, e in zip([n for n, _ in named_params], ema_params):
                if name in ema_state:
                    e.copy_(ema_state[name].to(device))
        if ema_params2 is not None:
            ema_state2 = resolve_model_state(
                raw_model,
                resume_ckpt.get('ema2', resume_ckpt.get('ema', {})),
                parameters_only=True,
            )
            for name, e in zip([n for n, _ in named_params], ema_params2):
                if name in ema_state2:
                    e.copy_(ema_state2[name].to(device))

    checkpointed_modules = configure_activation_checkpointing(
        raw_model, args.activation_checkpoint
    )
    if rank == 0:
        print(
            f'activation_checkpoint={args.activation_checkpoint} '
            f'wrapped_modules={checkpointed_modules}',
            flush=True,
        )

    model = raw_model
    if args.compile:
        torch._dynamo.config.cache_size_limit = 128
        compile_mode = args.compile_mode
        if compile_mode == 'auto':
            compile_mode = 'default' if args.grad_accum > 1 else 'reduce-overhead'
        if rank == 0:
            print(f"torch_compile_mode={compile_mode}", flush=True)
        model = torch.compile(raw_model, mode=compile_mode, dynamic=False)
    ddp_model = (
        DDP(
            model,
            device_ids=[local_rank],
            gradient_as_bucket_view=args.ddp_gradient_as_bucket_view,
        )
        if distributed
        else model
    )
    if rank == 0:
        print(
            f'ddp_gradient_as_bucket_view={args.ddp_gradient_as_bucket_view}',
            flush=True,
        )

    wb = None
    if rank == 0 and args.wandb:
        import wandb
        wandb_config = dict(vars(args))
        wandb_config.update({
            'model/class': type(raw_model).__name__,
            'model/num_parameters': sum(param.numel() for param in raw_model.parameters()),
            'model/depth': getattr(raw_model, 'depth', None),
            'model/stage_sizes': getattr(raw_model, 'stage_sizes', None),
            'model/hidden_size': getattr(raw_model, 'hidden_size', None),
            'model/num_heads': getattr(raw_model, 'num_heads', None),
            'model/in_context_len': getattr(raw_model, 'in_context_len', 0),
            'model/in_context_start': getattr(raw_model, 'in_context_start', 0),
        })
        wb = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity or None,
            group=args.wandb_group,
            id=args.wandb_id or None,
            name=args.run_name,
            config=wandb_config,
            resume='allow',
        )
        wandb.define_metric('train/global_step')
        wandb.define_metric('train/*', step_metric='train/global_step')
        wandb.define_metric('batch/*', step_metric='train/global_step')
        wandb.define_metric('loss_by_noise/*', step_metric='train/global_step')
        wandb.define_metric('norm/*', step_metric='train/global_step')

    csv_path = run_dir / 'logs' / 'train_metrics.csv'
    if rank == 0 and not csv_path.exists():
        with csv_path.open('w', newline='') as f:
            csv.writer(f).writerow(['step','loss','mse_loss','repa_loss','lr','samples_per_sec','data_time','step_time','grad_norm'])
    last_logged_step = read_max_logged_step(csv_path) if rank == 0 else -1

    data_epoch, data_batch_offset = data_position_for_step(
        start_step, len(loader), args.grad_accum
    )
    if distributed and hasattr(loader.sampler, 'set_epoch'):
        loader.sampler.set_epoch(data_epoch)
    iterator = iter(loader)
    for _ in range(data_batch_offset):
        next(iterator)
    if rank == 0 and start_step:
        print(
            f"resume_data_epoch={data_epoch} "
            f"resume_batch_offset={data_batch_offset}",
            flush=True,
        )

    last_log_time = time.time()
    for step in range(start_step + 1, args.max_steps + 1):
        current_lr = lr_for_step(args.lr, step, args.warmup_steps)
        set_optimizer_lr(optimizer, current_lr)
        optimizer.zero_grad(set_to_none=True)
        data_time = 0.0
        losses_for_log = []
        per_sample_loss_for_log = None
        noise_fraction_for_log = None
        noisy_for_log = None
        velocity_target_for_log = None
        velocity_prediction_for_log = None
        for accum_idx in range(args.grad_accum):
            data_start = time.time()
            try:
                x, y = next(iterator)
            except StopIteration:
                data_epoch += 1
                if distributed and hasattr(loader.sampler, 'set_epoch'):
                    loader.sampler.set_epoch(data_epoch)
                iterator = iter(loader)
                x, y = next(iterator)
            data_time += time.time() - data_start
            x_uint8 = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True).long()
            x01 = x_uint8.float().div(255.0)
            repa_targets = None
            if args.repa:
                with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    repa_input = (x01 - repa_mean) / repa_std
                    repa_targets = dinov3_patch_tokens(repa_teacher(pixel_values=repa_input)).detach()
                if repa_targets.shape[1] != 256 or repa_targets.shape[-1] != args.repa_z_dim:
                    raise RuntimeError(f'unexpected REPA target shape: {tuple(repa_targets.shape)}')
            x = x01.mul(2).sub(1)

            drop = torch.rand(y.shape[0], device=device) < args.label_drop_prob
            y_in = torch.where(drop, torch.full_like(y, 1000), y)
            t = torch.sigmoid(torch.randn(y.shape[0], device=device) * args.P_std + args.P_mean).view(-1,1,1,1)
            e = torch.randn_like(x) * args.noise_scale
            z = t * x + (1 - t) * e
            v = (x - z) / (1 - t).clamp_min(args.t_eps)
            sync_context = ddp_model.no_sync() if distributed and accum_idx < args.grad_accum - 1 else nullcontext()
            with sync_context:
                if args.compile:
                    mark_compile_step()
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    if args.repa:
                        pred, repa_outputs = ddp_model(z, t.flatten(), y_in, return_repa=True)
                        if not isinstance(repa_outputs, (tuple, list)):
                            repa_outputs = (repa_outputs,)
                    else:
                        pred = ddp_model(z, t.flatten(), y_in)
                        repa_outputs = ()
                    if args.prediction == 'clean':
                        v_pred = (pred - z) / (1 - t).clamp_min(args.t_eps)
                    else:
                        v_pred = pred
                    per_sample_loss = (v.float() - v_pred.float()).square().flatten(1).mean(1)
                    mse_loss = per_sample_loss.mean()
                    if args.repa:
                        repa_loss = repa_cosine_loss(repa_outputs[0], repa_targets)
                        loss = mse_loss + args.repa_coeff * repa_loss
                    else:
                        repa_loss = mse_loss.new_zeros(())
                        loss = mse_loss
                losses_for_log.append(
                    torch.stack(
                        (loss.detach(), mse_loss.detach(), repa_loss.detach())
                    )
                )
                per_sample_loss_for_log = per_sample_loss
                noise_fraction_for_log = 1 - t
                noisy_for_log = z
                velocity_target_for_log = v
                velocity_prediction_for_log = v_pred
                (loss / args.grad_accum).backward()

        should_log = step % args.log_every == 0
        payload = {}
        if should_log:
            reduced_losses = reduce_accumulated_losses(
                losses_for_log, args.grad_accum
            )
            payload = {
                'train/global_step': step,
                'train/epoch': step * args.grad_accum / len(loader),
                'train/loss': reduced_losses[0].item(),
                'train/mse_loss': reduced_losses[1].item(),
                'train/repa_loss': reduced_losses[2].item(),
                'train/lr': optimizer.param_groups[0]['lr'],
            }
            payload.update(norm_metrics(raw_model, ema_params, include_groups=step % 500 == 0))
            payload.update(batch_geometry_metrics(
                noisy_for_log,
                velocity_target_for_log,
                velocity_prediction_for_log,
                noise_fraction_for_log,
            ))
            payload.update(timestep_bin_metrics(
                per_sample_loss_for_log,
                noise_fraction_for_log,
                bins=10,
            ))

        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model_params, args.grad_clip, error_if_nonfinite=True)
        optimizer.step()
        update_ema(ema_params, model_params, args.ema_decay)
        if ema_params2 is not None:
            update_ema(ema_params2, model_params, args.ema_decay2)
        save_due = not args.no_save and (
            step % args.save_every == 0 or step == args.max_steps
        )
        if should_log:
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            now = time.time()
            log_window_sec = now - last_log_time
            sps = (
                args.batch_size * world * args.grad_accum * args.log_every
                / max(log_window_sec, 1e-9)
            )
            peak_allocated = (
                torch.cuda.max_memory_allocated(device) / 2**30
                if device.type == 'cuda'
                else 0.0
            )
            peak_reserved = (
                torch.cuda.max_memory_reserved(device) / 2**30
                if device.type == 'cuda'
                else 0.0
            )
            payload.update({
                'train/samples_per_sec': sps,
                'train/log_window_sec': log_window_sec,
                'train/peak_memory_allocated_gib': peak_allocated,
                'train/peak_memory_reserved_gib': peak_reserved,
            })
            if rank == 0 and step > last_logged_step:
                average_step_time = log_window_sec / args.log_every
                row = [
                    step,
                    payload['train/loss'],
                    payload['train/mse_loss'],
                    payload['train/repa_loss'],
                    optimizer.param_groups[0]['lr'],
                    sps,
                    data_time,
                    average_step_time,
                    payload['train/grad_norm'],
                ]
                with csv_path.open('a', newline='') as f:
                    csv.writer(f).writerow(row)
                if wb:
                    wb.log(payload, step=step)
                print(
                    f"step={step} loss={payload['train/loss']:.5f} "
                    f"grad_norm={payload['train/grad_norm']:.3f} sps={sps:.1f} "
                    f"window={log_window_sec:.3f}s"
                    + (
                        f" peak_alloc={payload['train/peak_memory_allocated_gib']:.2f}GiB"
                        f" peak_reserved={payload['train/peak_memory_reserved_gib']:.2f}GiB"
                        if args.log_cuda_memory
                        else ""
                    ),
                    flush=True,
                )
                last_logged_step = step
            last_log_time = now
        if rank == 0 and save_due:
            save_ckpt(
                run_dir / 'checkpoints' / f'step_{step:08d}.pt',
                raw_model,
                optimizer,
                ema_param_names,
                ema_params,
                ema_params2,
                step,
                args,
            )
    if wb:
        wb.finish()
    if distributed:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
