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

from flowmatching_lthc.checkpoint import load_model_state, migrate_checkpoint_state
from flowmatching_lthc.imagenet import build_dataset, build_loader
from flowmatching_lthc.models import MODEL_NAMES, build_model
from flowmatching_lthc.patch_noise import maybe_load_patch_noise


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_path', required=True)
    p.add_argument('--run_dir', default='runs/lthc_experiment')
    p.add_argument('--batch_size', type=int, default=128)
    p.add_argument('--grad_accum', type=int, default=1)
    p.add_argument('--num_workers', type=int, default=12)
    p.add_argument('--max_steps', type=int, default=750000)
    p.add_argument('--save_every', type=int, default=10000)
    p.add_argument('--log_every', type=int, default=100)
    p.add_argument('--lr', type=float, default=None)
    p.add_argument('--blr', type=float, default=5e-5, help='JiT-style base LR: absolute lr = blr * effective_batch_size / 256 when --lr is omitted.')
    p.add_argument('--warmup_steps', type=int, default=6250)
    p.add_argument('--weight_decay', type=float, default=0.0)
    p.add_argument('--optimizer', default='adamw', choices=['adamw', 'muon'])
    p.add_argument('--muon_aux_lr', type=float, default=0.0)
    p.add_argument('--muon_momentum', type=float, default=0.95)
    p.add_argument('--muon_ns_steps', type=int, default=5)
    p.add_argument('--muon_adjust_lr_fn', default='original', choices=['none', 'original', 'match_rms_adamw'])
    p.add_argument('--ema_decay', type=float, default=0.9999)
    p.add_argument('--ema_decay2', type=float, default=0.0, help='Optional second EMA decay; use 0.9996 to match JiT.')
    p.add_argument('--P_mean', type=float, default=-0.8)
    p.add_argument('--P_std', type=float, default=0.8)
    p.add_argument('--noise_scale', type=float, default=1.0)
    p.add_argument('--patch_noise_cov', default='', help='npz with p_sqrt_* for colored patch noise. Empty = isotropic Gaussian.')
    p.add_argument('--patch_noise_key', default='p_sqrt_uncentered')
    p.add_argument('--t_eps', type=float, default=5e-2)
    p.add_argument('--label_drop_prob', type=float, default=0.1)
    p.add_argument('--repa', action='store_true', help='Enable online DINO REPA auxiliary loss.')
    p.add_argument('--repa_coeff', type=float, default=0.5, help='REPA projection-loss coefficient; 0.5 matches the official REPA default.')
    p.add_argument('--repa_depth', type=int, default=8, help='1-indexed block depth where the model exposes the REPA feature.')
    p.add_argument('--repa_projector_dim', type=int, default=2048)
    p.add_argument('--repa_z_dim', type=int, default=768)
    p.add_argument('--repa_source', default='z_plus_dz', choices=['z_plus_dz', 'dz', 'semantic'])
    p.add_argument('--repa_encoder', default='dinov3_vitb16', choices=['dinov2_vitb14', 'dinov3_vitb16', 'dinov3_vitl16'])
    p.add_argument('--repa_dinov3_path', default=os.environ.get('DINOV3_PATH', ''))
    p.add_argument('--repa_dinov2_path', default='facebook/dinov2-base')
    p.add_argument('--prediction', default='velocity', choices=['clean', 'velocity'])
    p.add_argument('--precision', default='bf16', choices=['fp32', 'bf16'], help='Train autocast dtype. Use fp32 for numerical stability (required for virtual-width A/B/C).')
    p.add_argument('--compile', action='store_true')
    p.add_argument(
        '--compile_mode',
        default='auto',
        choices=['auto', 'default', 'reduce-overhead', 'max-autotune', 'max-autotune-no-cudagraphs'],
        help='torch.compile mode. auto keeps the historical train.py behavior.',
    )
    p.add_argument('--attn_backend', default='flash', choices=['flash','efficient','math','default'])
    p.add_argument('--model', default='jit_b16_fullpatch_longskip', choices=MODEL_NAMES)
    p.add_argument('--resume', default='')
    p.add_argument('--wandb', action='store_true')
    p.add_argument('--wandb_project', default='flowmatching-lthc')
    p.add_argument('--wandb_entity', default='')
    p.add_argument('--wandb_group', default='lthc-b4-velocity')
    p.add_argument('--wandb_id', default='')
    p.add_argument('--run_name', default='lthc_b4_velocity')
    return p.parse_args()


def init_dist():
    if 'RANK' not in os.environ:
        return False, 0, 1, 0
    dist.init_process_group('nccl')
    rank = dist.get_rank()
    world = dist.get_world_size()
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    torch.cuda.set_device(local_rank)
    return True, rank, world, local_rank


def update_ema(ema_params, model_params, decay):
    with torch.no_grad():
        for e, p in zip(ema_params, model_params):
            e.mul_(decay).add_(p.detach(), alpha=1 - decay)


def save_ckpt(path, model, optimizer, ema_params, ema_params2, step, args):
    state = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'ema': {name: ema.detach().cpu() for (name, _), ema in zip(model.named_parameters(), ema_params)},
        'step': step,
        'args': vars(args),
        'meta': {
            'model': args.model,
            'prediction': args.prediction,
            'num_classes': 1000,
            'image_size': getattr(model, 'input_size', 256),
            'patch_size': getattr(model, 'patch_size', 16),
            'num_parameters': sum(param.numel() for param in model.parameters()),
            'depth': getattr(model, 'depth', None),
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
            name: ema.detach().cpu() for (name, _), ema in zip(model.named_parameters(), ema_params2)
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


def set_optimizer_lr(optimizer, lr):
    for group in optimizer.param_groups:
        group['lr'] = lr * group.get('lr_scale', 1.0)


def grad_global_norm(parameters):
    norms = [p.grad.detach().float().norm(2) for p in parameters if p.grad is not None]
    if not norms:
        return 0.0
    return float(torch.linalg.vector_norm(torch.stack(norms), ord=2).item())


def mark_compile_step():
    if hasattr(torch, 'compiler') and hasattr(torch.compiler, 'cudagraph_mark_step_begin'):
        torch.compiler.cudagraph_mark_step_begin()


def log_skip_curves(raw_model, device, step, wb, csv_path):
    if not hasattr(raw_model, 'skip_effective_values'):
        return
    t_grid = torch.linspace(0.05, 0.95, 20, device=device)
    with torch.no_grad():
        values = raw_model.skip_effective_values(t_grid)
    alpha_eff = values['clean_alpha_effective'].detach().cpu().tolist()
    beta_eff = values['clean_beta_effective'].detach().cpu().tolist()
    alpha = values.get('alpha')
    beta = values.get('beta')
    alpha_raw = alpha.detach().cpu().tolist() if alpha is not None else [None] * len(alpha_eff)
    beta_raw = beta.detach().cpu().tolist() if beta is not None else [None] * len(beta_eff)

    write_header = not csv_path.exists()
    with csv_path.open('a', newline='') as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(['step', 't', 'alpha', 'beta', 'clean_alpha_effective', 'clean_beta_effective'])
        for t_val, a, b, ae, be in zip(t_grid.detach().cpu().tolist(), alpha_raw, beta_raw, alpha_eff, beta_eff):
            writer.writerow([step, t_val, a, b, ae, be])

    if wb:
        payload = {}
        for t_val, ae, be in zip(t_grid.detach().cpu().tolist(), alpha_eff, beta_eff):
            tag = f'{t_val:.2f}'
            payload[f'skip/clean_alpha_effective_t_{tag}'] = ae
            payload[f'skip/clean_beta_effective_t_{tag}'] = be
        wb.log(payload, step=step)


def build_repa_teacher(args, device):
    if not args.repa:
        return None
    if args.repa_encoder not in {'dinov2_vitb14', 'dinov3_vitb16', 'dinov3_vitl16'}:
        raise ValueError(f'unsupported REPA encoder: {args.repa_encoder}')
    from transformers import AutoModel

    teacher_path = args.repa_dinov2_path if args.repa_encoder == 'dinov2_vitb14' else args.repa_dinov3_path
    teacher = AutoModel.from_pretrained(
        teacher_path,
        local_files_only=True,
        trust_remote_code=True,
    )
    teacher.eval().to(device)
    for param in teacher.parameters():
        param.requires_grad_(False)
    return teacher


def repa_cosine_loss(pred, target):
    pred = F.normalize(pred.float(), dim=-1)
    target = F.normalize(target.float(), dim=-1)
    return -(pred * target).sum(dim=-1).mean()


def repa_patch_tokens(output, encoder):
    tokens = output.last_hidden_state
    if encoder == 'dinov2_vitb14':
        return tokens[:, 1:, :]
    # DINOv3 ViT-B/L/16 HF checkpoints expose 1 CLS token + 4 register tokens.
    return tokens[:, 5:, :]


def main():
    args = parse_args()
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
        print(f"Precision: {args.precision}", flush=True)

    dataset = build_dataset(args.data_path, split='train', image_size=256)
    loader = build_loader(dataset, args.batch_size, args.num_workers, distributed, rank, world, seed=0)
    iterator = iter(loader)
    if rank == 0:
        group_kind = getattr(dataset, 'sample_group_kind', '')
        group_suffix = f' sample_group_kind={group_kind}' if group_kind else ''
        print(f'Dataset: {type(dataset).__name__} n={len(dataset)}{group_suffix}', flush=True)

    patch_noise = maybe_load_patch_noise(args.patch_noise_cov, key=args.patch_noise_key)
    if patch_noise is not None:
        patch_noise = patch_noise.to(device=device, dtype=torch.float32)
        if rank == 0:
            print(
                f'Patch noise: cov={args.patch_noise_cov} key={args.patch_noise_key} '
                f'dim={patch_noise.dim} patches={patch_noise.num_patches}',
                flush=True,
            )
    elif rank == 0:
        print('Patch noise: isotropic Gaussian', flush=True)

    model_kwargs = {'attn_backend': args.attn_backend}
    if args.repa:
        model_kwargs.update(
            repa_depth=args.repa_depth,
            repa_z_dims=(args.repa_z_dim,),
            repa_projector_dim=args.repa_projector_dim,
            repa_source=args.repa_source,
        )
    raw_model = build_model(args.model, **model_kwargs).to(device)
    repa_teacher = build_repa_teacher(args, device)
    repa_mean = repa_std = None
    if args.repa:
        repa_mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        repa_std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
        if rank == 0:
            teacher_desc = args.repa_dinov2_path if args.repa_encoder == 'dinov2_vitb14' else args.repa_dinov3_path
            print(
                f'REPA enabled: coeff={args.repa_coeff} depth={args.repa_depth} '
                f'z_dim={args.repa_z_dim} projector_dim={args.repa_projector_dim} '
                f'source={args.repa_source} teacher={teacher_desc}',
                flush=True,
            )
    named_params = list(raw_model.named_parameters())
    model_params = [p for _, p in named_params]
    ema_params = [p.detach().clone() for p in model_params]
    ema_params2 = [p.detach().clone() for p in model_params] if args.ema_decay2 > 0 else None
    if rank == 0:
        secondary = f'{args.ema_decay2:.6f}' if ema_params2 is not None else 'disabled'
        print(f'EMA decays: primary={args.ema_decay:.6f} secondary={secondary}', flush=True)
    if args.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(model_params, lr=args.lr, betas=(0.9, 0.95), weight_decay=args.weight_decay, fused=True)
    else:
        from flowmatching_lthc.optim.muon import Muon as LocalMuon, split_muon_params
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
                self.adamw = torch.optim.AdamW(aux_params, lr=aux_lr, betas=(0.9, 0.95), weight_decay=args.weight_decay, fused=True)
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
    if args.resume:
        ckpt = torch.load(args.resume, map_location='cpu')
        load_model_state(raw_model, ckpt, 'model')
        optimizer.load_state_dict(ckpt['optimizer'])
        start_step = int(ckpt['step'])
        if 'ema' in ckpt:
            ema_state = migrate_checkpoint_state(ckpt['ema'])
            for name, e in zip([n for n, _ in named_params], ema_params):
                if name in ema_state:
                    e.copy_(ema_state[name].to(device))
        if ema_params2 is not None:
            ema_state2 = migrate_checkpoint_state(ckpt.get('ema2', ckpt.get('ema', {})))
            for name, e in zip([n for n, _ in named_params], ema_params2):
                if name in ema_state2:
                    e.copy_(ema_state2[name].to(device))

    model = raw_model
    if args.compile:
        torch._dynamo.config.cache_size_limit = 128
        compile_mode = args.compile_mode
        if compile_mode == 'auto':
            compile_mode = 'default' if args.grad_accum > 1 else 'reduce-overhead'
        if rank == 0:
            print(f"torch_compile_mode={compile_mode}", flush=True)
        model = torch.compile(raw_model, mode=compile_mode, dynamic=False)
    ddp_model = DDP(model, device_ids=[local_rank]) if distributed else model

    wb = None
    if rank == 0 and args.wandb:
        import wandb
        wandb_config = dict(vars(args))
        wandb_config.update({
            'model/class': type(raw_model).__name__,
            'model/num_parameters': sum(param.numel() for param in raw_model.parameters()),
            'model/depth': getattr(raw_model, 'depth', None),
            'model/hidden_size': getattr(raw_model, 'hidden_size', None),
            'model/num_heads': getattr(raw_model, 'num_heads', None),
            'model/in_context_len': getattr(raw_model, 'in_context_len', 0),
            'model/in_context_start': getattr(raw_model, 'in_context_start', 0),
            'model/has_long_skip': hasattr(raw_model, 'skip_effective_values'),
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

    csv_path = run_dir / 'logs' / 'train_metrics.csv'
    skip_csv_path = run_dir / 'logs' / 'skip_curves.csv'
    if rank == 0 and not csv_path.exists():
        with csv_path.open('w', newline='') as f:
            csv.writer(f).writerow(['step','loss','mse_loss','repa_loss','lr','samples_per_sec','data_time','step_time','grad_norm'])
    last_logged_step = read_max_logged_step(csv_path) if rank == 0 else -1

    last = time.time()
    for step in range(start_step + 1, args.max_steps + 1):
        current_lr = lr_for_step(args.lr, step, args.warmup_steps)
        set_optimizer_lr(optimizer, current_lr)
        optimizer.zero_grad(set_to_none=True)
        data_time = 0.0
        loss_for_log = 0.0
        mse_loss_for_log = 0.0
        repa_loss_for_log = 0.0
        for accum_idx in range(args.grad_accum):
            data_start = time.time()
            try:
                x, y = next(iterator)
            except StopIteration:
                if distributed and hasattr(loader.sampler, 'set_epoch'):
                    loader.sampler.set_epoch(step)
                iterator = iter(loader)
                x, y = next(iterator)
            data_time += time.time() - data_start
            x_uint8 = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True).long()
            x01 = x_uint8.float().div(255.0)
            repa_targets = None
            if args.repa:
                repa_dtype = torch.float32 if args.precision == 'fp32' else torch.bfloat16
                with torch.no_grad(), torch.amp.autocast('cuda', dtype=repa_dtype, enabled=args.precision != 'fp32'):
                    repa_input = x01
                    if args.repa_encoder == 'dinov2_vitb14':
                        repa_input = F.interpolate(repa_input, size=(224, 224), mode='bicubic', align_corners=False)
                    repa_input = (repa_input - repa_mean) / repa_std
                    repa_targets = repa_patch_tokens(repa_teacher(pixel_values=repa_input), args.repa_encoder).detach()
                if repa_targets.shape[1] != 256 or repa_targets.shape[-1] != args.repa_z_dim:
                    raise RuntimeError(f'unexpected REPA target shape: {tuple(repa_targets.shape)}')
            x = x01.mul(2).sub(1)

            drop = torch.rand(y.shape[0], device=device) < args.label_drop_prob
            y_in = torch.where(drop, torch.full_like(y, 1000), y)
            t = torch.sigmoid(torch.randn(y.shape[0], device=device) * args.P_std + args.P_mean).view(-1,1,1,1)
            if patch_noise is not None:
                e = patch_noise.sample_like(x, noise_scale=args.noise_scale)
            else:
                e = torch.randn_like(x) * args.noise_scale
            z = t * x + (1 - t) * e
            v = (x - z) / (1 - t).clamp_min(args.t_eps)
            sync_context = ddp_model.no_sync() if distributed and accum_idx < args.grad_accum - 1 else nullcontext()
            with sync_context:
                if args.compile:
                    mark_compile_step()
                train_autocast = (
                    nullcontext()
                    if args.precision == 'fp32'
                    else torch.amp.autocast('cuda', dtype=torch.bfloat16)
                )
                with train_autocast:
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
                    mse_loss = (v - v_pred).square().mean()
                    if args.repa:
                        repa_loss = repa_cosine_loss(repa_outputs[0], repa_targets)
                        loss = mse_loss + args.repa_coeff * repa_loss
                    else:
                        repa_loss = mse_loss.new_zeros(())
                        loss = mse_loss
                loss_for_log += float(loss.detach().cpu())
                mse_loss_for_log += float(mse_loss.detach().cpu())
                repa_loss_for_log += float(repa_loss.detach().cpu())
                (loss / args.grad_accum).backward()
        grad_norm = grad_global_norm(model_params)
        optimizer.step()
        update_ema(ema_params, model_params, args.ema_decay)
        if ema_params2 is not None:
            update_ema(ema_params2, model_params, args.ema_decay2)
        torch.cuda.synchronize(device)
        now = time.time()
        step_time = now - last
        last = now
        if rank == 0 and step % args.log_every == 0 and step > last_logged_step and not (step % args.save_every == 0 or step == args.max_steps):
            sps = args.batch_size * world * args.grad_accum / max(step_time, 1e-9)
            row = [
                step,
                loss_for_log / args.grad_accum,
                mse_loss_for_log / args.grad_accum,
                repa_loss_for_log / args.grad_accum,
                optimizer.param_groups[0]['lr'],
                sps,
                data_time,
                step_time,
                grad_norm,
            ]
            with csv_path.open('a', newline='') as f:
                csv.writer(f).writerow(row)
            if wb:
                wb.log({
                    'train/loss': row[1],
                    'train/total_loss': row[1],
                    'train/mse_loss': row[2],
                    'train/repa_loss': row[3],
                    'train/repa_coeff': args.repa_coeff if args.repa else 0.0,
                    'train/lr': row[4],
                    'train/samples_per_sec': row[5],
                    'train/data_time': row[6],
                    'train/step_time': row[7],
                    'train/grad_norm': row[8],
                }, step=step)
            log_skip_curves(raw_model, device, step, wb, skip_csv_path)
            print(
                f"step={step} loss={row[1]:.5f} mse={row[2]:.5f} repa={row[3]:.5f} "
                f"grad_norm={grad_norm:.3f} sps={sps:.1f} data={data_time:.3f}s step={step_time:.3f}s",
                flush=True,
            )
            last_logged_step = step
        if rank == 0 and (step % args.save_every == 0 or step == args.max_steps):
            save_ckpt(
                run_dir / 'checkpoints' / f'step_{step:08d}.pt',
                raw_model,
                optimizer,
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
