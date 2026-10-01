"""JiT B/L/H under the fusion study protocol. Results stay outside public source."""
import argparse, contextlib, datetime, gc, json, os, statistics, sys, tempfile, time, traceback
from pathlib import Path
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.attention import sdpa_kernel, SDPBackend

p = argparse.ArgumentParser()
p.add_argument('--size', choices=['B', 'L', 'H'], required=True)
p.add_argument('--phase', choices=['train', 'inference'], required=True)
p.add_argument('--repeat', type=int, default=0)
p.add_argument('--batch', type=int, default=128)
p.add_argument('--warmup', type=int, default=8)
p.add_argument('--windows', type=int, default=5)
p.add_argument('--steps', type=int, default=10)
p.add_argument('--jit-root', type=Path, required=True)
p.add_argument('--output-dir', type=Path, required=True)
a = p.parse_args()
root = a.output_dir.resolve()
root.mkdir(parents=True, exist_ok=True)
for directory in ('results', 'validation_outputs'):
    (root / directory).mkdir(exist_ok=True)
name = f'JIT_{a.size}_{a.phase}_flash_r{a.repeat}_b{a.batch}'
report = dict(
    case=name, family='jit', frequency='plain', implementation='official_flash',
    size=a.size, phase=a.phase, repeat=a.repeat, batch=a.batch,
    timestamp_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    pid=os.getpid(), physical_gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),
    torch=str(torch.__version__), cuda=torch.version.cuda,
    dtype='FP32 parameters and input; BF16 autocast', input_size=256, ctx=32,
    recompute='none', repa=False, mode='default' if a.phase == 'train' else 'reduce-overhead',
              fullgraph=False, dynamic=False, seed=12345, tf32=True, attention='flash',
              allocator=os.environ.get('PYTORCH_ALLOC_CONF'),
    ddp='1-rank compile-then-DDP' if a.phase == 'train' else None,
    optimizer=('AdamW fused=True lr=2e-4 betas=(0.9,0.95) weight_decay=0' if a.phase == 'train' else None),
    ema_decays=[.9999, .9996] if a.phase == 'train' else [],
    jit_root=str(a.jit_root),
)
stage = 'setup'


def mark(value):
    global stage
    stage = value
    print(json.dumps(dict(stage=value, case=name, timestamp_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())), flush=True)


def memory():
    stats = torch.cuda.memory_stats()
    return dict(
        allocated_gib=torch.cuda.memory_allocated() / 2**30,
        reserved_gib=torch.cuda.memory_reserved() / 2**30,
        peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
        peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30,
        active_gib=stats.get('active_bytes.all.current', 0) / 2**30,
        free_gib=torch.cuda.mem_get_info()[0] / 2**30,
        device_used_gib=(torch.cuda.mem_get_info()[1] - torch.cuda.mem_get_info()[0]) / 2**30,
    )


try:
    assert report['allocator']=='expandable_segments:True'
    from bench.fusion_observer import observe_ddp
    observe_ddp(report)
    torch.set_num_threads(4)
    torch.manual_seed(12345)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.cuda.set_device(0)
    report['gpu'] = torch.cuda.get_device_name()
    report['initial_memory'] = memory()
    sys.path.insert(0, str(a.jit_root))
    import model_jit
    for cls in (model_jit.JiTBlock, model_jit.FinalLayer):
        original = getattr(cls.forward, '_torchdynamo_orig_callable', None)
        if original is not None:
            cls.forward = original
    model_jit.scaled_dot_product_attention = torch.nn.functional.scaled_dot_product_attention
    raw = model_jit.JiT_models[f'JiT-{a.size}/16']()
    spec = dict(
        depth=len(raw.blocks), hidden_size=raw.hidden_size, patch_size=16,
        in_context_start=raw.in_context_start, in_context_len=raw.in_context_len,
        bottleneck_dim=raw.x_embedder.proj1.out_channels,
    )
    with torch.no_grad():
        for name_mod, module in raw.named_modules():
            if isinstance(module, torch.nn.Linear) and (
                'adaLN_modulation.1' in name_mod or name_mod == 'final_layer.linear'
            ):
                torch.nn.init.normal_(module.weight, std=.001)
    report.update(spec=spec, parameters=sum(q.numel() for q in raw.parameters()))
    raw = raw.cuda().train(a.phase == 'train')
    if a.phase == 'inference':
        raw.requires_grad_(False)
    trainable = [q for q in raw.parameters() if q.requires_grad]
    x = torch.randn(a.batch, 3, 256, 256, device='cuda')
    t = torch.full((a.batch,), .5, device='cuda')
    y = torch.arange(a.batch, device='cuda') % 1000
    target = torch.randn_like(x) if a.phase == 'train' else None
    def flash():
        return sdpa_kernel(SDPBackend.FLASH_ATTENTION)
    mark('eager_reference_batch1')
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16), flash():
        reference = raw(x[:1], t[:1], y[:1]).float().clone()
    if a.phase == 'train':
        optimizer = torch.optim.AdamW(trainable, lr=2e-4, betas=(.9, .95), weight_decay=0, fused=True)
        ema = [[q.detach().clone() for q in trainable] for _ in range(2)]
    else:
        optimizer, ema = None, []
    gc.collect(); torch.cuda.empty_cache()
    report['persistent_memory_before_compile'] = memory()
    torch.cuda.reset_peak_memory_stats()
    torch._dynamo.config.cache_size_limit = 128
    compiled = torch.compile(raw, mode=report['mode'], dynamic=False)
    runner = compiled
    if a.phase == 'train':
        store = tempfile.NamedTemporaryFile(prefix=f'jit_ddp_{os.getpid()}_', suffix='.store', delete=False)
        store.close()
        dist.init_process_group('nccl', init_method=f'file://{store.name}', rank=0, world_size=1)
        runner = DDP(compiled, device_ids=[0], gradient_as_bucket_view=False)
        report['ddp_gradient_as_bucket_view'] = False
    timings = []
    last_loss = None
    def mark_compile_step():
        if hasattr(torch, 'compiler') and hasattr(torch.compiler, 'cudagraph_mark_step_begin'):
            torch.compiler.cudagraph_mark_step_begin()
    def step(*, validate=False, timed=False):
        global last_loss
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        if a.phase == 'train':
            mark_compile_step()
        if timed:
            start_event = torch.cuda.Event(enable_timing=True)
            fb_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
        with torch.amp.autocast('cuda', dtype=torch.bfloat16), flash():
            output = runner(x, t, y)
            if optimizer is not None:
                per_sample = (output.float() - target).square().flatten(1).mean(1)
                loss = per_sample.mean()
        if optimizer is not None:
            loss.backward()
            if timed:
                fb_event.record()
            if validate:
                missing = [n for n, q in raw.named_parameters() if q.requires_grad and q.grad is None]
                assert not missing, missing
                assert all(torch.isfinite(q.grad).all() for q in trainable), 'Nonfinite gradient'
                report['initial_loss'] = loss.item()
                report['parameters_with_grad'] = len(trainable)
            optimizer.step()
            with torch.no_grad():
                for copies, decay in zip(ema, (.9999, .9996)):
                    torch._foreach_mul_(copies, decay)
                    torch._foreach_add_(copies, trainable, alpha=1 - decay)
            last_loss = loss.detach()
        elif timed:
            fb_event.record()
        if validate:
            candidate = output[:1].float()
            delta = candidate - reference
            report['compiled_eager_batch1_max_abs'] = delta.abs().max().item()
            report['compiled_eager_batch1_relative_l2'] = (delta.norm() / reference.norm().clamp_min(1e-12)).item()
            assert torch.isfinite(output).all(), 'Nonfinite compiled output'
            torch.testing.assert_close(candidate, reference, atol=.025, rtol=.03)
            torch.save(candidate.detach().cpu(), root/'validation_outputs'/f'{name}.pt')
            del candidate, delta
        if timed:
            end_event.record()
            timings.append((start_event, fb_event, end_event))
        del output
        if optimizer is not None:
            del loss, per_sample

    grad_context = contextlib.nullcontext() if a.phase == 'train' else torch.inference_mode()
    with grad_context:
        mark('compile_and_first_step')
        started = time.perf_counter()
        step(validate=True)
        torch.cuda.synchronize()
        report['first_step_seconds_including_compile'] = time.perf_counter() - started
        report['first_step_memory'] = memory()
        del reference
        mark('warmup')
        torch.cuda.reset_peak_memory_stats()
        for _ in range(a.warmup):
            step()
        torch.cuda.synchronize()
        report['warmup_memory'] = memory()
        mark('measure')
        torch.cuda.reset_peak_memory_stats()
        windows = []
        for window in range(a.windows):
            torch.cuda.synchronize()
            start = time.perf_counter()
            before = len(timings)
            for _ in range(a.steps):
                step(timed=True)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            samples = [dict(model_forward_backward_ms=s.elapsed_time(fb), total_step_ms=s.elapsed_time(e))
                       for s, fb, e in timings[before:]]
            windows.append(dict(window=window, wall_ms_per_step=elapsed * 1000 / a.steps,
                                images_per_second=a.batch * a.steps / elapsed, samples=samples))
        report['steady_memory'] = memory()
        report['windows_data'] = windows
        report['step_wall_ms_median'] = statistics.median(w['wall_ms_per_step'] for w in windows)
        report['images_per_second_median'] = statistics.median(w['images_per_second'] for w in windows)
        report['gpu_step_ms_median'] = statistics.median(s['total_step_ms'] for w in windows for s in w['samples'])
        report['gpu_model_fb_ms_median'] = statistics.median(s['model_forward_backward_ms'] for w in windows for s in w['samples'])
        if last_loss is not None:
            report['final_loss'] = last_loss.item()
            assert torch.isfinite(last_loss), 'Nonfinite final loss'
            assert all(torch.isfinite(q.grad).all() for q in trainable), 'Nonfinite final gradient'
        report['runtime_peak_allocated_gib'] = max(
            report[k]['peak_allocated_gib'] for k in ('first_step_memory', 'warmup_memory', 'steady_memory')
        )
        report['steady_peak_reserved_gib'] = report['steady_memory']['peak_reserved_gib']
        report['runtime_peak_note'] = (
            'Includes first execution/graph setup; first-step peak may include compilation allocations. '
            'Steady metrics reported separately.'
        )
        report['dynamo_counters'] = {k: dict(v) for k, v in torch._dynamo.utils.counters.items()}
        report['status'] = 'ok'
except Exception as error:
    message = str(error)
    is_oom = isinstance(error, torch.OutOfMemoryError) or 'out of memory' in message.lower()
    report.update(
        status='oom' if is_oom else 'error', failed_stage=stage,
        exception_type=type(error).__name__, error=message, traceback=traceback.format_exc(),
    )
    try:
        report['failure_memory'] = memory()
    except Exception:
        pass
    print(traceback.format_exc(), flush=True)
finally:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()
    report['finished_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    destination = root / 'results' / f'{name}.json'
    temporary = destination.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(report, indent=2, default=str) + '\n')
    temporary.replace(destination)
    print(json.dumps({k: report[k] for k in (
        'case', 'status', 'failed_stage', 'step_wall_ms_median',
        'runtime_peak_allocated_gib', 'steady_peak_reserved_gib',
    ) if k in report}), flush=True)
sys.exit(0 if report.get('status') in ('ok', 'oom') else 1)
