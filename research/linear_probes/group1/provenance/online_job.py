"""Frozen online backbone and independent batched linear heads, one GPU/job."""
import argparse
import json
import math
import os
from pathlib import Path
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from job import data_module, dump, save, loss_function
from models import load, registry


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--label', required=True)
    parser.add_argument('--alpha', type=float, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--smoke-batches', type=int, default=0)
    args = parser.parse_args()
    assert args.alpha in (.25, .5, .75)
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'complete.json').exists():
        return
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.cuda.set_device(0)
    device = torch.device('cuda:0')
    entry = next(e for e in registry() if e['label'] == args.label)
    module = data_module()
    train_data = module.TrainImages()
    extractor = load(entry, device)
    encoder = module.EncoderOnly().to(device) if entry['group'] == 2 else None
    micro, effective_batch, epochs = 128, 4096, 40

    @torch.no_grad()
    def encode(images):
        images = images.to(device, non_blocking=True).float().div_(255)
        if encoder is None:
            return images.mul_(2).sub_(1)
        from stage1.rae import RAE
        return RAE.encode(encoder, images)

    # Verify extraction boundaries, then compare BOTH BF16 paths against FP32.
    with torch.no_grad():
        check = encode(torch.stack([train_data[i][0] for i in range(2)]))
        rng = torch.Generator(device=device).manual_seed(710001)
        check = (1-args.alpha)*check + args.alpha*torch.randn(
            check.shape, device=device, generator=rng)
        times = torch.full((2,), args.alpha, device=device)
        attention_modes = []
        for module_ in extractor.modules():
            for attr in ('backend', 'attn_backend'):
                if getattr(module_, attr, None) == 'flash':
                    attention_modes.append((module_, attr))
                    setattr(module_, attr, 'math')
        try:
            fp32 = extractor(check, times)
        finally:
            for module_, attr in attention_modes:
                setattr(module_, attr, 'flash')

        with torch.autocast('cuda', dtype=torch.bfloat16):
            reference = extractor.native_hook_reference(check, times)
            eager = extractor(check, times)
        torch.testing.assert_close(eager, reference, rtol=.01, atol=.005)
    active = extractor
    info = dict(mode='default', native_hook_max_abs=float((eager-reference).abs().max()))
    try:
        candidate = torch.compile(extractor, mode='default', dynamic=False)
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            compiled = candidate(check, times)
        denominator = fp32.square().mean((0, 2)).sqrt().clamp_min(1e-8)
        eager_error = (eager-fp32).square().mean((0, 2)).sqrt()/denominator
        compiled_error = (compiled-fp32).square().mean((0, 2)).sqrt()/denominator
        info.update(eager_relative_rmse=eager_error.tolist(),
                    compiled_relative_rmse=compiled_error.tolist())
        if not torch.isfinite(compiled).all() or not torch.all(compiled_error <= 1.5*eager_error+1e-4):
            raise RuntimeError('Compiled BF16 error exceeds FP32-reference error budget')
        bench = check.repeat(micro//2, 1, 1, 1)
        bt = torch.full((micro,), args.alpha, device=device)
        timing = []
        outputs = []
        for fn in (extractor, candidate):
            with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                for _ in range(2):
                    result = fn(bench, bt)
                torch.cuda.synchronize()
                began = time.monotonic()
                for _ in range(5):
                    result = fn(bench, bt)
                torch.cuda.synchronize()
                timing.append((time.monotonic()-began)/5)
                outputs.append(result)
        # Actual batch shape must retain FP32-reference accuracy as well.
        expected = fp32.repeat(micro//2, 1, 1)
        errors = [(r-expected).square().mean((0, 2)).sqrt()/denominator for r in outputs]
        info.update(batch_eager_relative_rmse=errors[0].tolist(),
                    batch_compiled_relative_rmse=errors[1].tolist())
        if not torch.all(errors[1] <= 1.5*errors[0]+1e-4):
            raise RuntimeError('Actual-batch compiled error exceeds FP32-reference error budget')
        if timing[1] < timing[0]:
            active = candidate
        info.update(eager_seconds=timing[0], compiled_seconds=timing[1],
                    selected='compiled' if active is candidate else 'eager')
        del bench, outputs, expected
    except Exception as exc:
        info.update(selected='eager', reason=repr(exc))
    dump(out/'compile.json', info)
    print(json.dumps(info), flush=True)
    dump(out/'protocol.json', dict(model=entry, alpha=args.alpha, online=True,
        points=list(range(1, 12)), condition='null', pooling='patch RMS eps1e-6 then GAP',
        epochs=epochs, microbatch=micro, effective_batch=effective_batch,
        optimizer='AdamW lr cosine .01 to .0001, wd0, default betas',
        noise='fresh CUDA normal per epoch; shared seed and shuffle across models/t',
        train_seed_base=700001, validation_seeds=[800001,800002,800003],
        backbone='BF16 autocast', encoder='FP32', heads='FP32',
        checkpoint_state='model', cache=False, wandb=False))
    w = torch.nn.Parameter(torch.zeros(11, 768, 1000, device=device))
    b = torch.nn.Parameter(torch.zeros(11, 1, 1000, device=device))
    opt = torch.optim.AdamW([w, b], lr=.01, weight_decay=0., foreach=False)
    start, history = 0, []
    if (out/'latest.pt').exists():
        state = torch.load(out/'latest.pt', map_location='cpu', weights_only=True)
        with torch.no_grad():
            w.copy_(state['weight']); b.copy_(state['bias'])
        opt.load_state_dict(state['optimizer'])
        start, history = state['epoch'], state['history']
    shuffle = torch.Generator()
    loader = DataLoader(train_data, batch_size=micro,
        sampler=torch.utils.data.RandomSampler(train_data, generator=shuffle),
        num_workers=4, persistent_workers=True, pin_memory=True, prefetch_factor=4)
    for epoch in range(start, epochs):
        shuffle.manual_seed(42+epoch)
        rng = torch.Generator(device=device).manual_seed(700001+epoch)
        lr = .0001 + (.01-.0001)*.5*(1+math.cos(math.pi*epoch/39))
        for group in opt.param_groups:
            group['lr'] = lr
        began = time.monotonic()
        total = torch.zeros(11, device=device)
        seen, accumulated, updates = 0, 0, 0
        opt.zero_grad(set_to_none=True)
        for batch_index, (images, labels) in enumerate(loader):
            with torch.no_grad():
                x = encode(images)
                z = (1-args.alpha)*x + args.alpha*torch.randn(x.shape, device=device, generator=rng)
                at = torch.full((len(x),), args.alpha, device=device)
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    features = active(z, at).float().detach()
            losses = loss_function(features.transpose(0, 1).contiguous(), w, b,
                                   labels.to(device, non_blocking=True))
            # Only the small heads retain autograd; full backbone stays frozen.
            denominator = min(effective_batch, len(train_data)-(seen-accumulated))
            (losses.sum()*len(labels)/denominator).backward()
            total += losses.detach()*len(labels)
            seen += len(labels)
            accumulated += len(labels)
            if accumulated == effective_batch or seen == len(train_data):
                if not torch.isfinite(total).all():
                    raise RuntimeError('Nonfinite probe loss')
                opt.step(); opt.zero_grad(set_to_none=True)
                accumulated = 0; updates += 1
            if (batch_index+1) % 64 == 0 or seen == len(train_data) or args.smoke_batches:
                torch.cuda.synchronize()
                elapsed = time.monotonic()-began
                row = dict(epoch=epoch+1, images=seen, total=len(train_data),
                    updates=updates, seconds=elapsed, images_per_second=seen/elapsed,
                    loss=(total/seen).tolist())
                dump(out/'progress.json', row)
                print(json.dumps(row), flush=True)
            if args.smoke_batches and batch_index+1 >= args.smoke_batches:
                if not torch.isfinite(w.grad).all():
                    raise RuntimeError('Nonfinite smoke gradients')
                dump(out/'smoke_complete.json', row)
                return
        row.update(lr=lr)
        history.append(row)
        save(out/'latest.pt', dict(epoch=epoch+1, weight=w.detach().cpu(),
            bias=b.detach().cpu(), optimizer=opt.state_dict(), history=history))
    del loader
    validation = module.ValidationImages()
    val_loader = DataLoader(validation, batch_size=micro, num_workers=4,
        persistent_workers=True, pin_memory=True, prefetch_factor=4)
    rows = []
    for view in range(4):
        rng = torch.Generator(device=device).manual_seed(800000+view)
        counts = torch.zeros(11, 2, device=device, dtype=torch.float64)
        with torch.no_grad():
            for images, labels in val_loader:
                x = encode(images)
                a = args.alpha if view else 0.
                z = (1-a)*x+a*torch.randn(x.shape, device=device, generator=rng)
                at = torch.full((len(x),), a, device=device)
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    features = active(z, at).float()
                logits = torch.bmm(features.transpose(0, 1).contiguous(), w)+b
                top = logits.topk(5, -1).indices
                y = labels.to(device)[None]
                counts[:,0] += (top[:,:,0] == y).sum(1)
                counts[:,1] += (top == y[:,:,None]).any(-1).sum(1)
        rows.append((counts/len(validation)*100).cpu())
        dump(out/f'validation_view{view}.json', rows[-1].tolist())
    accuracy = torch.stack(rows)
    summary = []
    for layer in range(11):
        for mode, indices in [('clean',[0]), ('matched_noise',[1,2,3])]:
            selected = accuracy[indices,layer]
            summary.append(dict(after_block=layer+1, mode=mode,
                top1=float(selected[:,0].mean()), top5=float(selected[:,1].mean()),
                top1_seed_std=float(selected[:,0].std(unbiased=False)),
                top5_seed_std=float(selected[:,1].std(unbiased=False))))
    dump(out/'accuracy.json', summary)
    dump(out/'complete.json', dict(epochs=epochs, test_images=len(validation)))


if __name__ == '__main__':
    main()
