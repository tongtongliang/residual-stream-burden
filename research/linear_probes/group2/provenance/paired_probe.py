"""Two-GPU cached probes: preserve global microbatch order and CUDA noise stream."""
import argparse
import json
import math
import os
from pathlib import Path
import time

import torch
import torch.distributed as dist
from job import dump, save, loss_function
from models import registry, load
from latent_cache import CACHE, CachedLatents, SHAPE
from ring_latent_loader import RingLatentLoader

MICRO, GLOBAL = 128, 4096


def local_indices(order, rank, world):
    return [index for start in range(0, len(order), MICRO)
            if (start//MICRO)%world == rank for index in order[start:start+MICRO]]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--label', required=True)
    parser.add_argument('--alpha', type=float, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--resume', type=Path, required=True)
    parser.add_argument('--smoke-steps', type=int, default=0)
    args = parser.parse_args()
    rank, world, local = [int(os.environ[x]) for x in ('RANK','WORLD_SIZE','LOCAL_RANK')]
    assert world == 2
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.cuda.set_device(local)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device('cuda', local)
    dist.init_process_group('nccl')
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    entry = next(e for e in registry() if e['label'] == args.label)
    assert entry['group'] == 2 and args.alpha in (.25,.5,.75)
    data = CachedLatents(CACHE, 'train')
    model = load(entry, device)
    active = torch.compile(model, mode='default', dynamic=False)
    state = torch.load(args.resume, map_location='cpu', weights_only=True)
    w = torch.nn.Parameter(state['weight'].to(device))
    b = torch.nn.Parameter(state['bias'].to(device))
    optimizer = torch.optim.AdamW([w,b], lr=.01, weight_decay=0., foreach=False)
    optimizer.load_state_dict(state['optimizer'])
    start, history = state['epoch'], state['history']
    metadata = dict(start_epoch=start, world=world, global_batch=GLOBAL, microbatch=MICRO,
        checkpoint=str(args.resume), label=args.label, alpha=args.alpha, cache=str(CACHE),
        shuffle='same global randperm seed42+epoch; assign whole microbatches round-robin',
        noise='both ranks advance same seed700001+epoch through ALL global microbatches',
        gradients='local loss sum/actual global examples; SUM reduce once per optimizer update',
        numerical='same inputs/noise; gradient summation order differs from single GPU',
        backbone='BF16 compiled frozen', heads='FP32 resumed AdamW', wandb=False)
    if rank == 0:
        dump(out/'paired_resume.json', metadata)
        dump(out/'distributed_resume.json', metadata)
        print(json.dumps(metadata), flush=True)
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        warm = torch.stack([data[0][0]]*MICRO).to(device)
        active(warm, torch.full((MICRO,), args.alpha, device=device))
    torch.cuda.synchronize()
    dist.barrier()
    for epoch in range(start, 40):
        order = torch.randperm(len(data), generator=torch.Generator().manual_seed(42+epoch)).tolist()
        loader = RingLatentLoader(data, sampler=local_indices(order,rank,world), device=device)
        iterator = iter(loader)
        rng = torch.Generator(device=device).manual_seed(700001+epoch)
        lr = .0001+(.01-.0001)*.5*(1+math.cos(math.pi*epoch/39))
        for group in optimizer.param_groups:
            group['lr'] = lr
        total = torch.zeros(11, device=device)
        began = time.monotonic()
        for step, base in enumerate(range(0,len(data),GLOBAL)):
            count = min(GLOBAL,len(data)-base)
            optimizer.zero_grad(set_to_none=True)
            for offset in range(base,base+count,MICRO):
                n = min(MICRO,len(data)-offset)
                # Advancing on skipped microbatches preserves the original CUDA RNG
                # offsets exactly without copying inputs or evaluating the backbone.
                noise = torch.randn((n,*SHAPE), device=device, generator=rng)
                if (offset//MICRO)%world != rank:
                    continue
                x,y = next(iterator)
                assert len(x) == n
                with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                    z = (1-args.alpha)*x+args.alpha*noise
                    f = active(z,torch.full((n,),args.alpha,device=device)).float().detach()
                losses = loss_function(f.transpose(0,1).contiguous(),w,b,y)
                (losses.sum()*n/count).backward()
                total += losses.detach()*n
            dist.all_reduce(w.grad, op=dist.ReduceOp.SUM)
            dist.all_reduce(b.grad, op=dist.ReduceOp.SUM)
            if not torch.isfinite(w.grad).all() or not torch.isfinite(b.grad).all():
                raise RuntimeError('Nonfinite paired probe gradients')
            optimizer.step()
            if (step+1)%20 == 0 or base+count == len(data) or args.smoke_steps:
                summed = total.clone()
                dist.all_reduce(summed)
                torch.cuda.synchronize()
                elapsed = time.monotonic()-began
                row = dict(epoch=epoch+1,images=base+count,total=len(data),updates=step+1,
                    seconds=elapsed,images_per_second=(base+count)/elapsed,world=world,
                    loss=(summed/(base+count)).tolist(),lr=lr)
                if rank == 0:
                    dump(out/'progress.json',row)
                    print(json.dumps(row),flush=True)
            if args.smoke_steps and step+1 >= args.smoke_steps:
                low,high = w.detach().clone(),w.detach().clone()
                dist.all_reduce(low,op=dist.ReduceOp.MIN)
                dist.all_reduce(high,op=dist.ReduceOp.MAX)
                assert torch.equal(low,high)
                assert all(p.grad is None for p in model.parameters())
                if rank == 0:
                    save(out/'smoke_state.pt',dict(weight=w.detach().cpu(),bias=b.detach().cpu()))
                    dump(out/'smoke_complete.json',dict(**row,replicas_equal=True))
                iterator.close()
                dist.destroy_process_group()
                return
        iterator.close()
        history.append(row)
        if rank == 0:
            save(out/'latest.pt',dict(epoch=epoch+1,weight=w.detach().cpu(),bias=b.detach().cpu(),
                                     optimizer=optimizer.state_dict(),history=history))
        dist.barrier()
    validation = CachedLatents(CACHE,'val')
    ids = local_indices(list(range(len(validation))),rank,world)
    views = []
    for view in range(4):
        iterator = iter(RingLatentLoader(validation,sampler=ids,device=device))
        rng = torch.Generator(device=device).manual_seed(800000+view)
        alpha = args.alpha if view else 0.
        counts = torch.zeros(11,2,device=device,dtype=torch.float64)
        with torch.no_grad():
            for offset in range(0,len(validation),MICRO):
                n = min(MICRO,len(validation)-offset)
                noise = torch.randn((n,*SHAPE),device=device,generator=rng)
                if (offset//MICRO)%world != rank:
                    continue
                x,y = next(iterator)
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    f = active((1-alpha)*x+alpha*noise,torch.full((n,),alpha,device=device)).float()
                logits = torch.bmm(f.transpose(0,1).contiguous(),w)+b
                top = logits.topk(5,-1).indices
                counts[:,0] += (top[:,:,0] == y[None]).sum(1)
                counts[:,1] += (top == y[None,:,None]).any(-1).sum(1)
        iterator.close()
        dist.all_reduce(counts)
        views.append((counts/len(validation)*100).cpu())
        if rank == 0:
            dump(out/f'validation_view{view}.json',views[-1].tolist())
    if rank == 0:
        values = torch.stack(views)
        rows = []
        for layer in range(11):
            for mode,indices in [('clean',[0]),('matched_noise',[1,2,3])]:
                v = values[indices,layer]
                rows.append(dict(after_block=layer+1,mode=mode,top1=float(v[:,0].mean()),
                    top5=float(v[:,1].mean()),top1_seed_std=float(v[:,0].std(unbiased=False)),
                    top5_seed_std=float(v[:,1].std(unbiased=False))))
        dump(out/'accuracy.json',rows)
        dump(out/'complete.json',dict(epochs=40,test_images=len(validation),world=world))
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
