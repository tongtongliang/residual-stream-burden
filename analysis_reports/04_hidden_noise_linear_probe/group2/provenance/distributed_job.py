"""Data-parallel frozen SiHC: SUM globally normalized probe gradients only."""
import argparse
import json
import math
import os
from pathlib import Path
import time

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from job import data_module, dump, save, loss_function
from models import registry, load


class GlobalBatchShard:
    def __init__(self, size, rank, world, batch=4096):
        self.size, self.rank, self.world, self.batch = size, rank, world, batch
        self.epoch = 0

    def __iter__(self):
        order = torch.randperm(self.size, generator=torch.Generator().manual_seed(42+self.epoch)).tolist()
        for start in range(0, self.size, self.batch):
            yield order[start:start+self.batch][self.rank::self.world]

    def __len__(self):
        return math.ceil(self.size/self.batch)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--resume', type=Path, required=True)
    p.add_argument('--alpha', type=float, default=.75)
    p.add_argument('--smoke-steps', type=int, default=0)
    args = p.parse_args()
    rank, world = int(os.environ['RANK']), int(os.environ['WORLD_SIZE'])
    local = int(os.environ['LOCAL_RANK'])
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.cuda.set_device(local)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device('cuda', local)
    dist.init_process_group('nccl')
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    data = data_module()
    train_data = data.TrainImages()
    entry = next(e for e in registry() if e['label'] == 'sihc_sublayer_p4')
    model = load(entry, device)
    # Same extractor/compile mode already checked against native hooks and FP32.
    active = torch.compile(model, mode='default', dynamic=False)
    state = torch.load(args.resume, map_location='cpu', weights_only=True)
    w = torch.nn.Parameter(state['weight'].to(device))
    b = torch.nn.Parameter(state['bias'].to(device))
    optimizer = torch.optim.AdamW([w, b], lr=.01, weight_decay=0., foreach=False)
    optimizer.load_state_dict(state['optimizer'])
    start, history = state['epoch'], state['history']
    if rank == 0:
        metadata = dict(start_epoch=start, world=world, global_batch=4096, microbatch=128,
            resume=str(args.resume), alpha=args.alpha, optimizer='resumed AdamW',
            gradient='local loss SUM / actual global batch; all_reduce SUM, no world division',
            shuffle='same seed42+epoch global permutation; each example exactly once',
            noise='fresh CUDA Gaussian, seed700001+epoch+rank*10000000; not bitwise single-GPU noise',
            scope='SiHC only, Group2 remains paused', smoke_steps=args.smoke_steps)
        dump(out/'distributed_resume.json', metadata)
        print(json.dumps(metadata), flush=True)

    @torch.no_grad()
    def features(images, alpha, rng):
        x = images.to(device, non_blocking=True).float().div_(255).mul_(2).sub_(1)
        z = (1-alpha)*x+alpha*torch.randn(x.shape, device=device, generator=rng)
        n = len(z)
        if n < 128:
            z = torch.cat((z, z[-1:].expand(128-n, *z.shape[1:])))
        times = torch.full((128,), alpha, device=device)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            return active(z, times).float()[:n]

    # Compile outside epoch timing, including the input layout used in training.
    warm = torch.stack([train_data[0][0]]*128)
    features(warm, args.alpha, torch.Generator(device=device).manual_seed(42))
    torch.cuda.synchronize()
    dist.barrier()
    sampler = GlobalBatchShard(len(train_data), rank, world)
    loader = DataLoader(train_data, batch_sampler=sampler, num_workers=4,
        persistent_workers=True, pin_memory=True, prefetch_factor=2)
    for epoch in range(start, 40):
        sampler.epoch = epoch
        rng = torch.Generator(device=device).manual_seed(700001+epoch+rank*10000000)
        lr = .0001+(.01-.0001)*.5*(1+math.cos(math.pi*epoch/39))
        for group in optimizer.param_groups:
            group['lr'] = lr
        total = torch.zeros(11, device=device)
        began = time.monotonic()
        for step, (images, labels) in enumerate(loader):
            global_n = min(4096, len(train_data)-step*4096)
            optimizer.zero_grad(set_to_none=True)
            for offset in range(0, len(images), 128):
                y = labels[offset:offset+128].to(device, non_blocking=True)
                f = features(images[offset:offset+128], args.alpha, rng)
                losses = loss_function(f.transpose(0, 1).contiguous(), w, b, y)
                (losses.sum()*len(y)/global_n).backward()
                total += losses.detach()*len(y)
            dist.all_reduce(w.grad, op=dist.ReduceOp.SUM)
            dist.all_reduce(b.grad, op=dist.ReduceOp.SUM)
            if not torch.isfinite(w.grad).all() or not torch.isfinite(b.grad).all():
                raise RuntimeError('Nonfinite synchronized head gradients')
            optimizer.step()
            if (step+1)%20 == 0 or step+1 == len(sampler) or args.smoke_steps:
                totals = total.clone()
                dist.all_reduce(totals)
                torch.cuda.synchronize()
                count = min((step+1)*4096, len(train_data))
                elapsed = time.monotonic()-began
                row = dict(epoch=epoch+1, images=count, total=len(train_data), updates=step+1,
                    seconds=elapsed, images_per_second=count/elapsed, world=world,
                    loss=(totals/count).tolist())
                if rank == 0:
                    dump(out/'progress.json', row)
                    print(json.dumps(row), flush=True)
            if args.smoke_steps and step+1 >= args.smoke_steps:
                probe_sum = w.detach().double().sum()
                low, high = probe_sum.clone(), probe_sum.clone()
                dist.all_reduce(low, op=dist.ReduceOp.MIN)
                dist.all_reduce(high, op=dist.ReduceOp.MAX)
                assert float(high-low) == 0., 'Probe replicas diverged'
                assert all(param.grad is None for param in model.parameters())
                if rank == 0:
                    dump(out/'smoke_complete.json', dict(**row, replica_sum_difference=float(high-low)))
                dist.destroy_process_group()
                return
        row.update(lr=lr)
        history.append(row)
        if rank == 0:
            save(out/'latest.pt', dict(epoch=epoch+1, weight=w.detach().cpu(), bias=b.detach().cpu(),
                optimizer=optimizer.state_dict(), history=history))
        dist.barrier()
    del loader
    val = data.ValidationImages()
    val_loader = DataLoader(val, sampler=range(rank, len(val), world), batch_size=128,
        num_workers=4, persistent_workers=True, pin_memory=True, prefetch_factor=2)
    accuracies = []
    for view in range(4):
        rng = torch.Generator(device=device).manual_seed(800000+view+rank*10000000)
        counts = torch.zeros(11, 2, device=device, dtype=torch.float64)
        with torch.no_grad():
            for images, labels in val_loader:
                f = features(images, args.alpha if view else 0., rng)
                logits = torch.bmm(f.transpose(0, 1).contiguous(), w)+b
                top = logits.topk(5, -1).indices
                y = labels.to(device)[None]
                counts[:, 0] += (top[:, :, 0] == y).sum(1)
                counts[:, 1] += (top == y[:, :, None]).any(-1).sum(1)
        dist.all_reduce(counts)
        accuracies.append((counts/len(val)*100).cpu())
        if rank == 0:
            dump(out/f'validation_view{view}.json', accuracies[-1].tolist())
    if rank == 0:
        accuracy = torch.stack(accuracies)
        rows = []
        for layer in range(11):
            for mode, indices in [('clean', [0]), ('matched_noise', [1, 2, 3])]:
                values = accuracy[indices, layer]
                rows.append(dict(after_block=layer+1, mode=mode,
                    top1=float(values[:,0].mean()), top5=float(values[:,1].mean()),
                    top1_seed_std=float(values[:,0].std(unbiased=False)),
                    top5_seed_std=float(values[:,1].std(unbiased=False))))
        dump(out/'accuracy.json', rows)
        dump(out/'complete.json', dict(epochs=40, test_images=len(val), world=world))
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
