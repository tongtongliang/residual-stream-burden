"""Eight-GPU online normalized RAE DINOv2-B noise-specific GAP probes."""
import bisect
import csv
import json
import math
import os
from pathlib import Path
import sys
import time
from collections import OrderedDict
from datetime import datetime, timezone

os.environ.update(WANDB_MODE='disabled', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

ROOT = Path(__file__).resolve().parent
RAW = Path('/data/tongtong/flowmatching_shc_work/rebuildable/imagenet256_tensor_cache_full')
VAL = Path('/data/tongtong/RAEv2/data/imagenet-256/imagenet-latents-images/val')
MODEL = Path('/data/tongtong/sihc_artifacts/teachers/dinov2-with-registers-base')
STATS = ROOT / 'models/stats/dinov2/wReg_base/imagenet1k/stat.pt'
sys.path.insert(0, str(ROOT / 'RAE/src'))
TIMES = (0., .25, .5, .75)
EPOCHS, BATCH, MICRO, SEED = 40, 4096, 64, 42


def dump(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def save(path, value):
    tmp = path.with_suffix('.partial.pt')
    torch.save(value, tmp)
    tmp.replace(path)



class TrainImages(Dataset):
    def __init__(self):
        self.manifest = json.loads((RAW / 'manifest.json').read_text())
        assert self.manifest['layout'] == 'chw_uint8_adm_center_crop_no_flip'
        self.shards = self.manifest['shards']
        self.ends = np.cumsum([s['count'] for s in self.shards]).tolist()
        self.labels = np.concatenate([np.load(RAW / s['labels']) for s in self.shards])
        assert len(self.labels) == 1281167 and self.labels.min() == 0 and self.labels.max() == 999
        self.maps = OrderedDict()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        i = int(i)
        shard = bisect.bisect_right(self.ends, i)
        if shard not in self.maps:
            self.maps[shard] = np.load(RAW / self.shards[shard]['images'], mmap_mode='r')
            if len(self.maps) > 32:
                self.maps.popitem(last=False)
        self.maps.move_to_end(shard)
        offset = i - (self.ends[shard-1] if shard else 0)
        return torch.from_numpy(self.maps[shard][offset].copy()), int(self.labels[i])


class ValidationImages(Dataset):
    def __init__(self):
        from datasets import load_from_disk
        self.data = load_from_disk(str(VAL))
        assert len(self.data) == 50000
        np.testing.assert_array_equal(np.bincount(self.data['label'], minlength=1000), np.full(1000, 50))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        # Same ADM center crop as the DINOv3 probe's validation preprocessing.
        from PIL import Image
        row = self.data[int(i)]
        im = row['image'].convert('RGB')
        while min(im.size) >= 512:
            im = im.resize(tuple(x // 2 for x in im.size), resample=Image.Resampling.BOX)
        scale = 256 / min(im.size)
        im = im.resize(tuple(round(x * scale) for x in im.size), resample=Image.Resampling.BICUBIC)
        arr = np.array(im)
        y, x = (arr.shape[0]-256)//2, (arr.shape[1]-256)//2
        return torch.from_numpy(arr[y:y+256, x:x+256].copy()).permute(2,0,1), int(row['label'])


class EncoderOnly(torch.nn.Module):
    def __init__(self):
        super().__init__()
        from stage1.encoders.dinov2 import Dinov2withNorm
        from transformers import AutoImageProcessor
        self.encoder = Dinov2withNorm(str(MODEL), normalize=True)
        proc = AutoImageProcessor.from_pretrained(str(MODEL), local_files_only=True)
        self.register_buffer('encoder_mean', torch.tensor(proc.image_mean).view(1,3,1,1))
        self.register_buffer('encoder_std', torch.tensor(proc.image_std).view(1,3,1,1))
        stats = torch.load(STATS, map_location='cpu', weights_only=True)
        assert stats['var'] is not None
        self.register_buffer('latent_mean', stats['mean'])
        self.register_buffer('latent_var', stats['var'])
        self.encoder_input_size = 224
        self.reshape_to_2d, self.do_normalization = True, True
        self.noise_tau, self.eps = 0., 1e-5
        self.requires_grad_(False).eval()

    @torch.no_grad()
    def forward(self, images):
        from stage1.rae import RAE
        z = RAE.encode(self, images.float()/255.)
        assert z.shape[1:] == (768,16,16) and z.dtype == torch.float32
        return z.mean((2,3))


def noised(x, generator):
    t = x.new_tensor(TIMES).view(4,1,1)
    return (1-t)*x[None] + t/16*torch.randn((4,len(x),768), device=x.device, generator=generator)


def ce(logits, labels):
    return F.cross_entropy(logits.flatten(0,1), labels.repeat(4), reduction='none').view(4,-1)


def main():
    rank, world = int(os.environ['RANK']), int(os.environ['WORLD_SIZE'])
    assert world == 8
    torch.set_num_threads(2)
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    device = torch.device('cuda', int(os.environ['LOCAL_RANK']))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    dist.init_process_group('nccl')
    torch.manual_seed(SEED)
    data = TrainImages()
    encoder = EncoderOnly().to(device)
    weight = torch.nn.Parameter(torch.zeros(4,768,1000, device=device))
    bias = torch.nn.Parameter(torch.zeros(4,1,1000, device=device))
    opt = torch.optim.AdamW([weight,bias], lr=.01, weight_decay=0., foreach=False)
    results = ROOT / 'results'
    results.mkdir(exist_ok=True)
    start_epoch, history = 0, []
    if (results/'latest.pt').exists():
        state = torch.load(results/'latest.pt', map_location='cpu', weights_only=True)
        assert state['times'] == TIMES and state['epochs'] == EPOCHS and state['world'] == world
        with torch.no_grad():
            weight.copy_(state['weight']); bias.copy_(state['bias'])
        opt.load_state_dict(state['optimizer'])
        start_epoch, history = state['epoch'], state['history']
    if rank == 0:
        dump(ROOT/'protocol.json', dict(encoder='official RAE Dinov2withNorm, with-registers-base, normalize=True',
            shape=[768,16,16], encoder_resize=224,
            train_images=len(data), test_images=50000, noise_levels=TIMES,
            noise='(1-t)*GAP(normalized_latent)+(t/16)*N(0,I)',
            epochs=40, global_batch=4096, encode_microbatch=MICRO, world=8, seed=42,
            optimizer='AdamW betas=(0.9,0.999), eps=1e-8, weight_decay=0',
            lr='cosine .01 to .0001 over40 epochs', init='zero independent affine768->1000 heads',
            fp32=True, tf32=False, autocast=False, feature_cache=False, flip=False,
            rng='GPU per-rank noise seeds; CPU seed42+epoch shuffle; same distribution, not bitwise DINOv3 noise',
            checkpoint='every epoch; fixed epoch40 evaluation, no validation tuning', wandb=False))
        print(f'READY: 8 GPUs online RAE normalized DINOv2-B; resume_epoch={start_epoch}', flush=True)
    for epoch in range(start_epoch,EPOCHS):
        begin = time.monotonic()
        lr = .0001+(.01-.0001)*.5*(1+math.cos(math.pi*epoch/(EPOCHS-1)))
        for group in opt.param_groups:
            group['lr'] = lr
        order = torch.randperm(len(data), generator=torch.Generator().manual_seed(SEED+epoch))
        # Partition each global minibatch without duplication or dropping the tail.
        batches = [order[o:o+BATCH][rank::world].tolist() for o in range(0,len(data),BATCH)]
        loader = DataLoader(data, batch_sampler=batches, num_workers=2, pin_memory=True,
            prefetch_factor=2, multiprocessing_context='spawn')
        rng = torch.Generator(device=device).manual_seed(100042+epoch*world+rank)
        totals = torch.zeros(4, dtype=torch.float64, device=device)
        for step,(images,labels) in enumerate(loader):
            global_n = min(BATCH, len(data)-step*BATCH)
            with torch.no_grad():
                g = torch.cat([encoder(b.to(device, non_blocking=True)) for b in images.split(MICRO)])
            labels = labels.to(device, non_blocking=True)
            logits = torch.bmm(noised(g,rng),weight)+bias
            sums = ce(logits,labels).sum(1)
            if not torch.isfinite(sums).all():
                raise RuntimeError(f'Nonfinite epoch={epoch+1} step={step} rank={rank}')
            opt.zero_grad(set_to_none=True)
            (sums.sum()/global_n).backward()
            # SUM gradients of global-example-normalized losses, including uneven final batch.
            for p in (weight,bias):
                dist.all_reduce(p.grad)
            opt.step()
            totals += sums.detach().double()
            if step % 20 == 0:
                means = sums.detach().clone()
                dist.all_reduce(means)
                if rank == 0:
                    progress = dict(phase='train', epoch=epoch+1, epochs=40, step=step+1,
                        steps=len(batches), loss=(means/global_n).tolist(), lr=lr,
                        seconds=time.monotonic()-begin, time_utc=datetime.now(timezone.utc).isoformat())
                    dump(ROOT/'status.json', progress)
                    print(json.dumps(progress), flush=True)
        dist.all_reduce(totals)
        if rank == 0:
            row = dict(epoch=epoch+1, lr=lr, seconds=time.monotonic()-begin,
                **{f'loss_t{t}':float(v/len(data)) for t,v in zip(TIMES,totals)})
            history.append(row)
            state = dict(epoch=epoch+1,epochs=40,times=TIMES,world=world,
                weight=weight.detach().cpu(),bias=bias.detach().cpu(),optimizer=opt.state_dict(),history=history)
            save(results/'latest.pt',state)
            if (epoch+1)%10 == 0:
                save(results/f'epoch{epoch+1:02d}.pt',state)
            with (results/'training.csv').open('w') as f:
                writer=csv.DictWriter(f,fieldnames=list(row)); writer.writeheader(); writer.writerows(history)
            print(json.dumps(row),flush=True)
        dist.barrier()
    evaluate(encoder,weight,bias,device,rank,world)
    dist.destroy_process_group()


@torch.no_grad()
def evaluate(encoder,weight,bias,device,rank,world):
    dataset = ValidationImages()
    loader = DataLoader(dataset,batch_size=MICRO,sampler=range(rank,len(dataset),world),
        num_workers=2,pin_memory=True,multiprocessing_context='spawn')
    generators = [torch.Generator(device=device).manual_seed(s*world+rank) for s in (424200,424201,424202)]
    counts = torch.zeros(4,4,3,dtype=torch.float64,device=device)
    if rank == 0:
        dump(ROOT/'status.json',dict(phase='evaluate',epoch=40))
    for images,labels in loader:
        g = encoder(images.to(device,non_blocking=True))
        y = labels.to(device)
        for mode in range(4):
            inputs = g[None].expand(4,-1,-1) if mode==0 else noised(g,generators[mode-1])
            logits=torch.bmm(inputs,weight)+bias
            top=logits.topk(5,dim=-1).indices
            counts[mode,:,0] += (top[:,:,0]==y[None]).sum(1)
            counts[mode,:,1] += (top==y[None,:,None]).any(-1).sum(1)
            counts[mode,:,2] += ce(logits,y).sum(1)
    dist.all_reduce(counts)
    if rank != 0:
        return
    values=counts.cpu().numpy()/50000
    rows,summary=[],[]
    for head,t in enumerate(TIMES):
        for mode in range(4):
            if head==0 and mode>0:
                continue
            rows.append(dict(train_noise=t,test_noise=0. if mode==0 else t,
                mode='clean' if mode==0 else 'matched_noise',seed=-1 if mode==0 else 424199+mode,
                top1=values[mode,head,0]*100,top5=values[mode,head,1]*100,cross_entropy=values[mode,head,2]))
        for name,indices in [('clean',[0]),('matched_noise',[1,2,3])]:
            if head==0 and name=='matched_noise':
                continue
            summary.append(dict(train_noise=t,mode=name,
                top1_mean=float(values[indices,head,0].mean()*100),top1_std=float(values[indices,head,0].std()*100),
                top5_mean=float(values[indices,head,1].mean()*100),top5_std=float(values[indices,head,1].std()*100)))
    with (ROOT/'results/accuracy_by_seed.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    dump(ROOT/'results/accuracy_summary.json',summary)
    report=['# RAE DINOv2-B Normalized GAP Noise Probes','',
        '40 epochs; full ImageNet; independent noise-specific probes; no feature cache.',
        'Official RAE encoder (non-affine final LayerNorm) and official latent mean/variance.',
        'GAP over256 patches reduces independent Gaussian noise standard deviation by16.',
        'FP32 throughout, TF32 disabled. No validation tuning. Noise std is across3 test seeds.',
        '', '| Train noise | Test | Top-1 (%) | Top-5 (%) |','|---|---|---:|---:|']
    for r in summary:
        report.append(f"| {r['train_noise']} | {r['mode']} | {r['top1_mean']:.3f} +/- {r['top1_std']:.3f} | {r['top5_mean']:.3f} +/- {r['top5_std']:.3f} |")
    (ROOT/'REPORT.md').write_text('\n'.join(report)+'\n')
    dump(ROOT/'status.json',dict(phase='complete',epoch=40,time_utc=datetime.now(timezone.utc).isoformat()))
    dump(ROOT/'complete.json',dict(epochs=40,test_images=50000))
    print(json.dumps(summary),flush=True)


if __name__=='__main__':
    main()
