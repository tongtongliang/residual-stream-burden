"""One GPU, one model/noise level: extract once, train11 independent probes."""
import argparse
import csv
import gc
import importlib.util
import json
import math
import os
from pathlib import Path
import time
os.environ.update(WANDB_MODE='disabled',WANDB_DISABLED='true',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from models import ROOT,BASE,registry,load


def dump(path,value):
    temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temp.replace(path)


def save(path,value):
    temp=path.with_suffix('.partial.pt');torch.save(value,temp);temp.replace(path)


def data_module():
    path=BASE/'dinov2_noise_linear_probe/train.py'
    spec=importlib.util.spec_from_file_location('dinov2_feature_data',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def noise_like(x,ids,seed):
    # Stable per-image CPU RNG: same pixels/noise for all models in a group.
    return torch.stack([torch.randn(tuple(x.shape[1:]),generator=torch.Generator().manual_seed(
        int(seed)+int(i)*1000003)) for i in ids]).to(x.device)


def main():
    p=argparse.ArgumentParser();p.add_argument('--label',required=True);p.add_argument('--alpha',type=float,required=True)
    p.add_argument('--limit',type=int,default=0,help='Preflight only; use a separate output')
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();entry=next(e for e in registry() if e['label']==args.label)
    assert args.alpha in (.25,.5,.75)
    out=args.output;out.mkdir(parents=True,exist_ok=True)
    if (out/'complete.json').exists():return
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    device=torch.device('cuda:0');torch.cuda.set_device(device)
    module=data_module();datasets={'train':module.TrainImages(),'val':module.ValidationImages()}
    extractor=load(entry,device)
    encoder=module.EncoderOnly().to(device) if entry['group']==2 else None
    # EncoderOnly.forward is GAP: call native RAE.encode for the FULL normalized map.
    def clean_input(images):
        if encoder is None:return images.float()/255*2-1
        from stage1.rae import RAE
        return RAE.encode(encoder,images.float()/255)
    def run_features(x):
        alpha=torch.full((len(x),),args.alpha,device=device)
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
            return active(x,alpha).float()
    check_images=torch.stack([datasets['train'][i][0] for i in range(2)]).to(device)
    with torch.no_grad():
        check=clean_input(check_images)
        check=(1-args.alpha)*check+args.alpha*noise_like(check,[0,1],700001)
        times=torch.full((2,),args.alpha,device=device)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            reference=extractor.native_hook_reference(check,times)
            explicit=extractor(check,times)
        torch.testing.assert_close(explicit,reference,rtol=.01,atol=.005)
    active=extractor
    compile_info={'mode':'default','native_hook_max_abs':float((explicit-reference).abs().max())}
    try:
        candidate=torch.compile(extractor,mode='default',dynamic=False)
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
            compiled=candidate(check,times)
        torch.testing.assert_close(compiled,explicit,rtol=.02,atol=.01)
        # Benchmark the ACTUAL extraction batch, not tiny preflight tensors.
        bench=check[:1].expand(64,*check.shape[1:]).contiguous()
        bt=torch.full((64,),args.alpha,device=device)
        timings=[]
        for fn in (extractor,candidate):
            with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
                for _ in range(2):fn(bench,bt)
                torch.cuda.synchronize();start=time.monotonic()
                for _ in range(5):fn(bench,bt)
                torch.cuda.synchronize();timings.append((time.monotonic()-start)/5)
        if timings[1]<timings[0]:active=candidate
        compile_info.update(eager_seconds=timings[0],compiled_seconds=timings[1],selected='compiled' if active is candidate else 'eager')
        del bench
    except Exception as exc:
        compile_info.update(selected='eager',compile_error=repr(exc))
    dump(out/'compile.json',compile_info);print(json.dumps(compile_info),flush=True)
    dump(out/'protocol.json',dict(model=entry,alpha=args.alpha,condition='null',points=list(range(1,12)),
        pooling='non-affine patch RMS eps1e-6 then GAP; before AdaLN',
        train_noise_seeds=[700001,700002],test_noise_seeds=[800001,800002,800003],
        train_epochs=40,batch=4096,lr_start=.01,lr_end=.0001,optimizer='AdamW default betas, eps1e-8, wd0',
        cache_dtype='float32',encoder_dtype='float32',backbone_autocast='bfloat16',
        torch_tf32=False,te_mhc_use_tf32=True if entry['kind']=='mhc' else None,
        checkpoint_state='model',gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),limit=args.limit,wandb=False))
    for split,dataset in datasets.items():
        n=min(len(dataset),args.limit) if args.limit else len(dataset)
        views=2 if split=='train' else 4
        marker=out/f'{split}_features_complete.json'
        if marker.exists():
            assert json.loads(marker.read_text())['images']==n
            continue
        path=out/f'{split}_gap.npy'
        progress=out/f'{split}_extract_progress.json'
        offset=json.loads(progress.read_text())['images'] if progress.exists() else 0
        array=np.lib.format.open_memmap(path,mode='r+' if offset else 'w+',dtype=np.float32,shape=(n,views,11,768))
        labels=np.lib.format.open_memmap(out/f'{split}_labels.npy',mode='r+' if offset else 'w+',dtype=np.int64,shape=(n,))
        # Fork read-only image readers; no CUDA operations occur in the workers.
        loader=DataLoader(dataset,batch_size=64,sampler=range(offset,n),num_workers=2,pin_memory=True,prefetch_factor=2)
        started=time.monotonic()
        for images,y in loader:
            ids=list(range(offset,offset+len(y)))
            with torch.no_grad():
                x=clean_input(images.to(device,non_blocking=True))
                for view in range(views):
                    seed=(700001+view) if split=='train' else 800000+view
                    z=x if split=='val' and view==0 else (1-args.alpha)*x+args.alpha*noise_like(x,ids,seed)
                    # Clean test has clean pixels/latents AND clean time embedding.
                    alpha_value=0. if split=='val' and view==0 else args.alpha
                    padded=torch.cat((z,z[-1:].expand(64-len(z),*z.shape[1:]))) if len(z)<64 else z
                    at=torch.full((64,),alpha_value,device=device)
                    with torch.autocast('cuda',dtype=torch.bfloat16):
                        feature=active(padded,at).float()[:len(y)]
                    if not torch.isfinite(feature).all():raise RuntimeError('Nonfinite extracted features')
                    array[offset:offset+len(y),view]=feature.cpu().numpy()
            labels[offset:offset+len(y)]=y.numpy();offset+=len(y)
            if offset%4096==0 or offset==n:
                array.flush();labels.flush()
                dump(progress,dict(images=offset,total=n,seconds=time.monotonic()-started))
                print(f'{split} features {offset}/{n}',flush=True)
        dump(marker,dict(images=n,views=views));del array,labels
    del active,extractor,encoder;gc.collect();torch.cuda.empty_cache()
    train(out,args.limit)
    evaluate(out)


def loss_function(inputs,weight,bias,labels):
    logits=torch.bmm(inputs,weight)+bias
    return F.cross_entropy(logits.flatten(0,1),labels.repeat(11),reduction='none').reshape(11,-1).mean(1)


def train(out,limit):
    if (out/'training_complete.json').exists():return
    # One RAM copy; avoid repeated random disk reads over40 epochs.
    x=torch.from_numpy(np.array(np.load(out/'train_gap.npy',mmap_mode='r'),copy=True))
    y=torch.from_numpy(np.load(out/'train_labels.npy'))
    w=torch.nn.Parameter(torch.zeros(11,768,1000,device='cuda'))
    b=torch.nn.Parameter(torch.zeros(11,1,1000,device='cuda'))
    opt=torch.optim.AdamW([w,b],lr=.01,weight_decay=0.,foreach=False)
    start,history=0,[]
    if (out/'latest.pt').exists():
        s=torch.load(out/'latest.pt',map_location='cpu',weights_only=True)
        with torch.no_grad():w.copy_(s['weight']);b.copy_(s['bias'])
        opt.load_state_dict(s['optimizer']);start,history=s['epoch'],s['history']
    # GEMM dominates this tiny head; eager batches avoid per-head compile overhead.
    for epoch in range(start,40):
        began=time.monotonic();lr=.0001+(.01-.0001)*.5*(1+math.cos(math.pi*epoch/39))
        for g in opt.param_groups:g['lr']=lr
        order=torch.randperm(len(x),generator=torch.Generator().manual_seed(42+epoch))
        totals=torch.zeros(11,device='cuda',dtype=torch.float64)
        for offset in range(0,len(x),4096):
            ids=order[offset:offset+4096]
            inputs=x[ids,epoch%2].permute(1,0,2).contiguous().cuda()
            losses=loss_function(inputs,w,b,y[ids].cuda())
            if not torch.isfinite(losses).all():raise RuntimeError('Nonfinite probe loss')
            opt.zero_grad(set_to_none=True);losses.sum().backward();opt.step()
            totals+=losses.detach().double()*len(ids)
        row=dict(epoch=epoch+1,lr=lr,seconds=time.monotonic()-began,loss=(totals/len(x)).tolist())
        history.append(row)
        save(out/'latest.pt',dict(epoch=epoch+1,weight=w.detach().cpu(),bias=b.detach().cpu(),optimizer=opt.state_dict(),history=history))
        dump(out/'training_progress.json',row);print(json.dumps(row),flush=True)
    dump(out/'training_complete.json',dict(epochs=40,images=len(x)))


@torch.no_grad()
def evaluate(out):
    s=torch.load(out/'latest.pt',map_location='cpu',weights_only=True)
    assert s['epoch']==40
    w,b=s['weight'].cuda(),s['bias'].cuda()
    x=np.load(out/'val_gap.npy',mmap_mode='r');y=np.load(out/'val_labels.npy')
    counts=torch.zeros(4,11,2,device='cuda',dtype=torch.float64)
    for offset in range(0,len(x),2048):
        labels=torch.from_numpy(y[offset:offset+2048]).cuda()
        for v in range(4):
            inputs=torch.from_numpy(np.array(x[offset:offset+len(labels),v],copy=True)).permute(1,0,2).contiguous().cuda()
            top=(torch.bmm(inputs,w)+b).topk(5,dim=-1).indices
            counts[v,:,0]+=(top[:,:,0]==labels[None]).sum(1)
            counts[v,:,1]+=(top==labels[None,:,None]).any(-1).sum(1)
    acc=counts.cpu().numpy()/len(x)*100;rows=[]
    for layer in range(11):
        for mode,vs in [('clean',[0]),('matched_noise',[1,2,3])]:
            rows.append(dict(after_block=layer+1,mode=mode,top1=float(acc[vs,layer,0].mean()),
                top1_seed_std=float(acc[vs,layer,0].std()),top5=float(acc[vs,layer,1].mean()),
                top5_seed_std=float(acc[vs,layer,1].std())))
    dump(out/'accuracy.json',rows);dump(out/'complete.json',dict(epochs=40,test_images=len(x)))


if __name__=='__main__':main()
