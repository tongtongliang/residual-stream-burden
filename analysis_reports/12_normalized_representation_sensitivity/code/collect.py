"""Forward-only normalized representation sensitivity; no GAP, no W&B."""
import argparse
from contextlib import nullcontext
import csv
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'hidden_noise_linear_probe'))
from models import load
from job import data_module
from latent_cache import CACHE,CachedLatents
from collect_hidden_patch_pca import load_internal_model,load_external_model
from streaming_stats import Streaming,moments


def atomic(path,value):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False));tmp.replace(path)


class Observe:
    def __init__(self):
        self.stats=Streaming();self.reduce=torch.compile(moments,mode='default')
        self.names=[];self.seen=set()

    def add(self,name,x,tokens=256):
        assert name not in self.seen,name
        self.seen.add(name)
        assert x.ndim==3 and x.shape[0]==128 and x.shape[1]==tokens,(name,x.shape)
        mean,var=self.reduce(x)
        self.stats.add(name,mean,var)

    def hook(self,name,prefix=0):
        def call(module,args):self.add(name,args[0][:,prefix:])
        return call


def setup(entry,device,observe):
    if entry['kind'] in ('external','hyper'):
        loader=load_external_model if entry['kind']=='external' else load_internal_model
        args=(entry,device) if entry['kind']=='external' else (entry,device,'flash')
        model,name,pred,arch,forward=loader(*args)
        blocks=model.semantic_blocks if name=='hyperdit_b' else model.patch_blocks if name=='pixeldit_b' else model.blocks
        blocks=list(blocks)[:8 if name=='hyperdit_b' else 12]
        prefix=int(model.num_registers) if name=='hyperdit_b' else 0
    else:
        model=load(entry,device).model
        blocks=list(model.blocks);prefix=0
        if entry['group']==2:
            for block in blocks:
                block.attn.backend='auto'
        def forward(x,t,y):
            c=model.t_embedder(t)+model.y_embedder(y)
            state=model._embed_tokens(x)
            if entry['kind']=='mhc':state=model._expand(state)
            for block in blocks:state=block(state,c,model.feat_rope)
    if entry['kind']!='sihc':
        for i,block in enumerate(blocks):
            for part in (0,1):
                name=f'workspace/b{i+1:02d}_{"attn" if part==0 else "mlp"}'
                observe.names.append(name)
                getattr(block,'norm1' if part==0 else 'norm2').register_forward_pre_hook(observe.hook(name,prefix))
        return model,forward
    assert tuple(model.stage_sizes)==(12,) and model.patch_size==4
    for i in range(12):
        for part in ('attn','mlp'):
            observe.names.extend([f'carrier/b{i+1:02d}_{part}',f'workspace/b{i+1:02d}_{part}'])
    def sihc_forward(x,t,y):
        from sihc.models.sihc.sublayer_kernels import sublayer_read,sublayer_triangular,sublayer_write
        c=model.t_embedder(t)+model.y_embedder(y);x0=model.x_embedder(x)
        connections=[conn for block in blocks for conn in (block.attention_connection,block.mlp_connection)]
        alpha=torch.stack([conn.alpha(dtype=x0.dtype) for conn in connections]).contiguous()
        beta=torch.stack([conn.beta(dtype=x0.dtype) for conn in connections]).contiguous()
        base=sublayer_read(x0,alpha);gamma=(alpha[:,None]*beta[None,:]).sum(-1)
        updates=[];carrier=x0.float()
        for i,block in enumerate(blocks):
            mod=block.branch.conditioning(c)
            for part in (0,1):
                event=2*i+part;name=f'b{i+1:02d}_{"attn" if part==0 else "mlp"}'
                observe.add('carrier/'+name,carrier.flatten(1,2),4096)
                z=base[event]
                if updates:z=sublayer_triangular(z,updates,gamma[event,:event].contiguous())
                observe.add('workspace/'+name,z)
                delta=(block.branch.attention_update(z,mod,model.workspace_rope) if part==0 else block.branch.mlp_update(z,mod)).contiguous()
                updates.append(delta)
                # Diagnostic carrier only: FP32 lazy sum, never fed back to native workspace.
                if event<23:
                    from sihc.models.sihc.sublayer_kernels import _write
                    updated=torch.empty_like(carrier)
                    _write[(carrier.shape[0]*256,12)](
                        carrier,(delta,),beta[event:event+1].contiguous(),updated,
                        768,16,4,True,64,num_warps=4)
                    carrier=updated
    return model,sihc_forward


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--label',required=True)
    parser.add_argument('--t-clean',type=float,required=True);args=parser.parse_args()
    torch.set_num_threads(2);torch.cuda.set_device(0)
    torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
    device=torch.device('cuda');entry=next(e for e in json.loads((ROOT/'models.json').read_text()) if e['label']==args.label)
    folder=ROOT/'runs'/f'{args.label}_t{round(args.t_clean*100):03d}';folder.mkdir(parents=True,exist_ok=True)
    assert not (folder/'complete.json').exists(),'Already complete'
    sample=np.load(ROOT/'samples.npz');ids=sample['indices'];labels=sample['labels']
    dataset=CachedLatents(CACHE,'train') if entry['group']==2 else data_module().TrainImages()
    loader=torch.utils.data.DataLoader(dataset,sampler=ids.tolist(),batch_size=1,num_workers=2,pin_memory=True,prefetch_factor=4)
    observe=Observe();model,forward=setup(entry,device,observe)
    model.eval().requires_grad_(False)
    amp=nullcontext if entry['group']==2 else lambda:torch.autocast('cuda',dtype=torch.bfloat16)
    atomic(folder/'manifest.json',dict(model=entry,t_clean=args.t_clean,noise_fraction=1-args.t_clean,
        images=len(ids),noises=128,null_class=1000,flip=False,state='model',
        rms='per spatial token across channels, no trainable affine, eps=1e-6',
        covariance='Bessel N-1 and K-1; corrected between=raw between-within/K; no clipping',
        precision='FP32 with TF32' if entry['group']==2 else 'BF16 autocast; FP32 RMS; FP64 streaming',
        carrier='FP32 diagnostic incremental native write, not fed to workspace',points=observe.names))
    began=time.monotonic()
    with torch.inference_mode():
        for i,(image,y) in enumerate(loader):
            assert int(y)==int(labels[i]),(i,int(y),int(labels[i]))
            image=image.to(device,non_blocking=True).float()
            if entry['group']==1:image=image.div(255).mul(2).sub(1)
            seed=int((1234567+int(ids[i])*1000003)%(2**63-1))
            noise=torch.randn((128,*image.shape[1:]),device=device,generator=torch.Generator(device=device).manual_seed(seed))
            z=args.t_clean*image+(1-args.t_clean)*noise
            t=torch.full((128,),1-args.t_clean if entry['group']==2 else args.t_clean,device=device)
            null=torch.full((128,),1000,device=device,dtype=torch.long)
            observe.seen.clear()
            with amp():forward(z,t,null)
            assert observe.seen==set(observe.names),(observe.seen,set(observe.names))
            if (i+1)%64==0 or i==0 or i+1==len(ids):
                torch.cuda.synchronize();seconds=time.monotonic()-began
                atomic(folder/'progress.json',dict(images=i+1,total=len(ids),seconds=seconds,images_per_sec=(i+1)/seconds,
                    peak_gb=torch.cuda.max_memory_allocated()/1e9))
                if i:atomic(folder/'partial_metrics.json',observe.stats.rows(128))
                print(json.dumps(dict(images=i+1,total=len(ids),seconds=seconds,images_per_sec=(i+1)/seconds)),flush=True)
    rows=observe.stats.rows(128)
    assert all(r['images']==2048 for r in rows)
    atomic(folder/'metrics.json',rows)
    with (folder/'metrics.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    atomic(folder/'complete.json',dict(images=len(ids),noises=128,points=len(rows),seconds=time.monotonic()-began))


if __name__=='__main__':main()
