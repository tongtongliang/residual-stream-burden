"""Whitening extension of archived normalized sensitivity; forward only."""
import argparse,csv,gc,json,os,sys,time,traceback
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,'/data/tongtong/project/sihc-recovery-b90811d')
from sihc.checkpoint import load_checkpoint,build_model_from_checkpoint,load_model_state
from sihc.patch_geometry import PatchGeometry
from sihc.imagenet import CachedTensorImageNet256
from streaming_stats import Streaming,moments
ROOT=Path(__file__).resolve().parents[1]
BASE=Path('/data/tongtong')
def atomic(path,data):
 tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n');tmp.replace(path)
def collect(pred,model,geometry,dataset,samples,t,checkpoint,reduce):
 folder=ROOT/'runs'/f'whiten_{pred}_t{round(t*100):03d}';folder.mkdir(parents=True,exist_ok=True)
 if (folder/'complete.json').exists(): return
 stats=Streaming();seen=set();handles=[];names=[];validation={}
 def hook(name):
  def call(module,args):
   assert name not in seen;seen.add(name)
   x=args[0];assert x.shape==(128,256,768),(name,x.shape)
   mean,within=reduce(x)
   if not validation:
    rm,rv=moments(x)
    torch.testing.assert_close(mean,rm,rtol=2e-6,atol=2e-6)
    torch.testing.assert_close(within,rv,rtol=2e-6,atol=1e-8)
    validation.update(passed=True,mean_max_abs=float((mean-rm).abs().max()),within_abs=float((within-rv).abs()))
   stats.add(name,mean,within)
  return call
 for i,b in enumerate(model.blocks):
  for part in ('attn','mlp'):
   name=f'workspace/b{i+1:02d}_{part}';names.append(name)
   handles.append(getattr(b,'norm1' if part=='attn' else 'norm2').register_forward_pre_hook(hook(name)))
 atomic(folder/'manifest.json',{'model':f'whiten_{pred}','checkpoint':str(checkpoint),'checkpoint_step':250200,'state':'model','t_clean':t,'noise_fraction':1-t,'images':2048,'noises':128,'null_class':1000,'flip':False,'points':names,'dimensions':256*768,'input':'uint8 -> [-1,1] -> saved FP32 whitening -> t*xw+(1-t)*isotropic Gaussian noise','noise_seed_formula':'(1234567 + image_index * 1000003) % (2**63-1)','rms':'non-affine per spatial token across channels, eps=1e-6','covariance':'N-1 and K-1; B_corrected=B_raw-W/K; no clipping','precision':'BF16 autocast, FP32 RMS/moments, FP64 streaming; TF32 enabled','model_compile':False,'moment_compile':'default','physical_gpu':os.environ['CUDA_VISIBLE_DEVICES'],'allocator_cap_gib':2,'torch':torch.__version__,'source':'/data/tongtong/project/sihc-recovery-b90811d','sample_manifest':str(ROOT/'samples.npz'),'wandb':False})
 began=time.monotonic();torch.cuda.reset_peak_memory_stats()
 try:
  with torch.inference_mode():
   for i,index in enumerate(samples['indices']):
    image,label=dataset[int(index)];assert label==int(samples['labels'][i])
    image=image.to('cuda').float().div(255).mul(2).sub(1)[None]
    xw=geometry.encode(image)
    seed=int((1234567+int(index)*1000003)%(2**63-1))
    noise=torch.randn((128,3,256,256),device='cuda',generator=torch.Generator(device='cuda').manual_seed(seed))
    z=t*xw+(1-t)*noise;times=torch.full((128,),t,device='cuda');null=torch.full((128,),1000,device='cuda',dtype=torch.long)
    seen.clear()
    with torch.autocast('cuda',dtype=torch.bfloat16):
     cond=model.t_embedder(times)+model.y_embedder(null);state=model._embed_tokens(z)
     for block in model.blocks:state=block(state,cond,model.feat_rope)
    assert seen==set(names)
    if i==0 or (i+1)%64==0 or i==2047:
     torch.cuda.synchronize();rows=stats.rows(128)
     assert all(r['ratio_corrected'] is not None and all(np.isfinite(r[k]) for k in ('between_raw','within','between_corrected','ratio_corrected')) for r in rows)
     progress={'model':pred,'t_clean':t,'images':i+1,'total':2048,'seconds':time.monotonic()-began,'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30,'peak_reserved_gib':torch.cuda.max_memory_reserved()/2**30}
     atomic(folder/'progress.json',progress)
     if rows:atomic(folder/'partial_metrics.json',rows)
     print(json.dumps(progress),flush=True)
  rows=stats.rows(128);assert len(rows)==24 and all(r['images']==2048 for r in rows)
  atomic(folder/'metrics.json',rows)
  with (folder/'metrics.csv').open('w') as f:
   writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
  atomic(folder/'reduction_validation.json',validation)
  atomic(folder/'complete.json',dict(status='complete',images=2048,noises=128,points=24,seconds=time.monotonic()-began,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30))
 finally:
  for handle in handles:handle.remove()

def main():
 p=argparse.ArgumentParser();p.add_argument('--prediction',choices=['clean','velocity'],required=True);args=p.parse_args()
 torch.set_num_threads(2);torch.cuda.set_device(0)
 torch.cuda.set_per_process_memory_fraction(2*2**30/torch.cuda.get_device_properties(0).total_memory,0)
 torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
 checkpoint=BASE/f'artifacts/checkpoints/sihc-research-checkpoints/resume/group6/g6_jit_b16_whiten_{args.prediction}_seed0/step_00250200.pt'
 ckpt=load_checkpoint(checkpoint,mmap=True);assert ckpt['step']==250200
 model=build_model_from_checkpoint(ckpt,attn_backend='flash');load_model_state(model,ckpt,'model')
 model=model.cuda().eval().requires_grad_(False);geometry=PatchGeometry.from_checkpoint(ckpt,'cuda');assert geometry.mode=='whiten'
 del ckpt
 dataset=CachedTensorImageNet256(BASE/'artifacts/data/imagenet256-uint8-cache',train=False)
 samples=np.load(ROOT/'samples.npz');assert len(samples['indices'])==2048
 reduce=torch.compile(moments,mode='default')
 for t in (.25,.5,.75):
  collect(args.prediction,model,geometry,dataset,samples,t,checkpoint,reduce)
  gc.collect();torch.cuda.empty_cache()
 print('ALL_TIMES_COMPLETE '+args.prediction,flush=True)
if __name__=='__main__':main()
