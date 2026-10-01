"""Explicit next-block fan-in / read-workspace extraction, no live hooks."""
from pathlib import Path
import sys
import json
import os
os.environ.setdefault('CUDNN_PATH','/data/tongtong/transport_burden/.venv-mhc310/lib/python3.10/site-packages/nvidia/cudnn')
os.environ.setdefault('NVRTC_PATH','/data/tongtong/transport_burden/.venv-mhc310/lib/python3.10/site-packages/nvidia/cu13')
os.environ.setdefault('CURAND_PATH','/data/tongtong/transport_burden/.venv-mhc310/lib/python3.10/site-packages/nvidia/cu13')
import torch

ROOT=Path(__file__).resolve().parent
BASE=ROOT.parent
sys.path[:0]=['/data/tongtong/sihc',str(BASE/'dinov2_noise_linear_probe/RAE/src'),str(BASE/'hidden_patch_pca/code')]


def pooled(x):
    x=x.float()
    return (x*torch.rsqrt(x.square().mean(-1,keepdim=True)+1e-6)).mean(1)


def registry():
    old=json.loads((BASE/'semantic_noise_decomposition/model_registry.json').read_text())
    entries=[dict(e,group=1,kind='plain') for e in old if e['label'] in ('plain_jit_clean','plain_jit_velocity')]
    run=BASE/'runs/group1/g1_sihc_sublayer_b16_velocity_p4_seed0'
    entries.append(dict(label='sihc_sublayer_p4',group=1,kind='sihc',state_key='model',
        config=str(run/'config.json'),checkpoint=str(run/'checkpoints/step_00250200.pt')))
    names=['raev1-g2a-01-jit-b-clean-epoch200','raev1-g2a-02-jit-b-velocity-epoch200',
           'raev1-g2b-01-jit-b-mhc4-clean-epoch200','raev1-g2b-02-jit-b-mhc4-velocity-epoch200']
    for name in names:
        folder=BASE/'checkpoints/hf_group2a2b'/name
        entries.append(dict(label=name,group=2,kind='mhc' if 'mhc4' in name else 'plain',state_key='model',
            config=str(folder/'config.yaml'),checkpoint=str(folder/'checkpoints/ep-last.pt')))
    return entries


def load(entry,device):
    from collect_hidden_patch_pca import load_internal_model,add_checkpoint_safe_globals
    add_checkpoint_safe_globals()
    if entry['kind']=='sihc':
        from sihc.checkpoint import load_checkpoint,checkpoint_model_kwargs,load_model_state
        from sihc.models import build_model
        state=load_checkpoint(entry['checkpoint'],mmap=True,allow_unsafe=False)
        name='sihc_sublayer_1x12_d768_b4'
        kwargs=checkpoint_model_kwargs(state,model_name=name,attn_backend='flash' if device.type=='cuda' else 'math')
        model=build_model(name,**kwargs)
        load_model_state(model,state,entry['state_key'])
        assert model.patch_size==4 and tuple(model.stage_sizes)==(12,) and model.hidden_size==768
        model=model.to(device)
    elif entry['group']==1:
        model,*_=load_internal_model(entry,device,'flash' if device.type=='cuda' else 'math')
    else:
        import yaml
        from sihc.rae.models import RAEJiTB,RAEMHCJiTB
        config=yaml.safe_load(Path(entry['config']).read_text())
        cls=RAEMHCJiTB if entry['kind']=='mhc' else RAEJiTB
        params=dict(config['stage_2']['params'])
        params['attn_backend']='flash' if device.type=='cuda' else 'math'
        model=cls(**params)
        state=torch.load(entry['checkpoint'],map_location='cpu',mmap=True,weights_only=True)
        assert state['epoch']==200
        model.load_state_dict(state[entry['state_key']],strict=True)
        model=model.to(device)
    assert len(model.blocks)==12
    return Extractor(model.eval().requires_grad_(False),entry['kind'],entry['group'])


class Extractor(torch.nn.Module):
    def __init__(self,model,kind,group):
        super().__init__()
        self.model,self.kind,self.group=model,kind,group

    def forward(self,x,alpha):
        m=self.model
        t=alpha if self.group==2 else 1-alpha
        y=torch.full((x.shape[0],),1000,device=x.device,dtype=torch.long)
        c=m.t_embedder(t)+m.y_embedder(y)
        values=[]
        if self.kind=='sihc':
            from sihc.models.sihc.sublayer_kernels import sublayer_read,sublayer_triangular
            assert tuple(m.stage_sizes)==(12,)
            x0=m.x_embedder(x)
            connections=[conn for block in m.blocks for conn in (block.attention_connection,block.mlp_connection)]
            a=torch.stack([conn.alpha(dtype=x0.dtype) for conn in connections]).contiguous()
            b=torch.stack([conn.beta(dtype=x0.dtype) for conn in connections]).contiguous()
            base=sublayer_read(x0,a)
            gamma=(a[:,None]*b[None,:]).sum(-1)
            updates=[]
            for i,block in enumerate(m.blocks):
                z=base[2*i]
                if updates:
                    z=sublayer_triangular(z,updates,gamma[2*i,:2*i].contiguous())
                if i>0:
                    values.append(pooled(z))
                if i==11:
                    break
                modulation=block.branch.conditioning(c)
                updates.append(block.branch.attention_update(z,modulation,m.workspace_rope).contiguous())
                z=sublayer_triangular(base[2*i+1],updates,gamma[2*i+1,:2*i+1].contiguous())
                updates.append(block.branch.mlp_update(z,modulation).contiguous())
        else:
            state=m._embed_tokens(x)
            if self.kind=='mhc':
                state=m._expand(state)
            for i,block in enumerate(m.blocks):
                if i>0:
                    z=block.attn_hc.mix_and_aggregate(state)[0].transpose(0,1) if self.kind=='mhc' else state
                    values.append(pooled(z))
                if i==11:
                    break
                state=block(state,c,m.feat_rope)
        return torch.stack(values,1)

    def native_hook_reference(self,x,alpha):
        """Preflight only: compare extraction against actual native execution points."""
        m=self.model
        values=[]
        handles=[]
        for block in list(m.blocks)[1:]:
            target=block.branch.norm1 if self.kind=='sihc' else block.norm1 if self.kind=='mhc' else block
            handles.append(target.register_forward_pre_hook(lambda module,inputs: values.append(pooled(inputs[0]))))
        try:
            m(x,alpha if self.group==2 else 1-alpha,torch.full((len(x),),1000,device=x.device,dtype=torch.long))
        finally:
            for h in handles:h.remove()
        assert len(values)==11
        return torch.stack(values,1)
