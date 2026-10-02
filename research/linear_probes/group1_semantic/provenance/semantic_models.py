"""Group1 semantic-only frozen extractors; spatial tokens, no pixel decoders."""
import json
import os
from pathlib import Path
import torch
from models import BASE, pooled
from collect_hidden_patch_pca import load_internal_model, load_external_model

def registry():
    external=json.loads((BASE/'semantic_noise_decomposition/model_registry.json').read_text())
    hyper=json.loads((BASE/'residual_stream_noise_sensitivity/model_registry.json').read_text())
    entries=[e for e in external if e['family']=='external']
    entries += [e for e in hyper if e['label']=='hyperdit_b_velocity']
    assert len(entries)==4
    return [dict(e,group=1) for e in entries]

class SemanticExtractor(torch.nn.Module):
    def __init__(self,model,kind):
        super().__init__()
        self.model=model
        self.kind=kind
        self.points=7 if kind=='hyperdit_b' else 11
        if kind!='hyperdit_b':
            self.register_buffer('position',model.fetch_pos(16,16,next(model.parameters()).device),persistent=False)

    @property
    def blocks(self):
        m=self.model
        return m.semantic_blocks if self.kind=='hyperdit_b' else m.patch_blocks if self.kind=='pixeldit_b' else m.blocks

    def forward(self,x,alpha):
        m=self.model
        y=torch.full((len(x),),1000,device=x.device,dtype=torch.long)
        t=1-alpha
        values=[]
        if self.kind=='hyperdit_b':
            state=m._semantic_tokens(x)
            c=m.t_embedder(t)+m.y_embedder(y)
            for block in m.semantic_blocks[:self.points]:
                state=block(state,c,m.semantic_rope,m.semantic_coords,m.num_registers)
                spatial=state[:,m.num_registers:]
                assert spatial.shape[1:]==(256,768)
                values.append(pooled(spatial))
        else:
            patches=torch.nn.functional.unfold(x,kernel_size=m.patch_size,stride=m.patch_size).transpose(1,2)
            te=m.t_embedder(t).view(len(x),-1,m.hidden_size)
            ye=m.y_embedder(y).view(len(x),1,m.hidden_size)
            c=torch.nn.functional.silu(te+ye)
            state=m.s_embedder(patches)
            for block in self.blocks[:self.points]:
                state=block(state,c,self.position,None)
                assert state.shape[1:]==(256,768)
                values.append(pooled(state))
        return torch.stack(values,1)

    @torch.no_grad()
    def check_native(self,x,alpha):
        values=[]
        def capture(module,args,output):
            tokens=output[:,self.model.num_registers:] if self.kind=='hyperdit_b' else output
            values.append(pooled(tokens))
        handles=[block.register_forward_hook(capture) for block in self.blocks[:self.points]]
        try:
            self.model(x,1-alpha,torch.full((len(x),),1000,device=x.device,dtype=torch.long))
        finally:
            for handle in handles:handle.remove()
        expected=torch.stack(values,1)
        actual=self(x,alpha)
        torch.testing.assert_close(actual,expected,rtol=1e-3,atol=1e-3)
        return dict(native_max_abs=float((actual-expected).abs().max()),shape=list(actual.shape),
                    points=self.points,spatial_tokens=256,registers_excluded=self.kind=='hyperdit_b')

def load(entry,device):
    if entry['family']=='external':
        model,kind,_,_,_=load_external_model(entry,device)
    else:
        model,kind,_,_,_=load_internal_model(entry,device,'flash')
    return SemanticExtractor(model,kind).eval().requires_grad_(False)
