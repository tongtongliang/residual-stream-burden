"""Group 6 patch geometry. FP32 transforms, CHW patch order, checkpoint-owned bases."""
from pathlib import Path
import numpy as np
import torch
from torch import nn

MODES = ('none', 'whiten', 'pca_top16', 'random16')


def patchify(x):
    if x.ndim != 4 or x.shape[1:] != (3,256,256):
        raise ValueError('Group 6 requires [B,3,256,256]')
    return x.reshape(-1,3,16,16,16,16).permute(0,2,4,1,3,5).reshape(-1,256,768)


def unpatchify(x):
    return x.reshape(-1,16,16,3,16,16).permute(0,3,1,4,2,5).reshape(-1,3,256,256)


class PatchGeometry(nn.Module):
    def __init__(self, mode, basis, mean, scales, random_seed=20260916):
        super().__init__()
        if mode not in MODES[1:]: raise ValueError(mode)
        self.mode, self.random_seed = mode, int(random_seed)
        rows = 768 if mode == 'whiten' else 16
        if basis.shape != (rows,768) or mean.shape != (768,) or scales.shape != (768,):
            raise ValueError('Invalid geometry dimensions')
        if not all(torch.isfinite(x).all() for x in (basis,mean,scales)) or (scales<=0).any():
            raise ValueError('Nonfinite/singular patch transform')
        gram = basis.double() @ basis.double().T
        torch.testing.assert_close(gram,torch.eye(rows,dtype=torch.float64,device=gram.device),atol=2e-6,rtol=2e-6)
        self.register_buffer('basis',basis.float().contiguous())
        self.register_buffer('mean',mean.float().contiguous())
        self.register_buffer('scales',scales.float().contiguous())

    @classmethod
    def from_basis_file(cls,mode,path='',random_seed=20260916):
        if mode == 'random16':
            gen=torch.Generator(device='cpu').manual_seed(random_seed)
            q,r=torch.linalg.qr(torch.randn(768,16,generator=gen,dtype=torch.float64),mode='reduced')
            basis=(q*torch.sign(r.diag())[None]).T
            return cls(mode,basis,torch.zeros(768),torch.ones(768),random_seed)
        with np.load(Path(path),allow_pickle=False) as p:
            basis=torch.from_numpy(p['components'].copy())
            mean=torch.from_numpy(p['mean'].copy())
            eigenvalues=torch.from_numpy(p['eigenvalues'].copy())
        if basis.shape!=(768,768) or eigenvalues.shape!=(768,): raise ValueError('Full 768-D PCA required')
        if not torch.all(eigenvalues[:-1]>=eigenvalues[1:]): raise ValueError('PCA must be in descending variance order')
        if mode=='whiten' and (eigenvalues<=0).any(): raise ValueError('Whitening requires positive eigenvalues; no silent truncation/floor')
        return cls(mode,basis if mode=='whiten' else basis[:16],mean,
                   eigenvalues.sqrt() if mode=='whiten' else torch.ones(768),random_seed)

    def encode(self,x):
        if self.mode!='whiten': return x
        with torch.autocast(x.device.type,enabled=False):
            z=(patchify(x.float())-self.mean) @ self.basis.T
            return unpatchify(z/self.scales)

    def decode(self,z):
        if self.mode!='whiten': return z
        with torch.autocast(z.device.type,enabled=False):
            return unpatchify((patchify(z.float())*self.scales) @ self.basis + self.mean)

    def noise(self,batch,device,*,generator=None):
        with torch.autocast(torch.device(device).type,enabled=False):
            if self.mode=='whiten':
                return torch.randn(batch,3,256,256,device=device,generator=generator,dtype=torch.float32)
            coefficients=torch.randn(batch,256,16,device=device,generator=generator,dtype=torch.float32)
            return unpatchify(coefficients @ self.basis)

    def payload(self):
        return dict(version=1,mode=self.mode,random_seed=self.random_seed,
                    patch_layout='CHW-p16',**{k:v.detach().cpu() for k,v in self.state_dict().items()})

    @classmethod
    def from_checkpoint(cls,ckpt,device='cpu'):
        p=ckpt.get('patch_geometry')
        args=ckpt.get('args',{})
        args=vars(args) if hasattr(args,'__dict__') else args
        requested=args.get('patch_geometry','none')
        if p is None:
            if requested!='none': raise ValueError('Geometry checkpoint missing its basis; refusing RGB fallback')
            return None
        if p.get('version')!=1 or p.get('patch_layout')!='CHW-p16': raise ValueError('Unsupported geometry payload')
        if requested != p['mode']: raise ValueError('Geometry metadata mismatch')
        return cls(p['mode'],p['basis'],p['mean'],p['scales'],p['random_seed']).to(device)


def training_geometry(args,checkpoint,device):
    mode=args.patch_geometry
    if mode!='none' and (args.model!='jit_b16' or args.input_space!='pixel_rgb' or args.repa):
        raise ValueError('Group 6 requires plain JiT-B/16 pixel inputs without REPA')
    if checkpoint is not None:
        geometry=PatchGeometry.from_checkpoint(checkpoint,device)
        if (geometry.mode if geometry else 'none')!=mode: raise ValueError('Cannot change geometry on resume')
        if geometry and geometry.mode=='random16' and geometry.random_seed!=args.patch_random_seed:
            raise ValueError('Cannot change random subspace on resume')
        return geometry
    if mode=='none': return None
    return PatchGeometry.from_basis_file(mode,args.patch_basis,args.patch_random_seed).to(device)
