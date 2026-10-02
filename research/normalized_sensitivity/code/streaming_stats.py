"""One-pass nested variance traces, with Bessel and finite-K corrections."""
import torch


def moments(x):
    x=x.float()
    x=x*torch.rsqrt(x.square().mean(-1,keepdim=True)+1e-6)
    var,mean=torch.var_mean(x,dim=0,correction=1)
    return mean,var.double().sum()


class Streaming:
    def __init__(self):
        self.points={}

    @torch.no_grad()
    def add(self,name,mean,within):
        mean=mean.double()
        if name not in self.points:
            self.points[name]=dict(n=1,mean=mean.clone(),m2=within.new_zeros(()),
                                   within=within.clone(),dimension=mean.numel())
            return
        s=self.points[name];s['n']+=1
        delta=mean-s['mean']
        s['m2'].add_(delta.square().sum()*((s['n']-1)/s['n']))
        s['mean'].add_(delta/s['n']);s['within'].add_(within)

    def rows(self,k):
        rows=[]
        for name,s in self.points.items():
            if s['n']<2:continue
            b=float(s['m2']/(s['n']-1));w=float(s['within']/s['n']);d=s['dimension']
            corrected=b-w/k
            rows.append(dict(point=name,images=s['n'],noises=k,dimensions=d,
                between_raw=b,within=w,between_corrected=corrected,
                between_raw_per_coordinate=b/d,within_per_coordinate=w/d,
                between_corrected_per_coordinate=corrected/d,
                ratio_raw=b/w if w>0 else None,ratio_corrected=corrected/w if w>0 else None))
        return rows
