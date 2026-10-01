"""Render DINO target covariance spectra from cached RAE-normalized measurements."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[3]
SRC=ROOT/'analysis_reports/16_dinov2b_input_patch_spectrum/data/spectra/dino_rae_position_normalized.npz'
OUT=ROOT/'paper/figures/patch_geometry/dino_target_spectra'
OUT.parent.mkdir(parents=True, exist_ok=True)
d=np.load(SRC); clean=d['eigenvalues']; assert clean.shape==(768,) and np.all(clean>=0)
assert np.allclose(d['explained_variance_ratio'],clean/clean.sum())
plt.rcParams.update({'font.family':'serif','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
fig,axs=plt.subplots(1,2,figsize=(7.0,2.55)); rank=np.arange(1,769); stats={}
for label,a,color,ls in [('Clean target',clean,'#2679B2','-'),('Velocity target',clean+1,'#DA7C30','--')]:
 p=a/a.sum();cum=np.cumsum(p); r90=int(np.searchsorted(cum,.9)+1)
 axs[0].semilogy(rank,a,label=label,color=color,ls=ls,lw=1.8)
 axs[1].plot(rank,cum,label=f'{label} ($r_{{90}}={r90}$)',color=color,ls=ls,lw=1.8)
 axs[1].plot(r90,cum[r90-1],'o',color=color,ms=3)
 stats[label]={'ER':float(np.exp(-(p*np.log(p)).sum())),'SR':float(a.sum()/a.max()),'r90':r90,'r95':int(np.searchsorted(cum,.95)+1),'r99':int(np.searchsorted(cum,.99)+1),'trace':float(a.sum())}
for ax in axs:
 ax.set_xlim(1,768);ax.set_xlabel('Direction rank');ax.set_xticks([1,256,512,768]);ax.grid(alpha=.15)
axs[0].set_ylabel('Covariance eigenvalue');axs[0].legend(frameon=False,fontsize=9)
axs[1].set_ylabel('Cumulative variance');axs[1].set_ylim(0,1.025);axs[1].axhline(.9,color='gray',lw=.7,ls=':');axs[1].legend(frameon=False,fontsize=8,loc='lower right')
fig.tight_layout(pad=.5,w_pad=1.5);fig.savefig(OUT.with_suffix('.pdf'),bbox_inches='tight');fig.savefig(OUT.with_suffix('.png'),dpi=180,bbox_inches='tight')
meta={'source':str(SRC.relative_to(ROOT)),'source_commit':'39d19c6','observations':int(d['count']),'covariance':'globally centered bag of 256 tokens/image; 100000 images','normalization':'RAE position normalization after DINOv2-B final LayerNorm; no additional per-token RMS','velocity':'analytic covariance eigenvalues = measured clean eigenvalues + 1','statistics':stats}
OUT.with_name(OUT.name+'_sources.json').write_text(json.dumps(meta,indent=2)+'\n')
print(json.dumps(stats,indent=2))
