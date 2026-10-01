"""Render the UCF101 4x16x16 raw RGB tubelet spectrum from cached eigenvalues."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[3]
SRC=ROOT/'analysis_reports/18_ucf101_raw_tubelet_spectrum/data/spectra/ucf101_t16_r256_pt4_p16.npz'
OUT=ROOT/'paper/figures/patch_geometry/video_tubelet_spectra'
OUT.parent.mkdir(parents=True, exist_ok=True)
d=np.load(SRC); e=d['eigenvalues']; tr=float(d['total_variance'])
assert e.shape==(512,) and np.all(e>=0) and np.isclose(e.sum()+float(d['unresolved_variance']),tr)
cum=np.cumsum(e)/tr; rank=np.arange(1,513)
r={f'r{q}':int(np.searchsorted(cum,q/100)+1) for q in (90,95,99)}
plt.rcParams.update({'font.family':'serif','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
fig,axs=plt.subplots(1,2,figsize=(7.0,2.55)); c='#2679B2'
axs[0].semilogx(rank,cum,color=c,lw=1.8)
axs[0].plot(r['r90'],cum[r['r90']-1],'o',color=c,ms=3.5)
axs[0].annotate(f"$r_{{90}}={r['r90']}$",(r['r90'],cum[r['r90']-1]),xytext=(8,-14),textcoords='offset points',fontsize=9)
axs[0].axhline(.9,color='gray',lw=.7,ls=':'); axs[0].set_ylim(0.7,1.005)
axs[0].set_ylabel('Cumulative variance')
axs[1].loglog(rank,e/tr,color=c,lw=1.8); axs[1].set_ylabel('Eigenvalue / total variance')
for ax in axs: ax.set_xlabel('Direction rank'); ax.grid(alpha=.15)
fig.tight_layout(pad=.5,w_pad=1.5); fig.savefig(OUT.with_suffix('.pdf'),bbox_inches='tight'); fig.savefig(OUT.with_suffix('.png'),dpi=180,bbox_inches='tight')
meta={'source':str(SRC.relative_to(ROOT)),'setting':'UCF101, 50 videos/class, one 4x16x16 RGB tubelet per video from 16x256x256 clips (frame stride 4)','samples':int(d['sample_count']),'dimension':int(d['feature_dimension']),'ranks':r}
OUT.with_name(OUT.name+'_sources.json').write_text(json.dumps(meta,indent=2)+'\n'); print(json.dumps(meta,indent=2))
