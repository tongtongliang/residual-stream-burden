"""Compact endpoint/embedding cumulative spectra from cached eigenvalues only."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=next(p for p in Path(__file__).resolve().parents if (p / 'pyproject.toml').is_file())
OUT=ROOT/'research/figures/endpoint_spectra';OUT.mkdir(parents=True,exist_ok=True)
OUT.parent.mkdir(parents=True, exist_ok=True)
raw=ROOT/'research/patch_embedding/figure_cache/data'
endpoint=ROOT/'research/semantic_endpoints/figure_cache'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.labelsize':8,'axes.titlesize':8.5,'xtick.labelsize':7,'ytick.labelsize':7,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(1,4,figsize=(7.1,1.48))
fig.subplots_adjust(left=.055,right=.99,bottom=.29,top=.86,wspace=.23)
x=np.arange(1,769)
def cumulative(v):
 v=np.maximum(np.sort(np.asarray(v,dtype=float))[::-1],0);assert v.shape==(768,) and v.sum()>0
 y=np.cumsum(v)/v.sum();assert np.all(np.diff(y)>=-1e-12) and np.isclose(y[-1],1)
 return y
clean=np.load(raw/'raw_pixel_patch_pca.npz')['eigenvalues']
records=[]
for ax,model,title in zip(axes[:3],['deco_b','pixeldit_b','dip_b'],['DeCo endpoint','PixelDiT endpoint','DiP endpoint']):
 path=endpoint/f'{model}_spectra.npz';a=np.load(path);assert a['eigenvalues'].shape==(9,768)
 for t,v,col in zip(a['t'],a['eigenvalues'],plt.cm.Purples(np.linspace(.32,.98,9))):ax.plot(x,cumulative(v),c=col,lw=.85)
 ax.plot(x,cumulative(clean),c='#0072B2',lw=1.1,ls='--',label='Clean target')
 ax.plot(x,cumulative(clean+1),c='#D55E00',lw=1.1,ls=':',label='Velocity target')
 ax.set_title(title,pad=3);records.append({'source':str(path.relative_to(ROOT)),'t':a['t'].tolist(),'curves':9})
a=np.load(raw/'patch_embedding_matrices.npz');names=list(a['model_names'])
for name,label,color,ls in [('JiT-B clean','JiT clean','#0072B2','--'),('JiT-B velocity','JiT velocity','#D55E00',':'),('DeCO-B','DeCo','#A18100','-'),('PixelDiT-B','PixelDiT','#009E73','-'),('DiP-B','DiP','#CC79A7','-')]:
 v=np.linalg.eigvalsh(a[f'gram_{names.index(name)}'].astype(float));axes[3].plot(x,cumulative(v),label=label,color=color,ls=ls,lw=1)
axes[3].set_title(r'$W_{\mathrm{in}}$',pad=3);axes[3].legend(loc='lower right',frameon=False,fontsize=6.4,handlelength=1.6,labelspacing=.2,borderpad=0)
for i,ax in enumerate(axes):
 ax.set(xlim=(1,768),ylim=(0,1.02),xticks=[1,384,768],yticks=[0,.5,1],xlabel='Direction rank');ax.grid(axis='y',color='#E4E4E4',lw=.4);ax.tick_params(length=2,pad=1.5)
 if i:ax.tick_params(labelleft=False)
axes[0].set_ylabel('Cumulative energy')
axes[0].legend(loc='lower right',frameon=False,fontsize=6.2,handlelength=1.8,labelspacing=.2,borderpad=0)
axes[1].text(.96,.07,r'$t=0.1\rightarrow0.9$'+'\n(light to dark)',transform=axes[1].transAxes,ha='right',va='bottom',fontsize=6.3)
for ext in ['pdf','png']:fig.savefig(OUT/f'endpoint_embedding_spectra.{ext}',dpi=260,bbox_inches='tight',pad_inches=.025)
(OUT/'endpoint_embedding_spectra_sources.json').write_text(json.dumps({'endpoints':records,'target_source':str((raw/'raw_pixel_patch_pca.npz').relative_to(ROOT)),'embedding_source':str((raw/'patch_embedding_matrices.npz').relative_to(ROOT)),'embedding_models':['JiT clean','JiT velocity','DeCo','PixelDiT','DiP'],'energy':'covariance eigenvalues for endpoints/targets; eigenvalues of W.T W for embeddings','checks':'27 endpoint curves; nine actual time points per model; all CDFs monotone and end at one'},indent=2)+'\n')
