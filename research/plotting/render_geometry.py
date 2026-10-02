"""Render the manuscript whitening figure from cached data only."""
from pathlib import Path
import csv,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=next(p for p in Path(__file__).resolve().parents if (p / 'pyproject.toml').is_file()); P=R/'research'; W=R/'research/whitening'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.titlesize':9.5,'axes.labelsize':9,'xtick.labelsize':8,'ytick.labelsize':8,'legend.fontsize':7.5,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
def save(fig,path):
 path.parent.mkdir(parents=True,exist_ok=True)
 for ext in ['pdf','png']:fig.savefig(path.with_suffix('.'+ext),bbox_inches='tight',pad_inches=.03,dpi=190)
 plt.close(fig)
(P/'figures/provenance').mkdir(parents=True,exist_ok=True)
# Whitening: the covariance implied by the saved invertible transform, learned
# weight spectra in native coordinates, and the final observed FID.
a=np.load(W/'data/spectra.npz');m=np.load(W/'data/embedding_matrices.npz')
scales=m['clean_scales']; lam=scales.astype(float)**2
assert scales.shape==(768,) and np.min(scales)>0
hist=list(csv.DictReader((W/'data/evaluation_history.csv').open()))
final={r['model']:float(r['fid']) for r in hist if int(r['checkpoint_step'])==250200}
fig,ax=plt.subplots(1,3,figsize=(7.0,2.05));fig.subplots_adjust(left=.075,right=.99,bottom=.25,top=.77,wspace=.47)
x=np.arange(1,769)
ax[0].semilogy(x,lam/lam.mean(),color='#0072B2',lw=1.3,label='Raw pixels')
ax[0].axhline(1,color='#D55E00',ls='--',lw=1.3,label='Whitened')
ax[0].set(xlim=(1,768),xticks=[1,384,768],xlabel='Patch PCA rank',ylabel='Variance / mean',title='(a) Input geometry')
ax[0].legend(frameon=False,loc='upper right',fontsize=6.7)
for key,label,color,ls in [('pixel_clean_model_rgb_effective_sv','Pixel clean','#0072B2','-'),('pixel_velocity_model_rgb_effective_sv','Pixel velocity','#D55E00','-'),('whitening_clean_model_whiten_sv','Whitened clean','#0072B2','--'),('whitening_velocity_model_whiten_sv','Whitened velocity','#D55E00','--')]:
 sv=a[key];energy=sv**2;ax[1].plot(x,np.cumsum(energy)/energy.sum(),label=label,color=color,ls=ls,lw=1.3)
ax[1].set(xlim=(1,768),ylim=(0,1.03),xticks=[1,384,768],yticks=[0,.5,1],xlabel='Singular direction rank',ylabel='Cumulative energy',title='(b) Learned embedding')
ax[1].legend(frameon=False,loc='lower right',fontsize=6,labelspacing=.18)
raw=10.192;white=final['clean'];bars=ax[2].bar([0,1],[raw,white],color=['#0072B2','#D55E00'],width=.58)
for b,v in zip(bars,[raw,white]):ax[2].text(b.get_x()+b.get_width()/2,v+3,f'{v:.2f}',ha='center',fontsize=8)
ax[2].set(xticks=[0,1],xticklabels=['Raw','Whitened'],ylim=(0,160),yticks=[0,50,100,150],ylabel='FID (50K)',title='(c) Clean prediction')
for z in ax:z.grid(axis='y',alpha=.18);z.set_axisbelow(True)
save(fig,P/'figures/patch_geometry/whitening_geometry_and_fid')
(P/'figures/provenance/whitening_sources.json').write_text(json.dumps({'whitening_root':str(W.relative_to(R)),'final_fid':final,'raw_clean_fid':raw,'spectrum':'saved whitening scales squared and ideal normalized whitened covariance; model raw-weight native-coordinate Gram energy','min_scale':float(scales.min())},indent=2))
print('Whitening final FID:',final)
