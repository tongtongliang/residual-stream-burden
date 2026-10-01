"""Learned input-skip surfaces from cached training records."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
ROOT=Path(__file__).resolve().parents[3]
SRC=ROOT/'analysis_reports/16_jit_b16_long_skip_velocity'
OUT=ROOT/'paper/figures/patch_geometry/learned_long_skip'
OUT.parent.mkdir(parents=True, exist_ok=True)
h=pd.read_csv(SRC/'data/wandb_skip_history_sampled.csv.gz')
g=pd.read_csv(SRC/'data/neg_alpha_grid_full.csv')
assert np.array_equal(h['_step'],g['step'])
t=np.linspace(.05,.95,20)
q=h[[f'skip/clean_alpha_effective_t_{v:.2f}' for v in t]].to_numpy()
epoch=g['epoch'].to_numpy()
# Use dense recorded epochs early and throughout training; never interpolate in epoch.
wanted=np.r_[np.geomspace(epoch[0],40,130),np.linspace(40,epoch[-1],160)]
idx=np.unique([np.argmin(abs(epoch-v)) for v in wanted])
ep=np.r_[0.,epoch[idx]]
# Known zero initialization is exact, not a reconstructed observation.
qs=np.vstack([np.zeros(20),q[idx]])
tfine=np.linspace(t[0],t[-1],150)
Q=PchipInterpolator(t,qs,axis=1)(tfine)
T,E=np.meshgrid(tfine,ep)
plt.rcParams.update({'font.family':'serif','font.size':9,'pdf.fonttype':42})
fig=plt.figure(figsize=(9,6.1))
gs=fig.add_gridspec(2,2,height_ratios=[2.1,1],hspace=.48)
for n,(Z,ref,zlabel,title) in enumerate([
 (Q/(1-T),1/(1-tfine),r'$-\alpha(t)$','(a) Learned skip coefficient'),
 (Q,np.ones_like(tfine),r'$-(1-t)\alpha(t)$','(b) Normalized skip coefficient')]):
 ax=fig.add_subplot(gs[0,n],projection='3d',computed_zorder=False)
 ax.plot_surface(T,E,Z,cmap='YlGnBu',rstride=1,cstride=1,linewidth=0,antialiased=False,alpha=1.0,rasterized=True,zorder=1)
 ax.plot(tfine,np.full_like(tfine,epoch[-1]),Z[-1],color='#12446b',lw=1.8,zorder=4)
 ax.plot(tfine,np.full_like(tfine,epoch[-1]),ref,color='#C34A26',lw=2.3,ls='--',zorder=5)
 ax.set(xlim=(.05,.95),ylim=(0,400),xlabel=r'Clean coefficient $t$',ylabel='Training epoch',zlabel=zlabel)
 ax.set_xticks([.1,.5,.9]);ax.set_yticks([0,100,200,300,400])
 ax.set_zlabel('')
 ax.text2D(-.115,.52,zlabel,transform=ax.transAxes,rotation=90,va='center',fontsize=10)
 ax.set_zlim(0,21 if n==0 else 1.08)
 ax.set_zticks([0,5,10,15,20] if n==0 else [0,.5,1])
 ax.view_init(elev=25,azim=-125)
 ax.set_box_aspect((1.15,1.3,.85));ax.tick_params(labelsize=8,pad=0)
 ax.xaxis.labelpad=3;ax.yaxis.labelpad=4;ax.zaxis.labelpad=3
 for axis in [ax.xaxis,ax.yaxis,ax.zaxis]:
  axis.pane.fill=False;axis._axinfo['grid'].update(color=(.83,.83,.83,.5),linewidth=.5)
 ax.set_title(title,pad=5,fontsize=10)
fig.legend([Line2D([0],[0],color='#12446b',lw=1.8),Line2D([0],[0],color='#C34A26',lw=2.3,ls='--')],['Final learned curve','Analytic reference at final epoch'],loc='lower center',ncol=2,frameon=False,fontsize=9,bbox_to_anchor=(.5,.365))
fid=pd.read_csv(SRC/'data/fid_is_50k.csv')
ax=fig.add_subplot(gs[1,:])
ax.plot(fid['epoch'],fid['fid'],'o-',color='#12446b',lw=1.6,ms=4)
ax.set(xlabel='Training epoch',ylabel=r'FID $\downarrow$',xlim=(30,420),ylim=(0,85))
ax.set_xticks(np.arange(40,401,40))
ax.set_yticks([0,20,40,60,80])
ax.spines[['top','right']].set_visible(False)
ax.grid(axis='y',alpha=.2)
ax.set_title('(c) Generation quality',fontsize=10,pad=7)
for ep_label,offset in [(200,(7,10)),(400,(-35,10))]:
 row=fid.loc[fid.epoch==ep_label].iloc[0]
 ax.annotate(f"{row.fid:.2f}",(row.epoch,row.fid),xytext=offset,textcoords='offset points',fontsize=9)
fig.subplots_adjust(left=.085,right=.96,bottom=.08,top=.95,wspace=.08)
fig.savefig(OUT.with_suffix('.pdf'),bbox_inches='tight',dpi=220)
fig.savefig(OUT.with_suffix('.png'),bbox_inches='tight',dpi=180)
OUT.with_name(OUT.name+'_sources.json').write_text(json.dumps({'source':str(SRC.relative_to(ROOT)),'recorded_snapshots':len(epoch),'rendered_recorded_epochs':epoch[idx].tolist(),'initialization':'alpha=0 exactly at epoch0','time_grid':'linspace(0.05,0.95,20), metric keys rounded','interpolation':'PCHIP along t only; sampled measured epochs with dense early coverage; no epoch smoothing','fid_source':'data/fid_is_50k.csv','fid_display':'all 10 measured evaluations, joined by straight segments','final_epoch':float(epoch[-1]),'raw_alpha_reconstruction':'-alpha=clean_alpha_effective/(1-t), using unrounded grid'},indent=2)+'\n')
print(len(idx),'recorded epochs rendered; last',epoch[-1])
