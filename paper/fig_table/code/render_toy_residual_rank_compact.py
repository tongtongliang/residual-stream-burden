"""Compact two-panel 3D bar plot of toy residual-state r90 for the main text (cached data only)."""
from pathlib import Path
import json
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[3]
SRC=ROOT/'paper/fig_table/result_statistic/section03_analysis/04_toy_residual_rank/residual_rank_metrics.csv'
OUT=ROOT/'paper/figures/toy/residual_rank_compact'
OUT.parent.mkdir(parents=True, exist_ok=True)
df=pd.read_csv(SRC)
plt.rcParams.update({'font.family':'serif','font.size':10,'pdf.fonttype':42})
zmax=np.ceil((df.r90.max()+8)/10)*10
fig=plt.figure(figsize=(5.0,2.55))
for k,(pred,color,title) in enumerate([('clean','#B9DDF1','$\\boldsymbol{x}$-prediction'),('velocity','#C6E5B5','$\\boldsymbol{v}$-prediction')]):
    ax=fig.add_subplot(1,2,k+1,projection='3d',computed_zorder=False)
    sub=df[df.prediction==pred]; bars=[]
    for yi,t in enumerate(sorted(sub.t.unique())):
        for row in sub[np.isclose(sub.t,t)].itertuples():
            bars.append((row.block,yi,row.r90))
    bx,by,bz=np.array(bars).T
    ax.bar3d(bx-.24,by-.24,np.zeros_like(bz),.48,.48,bz,color=color,edgecolor='#65747B',linewidth=.5,shade=False,zorder=1)
    ax.set(xlim=(-.6,4.6),ylim=(4.5,-.5),zlim=(0,zmax),xticks=range(5),yticks=range(5),zticks=[0,100,200],yticklabels=['0.1','0.3','0.5','0.7','0.9'])
    ax.set_xlabel('Block',labelpad=-4);ax.set_ylabel('$t$',labelpad=-4);ax.set_zlabel('$r_{90}$',labelpad=-3)
    ax.set_title(title,pad=-2,fontsize=11)
    ax.view_init(elev=38,azim=-55);ax.set_box_aspect((1.2,1.4,.9))
    ax.tick_params(labelsize=8.5,pad=-1)
    for axis in [ax.xaxis,ax.yaxis,ax.zaxis]:
        axis.pane.fill=False;axis._axinfo['grid']['color']=(.7,.7,.7,.35)
fig.subplots_adjust(left=0,right=.94,bottom=0.02,top=.95,wspace=0.18)
fig.savefig(OUT.with_suffix('.pdf'),bbox_inches='tight',pad_inches=0.02);fig.savefig(OUT.with_suffix('.png'),dpi=180,bbox_inches='tight',pad_inches=0.02)
OUT.with_name(OUT.name+'_sources.json').write_text(json.dumps({'source':str(SRC.relative_to(ROOT)),'renderer':'paper/fig_table/code/render_toy_residual_rank_compact.py','content':'FCN-256 toy residual-state r90 at block inputs 0-4, t in {0.1,...,0.9}, x vs v prediction'},indent=2)+'\n')
print('ok')
