"""Pair cached embedding overlaps with top/bottom-64 gain shares. No inference."""
from pathlib import Path
import csv
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.ticker import PercentFormatter

PAPER=Path(__file__).resolve().parents[2]
DATA=PAPER/'fig_table/result_statistic/section03_analysis/01_patch_embedding_matrix/data'
OUT=PAPER/'fig_table/figure/manuscript_panels'
MODELS=['JiT-B velocity','JiT-B clean','DeCO-B','PixelDiT-B','DiP-B']
LABELS=['JiT velocity','JiT clean','DeCo','PixelDiT','DiP']
def rows(path):
    with path.open() as f:return list(csv.DictReader(f))
bins=rows(DATA/'gain_rank_bins.csv')
rankwise=rows(DATA/'rankwise_embedding_metrics.csv')
assert len(rankwise)==768 and [int(r['rank']) for r in rankwise]==list(range(1,769))
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':7,'axes.linewidth':.5,'pdf.fonttype':42,'ps.fonttype':42})
fig=plt.figure(figsize=(5.8,2.43))
panel_width=(.938-.069)/(5+4*.13)
heat_height=panel_width*5.8/2.43
heat_axes=[]
bar_axes=[]
with np.load(DATA/'raw_pixel_patch_pca.npz') as z:
    _,u=np.linalg.eigh(z['covariance']);u=u[:,::-1]
metrics=[]
with np.load(DATA/'patch_embedding_matrices.npz') as z:
    names=list(z['model_names'])
    for col,(name,label) in enumerate(zip(MODELS,LABELS)):
        left=.069+col*panel_width*1.13
        ax=fig.add_axes([left,.90-heat_height,panel_width,heat_height],sharey=heat_axes[0] if heat_axes else None)
        bar=fig.add_axes([left,.10,panel_width,.27],sharey=bar_axes[0] if bar_axes else None)
        heat_axes.append(ax);bar_axes.append(bar)
        idx=names.index(name);_,v=np.linalg.eigh(z[f'gram_{idx}']);v=v[:,::-1]
        overlap=(u[:,:64].T@v[:,:64])**2
        im=ax.imshow(overlap.T,origin='lower',cmap='magma',norm=LogNorm(1e-6,.5),extent=[.5,64.5,.5,64.5],interpolation='none')
        ax.set_title(label,fontsize=7.5,pad=4)
        ax.set_xticks([1,64]);ax.set_yticks([1,64]);ax.tick_params(length=1,pad=1,labelsize=5.8)
        ax.get_xticklabels()[0].set_ha('left');ax.get_xticklabels()[-1].set_ha('right')
        ax.set_xlabel('Raw Patch PCA Rank',fontsize=5.8,labelpad=2)
        if col==0:ax.set_ylabel(r'$W_{\mathrm{in}}$ PCA rank',fontsize=6.8,labelpad=2)
        else:ax.tick_params(labelleft=False)
        gain=np.array([float(r[name+' gain']) for r in rankwise])
        vals=[gain[:64].sum()/gain.sum(),gain[-64:].sum()/gain.sum()]
        for subset,val in zip(['Top 64','Bottom 64'],vals):
            archived=float(next(r['gain_share'] for r in bins if r['model']==name and r['rank_bin']==subset))
            assert np.isclose(val,archived,rtol=1e-10,atol=1e-12)
            metrics.append({'model':name,'rank_bin':subset,'raw_gain_sum':float(gain[:64].sum() if subset=='Top 64' else gain[-64:].sum()),'total_gain':float(gain.sum()),'gain_share':float(val)})
        bar.bar([0,1],vals,width=.58,color=['#9BC7DF','#EBC49D'],edgecolor=['#427D9F','#AB7748'],linewidth=.45,zorder=3)
        for x,val in enumerate(vals):
            labelval=f'{val*100:.2f}%' if val<.01 else f'{val*100:.1f}%'
            bar.text(x,val+.024,labelval,ha='center',va='bottom',fontsize=6.2,color='#26323B')
        bar.set(ylim=(0,.90),xlim=(-.55,1.55),xticks=[0,1],xticklabels=['Top 64','Bottom 64'],yticks=[0,.4,.8])
        bar.yaxis.set_major_formatter(PercentFormatter(xmax=1,decimals=0))
        bar.tick_params(axis='x',length=0,pad=3,labelsize=5.7)
        bar.tick_params(axis='y',length=2,pad=2,labelsize=6)
        bar.spines[['top','right']].set_visible(False)
        bar.grid(axis='y',color='#DDE2E6',linewidth=.4,zorder=0)
        if col==0:bar.set_ylabel('Gain share',fontsize=6.8,labelpad=2)
        else:bar.tick_params(labelleft=False)
        if col==4:last=ax
fig.canvas.draw()
pos=last.get_position()
cax=fig.add_axes([.949,pos.y0,.010,pos.height])
fig.colorbar(im,cax=cax,ticks=[1e-5,1e-3,1e-1]);cax.tick_params(labelsize=5.8,length=1,pad=1)
for heat,bar in zip(heat_axes,bar_axes):
    assert np.allclose([heat.get_position().x0,heat.get_position().width],[bar.get_position().x0,bar.get_position().width])
for ext in ['pdf','png']:
    fig.savefig(OUT/f'alignment_gain64.{ext}',dpi=300,bbox_inches='tight',pad_inches=.025)
plt.close(fig)
with (OUT/'alignment_gain64_values.csv').open('w') as f:
    writer=csv.DictWriter(f,fieldnames=list(metrics[0]));writer.writeheader();writer.writerows(metrics)
(OUT/'alignment_gain64_provenance.json').write_text(json.dumps({'sources':[str(p.relative_to(PAPER)) for p in [DATA/'raw_pixel_patch_pca.npz',DATA/'patch_embedding_matrices.npz',DATA/'rankwise_embedding_metrics.csv',DATA/'gain_rank_bins.csv']], 'models':MODELS,'directions':768,'heatmap':'squared overlap of top-64 eigenvectors, transposed: x raw patch PCA rank; y W_in PCA rank; heatmap and bar widths equal','bars':'sum of directional gains in ranks 1-64 or 705-768, divided by sum over all 768','checks':'10 shares recomputed from rankwise cache agree with archived bins at rtol 1e-10'},indent=2)+'\n')
print('Rendered five heatmaps and 10 verified gain-share bars.')
