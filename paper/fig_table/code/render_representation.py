from pathlib import Path
import csv,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'paper/fig_table/figure/manuscript_rewrite_2026_09_14'
OUT.parent.mkdir(parents=True, exist_ok=True)
models=[('plain_jit_velocity','JiT velocity','#D55E00','s','-'),('plain_jit_clean','JiT clean','#0072B2','o','-'),('deco_b_velocity','DeCo','#A07900','^','--'),('pixeldit_b_velocity','PixelDiT','#009E73','D','-.'),('dip_b_velocity','DiP','#B45A9A','X',':')]
senspath=ROOT/'analysis_reports/12_normalized_representation_sensitivity/per_event.csv'
sens=list(csv.DictReader(senspath.open()))
probes=[]
paths=[ROOT/f'analysis_reports/04_hidden_noise_linear_probe/{g}/data/accuracy.csv' for g in ('group1','group1_semantic')]
for p in paths:probes+=list(csv.DictReader(p.open()))
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':14,'axes.titlesize':14,'axes.labelsize':14,'xtick.labelsize':12,'ytick.labelsize':12,'legend.fontsize':14,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
fig=plt.figure(figsize=(12,4.05))
gs=fig.add_gridspec(2,3,left=.075,right=.99,bottom=.14,top=.81,wspace=.27,hspace=.45)
axes=[fig.add_subplot(gs[r,c]) for r in range(2) for c in range(3)]
selected_s=[];selected_p=[]
for j,t in enumerate([.25,.5,.75]):
    for model,label,color,marker,ls in models:
        rr=sorted([r for r in sens if r['model']==model and float(r['t_clean'])==t and r['point'].startswith('workspace/') and r['point'].endswith('_mlp')],key=lambda r:int(r['point'].split('/b')[1].split('_')[0]))
        assert len(rr)==12
        y=np.array([float(r['ratio_corrected']) for r in rr]);assert y.min()>=0 and y.max()<={.25:5,.5:12,.75:90}[t]
        axes[j].plot(range(1,13),y,label=label,color=color,marker=marker,ls=ls,lw=2.1,ms=5.3,markevery=2)
        selected_s+=rr
        rr=sorted([r for r in probes if r['model']==model and float(r['alpha'])==1-t and r['mode']=='matched_noise'],key=lambda r:int(r['after_block']))
        assert len(rr)==11
        x=[int(r['after_block']) for r in rr];y=np.array([float(r['top1']) for r in rr]);sd=np.array([float(r['top1_seed_std']) for r in rr]);assert min(y-sd)>=0 and max(y+sd)<45
        axes[j+3].plot(x,y,color=color,marker=marker,ls=ls,lw=2.1,ms=5.3,markevery=2)
        axes[j+3].fill_between(x,y-sd,y+sd,color=color,alpha=.16,lw=0)
        selected_p+=rr
    axes[j].set_ylim(0,{.25:5,.5:12,.75:90}[t]);axes[j].set_yticks({.25:[0,2,4],.5:[0,6,12],.75:[0,45,90]}[t])
    axes[j+3].set_ylim(0,45);axes[j+3].set_yticks([0,20,40])
    if j:axes[j+3].tick_params(labelleft=False)
    for k in [j,j+3]:
        if k<3: axes[k].set_title(f'$t={t:.2f}$',pad=4)
        axes[k].grid(axis='y',alpha=.22);axes[k].set_axisbelow(True)
        axes[k].set_xlim(.7,12.3 if k<3 else 11.3)
        axes[k].set_xticks([1,6,12] if k<3 else [1,6,11])
axes[0].set_ylabel('Image / noise\n$B/W$',labelpad=3)
axes[3].set_ylabel('Probe\nTop-1 (%)',labelpad=3)
for ax in axes[3:]: ax.set_xlabel('Residual stream',labelpad=2)
fig.legend(*axes[0].get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.53,1.005),ncol=5,frameon=False,columnspacing=1.1,handlelength=2)
for ext in ('pdf','png'):fig.savefig(OUT/f'representation_six.{ext}',dpi=200,bbox_inches='tight',pad_inches=.02)
(OUT/'representation_six_sources.json').write_text(json.dumps({'sensitivity_source':str(senspath.relative_to(ROOT)),'probe_sources':[str(p.relative_to(ROOT)) for p in paths],'t_clean':[.25,.5,.75],'noise_alpha':'1-t','sensitivity_observation':'MLP inputs, blocks 1-12, patch RMS','probe_observation':'after blocks 1-11, patch RMS then GAP','sensitivity_rows':selected_s,'probe_rows':selected_p},indent=2))
print('Validated 180 sensitivity and 165 probe points, five models, no clipped values.')
