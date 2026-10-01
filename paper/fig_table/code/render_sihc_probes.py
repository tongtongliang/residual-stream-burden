"""Compose paper panels from the archived validation means. No model execution."""
from pathlib import Path
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / 'analysis_reports/04_hidden_noise_linear_probe'
OUT = ROOT / 'paper/fig_table/figure/manuscript_rewrite_2026_09_14'
OUT.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.titlesize':9,
                     'axes.labelsize':8,'legend.fontsize':8,'pdf.fonttype':42,
                     'axes.spines.top':False,'axes.spines.right':False})
rows=[]
for group in ('group1',):
    with (DATA/group/'data/accuracy.csv').open() as f:
        rows += [r for r in csv.DictReader(f) if r['mode']=='matched_noise']

def curve(ax, model, alpha, label, color):
    values=sorted([r for r in rows if r['model']==model and float(r['alpha'])==alpha],
                  key=lambda r:int(r['after_block']))
    assert len(values)==11
    x=np.array([int(r['after_block']) for r in values])
    y=np.array([float(r['top1']) for r in values])
    sd=np.array([float(r['top1_seed_std']) for r in values])
    ax.plot(x,y,color=color,label=label,lw=1.6,marker='o',ms=2.7)
    ax.fill_between(x,y-sd,y+sd,color=color,alpha=.15,lw=0)

def style(ax):
    ax.set_xlim(.8,11.2);ax.set_xticks([1,3,5,7,9,11]);ax.set_xlabel('Block')
    ax.grid(axis='y',alpha=.17);ax.set_axisbelow(True)

fig,axes=plt.subplots(1,3,figsize=(6.55,1.94),sharey=True)
for ax,alpha in zip(axes,[.25,.5,.75]):
    curve(ax,'plain_jit_clean',alpha,'Pixel $\\boldsymbol{x}$-pred.','#2879b9')
    curve(ax,'plain_jit_velocity',alpha,'Pixel $\\boldsymbol{v}$-pred.','#e38127')
    curve(ax,'sihc_sublayer_p4',alpha,'SiHC','#009e73')
    style(ax);ax.set_title(f'{int(alpha*100)}% input noise');ax.set_ylim(0,40)
axes[0].set_ylabel('Top-1 accuracy (%)');axes[0].set_yticks([0,10,20,30,40])
fig.legend(*axes[0].get_legend_handles_labels(),loc='upper center',ncol=3,
           frameon=False,bbox_to_anchor=(.5,1.06))
fig.subplots_adjust(left=.085,right=.995,bottom=.25,top=.77,wspace=.15)
for ext in ('pdf','png'):fig.savefig(OUT/f'sihc_probes.{ext}',dpi=240,bbox_inches='tight',pad_inches=.02)
plt.close(fig)
print('Rendered SiHC probe panels from matched-noise CSV values.')
