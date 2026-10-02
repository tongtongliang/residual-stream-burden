"""Render archived whitening probes without checkpoints or GPUs."""
from pathlib import Path
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent
rows=list(csv.DictReader((ROOT/'data/accuracy.csv').open()))
fig,axes=plt.subplots(2,3,figsize=(12,6),sharex=True)
for col,q in enumerate([.25,.5,.75]):
    for model,color in [('whitening_jit_clean','tab:blue'),('whitening_jit_velocity','tab:orange')]:
        selected=sorted([r for r in rows if r['model']==model and r['mode']=='matched_noise' and float(r['noise_fraction'])==q],key=lambda r:int(r['after_block']))
        x=np.array([int(r['after_block']) for r in selected])
        for row,metric in enumerate(['top1','top5']):
            y=np.array([float(r[metric]) for r in selected]);sd=np.array([float(r[metric+'_seed_std']) for r in selected])
            ax=axes[row,col];ax.plot(x,y,label=model.removeprefix('whitening_jit_'),color=color);ax.fill_between(x,y-sd,y+sd,color=color,alpha=.2)
            ax.set_ylabel(metric+' (%)');ax.grid(alpha=.2)
    axes[0,col].set_title(f'Noise q={q}');axes[1,col].set_xlabel('After block');axes[1,col].set_xticks([1,3,5,7,9,11])
axes[0,0].legend();fig.suptitle('Whitening: matched-noise linear probe (mean ± test-noise SD)');fig.tight_layout()
(ROOT/'figures').mkdir(exist_ok=True)
for suffix in ['png','pdf']:fig.savefig(ROOT/'figures'/('matched_noise_depth.'+suffix),dpi=180)
