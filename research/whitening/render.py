"""Portable cached-data renderer; no model loading or external paths."""
from pathlib import Path
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parent
out = ROOT / 'figures'
out.mkdir(exist_ok=True)
a = np.load(ROOT / 'data/spectra.npz')
fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
rank = np.arange(1, 769)
for target, color in [('clean', 'tab:blue'), ('velocity', 'tab:orange')]:
    for state, style in [('model', '-'), ('ema', '--')]:
        s = a[f'whitening_{target}_{state}_whiten_sv']
        axes[0].semilogy(rank, s, color=color, ls=style, label=target+' '+state)
    for label, space, style in [('whitening', 'whiten', '-'), ('pixel', 'rgb_effective', ':')]:
        s = a[f'{label}_{target}_model_{space}_sv']
        axes[1].plot(rank, np.cumsum(s*s)/np.sum(s*s), color=color, ls=style, label=label+' '+target)
    for label, style in [('whitening', '-'), ('pixel', ':')]:
        s = a[f'{label}_{target}_model_rgb_effective_sv']
        axes[2].semilogy(rank, s, color=color, ls=style, label=label+' '+target)
for ax, title in zip(axes, ['Whitening-coordinate singular values', 'Native Gram cumulative energy (raw)', 'RGB-effective singular values (raw)']):
    ax.set_title(title); ax.set_xlabel('Singular value rank'); ax.grid(alpha=.2); ax.legend(fontsize=8)
fig.tight_layout()
for ext in ['png', 'pdf']: fig.savefig(out/f'spectrum.{ext}', dpi=180)
plt.close(fig)
with (ROOT/'data/evaluation_history.csv').open() as f: rows=list(csv.DictReader(f))
fig, axes = plt.subplots(1,2,figsize=(9,3.8))
for target in ['clean','velocity']:
    rs=sorted([r for r in rows if r['model']==target],key=lambda r:int(r['checkpoint_step']))
    epochs=[int(r['checkpoint_step'])/1251 for r in rs]
    for ax,key in zip(axes,['fid','inception_score_mean']):
        ax.plot(epochs,[float(r[key]) for r in rs],marker='o',label=target)
for ax,label in zip(axes,['FID (50K, primary EMA)','IS (50K, primary EMA)']):
    ax.set_xlabel('Epoch'); ax.set_ylabel(label); ax.legend(); ax.grid(alpha=.2)
fig.tight_layout()
for ext in ['png','pdf']: fig.savefig(out/f'fid_is.{ext}',dpi=180)
print('Rendered figures to',out)
