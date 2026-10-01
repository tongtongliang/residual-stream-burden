"""Render the measured SiHC scaling results; no fitted or interpolated data."""
from pathlib import Path
import json
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / 'paper/notes/implementation_sources/scaling'
OUT = ROOT / 'paper/figures/scaling/sihc_scaling'
OUT.parent.mkdir(parents=True, exist_ok=True)
df = pd.read_csv(SOURCE / 'fid_history.csv')
costs = json.loads((SOURCE / 'compute/counts.json').read_text())['rows']
params = {r['model'].removeprefix('SiHC-'): r['parameters']/1e6
          for r in costs if r['model'].startswith('SiHC-')}
plt.rcParams.update({'font.family':'serif', 'font.size':10, 'pdf.fonttype':42,
                     'axes.spines.top':False, 'axes.spines.right':False})
fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.55),
                        gridspec_kw={'width_ratios':[1, 1.3]})
s = df[df.epoch == 200].set_index('size').loc[['B','L','H']]
x = [params[k] for k in s.index]
axs[0].plot(x, s.fid, 'o-', color='#1B637A', lw=1.8, ms=4.5)
axs[0].set(xlabel='Parameters (M)', ylabel=r'FID $\downarrow$',
           title='(a) Scaling at 200 epochs', xlim=(50,1060), ylim=(0,7.6))
axs[0].set_xticks([131,460,954])
for p, size, v in zip(x, s.index, s.fid):
    axs[0].annotate(f'{size}: {v:.2f}', (p,v), xytext=(0,8),
                    textcoords='offset points', ha='center', fontsize=9)
h = df[(df['size']=='H') & (df.epoch>=200)]
axs[1].plot(h.epoch, h.fid, 'o-', color='#AB542F', ms=3.5, lw=1.6)
axs[1].set(xlabel='Training epoch', ylabel=r'FID $\downarrow$',
           title='(b) SiHC-H: continued training',
           xlim=(180,620), ylim=(2.0,2.6))
axs[1].set_xticks([200,300,400,500,600])
axs[1].annotate('2.06', (400,2.0568455548333304), xytext=(-12,13),
                textcoords='offset points', fontsize=9)
for a in axs:
    a.grid(axis='y', alpha=.18)
fig.tight_layout(pad=.4, w_pad=1.5)
for ext in ['pdf','png']:
    fig.savefig(OUT.with_suffix('.'+ext), bbox_inches='tight', dpi=180)
meta = dict(source=str((SOURCE/'fid_history.csv').relative_to(ROOT)),
            costs=str((SOURCE/'compute/counts.json').relative_to(ROOT)),
            protocol='Sublayer SiHC, ctx32, no REPA; 50K samples, Heun50; CFG B/L/H=2.9/2.4/2.1.',
            display='Measured points joined by straight segments; no fitted scaling exponent or smoothed trajectory.')
OUT.with_name(OUT.name+'_sources.json').write_text(json.dumps(meta,indent=2)+'\n')
