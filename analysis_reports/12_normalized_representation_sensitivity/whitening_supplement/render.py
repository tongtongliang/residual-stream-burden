#!/usr/bin/env python3
"""Render cached scientific figures; CPU only, no checkpoints/network/inference."""
import argparse,csv
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent
p=argparse.ArgumentParser()
p.add_argument('--output-dir',type=Path,default=ROOT/'figures')
args=p.parse_args();out=args.output_dir;out.mkdir(parents=True,exist_ok=True)
with (ROOT/'per_event.csv').open() as f: per_event=list(csv.DictReader(f))
assert len(per_event)==144
for row in per_event:
    for key in row:
        if key not in ('model','point'):row[key]=float(row[key])
colors={'clean':'#2166ac','velocity':'#d6604d'}
plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
fig,axes=plt.subplots(1,3,figsize=(14,4.3),layout='constrained')
for ax,t in zip(axes,(.25,.5,.75)):
 for pred in ('clean','velocity'):
  rs=[r for r in per_event if r['model']=='whiten_'+pred and r['t_clean']==t]
  ax.plot(range(1,25),[r['ratio_corrected'] for r in rs],label='Whiten '+pred,color=colors[pred],lw=2)
 ax.set(title=f'Clean fraction t = {t:.2f}',xlabel='Residual event (attention, MLP)')
 ax.grid(alpha=.2);ax.set_xticks([1,4,8,12,16,20,24])
axes[0].set_ylabel('R = corrected B / W');axes[-1].legend()
for ext in ('png','pdf'):fig.savefig(out/f'ratio_by_depth.{ext}',dpi=180)
plt.close(fig)
fig,axes=plt.subplots(2,3,figsize=(14,7),layout='constrained')
for col,t in enumerate((.25,.5,.75)):
 for row,(key,label) in enumerate((('between_corrected_per_coordinate','B / D (corrected)'),('within_per_coordinate','W / D'))):
  ax=axes[row,col]
  for pred in ('clean','velocity'):
   rs=[r for r in per_event if r['model']=='whiten_'+pred and r['t_clean']==t]
   ax.plot(range(1,25),[r[key] for r in rs],label=pred,color=colors[pred],lw=2)
  ax.set_title(f'Clean fraction t = {t:.2f}');ax.set_ylabel(label);ax.grid(alpha=.2)
  ax.set_xticks([1,4,8,12,16,20,24])
  if row==1:ax.set_xlabel('Residual event (attention, MLP)')
axes[0,-1].legend()
for ext in ('png','pdf'):fig.savefig(out/f'variance_by_depth.{ext}',dpi=180)
plt.close(fig)

print(f'Rendered four PNG/PDF figures into {out}')
