"""Matched next-attention workspace diagnostics from archived CSVs only."""
from pathlib import Path
import csv,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'paper/fig_table/figure/manuscript_rewrite_2026_09_14'
OUT.parent.mkdir(parents=True, exist_ok=True)
SENS=ROOT/'analysis_reports/12_normalized_representation_sensitivity/data'
models=[('plain_jit_clean','Pixel $\\boldsymbol{x}$-pred.','#2879b9','o'),('plain_jit_velocity','Pixel $\\boldsymbol{v}$-pred.','#d77419','s'),('sihc_sublayer_p4','SiHC','#00866b','^')]
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.labelsize':10.5,'axes.titlesize':11,'legend.fontsize':10,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
provenance={'sensitivity_sources':[], 'normalization':'per-token RMS','observation':'all 24 attention/MLP workspace inputs, before native normalization/modulation','times':[.25,.5,.75],'ratio':'ratio_corrected = (B_raw-W/128)/W','raw_rows':[]}
def rows(model,t):
 p=SENS/f'{model}_t{int(t*100):03d}'/'metrics.csv'
 rs=[r for r in csv.DictReader(p.open()) if r['point'].startswith('workspace/')]
 assert len(rs)==24
 if str(p.relative_to(ROOT)) not in provenance['sensitivity_sources']:
  provenance['sensitivity_sources'].append(str(p.relative_to(ROOT)))
  provenance['raw_rows'] += [dict(model=model,t=t,**r) for r in rs]
 return rs
def line(ax,x,y,label,color,marker):
 ax.plot(x,y,lw=1.7,marker=marker,ms=4.2,mew=.4,markeredgecolor='white',label=label,color=color)
def style(ax,xend):
 ax.set_xlim(.65,xend+.35);ax.set_xticks([1,3,5,7,9,11] if xend==11 else [1,6,12,18,24]);ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True);ax.set_ylim(bottom=0)
def save(fig,name):
 for ext in ('pdf','png'): fig.savefig(OUT/f'{name}.{ext}',dpi=240,bbox_inches='tight',pad_inches=.025)
 plt.close(fig)
fig,axes=plt.subplots(2,3,figsize=(8.1,4.8),sharex=True)
for col,t in enumerate([.25,.5,.75]):
 for model,label,color,marker in models:
  ss=rows(model,t)
  for row,key in enumerate(['within_per_coordinate','ratio_corrected']):
   line(axes[row,col],range(1,25),[float(r[key]) for r in ss],label,color,marker)
 for row in [0,1]:style(axes[row,col],24)
 axes[0,col].set_title(f'$t={t:.2f}$');axes[1,col].set_xlabel('Update')
axes[0,0].set_ylabel('Noise variance $W/D$');axes[1,0].set_ylabel('Between / within $B/W$')
fig.legend(*axes[0,0].get_legend_handles_labels(),loc='upper center',ncol=3,frameon=False,bbox_to_anchor=(.52,1.02))
fig.subplots_adjust(left=.085,right=.995,bottom=.13,top=.87,wspace=.30,hspace=.23)
save(fig,'sihc_sensitivity_all')
(OUT/'sihc_sensitivity_provenance.json').write_text(json.dumps(provenance,indent=2))
print('Rendered all-time SiHC sensitivity; archived 216 full-precision observations.')
