"""Render anonymous paper figures from repository-relative aggregate CSV.

Requires numpy and matplotlib. Run this script from any working directory.
No model execution, external requests or file hashes are involved.
"""
import csv
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'paper_assets'
OUT.mkdir(exist_ok=True)
rows=list(csv.DictReader((ROOT/'data/aggregate.csv').open()))
cells={tuple(r[k] for k in ('family','size','frequency','phase','recompute','implementation')):r for r in rows}
SIZES=['B','L','XL','H']
ARMS=['literal','fused']
COLORS=['#8998a5','#247b87']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
                     'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42})

def get(size,freq,phase,mode,arm):
    return cells['sihc',size,freq,phase,mode,arm]

def save(fig,name):
    fig.savefig(OUT/f'{name}.pdf',bbox_inches='tight',metadata={
        'Creator':None,'Producer':None,'CreationDate':None,'ModDate':None,
        'Title':name.replace('_',' ')})
    fig.savefig(OUT/f'{name}.png',dpi=170,bbox_inches='tight',metadata={'Software':'SiHC figure renderer'})
    plt.close(fig)

def grouped(ax,records,key,labels,colors):
    width=.23 if len(labels)==3 else .30
    maximum=max(float(r[key]) for group in records for r in group if r.get(key))
    top=maximum*1.23
    for j,(label,color) in enumerate(zip(labels,colors)):
        for i,group in enumerate(records):
            r=group[j]; x=i+(j-(len(labels)-1)/2)*width
            if not r.get(key):
                ax.text(x,top*.93,'OOM',rotation=90,ha='center',va='top',fontsize=8,color='#7f3434')
                ax.plot(x,top*.97,marker='x',color='#7f3434',ms=5)
                continue
            value=float(r[key]);ax.bar(x,value,width*.9,color=color,zorder=3)
            if key=='ms':
                lo,hi=float(r['min_ms']),float(r['max_ms'])
                ax.errorbar(x,value,yerr=np.array([[value-lo],[hi-value]]),fmt='none',ecolor='#333333',capsize=2,lw=.8,zorder=4)
            ax.annotate(f'{value:.0f}' if key=='ms' else f'{value:.1f}',(x,value),xytext=(0,4),
                        textcoords='offset points',ha='center',fontsize=7)
    ax.set_ylim(0,top);ax.set_xticks(range(4),SIZES)
    ax.grid(axis='y',alpha=.18,zorder=0)
    ax.set_ylabel('Step time (ms)' if key=='ms' else 'Peak allocated (GiB)')

for phase in ['train','inference']:
    fig,axes=plt.subplots(2,2,figsize=(10,6.5),constrained_layout=True)
    for col,freq in enumerate(['block','sublayer']):
        records=[]
        for size in SIZES:
            mode='mlp' if phase=='train' and size in ('XL','H') else 'none'
            records.append([get(size,freq,phase,mode,arm) for arm in ARMS])
        for row,key in enumerate(['ms','allocated_gib']):grouped(axes[row,col],records,key,ARMS,COLORS)
        axes[0,col].set_title('Block-wise' if freq=='block' else 'Sublayer-wise')
        for ax in axes[:,col]:
            if phase=='train':ax.set_xticks(range(4),['B\nnone','L\nnone','XL\nMLP','H\nMLP'])
    title='Training: compile default + DDP(1), AdamW + two EMAs' if phase=='train' else 'Inference: compile reduce-overhead, one denoiser forward'
    fig.suptitle(title+'\nH100 80GB · batch 128 · FP32 parameters / BF16 autocast',fontsize=12)
    fig.legend(handles=[Patch(color=c,label=l.capitalize()+' Torch' if l!='fused' else 'Fused kernels')
                        for l,c in zip(ARMS,COLORS)],loc='outside lower center',ncol=3,frameon=False)
    save(fig,f'fusion_{phase}')

fig,axes=plt.subplots(2,2,figsize=(10,6.5),constrained_layout=True)
for col,freq in enumerate(['block','sublayer']):
    records=[[get(size,freq,'train',mode,'fused') for mode in ['none','mlp']] for size in SIZES]
    for row,key in enumerate(['ms','allocated_gib']):grouped(axes[row,col],records,key,['none','mlp'],['#54799b','#dfab50'])
    axes[0,col].set_title('Block-wise' if freq=='block' else 'Sublayer-wise')
fig.suptitle('Fused training: recomputation trade-off\nH100 80GB · batch 128 · compile default + DDP(1)',fontsize=12)
fig.legend(handles=[Patch(color=c,label=l) for c,l in zip(['#54799b','#dfab50'],['No recompute','MLP recompute'])],
           loc='outside lower center',ncol=2,frameon=False)
save(fig,'fusion_recompute')

lines=[r'\begin{table}[t]',r'\centering\scriptsize',
       r'\caption{Matched SiHC routing implementations at batch 128 on H100 80GB. Training uses compile default followed by DDP with world size 1, AdamW and two EMAs; inference uses reduce-overhead. No recomputation for B/L, MLP recomputation for XL/H. Entries are medians of three fresh processes. OOM retains the requested batch. Allocated memory excludes driver/library allocations.}',
       r'\label{tab:fusion-matched}',r'\begin{tabular}{lllrrrr}',r'\toprule',
       r'Topology & Size & Implementation & Train ms & Train GiB & Infer ms & Infer GiB \\',r'\midrule']
for freq in ['block','sublayer']:
    for size in SIZES:
        for arm in ARMS:
            train=get(size,freq,'train','mlp' if size in ('XL','H') else 'none',arm)
            infer=get(size,freq,'inference','none',arm)
            values=[f'{float(r[k]):.2f}' if r.get(k) else 'OOM' for r,k in [(train,'ms'),(train,'allocated_gib'),(infer,'ms'),(infer,'allocated_gib')]]
            lines.append(' & '.join([freq,size,arm]+values)+r' \\')
    lines.append(r'\midrule')
lines[-1:]=[r'\bottomrule',r'\end{tabular}',r'\end{table}']
(OUT/'fusion_table.tex').write_text('\n'.join(lines)+'\n')
print('Rendered three PDF/PNG figures and a LaTeX table from data/aggregate.csv.')
