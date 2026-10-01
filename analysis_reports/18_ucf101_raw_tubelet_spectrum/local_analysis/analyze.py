from pathlib import Path
from collections import Counter
import numpy as np,csv,json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(__file__).resolve().parents[1]; out=root/'local_analysis'
items=[('imagenet_val_r1024_p32','ImageNet: 32×32'),('ucf101_t16_r256_pt4_p16','UCF101: 4×16×16'),('ucf101_t128_r128_pt16_p8','UCF101: 16×8×8')]
res={}; fig,ax=plt.subplots(1,2,figsize=(9,3.3),layout='constrained')
for name,label in items:
 d=np.load(root/'data/spectra'/f'{name}.npz'); e=d['eigenvalues']; tr=float(d['total_variance']); cum=np.cumsum(e)/tr
 assert len(e)==512 and np.all(e>=0) and np.all(np.diff(e)<=1e-8)
 assert np.isclose(e.sum()+float(d['unresolved_variance']),tr)
 ranks={f'r{int(q*100)}':int(np.searchsorted(cum,q)+1) for q in [.9,.95,.99]}
 tail=np.cumsum(e[1:])/(tr-e[0]); ranks['r90_after_removing_PC1']=int(np.searchsorted(tail,.9)+1)
 res[name]=dict(**ranks,pc1_share=float(e[0]/tr),top8_share=float(cum[7]),top512_share=float(cum[-1]),sample_count=int(d['sample_count']),stable_rank=float(tr/e[0]))
 ax[0].plot(np.arange(1,513),cum*100,label=label)
 ax[1].loglog(np.arange(1,513),e/tr,label=label)
ax[0].set(xscale='log',xlabel='Principal component rank',ylabel='Cumulative variance (%)',ylim=(70,100.2));ax[0].axhline(90,color='.6',ls=':',lw=1)
ax[1].set(xlabel='Principal component rank',ylabel='Eigenvalue / total variance')
for a in ax:a.grid(alpha=.2);a.spines[['top','right']].set_visible(False)
ax[0].legend(fontsize=8,loc='lower right')
fig.savefig(out/'tubelet_spectrum_comparison.png',dpi=180);fig.savefig(out/'tubelet_spectrum_comparison.pdf')
for f in (root/'data/manifests').glob('*.csv'):
 with f.open() as h: rows=list(csv.DictReader(h))
 classes=Counter(r['class_name'] for r in rows)
 res[f.stem]={'rows':len(rows),'classes':len(classes),'videos_per_class':sorted(set(classes.values())),'unique_paths':len(set(r['relative_path'] for r in rows)),'temporal_sampling':dict(Counter(r['temporal_sampling'] for r in rows))}
(out/'verified_metrics.json').write_text(json.dumps(res,indent=2)+'\n');print(json.dumps(res,indent=2))
