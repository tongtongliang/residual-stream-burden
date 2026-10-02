"""Single-row target spectra and native-coordinate embedding alignment, cached data only.
Run with the Miniforge ml Python documented in AGENTS.md.
"""
from pathlib import Path
import json
import os
import numpy as np
ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'pyproject.toml').is_file())
OUT = ROOT / 'research/figures/section4_gain'
OUT.mkdir(parents=True, exist_ok=True)
os.environ.setdefault('MPLCONFIGDIR', str(OUT / '.matplotlib_cache'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from _alignment import draw_four, LABELS, COLORS
RAW = ROOT / 'research/patch_embedding/figure_cache/data'
WHITE = ROOT / 'research/whitening/data'
raw = np.load(RAW / 'raw_pixel_patch_pca.npz')
weights = np.load(RAW / 'patch_embedding_matrices.npz')
white = np.load(WHITE / 'embedding_matrices.npz')
_, raw_basis = np.linalg.eigh(raw['covariance'])
raw_basis = raw_basis[:, ::-1]
models = [weights['weight_0'], weights['weight_1'], white['whitening_clean_model_weight'], white['whitening_velocity_model_weight']]
targets = [raw['eigenvalues'], raw['eigenvalues'] + 1, np.ones(768), 2*np.ones(768)]
grams, overlaps = [], []
for i, w in enumerate(models):
    eigenvalues, vectors = np.linalg.eigh(w.astype(float).T @ w.astype(float))
    grams.append(eigenvalues[::-1])
    # Whitening outputs PCA coordinates, so the inherited input axes are I.
    basis = raw_basis if i < 2 else np.eye(768)
    overlap = (basis[:, :64].T @ vectors[:, ::-1][:, :64])**2
    assert np.isfinite(overlap).all() and overlap.min() >= 0 and overlap.max() <= 1+1e-12
    overlaps.append(overlap.T)

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'pdf.fonttype':42,'axes.linewidth':.6})
gains=[]
for i,w in enumerate(models):
    basis=raw_basis if i<2 else np.eye(768)
    g=np.sum((w@basis)**2,axis=0); gains.append(g/g.sum())
fig,axs=plt.subplots(2,4,figsize=(7.4,3.0),gridspec_kw={'height_ratios':[2.1,1]})
fig.subplots_adjust(left=.075,right=.935,bottom=.10,top=.9,wspace=.22,hspace=.55)
for i in range(4):
    ax=axs[0,i]
    im=ax.imshow(np.maximum(overlaps[i],1e-6),origin='lower',extent=[.5,64.5,.5,64.5],aspect='auto',cmap='magma',norm=LogNorm(1e-6,1),rasterized=True)
    ax.set(xticks=[1,64],yticks=[1,64],xlabel='Input rank'); ax.set_title(LABELS[i],color=COLORS[i],fontsize=9)
    if i: ax.tick_params(labelleft=False)
    else: ax.set_ylabel('Gram rank')
    ax=axs[1,i]; vals=np.array([gains[i][:64].sum(),gains[i][-64:].sum()])*100
    ax.bar([0,1],vals,color=['#91bdd4','#e2b889'],width=.62)
    ax.set(xticks=[0,1],xticklabels=['Top 64','Bottom 64'],ylim=(0,80),yticks=[0,40,80])
    ax.spines[['top','right']].set_visible(False)
    for j,v in enumerate(vals): ax.text(j,v+2,f'{v:.2f}%' if v<1 else f'{v:.1f}%',ha='center',fontsize=8)
    if i: ax.tick_params(labelleft=False)
    else: ax.set_ylabel('Gain share (%)')
cax=fig.add_axes([.95,.445,.01,.455]);fig.colorbar(im,cax=cax,ticks=[1e-6,1e-3,1])
for ext in ['pdf','png']:fig.savefig(OUT/f'prediction_spectra_alignment.{ext}',dpi=240,bbox_inches='tight',pad_inches=.03)
plt.close(fig)
fig,axs=plt.subplots(1,2,figsize=(6.5,2.6))
for ax,vs,title in zip(axs,[targets,grams],['Target covariance','Embedding Gram']):
    draw_four(ax,vs);ax.set_title(title);ax.set_xticks([1,384,768])
axs[1].set_ylabel('')
fig.legend(*axs[0].get_legend_handles_labels(),loc='upper center',ncol=2,frameon=False,fontsize=8)
fig.tight_layout(rect=[0,0,1,.80])
for ext in ['pdf','png']:fig.savefig(OUT/f'prediction_covariance_gram.{ext}',dpi=240,bbox_inches='tight',pad_inches=.03)
meta={'sources':[str(p.relative_to(ROOT)) for p in [RAW/'raw_pixel_patch_pca.npz',RAW/'patch_embedding_matrices.npz',WHITE/'embedding_matrices.npz']], 'weights':'raw epoch 200; native training coordinates', 'models':LABELS,'gain_top64_bottom64':[[float(g[:64].sum()),float(g[-64:].sum())] for g in gains], 'whitened_reference':'Native axes in original clean-PCA order; gain normalized over all 768 directions.'}
(OUT/'sources.json').write_text(json.dumps(meta,indent=2)+'\n')
print(json.dumps(meta,indent=2))
