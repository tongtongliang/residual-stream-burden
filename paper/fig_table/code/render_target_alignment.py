"""Single-row target spectra and native-coordinate embedding alignment, cached data only.
Run with the Miniforge ml Python documented in AGENTS.md.
"""
from pathlib import Path
import json
import os
import numpy as np
ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'paper/fig_table/figure/r46'
OUT.mkdir(parents=True, exist_ok=True)
os.environ.setdefault('MPLCONFIGDIR', str(OUT / '.matplotlib_cache'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from render_section4_r45 import draw_four, LABELS, COLORS
RAW = ROOT / 'paper/fig_table/result_statistic/section03_analysis/01_patch_embedding_matrix/data'
WHITE = ROOT / 'analysis_reports/14_whitening_patch_embedding/data'
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
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':13,'axes.labelsize':13,'axes.titlesize':13,'xtick.labelsize':12,'ytick.labelsize':12,'legend.fontsize':12.5,'pdf.fonttype':42,'axes.linewidth':.65})
fig=plt.figure(figsize=(12,2.35))
# All six panels share a baseline. Space between the two groups carries the heatmap y-label.
axes=[fig.add_axes([x,.25,w,.55]) for x,w in [( .048,.172),(.248,.172),(.490,.105),(.610,.105),(.730,.105),(.850,.105)]]
for i,(ax,values) in enumerate(zip(axes[:2],[targets,grams])):
    draw_four(ax,values)
    ax.set_xticks([1,384,768]); ax.set_title(['Target covariance','Embedding Gram'][i],pad=5)
axes[1].set_ylabel(''); axes[1].tick_params(labelleft=False)
for i,ax in enumerate(axes[2:]):
    im=ax.imshow(np.maximum(overlaps[i],1e-6),origin='lower',extent=[.5,64.5,.5,64.5],aspect='auto',cmap='magma',norm=LogNorm(1e-6,1),interpolation='nearest',rasterized=True)
    ax.set(xticks=[1,64],yticks=[1,64],xlabel='Input rank')
    ax.set_title(LABELS[i],color=COLORS[i],pad=5,fontsize=11)
    ax.tick_params(length=2,pad=2)
    if i: ax.tick_params(labelleft=False)
    else: ax.set_ylabel('Embedding-Gram\neigenvector rank',labelpad=2)
cax=fig.add_axes([.967,.25,.008,.55]); fig.colorbar(im,cax=cax,ticks=[1e-6,1e-3,1]);cax.tick_params(length=2,pad=2,labelsize=11)
fig.legend(*axes[0].get_legend_handles_labels(),loc='lower left',bbox_to_anchor=(.044,.885),ncol=4,frameon=False,columnspacing=.8,handlelength=2.1,handletextpad=.4,borderaxespad=0,fontsize=11.5)
for ext in ['pdf','png']: fig.savefig(OUT/f'prediction_spectra_alignment.{ext}',dpi=260,bbox_inches='tight',pad_inches=.035)
plt.close(fig)
np.savez_compressed(OUT/'prediction_spectra_alignment_data.npz',**{f'overlap_{i}':v for i,v in enumerate(overlaps)},**{f'gram_spectrum_{i}':v for i,v in enumerate(grams)})
metadata={'models':LABELS,'sources':[str(p.relative_to(ROOT)) for p in [RAW/'raw_pixel_patch_pca.npz',RAW/'patch_embedding_matrices.npz',WHITE/'embedding_matrices.npz']], 'weights':'raw epoch 200, native training coordinates','heatmap':'Squared overlaps of the first 64 input-reference directions with the first 64 eigenvectors of W.T W, transposed: x input rank, y embedding-Gram eigenvector rank. Rank 1 lower left.','raw_reference':'Raw clean covariance eigenvectors in descending eigenvalue order.','whitened_reference':'Coordinate axes inherited from the fitted PCA whitening transform; not a numerically selected eigenbasis of identity covariance.','color_scale':{'log_min':1e-6,'max':1},'gram_effective_ranks':[float(np.exp(-np.sum((v/v.sum())*np.log(v/v.sum())))) for v in grams],'heatmap_shapes':[list(x.shape) for x in overlaps]}
(OUT/'prediction_spectra_alignment_sources.json').write_text(json.dumps(metadata,indent=2)+'\n')
print(json.dumps(metadata,indent=2))
