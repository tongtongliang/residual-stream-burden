"""Render paper plots from packaged local data on CPU."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'result_statistic/section03_analysis'
ORDER=['JiT-B velocity','JiT-B clean','DeCO-B','PixelDiT-B','DiP-B','HyperDiT-B']
COLORS=['#D55E00','#0072B2','#E69F00','#009E73','#CC79A7','#56B4E9']
STYLES=['--','-','-.',':',(0,(5,1)),(0,(3,1,1,1))]
DISPLAY={'JiT-B velocity':'Pixel $\\boldsymbol{v}$-pred.','JiT-B clean':'Pixel $\\boldsymbol{x}$-pred.','DeCO-B':'DeCo','PixelDiT-B':'PixelDiT','DiP-B':'DiP','HyperDiT-B':'HyperDiT'}


def save(fig, part, name):
    p=ROOT/'figure/section03_analysis'/part
    p.mkdir(parents=True,exist_ok=True)
    for ext in ['pdf','png']:
        fig.savefig(p/f'{name}.{ext}',dpi=200,bbox_inches='tight')
    plt.close(fig)


def square(xlabel,ylabel):
    fig,ax=plt.subplots(figsize=(8,8),layout='constrained')
    ax.set_box_aspect(1)
    ax.set(xlabel=xlabel,ylabel=ylabel)
    ax.grid(alpha=.18)
    return fig,ax


def main():
    plt.rcParams.update({'font.size':16,'axes.labelsize':22,'axes.titlesize':22,
                         'legend.fontsize':12,'pdf.fonttype':42,'ps.fonttype':42})
    part='01_patch_embedding_matrix'
    with np.load(DATA/part/'data/raw_pixel_patch_pca.npz') as p:
        lam=p['eigenvalues']; vectors=p['components'].T
    ranks=np.arange(1,769)
    for cumulative in [False,True]:
        fig,ax=square('PCA rank','Cumulative variance' if cumulative else 'Eigenvalue')
        for t,c,style in zip([1,.75,.5,.25,0],COLORS,STYLES):
            e=t*t*lam+(1-t)**2
            ax.plot(ranks,np.cumsum(e)/e.sum() if cumulative else e,color=c,ls=style,label=f't={t:g}')
        ax.set_xlim(1,768)
        if not cumulative: ax.set_yscale('log')
        ax.legend(loc='lower right' if cumulative else 'upper right')
        save(fig,part,'raw_patch_cumulative' if cumulative else 'raw_patch_spectrum')
    rankwise=pd.read_csv(DATA/part/'data/rankwise_embedding_metrics.csv')
    for cumulative in [False,True]:
        fig,ax=square('PCA rank','Cumulative share' if cumulative else 'Energy / gain share')
        base=lam/lam.sum()
        ax.plot(ranks,np.cumsum(base) if cumulative else base,'k--',label='Clean patch energy')
        for m,c,s in zip(ORDER,COLORS,STYLES):
            g=rankwise[f'{m} gain'].to_numpy();g=g/g.sum()
            ax.plot(ranks,np.cumsum(g) if cumulative else g,color=c,ls=s,label=DISPLAY.get(m,m))
        ax.set_xlim(1,768)
        if not cumulative: ax.set_yscale('log')
        ax.legend(loc='lower right' if cumulative else 'upper right')
        save(fig,part,'gain_cumulative' if cumulative else 'gain_share')
    with np.load(DATA/part/'data/patch_embedding_matrices.npz') as p:
        fig,axes=plt.subplots(1,6,figsize=(25,5),layout='constrained')
        for ax,m in zip(axes,ORDER):
            idx=list(p['model_names']).index(m)
            _,u=np.linalg.eigh(p[f'gram_{idx}'].astype(float))
            overlap=(vectors[:,:64].T@u[:,::-1][:,:64])**2
            im=ax.imshow(np.log10(overlap+1e-8),origin='lower',extent=(.5,64.5,.5,64.5),cmap='magma',vmin=-6,vmax=-.25)
            ax.set_title(m,fontsize=20);ax.set_xticks([1,32,64]);ax.set_yticks([1,32,64])
        fig.supxlabel('Gram direction rank');fig.supylabel('PCA direction rank')
        fig.colorbar(im,ax=axes,label='log10 overlap squared',shrink=.85)
        save(fig,part,'directional_overlap')
    part='02_residual_stream_sensitivity'
    df=pd.read_csv(DATA/part/'data/all_aggregate_metrics.csv')
    ci=pd.read_csv(DATA/part/'data/bootstrap_intervals.csv')
    for t in [.25,.5,.75]:
        for metric,label in [('within_noise','Noise variance W'),('between_image','Image variance B'),('noise_conditioned_variance_fraction','Noise fraction F')]:
            fig,ax=plt.subplots(figsize=(12,6),layout='constrained')
            for m,c,s in zip(ORDER,COLORS,STYLES):
                name=m if m.startswith('JiT') else m+' velocity'
                d=df[(df.model_display==name)&np.isclose(df.t_clean,t)].sort_values('point')
                ax.plot(d.point,d[metric],color=c,ls=s,label=m)
                if metric=='noise_conditioned_variance_fraction':
                    b=ci[(ci.model_display==name)&np.isclose(ci.t_clean,t)].sort_values('point')
                    ax.fill_between(b.point,b.bootstrap_low,b.bootstrap_high,color=c,alpha=.15)
            ax.set_xticks(range(24),[f'{"A" if i%2==0 else "M"}{i//2}' for i in range(24)],rotation=60)
            ax.set(xlabel='Fan-in point',ylabel=label,title=f't={t:g}');ax.legend(ncol=3);ax.grid(alpha=.18)
            save(fig,part,f'{metric}_t{t:.2f}')
    part='03_semantic_endpoint_patch_pca'
    for model in ['deco_b','pixeldit_b','dip_b']:
        with np.load(DATA/part/f'{model}_spectra.npz') as p:
            fig,ax=square('PCA rank','Cumulative variance')
            for i,(t,e) in enumerate(zip(p['t'],p['eigenvalues'])):
                ax.plot(ranks,np.cumsum(e)/e.sum(),color=plt.cm.viridis(i/8),ls=['-','--',':'][i%3],label=f't={t:.1f}')
            ax.set(xlim=(1,768),ylim=(0,1.01),title={'deco_b':'DeCO-B','pixeldit_b':'PixelDiT-B','dip_b':'DiP-B'}[model])
            ax.legend(loc='lower right',ncol=2)
            save(fig,part,f'{model}_cumulative')
    print('Rendered PDF and PNG from packaged data')


if __name__=='__main__':
    main()
