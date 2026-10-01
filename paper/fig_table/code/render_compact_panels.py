"""Compact manuscript figures, using archived measurements/samples only.

No training, model inference, sample generation, or metric re-evaluation. Coordinates and
display limits for the cached point clouds are recorded beside the figures.
"""
from pathlib import Path
import csv,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
REPO=ROOT.parents[1]
DATA=ROOT/'result_statistic/section03_analysis'
OUT=ROOT/'figure/manuscript_panels'
OUT.parent.mkdir(parents=True, exist_ok=True)
MODELS=['JiT-B velocity','JiT-B clean','DeCO-B','PixelDiT-B','DiP-B','HyperDiT-B']
LABELS=['Pixel $\\boldsymbol{v}$-pred.','Pixel $\\boldsymbol{x}$-pred.','DeCo','PixelDiT','DiP','HyperDiT']
COLORS=['#D55E00','#0072B2','#B38800','#009E73','#AC5598','#54A4D0']

def rows(p):
    with p.open() as f:return list(csv.DictReader(f))
def save(fig,n):
    OUT.mkdir(exist_ok=True,parents=True)
    fig.savefig(OUT/(n+'.pdf'),bbox_inches='tight',pad_inches=.018,dpi=400)
    fig.savefig(OUT/(n+'.png'),bbox_inches='tight',pad_inches=.018,dpi=260)
    plt.close(fig)
def style(ax):
    ax.spines[['top','right']].set_visible(False)
    ax.tick_params(length=2,pad=1.5,width=.5)
    ax.grid(alpha=.15,lw=.4,axis='y')

def embedding():
    p=DATA/'01_patch_embedding_matrix/data'
    source=rows(p/'rankwise_embedding_metrics.csv')
    bins=rows(p/'gain_rank_bins.csv')
    assert len(source)==768 and len(bins)==36
    # Separate alignment: six tightly spaced squares with one common scale.
    with np.load(p/'raw_pixel_patch_pca.npz') as z:
        _,u=np.linalg.eigh(z['covariance']);u=u[:,::-1]
        lam=z['eigenvalues']
    fig,axes=plt.subplots(1,6,figsize=(5.65,1.45))
    fig.subplots_adjust(left=.045,right=.947,bottom=.22,top=.88,wspace=.055)
    with np.load(p/'patch_embedding_matrices.npz') as z:
        names=list(z['model_names'])
        for ax,name,label in zip(axes,MODELS,LABELS):
            idx=names.index(name);_,v=np.linalg.eigh(z[f'gram_{idx}']);v=v[:,::-1]
            overlap=(u[:,:64].T@v[:,:64])**2
            im=ax.imshow(overlap,origin='lower',cmap='magma',norm=LogNorm(1e-6,.5),extent=[.5,64.5,.5,64.5],interpolation='none')
            ax.set_title(label,fontsize=6.5,pad=3)
            ax.set_xticks([1,64]);ax.set_yticks([1,64]);ax.tick_params(length=1,pad=1,labelsize=6)
            ax.get_xticklabels()[0].set_ha('left')
            ax.get_xticklabels()[-1].set_ha('right')
            if ax is not axes[0]:ax.set_yticklabels([])
    axes[0].set_ylabel('Clean-patch\nPCA rank',fontsize=7,labelpad=1)
    fig.text(.5,.055,'Rank of principal directions in the embedding matrix',ha='center',fontsize=7.3)
    fig.canvas.draw();pos=axes[-1].get_position()
    cax=fig.add_axes([.96,pos.y0,.012,pos.height]);fig.colorbar(im,cax=cax,ticks=[1e-5,1e-3,1e-1]);cax.tick_params(labelsize=6,length=1,pad=1)
    save(fig,'alignment_overlap')

    fig,(c,a,b)=plt.subplots(1,3,figsize=(5.65,1.75),gridspec_kw={'width_ratios':[.96,1,1]})
    fig.subplots_adjust(left=.074,right=.98,bottom=.23,top=.90,wspace=.52)
    for name,label,color in zip(MODELS,LABELS,COLORS):
        eig=np.array([float(x[name+' gram_eigenvalue']) for x in source]).clip(0)
        a.plot(np.arange(1,769),np.cumsum(eig)/eig.sum(),color=color,lw=.85,label=label)
        g=np.array([float(x[name+' gain']) for x in source]);g/=g.sum()
        b.plot(np.arange(1,513),g[:512],color=color,lw=.85)
    a.plot(np.arange(1,769),np.cumsum(lam)/lam.sum(),color='#202830',lw=1.15,ls='--',label='Clean patches')
    a.set(xlim=(1,768),ylim=(0,1.03),xticks=[1,256,512,768],yticks=[0,.5,1],xlabel='Principal-direction rank',ylabel='Cumulative energy',title='(b) Cumulative energy')
    b.set(xlim=(1,512),yscale='log',ylim=(1e-6,.1),xticks=[1,128,256,384,512],yticks=[1e-1,1e-3,1e-5],xlabel='Clean-patch PCA rank',ylabel='Gain share',title='(c) Directional gain')
    rs=rows(DATA/'03_semantic_endpoint_patch_pca/all_endpoint_metrics.csv');assert len(rs)==27
    rank90=lambda x:int(np.searchsorted(np.cumsum(x)/np.sum(x),.9)+1)
    clean,velocity=rank90(lam),rank90(lam+1)
    for name,color in zip(['DeCO-B','PixelDiT-B','DiP-B'],[COLORS[2],COLORS[3],COLORS[4]]):
        rr=sorted([r for r in rs if r['display_name']==name],key=lambda r:float(r['t_clean']))
        c.plot([float(r['t_clean']) for r in rr],[int(r['endpoint_rank90']) for r in rr],'o-',color=color,lw=1.05,ms=2.1)
    c.axhline(velocity,color='#56616B',ls='--',lw=.75)
    c.text(.12,velocity+26,f'Velocity target: {velocity}',fontsize=5.8,color='#56616B')
    c.axhline(clean,color='#202830',ls=':',lw=.85)
    c.text(.45,22,'Clean: 8',fontsize=6.1,color='#202830')
    c.set(xlim=(.08,.94),ylim=(0,800),xticks=[.1,.5,.9],yticks=[0,200,400,600],xlabel='Clean coefficient $t$',ylabel='$r_{90}$ (dimensions)',title='(a) Endpoint')
    for ax in [a,b,c]:style(ax)
    a.legend(loc='lower right',frameon=False,handlelength=1.35,handletextpad=.35,labelspacing=0,borderaxespad=.2,fontsize=5.8)
    b.tick_params(which='minor',length=0)
    save(fig,'spectrum_gain_endpoint')
    lines=[r'\begin{tabular}{@{}lrrrrrr@{}}',r'\toprule',r'Gain share (\%) & JiT-$v$ & JiT-$x$ & DeCo & PixelDiT & DiP & HyperDiT \\',r'\midrule']
    for name in ['Top 256','Bottom 256']:
        vals=[100*float(next(r['gain_share'] for r in bins if r['model']==m and r['rank_bin']==name)) for m in MODELS]
        lines.append(name+' & '+' & '.join(f'{v:.1f}' for v in vals)+r' \\')
    lines += [r'\bottomrule',r'\end{tabular}']
    (OUT/'gain_top_bottom_256.tex').write_text('\n'.join(lines)+'\n')


def toy_rank():
    rs=rows(DATA/'04_toy_residual_rank/residual_rank_metrics.csv');assert len(rs)==50
    from matplotlib.patches import Polygon,Rectangle,FancyArrow
    from matplotlib.colors import to_rgb
    fig,axes=plt.subplots(1,2,figsize=(3.65,1.70))
    fig.subplots_adjust(left=.08,right=.995,bottom=.20,top=.86,wspace=.12)
    for ax,mode,color,title in zip(axes,['clean','velocity'],['#AED7EA','#CAE2B4'],['Predict clean $x$','Predict velocity $v$']):
        rgb=np.array(to_rgb(color));edge='#81929C'
        for row,t in enumerate([.1,.3,.5,.7,.9]):
            for col in range(5):
                rank=int(next(r['r90'] for r in rs if r['prediction']==mode and int(r['block'])==col and abs(float(r['t'])-t)<1e-6))
                ax.add_patch(Rectangle((col,row),1,1,facecolor='#FAFCFD',edgecolor='#DDE4E8',linewidth=.35))
                # The orthographic grid fixes layer/time positions. Only bar height
                # encodes rank; the top/side faces give depth without occlusion.
                x=col+.22;y=row+.075;w=.43;h=.48*rank/225;dx=.10;dy=.065
                ax.add_patch(Polygon([(x,y),(x+w,y),(x+w,y+h),(x,y+h)],facecolor=rgb,edgecolor=edge,linewidth=.28))
                ax.add_patch(Polygon([(x+w,y),(x+w+dx,y+dy),(x+w+dx,y+h+dy),(x+w,y+h)],facecolor=rgb*.84,edgecolor=edge,linewidth=.28))
                ax.add_patch(Polygon([(x,y+h),(x+dx,y+h+dy),(x+w+dx,y+h+dy),(x+w,y+h)],facecolor=rgb*.6+.4,edgecolor=edge,linewidth=.28))
                ax.text(col+.485,y+h+dy+.04,str(rank),ha='center',va='bottom',fontsize=6.4,color='#1F2933')
        ax.set(xlim=(0,5),ylim=(0,5),xticks=np.arange(5)+.5,xticklabels=range(1,6),yticks=np.arange(5)+.5,yticklabels=['.1','.3','.5','.7','.9'])
        ax.set_title(title,fontsize=8,pad=4)
        ax.tick_params(length=0,pad=2,labelsize=6.5)
        ax.spines[:].set_visible(False)
        ax.add_patch(FancyArrow(.01,-.28,.98,0,width=.023,head_width=.07,head_length=.055,length_includes_head=True,transform=ax.transAxes,clip_on=False,color='#34495E',linewidth=0))
        ax.text(.5,-.155,'Layer / residual stream',transform=ax.transAxes,ha='center',va='top',fontsize=6.6)
    axes[0].set_ylabel('$t$',rotation=0,labelpad=7,fontsize=8)
    axes[1].set_yticklabels([])
    save(fig,'toy_rank_grid_bars')


def sensitivity_table():
    rs=rows(DATA/'02_residual_stream_sensitivity/data/paper_depth_mean_table.csv');assert len(rs)==18
    lines=[r'\begin{tabular}{@{}lrrrrrr@{}}',r'\toprule',r'Noise $1-t$ & JiT-$v$ & JiT-$x$ & DeCo & PixelDiT & DiP & HyperDiT \\',r'\midrule']
    for noise in [.25,.50,.75]:
        vals=[]
        for model in MODELS:
            display=model if model.startswith('JiT') else model+' velocity'
            vals.append(float(next(r['mean_ratio_F'] for r in rs if r['model_display']==display and abs(float(r['t'])-(1-noise))<1e-6)))
        lines.append(f'{noise:.2f}'+' & '+' & '.join(f'{v:.3f}' for v in vals)+r' \\')
    lines += [r'\bottomrule',r'\end{tabular}']
    (OUT/'sensitivity_by_noise.tex').write_text('\n'.join(lines)+'\n')

def pointclouds():
    paths=['fcn_w256_adaln/runs/fcn_w256_adaln_xv','fcn_w256_adaln/runs/fcn_w256_adaln_xv','fcn_w256_adaln/runs/fcn_w256_adaln_xv','fcn_w256_branch1024_adaln/runs/fcn_w256_branch1024_adaln_v','fcn_w512_adaln/runs/fcn_w512_adaln_xv','sihc_patch_adaln/runs/sihc_patch_adaln_xv']
    titles=['Ground truth data','FCN-256 / clean','FCN-256 / velocity','FCN-256-Wide / velocity','FCN-512 / velocity','FCN-256-HC / velocity']
    cost_rows=rows(ROOT/'result_statistic/section04_swiss_roll/04_swiss_roll/paper_main_table.csv')
    cost_keys=[None,('FCN-256','x'),('FCN-256','v'),('FCN-256 wide','v'),('FCN-512','v'),('SiHC N=4','v')]
    costs=[None if key is None else next(r for r in cost_rows if (r['model'],r['target'])==key) for key in cost_keys]
    points=[]
    for i,pp in enumerate(paths):
        p=REPO/'toy/results'/pp
        with np.load(p/'training_data_snapshot.npz') as z:
            offset=-z['mean']/z['std'];basis=z['P']/z['std'][:,None]
            if i==0:x=z['x0'].astype(float)
        if i:
            with np.load(p/'analysis/samples.npz') as z:x=z[('x' if i==1 else 'v')+'_highd'].astype(float)
        xy=(x-offset)@basis@np.linalg.inv(basis.T@basis)
        residual=x-offset-xy@basis.T
        points.append([xy,residual])
    _,_,vh=np.linalg.svd(points[2][1],full_matrices=False);normal=vh[0]
    xyall=np.concatenate([p[0] for p in points]);lo=xyall.min(0);hi=xyall.max(0);pad=(hi-lo)*.035
    xlim=(lo[0]-pad[0],hi[0]+pad[0]);ylim=(lo[1]-pad[1],hi[1]+pad[1])
    heights=[np.sign(r@normal+1e-12)*np.linalg.norm(r,axis=1) for xy,r in points]
    zmax=max(np.max(np.abs(h)) for h in heights)*1.06
    fig=plt.figure(figsize=(8.6,1.65))
    for i,((xy,r),h,title) in enumerate(zip(points,heights,titles)):
        ax=fig.add_axes([i/6+.005,.12,.155,.64],projection='3d')
        rng=np.random.default_rng(10 if i==0 else 20);ids=rng.choice(len(xy),min(1400,len(xy)),replace=False)
        xx,yy=np.meshgrid(xlim,ylim);ax.plot_surface(xx,yy,np.zeros_like(xx),color='#F4DB86',alpha=.23,shade=False,linewidth=0)
        ax.scatter(xy[ids,0],xy[ids,1],h[ids],s=.6,color='#CF811E' if i==0 else '#1685B8',alpha=.75,linewidths=0,depthshade=False,rasterized=True)
        ax.set(xlim=xlim,ylim=ylim,zlim=(-zmax,zmax),xticks=[-10,0,10],yticks=[-10,0,10],zticks=[-20,0,20])
        ax.set_box_aspect((1,1,.66),zoom=1.30);ax.view_init(elev=23,azim=-58);ax.grid(False)
        for axis in [ax.xaxis,ax.yaxis,ax.zaxis]:
            axis.pane.fill=False;axis.pane.set_edgecolor('#FFFFFF');axis.line.set_color('#BCC5CC');axis.set_tick_params(labelsize=7,pad=-3)
        ax.text2D(.5,1.30,title.replace(' / ', '\n'),transform=ax.transAxes,ha='center',fontsize=8.5)
        if costs[i] is not None:
            params=int(costs[i]['parameters'])/1e6;flops=int(costs[i]['forward_flops'])/1e6
            ax.text2D(.5,1.025,f'{params:.2f}M params\n{flops:.2f}M FLOPs',transform=ax.transAxes,ha='center',fontsize=7,color='#536271')
    save(fig,'toy_samples_six')
    (OUT/'toy_samples_display.json').write_text(json.dumps({'source':'archived toy/results/*/analysis/samples.npz','cost_source':'paper/fig_table/result_statistic/section04_swiss_roll/04_swiss_roll/paper_main_table.csv','costs':costs,'xlim':list(xlim),'ylim':list(ylim),'zlim':[-float(zmax),float(zmax)],'points_per_panel':1400,'models':titles,'coordinates':'known data-plane coordinates and signed full off-plane norm; shared view and limits; all displayed samples inside limits'},indent=2))

def reconstruction():
    # Keep native pixels from the archived scientific grid, without resampling.
    from reportlab.pdfgen import canvas
    from reportlab.lib.utils import ImageReader
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    im=Image.open(ROOT/'figure/section03_analysis/01_patch_embedding_matrix/patch_pca_visual_reconstruction.png').convert('RGB')
    a=np.asarray(im);active=np.where(np.any(a<245,axis=2).sum(axis=1)>im.width*.3)[0]
    groups=np.split(active,np.where(np.diff(active)>2)[0]+1)
    groups=[g for g in groups if len(g)>100];assert len(groups)==5
    crops=[im.crop((0,int(groups[i][0]),im.width,int(groups[i][-1])+1)) for i in [0,1,3]]
    gap=10;out=Image.new('RGB',(im.width,sum(c.height for c in crops)+2*gap),'white')
    y=0
    for crop in crops:out.paste(crop,(0,y));y+=crop.height+gap
    out.save(OUT/'patch_reconstruction_three.png')
    width=408;h=out.height*width/out.width
    pdfmetrics.registerFont(TTFont('ReconSans',str(Path(matplotlib.get_data_path())/'fonts/ttf/DejaVuSans.ttf')))
    c=canvas.Canvas(str(OUT/'patch_reconstruction_three.pdf'),pagesize=(width,h+13))
    c.drawImage(ImageReader(out),0,13,width=width,height=h)
    c.setFont('ReconSans',8)
    for j,label in enumerate(['Original','Top 8','Top 16','Top 32','Top 64','Top 128']):c.drawCentredString((j+.5)*width/6,2,label)
    c.save()
    print('Reconstruction embedded at native resolution:',out.size)

if __name__=='__main__':
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':7,'axes.labelsize':7,'axes.titlesize':7.5,'xtick.labelsize':6.3,'ytick.labelsize':6.3,'axes.linewidth':.5,'pdf.fonttype':42})
    embedding();toy_rank();sensitivity_table();pointclouds();reconstruction()
    print('Rendered revised scientific panels from archived data.')
