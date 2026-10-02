"""Render cached normalized sensitivity; no model execution or data mutation."""
from pathlib import Path
import csv
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter

ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'pyproject.toml').is_file())
SOURCE = ROOT / 'research/normalized_sensitivity'
OUT = ROOT / 'research/figures/normalized_sensitivity'
OUT.mkdir(parents=True, exist_ok=True)
MODELS = [
    ('plain_jit_velocity', 'Pixel $\\boldsymbol{v}$-pred.', '#D55E00', 's', '-'),
    ('plain_jit_clean', 'Pixel $\\boldsymbol{x}$-pred.', '#0072B2', 'o', '-'),
    ('deco_b_velocity', 'DeCo', '#A07900', '^', '--'),
    ('pixeldit_b_velocity', 'PixelDiT', '#009E73', 'D', '-.'),
    ('dip_b_velocity', 'DiP', '#B45A9A', 'X', ':'),
    ('hyperdit_b_velocity', 'HyperDiT', '#4899BE', 'v', '--'),
]
MAIN_MODELS = MODELS[:-1]
TIMES = (.25, .50, .75)
rows = list(csv.DictReader((SOURCE / 'per_event.csv').open()))
selected = [r for r in rows if r['model'] in {m[0] for m in MODELS}
            and r['point'].startswith('workspace/')]
for r in selected:
    assert int(r['images']) == 2048 and int(r['noises']) == 128
    assert int(r['dimensions']) == 256 * 768
    w, b, raw = map(float, (r['within'], r['between_corrected'], r['between_raw']))
    assert np.isclose(b, raw - w / 128, rtol=1e-12)
    assert np.isclose(float(r['within_per_coordinate']), w / 196608, rtol=1e-12)
    assert np.isclose(float(r['ratio_corrected']), b / w, rtol=1e-12)
assert len(selected) == 408

def series(model, t, mlp=False):
    rr = [r for r in selected if r['model'] == model and float(r['t_clean']) == t
          and (not mlp or r['point'].endswith('_mlp'))]
    rr.sort(key=lambda r: (int(r['point'].split('/b')[1].split('_')[0]),
                           r['point'].endswith('_mlp')))
    expected = (8 if model == 'hyperdit_b_velocity' else 12) * (1 if mlp else 2)
    assert len(rr) == expected
    return rr

plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':16,
                     'axes.labelsize':17, 'axes.titlesize':19,
                     'xtick.labelsize':15, 'ytick.labelsize':15,
                     'legend.fontsize':16, 'pdf.fonttype':42, 'ps.fonttype':42,
                     'axes.spines.top':False, 'axes.spines.right':False})

def style(ax, xlim, ylims, ticks, scale="log"):
    ax.set_xlim(.7, xlim + .3)
    ax.set_xticks([1,4,8,12] if xlim == 12 else [1,6,12,18,24])
    ax.set_yscale(scale)
    ax.set_ylim(*ylims)
    ax.yaxis.set_major_locator(FixedLocator(ticks))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:g}'))
    ax.tick_params(which='minor', left=False)
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.grid(axis='y', which='major', color='#D9DEE3', linewidth=.65)
    ax.set_axisbelow(True)

def legend(fig, ax, y, ncol=3):
    handles, labels = ax.get_legend_handles_labels()
    # Three columns, two rows: retain the alignment figure's order.
    fig.legend(handles, labels, ncol=ncol, loc='upper center',
               bbox_to_anchor=(.53,y), frameon=False, columnspacing=1.2 if ncol==5 else 1.8,
               handlelength=1.8 if ncol==5 else 2.5, handletextpad=.5)

def save(fig, name):
    fig.savefig(OUT / f'{name}.pdf', bbox_inches='tight', pad_inches=.035)
    fig.savefig(OUT / f'{name}.png', dpi=200, bbox_inches='tight', pad_inches=.035)
    plt.close(fig)

# Main figure: one consistently located observation per block, five models only.
fig, axes = plt.subplots(1,3,figsize=(10.8,3.55),sharey=False)
for ax,t in zip(axes,TIMES):
    for model,label,color,marker,ls in MAIN_MODELS:
        rr=series(model,t,True)
        ax.plot(range(1,len(rr)+1),[float(r['ratio_corrected']) for r in rr],
                label=label,color=color,marker=marker,linestyle=ls,
                lw=2.2,ms=6.2,mew=.7,markeredgecolor='white')
    upper = {.25:5, .50:12, .75:90}[t]
    style(ax,12,(0,upper),np.arange(0,upper+1,{.25:1,.50:3,.75:30}[t]),scale='linear')
    ax.set_title(f'$t={t:.2f}$',pad=9)
    ax.set_xlabel('Residual stream',labelpad=7)
axes[0].set_ylabel('Between / within $B/W$',labelpad=8)
legend(fig,axes[0],1.02,ncol=5)
fig.subplots_adjust(left=.085,right=.995,bottom=.21,top=.77,wspace=.27)
save(fig,'sensitivity_main')

# Full appendix: both attention and MLP inputs, with both variance components.
fig, axes=plt.subplots(3,3,figsize=(10.8,9.0),sharex=True,sharey='row')
for col,t in enumerate(TIMES):
    for row,(key,ylabel,lims,ticks) in enumerate([
        ('within_per_coordinate','Sensitivity $S=W/D$',(.002,1.05),[.003,.01,.03,.1,.3,1]),
        ('between_corrected_per_coordinate','Between-image $B/D$',(.03,1.05),[.03,.1,.3,1]),
        ('ratio_corrected','Between / within $B/W$',(.02,150),[.03,.1,1,10,100]),
    ]):
        ax=axes[row,col]
        for model,label,color,marker,ls in MODELS:
            rr=series(model,t)
            yy=[float(r[key]) for r in rr]
            assert min(yy)>0
            assert min(yy)>=lims[0] and max(yy)<=lims[1]
            ax.plot(range(1,len(rr)+1),yy,label=label,color=color,marker=marker,
                    linestyle=ls,lw=1.8,ms=4.3,markevery=2,mew=.5,markeredgecolor='white')
        style(ax,24,lims,ticks)
        if row==0: ax.set_title(f'$t={t:.2f}$',pad=9)
        if col==0: ax.set_ylabel(ylabel,labelpad=8)
        if row==2: ax.set_xlabel('Residual stream',labelpad=7)
legend(fig,axes[0,0],1.005)
fig.subplots_adjust(left=.11,right=.995,bottom=.09,top=.885,wspace=.15,hspace=.2)
save(fig,'sensitivity_full')

with (OUT/'source_values.csv').open('w') as f:
    writer=csv.DictWriter(f,fieldnames=list(selected[0]))
    writer.writeheader();writer.writerows(selected)
mainrows=[r for r in selected if r['point'].endswith('_mlp') and r['model'] in {m[0] for m in MAIN_MODELS}]
with (OUT/'main_values.csv').open('w') as f:
    writer=csv.DictWriter(f,fieldnames=list(selected[0]))
    writer.writeheader();writer.writerows(mainrows)
assert len(mainrows)==180

# A compact appendix table, aggregating all observed events.
table=[r'\begin{tabular}{@{}lrrr|rrr|rrr@{}}',r'\toprule',
       r'& \multicolumn{3}{c}{$t=.25$} & \multicolumn{3}{c}{$t=.50$} & \multicolumn{3}{c}{$t=.75$} \\',
       r'Model & $\bar S$ & $\overline{B/D}$ & $\bar R$ & $\bar S$ & $\overline{B/D}$ & $\bar R$ & $\bar S$ & $\overline{B/D}$ & $\bar R$ \\',r'\midrule']
for model,label,*_ in MODELS:
    vals=[]
    for t in TIMES:
        rr=series(model,t)
        for key,fmt in [('within_per_coordinate','.3f'),('between_corrected_per_coordinate','.3f'),('ratio_corrected','.2f')]:
            vals.append(format(np.mean([float(r[key]) for r in rr]),fmt))
    table.append(label+' & '+' & '.join(vals)+r' \\')
table += [r'\bottomrule',r'\end{tabular}']
(OUT/'summary.tex').write_text('\n'.join(table)+'\n')

provenance={'source':str((SOURCE/'per_event.csv').relative_to(ROOT)),
            'source_rows':len(rows),'selected_rows':len(selected),'main_rows':len(mainrows),
            'models':[m[0] for m in MODELS], 'main_models':[m[0] for m in MAIN_MODELS],'times_clean':TIMES,
            'main_metric':'ratio_corrected = (between_raw - within/128) / within',
            'appendix_metrics':'W/D, B/D, corrected B/W',
            'normalization':'per patch RMS, eps1e-6, no affine, no GAP',
            'main_observation':'MLP input before native normalization/modulation, one per block',
            'appendix_observation':'attention input then MLP input, in native depth order',
            'x_axis':'native block index in main, native observation index in appendix; HyperDiT not stretched',
            'y_axis':'main: linear, panel-specific ranges starting at zero; appendix: logarithmic, shared across times',
            'uncertainty':'No CIs in the new archive; none synthesized.',
            'preserved_fields':'all original covariance traces, per-coordinate values and ratios',
            'checks':'408 rows, 180 main markers, formulas, sample counts, native observation counts'}
(OUT/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
print(f'Rendered main and appendix figures from {len(selected)} cached rows.')
