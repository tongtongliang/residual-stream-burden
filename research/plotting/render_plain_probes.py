"""Compare pixel and DINO representations on one axis, from cached probe values."""
from pathlib import Path
import csv
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'pyproject.toml').is_file())
OUT = ROOT / 'research/figures/plain_probes'
paths = [ROOT / f'research/linear_probes/{group}/data/accuracy.csv' for group in ('group1', 'group2')]
rows = [row for path in paths for row in csv.DictReader(path.open())]
models = [
    ('plain_jit_clean', 'Pixel $\\boldsymbol{x}$-pred.', '#2879b9', 'o', '-'),
    ('plain_jit_velocity', 'Pixel $\\boldsymbol{v}$-pred.', '#2879b9', 's', '--'),
    ('raev1-g2a-01-jit-b-clean-epoch200', 'DINO $\\boldsymbol{x}$-pred.', '#d46a16', 'o', '-'),
    ('raev1-g2a-02-jit-b-velocity-epoch200', 'DINO $\\boldsymbol{v}$-pred.', '#d46a16', 's', '--'),
]
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.labelsize': 9,
                     'legend.fontsize': 8, 'pdf.fonttype': 42,
                     'axes.spines.top': False, 'axes.spines.right': False})
fig, ax = plt.subplots(figsize=(2.6, 2.6))
selected = []
for model, label, color, marker, linestyle in models:
    values = sorted([row for row in rows if row['model'] == model and float(row['alpha']) == .5 and row['mode'] == 'matched_noise'], key=lambda row: int(row['after_block']))
    assert len(values) == 11
    x = np.array([int(row['after_block']) for row in values])
    y = np.array([float(row['top1']) for row in values])
    sd = np.array([float(row['top1_seed_std']) for row in values])
    assert np.all(y - sd >= 0) and np.all(y + sd < 90)
    ax.plot(x, y, color=color, label=label, lw=1.5, marker=marker, ms=3, linestyle=linestyle)
    ax.fill_between(x, y - sd, y + sd, color=color, alpha=.13, lw=0)
    selected.extend(values)
ax.set(xlim=(.8, 11.2), ylim=(0, 90), xticks=[1, 6, 11], yticks=[0, 20, 40, 60, 80], xlabel='Block boundary', ylabel='Top-1 accuracy (%)')
ax.grid(axis='y', alpha=.18)
ax.set_axisbelow(True)
ax.legend(loc='upper center', ncol=1, frameon=False, bbox_to_anchor=(.54, .76), columnspacing=1.1, handlelength=2, borderaxespad=0)
fig.subplots_adjust(left=.22, right=.98, bottom=.19, top=.98)
for extension in ('pdf', 'png'):
    fig.savefig(OUT / f'plain_probes.{extension}', dpi=240, bbox_inches='tight', pad_inches=.02)
(OUT / 'plain_probes_sources.json').write_text(json.dumps({'sources': [str(path.relative_to(ROOT)) for path in paths], 'alpha_noise': .5, 't_clean': .5, 'observation': 'after blocks 1-11, before the next attention; token RMS then GAP', 'x_axis': 'Block boundary', 'rows': selected}, indent=2))
print('Rendered one axis with all 44 original probe means and validation-noise standard deviations.')
