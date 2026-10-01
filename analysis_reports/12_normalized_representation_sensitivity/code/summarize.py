"""Summaries of completed tasks only; preserve per-point metrics separately."""
import csv
import json
from pathlib import Path
import statistics
ROOT=Path(__file__).resolve().parent
rows=[]
for folder in sorted((ROOT/'runs').glob('*')):
    if not (folder/'complete.json').exists():continue
    metrics=json.loads((folder/'metrics.json').read_text())
    manifest=json.loads((folder/'manifest.json').read_text())
    for scope in sorted(set(r['point'].split('/')[0] for r in metrics)):
        points=[r for r in metrics if r['point'].startswith(scope+'/')]
        b=statistics.mean(r['between_corrected_per_coordinate'] for r in points)
        w=statistics.mean(r['within_per_coordinate'] for r in points)
        rows.append(dict(model=manifest['model']['label'],t_clean=manifest['t_clean'],scope=scope,
            points=len(points),B_per_coordinate=b,W_per_coordinate=w,
            mean_point_ratio=statistics.mean(r['ratio_corrected'] for r in points),ratio_of_mean_traces=b/w,
            last_point=points[-1]['point'],last_ratio=points[-1]['ratio_corrected']))
(ROOT/'summary.json').write_text(json.dumps(rows,indent=2))
if rows:
    with (ROOT/'summary.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
lines=['# Normalized Representation Sensitivity','',
    'Completed tasks only. N=2048, K=128, null class, raw historical weights. Time is CLEAN fraction.',
    'B is finite-K corrected between-image covariance trace; W is within-image noise covariance trace.',
    'B and W below are divided by feature dimension then averaged over observed events. R is mean of per-event ratios, not ratio of these displayed mean traces.',
    'HyperDiT has 16 events; others have 24. Last event is the last MLP input, not the final model output.',
    '', '| Model | t clean | Scope | B / D | W / D | Mean event R |', '|---|---:|---|---:|---:|---:|']
for r in rows:lines.append(f"| {r['model']} | {r['t_clean']} | {r['scope']} | {r['B_per_coordinate']:.6f} | {r['W_per_coordinate']:.6f} | {r['mean_point_ratio']:.4f} |")
lines+=['','Per-depth raw traces and both corrected/uncorrected ratios are in runs/*/metrics.csv. See README.md and validation JSONs for protocol and estimator checks.']
(ROOT/'REPORT.md').write_text('\n'.join(lines)+'\n')
print(json.dumps(dict(completed_tasks=len(list((ROOT/'runs').glob('*/complete.json'))),summaries=len(rows))))
