"""Audit six completed tasks and render raw-valued tables and scientific plots."""
import csv,json,math,statistics
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
ARCHIVE=Path('/data/tongtong/project/elucidating_residual_stream_burden/analysis_reports/12_normalized_representation_sensitivity')
def csv_write(path,rows):
 with path.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
summary=[];per_event=[];audits=[]
for pred in ('clean','velocity'):
 for t in (.25,.5,.75):
  folder=ROOT/'runs'/f'whiten_{pred}_t{round(t*100):03d}'
  complete=json.loads((folder/'complete.json').read_text());manifest=json.loads((folder/'manifest.json').read_text())
  validation=json.loads((folder/'reduction_validation.json').read_text())
  rows=json.loads((folder/'metrics.json').read_text())
  assert complete['status']=='complete' and validation['passed']
  assert len(rows)==24 and len({r['point'] for r in rows})==24
  assert manifest['state']=='model' and manifest['null_class']==1000 and not manifest['flip']
  for r in rows:
   assert r['images']==2048 and r['noises']==128 and r['dimensions']==196608
   for k,v in r.items():
    if isinstance(v,(float,int)):assert math.isfinite(v)
   assert math.isclose(r['between_corrected'],r['between_raw']-r['within']/128,rel_tol=1e-12,abs_tol=1e-9)
   assert math.isclose(r['ratio_corrected'],r['between_corrected']/r['within'],rel_tol=1e-12)
   per_event.append(dict(model='whiten_'+pred,t_clean=t,**r))
  b=statistics.mean(r['between_corrected_per_coordinate'] for r in rows)
  w=statistics.mean(r['within_per_coordinate'] for r in rows)
  summary.append(dict(model='whiten_'+pred,t_clean=t,scope='workspace',points=24,B_per_coordinate=b,W_per_coordinate=w,mean_point_ratio=statistics.mean(r['ratio_corrected'] for r in rows),ratio_of_mean_traces=b/w,last_point=rows[-1]['point'],last_ratio=rows[-1]['ratio_corrected']))
  audits.append(dict(model=pred,t_clean=t,**complete,reduction_validation=validation))
assert len(per_event)==144
current=np.load(ROOT/'samples.npz');old=np.load(ARCHIVE/'samples.npz')
assert all(np.array_equal(current[k],old[k]) for k in current.files)
csv_write(ROOT/'summary.csv',summary);csv_write(ROOT/'per_event.csv',per_event)
(ROOT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
(ROOT/'audit.json').write_text(json.dumps({'status':'passed','tasks':6,'records':144,'original_sample_ids_labels_exact':True,'task_audits':audits},indent=2)+'\n')
with (ARCHIVE/'summary.csv').open() as f: historical=[r for r in csv.DictReader(f) if r['model'] in ('plain_jit_clean','plain_jit_velocity')]
comparisons=[]
for r in summary:
 pred=r['model'].removeprefix('whiten_')
 h=next(h for h in historical if h['model']=='plain_jit_'+pred and float(h['t_clean'])==r['t_clean'])
 comparisons.append(dict(model=r['model'],t_clean=r['t_clean'],whiten_R=r['mean_point_ratio'],historical_plain_R=float(h['mean_point_ratio']),whiten_over_historical_plain_R=r['mean_point_ratio']/float(h['mean_point_ratio']),comparison_scope='historical reference; plain models not rerun in current environment'))
csv_write(ROOT/'historical_plain_comparison.csv',comparisons)
(ROOT/'figures').mkdir(exist_ok=True)
colors={'clean':'#2166ac','velocity':'#d6604d'}
plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
fig,axes=plt.subplots(1,3,figsize=(14,4.3),layout='constrained')
for ax,t in zip(axes,(.25,.5,.75)):
 for pred in ('clean','velocity'):
  rs=[r for r in per_event if r['model']=='whiten_'+pred and r['t_clean']==t]
  ax.plot(range(1,25),[r['ratio_corrected'] for r in rs],label='Whiten '+pred,color=colors[pred],lw=2)
 ax.set(title=f'Clean fraction t = {t:.2f}',xlabel='Residual event (attention, MLP)')
 ax.grid(alpha=.2);ax.set_xticks([1,4,8,12,16,20,24])
axes[0].set_ylabel('R = corrected B / W');axes[-1].legend()
for ext in ('png','pdf'):fig.savefig(ROOT/'figures'/f'ratio_by_depth.{ext}',dpi=180)
plt.close(fig)
fig,axes=plt.subplots(2,3,figsize=(14,7),layout='constrained')
for col,t in enumerate((.25,.5,.75)):
 for row,(key,label) in enumerate((('between_corrected_per_coordinate','B / D (corrected)'),('within_per_coordinate','W / D'))):
  ax=axes[row,col]
  for pred in ('clean','velocity'):
   rs=[r for r in per_event if r['model']=='whiten_'+pred and r['t_clean']==t]
   ax.plot(range(1,25),[r[key] for r in rs],label=pred,color=colors[pred],lw=2)
  ax.set_title(f'Clean fraction t = {t:.2f}');ax.set_ylabel(label);ax.grid(alpha=.2)
  ax.set_xticks([1,4,8,12,16,20,24])
  if row==1:ax.set_xlabel('Residual event (attention, MLP)')
axes[0,-1].legend()
for ext in ('png','pdf'):fig.savefig(ROOT/'figures'/f'variance_by_depth.{ext}',dpi=180)
plt.close(fig)
lines=['# Whitening normalized representation sensitivity','',
'完整复用第 12 项协议：2048 个固定 ImageNet 训练样本，每图 128 个固定 CUDA Gaussian 噪声，t 为 clean fraction，null class 1000，epoch200/step250200 raw model 权重，不用 EMA。24 个 norm1/norm2 输入，逐 patch 非仿射 RMS（eps=1e-6），无 GAP、无 head。',
'', '输入先从 uint8 转到 [-1,1]，用各 checkpoint 保存且完全相同的 basis/mean/scales 做 FP32 whitening，然后在 whiten 坐标加入 isotropic noise。未重新拟合 whitening。',
'', 'B 为 N-1 的 between covariance trace 减去 W/128；W 使用 K-1 的 within covariance trace，不裁剪负修正值。下面 B/D 与 W/D 是 24 个观测点的均值；R 是 24 个点的 B/W 均值，并非表内均值 B 除以均值 W。',
'', '| Prediction | t clean | Mean B/D | Mean W/D | Mean event R | Last-event R |', '|---|---:|---:|---:|---:|---:|']
for r in summary:lines.append(f"| {r['model']} | {r['t_clean']:.2f} | {r['B_per_coordinate']:.8f} | {r['W_per_coordinate']:.8f} | {r['mean_point_ratio']:.6f} | {r['last_ratio']:.6f} |")
lines += ['', '## 同次运行的 clean/velocity 对比','', '| t clean | Clean R / velocity R | Clean W / velocity W | Clean B / velocity B |','|---|---:|---:|---:|']
for t in (.25,.5,.75):
 c=next(r for r in summary if r['model']=='whiten_clean' and r['t_clean']==t);v=next(r for r in summary if r['model']=='whiten_velocity' and r['t_clean']==t)
 lines.append(f"| {t:.2f} | {c['mean_point_ratio']/v['mean_point_ratio']:.6f} | {c['W_per_coordinate']/v['W_per_coordinate']:.6f} | {c['B_per_coordinate']/v['B_per_coordinate']:.6f} |")
lines += ['', '## 历史未 whitening 参考','', '以下 plain 值来自原报告，未在本机重新运行。样本 IDs/labels 完全一致，但历史环境为 Torch2.13，本次为 Torch2.9.1，因此只作为历史参考，不声称跨环境严格复现实验或因果效应。', '', '| Model | t clean | Whiten mean R | Historical plain mean R | Whiten / historical plain |','|---|---:|---:|---:|---:|']
for r in comparisons:lines.append(f"| {r['model']} | {r['t_clean']:.2f} | {r['whiten_R']:.6f} | {r['historical_plain_R']:.6f} | {r['whiten_over_historical_plain_R']:.6f} |")
lines += ['', '## 边界与运行记录','', 'B 表示跨图变化，不等于纯语义；W 是 patch RMS 后的噪声敏感度，不能单独当作绝对幅值敏感度。较低 W 可能伴随表征塌缩，应同时检查 B 和 R。结果没有置信区间，不能从这些汇总量直接声称统计显著性。', '', '纯 inference_mode、无优化器、无 W&B、未写原 checkpoint。clean/velocity 分别共用 GPU0/GPU4，PyTorch allocator cap 为每进程 2 GiB；CUDA runtime 额外占用不包含在 cap 内。原 H-wide 训练未停、未重启、未改配置。', '', '统计直接复用原 streaming_stats.py；只编译 moment reduction，并在每项任务首个观测点与 eager moments 校验。全部 6 项 complete 和 144 条完整观测才纳入汇总。', '', '- `summary.csv/json`：深度均值。', '- `per_event.csv`：保留原始 B_raw、B_corrected、W、两个比值及每维结果。', '- `runs/*/`：每项 manifest、metrics、complete、reduction validation。', '- `audit.json`：完整性、有限值与有限 K 修正检查。', '- `figures/ratio_by_depth.png/pdf`、`figures/variance_by_depth.png/pdf`：独立图件。', '', '![逐层 B/W](figures/ratio_by_depth.png)', '', '![逐层 B 和 W](figures/variance_by_depth.png)']
(ROOT/'REPORT.md').write_text('\n'.join(lines)+'\n')
print(json.dumps(summary,indent=2))
