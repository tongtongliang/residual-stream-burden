"""Validate completed records and package only diagnostic data/code, not weights."""
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parent
while subprocess.run(['systemctl','--user','is-active','--quiet','normalized-sensitivity-retry.service']).returncode==0:
    time.sleep(10)
entries=json.loads((ROOT/'models.json').read_text())
for entry in entries:
    for t in (25,50,75):
        p=ROOT/'runs'/f"{entry['label']}_t{t:03d}"
        assert (p/'complete.json').exists(),str(p)
        rows=json.loads((p/'metrics.json').read_text())
        for r in rows:
            assert r['images']==2048 and r['noises']==128
            assert all(math.isfinite(r[k]) for k in ('between_raw','within','between_corrected','ratio_corrected'))
            assert 0<=r['within_per_coordinate']<=128/127+1e-5
            assert abs(r['between_corrected']-(r['between_raw']-r['within']/128))<1e-8
(ROOT/'audit.json').write_text(json.dumps(dict(passed=True,tasks=33,images_per_task=2048,noises=128),indent=2))
subprocess.run([sys.executable,str(ROOT/'summarize.py')],check=True)
dest=Path('/data/tongtong/elucidating_residual_stream_burden/analysis_reports/12_normalized_representation_sensitivity')
dest.mkdir(parents=True,exist_ok=True)
for name in ('README.md','REPORT.md','summary.csv','summary.json','models.json','samples.npz','audit.json',
             'streaming_validation.json','production_reduction_validation.json','carrier_write_validation.json'):
    shutil.copy2(ROOT/name,dest/name)
for source in ROOT.glob('*.py'):
    (dest/'code').mkdir(exist_ok=True);shutil.copy2(source,dest/'code'/source.name)
for source in (ROOT/'runs').glob('*'):
    target=dest/'data'/source.name;target.mkdir(parents=True,exist_ok=True)
    for name in ('manifest.json','metrics.json','metrics.csv','complete.json'):
        shutil.copy2(source/name,target/name)
(ROOT/'archive_complete.json').write_text(json.dumps(dict(path=str(dest),tasks=33),indent=2))
print(str(dest),flush=True)
