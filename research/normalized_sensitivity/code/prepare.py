"""Validate streaming against a direct oracle and register fixed inputs."""
import json
from pathlib import Path
import sys
import numpy as np
import torch
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'hidden_noise_linear_probe'))
from models import registry
from job import data_module
from latent_cache import CACHE,CachedLatents
from streaming_stats import Streaming

torch.manual_seed(42)
x=torch.randn(9,13,7,5,dtype=torch.float64)
x=x*torch.rsqrt(x.square().mean(-1,keepdim=True)+1e-6)
means=x.mean(1);v=x.var(1,correction=1).flatten(1).sum(1)
expected_b=float(means.var(0,correction=1).sum());expected_w=float(v.mean())
errors=[]
for order in (list(range(9)),list(reversed(range(9)))):
    s=Streaming()
    for i in order:s.add('test',means[i],v[i])
    r=s.rows(13)[0]
    errors.extend([abs(r['between_raw']-expected_b),abs(r['within']-expected_w),
                   abs(r['between_corrected']-(expected_b-expected_w/13))])
assert max(errors)<1e-12
(ROOT/'streaming_validation.json').write_text(json.dumps(dict(passed=True,max_absolute_error=max(errors),
    oracle='FP64 explicit per-image unbiased noise covariance and unbiased covariance of image means; forward and reversed image orders'),indent=2))
entries=registry()
old=json.loads((ROOT.parent/'residual_stream_noise_sensitivity/model_registry.json').read_text())
entries += [dict(e,group=1,kind='external' if e['family']=='external' else 'hyper')
            for e in old if e['label'] not in ('plain_jit_clean','plain_jit_velocity')]
assert len(entries)==11
for e in entries: assert Path(e['checkpoint']).is_file(),e['checkpoint']
(ROOT/'models.json').write_text(json.dumps(entries,indent=2))
dataset=data_module().TrainImages();latents=CachedLatents(CACHE,'train')
assert len(dataset)==len(latents)
np.testing.assert_array_equal(dataset.labels,latents.labels)
ids=np.random.default_rng(20260914).choice(len(dataset),2048,replace=False)
np.savez(ROOT/'samples.npz',indices=ids,labels=np.asarray(dataset.labels)[ids])
print(json.dumps(dict(validation='passed',models=len(entries),images=len(ids),dataset_size=len(dataset))),flush=True)
