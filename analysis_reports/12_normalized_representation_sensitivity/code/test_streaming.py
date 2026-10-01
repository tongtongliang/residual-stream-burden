"""Direct tensor oracle for the exact production normalization and reductions."""
import json
from pathlib import Path
import torch
from streaming_stats import Streaming,moments

torch.manual_seed(17)
x=torch.randn(17,128,8,12)
normalized=x.double()*torch.rsqrt(x.double().square().mean(-1,keepdim=True)+1e-6)
means=normalized.mean(1)
reference_b=means.var(0,correction=1).sum()
reference_w=normalized.var(1,correction=1).flatten(1).sum(1).mean()
s=Streaming()
for image in x:
    mean,within=moments(image);s.add('point',mean,within)
row=s.rows(128)[0]
torch.testing.assert_close(torch.tensor(row['between_raw'],dtype=torch.float64),reference_b,rtol=2e-6,atol=1e-8)
torch.testing.assert_close(torch.tensor(row['within'],dtype=torch.float64),reference_w,rtol=2e-6,atol=1e-8)
constant=Streaming()
for _ in range(4):constant.add('same_means',torch.ones(3,4),torch.tensor(12.,dtype=torch.float64))
assert constant.rows(128)[0]['between_corrected']<0
zero=Streaming()
for i in range(4):zero.add('no_noise',torch.ones(3,4)*i,torch.tensor(0.,dtype=torch.float64))
assert zero.rows(128)[0]['ratio_corrected'] is None
result=dict(passed=True,between_relative=float(abs(row['between_raw']-reference_b)/reference_b),
            within_relative=float(abs(row['within']-reference_w)/reference_w),
            negative_correction_preserved=True,zero_noise_ratio_undefined=True)
(Path(__file__).parent/'production_reduction_validation.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result))
