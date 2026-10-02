"""Audit the incremental diagnostic write against explicit spatial-slot algebra."""
import json
from pathlib import Path
import sys
import torch
sys.path.insert(0,'/data/tongtong/sihc')
from sihc.models.sihc.sublayer_kernels import _write
torch.manual_seed(42)
b,g,k,c=2,3,4,64
carrier=torch.randn(b,g*k,g*k,c,device='cuda')
reference=carrier.clone()
errors=[]
for event in range(4):
    delta=torch.randn(b,g*g,c,device='cuda',dtype=torch.bfloat16)
    beta=torch.randn(1,c,k*k,device='cuda',dtype=torch.bfloat16)
    output=torch.empty_like(carrier)
    _write[(b*g*g,1)](carrier,(delta,),beta,output,c,g,k,True,64,num_warps=4)
    update=delta.float().reshape(b,g,1,g,1,c)
    weights=beta.float().reshape(c,k,k).permute(1,2,0).reshape(1,1,k,1,k,c)
    reference=(reference.reshape(b,g,k,g,k,c)+update*weights).reshape_as(carrier)
    errors.append(float((output-reference).abs().max()))
    torch.testing.assert_close(output,reference,rtol=2e-6,atol=2e-6)
    carrier=output
result=dict(passed=True,max_absolute_error=max(errors),events=4,
            layout='B,G,K,G,K,C; beta[event,channel,slot]; FP32 carrier and BF16 updates/coefficients')
(Path(__file__).parent/'carrier_write_validation.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result))
