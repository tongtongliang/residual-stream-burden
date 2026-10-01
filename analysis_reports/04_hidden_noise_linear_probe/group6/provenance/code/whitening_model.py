import sys
sys.path[:0]=['/data/tongtong/sihc','/data/tongtong/sihc_artifacts/hidden_noise_linear_probe','/data/tongtong/sihc_artifacts/normalized_rep_sensitivity']
import json
from pathlib import Path
from models import Extractor
from sihc.checkpoint import load_checkpoint, build_model_from_checkpoint, load_model_state
from sihc.patch_geometry import PatchGeometry

def registry():
    return json.loads((Path(__file__).parent/'models.json').read_text())

def load(entry,device):
    ck = load_checkpoint(entry['checkpoint'],mmap=True)
    assert ck['step']==250200 and ck['args']['prediction']==entry.get('prediction','clean')
    assert ck['patch_geometry']['mode']=='whiten'
    model=build_model_from_checkpoint(ck,attn_backend='flash' if device.type=='cuda' else 'math')
    load_model_state(model,ck,'model')
    ext=Extractor(model.to(device).eval().requires_grad_(False),'plain',1)
    ext.geometry=PatchGeometry.from_checkpoint(ck,device)
    return ext
