"""Package existing results and provenance without inference or checkpoint loads."""
import csv
import importlib.metadata
import json
from pathlib import Path
import shutil
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
WORK = Path('/data/tongtong')
models = json.loads((ROOT / 'models.json').read_text())
meta = ROOT / 'metadata'
meta.mkdir(exist_ok=True)
configs = meta / 'configs'
configs.mkdir(exist_ok=True)
inventory = []
for model in models:
    checkpoint = Path(model['checkpoint'])
    config = Path(model['config']) if 'config' in model else checkpoint.parent.parent / 'config.json'
    item = dict(label=model['label'], checkpoint=str(checkpoint),
                checkpoint_bytes=checkpoint.stat().st_size, state_key=model['state_key'])
    if config.is_file():
        target = configs / (model['label'] + config.suffix)
        shutil.copy2(config, target)
        item.update(config_source=str(config), config_snapshot=str(target.relative_to(ROOT)))
    else:
        item['config_snapshot'] = None
        item['config_note'] = 'No separate config found; see loader source and model registry.'
    inventory.append(item)
(meta / 'checkpoint_inventory.json').write_text(json.dumps(inventory, indent=2) + '\n')
samples = np.load(ROOT / 'samples.npz')
with (meta / 'samples.csv').open('w') as handle:
    writer = csv.writer(handle)
    writer.writerow(['sample_order', 'imagenet_train_index', 'class_label', 'cuda_noise_seed'])
    for order, (index, label) in enumerate(zip(samples['indices'], samples['labels'])):
        writer.writerow([order, int(index), int(label), (1234567 + int(index)*1000003) % (2**63-1)])
rows = []
for task in sorted((ROOT / 'data').iterdir()):
    manifest = json.loads((task / 'manifest.json').read_text())
    assert (task / 'complete.json').is_file()
    for row in json.loads((task / 'metrics.json').read_text()):
        assert row['images'] == 2048 and row['noises'] == 128
        rows.append(dict(model=manifest['model']['label'], t_clean=manifest['t_clean'], **row))
with (ROOT / 'per_event.csv').open('w') as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
cache = WORK / 'latent_caches/raev1_dinov2b_normalized_fp32_noflip'
for name in ('complete.json', 'verification.json'):
    if (cache / name).is_file():
        shutil.copy2(cache / name, meta / ('rae_cache_' + name))
versions = {}
for name in ('torch', 'numpy', 'triton', 'transformer-engine', 'torchvision'):
    try:
        versions[name] = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        versions[name] = None
(meta / 'environment.json').write_text(json.dumps(dict(
    recorded_at_packaging=True, packages=versions,
    worker_python='/data/tongtong/sihc_artifacts/hidden_noise_linear_probe/.venv/bin/python',
    sample_seed=20260914, sample_population=1281167,
    sampling='numpy default_rng choice without replacement',
    tf32_enabled=True, noise_seed_formula='(1234567 + image_index * 1000003) % (2**63-1)',
    cache_root=str(cache), wandb=False), indent=2) + '\n')
for name in ('README.md', 'REPORT.md'):
    path = ROOT / name
    path.write_text(path.read_text().replace('runs/*/metrics.csv', 'data/*/metrics.csv'))
print(json.dumps(dict(tasks=len(list((ROOT/'data').iterdir())), event_rows=len(rows), samples=len(samples['indices']))))
