import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from sihc.imagenet import ParquetImageNet, build_transform


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_path', required=True)
    p.add_argument('--cache_root', required=True)
    p.add_argument('--image_size', type=int, default=256)
    p.add_argument('--batch_size', type=int, default=256)
    p.add_argument('--num_workers', type=int, default=24)
    p.add_argument('--shard_size', type=int, default=4096)
    p.add_argument('--max_samples', type=int, default=0)
    return p.parse_args()


def write_shard(cache_root, shard_idx, images, labels):
    images = np.concatenate(images, axis=0)
    labels = np.concatenate(labels, axis=0).astype(np.int64, copy=False)
    image_name = f'images_{shard_idx:05d}.npy'
    label_name = f'labels_{shard_idx:05d}.npy'
    tmp_image = cache_root / f'.{image_name}.tmp'
    tmp_label = cache_root / f'.{label_name}.tmp'
    with tmp_image.open('wb') as f:
        np.save(f, images, allow_pickle=False)
    with tmp_label.open('wb') as f:
        np.save(f, labels, allow_pickle=False)
    tmp_image.rename(cache_root / image_name)
    tmp_label.rename(cache_root / label_name)
    return {'images': image_name, 'labels': label_name, 'count': int(labels.shape[0])}


def main():
    args = parse_args()
    cache_root = Path(args.cache_root)
    cache_root.mkdir(parents=True, exist_ok=True)
    transform = build_transform(image_size=args.image_size, train=False)
    dataset = ParquetImageNet(args.data_path, split='train', transform=transform)
    if args.max_samples > 0:
        dataset.total_rows = min(dataset.total_rows, args.max_samples)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
        drop_last=False,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=4 if args.num_workers > 0 else None,
    )
    shards = []
    buf_images, buf_labels = [], []
    buffered = 0
    written = 0
    t0 = time.time()
    for x, y in loader:
        x_np = x.numpy()
        y_np = y.numpy()
        pos = 0
        while pos < x_np.shape[0]:
            take = min(args.shard_size - buffered, x_np.shape[0] - pos)
            buf_images.append(x_np[pos:pos + take].copy())
            buf_labels.append(y_np[pos:pos + take].copy())
            buffered += take
            pos += take
            if buffered == args.shard_size:
                shards.append(write_shard(cache_root, len(shards), buf_images, buf_labels))
                written += buffered
                dt = time.time() - t0
                print(f'wrote={written} shards={len(shards)} rate={written / max(dt, 1e-9):.2f} img/s', flush=True)
                buf_images, buf_labels, buffered = [], [], 0
    if buffered:
        shards.append(write_shard(cache_root, len(shards), buf_images, buf_labels))
        written += buffered
    manifest = {
        'version': 1,
        'source': os.path.abspath(args.data_path),
        'image_size': args.image_size,
        'layout': 'chw_uint8_adm_center_crop_no_flip',
        'num_samples': written,
        'shards': shards,
    }
    (cache_root / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    dt = time.time() - t0
    print(f'done samples={written} shards={len(shards)} seconds={dt:.1f} rate={written / max(dt, 1e-9):.2f} img/s')


if __name__ == '__main__':
    main()
