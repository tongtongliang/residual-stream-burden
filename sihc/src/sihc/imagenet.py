"""ImageNet-256 dataset helpers.

Supports standard ImageFolder directories and EDM/REPA-style zip datasets with
`dataset.json` labels. Images are returned as uint8 tensors so training can use
in-place normalization on GPU.
"""
import json
import os
import zipfile
from bisect import bisect_right
from io import BytesIO
from pathlib import Path

import numpy as np
try:
    import pyarrow.parquet as pq
except ImportError:  # Parquet is optional; tensor caches and ImageFolder do not need it.
    pq = None
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, DistributedSampler, Sampler
from torchvision import datasets, transforms


def center_crop_arr(pil_image, image_size):
    # Matches the ADM/EDM center-crop convention commonly used for ImageNet-256.
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(tuple(x // 2 for x in pil_image.size), resample=Image.Resampling.BOX)
    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(tuple(round(x * scale) for x in pil_image.size), resample=Image.Resampling.BICUBIC)
    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])


class ZipImageNet256(Dataset):
    def __init__(self, zip_path, transform=None):
        self.zip_path = str(zip_path)
        self.transform = transform
        self.zip = None
        with zipfile.ZipFile(self.zip_path) as zf:
            names = sorted(n for n in zf.namelist() if n.lower().endswith((".png", ".jpg", ".jpeg")))
            if "dataset.json" in zf.namelist():
                labels = dict(json.loads(zf.read("dataset.json"))["labels"])
            else:
                labels = self._labels_from_paths(names)
        self.names = names
        self.labels = [int(labels[n.replace('\\', '/')]) for n in names]

    @staticmethod
    def _labels_from_paths(names):
        roots = sorted({n.split('/')[0] for n in names if '/' in n})
        if len(roots) != 1000:
            raise RuntimeError("zip has no dataset.json and does not look like class-folder ImageNet")
        class_to_idx = {c: i for i, c in enumerate(roots)}
        return {n: class_to_idx[n.split('/')[0]] for n in names}

    def _zipfile(self):
        if self.zip is None:
            self.zip = zipfile.ZipFile(self.zip_path)
        return self.zip

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        with self._zipfile().open(self.names[idx], 'r') as f:
            img = Image.open(f).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, self.labels[idx]


class ParquetImageNet(Dataset):
    def __init__(self, root, split="train", transform=None):
        if pq is None:
            raise RuntimeError("Parquet datasets require pyarrow")
        self.root = Path(root)
        self.split = "validation" if split == "val" else split
        self.transform = transform
        shard_dir = self.root / self.split
        files = sorted(shard_dir.glob("*.parquet"))
        if not files:
            raise FileNotFoundError(f"no parquet shards found in {shard_dir}")

        self.row_groups = []
        offsets = [0]
        for path in files:
            pf = pq.ParquetFile(path)
            for rg in range(pf.num_row_groups):
                n = pf.metadata.row_group(rg).num_rows
                self.row_groups.append((str(path), rg, n))
                offsets.append(offsets[-1] + n)
        self.offsets = np.asarray(offsets, dtype=np.int64)
        self.total_rows = int(self.offsets[-1])
        self._cache_key = None
        self._cache_batch = None
        self._parquet_files = {}

    def __len__(self):
        return self.total_rows

    def _parquet_file(self, path):
        pf = self._parquet_files.get(path)
        if pf is None:
            pf = pq.ParquetFile(path)
            self._parquet_files[path] = pf
        return pf

    def _read_row_group(self, group_idx):
        path, row_group, _ = self.row_groups[int(group_idx)]
        key = (path, row_group)
        if self._cache_key != key:
            table = self._parquet_file(path).read_row_group(row_group, columns=["image", "label"])
            self._cache_batch = table.to_pydict()
            self._cache_key = key
        return self._cache_batch

    def __getitem__(self, idx):
        idx = int(idx)
        if idx < 0 or idx >= self.total_rows:
            raise IndexError(idx)
        group_idx = bisect_right(self.offsets, idx) - 1
        local_idx = idx - int(self.offsets[group_idx])
        batch = self._read_row_group(group_idx)
        image = batch["image"][local_idx]
        if isinstance(image, Image.Image):
            img = image.convert("RGB")
        elif image.get("bytes") is not None:
            img = Image.open(BytesIO(image["bytes"])).convert("RGB")
        else:
            img = Image.open(image["path"]).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, int(batch["label"][local_idx])


class ParquetDistributedSampler(Sampler):
    def __init__(self, dataset, num_replicas, rank, batch_size, shuffle=True, seed=0, drop_last=True):
        self.dataset = dataset
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.drop_last = bool(drop_last)
        self.epoch = 0

        global_batch = self.num_replicas * self.batch_size
        if self.drop_last:
            self.num_batches = len(self.dataset) // global_batch
        else:
            self.num_batches = int(np.ceil(len(self.dataset) / global_batch))
        self.num_samples = self.num_batches * self.batch_size
        self.total_size = self.num_batches * global_batch

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def _ordered_indices(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        group_order = np.arange(len(self.dataset.row_groups), dtype=np.int64)
        if self.shuffle:
            rng.shuffle(group_order)

        parts = []
        remaining = self.total_size
        for group_idx in group_order:
            start = int(self.dataset.offsets[int(group_idx)])
            n = int(self.dataset.row_groups[int(group_idx)][2])
            local = np.arange(n, dtype=np.int64)
            if self.shuffle:
                rng.shuffle(local)
            values = start + local
            if values.size >= remaining:
                parts.append(values[:remaining])
                remaining = 0
                break
            parts.append(values)
            remaining -= values.size

        indices = np.concatenate(parts) if parts else np.empty(0, dtype=np.int64)
        if indices.size < self.total_size:
            if self.drop_last:
                raise RuntimeError("not enough parquet samples to form one distributed epoch")
            repeats = int(np.ceil((self.total_size - indices.size) / max(indices.size, 1)))
            indices = np.concatenate([indices, np.tile(indices, repeats)[: self.total_size - indices.size]])
        return indices[: self.total_size]

    def __iter__(self):
        indices = self._ordered_indices()
        indices = indices.reshape(self.num_batches, self.num_replicas, self.batch_size)
        rank_indices = indices[:, self.rank, :].reshape(-1)
        return iter(rank_indices.tolist())


class CachedTensorImageNet256(Dataset):
    """Lossless ADM-cropped uint8 tensor cache.

    The cache stores center-cropped CHW uint8 tensors and labels. Training keeps
    RandomHorizontalFlip online so the model sees the same tensor convention as
    the live parquet/PIL path, just without repeated JPEG decode and ADM crop.
    """
    def __init__(self, cache_root, train=True):
        self.cache_root = Path(cache_root)
        manifest_path = self.cache_root / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"missing tensor cache manifest: {manifest_path}")
        manifest = json.loads(manifest_path.read_text())
        if int(manifest.get("image_size", 0)) != 256:
            raise ValueError(f"expected image_size=256 cache, got {manifest.get('image_size')}")
        self.train = bool(train)
        self.shards = manifest["shards"]
        offsets = [0]
        for shard in self.shards:
            offsets.append(offsets[-1] + int(shard["count"]))
        self.offsets = np.asarray(offsets, dtype=np.int64)
        self.total_rows = int(self.offsets[-1])
        group_mode = os.environ.get("IMAGENET256_TENSOR_CACHE_GROUPS", "auto").strip().lower()
        if group_mode in {"cache_shard", "cache-shard", "cache", "shard"}:
            group_counts = [int(shard["count"]) for shard in self.shards]
            self.sample_group_kind = "cache_shard"
        elif group_mode in {"auto", "source_parquet_row_group", "source-parquet-row-group", "row_group", "row-group", "parquet"}:
            group_counts = self._source_parquet_row_group_counts(manifest)
            if group_counts is None:
                group_counts = [int(shard["count"]) for shard in self.shards]
                self.sample_group_kind = "cache_shard"
            else:
                self.sample_group_kind = "source_parquet_row_group"
        else:
            raise ValueError(
                "IMAGENET256_TENSOR_CACHE_GROUPS must be 'auto', 'source_parquet_row_group', or 'cache_shard', "
                f"got {group_mode!r}"
            )
        group_offsets = [0]
        for count in group_counts:
            group_offsets.append(group_offsets[-1] + int(count))
        self.sample_group_counts = group_counts
        self.sample_group_offsets = np.asarray(group_offsets, dtype=np.int64)
        self._cache_idx = None
        self._cache_images = None
        self._cache_labels = None

    def _source_parquet_row_group_counts(self, manifest):
        if pq is None:
            return None
        source = manifest.get("source")
        if not source:
            return None
        shard_dir = Path(source) / "train"
        if not shard_dir.is_dir():
            return None
        files = sorted(shard_dir.glob("*.parquet"))
        if not files:
            return None
        counts = []
        for path in files:
            pf = pq.ParquetFile(path)
            for rg in range(pf.num_row_groups):
                counts.append(int(pf.metadata.row_group(rg).num_rows))
        if sum(counts) != self.total_rows:
            return None
        return counts

    def __len__(self):
        return self.total_rows

    def _load_shard(self, shard_idx):
        shard_idx = int(shard_idx)
        if self._cache_idx != shard_idx:
            shard = self.shards[shard_idx]
            self._cache_images = np.load(self.cache_root / shard["images"], mmap_mode="r")
            self._cache_labels = np.load(self.cache_root / shard["labels"], mmap_mode="r")
            self._cache_idx = shard_idx
        return self._cache_images, self._cache_labels

    def __getitem__(self, idx):
        idx = int(idx)
        if idx < 0 or idx >= self.total_rows:
            raise IndexError(idx)
        shard_idx = bisect_right(self.offsets, idx) - 1
        local_idx = idx - int(self.offsets[shard_idx])
        images, labels = self._load_shard(shard_idx)
        x = torch.from_numpy(np.array(images[local_idx], copy=True))
        if self.train and torch.rand(()) < 0.5:
            x = torch.flip(x, dims=(2,))
        return x, int(labels[local_idx])


class CachedTensorImageNet256Float(CachedTensorImageNet256):
    """Cached uint8 images exposed in the [0, 1] range."""

    def __getitem__(self, idx):
        image, label = super().__getitem__(idx)
        return image.float().mul_(1.0 / 255.0), label


class CachedTensorDistributedSampler(Sampler):
    def __init__(self, dataset, num_replicas, rank, batch_size, shuffle=True, seed=0, drop_last=True):
        self.dataset = dataset
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.drop_last = bool(drop_last)
        self.epoch = 0

        global_batch = self.num_replicas * self.batch_size
        if self.drop_last:
            self.num_batches = len(self.dataset) // global_batch
        else:
            self.num_batches = int(np.ceil(len(self.dataset) / global_batch))
        self.num_samples = self.num_batches * self.batch_size
        self.total_size = self.num_batches * global_batch

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def _ordered_indices(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        group_order = np.arange(len(self.dataset.sample_group_counts), dtype=np.int64)
        if self.shuffle:
            rng.shuffle(group_order)

        parts = []
        remaining = self.total_size
        for group_idx in group_order:
            start = int(self.dataset.sample_group_offsets[int(group_idx)])
            n = int(self.dataset.sample_group_counts[int(group_idx)])
            local = np.arange(n, dtype=np.int64)
            if self.shuffle:
                rng.shuffle(local)
            values = start + local
            if values.size >= remaining:
                parts.append(values[:remaining])
                remaining = 0
                break
            parts.append(values)
            remaining -= values.size

        indices = np.concatenate(parts) if parts else np.empty(0, dtype=np.int64)
        if indices.size < self.total_size:
            if self.drop_last:
                raise RuntimeError("not enough cached samples to form one distributed epoch")
            repeats = int(np.ceil((self.total_size - indices.size) / max(indices.size, 1)))
            indices = np.concatenate([indices, np.tile(indices, repeats)[: self.total_size - indices.size]])
        return indices[: self.total_size]

    def __iter__(self):
        indices = self._ordered_indices()
        indices = indices.reshape(self.num_batches, self.num_replicas, self.batch_size)
        rank_indices = indices[:, self.rank, :].reshape(-1)
        return iter(rank_indices.tolist())


def build_transform(image_size=256, train=True):
    ops = [transforms.Lambda(lambda img: center_crop_arr(img, image_size))]
    if train:
        ops.append(transforms.RandomHorizontalFlip())
    ops.append(transforms.PILToTensor())
    return transforms.Compose(ops)


def _split_names(split):
    if split in {"val", "validation"}:
        return ("val", "validation")
    return (split,)


def build_dataset(data_path, split='train', image_size=256):
    data_path = Path(data_path)
    if (data_path / "manifest.json").is_file():
        return CachedTensorImageNet256(data_path, train=(split == 'train'))
    transform = build_transform(image_size=image_size, train=(split == 'train'))
    for name in _split_names(split):
        split_dir = data_path / name
        if split_dir.is_dir() and any(split_dir.glob("*.parquet")):
            return ParquetImageNet(data_path, split=name, transform=transform)
        if split_dir.is_dir():
            return datasets.ImageFolder(split_dir, transform=transform)
    zip_candidates = [data_path / f'{name}.zip' for name in _split_names(split)]
    zip_candidates.append(data_path / 'images.zip')
    for z in zip_candidates:
        if z.is_file():
            return ZipImageNet256(z, transform=transform)
    raise FileNotFoundError(f"no ImageFolder split or zip found under {data_path}")


def build_loader(dataset, batch_size, num_workers=12, distributed=False, rank=0, world_size=1, seed=0, drop_last=True):
    sampler = None
    shuffle = True
    if distributed:
        if isinstance(dataset, ParquetImageNet):
            sampler = ParquetDistributedSampler(
                dataset,
                num_replicas=world_size,
                rank=rank,
                batch_size=batch_size,
                shuffle=True,
                seed=seed,
                drop_last=drop_last,
            )
        elif isinstance(dataset, CachedTensorImageNet256):
            sampler = CachedTensorDistributedSampler(
                dataset,
                num_replicas=world_size,
                rank=rank,
                batch_size=batch_size,
                shuffle=True,
                seed=seed,
                drop_last=drop_last,
            )
        else:
            sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=seed)
        shuffle = False
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=drop_last,
        persistent_workers=num_workers > 0,
        prefetch_factor=4 if num_workers > 0 else None,
    )
