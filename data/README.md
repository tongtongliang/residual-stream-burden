# Image data

The image trainer accepts three local inputs:

- ImageFolder: `ROOT/train/<class_name>/*.JPEG` (directory class names sort into labels).
- Parquet shards under `ROOT/train/`, with encoded `image` bytes and a `label` column; install the `parquet` extra.
- A prepared uint8 cache whose root contains `manifest.json`, CHW image shards and label shards.

All recipes use 256px ADM-style center cropping. Training adds random horizontal
flips; cache construction stores the center crop without flips. Inputs are
scaled to [-1,1] by the image trainer. ImageNet itself is not redistributed.

For an existing parquet dataset:

```bash
python -m pip install -e '.[parquet]'
python -m data.prepare_cache --data_path /path/to/parquet \
  --cache_root /path/to/cache --num_workers 12
```

Cache files can be large. Generated manifests record their local input path;
keep datasets/caches outside the anonymous source release. A full ImageNet
cache has 1,281,167 examples; a partial cache is useful for local validation but
requires a different epoch/step interpretation.
