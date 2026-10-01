# Supplementary source guide

Start with [README.md](README.md) for installation, model use, training/resume
and evaluation. The default is sublayer SiHC-B + CTX32. Research examples
are independent of the image-model workflows.

- [Flagship XL + REPA](training/FLAGSHIP.md): teacher, training, resume and periodic evaluation.
- [Kernel fusion notes](docs/kernel_fusion/README.md): both topologies and implementation.
- [Toy diagnostics](research/toy/README.md): synthetic models and examples.
- [Models and checkpoints](docs/REPRODUCIBILITY.md).

The archive includes source, configs and tests.
It excludes data, weights, logs, account identifiers and Git history. Required
third-party attribution stays. `SOURCE_INVENTORY.json` records file sizes.

To export a modified copy, keep generated outputs outside the source tree:

```bash
python tools/audit_release.py .
python tools/build_supplement.py --output ../sihc-supplement.zip
```

An optional private denylist must remain outside the release. Archive timestamps
and permissions are normalized. Ordinary checkpoints may contain local metadata;
share weights through the separate allowlisted exporter.
