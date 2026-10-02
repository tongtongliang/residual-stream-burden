# Part 3: Semantic Endpoint Patch PCA

This directory measures the semantic residual-stream output immediately before
the separate pixel decoder in DeCO-B, PixelDiT-B, and DiP-B.

- `METRICS.md`: estimator and metric definitions.
- `RESULTS.md`: paper-facing absolute and ratio tables.
- `semantic_endpoint_patch_pca.ipynb`: rendered visual report.
- `data/all_endpoint_metrics.csv`: merged scalar statistics.
- `data/merged/<model>/`: sufficient statistics, eigensystems, provenance,
  and audits.
- `data/shards/<model>/`: eight independently collected GPU shards.

No W&B logging is used.
