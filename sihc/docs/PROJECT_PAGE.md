# Updating the project page

Live page: https://tongtongliang.github.io/residual-stream-burden/

The editable blog remains in `elucidating_residual_stream_burden/blog/`. Its `writing/manuscript.md` and build scripts are the writing source. This repository publishes only the built `index.html`, referenced assets and the KaTeX license under `docs/`.

After updating and rebuilding the original blog, run from this repository:

```bash
python sihc/tools/sync_project_page.py --source ../elucidating_residual_stream_burden/blog
git diff --stat
git add docs
git commit -m "Update project blog"
git push origin main
```

GitHub Pages automatically publishes `main:/docs`. The sync script replaces the local Paper PDF link with the arXiv link in the published copy, verifies local asset references, and removes obsolete files listed in its previous export manifest. It does not edit the original blog or upload drafts, backups, writing notes or citation-audit files. Do not edit `docs/index.html` directly: the next sync replaces it.
