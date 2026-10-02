"""Download a catalogued checkpoint or evaluation asset without changing its architecture or state."""
import argparse, json
from pathlib import Path

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--id", choices=["sihc-xl-repa", "flagship-xl", "imagenet256-fid-stats"])
    p.add_argument("--repo")
    p.add_argument("--file")
    p.add_argument("--output", default="checkpoints")
    a = p.parse_args()
    catalog = json.loads((Path(__file__).resolve().parents[2] / "sihc/pretrained/checkpoints.json").read_text())
    if a.id == "imagenet256-fid-stats":
        assets = json.loads((Path(__file__).resolve().parents[1] / "pretrained/assets.json").read_text())
        pick = assets[a.id]
    elif a.id:
        pick = catalog["paper_flagship"]
    else:
        if not a.repo or not a.file:
            p.error("provide --id or both --repo and --file")
        pick = next((e for e in catalog["checkpoints"] if e["repo_id"] == a.repo and e["file"] == a.file), None)
        if pick is None:
            p.error("file is not in the checkpoint catalog")
    from huggingface_hub import hf_hub_download
    result = hf_hub_download(repo_id=pick["repo_id"], filename=pick["file"], revision=pick["revision"], local_dir=a.output)
    print(result)

if __name__ == "__main__":
    main()
