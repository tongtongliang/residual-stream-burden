"""Run the archived long-skip trainer/evaluator with its original module names."""
import argparse, runpy, sys
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("task", choices=["train", "evaluate"])
    args, rest = parser.parse_known_args()
    root = Path(__file__).resolve().parent / "vendor"
    sys.path.insert(0, str(root))
    name = "train_imagenet256.py" if args.task == "train" else "evaluate_fid.py"
    sys.argv = [name, *rest]
    runpy.run_path(str(root / "scripts" / name), run_name="__main__")

if __name__ == "__main__":
    main()
