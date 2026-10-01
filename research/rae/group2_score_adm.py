#!/usr/bin/env python3
"""Score a final Group 2 50K sample archive with the ADM evaluator."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("sample_npz")
    args = parser.parse_args()
    sample = Path(args.sample_npz).resolve()
    reference = Path(os.environ["ADM_REFERENCE_NPZ"]).resolve()
    evaluator = Path(os.environ["ADM_EVAL_ROOT"]).resolve() / "evaluator.py"
    python = os.environ.get("ADM_PYTHON", sys.executable)
    for path in (sample, reference, evaluator):
        if not path.is_file():
            raise FileNotFoundError(path)
    command = [python, str(evaluator), str(reference), str(sample)]
    print("command:", " ".join(command), flush=True)
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
