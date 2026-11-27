#!/usr/bin/env python3
"""Generate all synthetic datasets used by the multi-commit experiment.

No arguments. Run from the repository root:

    PYTHONPATH=src python experiments/multicommit-memory/generate_datasets.py

The datasets are reproducible from the fixed seed and generator parameters.
They are generated locally and are not committed to git.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = REPO_ROOT / "experiments" / "multicommit-memory"

DATASETS = [
    ("generate.py", "datasets/multicommit-memory", 1200, 300),
    ("generate_simple.py", "datasets/multicommit-simple3", 2000, 400),
    ("generate_two.py", "datasets/multicommit-two", 2000, 400),
]


def run_generator(script: str, output_dir: str, train_episodes: int, test_episodes: int) -> None:
    command = [
        sys.executable,
        str(EXPERIMENT_DIR / script),
        "--output-dir",
        output_dir,
        "--seed",
        "20261125",
        "--train-episodes",
        str(train_episodes),
        "--test-episodes",
        str(test_episodes),
    ]
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def main() -> None:
    for script, output_dir, train_episodes, test_episodes in DATASETS:
        run_generator(script, output_dir, train_episodes, test_episodes)
    print("Generated datasets:")
    for _, output_dir, _, _ in DATASETS:
        print(f"  {output_dir}")


if __name__ == "__main__":
    main()
