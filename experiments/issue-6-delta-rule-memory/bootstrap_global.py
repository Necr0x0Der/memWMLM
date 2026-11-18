#!/usr/bin/env python3
"""Paired bootstrap 95% CIs (10,000 resamples, seed 20261107) for global-memory comparisons on dev_in."""

from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path

# run dir with closed_book / rag_bm25@1 files (BOOTSTRAP_REF overrides, e.g. the 4B references)
REF = Path(os.environ.get("BOOTSTRAP_REF", "runs/delta-rule/global-v1/runs/frozen-delta-g0.3"))


def load(path: Path) -> dict[str, float]:
    return {r["question_id"]: r["exact_match"] for r in map(json.loads, path.read_text().splitlines())}


def ci(a: dict, b: dict, n: int = 10_000, seed: int = 20261107) -> tuple[float, float, float]:
    ids = sorted(a)
    diff = [a[i] - b[i] for i in ids]
    rng = random.Random(seed)
    stats = sorted(sum(diff[rng.randrange(len(ids))] for _ in ids) / len(ids) for _ in range(n))
    return 100 * sum(diff) / len(ids), 100 * stats[int(0.025 * n)], 100 * stats[int(0.975 * n)]


def main() -> None:
    closed = load(REF / "closed_book-dev_in.jsonl")
    pairs = [("rag_bm25@1", REF / "rag_bm25@1-dev_in.jsonl", None)]
    for run in sys.argv[1:]:
        d = Path(run)
        label = d.parent.name if d.name == "eval" else d.name
        pairs.append((label, d / "memory-dev_in.jsonl", d / "counterfactual-dev_in.jsonl"))
        if (d / "null-dev_in.jsonl").exists():
            pairs.append((label + " S=0", d / "null-dev_in.jsonl", None))
    print("| run (dev_in) | vs closed-book | vs counterfactual S |")
    print("|---|---|---|")
    for name, path, cf in pairs:
        m = load(path)
        a = "{:+.1f} [{:+.1f}, {:+.1f}]".format(*ci(m, closed))
        b = "{:+.1f} [{:+.1f}, {:+.1f}]".format(*ci(m, load(cf))) if cf and cf.exists() else "—"
        print(f"| {name} | {a} | {b} |")


if __name__ == "__main__":
    main()
