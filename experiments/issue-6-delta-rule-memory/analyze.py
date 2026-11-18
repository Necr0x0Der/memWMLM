#!/usr/bin/env python3
"""Summarise issue #6 result files: EM, paired bootstrap CIs, contamination split.

Every ``*.jsonl`` under the given directories is one (condition, N) run keyed by
question id. For each run this reports exact match, the paired difference to a
reference run (default ``closed_book-N1``) with a bootstrap 95% CI, EM split by
whether the reference was correct (contamination / parametric-knowledge split),
and for counterfactual runs the follow (swapped answer) and resist (original
answer) rates over valid counterfactuals.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def load(path: Path) -> dict[str, dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").split("\n") if line]
    return {r["question_id"]: r for r in rows}


def bootstrap_diff(a: list[float], b: list[float], resamples: int, seed: int) -> tuple[float, float]:
    rng = random.Random(seed)
    n = len(a)
    diffs = []
    for _ in range(resamples):
        idx = [rng.randrange(n) for _ in range(n)]
        diffs.append(sum(a[i] - b[i] for i in idx) / n)
    diffs.sort()
    return 100 * diffs[int(0.025 * resamples)], 100 * diffs[int(0.975 * resamples) - 1]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("dirs", nargs="+", type=Path)
    p.add_argument("--reference", type=Path, default=Path("runs/delta-rule/eval-baselines/closed_book-N1.jsonl"))
    p.add_argument("--resamples", type=int, default=10_000)
    p.add_argument("--seed", type=int, default=20261111)
    p.add_argument("--output", type=Path, default=None)
    args = p.parse_args()

    reference = load(args.reference)
    table = []
    for directory in args.dirs:
        for path in sorted(directory.glob("*.jsonl")):
            run = load(path)
            ids = [i for i in run if i in reference]
            if len(ids) != len(run):
                raise SystemExit(f"{path}: {len(run) - len(ids)} ids missing from reference")
            em = [run[i]["exact_match"] for i in ids]
            ref = [reference[i]["exact_match"] for i in ids]
            known = [i for i in ids if reference[i]["exact_match"]]
            unknown = [i for i in ids if not reference[i]["exact_match"]]
            low, high = bootstrap_diff(em, ref, args.resamples, args.seed)
            entry = {
                "run": f"{directory.name}/{path.stem}",
                "n": len(ids),
                "em": 100 * sum(em) / len(ids),
                "diff_vs_ref": 100 * (sum(em) - sum(ref)) / len(ids),
                "ci95": [low, high],
                "em_ref_correct": 100 * sum(run[i]["exact_match"] for i in known) / max(len(known), 1),
                "em_ref_wrong": 100 * sum(run[i]["exact_match"] for i in unknown) / max(len(unknown), 1),
            }
            cf = [run[i] for i in ids if run[i].get("cf_valid")]
            if cf:
                entry["cf_follow"] = 100 * sum(r["cf_follow_exact_match"] for r in cf) / len(cf)
                entry["cf_resist"] = 100 * sum(r["exact_match"] for r in cf) / len(cf)
            table.append(entry)

    header = f"{'run':58} {'n':>5} {'EM':>6} {'Δref':>6} {'95% CI':>15} {'EM|ref✓':>8} {'EM|ref✗':>8} {'follow':>7}"
    print(header)
    for e in table:
        ci = f"[{e['ci95'][0]:+.1f},{e['ci95'][1]:+.1f}]"
        follow = f"{e['cf_follow']:.1f}" if "cf_follow" in e else ""
        print(f"{e['run']:58} {e['n']:5d} {e['em']:6.1f} {e['diff_vs_ref']:+6.1f} {ci:>15} "
              f"{e['em_ref_correct']:8.1f} {e['em_ref_wrong']:8.1f} {follow:>7}")
    if args.output:
        args.output.write_text(json.dumps(table, indent=2) + "\n")


if __name__ == "__main__":
    main()
