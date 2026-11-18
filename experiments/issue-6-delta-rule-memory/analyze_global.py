#!/usr/bin/env python3
"""Contamination split for the global-memory runs (V1 and V2).

"Known" = questions the frozen model answers correctly closed-book (its
pretraining already holds the fact); "unknown" = the rest. A memory that
retrieves facts should help on *unknown* dev_in questions; a generic effect
moves known and unknown alike, and harm shows as losses on known ones.
Prints EM on known / unknown questions per group and run (seed-averaged) and
gained / lost counts vs closed-book.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path


def load(path: Path) -> dict[str, float]:
    return {r["question_id"]: r["exact_match"] for r in map(json.loads, path.read_text().splitlines())}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--global-runs", type=Path, default=Path("runs/delta-rule/global-v1/runs"))
    p.add_argument("--v2", type=Path, nargs="*", default=sorted(Path("runs/delta-rule/v2").glob("*/eval")))
    p.add_argument("--reference", default="frozen-delta-g0.3", help="run dir holding closed_book/rag files")
    args = p.parse_args()
    ref = args.global_runs / args.reference
    groups = ("dev_in", "dev_out", "train_heldout")
    closed = {g: load(ref / f"closed_book-{g}.jsonl") for g in groups}
    runs: dict[str, list[Path]] = defaultdict(list)
    for d in sorted(args.global_runs.iterdir()):
        runs[re.sub(r"-s\d+$", "", d.name)].append(d)
    for d in args.v2:
        runs["v2-" + re.sub(r"-s\d+$", "", d.parent.name)].append(d)
    rows = [("closed_book", ref, "closed_book"), ("rag_bm25@1", ref, "rag_bm25@1")]
    print(f"known questions (closed-book correct): " +
          ", ".join(f"{g} {int(sum(closed[g].values()))}/{len(closed[g])}" for g in groups))
    print("| run | " + " | ".join(f"{g} known / unknown | gained / lost" for g in groups) + " |")
    print("|---|" + "---|---|" * len(groups))
    entries = [(name, [path], prefix) for name, path, prefix in rows]
    entries += [(name, dirs, "memory") for name, dirs in runs.items() if name != "lora-null"]
    entries += [(name + " S=0", dirs, "null") for name, dirs in runs.items() if name.startswith("lora")]
    entries += [("v2-" + name[3:] + " cf", dirs, "counterfactual") for name, dirs in runs.items() if name.startswith("v2-")]
    for name, dirs, prefix in entries:
        cells = []
        for g in groups:
            files = [d / f"{prefix}-{g}.jsonl" for d in dirs if (d / f"{prefix}-{g}.jsonl").exists()]
            if not files:
                cells += ["", ""]
                continue
            kn, un, gain, lost = [], [], [], []
            for f in files:
                em = load(f)
                known = [q for q in em if closed[g].get(q) == 1]
                unknown = [q for q in em if closed[g].get(q) == 0]
                kn.append(100 * sum(em[q] for q in known) / len(known))
                un.append(100 * sum(em[q] for q in unknown) / len(unknown))
                gain.append(sum(em[q] for q in unknown))
                lost.append(sum(1 - em[q] for q in known))
            avg = lambda xs: sum(xs) / len(xs)  # noqa: E731
            cells += [f"{avg(kn):.1f} / {avg(un):.1f}", f"{avg(gain):.0f} / {avg(lost):.0f}"]
        if any(cells):
            print(f"| {name} (n={len(dirs)}) | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
