#!/usr/bin/env python3
"""Pick a memory gate on the calibration split: max memory - shuffled EM, ties -> smaller.

Usage: select_capacity_gate.py CALIB_DIR CONFIG [N] [CONDITION] [--skip-collapsed]
  N          bank size of the calibration runs (default 16, the capacity test)
  CONDITION  memory condition (default "memory"; its control is "shuffled_" + CONDITION)
  --skip-collapsed  ignore gates where both EMs are below 1% (output collapsed; 0 - 0
                    would otherwise beat negative differences). Added 2026-11-19 for
                    the QA-pair check; earlier selections ran without it.
Prints the gate; details on stderr.
"""

import json
import sys
from pathlib import Path


def em(path: Path) -> float:
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    return 100 * sum(r["exact_match"] for r in rows) / len(rows)


def main() -> None:
    skip = "--skip-collapsed" in sys.argv
    argv = [a for a in sys.argv[1:] if a != "--skip-collapsed"]
    root, config = Path(argv[0]), argv[1]
    n = argv[2] if len(argv) > 2 else "16"
    condition = argv[3] if len(argv) > 3 else "memory"
    scores = []
    for g in ("0.05", "0.1", "0.2", "0.5", "1"):
        m = em(root / f"{condition}-{config}-g{g}-N{n}.jsonl")
        s = em(root / f"shuffled_{condition}-{config}-g{g}-N{n}.jsonl")
        collapsed = skip and m < 1 and s < 1
        print(f"{config} gate {g}: {condition} {m:.2f} shuffled {s:.2f} diff {m - s:+.2f}"
              + (" (collapsed, skipped)" if collapsed else ""), file=sys.stderr)
        if not collapsed:
            scores.append((m - s, -float(g), g))
    print(max(scores)[2])


if __name__ == "__main__":
    main()
