#!/usr/bin/env python3
"""Pick the capacity-test gate on the calibration split: max memory - shuffled EM at N=16, ties -> smaller.

Usage: select_capacity_gate.py CALIB_DIR CONFIG   (prints the gate; details on stderr)
"""

import json
import sys
from pathlib import Path


def em(path: Path) -> float:
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    return 100 * sum(r["exact_match"] for r in rows) / len(rows)


def main() -> None:
    root, config = Path(sys.argv[1]), sys.argv[2]
    scores = []
    for g in ("0.05", "0.1", "0.2", "0.5", "1"):
        m, s = em(root / f"memory-{config}-g{g}-N16.jsonl"), em(root / f"shuffled_memory-{config}-g{g}-N16.jsonl")
        print(f"{config} gate {g}: memory {m:.2f} shuffled {s:.2f} diff {m - s:+.2f}", file=sys.stderr)
        scores.append((m - s, -float(g), g))
    print(max(scores)[2])


if __name__ == "__main__":
    main()
