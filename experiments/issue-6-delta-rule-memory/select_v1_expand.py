#!/usr/bin/env python3
"""Apply the pre-registered gate selection to the expanded-key V1 runs.

Per run directory: gate = argmax train_seen memory EM (ties -> smaller gate);
report that gate's EM on every group and counterfactual follow, next to the
closed-book references.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

# argv[1] = "4b" selects the Qwen3-4B runs (README amendment 2026-11-17)
# optional argv[2] = another run root, e.g. runs/delta-rule/expand-fixed/4b (README amendment 2026-11-18)
FOUR_B = len(sys.argv) > 1 and sys.argv[1] == "4b"
ROOT = Path(sys.argv[2] if len(sys.argv) > 2
            else "runs/delta-rule/qwen3-4b/global/expand" if FOUR_B else "runs/delta-rule/global-v1/expand")
CLOSED = ({"train_seen": 39.65, "train_heldout": 40.62, "dev_in": 35.16, "dev_out": 32.42} if FOUR_B
          else {"train_seen": 21.9, "train_heldout": 25.6, "dev_in": 18.2, "dev_out": 16.0})


def main() -> None:
    print("| run | gate | train_seen | train_heldout | dev_in | dev_out | cf dev_in (follow) |")
    print("|---|---|---|---|---|---|---|")
    print("| closed-book | — | " + " | ".join(f"{CLOSED[g]}" for g in CLOSED) + " | — |")
    for d in sorted(ROOT.iterdir(), key=lambda p: (p.name.split("-D")[0], int(p.name.split("-D")[1]))):
        if not (d / "meta.json").exists():
            continue
        s = {x["run"]: x for x in json.loads((d / "meta.json").read_text())["summaries"]}
        gates = sorted({float(m.group(1)) for k in s if (m := re.search(r"-g([\d.]+)$", k))})
        best = max(gates, key=lambda g: (s[f"memory-train_seen-g{g}"]["em"], -g))
        em = lambda r: s[f"{r}-g{best}"]["em"]  # noqa: E731
        cf = s[f"counterfactual-dev_in-g{best}"]
        print(f"| {d.name} | {best:g} | {em('memory-train_seen'):.1f} | {em('memory-train_heldout'):.1f} | "
              f"{em('memory-dev_in'):.1f} | {em('memory-dev_out'):.1f} | "
              f"{cf['em']:.1f} ({cf.get("cf_follow", float("nan")):.1f}) |")


if __name__ == "__main__":
    main()
