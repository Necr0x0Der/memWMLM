#!/usr/bin/env python3
"""Does the dropped first token (2026-11-18 review) hide a memory effect?

dev_in passages whose text starts with an answer alias lost (part of) the
answer from the write. Recompute memory − counterfactual on the remaining
dev_in questions for every global run directory with both prediction files.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bootstrap_global import ci, load  # noqa: E402

data = Path("runs/delta-rule/global-v1")
corpus = [json.loads(line) for line in (data / "corpus.jsonl").open()]
questions = {r["question_id"]: r for r in map(json.loads, (data / "questions.jsonl").open())}
affected = {r["question_id"] for r in corpus if r["group"] == "dev_in" and any(
    a and r["passage"].lower().lstrip().startswith(a.lower()) for a in questions[r["question_id"]]["aliases"])}
print(f"dev_in passages starting with an answer alias: {len(affected)}")
print("| run | all: memory − cf | answer fully written: memory − cf |\n|---|---|---|")
for m in sorted(Path("runs/delta-rule").rglob("memory-dev_in*.jsonl")):
    cf = m.with_name(m.name.replace("memory-", "counterfactual-"))
    if "corpus" in m.name or "smoke" in m.parts or not cf.exists():
        continue
    a, b = load(m), load(cf)
    ids = sorted(set(a) & set(b))
    keep = [i for i in ids if i not in affected]
    full = ci({i: a[i] for i in ids}, {i: b[i] for i in ids})
    part = ci({i: a[i] for i in keep}, {i: b[i] for i in keep})
    print(f"| {m.parent.relative_to('runs/delta-rule')}/{m.stem.replace('memory-dev_in', '')} | "
          f"{full[0]:+.1f} [{full[1]:+.1f}, {full[2]:+.1f}] | {part[0]:+.1f} [{part[1]:+.1f}, {part[2]:+.1f}] (n {len(keep)}) |")
