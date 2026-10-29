"""Parser for the Phase-0 reproduction dataset (JSONL windows).

Conservative: does not modify memWMLM's bAbI loader. The reproduction dataset
is a plain JSONL of {"id", "split", "text"} windows (see dataset/make_dataset.py).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ReproExample:
    """One reproduction window: an id, a split, and the full text to reproduce."""

    example_id: str
    split: str
    text: str


def load_repro_task(path: str | Path) -> list[ReproExample]:
    """Load a JSONL file of ``{"id", "split", "text"}`` windows."""

    examples: list[ReproExample] = []
    path = Path(path)
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw_line.strip():
            continue
        record = json.loads(raw_line)
        examples.append(
            ReproExample(
                example_id=str(record.get("id", line_number)),
                split=str(record.get("split", "train")),
                text=record["text"],
            )
        )
    if not examples:
        raise ValueError(f"No reproduction examples found in {path}")
    return examples
