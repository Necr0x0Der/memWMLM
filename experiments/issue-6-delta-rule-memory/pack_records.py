#!/usr/bin/env python3
"""Pack every run's small provenance files into one ``records/runs.jsonl``.

``runs/`` is gitignored, so the committed record of a run is its metadata:
``*meta.json`` (arguments, seed, steps, data hash, prompt, summary metrics),
dataset ``manifest.json`` files, training logs (``*.log.json``),
``probe_v2.json`` and the plain-text queue logs (``*.out`` / ``*.log``).
They used to be mirrored as ~700 separate files; this writes one JSON line per
file instead: ``{"path": "delta-rule/...", "content": <parsed JSON or text>}``.

Sources, later overriding earlier: entries already in ``runs.jsonl``, loose
files under ``records/`` and ``runs/delta-rule/`` (adds new runs). Rerunning
after new runs therefore only adds or refreshes entries.
Checkpoints and per-question predictions are not included.

    python experiments/issue-6-delta-rule-memory/pack_records.py
"""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RECORDS = HERE / "records"
RUNS = ROOT / "runs"
JSON_NAMES = ("meta.json", "manifest.json", ".log.json", "probe_v2.json")
TEXT_SUFFIXES = (".out", ".log")


def wanted(path: Path) -> str | None:
    if path.name.endswith(JSON_NAMES):
        return "json"
    if path.suffix in TEXT_SUFFIXES:
        return "text"
    return None


def collect(base: Path, found: dict[str, tuple[str, Path]]) -> None:
    for path in sorted(base.rglob("*")):
        kind = wanted(path) if path.is_file() else None
        if kind:
            found[path.relative_to(base).as_posix()] = (kind, path)


def main() -> None:
    out = RECORDS / "runs.jsonl"
    packed: dict[str, object] = {}
    if out.exists():  # entries packed earlier (their loose files may be gone)
        for line in out.read_text().splitlines():
            row = json.loads(line)
            packed[row["path"]] = row["content"]
    found: dict[str, tuple[str, Path]] = {}
    collect(RECORDS, found)
    collect(RUNS, found)  # later wins: the live run directory is the source of truth
    for key, (kind, path) in found.items():
        if key.startswith("delta-rule/") or "/" not in key:
            text = path.read_text().replace(str(Path.home()), "~")  # no personal paths
            packed[key] = json.loads(text) if kind == "json" else text
    with out.open("w") as f:
        for key in sorted(packed):
            f.write(json.dumps({"path": key, "content": packed[key]}, ensure_ascii=False) + "\n")
    print(f"{len(packed)} files -> {out.relative_to(ROOT)} ({out.stat().st_size / 1e6:.1f} MB)")

if __name__ == "__main__":
    main()
