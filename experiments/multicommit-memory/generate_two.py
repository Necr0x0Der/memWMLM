#!/usr/bin/env python3
"""Generate a two-entity, three-commit versioned-location dataset.

Alice moves twice (commit 1 -> 2 -> 3), Bob moves once (commit 1 -> 2) and
remains in commit 3.  Questions probe current, previous, and original
locations.  The last-commit-only control can answer current facts but cannot
answer earlier facts, so this is a direct multi-commit test.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

CITIES = ["Paris", "Rome", "Berlin", "Madrid", "London", "Lisbon"]
TRAIN_NAMES = ["Alice", "Bob", "Carol", "David", "Erin", "Frank"]
TEST_NAMES = ["Grace", "Henry", "Iris", "Jack", "Kate", "Leo"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_split(split: str, names: list[str], episodes: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    records = []
    for index in range(episodes):
        mover, stayer = rng.sample(names, 2)
        a1, a2, a3, b1, b2 = rng.sample(CITIES, 5)
        docs = [
            f"{mover} lives in {a1}. {stayer} lives in {b1}.",
            f"{mover} moved from {a1} to {a2}. {stayer} moved from {b1} to {b2}.",
            f"{mover} moved from {a2} to {a3}. {stayer} remains in {b2}.",
        ]
        questions = [
            ("current", f"Where is {mover} now?", a3),
            ("previous", f"Where was {mover} before the last move?", a2),
            ("original", f"Where did {mover} live originally?", a1),
            ("current", f"Where is {stayer} now?", b2),
            ("previous", f"Where was {stayer} before the last move?", b1),
        ]
        for q_index, (q_type, question, target) in enumerate(questions):
            records.append(
                {
                    "example_id": f"{split}_{index:05d}_{q_index}",
                    "split": split,
                    "episode_id": f"{split}_{index:05d}",
                    "question_type": q_type,
                    "question": question,
                    "target": target,
                    "source_documents": docs,
                    "metadata": {
                        "mover": mover,
                        "stayer": stayer,
                        "mover_history": [a1, a2, a3],
                        "stayer_history": [b1, b2],
                        "cities_used": [a1, a2, a3, b1, b2],
                    },
                }
            )
    return records


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261125)
    parser.add_argument("--train-episodes", type=int, default=2000)
    parser.add_argument("--test-episodes", type=int, default=400)
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    train = build_split("train", TRAIN_NAMES, args.train_episodes, args.seed)
    test = build_split("test", TEST_NAMES, args.test_episodes, args.seed + 1)
    train_path = output_dir / "train.jsonl"
    test_path = output_dir / "test.jsonl"
    write_jsonl(train_path, train)
    write_jsonl(test_path, test)
    manifest = {
        "dataset": "two-entity three-commit versioned locations",
        "seed": args.seed,
        "cities": CITIES,
        "train_names": TRAIN_NAMES,
        "test_names": TEST_NAMES,
        "train_episodes": args.train_episodes,
        "test_episodes": args.test_episodes,
        "train_examples": len(train),
        "test_examples": len(test),
        "train_sha256": sha256(train_path),
        "test_sha256": sha256(test_path),
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
