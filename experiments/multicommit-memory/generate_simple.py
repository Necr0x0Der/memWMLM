#!/usr/bin/env python3
"""Generate a simpler single-entity, three-commit versioned-location dataset.

This is a diagnostic rung: one person, one location fact per commit, and one
current-location question.  It isolates whether the memory sidecar can read a
fact across commits at all before adding multiple people and question types.
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
        name = rng.choice(names)
        c1, c2, c3 = rng.sample(CITIES, 3)
        docs = [
            f"{name} lives in {c1}.",
            f"{name} moved from {c1} to {c2}.",
            f"{name} moved from {c2} to {c3}.",
        ]
        questions = [
            ("current", f"Where is {name} now?", c3),
            ("original", f"Where did {name} live originally?", c1),
            ("previous", f"Where was {name} before the last move?", c2),
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
                        "entity": name,
                        "history": [c1, c2, c3],
                        "cities_used": [c1, c2, c3],
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
        "dataset": "simple single-entity multi-commit locations",
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
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
