#!/usr/bin/env python3
"""Generate synthetic multi-commit versioned-fact episodes.

Each episode has three commits.  Two people move twice; one person never moves.
Questions ask either a current location (latest fact) or a previous location
(before the last move).  Source documents are separate commits, so a model that
only reads the last commit cannot answer facts committed earlier.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

CITIES = [
    "Paris", "Rome", "Berlin", "Madrid", "London", "Lisbon",
    "Vienna", "Prague", "Oslo", "Cairo", "Tokyo", "Sydney",
]

TRAIN_NAMES = ["Alice", "Bob", "Carol", "David", "Erin", "Frank"]
TEST_NAMES = ["Grace", "Henry", "Iris", "Jack", "Kate", "Leo"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def make_episode(split: str, names: list[str], rng: random.Random, index: int) -> list[dict]:
    a, b, c = rng.sample(names, 3)
    pool = rng.sample(CITIES, 7)
    a1, a2, a3, b1, c1, c2, c3 = pool

    doc1 = [
        f"{a} lives in {a1}.",
        f"{b} lives in {b1}.",
        f"{c} lives in {c1}.",
    ]
    doc2 = [
        f"{a} moved from {a1} to {a2}.",
        f"{c} moved from {c1} to {c2}.",
    ]
    doc3 = [
        f"{a} moved from {a2} to {a3}.",
        f"{c} moved from {c2} to {c3}.",
    ]

    questions = [
        ("current", f"Where is {a} now?", a3),
        ("current", f"Where is {b} now?", b1),
        ("current", f"Where is {c} now?", c3),
        ("previous", f"Where was {a} before the last move?", a2),
        ("previous", f"Where was {c} before the last move?", c2),
    ]

    records = []
    for q_index, (q_type, question, target) in enumerate(questions):
        records.append(
            {
                "example_id": f"{split}_{index:05d}_{q_index}",
                "split": split,
                "episode_id": f"{split}_{index:05d}",
                "question_type": q_type,
                "question": question,
                "target": target,
                "source_documents": ["\n".join(doc1), "\n".join(doc2), "\n".join(doc3)],
                "metadata": {
                    "entities": {"a": a, "b": b, "c": c},
                    "history": {"a": [a1, a2, a3], "b": [b1], "c": [c1, c2, c3]},
                    "cities_used": [a1, a2, a3, b1, c1, c2, c3],
                },
            }
        )
    return records


def build_split(split: str, names: list[str], episodes: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    records: list[dict] = []
    for index in range(episodes):
        records.extend(make_episode(split, names, rng, index))
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
    parser.add_argument("--train-episodes", type=int, default=1200)
    parser.add_argument("--test-episodes", type=int, default=300)
    args = parser.parse_args()

    if args.train_episodes <= 0 or args.test_episodes <= 0:
        raise ValueError("episode counts must be positive")

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    train = build_split("train", TRAIN_NAMES, args.train_episodes, args.seed)
    test = build_split("test", TEST_NAMES, args.test_episodes, args.seed + 1)

    train_path = output_dir / "train.jsonl"
    test_path = output_dir / "test.jsonl"
    write_jsonl(train_path, train)
    write_jsonl(test_path, test)

    manifest = {
        "dataset": "synthetic multi-commit versioned facts",
        "seed": args.seed,
        "cities": CITIES,
        "train_names": TRAIN_NAMES,
        "test_names": TEST_NAMES,
        "train_episodes": args.train_episodes,
        "test_episodes": args.test_episodes,
        "train_questions": len(train),
        "test_questions": len(test),
        "train_path": str(train_path),
        "test_path": str(test_path),
        "train_sha256": sha256(train_path),
        "test_sha256": sha256(test_path),
        "question_types": {
            "current": 3,
            "previous": 2,
        },
        "protocol": {
            "commits_per_episode": 3,
            "source_documents_are_separate_commits": True,
            "question_types": ["current", "previous"],
            "entity_split": "train names and test names are disjoint",
        },
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
