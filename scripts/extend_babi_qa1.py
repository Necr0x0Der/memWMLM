#!/usr/bin/env python3
"""Append independently sampled, bAbI Task 1-shaped episodes to the original train split.

This is a format/distribution-matched *approximation*, not the unpublished v1.2
original generator. Preserve the published test split and store a provenance
manifest alongside the ignored local artifacts.
"""

from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import random

ACTORS = ("Mary", "John", "Daniel", "Sandra")
LOCATIONS = ("bathroom", "hallway", "garden", "office", "bedroom", "kitchen")
MOVEMENTS = ("moved to", "went to", "journeyed to", "travelled to", "went back to")
QUESTIONS_PER_EPISODE = 5
ACTIONS_PER_QUESTION = 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-file", type=Path, required=True)
    parser.add_argument("--test-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--additional-examples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20261020)
    return parser.parse_args()


def split_episodes(lines: list[str]) -> list[tuple[str, ...]]:
    episodes: list[tuple[str, ...]] = []
    current: list[str] = []
    for line in lines:
        if not line.strip():
            continue
        if line.startswith("1 "):
            if current:
                episodes.append(tuple(current))
            current = []
        if not current and not line.startswith("1 "):
            raise ValueError("Input does not start with a numbered episode")
        expected = len(current) + 1
        if not line.startswith(f"{expected} "):
            raise ValueError(f"Unexpected line number in episode: {line!r}")
        current.append(line)
    if current:
        episodes.append(tuple(current))
    return episodes


def example_count(episodes: list[tuple[str, ...]]) -> int:
    return sum("\t" in line for episode in episodes for line in episode)


def episode_text(episodes: list[tuple[str, ...]]) -> str:
    return "".join("\n".join(episode) + "\n" for episode in episodes)


def file_hash(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def episode_key(episode: tuple[str, ...]) -> str:
    return sha256("\n".join(episode).encode("utf-8")).hexdigest()


def example_keys(episode: tuple[str, ...]) -> set[tuple[tuple[str, ...], str, str]]:
    """Match the parser's source/question/answer contract, ignoring line ids."""
    story: list[str] = []
    examples = set()
    for line in episode:
        _, content = line.split(" ", 1)
        if "\t" in content:
            question, answer, _support = (part.strip() for part in content.split("\t"))
            examples.add((tuple(story), question, answer))
        else:
            story.append(content.strip())
    return examples


def story_keys(episode: tuple[str, ...]) -> set[tuple[str, ...]]:
    """All source stories at query boundaries, even if questions differ."""
    return {story for story, _question, _answer in example_keys(episode)}


def generate_episode(rng: random.Random) -> tuple[str, ...]:
    """Five questions, two new movements before each, support = latest queried movement."""
    lines: list[str] = []
    latest: dict[str, tuple[str, int]] = {}
    for _question_index in range(QUESTIONS_PER_EPISODE):
        for _ in range(ACTIONS_PER_QUESTION):
            actor = rng.choice(ACTORS)
            old_location = latest.get(actor, (None, 0))[0]
            location = rng.choice([place for place in LOCATIONS if place != old_location])
            movement = rng.choice(MOVEMENTS)
            event_id = len(lines) + 1
            lines.append(f"{event_id} {actor} {movement} the {location}.")
            latest[actor] = (location, event_id)
        # Select an actor from the most recent five movement events. Repeated
        # actors have proportionally more chances of being selected. This
        # reproduces the original sample's support-age range/distribution
        # much better than selecting uniformly from the known actors.
        recent_actors = [line.split()[1] for line in lines if "\t" not in line][-5:]
        queried = rng.choice(recent_actors)
        answer, supporting_id = latest[queried]
        lines.append(f"{len(lines) + 1} Where is {queried}? \t{answer}\t{supporting_id}")
    return tuple(lines)


def generate_unique_episodes(
    count: int,
    seed: int,
    forbidden: set[str],
    forbidden_examples: set[tuple[tuple[str, ...], str, str]],
    forbidden_test_stories: set[tuple[str, ...]],
) -> tuple[list[tuple[str, ...]], int, int, int]:
    if count <= 0:
        raise ValueError("count must be positive")
    rng = random.Random(seed)
    episodes: list[tuple[str, ...]] = []
    seen = set(forbidden)
    collisions = 0
    example_collisions = 0
    test_story_collisions = 0
    seen_examples = set(forbidden_examples)
    while len(episodes) < count:
        episode = generate_episode(rng)
        key = episode_key(episode)
        if key in seen:
            collisions += 1
            if collisions > 100_000:
                raise RuntimeError("Too many episode collisions")
            continue
        proposed_examples = example_keys(episode)
        if proposed_examples & seen_examples:
            example_collisions += 1
            if example_collisions > 100_000:
                raise RuntimeError("Too many example collisions")
            continue
        if story_keys(episode) & forbidden_test_stories:
            test_story_collisions += 1
            if test_story_collisions > 100_000:
                raise RuntimeError("Too many test-story collisions")
            continue
        seen.add(key)
        seen_examples.update(proposed_examples)
        episodes.append(episode)
    return episodes, collisions, example_collisions, test_story_collisions


def statistics(episodes: list[tuple[str, ...]]) -> dict[str, object]:
    actors: Counter[str] = Counter()
    locations: Counter[str] = Counter()
    answer_counts: Counter[str] = Counter()
    query_recency: Counter[str] = Counter()
    for episode in episodes:
        movement_count = 0
        movement_number_by_line: dict[int, int] = {}
        for line in episode:
            line_id_text, content = line.split(" ", 1)
            if "\t" in content:
                _question, answer, supporting = content.split("\t")
                answer_counts[answer] += 1
                query_recency[str(movement_count - movement_number_by_line[int(supporting)] + 1)] += 1
            else:
                tokens = content.split()
                actors[tokens[0]] += 1
                locations[tokens[-1].rstrip(".")] += 1
                movement_count += 1
                movement_number_by_line[int(line_id_text)] = movement_count
    return {
        "episodes": len(episodes),
        "questions": example_count(episodes),
        "movement_actors": dict(sorted(actors.items())),
        "movement_locations": dict(sorted(locations.items())),
        "answers": dict(sorted(answer_counts.items())),
        "question_support_movement_age": dict(sorted(query_recency.items(), key=lambda x: int(x[0]))),
    }


def build_dataset(train_file: Path, test_file: Path, output_dir: Path,
                  additional_examples: int, seed: int) -> dict[str, object]:
    if additional_examples <= 0 or additional_examples % QUESTIONS_PER_EPISODE:
        raise ValueError("additional-examples must be positive and a multiple of five")
    original_train = split_episodes(train_file.read_text(encoding="utf-8").splitlines())
    original_test = split_episodes(test_file.read_text(encoding="utf-8").splitlines())
    if not original_train or not original_test:
        raise ValueError("Original train and test must be nonempty")
    if any(len(episode) != 15 or sum("\t" in line for line in episode) != 5
           for episode in (*original_train, *original_test)):
        raise ValueError("Expected published QA1 episodes of 15 lines/5 questions")
    forbidden = {episode_key(episode) for episode in (*original_train, *original_test)}
    forbidden_examples = set().union(*(example_keys(episode) for episode in (*original_train, *original_test)))
    extra, collisions, example_collisions, test_story_collisions = generate_unique_episodes(
        additional_examples // QUESTIONS_PER_EPISODE,
        seed,
        forbidden,
        forbidden_examples,
        set().union(*(story_keys(episode) for episode in original_test)),
    )
    merged = original_train + extra
    original_count = example_count(original_train)
    extra_path = output_dir / f"qa1_extra_{additional_examples}_train.txt"
    merged_path = output_dir / f"qa1_merged_{original_count + additional_examples}_train.txt"
    manifest_path = output_dir / "manifest.json"
    if any(path.exists() for path in (extra_path, merged_path, manifest_path)):
        raise FileExistsError(f"Output already exists in {output_dir}; choose a new directory")
    output_dir.mkdir(parents=True, exist_ok=True)
    extra_path.write_text(episode_text(extra), encoding="utf-8")
    merged_path.write_text(episode_text(merged), encoding="utf-8")
    if not merged_path.read_bytes().startswith(train_file.read_bytes()):
        raise RuntimeError("Merged dataset did not preserve original train prefix exactly")
    manifest = {
        "format": "babi-qa1-extension-v1",
        "note": "Approximate QA1 generator, not an exact reproduction of the v1.2 simulator; original test untouched.",
        "seed": seed,
        "additional_examples": additional_examples,
        "collisions_rejected": collisions,
        "example_collisions_rejected": example_collisions,
        "test_story_collisions_rejected": test_story_collisions,
        "original_train": {"path": str(train_file), "sha256": file_hash(train_file),
                           "stats": statistics(original_train)},
        "original_test": {"path": str(test_file), "sha256": file_hash(test_file),
                          "stats": statistics(original_test)},
        "generated_train": {"path": str(extra_path), "sha256": file_hash(extra_path),
                            "stats": statistics(extra)},
        "merged_train": {"path": str(merged_path), "sha256": file_hash(merged_path),
                         "stats": statistics(merged)},
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    args = parse_args()
    print(json.dumps(build_dataset(args.train_file, args.test_file, args.output_dir,
                                   args.additional_examples, args.seed), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
