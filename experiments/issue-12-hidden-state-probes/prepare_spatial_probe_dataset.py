#!/usr/bin/env python3
"""Build the grouped spatial-world dataset proposed in GitHub issue #12.

The split unit is an abstract world state (the ranks occupied by Mary, John,
and Bob), not a rendered prompt.  Consequently, room-name permutations,
instruction variants, and fact-order variants of the same state cannot leak
across train, validation, and test.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import random
import shutil
import tempfile
from collections import Counter
from pathlib import Path
from typing import Iterable, Sequence


SCHEMA_VERSION = "spatial-hidden-state-probes-v2"
PEOPLE = ("Mary", "John", "Bob")
ROOMS = ("A", "B", "C", "D")
SPLITS = ("train", "validation", "test")
DEFAULT_RATIOS = (0.70, 0.15, 0.15)
INSTRUCTIONS = {
    "none": "",
    "imagine": "Imagine the situation.",
    "track_people": "Think about who is where.",
    "track_mary": "Think about where Mary is.",
}
FACT_ORDERS = ("canonical", "shuffled")


def stable_seed(seed: int, *parts: object) -> int:
    payload = "\0".join([str(seed), *(str(part) for part in parts)]).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def abstract_layouts() -> list[tuple[int, int, int]]:
    """All placements of three named people in four ordered positions."""
    return list(itertools.permutations(range(len(ROOMS)), len(PEOPLE)))


def group_id(layout: Sequence[int]) -> str:
    return "ranks-" + "-".join(f"{person.lower()}-{rank}" for person, rank in zip(PEOPLE, layout))


def targets_for(layout: Sequence[int]) -> dict[str, str | int]:
    ranks = dict(zip(PEOPLE, layout))
    targets: dict[str, str | int] = {
        "leftmost_person": min(PEOPLE, key=ranks.__getitem__),
    }
    for first, second in itertools.permutations(PEOPLE, 2):
        key = f"{first.lower()}_left_of_{second.lower()}"
        targets[key] = int(ranks[first] < ranks[second])
    return targets


def probe_definitions() -> dict[str, dict[str, object]]:
    probes: dict[str, dict[str, object]] = {
        "leftmost_person": {
            "question": "Who is leftmost?",
            "classes": list(PEOPLE),
        }
    }
    for first, second in itertools.permutations(PEOPLE, 2):
        probes[f"{first.lower()}_left_of_{second.lower()}"] = {
            "question": f"Is {first} left of {second}?",
            "classes": [0, 1],
        }
    return probes


def qwen_questions(targets: dict[str, str | int]) -> list[dict[str, object]]:
    """Return self-contained QA metadata for sending a record to Qwen."""
    questions: list[dict[str, object]] = []
    for probe_id, definition in probe_definitions().items():
        target = targets[probe_id]
        classes = definition["classes"]
        if classes == [0, 1]:
            allowed_answers = ["No", "Yes"]
            answer = allowed_answers[int(target)]
        else:
            allowed_answers = list(classes)
            answer = str(target)
        questions.append(
            {
                "id": probe_id,
                "question": definition["question"],
                "allowed_answers": allowed_answers,
                "answer": answer,
            }
        )
    return questions


def split_counts(total: int, ratios: Sequence[float]) -> tuple[int, int, int]:
    if len(ratios) != len(SPLITS) or any(ratio <= 0 for ratio in ratios):
        raise ValueError("Expected three positive split ratios")
    ratio_sum = sum(ratios)
    normalized = [ratio / ratio_sum for ratio in ratios]
    raw = [total * ratio for ratio in normalized]
    counts = [math.floor(value) for value in raw]
    # Prefer test over validation when their fractional remainders are tied.
    tie_priority = {"test": 0, "validation": 1, "train": 2}
    ## where to add reminder
    order = sorted(
        range(len(SPLITS)),
        key=lambda index: (-(raw[index] - counts[index]), tie_priority[SPLITS[index]]),
    )
    for index in order[: total - sum(counts)]:
        counts[index] += 1
    if any(count == 0 for count in counts):
        raise ValueError(f"Too few groups ({total}) for three non-empty splits")
    return tuple(counts)  # type: ignore[return-value]


def _balance_score(assignments: dict[str, list[tuple[int, int, int]]]) -> float:
    """Score label balance; lower is better and missing classes are forbidden."""
    probes = probe_definitions()
    global_targets = [targets_for(layout) for layouts in assignments.values() for layout in layouts]
    score = 0.0
    for probe_name, definition in probes.items():
        classes = definition["classes"]
        global_counts = Counter(target[probe_name] for target in global_targets)
        global_total = len(global_targets)
        for split, layouts in assignments.items():
            counts = Counter(targets_for(layout)[probe_name] for layout in layouts)
            if any(counts[value] == 0 for value in classes):
                return math.inf
            for value in classes:
                observed = counts[value] / len(layouts)
                expected = global_counts[value] / global_total
                score += (observed - expected) ** 2
    return score


def assign_layouts(
    layouts: Sequence[tuple[int, int, int]],
    ratios: Sequence[float],
    seed: int,
    attempts: int = 5_000,
) -> dict[str, str]:
    """Find a deterministic, label-balanced grouped split."""
    counts = split_counts(len(layouts), ratios)
    rng = random.Random(stable_seed(seed, "group-split"))
    best: tuple[float, tuple[str, ...], dict[str, list[tuple[int, int, int]]]] | None = None
    for _ in range(attempts):
        candidate = list(layouts)
        rng.shuffle(candidate)
        assignments: dict[str, list[tuple[int, int, int]]] = {}
        offset = 0
        for split, count in zip(SPLITS, counts):
            assignments[split] = candidate[offset : offset + count]
            offset += count
        score = _balance_score(assignments)
        signature = tuple(group_id(layout) for split in SPLITS for layout in assignments[split])
        comparable = (score, signature, assignments)
        if best is None or comparable[:2] < best[:2]:
            best = comparable
            if score == 0:
                break
    if best is None or not math.isfinite(best[0]):
        raise RuntimeError("Could not construct a split with every probe class represented")
    return {
        group_id(layout): split
        for split, split_layouts in best[2].items()
        for layout in split_layouts
    }


def facts_for(layout: Sequence[int], room_order: Sequence[str]) -> list[str]:
    facts = [
        f"Room {left} is left of room {right}."
        for left, right in zip(room_order, room_order[1:])
    ]
    facts.extend(
        f"{person} is in room {room_order[rank]}."
        for person, rank in zip(PEOPLE, layout)
    )
    return facts


def shuffled_facts(facts: Sequence[str], seed: int) -> list[str]:
    shuffled = list(facts)
    random.Random(seed).shuffle(shuffled)
    if shuffled == list(facts):
        shuffled = shuffled[1:] + shuffled[:1]
    return shuffled


def render_text(instruction: str, facts: Sequence[str]) -> str:
    lines = [instruction] if instruction else []
    lines.extend(facts)
    return "\n".join(lines)


def generate_records(
    seed: int,
    episodes_per_group: int,
    ratios: Sequence[float] = DEFAULT_RATIOS,
) -> list[dict[str, object]]:
    if episodes_per_group <= 0:
        raise ValueError("episodes_per_group must be positive")
    layouts = abstract_layouts()
    split_by_group = assign_layouts(layouts, ratios, seed)
    room_orders = list(itertools.permutations(ROOMS))
    records: list[dict[str, object]] = []

    for layout in layouts:
        current_group_id = group_id(layout)
        split = split_by_group[current_group_id]
        ordered_rooms = list(room_orders)
        random.Random(stable_seed(seed, current_group_id, "room-orders")).shuffle(ordered_rooms)
        for episode_index in range(episodes_per_group):
            rooms = ordered_rooms[episode_index % len(ordered_rooms)]
            episode_id = f"{current_group_id}:episode-{episode_index:03d}"
            canonical_facts = facts_for(layout, rooms)
            episode_targets = targets_for(layout)
            episode_questions = qwen_questions(episode_targets)
            order_variants = {
                "canonical": canonical_facts,
                "shuffled": shuffled_facts(
                    canonical_facts,
                    stable_seed(seed, episode_id, "fact-order"),
                ),
            }
            person_room = {
                person: rooms[rank]
                for person, rank in zip(PEOPLE, layout)
            }
            for instruction_id, instruction in INSTRUCTIONS.items():
                for fact_order, facts in order_variants.items():
                    record_id = f"{episode_id}:{instruction_id}:{fact_order}"
                    records.append(
                        {
                            "id": record_id,
                            "split": split,
                            "group_id": current_group_id,
                            "episode_id": episode_id,
                            "instruction_id": instruction_id,
                            "fact_order": fact_order,
                            # Keep the exact facts separate from the optional instruction.
                            # The paired fields make the canonical/shuffled relationship
                            # explicit instead of requiring callers to join adjacent rows.
                            "facts": list(facts),
                            "canonical_facts": list(canonical_facts),
                            "shuffled_facts": list(order_variants["shuffled"]),
                            "text": render_text(instruction, facts),
                            "state": {
                                "left_to_right_rooms": list(rooms),
                                "person_room": person_room,
                                "person_rank": dict(zip(PEOPLE, layout)),
                            },
                            "targets": episode_targets,
                            "qwen_qa": episode_questions,
                        }
                    )
    return records


def validate_records(records: Sequence[dict[str, object]]) -> None:
    if not records:
        raise ValueError("Dataset is empty")
    split_by_group: dict[str, str] = {}
    seen_ids: set[str] = set()
    variants_by_condition: dict[tuple[str, str], dict[str, dict[str, object]]] = {}
    for record in records:
        record_id = str(record["id"])
        split = str(record["split"])
        current_group_id = str(record["group_id"])
        if record_id in seen_ids:
            raise ValueError(f"Duplicate record id: {record_id}")
        seen_ids.add(record_id)
        previous = split_by_group.setdefault(current_group_id, split)
        if previous != split:
            raise ValueError(f"Group leakage: {current_group_id} is in {previous} and {split}")
        facts = list(record["facts"])  # type: ignore[arg-type]
        canonical = list(record["canonical_facts"])  # type: ignore[arg-type]
        shuffled = list(record["shuffled_facts"])  # type: ignore[arg-type]
        if Counter(canonical) != Counter(shuffled) or canonical == shuffled:
            raise ValueError(f"Invalid canonical/shuffled fact pair: {record_id}")
        fact_order = str(record["fact_order"])
        if fact_order not in FACT_ORDERS:
            raise ValueError(f"Unknown fact_order in {record_id}: {fact_order}")
        expected_facts = canonical if fact_order == "canonical" else shuffled
        if facts != expected_facts:
            raise ValueError(f"facts do not match fact_order in {record_id}")
        instruction = INSTRUCTIONS[str(record["instruction_id"])]
        if record["text"] != render_text(instruction, facts):
            raise ValueError(f"text does not render facts in {record_id}")
        qa = list(record["qwen_qa"])  # type: ignore[arg-type]
        if qa != qwen_questions(record["targets"]):  # type: ignore[arg-type]
            raise ValueError(f"Qwen QA metadata does not match targets in {record_id}")
        condition = (str(record["episode_id"]), str(record["instruction_id"]))
        variants = variants_by_condition.setdefault(condition, {})
        if fact_order in variants:
            raise ValueError(f"Duplicate {fact_order} variant for {condition}")
        variants[fact_order] = record
    if set(split_by_group.values()) != set(SPLITS):
        raise ValueError("All three splits must be present")
    for condition, variants in variants_by_condition.items():
        if set(variants) != set(FACT_ORDERS):
            raise ValueError(f"Missing canonical/shuffled pair for {condition}")
        canonical_record = variants["canonical"]
        shuffled_record = variants["shuffled"]
        for field in ("canonical_facts", "shuffled_facts", "targets", "qwen_qa"):
            if canonical_record[field] != shuffled_record[field]:
                raise ValueError(f"Mismatched {field} in pair for {condition}")


def write_jsonl(path: Path, records: Iterable[dict[str, object]]) -> int:
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    return count


def build_dataset(
    output_dir: Path,
    seed: int = 12,
    episodes_per_group: int = 24,
    ratios: Sequence[float] = DEFAULT_RATIOS,
) -> Path:
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output directory: {output_dir}")
    records = generate_records(seed, episodes_per_group, ratios)
    validate_records(records)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}-", dir=output_dir.parent))
    try:
        file_stats: dict[str, dict[str, object]] = {}
        group_stats: dict[str, int] = {}
        for split in SPLITS:
            split_records = [record for record in records if record["split"] == split]
            path = temporary / f"{split}.jsonl"
            count = write_jsonl(path, split_records)
            groups = {str(record["group_id"]) for record in split_records}
            group_stats[split] = len(groups)
            file_stats[path.name] = {"examples": count, "sha256": sha256(path)}

        total_groups = sum(group_stats.values())
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "issue": "https://github.com/Necr0x0Der/memWMLM/issues/12",
            "seed": seed,
            "episodes_per_group": episodes_per_group,
            "conditions_per_episode": len(INSTRUCTIONS) * len(FACT_ORDERS),
            "people": list(PEOPLE),
            "rooms": list(ROOMS),
            "instructions": INSTRUCTIONS,
            "fact_orders": list(FACT_ORDERS),
            "record_fields": {
                "facts": "facts in the order used by this row's text",
                "canonical_facts": "canonical facts for the matched episode",
                "shuffled_facts": "a non-identity permutation of canonical_facts",
                "qwen_qa": (
                    "questions, allowed short answers, and expected answers; combine an "
                    "item's question with text (or facts) to query Qwen"
                ),
            },
            "probes": probe_definitions(),
            "split": {
                "unit": "abstract_person_rank_layout",
                "policy": (
                    "All room-name, instruction, and fact-order variants of one abstract "
                    "person layout stay in the same split. Groups are selected by a seeded "
                    "search that balances every probe label."
                ),
                "requested_ratios": dict(zip(SPLITS, ratios)),
                "groups": group_stats,
                "actual_group_ratios": {
                    split: group_stats[split] / total_groups for split in SPLITS
                },
            },
            "files": file_stats,
        }
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(output_dir)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return output_dir / "manifest.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path,  default="artifacts/test.db")
    parser.add_argument("--seed", type=int, default=12)
    parser.add_argument("--episodes-per-group", type=int, default=24)
    parser.add_argument("--train-ratio", type=float, default=DEFAULT_RATIOS[0])
    parser.add_argument("--validation-ratio", type=float, default=DEFAULT_RATIOS[1])
    parser.add_argument("--test-ratio", type=float, default=DEFAULT_RATIOS[2])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ratios = (args.train_ratio, args.validation_ratio, args.test_ratio)
    manifest_path = build_dataset(
        args.output_dir,
        seed=args.seed,
        episodes_per_group=args.episodes_per_group,
        ratios=ratios,
    )
    print(manifest_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
