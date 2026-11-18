from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path


EXPERIMENT_DIR = Path(__file__).resolve().parents[1] / "experiments" / "issue-12-hidden-state-probes"
sys.path.insert(0, str(EXPERIMENT_DIR))

from prepare_spatial_probe_dataset import (  # noqa: E402
    INSTRUCTIONS,
    SPLITS,
    abstract_layouts,
    assign_layouts,
    build_dataset,
    generate_records,
    split_counts,
    targets_for,
)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_targets_are_derived_from_person_ranks() -> None:
    targets = targets_for((0, 3, 2))
    assert targets["leftmost_person"] == "Mary"
    assert targets["mary_left_of_bob"] == 1
    assert targets["bob_left_of_john"] == 1
    assert targets["john_left_of_mary"] == 0


def test_grouped_split_keeps_all_near_duplicate_variants_together() -> None:
    records = generate_records(seed=12, episodes_per_group=2)
    splits_by_group: dict[str, set[str]] = defaultdict(set)
    variants_by_episode: dict[str, set[tuple[str, str]]] = defaultdict(set)
    for record in records:
        splits_by_group[record["group_id"]].add(record["split"])
        variants_by_episode[record["episode_id"]].add(
            (record["instruction_id"], record["fact_order"])
        )

    assert len(splits_by_group) == 24
    assert all(len(splits) == 1 for splits in splits_by_group.values())
    assert {next(iter(splits)) for splits in splits_by_group.values()} == set(SPLITS)
    assert all(len(variants) == len(INSTRUCTIONS) * 2 for variants in variants_by_episode.values())


def test_split_is_deterministic_balanced_and_close_to_requested_ratio() -> None:
    records = generate_records(seed=98, episodes_per_group=1)
    layouts_by_split = defaultdict(set)
    labels_by_split = defaultdict(lambda: defaultdict(set))
    for record in records:
        split = record["split"]
        layouts_by_split[split].add(record["group_id"])
        for target, value in record["targets"].items():
            labels_by_split[split][target].add(value)

    assert split_counts(24, (0.70, 0.15, 0.15)) == (17, 3, 4)
    assert {split: len(groups) for split, groups in layouts_by_split.items()} == {
        "train": 17,
        "validation": 3,
        "test": 4,
    }
    assert all(
        len(labels_by_split[split][target]) > 1
        for split in SPLITS
        for target in labels_by_split[split]
    )
    layouts = abstract_layouts()
    assert assign_layouts(layouts, (0.70, 0.15, 0.15), 98) == assign_layouts(
        layouts, (0.70, 0.15, 0.15), 98
    )


def test_build_dataset_writes_manifest_hashes_and_refuses_overwrite(tmp_path: Path) -> None:
    output = tmp_path / "dataset"
    manifest_path = build_dataset(output, seed=7, episodes_per_group=1)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["split"]["unit"] == "abstract_person_rank_layout"
    assert sum(item["examples"] for item in manifest["files"].values()) == 24 * 8
    for split in SPLITS:
        rows = read_jsonl(output / f"{split}.jsonl")
        assert rows
        assert all(row["split"] == split for row in rows)

    try:
        build_dataset(output, seed=7, episodes_per_group=1)
    except FileExistsError:
        pass
    else:
        raise AssertionError("Expected overwrite protection")
