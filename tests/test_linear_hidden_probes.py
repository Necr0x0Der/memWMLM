from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import torch


EXPERIMENT_DIR = Path(__file__).resolve().parents[1] / "experiments" / "issue-12-hidden-state-probes"
sys.path.insert(0, str(EXPERIMENT_DIR))

from train_linear_probes import (  # noqa: E402
    fit_probes,
    group_bootstrap_intervals,
    load_feature_split,
    metrics,
    predict,
    run_experiment,
    shuffle_targets_by_group,
)


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_probe_bank_learns_multiple_fixed_questions() -> None:
    features = torch.tensor(
        [
            [-3.0, -2.0],
            [-2.0, -3.0],
            [-2.0, 3.0],
            [-3.0, 2.0],
            [2.0, -3.0],
            [3.0, -2.0],
            [2.0, 3.0],
            [3.0, 2.0],
        ]
    )
    targets = {
        "x_positive": (features[:, 0] > 0).to(torch.long),
        "y_positive": (features[:, 1] > 0).to(torch.long),
        "quadrant": torch.tensor([0, 0, 1, 1, 2, 2, 3, 3]),
    }
    class_counts = {"x_positive": 2, "y_positive": 2, "quadrant": 4}
    fitted = fit_probes(
        features,
        targets,
        class_counts,
        l2=1e-4,
        max_iter=60,
        device="cpu",
    )
    predictions = predict(fitted, features, device="cpu")
    report = metrics(predictions, targets, class_counts)

    assert report["mean_macro_accuracy"] == 1.0
    assert report["joint_exact_state_accuracy"] == 1.0


def test_group_shuffle_keeps_matched_variants_together() -> None:
    rows = [
        {"id": "a-1", "group_id": "a"},
        {"id": "a-2", "group_id": "a"},
        {"id": "b-1", "group_id": "b"},
        {"id": "b-2", "group_id": "b"},
        {"id": "c-1", "group_id": "c"},
        {"id": "c-2", "group_id": "c"},
    ]
    targets = {
        "first": torch.tensor([0, 0, 1, 1, 2, 2]),
        "second": torch.tensor([1, 1, 0, 0, 2, 2]),
    }
    shuffled = shuffle_targets_by_group(targets, rows, seed=12)

    state_pairs = []
    for start in (0, 2, 4):
        assert shuffled["first"][start] == shuffled["first"][start + 1]
        assert shuffled["second"][start] == shuffled["second"][start + 1]
        state_pairs.append((int(shuffled["first"][start]), int(shuffled["second"][start])))
    assert sorted(state_pairs) == [(0, 1), (1, 0), (2, 2)]


def test_feature_loader_joins_shards_by_id(tmp_path: Path) -> None:
    feature_dir = tmp_path / "features"
    split_dir = feature_dir / "train"
    split_dir.mkdir(parents=True)
    shard_path = split_dir / "shard-00000.pt"
    torch.save(
        {
            "ids": ["second", "first"],
            "layer_indices": torch.tensor([4, 8], dtype=torch.int16),
            "last_token": torch.tensor([[[2.0], [20.0]], [[1.0], [10.0]]]),
            "mean_pool": torch.tensor([[[3.0], [30.0]], [[1.5], [15.0]]]),
        },
        shard_path,
    )
    manifest = {
        "shards": [
            {"path": shard_path.name, "examples": 2, "sha256": file_hash(shard_path)}
        ]
    }

    layers, features = load_feature_split(
        feature_dir, "train", manifest, ["first", "second"]
    )

    assert layers == (4, 8)
    assert features["last_token"][:, 0, 0].tolist() == [1.0, 2.0]
    assert features["mean_pool"][:, 1, 0].tolist() == [15.0, 30.0]


def test_group_bootstrap_treats_matched_rows_as_one_unit() -> None:
    rows = [
        {"id": "a-1", "group_id": "a"},
        {"id": "a-2", "group_id": "a"},
        {"id": "b-1", "group_id": "b"},
        {"id": "b-2", "group_id": "b"},
    ]
    targets = {"probe": torch.tensor([0, 0, 1, 1])}
    predictions = {"probe": torch.tensor([0, 0, 0, 0])}

    intervals = group_bootstrap_intervals(
        rows, predictions, targets, {"probe": 2}, samples=100, seed=7
    )

    assert intervals["joint_exact_state_accuracy"] == [0.0, 1.0]
    assert all(0.0 <= value <= 1.0 for value in intervals["mean_macro_accuracy"])


def test_end_to_end_selection_writes_locked_report(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    feature_dir = tmp_path / "features"
    dataset_dir.mkdir()
    dataset_hashes = {}
    split_rows = {}
    for split in ("train", "validation", "test"):
        rows = []
        feature_values = []
        for group_index, sign in enumerate((-1.0, 1.0)):
            for instruction in ("none", "track"):
                for fact_order in ("canonical", "shuffled"):
                    row_id = f"{split}-{group_index}-{instruction}-{fact_order}"
                    rows.append(
                        {
                            "id": row_id,
                            "group_id": f"{split}-group-{group_index}",
                            "instruction_id": instruction,
                            "fact_order": fact_order,
                            "targets": {"positive": int(sign > 0)},
                        }
                    )
                    feature_values.append([[sign, sign * 0.5]])
        split_rows[split] = rows
        dataset_path = dataset_dir / f"{split}.jsonl"
        dataset_path.write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )
        dataset_hashes[dataset_path.name] = file_hash(dataset_path)
        split_feature_dir = feature_dir / split
        split_feature_dir.mkdir(parents=True)
        shard_path = split_feature_dir / "shard-00000.pt"
        tensor = torch.tensor(feature_values, dtype=torch.float32)
        torch.save(
            {
                "ids": [row["id"] for row in reversed(rows)],
                "layer_indices": torch.tensor([4], dtype=torch.int16),
                "last_token": tensor.flip(0),
                "mean_pool": tensor.flip(0) * 2,
            },
            shard_path,
        )
    (dataset_dir / "manifest.json").write_text(
        json.dumps({"probes": {"positive": {"classes": [0, 1]}}}), encoding="utf-8"
    )
    feature_manifest = {
        "dataset": {"files": dataset_hashes},
        "splits": {
            split: {
                "shards": [
                    {
                        "path": "shard-00000.pt",
                        "examples": len(split_rows[split]),
                        "sha256": file_hash(feature_dir / split / "shard-00000.pt"),
                    }
                ]
            }
            for split in ("train", "validation", "test")
        },
    }
    (feature_dir / "manifest.json").write_text(json.dumps(feature_manifest), encoding="utf-8")

    report_path = run_experiment(
        dataset_dir,
        feature_dir,
        tmp_path / "output",
        l2_values=(1e-4,),
        max_iter=30,
        device="cpu",
        shuffle_seed=12,
        bootstrap_samples=20,
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))

    assert report["selected"]["layer"] == 4
    assert report["evaluation"]["test"]["overall"]["mean_macro_accuracy"] == 1.0
    assert "group_bootstrap_95ci" in report["evaluation"]["test"]
    assert (report_path.parent / "selected_probe.pt").is_file()
