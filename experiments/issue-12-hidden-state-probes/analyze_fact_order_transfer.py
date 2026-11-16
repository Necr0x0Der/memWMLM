#!/usr/bin/env python3
"""Measure within- and cross-order transfer of issue #12 linear probes.

For each source condition (canonical or shuffled), this script selects pooling,
layer, and L2 using only validation rows from that same condition. It then
locks the probe bank and evaluates it on both canonical and shuffled test rows,
producing the four train/test cells needed to distinguish condition-specific
linear decodability from a shared, order-invariant readout.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import tempfile
from pathlib import Path
from typing import Mapping, Sequence

import torch
from torch import Tensor

from train_linear_probes import (
    POOLINGS,
    SPLITS,
    condition_metrics,
    encode_targets,
    fit_probes,
    group_bootstrap_intervals,
    load_feature_split,
    metrics,
    predict,
    read_jsonl,
    save_probe,
    sha256,
    shuffle_targets_by_group,
)


FACT_ORDERS = ("canonical", "shuffled")
RESULT_SCHEMA_VERSION = "spatial-linear-probe-fact-order-transfer-v1"


def condition_indices(
    rows: Sequence[Mapping[str, object]], fact_order: str
) -> Tensor:
    if fact_order not in FACT_ORDERS:
        raise ValueError(f"Unknown fact order {fact_order!r}")
    indices = [
        index for index, row in enumerate(rows) if row.get("fact_order") == fact_order
    ]
    if not indices:
        raise ValueError(f"No {fact_order!r} rows found")
    return torch.tensor(indices, dtype=torch.long)


def subset_rows(
    rows: Sequence[Mapping[str, object]], indices: Tensor
) -> list[Mapping[str, object]]:
    return [rows[int(index)] for index in indices.tolist()]


def subset_targets(targets: Mapping[str, Tensor], indices: Tensor) -> dict[str, Tensor]:
    return {name: labels.index_select(0, indices) for name, labels in targets.items()}


def load_inputs(
    dataset_dir: Path,
    feature_dir: Path,
) -> tuple[
    dict[str, object],
    dict[str, object],
    dict[str, list[dict[str, object]]],
    dict[str, dict[str, Tensor]],
    tuple[int, ...],
]:
    """Load and integrity-check the same inputs used by the shared probe run."""
    dataset_manifest = json.loads(
        (dataset_dir / "manifest.json").read_text(encoding="utf-8")
    )
    feature_manifest = json.loads(
        (feature_dir / "manifest.json").read_text(encoding="utf-8")
    )
    for split in SPLITS:
        dataset_path = dataset_dir / f"{split}.jsonl"
        expected_hash = (
            feature_manifest.get("dataset", {}).get("files", {}).get(dataset_path.name)
        )
        actual_hash = sha256(dataset_path)
        if expected_hash != actual_hash:
            raise ValueError(
                f"Feature manifest does not match {dataset_path}: "
                f"feature={expected_hash!r}, actual={actual_hash}"
            )

    rows = {split: read_jsonl(dataset_dir / f"{split}.jsonl") for split in SPLITS}
    features: dict[str, dict[str, Tensor]] = {}
    layer_indices: tuple[int, ...] | None = None
    for split in SPLITS:
        split_layers, split_features = load_feature_split(
            feature_dir,
            split,
            feature_manifest["splits"][split],
            [str(row["id"]) for row in rows[split]],
        )
        if layer_indices is None:
            layer_indices = split_layers
        elif split_layers != layer_indices:
            raise ValueError("Feature splits have different layer indices")
        features[split] = split_features
    if layer_indices is None:
        raise ValueError("No feature layers available")
    return dataset_manifest, feature_manifest, rows, features, layer_indices


def select_configuration(
    train_features: Mapping[str, Tensor],
    validation_features: Mapping[str, Tensor],
    train_targets: Mapping[str, Tensor],
    validation_targets: Mapping[str, Tensor],
    class_counts: Mapping[str, int],
    layer_indices: Sequence[int],
    l2_values: Sequence[float],
    max_iter: int,
    device: str,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    """Select a source-condition probe without consulting either test condition."""
    selection: list[dict[str, object]] = []
    best_key: tuple[float, float, float] | None = None
    best_config: dict[str, object] | None = None
    for pooling in POOLINGS:
        for layer_offset, layer in enumerate(layer_indices):
            train_x = train_features[pooling][:, layer_offset, :]
            validation_x = validation_features[pooling][:, layer_offset, :]
            for l2 in l2_values:
                fitted = fit_probes(
                    train_x,
                    train_targets,
                    class_counts,
                    l2,
                    max_iter,
                    device,
                )
                validation_predictions = predict(fitted, validation_x, device)
                validation_metrics = metrics(
                    validation_predictions, validation_targets, class_counts
                )
                record = {
                    "pooling": pooling,
                    "layer": layer,
                    "l2": l2,
                    "validation": validation_metrics,
                }
                selection.append(record)
                key = (
                    float(validation_metrics["mean_macro_accuracy"]),
                    float(validation_metrics["joint_exact_state_accuracy"]),
                    l2,
                )
                if best_key is None or key > best_key:
                    best_key = key
                    best_config = {"pooling": pooling, "layer": layer, "l2": l2}
    if best_config is None:
        raise RuntimeError("No probe candidate was fitted")
    return best_config, selection


def run_fact_order_transfer(
    dataset_dir: Path,
    feature_dir: Path,
    output_dir: Path,
    l2_values: Sequence[float],
    max_iter: int,
    device: str,
    shuffle_seed: int,
    bootstrap_samples: int,
) -> Path:
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output directory: {output_dir}")
    if not l2_values or any(value < 0 or not math.isfinite(value) for value in l2_values):
        raise ValueError("--l2-values must contain finite non-negative values")
    if bootstrap_samples <= 0:
        raise ValueError("bootstrap samples must be positive")

    dataset_manifest, feature_manifest, rows, features, layer_indices = load_inputs(
        dataset_dir, feature_dir
    )
    probes = dataset_manifest.get("probes")
    if not isinstance(probes, dict) or not probes:
        raise ValueError("Dataset manifest contains no probe definitions")
    targets = {split: encode_targets(rows[split], probes) for split in SPLITS}
    class_counts = {
        name: len(definition["classes"]) for name, definition in probes.items()
    }
    indices = {
        split: {
            fact_order: condition_indices(rows[split], fact_order)
            for fact_order in FACT_ORDERS
        }
        for split in SPLITS
    }

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}-", dir=output_dir.parent)
    )
    try:
        regimes: dict[str, object] = {}
        for source_offset, train_order in enumerate(FACT_ORDERS):
            train_indices = indices["train"][train_order]
            validation_indices = indices["validation"][train_order]
            source_train_features = {
                pooling: tensor.index_select(0, train_indices)
                for pooling, tensor in features["train"].items()
            }
            source_validation_features = {
                pooling: tensor.index_select(0, validation_indices)
                for pooling, tensor in features["validation"].items()
            }
            source_train_targets = subset_targets(targets["train"], train_indices)
            source_validation_targets = subset_targets(
                targets["validation"], validation_indices
            )
            source_train_rows = subset_rows(rows["train"], train_indices)

            selected_config, selection = select_configuration(
                source_train_features,
                source_validation_features,
                source_train_targets,
                source_validation_targets,
                class_counts,
                layer_indices,
                l2_values,
                max_iter,
                device,
            )
            layer_offset = layer_indices.index(int(selected_config["layer"]))
            pooling = str(selected_config["pooling"])
            selected_train_x = source_train_features[pooling][:, layer_offset, :]
            fitted = fit_probes(
                selected_train_x,
                source_train_targets,
                class_counts,
                float(selected_config["l2"]),
                max_iter,
                device,
            )
            shuffled_train_targets = shuffle_targets_by_group(
                source_train_targets,
                source_train_rows,
                shuffle_seed + source_offset,
            )
            shuffled_control = fit_probes(
                selected_train_x,
                shuffled_train_targets,
                class_counts,
                float(selected_config["l2"]),
                max_iter,
                device,
            )
            save_probe(
                temporary / f"probe-trained-{train_order}.pt",
                fitted,
                {"train_fact_order": train_order, **selected_config},
            )

            evaluations: dict[str, object] = {}
            control_evaluations: dict[str, object] = {}
            for target_offset, test_order in enumerate(FACT_ORDERS):
                test_indices = indices["test"][test_order]
                test_rows = subset_rows(rows["test"], test_indices)
                test_targets = subset_targets(targets["test"], test_indices)
                test_x = features["test"][pooling].index_select(0, test_indices)[
                    :, layer_offset, :
                ]
                predictions = predict(fitted, test_x, device)
                control_predictions = predict(shuffled_control, test_x, device)
                evaluation = condition_metrics(
                    test_rows, predictions, test_targets, class_counts
                )
                evaluation["group_bootstrap_95ci"] = group_bootstrap_intervals(
                    test_rows,
                    predictions,
                    test_targets,
                    class_counts,
                    bootstrap_samples,
                    shuffle_seed + 100 + source_offset * 10 + target_offset,
                )
                control_evaluation = condition_metrics(
                    test_rows, control_predictions, test_targets, class_counts
                )
                control_evaluation["group_bootstrap_95ci"] = group_bootstrap_intervals(
                    test_rows,
                    control_predictions,
                    test_targets,
                    class_counts,
                    bootstrap_samples,
                    shuffle_seed + 200 + source_offset * 10 + target_offset,
                )
                evaluations[test_order] = evaluation
                control_evaluations[test_order] = control_evaluation

            regimes[train_order] = {
                "selected": selected_config,
                "selection": selection,
                "evaluation": evaluations,
                "shuffled_label_control": control_evaluations,
            }

        report = {
            "schema_version": RESULT_SCHEMA_VERSION,
            "protocol": {
                "matrix": [
                    "canonical -> canonical",
                    "canonical -> shuffled",
                    "shuffled -> canonical",
                    "shuffled -> shuffled",
                ],
                "selection": (
                    "pooling/layer/L2 selected independently for each train fact order "
                    "using validation rows of that same order only"
                ),
                "test_access": (
                    "both test fact orders evaluated only after the source-condition "
                    "configuration is locked"
                ),
                "standardization": "source-condition train statistics only",
                "confidence_intervals": (
                    "95% percentile bootstrap over abstract-layout groups within each cell"
                ),
                "negative_control": (
                    "complete target vectors permuted between source-condition train groups"
                ),
            },
            "dataset_manifest_sha256": sha256(dataset_dir / "manifest.json"),
            "feature_manifest_sha256": sha256(feature_dir / "manifest.json"),
            "layer_indices": list(layer_indices),
            "l2_values": list(l2_values),
            "max_iter": max_iter,
            "device": device,
            "shuffle_seed": shuffle_seed,
            "bootstrap_samples": bootstrap_samples,
            "regimes": regimes,
        }
        report_path = temporary / "report.json"
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(output_dir)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return output_dir / "report.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--feature-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--l2-values", default="0,1e-5,1e-4,1e-3")
    parser.add_argument("--max-iter", type=int, default=100)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--shuffle-seed", type=int, default=12012)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = run_fact_order_transfer(
        args.dataset_dir,
        args.feature_dir,
        args.output_dir,
        tuple(float(value.strip()) for value in args.l2_values.split(",")),
        args.max_iter,
        args.device,
        args.shuffle_seed,
        args.bootstrap_samples,
    )
    print(json.dumps({"report": str(report)}))


if __name__ == "__main__":
    main()
