#!/usr/bin/env python3
"""Measure canonical/shuffled transfer of shallow issue #12 MLP probes."""

from __future__ import annotations

import argparse
import json
import shutil
import statistics
import tempfile
from pathlib import Path
from typing import Mapping, Sequence

from torch import Tensor

from analyze_fact_order_transfer import (
    FACT_ORDERS,
    condition_indices,
    subset_rows,
    subset_targets,
)
from mlp_probe_utils import (
    ensemble_logits,
    predict_logits,
    predictions_from_logits,
    save_ensemble,
)
from train_linear_probes import (
    DEFAULT_POOLINGS,
    SPLITS,
    condition_metrics,
    encode_targets,
    group_bootstrap_intervals,
    load_feature_split,
    metrics,
    read_jsonl,
    sha256,
    shuffle_targets_by_group,
)
from train_mlp_probes import fit_configuration, select_configuration


RESULT_SCHEMA_VERSION = "spatial-mlp-fact-order-transfer-v1"
MATCHED_LAYERS = {"canonical": 8, "shuffled": 28}


def evaluate_orders(
    fitted: Sequence[object],
    config: Mapping[str, object],
    test_features: Mapping[str, Tensor],
    test_rows: Sequence[Mapping[str, object]],
    test_targets: Mapping[str, Tensor],
    class_counts: Mapping[str, int],
    layer_indices: Sequence[int],
    device: str,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, object]:
    pooling = str(config["pooling"])
    layer_offset = layer_indices.index(int(config["layer"]))
    result: dict[str, object] = {}
    for offset, order in enumerate(FACT_ORDERS):
        indices = condition_indices(test_rows, order)
        rows = subset_rows(test_rows, indices)
        targets = subset_targets(test_targets, indices)
        x = test_features[pooling].index_select(0, indices)[:, layer_offset, :]
        predictions = predictions_from_logits(ensemble_logits(fitted, x, device))
        evaluation = condition_metrics(rows, predictions, targets, class_counts)
        evaluation["group_bootstrap_95ci"] = group_bootstrap_intervals(
            rows,
            predictions,
            targets,
            class_counts,
            bootstrap_samples,
            bootstrap_seed + offset,
        )
        seed_runs = []
        for model in fitted:
            seed_overall = metrics(
                predictions_from_logits(predict_logits(model, x, device)),
                targets,
                class_counts,
            )
            seed_runs.append(
                {
                    "seed": model.seed,
                    "mean_macro_accuracy": seed_overall["mean_macro_accuracy"],
                    "joint_exact_state_accuracy": seed_overall[
                        "joint_exact_state_accuracy"
                    ],
                }
            )
        evaluation["seed_runs"] = seed_runs
        evaluation["seed_summary"] = {
            metric: {
                "mean": statistics.fmean(float(run[metric]) for run in seed_runs),
                "population_std": statistics.pstdev(
                    float(run[metric]) for run in seed_runs
                ),
            }
            for metric in ("mean_macro_accuracy", "joint_exact_state_accuracy")
        }
        result[order] = evaluation
    return result


def run_mlp_fact_order_transfer(
    dataset_dir: Path,
    feature_dir: Path,
    output_dir: Path,
    poolings: Sequence[str] = DEFAULT_POOLINGS,
    hidden_sizes: Sequence[int] = (128,),
    weight_decays: Sequence[float] = (1e-4, 1e-3),
    seeds: Sequence[int] = (12012, 12013, 12014, 12015, 12016),
    dropout: float = 0.1,
    learning_rate: float = 1e-3,
    batch_size: int = 128,
    max_epochs: int = 200,
    patience: int = 20,
    device: str = "cpu",
    bootstrap_samples: int = 2000,
    bootstrap_seed: int = 12012,
) -> Path:
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output directory: {output_dir}")
    dataset_manifest = json.loads((dataset_dir / "manifest.json").read_text())
    feature_manifest = json.loads((feature_dir / "manifest.json").read_text())
    probes = dataset_manifest.get("probes")
    if not isinstance(probes, dict) or not probes:
        raise ValueError("Dataset manifest contains no probe definitions")
    rows = {split: read_jsonl(dataset_dir / f"{split}.jsonl") for split in SPLITS}
    features: dict[str, dict[str, Tensor]] = {}
    layer_indices: tuple[int, ...] | None = None
    for split in SPLITS:
        dataset_path = dataset_dir / f"{split}.jsonl"
        if feature_manifest.get("dataset", {}).get("files", {}).get(dataset_path.name) != sha256(dataset_path):
            raise ValueError(f"Feature manifest does not match {dataset_path}")
        split_layers, split_features = load_feature_split(
            feature_dir,
            split,
            feature_manifest["splits"][split],
            [str(row["id"]) for row in rows[split]],
            poolings,
        )
        if layer_indices is None:
            layer_indices = split_layers
        elif layer_indices != split_layers:
            raise ValueError("Feature splits have different layers")
        features[split] = split_features
    if layer_indices is None:
        raise ValueError("No feature layers available")
    targets = {split: encode_targets(rows[split], probes) for split in SPLITS}
    class_counts = {name: len(definition["classes"]) for name, definition in probes.items()}
    training: dict[str, object] = {
        "dropout": dropout,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "max_epochs": max_epochs,
        "patience": patience,
        "device": device,
    }
    representations = [(pooling, layer) for pooling in poolings for layer in layer_indices]
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}-", dir=output_dir.parent))
    try:
        regimes: dict[str, object] = {}
        for source_offset, train_order in enumerate(FACT_ORDERS):
            train_indices = condition_indices(rows["train"], train_order)
            validation_indices = condition_indices(rows["validation"], train_order)
            source_features = {
                "train": {
                    name: tensor.index_select(0, train_indices)
                    for name, tensor in features["train"].items()
                },
                "validation": {
                    name: tensor.index_select(0, validation_indices)
                    for name, tensor in features["validation"].items()
                },
                "test": features["test"],
            }
            source_targets = {
                "train": subset_targets(targets["train"], train_indices),
                "validation": subset_targets(targets["validation"], validation_indices),
                "test": targets["test"],
            }
            source_rows = subset_rows(rows["train"], train_indices)
            selected, selection = select_configuration(
                representations,
                hidden_sizes,
                weight_decays,
                source_features,
                source_targets,
                class_counts,
                layer_indices,
                seeds,
                training,
            )
            matched, matched_selection = select_configuration(
                [("mean_pool", MATCHED_LAYERS[train_order])],
                hidden_sizes,
                weight_decays,
                source_features,
                source_targets,
                class_counts,
                layer_indices,
                seeds,
                training,
            )
            selected_models, _ = fit_configuration(
                selected, source_features, source_targets, class_counts,
                layer_indices, seeds, training
            )
            matched_models, _ = fit_configuration(
                matched, source_features, source_targets, class_counts,
                layer_indices, seeds, training
            )
            shuffled_source_targets = dict(source_targets)
            shuffled_source_targets["train"] = shuffle_targets_by_group(
                source_targets["train"], source_rows, bootstrap_seed + source_offset
            )
            control_models, _ = fit_configuration(
                selected, source_features, shuffled_source_targets, class_counts,
                layer_indices, seeds, training
            )
            save_ensemble(
                temporary / f"selected-mlp-trained-{train_order}.pt",
                selected_models,
                {"train_fact_order": train_order, **selected},
            )
            save_ensemble(
                temporary / f"matched-mlp-trained-{train_order}.pt",
                matched_models,
                {"train_fact_order": train_order, **matched},
            )
            regimes[train_order] = {
                "selected": selected,
                "selection": selection,
                "evaluation": evaluate_orders(
                    selected_models, selected, features["test"], rows["test"],
                    targets["test"], class_counts, layer_indices, device,
                    bootstrap_samples, bootstrap_seed + 100 * source_offset
                ),
                "matched_linear_representation": {
                    "selected": matched,
                    "selection": matched_selection,
                    "evaluation": evaluate_orders(
                        matched_models, matched, features["test"], rows["test"],
                        targets["test"], class_counts, layer_indices, device,
                        bootstrap_samples, bootstrap_seed + 20 + 100 * source_offset
                    ),
                },
                "shuffled_label_control": evaluate_orders(
                    control_models, selected, features["test"], rows["test"],
                    targets["test"], class_counts, layer_indices, device,
                    bootstrap_samples, bootstrap_seed + 40 + 100 * source_offset
                ),
            }
        report = {
            "schema_version": RESULT_SCHEMA_VERSION,
            "protocol": {
                "matrix": [f"{a} -> {b}" for a in FACT_ORDERS for b in FACT_ORDERS],
                "selection": "pooling/layer/MLP hyperparameters selected on matching-order validation only",
                "ensemble": "mean logits across random seeds",
                "interpretation": "condition-specific nonlinear decodability, not causal use",
            },
            "dataset_manifest_sha256": sha256(dataset_dir / "manifest.json"),
            "feature_manifest_sha256": sha256(feature_dir / "manifest.json"),
            "layer_indices": list(layer_indices),
            "poolings": list(poolings),
            "hidden_sizes": list(hidden_sizes),
            "weight_decays": list(weight_decays),
            "seeds": list(seeds),
            "training": training,
            "regimes": regimes,
        }
        report_path = temporary / "report.json"
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        temporary.replace(output_dir)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return output_dir / "report.json"


def _ints(value: str) -> tuple[int, ...]:
    return tuple(int(item) for item in value.split(",") if item.strip())


def _floats(value: str) -> tuple[float, ...]:
    return tuple(float(item) for item in value.split(",") if item.strip())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--feature-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--poolings", default=",".join(DEFAULT_POOLINGS))
    parser.add_argument("--hidden-sizes", default="128")
    parser.add_argument("--weight-decays", default="1e-4,1e-3")
    parser.add_argument("--seeds", default="12012,12013,12014,12015,12016")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=12012)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = run_mlp_fact_order_transfer(
        args.dataset_dir,
        args.feature_dir,
        args.output_dir,
        tuple(item.strip() for item in args.poolings.split(",") if item.strip()),
        _ints(args.hidden_sizes),
        _floats(args.weight_decays),
        _ints(args.seeds),
        args.dropout,
        args.learning_rate,
        args.batch_size,
        args.max_epochs,
        args.patience,
        args.device,
        args.bootstrap_samples,
        args.bootstrap_seed,
    )
    print(json.dumps({"report": str(report)}))


if __name__ == "__main__":
    main()
