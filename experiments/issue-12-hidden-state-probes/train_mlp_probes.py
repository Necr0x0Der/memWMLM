#!/usr/bin/env python3
"""Train shallow nonlinear state probes over frozen issue #12 features."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import statistics
import tempfile
from pathlib import Path
from typing import Mapping, Sequence

from torch import Tensor

from mlp_probe_utils import (
    ensemble_logits,
    fit_ensemble,
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


RESULT_SCHEMA_VERSION = "spatial-mlp-probes-v1"


def fit_configuration(
    config: Mapping[str, object],
    features: Mapping[str, Mapping[str, Tensor]],
    targets: Mapping[str, Mapping[str, Tensor]],
    class_counts: Mapping[str, int],
    layer_indices: Sequence[int],
    seeds: Sequence[int],
    training: Mapping[str, object],
) -> tuple[list[object], dict[str, object]]:
    pooling = str(config["pooling"])
    layer_offset = layer_indices.index(int(config["layer"]))
    fitted = fit_ensemble(
        features["train"][pooling][:, layer_offset, :],
        targets["train"],
        features["validation"][pooling][:, layer_offset, :],
        targets["validation"],
        class_counts,
        seeds,
        hidden_size=int(config["hidden_size"]),
        dropout=float(training["dropout"]),
        weight_decay=float(config["weight_decay"]),
        learning_rate=float(training["learning_rate"]),
        batch_size=int(training["batch_size"]),
        max_epochs=int(training["max_epochs"]),
        patience=int(training["patience"]),
        device=str(training["device"]),
    )
    logits = ensemble_logits(
        fitted,
        features["validation"][pooling][:, layer_offset, :],
        str(training["device"]),
    )
    validation = metrics(
        predictions_from_logits(logits), targets["validation"], class_counts
    )
    return fitted, validation


def select_configuration(
    representations: Sequence[tuple[str, int]],
    hidden_sizes: Sequence[int],
    weight_decays: Sequence[float],
    features: Mapping[str, Mapping[str, Tensor]],
    targets: Mapping[str, Mapping[str, Tensor]],
    class_counts: Mapping[str, int],
    layer_indices: Sequence[int],
    seeds: Sequence[int],
    training: Mapping[str, object],
) -> tuple[dict[str, object], list[dict[str, object]]]:
    records: list[dict[str, object]] = []
    best_key: tuple[float, float, float] | None = None
    best_config: dict[str, object] | None = None
    for pooling, layer in representations:
        for hidden_size in hidden_sizes:
            for weight_decay in weight_decays:
                config: dict[str, object] = {
                    "pooling": pooling,
                    "layer": layer,
                    "hidden_size": hidden_size,
                    "weight_decay": weight_decay,
                }
                fitted, validation = fit_configuration(
                    config,
                    features,
                    targets,
                    class_counts,
                    layer_indices,
                    seeds,
                    training,
                )
                records.append(
                    {
                        **config,
                        "best_epochs": [model.best_epoch for model in fitted],
                        "validation": validation,
                    }
                )
                print(
                    json.dumps(
                        {
                            "mlp_candidate": config,
                            "validation_mean_macro_accuracy": validation[
                                "mean_macro_accuracy"
                            ],
                            "validation_joint_exact_state_accuracy": validation[
                                "joint_exact_state_accuracy"
                            ],
                            "best_epochs": [model.best_epoch for model in fitted],
                        }
                    ),
                    flush=True,
                )
                key = (
                    float(validation["mean_macro_accuracy"]),
                    float(validation["joint_exact_state_accuracy"]),
                    -float(weight_decay),
                )
                if best_key is None or key > best_key:
                    best_key = key
                    best_config = config
    if best_config is None:
        raise RuntimeError("No MLP candidate was fitted")
    return best_config, records


def evaluate(
    fitted: Sequence[object],
    config: Mapping[str, object],
    features: Mapping[str, Mapping[str, Tensor]],
    rows: Mapping[str, Sequence[Mapping[str, object]]],
    targets: Mapping[str, Mapping[str, Tensor]],
    class_counts: Mapping[str, int],
    layer_indices: Sequence[int],
    device: str,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, object]:
    pooling = str(config["pooling"])
    layer_offset = layer_indices.index(int(config["layer"]))
    result: dict[str, object] = {}
    for split_offset, split in enumerate(("validation", "test")):
        split_x = features[split][pooling][:, layer_offset, :]
        logits = ensemble_logits(
            fitted, split_x, device
        )
        predictions = predictions_from_logits(logits)
        split_metrics = condition_metrics(
            rows[split], predictions, targets[split], class_counts
        )
        split_metrics["group_bootstrap_95ci"] = group_bootstrap_intervals(
            rows[split],
            predictions,
            targets[split],
            class_counts,
            bootstrap_samples,
            bootstrap_seed + split_offset,
        )
        seed_runs = []
        for model in fitted:
            seed_overall = metrics(
                predictions_from_logits(predict_logits(model, split_x, device)),
                targets[split],
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
        split_metrics["seed_runs"] = seed_runs
        split_metrics["seed_summary"] = {
            metric: {
                "mean": statistics.fmean(float(run[metric]) for run in seed_runs),
                "population_std": statistics.pstdev(
                    float(run[metric]) for run in seed_runs
                ),
            }
            for metric in ("mean_macro_accuracy", "joint_exact_state_accuracy")
        }
        result[split] = split_metrics
    return result


def run_mlp_experiment(
    dataset_dir: Path,
    feature_dir: Path,
    output_dir: Path,
    poolings: Sequence[str] = DEFAULT_POOLINGS,
    hidden_sizes: Sequence[int] = (128,),
    weight_decays: Sequence[float] = (1e-4, 1e-3),
    seeds: Sequence[int] = (12012, 12013, 12014, 12015, 12016),
    matched_pooling: str = "mean_pool",
    matched_layer: int = 16,
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
    if not poolings or not hidden_sizes or not weight_decays or not seeds:
        raise ValueError("Pooling, hidden-size, weight-decay, and seed grids must be non-empty")
    if any(size <= 0 for size in hidden_sizes):
        raise ValueError("Hidden sizes must be positive")
    if any(value < 0 or not math.isfinite(value) for value in weight_decays):
        raise ValueError("Weight decays must be finite and non-negative")

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
    if matched_pooling not in poolings or matched_layer not in layer_indices:
        raise ValueError("Matched representation is absent from the feature candidates")
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
    selected, selection = select_configuration(
        representations,
        hidden_sizes,
        weight_decays,
        features,
        targets,
        class_counts,
        layer_indices,
        seeds,
        training,
    )
    matched, matched_selection = select_configuration(
        [(matched_pooling, matched_layer)],
        hidden_sizes,
        weight_decays,
        features,
        targets,
        class_counts,
        layer_indices,
        seeds,
        training,
    )
    selected_models, _ = fit_configuration(
        selected, features, targets, class_counts, layer_indices, seeds, training
    )
    matched_models, _ = fit_configuration(
        matched, features, targets, class_counts, layer_indices, seeds, training
    )
    shuffled_targets = dict(targets)
    shuffled_targets["train"] = shuffle_targets_by_group(
        targets["train"], rows["train"], bootstrap_seed
    )
    control_models, _ = fit_configuration(
        selected,
        features,
        shuffled_targets,
        class_counts,
        layer_indices,
        seeds,
        training,
    )

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}-", dir=output_dir.parent))
    try:
        save_ensemble(temporary / "selected_mlp.pt", selected_models, selected)
        save_ensemble(temporary / "matched_mlp.pt", matched_models, matched)
        report = {
            "schema_version": RESULT_SCHEMA_VERSION,
            "protocol": {
                "probe_scope": "one independent shallow MLP per state field",
                "selection_split": "validation only",
                "test_access": "test evaluated after representation and MLP hyperparameters are locked",
                "ensemble": "mean logits across random seeds",
                "negative_control": "complete train state vectors shuffled between abstract-layout groups",
                "interpretation": "nonlinear decodability, not causal use",
            },
            "dataset_manifest_sha256": sha256(dataset_dir / "manifest.json"),
            "feature_manifest_sha256": sha256(feature_dir / "manifest.json"),
            "layer_indices": list(layer_indices),
            "poolings": list(poolings),
            "hidden_sizes": list(hidden_sizes),
            "weight_decays": list(weight_decays),
            "seeds": list(seeds),
            "training": training,
            "selected": selected,
            "selection": selection,
            "evaluation": evaluate(
                selected_models, selected, features, rows, targets, class_counts,
                layer_indices, device, bootstrap_samples, bootstrap_seed
            ),
            "matched_linear_representation": {
                "selected": matched,
                "selection": matched_selection,
                "evaluation": evaluate(
                    matched_models, matched, features, rows, targets, class_counts,
                    layer_indices, device, bootstrap_samples, bootstrap_seed + 10
                ),
            },
            "shuffled_label_control": evaluate(
                control_models, selected, features, rows, targets, class_counts,
                layer_indices, device, bootstrap_samples, bootstrap_seed + 20
            ),
        }
        report_path = temporary / "report.json"
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        temporary.replace(output_dir)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return output_dir / "report.json"


def _csv_ints(value: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def _csv_floats(value: str) -> tuple[float, ...]:
    return tuple(float(item.strip()) for item in value.split(",") if item.strip())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--feature-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--poolings", default=",".join(DEFAULT_POOLINGS))
    parser.add_argument("--hidden-sizes", default="128")
    parser.add_argument("--weight-decays", default="1e-4,1e-3")
    parser.add_argument("--seeds", default="12012,12013,12014,12015,12016")
    parser.add_argument("--matched-pooling", default="mean_pool")
    parser.add_argument("--matched-layer", type=int, default=16)
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
    report = run_mlp_experiment(
        args.dataset_dir,
        args.feature_dir,
        args.output_dir,
        tuple(item.strip() for item in args.poolings.split(",") if item.strip()),
        _csv_ints(args.hidden_sizes),
        _csv_floats(args.weight_decays),
        _csv_ints(args.seeds),
        args.matched_pooling,
        args.matched_layer,
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
