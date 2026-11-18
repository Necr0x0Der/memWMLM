#!/usr/bin/env python3
"""Select and evaluate shared linear world-state probes for issue #12.

The script joins frozen feature shards to dataset rows by ID, trains one
multinomial linear classifier per fixed question, and selects one shared
(pooling, layer, L2) configuration using validation data only.  Test data is
evaluated only after that choice is locked.  A group-wise shuffled-label fit is
reported as a negative control.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import torch
from torch import Tensor, nn


SPLITS = ("train", "validation", "test")
POOLINGS = ("last_token", "mean_pool", "sentence_end_mean")
RESULT_SCHEMA_VERSION = "spatial-linear-probes-v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    seen: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON in {path}:{line_number}") from error
            row_id = row.get("id")
            if not isinstance(row_id, str) or row_id in seen:
                raise ValueError(f"Missing or duplicate id in {path}:{line_number}")
            seen.add(row_id)
            rows.append(row)
    return rows


def load_feature_split(
    feature_dir: Path,
    split: str,
    split_manifest: Mapping[str, object],
    expected_ids: Sequence[str],
) -> tuple[tuple[int, ...], dict[str, Tensor]]:
    """Load one split, verify shard hashes, and reorder tensors to dataset IDs."""
    ids: list[str] = []
    chunks: dict[str, list[Tensor]] = {pooling: [] for pooling in POOLINGS}
    layer_indices: tuple[int, ...] | None = None
    shards = split_manifest.get("shards")
    if not isinstance(shards, list) or not shards:
        raise ValueError(f"No feature shards recorded for split {split!r}")
    for shard_record in shards:
        if not isinstance(shard_record, dict) or not isinstance(shard_record.get("path"), str):
            raise ValueError(f"Invalid shard record for split {split!r}")
        shard_path = feature_dir / split / str(shard_record["path"])
        expected_hash = shard_record.get("sha256")
        actual_hash = sha256(shard_path)
        if expected_hash != actual_hash:
            raise ValueError(
                f"Feature hash mismatch for {shard_path}: "
                f"manifest={expected_hash!r}, actual={actual_hash}"
            )
        payload = torch.load(shard_path, map_location="cpu", weights_only=True)
        shard_layers = tuple(int(value) for value in payload["layer_indices"].tolist())
        if layer_indices is None:
            layer_indices = shard_layers
        elif shard_layers != layer_indices:
            raise ValueError(f"Inconsistent layer indices in {shard_path}")
        shard_ids = payload["ids"]
        if not isinstance(shard_ids, list) or not all(isinstance(item, str) for item in shard_ids):
            raise ValueError(f"Invalid ids in {shard_path}")
        ids.extend(shard_ids)
        for pooling in POOLINGS:
            tensor = payload[pooling]
            if tensor.ndim != 3 or tensor.shape[0] != len(shard_ids):
                raise ValueError(f"Invalid {pooling} shape in {shard_path}: {tuple(tensor.shape)}")
            chunks[pooling].append(tensor)
    if layer_indices is None:
        raise ValueError(f"No layers found for split {split!r}")
    if len(set(ids)) != len(ids):
        raise ValueError(f"Duplicate feature IDs in split {split!r}")
    if set(ids) != set(expected_ids):
        missing = sorted(set(expected_ids) - set(ids))[:3]
        extra = sorted(set(ids) - set(expected_ids))[:3]
        raise ValueError(f"Feature/dataset ID mismatch in {split}: missing={missing}, extra={extra}")
    index = {row_id: position for position, row_id in enumerate(ids)}
    order = torch.tensor([index[row_id] for row_id in expected_ids], dtype=torch.long)
    features = {
        pooling: torch.cat(chunks[pooling], dim=0).index_select(0, order)
        for pooling in POOLINGS
    }
    return layer_indices, features


def encode_targets(
    rows: Sequence[Mapping[str, object]],
    probes: Mapping[str, Mapping[str, object]],
) -> dict[str, Tensor]:
    encoded: dict[str, Tensor] = {}
    for probe_name, definition in probes.items():
        classes = definition.get("classes")
        if not isinstance(classes, list) or len(classes) < 2:
            raise ValueError(f"Invalid classes for probe {probe_name!r}")
        class_to_index = {json.dumps(value, sort_keys=True): index for index, value in enumerate(classes)}
        labels: list[int] = []
        for row in rows:
            targets = row.get("targets")
            if not isinstance(targets, dict) or probe_name not in targets:
                raise ValueError(f"Missing target {probe_name!r} for row {row.get('id')!r}")
            key = json.dumps(targets[probe_name], sort_keys=True)
            if key not in class_to_index:
                raise ValueError(f"Unknown class for target {probe_name!r}: {targets[probe_name]!r}")
            labels.append(class_to_index[key])
        encoded[probe_name] = torch.tensor(labels, dtype=torch.long)
    return encoded


def shuffle_targets_by_group(
    targets: Mapping[str, Tensor],
    rows: Sequence[Mapping[str, object]],
    seed: int,
) -> dict[str, Tensor]:
    """Permute complete state vectors between abstract-layout groups."""
    group_rows: dict[str, list[int]] = {}
    for index, row in enumerate(rows):
        group = row.get("group_id")
        if not isinstance(group, str):
            raise ValueError(f"Missing group_id for row {row.get('id')!r}")
        group_rows.setdefault(group, []).append(index)
    groups = sorted(group_rows)
    source_groups = groups.copy()
    random.Random(seed).shuffle(source_groups)
    if len(groups) > 1 and source_groups == groups:
        source_groups = source_groups[1:] + source_groups[:1]
    shuffled = {name: labels.clone() for name, labels in targets.items()}
    for destination, source in zip(groups, source_groups):
        source_index = group_rows[source][0]
        for name, labels in targets.items():
            source_label = labels[source_index]
            shuffled[name][group_rows[destination]] = source_label
    return shuffled


class ProbeBank(nn.Module):
    def __init__(self, width: int, class_counts: Mapping[str, int]) -> None:
        super().__init__()
        self.heads = nn.ModuleDict(
            {name: nn.Linear(width, count) for name, count in class_counts.items()}
        )
        for head in self.heads.values():
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def forward(self, features: Tensor) -> dict[str, Tensor]:
        return {name: head(features) for name, head in self.heads.items()}


@dataclass
class FittedProbes:
    model: ProbeBank
    mean: Tensor
    scale: Tensor


def fit_probes(
    features: Tensor,
    targets: Mapping[str, Tensor],
    class_counts: Mapping[str, int],
    l2: float,
    max_iter: int,
    device: str,
) -> FittedProbes:
    if l2 < 0 or max_iter <= 0:
        raise ValueError("l2 must be non-negative and max_iter must be positive")
    x = features.to(device=device, dtype=torch.float32)
    mean = x.mean(dim=0)
    scale = x.std(dim=0, unbiased=False).clamp_min(1e-6)
    x = (x - mean) / scale
    y = {name: labels.to(device) for name, labels in targets.items()}
    model = ProbeBank(x.shape[1], class_counts).to(device)
    optimizer = torch.optim.LBFGS(
        model.parameters(),
        lr=1.0,
        max_iter=max_iter,
        tolerance_grad=1e-7,
        tolerance_change=1e-9,
        line_search_fn="strong_wolfe",
    )

    def closure() -> Tensor:
        optimizer.zero_grad()
        logits = model(x)
        losses = [nn.functional.cross_entropy(logits[name], y[name]) for name in logits]
        objective = torch.stack(losses).mean()
        if l2:
            penalty = torch.stack([head.weight.square().sum() for head in model.heads.values()]).mean()
            objective = objective + 0.5 * l2 * penalty
        objective.backward()
        return objective

    optimizer.step(closure)
    model.eval()
    return FittedProbes(model=model, mean=mean, scale=scale)


def predict(fitted: FittedProbes, features: Tensor, device: str) -> dict[str, Tensor]:
    x = features.to(device=device, dtype=torch.float32)
    x = (x - fitted.mean) / fitted.scale
    with torch.inference_mode():
        return {name: logits.argmax(dim=1).cpu() for name, logits in fitted.model(x).items()}


def macro_accuracy(prediction: Tensor, target: Tensor, class_count: int) -> float:
    recalls = []
    for class_index in range(class_count):
        mask = target == class_index
        if torch.any(mask):
            recalls.append((prediction[mask] == target[mask]).to(torch.float32).mean().item())
    return float(sum(recalls) / len(recalls))


def metrics(
    predictions: Mapping[str, Tensor],
    targets: Mapping[str, Tensor],
    class_counts: Mapping[str, int],
    indices: Tensor | None = None,
) -> dict[str, object]:
    if indices is None:
        indices = torch.arange(next(iter(targets.values())).shape[0])
    per_probe: dict[str, object] = {}
    correct_columns: list[Tensor] = []
    for name, target in targets.items():
        selected_target = target[indices]
        selected_prediction = predictions[name][indices]
        correct = selected_prediction == selected_target
        correct_columns.append(correct)
        per_probe[name] = {
            "accuracy": correct.to(torch.float32).mean().item(),
            "macro_accuracy": macro_accuracy(
                selected_prediction, selected_target, class_counts[name]
            ),
        }
    return {
        "examples": int(indices.numel()),
        "mean_macro_accuracy": float(
            sum(float(item["macro_accuracy"]) for item in per_probe.values()) / len(per_probe)
        ),
        "joint_exact_state_accuracy": torch.stack(correct_columns, dim=1)
        .all(dim=1)
        .to(torch.float32)
        .mean()
        .item(),
        "per_probe": per_probe,
    }


def condition_metrics(
    rows: Sequence[Mapping[str, object]],
    predictions: Mapping[str, Tensor],
    targets: Mapping[str, Tensor],
    class_counts: Mapping[str, int],
) -> dict[str, object]:
    result: dict[str, object] = {"overall": metrics(predictions, targets, class_counts)}
    for field in ("instruction_id", "fact_order"):
        values = sorted({str(row[field]) for row in rows})
        result[field] = {
            value: metrics(
                predictions,
                targets,
                class_counts,
                torch.tensor(
                    [index for index, row in enumerate(rows) if row[field] == value],
                    dtype=torch.long,
                ),
            )
            for value in values
        }
    cells: dict[str, object] = {}
    for instruction in sorted({str(row["instruction_id"]) for row in rows}):
        for fact_order in sorted({str(row["fact_order"]) for row in rows}):
            key = f"{instruction}/{fact_order}"
            cells[key] = metrics(
                predictions,
                targets,
                class_counts,
                torch.tensor(
                    [
                        index
                        for index, row in enumerate(rows)
                        if row["instruction_id"] == instruction and row["fact_order"] == fact_order
                    ],
                    dtype=torch.long,
                ),
            )
    result["instruction_x_fact_order"] = cells
    return result


def group_bootstrap_intervals(
    rows: Sequence[Mapping[str, object]],
    predictions: Mapping[str, Tensor],
    targets: Mapping[str, Tensor],
    class_counts: Mapping[str, int],
    samples: int,
    seed: int,
) -> dict[str, list[float]]:
    """Percentile intervals resampling abstract-layout groups, not rows."""
    if samples <= 0:
        raise ValueError("bootstrap samples must be positive")
    group_rows: dict[str, list[int]] = {}
    for index, row in enumerate(rows):
        group = row.get("group_id")
        if not isinstance(group, str):
            raise ValueError(f"Missing group_id for row {row.get('id')!r}")
        group_rows.setdefault(group, []).append(index)
    groups = sorted(group_rows)
    generator = random.Random(seed)
    values: dict[str, list[float]] = {
        "mean_macro_accuracy": [],
        "joint_exact_state_accuracy": [],
    }
    for _ in range(samples):
        sampled_groups = [generator.choice(groups) for _ in groups]
        indices = torch.tensor(
            [index for group in sampled_groups for index in group_rows[group]],
            dtype=torch.long,
        )
        result = metrics(predictions, targets, class_counts, indices)
        for name in values:
            values[name].append(float(result[name]))

    def percentile(sorted_values: Sequence[float], quantile: float) -> float:
        position = (len(sorted_values) - 1) * quantile
        lower = math.floor(position)
        upper = math.ceil(position)
        if lower == upper:
            return sorted_values[lower]
        fraction = position - lower
        return sorted_values[lower] * (1 - fraction) + sorted_values[upper] * fraction

    return {
        name: [percentile(sorted(items), 0.025), percentile(sorted(items), 0.975)]
        for name, items in values.items()
    }


def save_probe(path: Path, fitted: FittedProbes, config: Mapping[str, object]) -> None:
    torch.save(
        {
            "config": dict(config),
            "mean": fitted.mean.cpu(),
            "scale": fitted.scale.cpu(),
            "state_dict": {name: value.cpu() for name, value in fitted.model.state_dict().items()},
        },
        path,
    )


def run_experiment(
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
    dataset_manifest = json.loads((dataset_dir / "manifest.json").read_text(encoding="utf-8"))
    feature_manifest = json.loads((feature_dir / "manifest.json").read_text(encoding="utf-8"))
    probes = dataset_manifest.get("probes")
    if not isinstance(probes, dict) or not probes:
        raise ValueError("Dataset manifest contains no probe definitions")
    for split in SPLITS:
        dataset_path = dataset_dir / f"{split}.jsonl"
        expected_hash = feature_manifest.get("dataset", {}).get("files", {}).get(dataset_path.name)
        actual_hash = sha256(dataset_path)
        if expected_hash != actual_hash:
            raise ValueError(
                f"Feature manifest does not match {dataset_path}: "
                f"feature={expected_hash!r}, actual={actual_hash}"
            )
    rows = {split: read_jsonl(dataset_dir / f"{split}.jsonl") for split in SPLITS}
    all_features: dict[str, dict[str, Tensor]] = {}
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
        all_features[split] = split_features
    if layer_indices is None:
        raise ValueError("No feature layers available")
    targets = {split: encode_targets(rows[split], probes) for split in SPLITS}
    class_counts = {name: len(definition["classes"]) for name, definition in probes.items()}

    selection: list[dict[str, object]] = []
    best_key: tuple[float, float, float] | None = None
    best_config: dict[str, object] | None = None
    for pooling in POOLINGS:
        for layer_offset, layer in enumerate(layer_indices):
            train_x = all_features["train"][pooling][:, layer_offset, :]
            validation_x = all_features["validation"][pooling][:, layer_offset, :]
            for l2 in l2_values:
                fitted = fit_probes(
                    train_x, targets["train"], class_counts, l2, max_iter, device
                )
                validation_predictions = predict(fitted, validation_x, device)
                validation_metrics = metrics(
                    validation_predictions, targets["validation"], class_counts
                )
                record = {
                    "pooling": pooling,
                    "layer": layer,
                    "l2": l2,
                    "validation": validation_metrics,
                }
                selection.append(record)
                # The final term deterministically prefers stronger regularization on exact ties.
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
    selected_offset = layer_indices.index(int(best_config["layer"]))
    selected_pooling = str(best_config["pooling"])
    selected_train = all_features["train"][selected_pooling][:, selected_offset, :]
    selected = fit_probes(
        selected_train,
        targets["train"],
        class_counts,
        float(best_config["l2"]),
        max_iter,
        device,
    )
    shuffled_targets = shuffle_targets_by_group(targets["train"], rows["train"], shuffle_seed)
    shuffled = fit_probes(
        selected_train,
        shuffled_targets,
        class_counts,
        float(best_config["l2"]),
        max_iter,
        device,
    )

    evaluations: dict[str, object] = {}
    shuffled_evaluations: dict[str, object] = {}
    for split in ("validation", "test"):
        split_x = all_features[split][selected_pooling][:, selected_offset, :]
        selected_predictions = predict(selected, split_x, device)
        shuffled_predictions = predict(shuffled, split_x, device)
        evaluations[split] = condition_metrics(
            rows[split], selected_predictions, targets[split], class_counts
        )
        evaluations[split]["group_bootstrap_95ci"] = group_bootstrap_intervals(
            rows[split],
            selected_predictions,
            targets[split],
            class_counts,
            bootstrap_samples,
            shuffle_seed + (1 if split == "validation" else 2),
        )
        shuffled_evaluations[split] = condition_metrics(
            rows[split], shuffled_predictions, targets[split], class_counts
        )

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}-", dir=output_dir.parent))
    try:
        save_probe(temporary / "selected_probe.pt", selected, best_config)
        report = {
            "schema_version": RESULT_SCHEMA_VERSION,
            "protocol": {
                "probe_scope": "one shared linear readout per question across all conditions",
                "selection_split": "validation",
                "selection_metric": "mean per-probe macro accuracy; joint exact state then L2 break ties",
                "test_access": "evaluated only after pooling/layer/L2 selection",
                "standardization": "per-feature train mean and population standard deviation",
                "negative_control": "complete target vectors permuted between train abstract-layout groups",
                "confidence_intervals": "95% percentile bootstrap over abstract-layout groups",
            },
            "dataset_manifest_sha256": sha256(dataset_dir / "manifest.json"),
            "feature_manifest_sha256": sha256(feature_dir / "manifest.json"),
            "layer_indices": list(layer_indices),
            "l2_values": list(l2_values),
            "max_iter": max_iter,
            "device": device,
            "shuffle_seed": shuffle_seed,
            "bootstrap_samples": bootstrap_samples,
            "selected": best_config,
            "selection": selection,
            "evaluation": evaluations,
            "shuffled_label_control": shuffled_evaluations,
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
    parser.add_argument(
        "--l2-values",
        default="0,1e-5,1e-4,1e-3",
        help="Comma-separated L2 coefficients selected on validation data.",
    )
    parser.add_argument("--max-iter", type=int, default=100)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--shuffle-seed", type=int, default=12012)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    l2_values = tuple(float(value.strip()) for value in args.l2_values.split(","))
    report = run_experiment(
        args.dataset_dir,
        args.feature_dir,
        args.output_dir,
        l2_values,
        args.max_iter,
        args.device,
        args.shuffle_seed,
        args.bootstrap_samples,
    )
    print(json.dumps({"report": str(report)}))


if __name__ == "__main__":
    main()
