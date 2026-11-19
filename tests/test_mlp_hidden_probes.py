from __future__ import annotations

import sys
import hashlib
import json
from pathlib import Path

import torch


EXPERIMENT_DIR = Path(__file__).resolve().parents[1] / "experiments" / "issue-12-hidden-state-probes"
sys.path.insert(0, str(EXPERIMENT_DIR))

from mlp_probe_utils import (  # noqa: E402
    FittedMLP,
    MLPProbeBank,
    fit_ensemble,
    predictions_from_logits,
    ensemble_logits,
    save_ensemble,
)
from train_linear_probes import fit_probes, predict  # noqa: E402
from train_mlp_probes import run_mlp_experiment  # noqa: E402
from analyze_mlp_fact_order_transfer import run_mlp_fact_order_transfer  # noqa: E402


def test_shallow_mlp_learns_xor_that_linear_probe_cannot() -> None:
    base_x = torch.tensor(
        [[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]]
    )
    base_y = torch.tensor([0, 1, 1, 0])
    train_x = base_x.repeat(32, 1)
    train_targets = {"xor": base_y.repeat(32)}
    validation_x = base_x.repeat(8, 1)
    validation_targets = {"xor": base_y.repeat(8)}
    class_counts = {"xor": 2}

    linear = fit_probes(
        train_x, train_targets, class_counts, l2=1e-4, max_iter=50, device="cpu"
    )
    linear_accuracy = (
        predict(linear, validation_x, "cpu")["xor"] == validation_targets["xor"]
    ).to(torch.float32).mean()

    fitted = fit_ensemble(
        train_x,
        train_targets,
        validation_x,
        validation_targets,
        class_counts,
        seeds=(7,),
        hidden_size=8,
        dropout=0.0,
        weight_decay=0.0,
        learning_rate=0.03,
        batch_size=16,
        max_epochs=200,
        patience=30,
        device="cpu",
    )
    predictions = predictions_from_logits(
        ensemble_logits(fitted, validation_x, "cpu")
    )
    mlp_accuracy = (
        predictions["xor"] == validation_targets["xor"]
    ).to(torch.float32).mean()

    assert linear_accuracy <= 0.5
    assert mlp_accuracy == 1.0


def test_saving_ensemble_does_not_move_live_model_to_cpu(
    tmp_path: Path, monkeypatch: object
) -> None:
    model = MLPProbeBank(2, 4, {"state": 2}, dropout=0.0)
    fitted = FittedMLP(
        model=model,
        mean=torch.zeros(2),
        scale=torch.ones(2),
        best_epoch=1,
        seed=7,
    )

    def forbidden_cpu_call() -> None:
        raise AssertionError("save_ensemble must not mutate the live model device")

    monkeypatch.setattr(model, "cpu", forbidden_cpu_call)  # type: ignore[attr-defined]
    save_ensemble(tmp_path / "ensemble.pt", [fitted], {"pooling": "mean_pool"})

    checkpoint = torch.load(tmp_path / "ensemble.pt", weights_only=True)
    assert checkpoint["models"][0]["state_dict"]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_mlp_shared_and_transfer_protocols_write_reports(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    feature_dir = tmp_path / "features"
    dataset_dir.mkdir()
    dataset_hashes: dict[str, str] = {}
    split_rows: dict[str, list[dict[str, object]]] = {}
    for split in ("train", "validation", "test"):
        rows: list[dict[str, object]] = []
        values: list[list[list[float]]] = []
        for group, sign in (("negative", -1.0), ("positive", 1.0)):
            for instruction in ("none", "track"):
                for order in ("canonical", "shuffled"):
                    row_id = f"{split}-{group}-{instruction}-{order}"
                    rows.append(
                        {
                            "id": row_id,
                            "group_id": f"{split}-{group}",
                            "instruction_id": instruction,
                            "fact_order": order,
                            "targets": {"positive": int(sign > 0)},
                        }
                    )
                    values.append([[sign, sign * 0.5], [sign * 0.5, sign]])
        split_rows[split] = rows
        dataset_path = dataset_dir / f"{split}.jsonl"
        dataset_path.write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )
        dataset_hashes[dataset_path.name] = _sha256(dataset_path)
        split_dir = feature_dir / split
        split_dir.mkdir(parents=True)
        shard = split_dir / "shard-00000.pt"
        tensor = torch.tensor(values)
        torch.save(
            {
                "ids": [str(row["id"]) for row in rows],
                "layer_indices": torch.tensor([8, 28], dtype=torch.int16),
                "mean_pool": tensor,
            },
            shard,
        )
    (dataset_dir / "manifest.json").write_text(
        json.dumps({"probes": {"positive": {"classes": [0, 1]}}})
    )
    (feature_dir / "manifest.json").write_text(
        json.dumps(
            {
                "dataset": {"files": dataset_hashes},
                "splits": {
                    split: {
                        "shards": [
                            {
                                "path": "shard-00000.pt",
                                "examples": len(split_rows[split]),
                                "sha256": _sha256(feature_dir / split / "shard-00000.pt"),
                            }
                        ]
                    }
                    for split in ("train", "validation", "test")
                },
            }
        )
    )

    shared_path = run_mlp_experiment(
        dataset_dir,
        feature_dir,
        tmp_path / "shared",
        poolings=("mean_pool",),
        hidden_sizes=(2,),
        weight_decays=(0.0,),
        seeds=(7,),
        matched_layer=8,
        dropout=0.0,
        learning_rate=0.02,
        batch_size=8,
        max_epochs=3,
        patience=1,
        bootstrap_samples=5,
    )
    shared = json.loads(shared_path.read_text())
    assert shared["selected"]["pooling"] == "mean_pool"
    assert shared["evaluation"]["test"]["overall"]["examples"] == 8

    transfer_path = run_mlp_fact_order_transfer(
        dataset_dir,
        feature_dir,
        tmp_path / "transfer",
        poolings=("mean_pool",),
        hidden_sizes=(2,),
        weight_decays=(0.0,),
        seeds=(7,),
        dropout=0.0,
        learning_rate=0.02,
        batch_size=8,
        max_epochs=3,
        patience=1,
        bootstrap_samples=5,
    )
    transfer = json.loads(transfer_path.read_text())
    assert set(transfer["regimes"]) == {"canonical", "shuffled"}
    assert set(transfer["regimes"]["canonical"]["evaluation"]) == {
        "canonical",
        "shuffled",
    }
