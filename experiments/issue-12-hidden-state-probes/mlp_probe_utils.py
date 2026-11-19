"""Shared shallow-MLP fitting utilities for issue #12 state probes."""

from __future__ import annotations

import copy
import math
import random
from dataclasses import dataclass
from typing import Mapping, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from train_linear_probes import metrics


class MLPProbeBank(nn.Module):
    """One independent one-hidden-layer classifier per state field."""

    def __init__(
        self,
        input_dim: int,
        hidden_size: int,
        class_counts: Mapping[str, int],
        dropout: float,
    ) -> None:
        super().__init__()
        self.heads = nn.ModuleDict(
            {
                name: nn.Sequential(
                    nn.Linear(input_dim, hidden_size),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_size, classes),
                )
                for name, classes in class_counts.items()
            }
        )

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        return {name: head(x) for name, head in self.heads.items()}


@dataclass
class FittedMLP:
    model: MLPProbeBank
    mean: Tensor
    scale: Tensor
    best_epoch: int
    seed: int


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _class_weights(target: Tensor, classes: int, device: str) -> Tensor:
    counts = torch.bincount(target, minlength=classes).to(torch.float32)
    if torch.any(counts == 0):
        raise ValueError("Every probe class must occur in the training subset")
    weights = counts.sum() / (classes * counts)
    return weights.to(device)


def predict_logits(fitted: FittedMLP, x: Tensor, device: str) -> dict[str, Tensor]:
    fitted.model.eval()
    normalized = (x.to(torch.float32) - fitted.mean) / fitted.scale
    with torch.inference_mode():
        logits = fitted.model(normalized.to(device))
    return {name: value.cpu() for name, value in logits.items()}


def predictions_from_logits(logits: Mapping[str, Tensor]) -> dict[str, Tensor]:
    return {name: value.argmax(dim=-1) for name, value in logits.items()}


def ensemble_logits(
    fitted_models: Sequence[FittedMLP], x: Tensor, device: str
) -> dict[str, Tensor]:
    if not fitted_models:
        raise ValueError("At least one fitted MLP is required")
    totals: dict[str, Tensor] = {}
    for fitted in fitted_models:
        for name, value in predict_logits(fitted, x, device).items():
            totals[name] = totals.get(name, torch.zeros_like(value)) + value
    return {name: value / len(fitted_models) for name, value in totals.items()}


def fit_mlp(
    train_x: Tensor,
    train_targets: Mapping[str, Tensor],
    validation_x: Tensor,
    validation_targets: Mapping[str, Tensor],
    class_counts: Mapping[str, int],
    hidden_size: int,
    dropout: float,
    weight_decay: float,
    learning_rate: float,
    batch_size: int,
    max_epochs: int,
    patience: int,
    seed: int,
    device: str,
) -> FittedMLP:
    if hidden_size <= 0 or batch_size <= 0 or max_epochs <= 0 or patience <= 0:
        raise ValueError("MLP sizes, epochs, batch size, and patience must be positive")
    if not 0 <= dropout < 1:
        raise ValueError("dropout must be in [0, 1)")
    if weight_decay < 0 or learning_rate <= 0:
        raise ValueError("weight decay must be non-negative and learning rate positive")
    _seed_everything(seed)
    train_x = train_x.to(torch.float32)
    validation_x = validation_x.to(torch.float32)
    mean = train_x.mean(dim=0)
    scale = train_x.std(dim=0, unbiased=False).clamp_min(1e-6)
    normalized_train = (train_x - mean) / scale
    normalized_validation = (validation_x - mean) / scale
    model = MLPProbeBank(train_x.shape[1], hidden_size, class_counts, dropout).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    weights = {
        name: _class_weights(train_targets[name], classes, device)
        for name, classes in class_counts.items()
    }
    best_score = -math.inf
    best_epoch = 0
    best_state: dict[str, Tensor] | None = None
    stale_epochs = 0
    generator = torch.Generator(device="cpu").manual_seed(seed)

    for epoch in range(1, max_epochs + 1):
        model.train()
        order = torch.randperm(len(normalized_train), generator=generator)
        for start in range(0, len(order), batch_size):
            indices = order[start : start + batch_size]
            batch_x = normalized_train.index_select(0, indices).to(device)
            logits = model(batch_x)
            losses = [
                F.cross_entropy(
                    logits[name],
                    target.index_select(0, indices).to(device),
                    weight=weights[name],
                )
                for name, target in train_targets.items()
            ]
            optimizer.zero_grad(set_to_none=True)
            torch.stack(losses).mean().backward()
            optimizer.step()

        model.eval()
        with torch.inference_mode():
            validation_logits = model(normalized_validation.to(device))
        validation_predictions = {
            name: value.argmax(dim=-1).cpu()
            for name, value in validation_logits.items()
        }
        score = float(
            metrics(validation_predictions, validation_targets, class_counts)[
                "mean_macro_accuracy"
            ]
        )
        if score > best_score + 1e-6:
            best_score = score
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break
    if best_state is None:
        raise RuntimeError("MLP training did not produce a checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    return FittedMLP(
        model=model,
        mean=mean,
        scale=scale,
        best_epoch=best_epoch,
        seed=seed,
    )


def fit_ensemble(
    train_x: Tensor,
    train_targets: Mapping[str, Tensor],
    validation_x: Tensor,
    validation_targets: Mapping[str, Tensor],
    class_counts: Mapping[str, int],
    seeds: Sequence[int],
    **kwargs: object,
) -> list[FittedMLP]:
    if not seeds:
        raise ValueError("At least one seed is required")
    return [
        fit_mlp(
            train_x,
            train_targets,
            validation_x,
            validation_targets,
            class_counts,
            seed=seed,
            **kwargs,
        )
        for seed in seeds
    ]


def save_ensemble(
    path: str,
    fitted_models: Sequence[FittedMLP],
    config: Mapping[str, object],
) -> None:
    torch.save(
        {
            "config": dict(config),
            "models": [
                {
                    # Copy checkpoint tensors to CPU without mutating the live
                    # model. Evaluation may still need that model on CUDA after
                    # the checkpoint is written.
                    "state_dict": {
                        name: tensor.detach().cpu()
                        for name, tensor in fitted.model.state_dict().items()
                    },
                    "mean": fitted.mean.cpu(),
                    "scale": fitted.scale.cpu(),
                    "best_epoch": fitted.best_epoch,
                    "seed": fitted.seed,
                }
                for fitted in fitted_models
            ],
        },
        path,
    )
