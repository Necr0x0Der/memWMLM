"""Evaluation of frozen causal language models on bAbI Task 1."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Callable
from typing import Sequence

from memwmlm.data.babi import BabiExample
from memwmlm.models.candidate_scorer import AnswerScorer
from memwmlm.tasks.babi import BabiVariant, render_babi_prompt


@dataclass(frozen=True)
class BabiEvaluationResult:
    variant: BabiVariant
    examples: int
    correct: int
    accuracy: float


def evaluate_babi_variant(
    scorer: AnswerScorer,
    examples: Sequence[BabiExample],
    candidates: Sequence[str],
    variant: BabiVariant,
    prompt_formatter: Callable[[str], str] | None = None,
) -> BabiEvaluationResult:
    """Run exact candidate scoring for one fixed source-visibility variant."""

    if not examples:
        raise ValueError("Cannot evaluate an empty example sequence")

    correct = 0
    for example in examples:
        prompt = render_babi_prompt(example, variant)
        prefix = prompt_formatter(prompt.prefix) if prompt_formatter is not None else prompt.prefix
        prediction = scorer.predict(prefix, candidates)
        correct += prediction.candidate == prompt.target

    count = len(examples)
    return BabiEvaluationResult(
        variant=variant,
        examples=count,
        correct=correct,
        accuracy=correct / count,
    )
