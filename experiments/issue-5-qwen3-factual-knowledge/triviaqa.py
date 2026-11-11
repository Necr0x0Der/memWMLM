"""TriviaQA v1.0 loading and official-style answer metrics."""

from __future__ import annotations

import json
import re
import string
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence


@dataclass(frozen=True)
class TriviaQAExample:
    question_id: str
    question: str
    ground_truths: tuple[str, ...]


def normalize_answer(text: str) -> str:
    """Match the normalization in TriviaQA's official v1.0 evaluator."""

    text = text.replace("_", " ").lower()
    punctuation = set(string.punctuation + "‘’´`")
    text = "".join(" " if character in punctuation else character for character in text)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split()).strip()


def load_triviaqa(path: Path) -> list[TriviaQAExample]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("Version") != 1.0:
        raise ValueError(f"Expected TriviaQA version 1.0, got {payload.get('Version')!r}")
    if not isinstance(payload.get("Data"), list):
        raise ValueError("TriviaQA payload must contain a Data list")

    examples = []
    seen_ids = set()
    for record in payload["Data"]:
        question_id = record.get("QuestionId")
        question = record.get("Question")
        answer = record.get("Answer")
        if not isinstance(question_id, str) or not isinstance(question, str):
            raise ValueError("Each TriviaQA record requires string QuestionId and Question")
        if question_id in seen_ids:
            raise ValueError(f"Duplicate TriviaQA QuestionId: {question_id}")
        if not isinstance(answer, dict):
            raise ValueError(f"TriviaQA record {question_id} has no Answer object")

        aliases = answer.get("NormalizedAliases")
        if not isinstance(aliases, list) or not all(isinstance(item, str) for item in aliases):
            raise ValueError(f"TriviaQA record {question_id} has invalid NormalizedAliases")
        human_answers = answer.get("HumanAnswers", [])
        if not isinstance(human_answers, list) or not all(
            isinstance(item, str) for item in human_answers
        ):
            raise ValueError(f"TriviaQA record {question_id} has invalid HumanAnswers")
        ground_truths = tuple(dict.fromkeys([*aliases, *(normalize_answer(x) for x in human_answers)]))
        if not ground_truths:
            raise ValueError(f"TriviaQA record {question_id} has no ground truths")

        examples.append(TriviaQAExample(question_id, question, ground_truths))
        seen_ids.add(question_id)
    return examples


def exact_match(prediction: str, ground_truth: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def token_f1(prediction: str, ground_truth: str) -> float:
    predicted = normalize_answer(prediction).split()
    expected = normalize_answer(ground_truth).split()
    if not predicted or not expected:
        return float(predicted == expected)
    common = Counter(predicted) & Counter(expected)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(predicted)
    recall = overlap / len(expected)
    return 2 * precision * recall / (precision + recall)


def score_prediction(prediction: str, ground_truths: Sequence[str]) -> tuple[float, float]:
    if not ground_truths:
        raise ValueError("At least one ground truth is required")
    return (
        max(exact_match(prediction, answer) for answer in ground_truths),
        max(token_f1(prediction, answer) for answer in ground_truths),
    )


def evaluate_predictions(
    examples: Sequence[TriviaQAExample], predictions: Mapping[str, str]
) -> dict[str, float | int]:
    if not examples:
        raise ValueError("Cannot evaluate an empty TriviaQA split")
    exact_total = 0.0
    f1_total = 0.0
    answered = 0
    for example in examples:
        prediction = predictions.get(example.question_id)
        if prediction is None:
            continue
        if not isinstance(prediction, str):
            raise ValueError(f"Prediction for {example.question_id} must be a string")
        exact, f1 = score_prediction(prediction, example.ground_truths)
        exact_total += exact
        f1_total += f1
        answered += 1
    denominator = len(examples)
    return {
        "exact_match": 100.0 * exact_total / denominator,
        "f1": 100.0 * f1_total / denominator,
        "answered": answered,
        "denominator": denominator,
        "prediction_count": len(predictions),
    }
