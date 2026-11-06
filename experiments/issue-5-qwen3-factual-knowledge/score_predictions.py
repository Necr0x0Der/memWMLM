#!/usr/bin/env python3
"""Score a QuestionId-to-answer JSON object against TriviaQA v1.0."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from triviaqa import evaluate_predictions, load_triviaqa


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--max-examples", type=int)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    examples = load_triviaqa(args.dataset)
    if args.max_examples is not None:
        if args.max_examples <= 0:
            raise ValueError("--max-examples must be positive")
        examples = examples[: args.max_examples]
    predictions = json.loads(args.predictions.read_text(encoding="utf-8"))
    if not isinstance(predictions, dict):
        raise ValueError("Predictions must be a JSON object keyed by QuestionId")
    result = evaluate_predictions(examples, predictions)
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    print(rendered, end="")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")


if __name__ == "__main__":
    main()
