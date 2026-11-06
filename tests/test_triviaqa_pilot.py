from __future__ import annotations

import json
import sys
from pathlib import Path


EXPERIMENT_DIR = (
    Path(__file__).resolve().parents[1] / "experiments" / "issue-5-qwen3-factual-knowledge"
)
sys.path.insert(0, str(EXPERIMENT_DIR))

from run_qwen import read_completed, render_task  # noqa: E402
from triviaqa import evaluate_predictions, load_triviaqa, normalize_answer, score_prediction  # noqa: E402


def write_fixture(path: Path) -> None:
    payload = {
        "Version": 1.0,
        "Data": [
            {
                "QuestionId": "q1",
                "Question": "Which musical?",
                "Answer": {
                    "NormalizedAliases": ["sunset boulevard", "sunset blvd"],
                    "HumanAnswers": ["Sunset Blvd."],
                },
            },
            {
                "QuestionId": "q2",
                "Question": "Which city?",
                "Answer": {"NormalizedAliases": ["new york city"]},
            },
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_normalize_answer_matches_official_rules() -> None:
    assert normalize_answer("The_Sunset Blvd.!") == "sunset blvd"
    assert normalize_answer("An `Answer`") == "answer"


def test_load_and_score_aliases(tmp_path: Path) -> None:
    dataset = tmp_path / "triviaqa.json"
    write_fixture(dataset)
    examples = load_triviaqa(dataset)

    assert [example.question_id for example in examples] == ["q1", "q2"]
    assert score_prediction("Sunset Blvd.", examples[0].ground_truths) == (1.0, 1.0)


def test_missing_predictions_score_zero(tmp_path: Path) -> None:
    dataset = tmp_path / "triviaqa.json"
    write_fixture(dataset)
    result = evaluate_predictions(load_triviaqa(dataset), {"q1": "Sunset Boulevard"})

    assert result == {
        "exact_match": 50.0,
        "f1": 50.0,
        "answered": 1,
        "denominator": 2,
        "prediction_count": 1,
    }


def test_prompt_and_resume_records(tmp_path: Path) -> None:
    prompt = render_task("Who wrote Hamlet?")
    assert prompt.endswith("Question: Who wrote Hamlet?\nAnswer:")
    assert "Do not explain" in prompt

    output = tmp_path / "predictions.jsonl"
    output.write_text('{"question_id": "q1"}\n{"question_id": "q2"}\n', encoding="utf-8")
    assert read_completed(output) == {"q1", "q2"}
