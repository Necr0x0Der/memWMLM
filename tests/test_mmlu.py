from __future__ import annotations

import csv
import json
import sys
from pathlib import Path


EXPERIMENT_DIR = Path(__file__).resolve().parents[1] / "experiments" / "issue-5-qwen3-factual-knowledge"
sys.path.insert(0, str(EXPERIMENT_DIR))

from mmlu import (  # noqa: E402
    MMLUExample,
    load_few_shots,
    load_mmlu,
    render_task,
    summarize_records,
    write_summary,
)
from prepare_mmlu_shuffled import permutation_for, prepare_dataset  # noqa: E402


def write_rows(path: Path, rows: list[list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        csv.writer(handle).writerows(rows)


def test_load_prompt_and_five_shot(tmp_path: Path) -> None:
    row = ["2 + 2 = ?", "3", "4", "5", "6", "B"]
    write_rows(tmp_path / "test" / "abstract_algebra_test.csv", [row])
    write_rows(tmp_path / "dev" / "abstract_algebra_dev.csv", [row] * 5)

    example = load_mmlu(tmp_path)[0]
    few_shots = load_few_shots(tmp_path, "abstract_algebra", 5)
    prompt = render_task(example, few_shots)

    assert example.example_id == "abstract_algebra:test:0"
    assert example.domain == "STEM"
    assert len(few_shots) == 5
    assert prompt.count("Answer: B") == 5
    assert prompt.endswith("Answer:")


def test_summary_is_micro_averaged_by_domain_and_subject() -> None:
    records = [
        {"correct": True, "domain": "STEM", "subject": "astronomy"},
        {"correct": False, "domain": "STEM", "subject": "astronomy"},
        {"correct": True, "domain": "humanities", "subject": "philosophy"},
    ]
    summary = summarize_records(records)

    assert summary["accuracy"] == 2 / 3
    assert summary["per_domain"]["STEM"] == {"accuracy": 0.5, "correct": 1, "count": 2}
    assert summary["per_subject"]["philosophy"]["accuracy"] == 1.0


def test_write_summary_preserves_unicode_line_separator_inside_json(tmp_path: Path) -> None:
    records = tmp_path / "predictions.jsonl"
    row = {
        "correct": True,
        "domain": "humanities",
        "subject": "philosophy",
        "question": "first line\u2028second line",
    }
    records.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")

    summary_path = write_summary(records)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    assert summary["count"] == 1
    assert summary["accuracy"] == 1.0


def test_prepare_shuffled_dataset_is_deterministic_and_remaps_answer(tmp_path: Path) -> None:
    source = tmp_path / "source"
    row = ["Select beta", "alpha", "beta", "gamma", "delta", "B"]
    for index in range(57):
        subject = "abstract_algebra" if index == 0 else f"subject_{index}"
        write_rows(source / "dev" / f"{subject}_dev.csv", [row])
        write_rows(source / "test" / f"{subject}_test.csv", [row])

    output = tmp_path / "shifted"
    manifest_path = prepare_dataset(source, output, seed=20261109)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    transformed = next(csv.reader((output / "test" / "abstract_algebra_test.csv").open()))
    order = permutation_for(20261109, "abstract_algebra:test:0")

    assert (output / "dev" / "abstract_algebra_dev.csv").read_bytes() == (
        source / "dev" / "abstract_algebra_dev.csv"
    ).read_bytes()
    assert transformed[1:5] == [row[1 + index] for index in order]
    assert transformed[5] == "ABCD"[order.index(1)]
    assert transformed[1 + "ABCD".index(transformed[5])] == "beta"
    assert manifest["example_count"] == 57
    assert manifest["examples"][0]["new_to_original_choice_indices"] == list(order)
