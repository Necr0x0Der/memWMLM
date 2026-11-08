from __future__ import annotations

import csv
import sys
from pathlib import Path


EXPERIMENT_DIR = Path(__file__).resolve().parents[1] / "experiments" / "issue-5-qwen3-factual-knowledge"
sys.path.insert(0, str(EXPERIMENT_DIR))

from mmlu import MMLUExample, load_few_shots, load_mmlu, render_task, summarize_records  # noqa: E402


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
