"""Load and summarize the original CSV release of MMLU."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


CHOICES = ("A", "B", "C", "D")

DOMAIN_SUBJECTS = {
    "STEM": {
        "abstract_algebra", "astronomy", "college_biology", "college_chemistry",
        "college_computer_science", "college_mathematics", "college_physics",
        "computer_security", "conceptual_physics", "electrical_engineering",
        "elementary_mathematics", "high_school_biology", "high_school_chemistry",
        "high_school_computer_science", "high_school_mathematics",
        "high_school_physics", "high_school_statistics", "machine_learning",
    },
    "humanities": {
        "formal_logic", "high_school_european_history", "high_school_us_history",
        "high_school_world_history", "international_law", "jurisprudence",
        "logical_fallacies", "moral_disputes", "moral_scenarios", "philosophy",
        "prehistory", "professional_law", "world_religions",
    },
    "social_sciences": {
        "econometrics", "high_school_geography", "high_school_government_and_politics",
        "high_school_macroeconomics", "high_school_microeconomics",
        "high_school_psychology", "human_sexuality", "professional_psychology",
        "public_relations", "security_studies", "sociology", "us_foreign_policy",
    },
    "other": {
        "anatomy", "business_ethics", "clinical_knowledge", "college_medicine",
        "global_facts", "human_aging", "management", "marketing", "medical_genetics",
        "miscellaneous", "nutrition", "professional_accounting", "professional_medicine",
        "virology",
    },
}

SUBJECT_TO_DOMAIN = {
    subject: domain for domain, subjects in DOMAIN_SUBJECTS.items() for subject in subjects
}


@dataclass(frozen=True)
class MMLUExample:
    example_id: str
    subject: str
    domain: str
    question: str
    choices: tuple[str, str, str, str]
    answer: str


def _read_rows(path: Path) -> list[list[str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle))
    for line_number, row in enumerate(rows, 1):
        if len(row) != 6 or row[-1] not in CHOICES:
            raise ValueError(f"Malformed MMLU row in {path}:{line_number}")
    return rows


def load_mmlu(data_dir: Path, split: str = "test") -> list[MMLUExample]:
    """Load all subjects in deterministic subject/file order."""
    split_dir = data_dir / split
    files = sorted(split_dir.glob(f"*_{split}.csv"))
    if not files:
        raise ValueError(f"No MMLU {split} CSV files found under {split_dir}")

    examples = []
    for path in files:
        subject = path.name.removesuffix(f"_{split}.csv")
        if subject not in SUBJECT_TO_DOMAIN:
            raise ValueError(f"No domain mapping for MMLU subject {subject!r}")
        for index, row in enumerate(_read_rows(path)):
            examples.append(
                MMLUExample(
                    example_id=f"{subject}:{split}:{index}",
                    subject=subject,
                    domain=SUBJECT_TO_DOMAIN[subject],
                    question=row[0],
                    choices=tuple(row[1:5]),  # type: ignore[arg-type]
                    answer=row[5],
                )
            )
    return examples


def load_few_shots(data_dir: Path, subject: str, n_shot: int) -> list[MMLUExample]:
    if n_shot == 0:
        return []
    path = data_dir / "dev" / f"{subject}_dev.csv"
    rows = _read_rows(path)
    if len(rows) < n_shot:
        raise ValueError(f"{path} has only {len(rows)} rows, need {n_shot}")
    return [
        MMLUExample(
            example_id=f"{subject}:dev:{index}",
            subject=subject,
            domain=SUBJECT_TO_DOMAIN[subject],
            question=row[0],
            choices=tuple(row[1:5]),  # type: ignore[arg-type]
            answer=row[5],
        )
        for index, row in enumerate(rows[:n_shot])
    ]


def format_example(example: MMLUExample, include_answer: bool) -> str:
    lines = [example.question]
    lines.extend(f"{letter}. {choice}" for letter, choice in zip(CHOICES, example.choices))
    lines.append(f"Answer:{' ' + example.answer if include_answer else ''}")
    return "\n".join(lines)


def render_task(example: MMLUExample, few_shots: Iterable[MMLUExample]) -> str:
    heading = (
        "The following are multiple choice questions (with answers) about "
        f"{example.subject.replace('_', ' ')}."
    )
    sections = [heading, *(format_example(item, True) for item in few_shots)]
    sections.append(format_example(example, False))
    return "\n\n".join(sections)


def summarize_records(records: Iterable[dict]) -> dict:
    groups: dict[str, dict[str, list[bool]]] = {
        "domain": defaultdict(list),
        "subject": defaultdict(list),
    }
    correct = []
    for record in records:
        value = bool(record["correct"])
        correct.append(value)
        groups["domain"][record["domain"]].append(value)
        groups["subject"][record["subject"]].append(value)

    def summarize_group(values: dict[str, list[bool]]) -> dict:
        return {
            key: {"accuracy": sum(items) / len(items), "correct": sum(items), "count": len(items)}
            for key, items in sorted(values.items())
        }

    return {
        "accuracy": sum(correct) / len(correct) if correct else None,
        "correct": sum(correct),
        "count": len(correct),
        "per_domain": summarize_group(groups["domain"]),
        "per_subject": summarize_group(groups["subject"]),
    }


def write_summary(records_path: Path) -> Path:
    records = [json.loads(line) for line in records_path.read_text(encoding="utf-8").splitlines()]
    output = records_path.with_suffix(records_path.suffix + ".summary.json")
    output.write_text(json.dumps(summarize_records(records), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output
