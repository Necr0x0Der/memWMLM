"""Parser for the original bAbI Tasks text format."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class BabiExample:
    """One bAbI question and the preceding story in its episode."""

    example_id: str
    story: tuple[str, ...]
    question: str
    answer: str
    supporting_fact_ids: tuple[int, ...]


def load_babi_task(path: str | Path) -> list[BabiExample]:
    """Load a bAbI ``qa*_*.txt`` file without altering its text semantics."""

    examples: list[BabiExample] = []
    story: list[tuple[int, str]] = []
    path = Path(path)

    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw_line.strip():
            continue
        try:
            event_id_text, payload = raw_line.split(" ", 1)
            event_id = int(event_id_text)
        except ValueError as error:
            raise ValueError(f"Malformed bAbI line {line_number} in {path}: {raw_line!r}") from error

        # bAbI restarts event numbering at one for every new story.
        if event_id == 1:
            story = []

        if "\t" not in payload:
            story.append((event_id, payload.strip()))
            continue

        fields = payload.split("\t")
        if len(fields) != 3:
            raise ValueError(f"Malformed bAbI question at line {line_number} in {path}")
        question, answer, supporting_ids = (field.strip() for field in fields)
        supporting = tuple(int(item) for item in supporting_ids.split() if item)
        examples.append(
            BabiExample(
                example_id=f"{path.stem}:{line_number}",
                story=tuple(sentence for _, sentence in story),
                question=question,
                answer=answer,
                supporting_fact_ids=supporting,
            )
        )

    if not examples:
        raise ValueError(f"No bAbI examples found in {path}")
    return examples


def answer_vocabulary(examples: list[BabiExample]) -> tuple[str, ...]:
    """Return deterministic candidate answers from a training split."""

    return tuple(sorted({example.answer for example in examples}))
