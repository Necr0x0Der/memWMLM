"""Task rendering and exact candidate scoring contracts for bAbI baselines."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from memwmlm.data.babi import BabiExample


class BabiVariant(StrEnum):
    APPLICATION_ONLY = "application_only"
    FULL_CONTEXT = "full_context"


LEGACY_LOCATION_INSTRUCTION = (
    "This is a bAbI question-answering task. "
    "Answer with exactly one location word from the story. Do not explain your answer."
)
GENERAL_INSTRUCTION = (
    "This is a bAbI question-answering task. "
    "Answer with exactly one word. Do not explain your answer."
)
INSTRUCTION = GENERAL_INSTRUCTION
PROMPT_VERSIONS = {"legacy_location": LEGACY_LOCATION_INSTRUCTION, "general_one_word": GENERAL_INSTRUCTION}


def prompt_instruction(version: str) -> str:
    try:
        return PROMPT_VERSIONS[version]
    except KeyError as error:
        raise ValueError(f"Unknown prompt version: {version!r}") from error


@dataclass(frozen=True)
class RenderedBabiPrompt:
    """Prompt before its answer continuation."""

    example_id: str
    variant: BabiVariant
    prefix: str
    target: str


def render_babi_prompt(
    example: BabiExample,
    variant: BabiVariant,
    instruction: str = INSTRUCTION,
) -> RenderedBabiPrompt:
    """Render the fixed prompt used by all frozen-Qwen baseline variants."""

    sections = [instruction]
    if variant is BabiVariant.FULL_CONTEXT:
        sections.extend(("Story:", "\n".join(example.story)))
    sections.extend((f"Question: {example.question}", "Answer:"))
    return RenderedBabiPrompt(
        example_id=example.example_id,
        variant=variant,
        prefix="\n\n".join(sections),
        target=example.answer,
    )
