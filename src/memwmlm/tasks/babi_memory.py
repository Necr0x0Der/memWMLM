"""Source/application rendering for the reset-pair bAbI memory experiment."""

from __future__ import annotations

from memwmlm.data.babi import BabiExample
from memwmlm.tasks.babi import INSTRUCTION


SOURCE_INSTRUCTION = "Read and retain the following story for a later question."


def render_source_episode(example: BabiExample) -> str:
    """Render source without exposing its later question or answer."""

    return "\n\n".join((SOURCE_INSTRUCTION, "Story:", "\n".join(example.story)))


def render_application_episode(example: BabiExample, instruction: str = INSTRUCTION) -> str:
    """Render application in the same task/output contract as the baseline."""

    return "\n\n".join((instruction, f"Question: {example.question}", "Answer:"))
