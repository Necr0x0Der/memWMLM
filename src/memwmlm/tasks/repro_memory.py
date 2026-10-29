"""Source/application rendering for the reset-pair reproduction memory experiment.

Mirrors tasks.babi_memory but for whole-text reproduction. The application prompt
supports two modes (the "answer" is appended by the batch builder, not here):

  * ``sentence_index=None``  -> reproduce the whole passage (Stage A default).
  * ``sentence_index=n``     -> print only the n-th (1-based) sentence.

The second mode is the "print exactly the n-th sentence" selective-reproduction
task; it is wired here so Stage A can add it without touching the rendering.
"""

from __future__ import annotations

import re
from typing import Optional

from memwmlm.data.repro import ReproExample

SOURCE_INSTRUCTION = "Read and retain the following passage for later reproduction."
REPRO_INSTRUCTION = "Reproduce the passage from memory."


def render_source_episode(example: ReproExample) -> str:
    """Render source without exposing the application prompt."""

    return "\n\n".join((SOURCE_INSTRUCTION, "Passage:", example.text))


def render_application_episode(
    example: ReproExample,
    instruction: str = REPRO_INSTRUCTION,
    sentence_index: Optional[int] = None,
) -> str:
    """Render the application prompt prefix (target text is appended by the batch)."""

    if sentence_index is None:
        head = "Passage:"
    else:
        head = f"Sentence {sentence_index}:"
    return "\n\n".join((instruction, head))


def split_sentences(text: str, min_len: int = 8) -> list[str]:
    """Split text into sentences (regex fallback; used for the sentence-index task)."""

    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sentences if len(s.strip()) >= min_len]


def sentence_target(text: str, index: int) -> str:
    """Return the 1-based index-th sentence of the text."""

    sentences = split_sentences(text)
    if not sentences:
        raise ValueError("Empty text has no sentences")
    if not (1 <= index <= len(sentences)):
        raise ValueError(f"Sentence index {index} out of range (1..{len(sentences)})")
    return sentences[index - 1]
