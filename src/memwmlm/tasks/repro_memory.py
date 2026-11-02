"""Source/application rendering for the reset-pair reproduction memory experiment.

Mirrors tasks.babi_memory but for whole-text reproduction. The application prompt
supports two modes (the "answer" is appended by the batch builder, not here):

  * ``sentence_index=None``  -> reproduce the whole passage (Stage A default).
  * ``sentence_index=n``     -> print only the n-th (1-based) sentence.

The second mode is the "print exactly the n-th sentence" selective-reproduction
task; it is wired here so Stage A can add it without touching the rendering.
"""

from __future__ import annotations

import random
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


def target_text(text: str, index: Optional[int]) -> str:
    """The text to reproduce: the whole passage (index=None) or the index-th sentence."""

    if index is None:
        return text
    return sentence_target(text, index)


def choose_sentence_index(
    text: str,
    rng: random.Random,
    whole_prob: float,
    exclude_last: bool,
) -> Optional[int]:
    """Sample a reproduction target for the selective-repro task.

    With probability ``whole_prob`` the whole passage is asked (None); otherwise a
    random 1-based sentence is chosen. When ``exclude_last`` is set, the final
    sentence (which may be token-truncated at the window boundary) is skipped, so
    only *full* sentences are queried. Returns None when no sentence is selectable.
    """

    if rng.random() < whole_prob:
        return None
    sentences = split_sentences(text)
    if not sentences:
        return None
    hi = len(sentences) - 1 if exclude_last else len(sentences)
    hi = max(1, hi)
    return rng.randint(1, hi)
