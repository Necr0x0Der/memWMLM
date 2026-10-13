"""Find a token boundary before the application question in the *rendered* prompt.

Tokenize the complete chat-formatted prompt once. Tokenizing its two text pieces
separately can change token IDs at the seam and invalidate a KV equivalence test.
"""

from __future__ import annotations


def split_at_question(tokenizer: object, rendered_prompt: str) -> tuple[list[int], list[int]]:
    marker = "Question: "
    question_start = rendered_prompt.rfind(marker)
    if question_start < 0:
        raise ValueError("Rendered prompt has no application question")
    encoding = tokenizer(
        rendered_prompt, add_special_tokens=True, return_offsets_mapping=True
    )
    ids = encoding["input_ids"]
    offsets = encoding["offset_mapping"]
    if len(ids) != len(offsets):
        raise ValueError("Tokenizer IDs and offsets have different lengths")
    # Special tokens can have zero-width offsets; skip them when locating the seam.
    # If a token joins the separator to "Question:", move the *whole* token to
    # the application side. KV caching can split only between complete tokens.
    split = next(
        (i for i, (start, end) in enumerate(offsets) if end > start and end > question_start),
        None,
    )
    if split is None or split == 0 or split == len(ids):
        raise ValueError("No nonempty token split near the question boundary")
    return ids[:split], ids[split:]
