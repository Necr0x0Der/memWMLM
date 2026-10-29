"""Batch construction and loss for the source -> commit -> reproduction check.

Mirrors training.reset_pair but the "answer" is the whole passage (or one
sentence), so the loss covers the reproduced tokens rather than a single word.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import Tensor

from memwmlm.data.repro import ReproExample
from memwmlm.models.qwen import QwenChatPromptFormatter
from memwmlm.tasks.repro_memory import (
    REPRO_INSTRUCTION,
    render_application_episode,
    render_source_episode,
    sentence_target,
)
from memwmlm.training.reset_pair import _pad_id, _tokenize_padded


@dataclass(frozen=True)
class ReproBatch:
    source_input_ids: Tensor
    source_attention_mask: Tensor
    application_input_ids: Tensor
    application_attention_mask: Tensor
    labels: Tensor


def make_repro_batch(
    examples: Sequence[ReproExample],
    tokenizer: object,
    formatter: QwenChatPromptFormatter,
    device: torch.device,
    append_eos: bool = True,
    instruction: str = REPRO_INSTRUCTION,
    sentence_index: int | None = None,
    max_tokens: int | None = None,
) -> ReproBatch:
    """Create source/application inputs with loss only on the reproduced text.

    ``max_tokens`` (when set) caps each application sequence length (prefix +
    target) by truncating the tail of the target passage, which tames the long
    windows (up to ~713 tokens) that otherwise spike activation memory."""

    source_texts = [formatter(render_source_episode(example)) for example in examples]
    application_prefixes = [
        formatter(render_application_episode(example, instruction, sentence_index))
        for example in examples
    ]
    src_kwargs: dict[str, object] = {"add_special_tokens": True, "padding": True, "return_tensors": "pt"}
    if max_tokens is not None:
        src_kwargs["max_length"] = max_tokens
        src_kwargs["truncation"] = True
    source_encoded = tokenizer(list(source_texts), **src_kwargs)
    source = {name: value.to(device) for name, value in source_encoded.items()}

    sequences: list[list[int]] = []
    label_starts: list[int] = []
    for prefix, example in zip(application_prefixes, examples, strict=True):
        prefix_ids = tokenizer(prefix, add_special_tokens=True).input_ids
        target_text = example.text if sentence_index is None else sentence_target(example.text, sentence_index)
        target_ids = tokenizer(target_text, add_special_tokens=False).input_ids
        if not target_ids:
            raise ValueError(f"Empty target tokenization for {example.example_id!r}")
        if append_eos:
            eos_id = tokenizer.eos_token_id
            if eos_id is None:
                raise ValueError("append_eos requires tokenizer.eos_token_id")
            target_ids = [*target_ids, int(eos_id)]
        if max_tokens is not None:
            budget = max(0, max_tokens - len(prefix_ids))
            if len(target_ids) > budget:
                target_ids = target_ids[:budget]
        sequences.append(prefix_ids + target_ids)
        label_starts.append(len(prefix_ids))

    pad_id = _pad_id(tokenizer)
    max_length = max(map(len, sequences))
    input_ids = torch.full((len(sequences), max_length), pad_id, dtype=torch.long, device=device)
    attention_mask = torch.zeros_like(input_ids)
    labels = torch.full_like(input_ids, -100)
    for row, (sequence, label_start) in enumerate(zip(sequences, label_starts, strict=True)):
        length = len(sequence)
        input_ids[row, :length] = torch.tensor(sequence, dtype=torch.long, device=device)
        attention_mask[row, :length] = 1
        labels[row, label_start:length] = torch.tensor(sequence[label_start:], dtype=torch.long, device=device)

    return ReproBatch(
        source_input_ids=source["input_ids"],
        source_attention_mask=source["attention_mask"],
        application_input_ids=input_ids,
        application_attention_mask=attention_mask,
        labels=labels,
    )


def causal_repro_loss(logits: Tensor, labels: Tensor) -> Tensor:
    """Mean causal cross-entropy over reproduced tokens only."""

    return torch.nn.functional.cross_entropy(
        logits[:, :-1].reshape(-1, logits.shape[-1]),
        labels[:, 1:].reshape(-1),
        ignore_index=-100,
    )
