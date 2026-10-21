"""Batch construction and loss for the source → commit → application sanity check."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import Tensor

from memwmlm.data.babi import BabiExample
from memwmlm.models.qwen import QwenChatPromptFormatter
from memwmlm.tasks.babi import INSTRUCTION
from memwmlm.tasks.babi_memory import render_application_episode, render_source_episode


@dataclass(frozen=True)
class ResetPairBatch:
    source_input_ids: Tensor
    source_attention_mask: Tensor
    application_input_ids: Tensor
    application_attention_mask: Tensor
    labels: Tensor


def make_reset_pair_batch(
    examples: Sequence[BabiExample],
    tokenizer: object,
    formatter: QwenChatPromptFormatter,
    device: torch.device,
    append_eos: bool = True,
    instruction: str = INSTRUCTION,
) -> ResetPairBatch:
    """Create source/application inputs with loss only on application answers."""

    source_texts = [formatter(render_source_episode(example)) for example in examples]
    application_prefixes = [formatter(render_application_episode(example, instruction)) for example in examples]
    source = _tokenize_padded(tokenizer, source_texts, device)

    sequences: list[list[int]] = []
    label_starts: list[int] = []
    for prefix, example in zip(application_prefixes, examples, strict=True):
        prefix_ids = tokenizer(prefix, add_special_tokens=True).input_ids
        target_ids = tokenizer(example.answer, add_special_tokens=False).input_ids
        if not target_ids:
            raise ValueError(f"Empty answer tokenization for {example.answer!r}")
        if append_eos:
            eos_id = tokenizer.eos_token_id
            if eos_id is None:
                raise ValueError("append_eos requires tokenizer.eos_token_id")
            target_ids = [*target_ids, int(eos_id)]
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

    return ResetPairBatch(
        source_input_ids=source["input_ids"],
        source_attention_mask=source["attention_mask"],
        application_input_ids=input_ids,
        application_attention_mask=attention_mask,
        labels=labels,
    )


def causal_answer_loss(logits: Tensor, labels: Tensor) -> Tensor:
    """Mean causal cross-entropy over answer tokens only."""

    return torch.nn.functional.cross_entropy(
        logits[:, :-1].reshape(-1, logits.shape[-1]),
        labels[:, 1:].reshape(-1),
        ignore_index=-100,
    )


def _tokenize_padded(tokenizer: object, texts: Sequence[str], device: torch.device) -> dict[str, Tensor]:
    encoded = tokenizer(list(texts), add_special_tokens=True, padding=True, return_tensors="pt")
    return {name: value.to(device) for name, value in encoded.items()}


def _pad_id(tokenizer: object) -> int:
    token_id = tokenizer.pad_token_id
    if token_id is None:
        token_id = tokenizer.eos_token_id
    if token_id is None:
        raise ValueError("Tokenizer has neither pad_token_id nor eos_token_id")
    return int(token_id)
