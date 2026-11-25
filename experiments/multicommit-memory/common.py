"""Shared data, batch, memory, and candidate-scoring helpers for the
multi-commit memory experiment."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor

from memwmlm.memory.embedding_memory import MemoryState
from memwmlm.models.qwen import QwenChatPromptFormatter
from memwmlm.training.reset_pair import causal_answer_loss


SOURCE_INSTRUCTION = "Read and retain the following document for later questions."
APPLICATION_INSTRUCTION = (
    "This is a memory question. Answer with exactly one city name. Do not explain."
)


def load_records(path: Path) -> list[dict]:
    records = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            record = json.loads(line)
            for key in ("example_id", "question", "target", "source_documents", "question_type"):
                if key not in record:
                    raise ValueError(f"Missing {key!r} in {path}:{line_number}")
            if not isinstance(record["source_documents"], list) or not record["source_documents"]:
                raise ValueError(f"Invalid source_documents in {path}:{line_number}")
            records.append(record)
    return records


def render_source_document(index: int, text: str) -> str:
    return f"{SOURCE_INSTRUCTION}\n\nDocument {index + 1}:\n{text}"


def render_application(record: dict, instruction: str = APPLICATION_INSTRUCTION) -> str:
    return f"{instruction}\n\nQuestion: {record['question']}\nAnswer:"


def render_full_context(record: dict, instruction: str = APPLICATION_INSTRUCTION) -> str:
    parts = [SOURCE_INSTRUCTION]
    for index, text in enumerate(record["source_documents"], 1):
        parts.append(f"Document {index}:\n{text}")
    parts.append(instruction)
    parts.append(f"Question: {record['question']}\nAnswer:")
    return "\n\n".join(parts)


def tokenize_padded(tokenizer: object, texts: list[str], device: torch.device) -> tuple[Tensor, Tensor]:
    encoded = tokenizer(texts, add_special_tokens=True, padding=True, return_tensors="pt")
    return encoded["input_ids"].to(device), encoded["attention_mask"].to(device)


@dataclass(frozen=True)
class MultiCommitBatch:
    source_input_ids: list[Tensor]
    source_attention_masks: list[Tensor]
    application_input_ids: Tensor
    application_attention_mask: Tensor
    labels: Tensor


def make_multicommit_batch(
    records: list[dict],
    tokenizer: object,
    formatter: QwenChatPromptFormatter,
    device: torch.device,
    append_eos: bool = True,
) -> MultiCommitBatch:
    n_documents = len(records[0]["source_documents"])
    source_ids: list[Tensor] = []
    source_masks: list[Tensor] = []
    for doc_index in range(n_documents):
        texts = [
            formatter(render_source_document(doc_index, record["source_documents"][doc_index]))
            for record in records
        ]
        ids, mask = tokenize_padded(tokenizer, texts, device)
        source_ids.append(ids)
        source_masks.append(mask)

    prefixes = [formatter(render_application(record)) for record in records]
    sequences: list[list[int]] = []
    label_starts: list[int] = []
    for prefix, record in zip(prefixes, records, strict=True):
        prefix_ids = tokenizer(prefix, add_special_tokens=True).input_ids
        target_ids = tokenizer(record["target"], add_special_tokens=False).input_ids
        if not target_ids:
            raise ValueError(f"Empty target tokenization for {record['target']!r}")
        if append_eos:
            eos_id = tokenizer.eos_token_id
            if eos_id is None:
                raise ValueError("append_eos requires eos_token_id")
            target_ids = [*target_ids, int(eos_id)]
        sequences.append(prefix_ids + target_ids)
        label_starts.append(len(prefix_ids))

    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    if pad_id is None:
        raise ValueError("Tokenizer has neither pad_token_id nor eos_token_id")

    max_length = max(len(sequence) for sequence in sequences)
    input_ids = torch.full((len(sequences), max_length), int(pad_id), dtype=torch.long, device=device)
    attention_mask = torch.zeros_like(input_ids)
    labels = torch.full_like(input_ids, -100)
    for row, (sequence, label_start) in enumerate(zip(sequences, label_starts, strict=True)):
        length = len(sequence)
        input_ids[row, :length] = torch.tensor(sequence, dtype=torch.long, device=device)
        attention_mask[row, :length] = 1
        labels[row, label_start:length] = torch.tensor(sequence[label_start:], dtype=torch.long, device=device)

    return MultiCommitBatch(source_ids, source_masks, input_ids, attention_mask, labels)


def concat_memory(states: list[MemoryState]) -> MemoryState:
    keys = torch.cat([state.keys for state in states], dim=1)
    values = torch.cat([state.values for state in states], dim=1)
    masks = torch.cat([state.mask for state in states], dim=1) if all(state.mask is not None for state in states) else None
    return MemoryState(keys=keys, values=values, mask=masks)


@torch.inference_mode()
def write_selected_documents(
    model: object,
    tokenizer: object,
    formatter: QwenChatPromptFormatter,
    record: dict,
    indices: list[int],
) -> MemoryState:
    states = []
    for index in indices:
        text = formatter(render_source_document(index, record["source_documents"][index]))
        encoded = tokenizer(text, add_special_tokens=True, return_tensors="pt")
        ids = encoded["input_ids"].to(model.device)
        mask = encoded["attention_mask"].to(model.device)
        states.append(model.write_memory(ids, mask))
    return concat_memory(states)


def _candidate_sequences(tokenizer: object, prefix: str, candidates: list[str]) -> tuple[list[int], list[tuple[str, list[int]]]]:
    prefix_ids = tokenizer(prefix, add_special_tokens=True).input_ids
    encoded = [(candidate, tokenizer(candidate, add_special_tokens=False).input_ids) for candidate in candidates]
    if not all(tokens for _, tokens in encoded):
        raise ValueError("At least one candidate tokenized to an empty sequence")
    return prefix_ids, encoded


def _score_logits(logits: Tensor, labels: Tensor) -> Tensor:
    log_probs = logits[:, :-1].log_softmax(dim=-1)
    target_ids = labels[:, 1:]
    target_mask = target_ids.ne(-100)
    gathered = log_probs.gather(-1, target_ids.clamp_min(0).unsqueeze(-1)).squeeze(-1)
    return (gathered * target_mask).sum(dim=-1)


def _build_candidate_batch(
    tokenizer: object,
    prefix_ids: list[int],
    encoded: list[tuple[str, list[int]]],
    device: torch.device,
) -> tuple[Tensor, Tensor, Tensor]:
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    if pad_id is None:
        raise ValueError("Tokenizer has neither pad_token_id nor eos_token_id")
    max_length = max(len(prefix_ids) + len(tokens) for _, tokens in encoded)
    input_ids = torch.full((len(encoded), max_length), int(pad_id), dtype=torch.long, device=device)
    attention_mask = torch.zeros_like(input_ids)
    labels = torch.full_like(input_ids, -100)
    for row, (_, suffix) in enumerate(encoded):
        sequence = prefix_ids + suffix
        input_ids[row, : len(sequence)] = torch.tensor(sequence, dtype=torch.long, device=device)
        attention_mask[row, : len(sequence)] = 1
        labels[row, len(prefix_ids) : len(sequence)] = torch.tensor(suffix, dtype=torch.long, device=device)
    return input_ids, attention_mask, labels


@torch.inference_mode()
def predict_with_memory(
    model: object,
    tokenizer: object,
    prefix: str,
    candidates: list[str],
    memory: MemoryState,
) -> str:
    prefix_ids, encoded = _candidate_sequences(tokenizer, prefix, candidates)
    input_ids, attention_mask, labels = _build_candidate_batch(tokenizer, prefix_ids, encoded, model.device)
    repeated_memory = MemoryState(
        keys=memory.keys.expand(len(encoded), -1, -1),
        values=memory.values.expand(len(encoded), -1, -1),
        mask=memory.mask.expand(len(encoded), -1) if memory.mask is not None else None,
    )
    logits = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        memory=repeated_memory,
        use_cache=False,
        return_dict=True,
    ).logits
    scores = _score_logits(logits, labels)
    return encoded[int(scores.argmax())][0]


@torch.inference_mode()
def predict_raw(
    qwen: object,
    tokenizer: object,
    prefix: str,
    candidates: list[str],
) -> str:
    prefix_ids, encoded = _candidate_sequences(tokenizer, prefix, candidates)
    input_ids, attention_mask, labels = _build_candidate_batch(tokenizer, prefix_ids, encoded, qwen.device)
    logits = qwen(input_ids=input_ids, attention_mask=attention_mask, use_cache=False, return_dict=True).logits
    scores = _score_logits(logits, labels)
    return encoded[int(scores.argmax())][0]


@torch.inference_mode()
def predict_with_memory_prefix(
    model: object,
    tokenizer: object,
    prefix: str,
    candidates: list[str],
    memory: MemoryState,
) -> str:
    return predict_with_memory(model, tokenizer, prefix, candidates, memory)


def zero_memory(memory: MemoryState) -> MemoryState:
    return MemoryState(
        keys=torch.zeros_like(memory.keys),
        values=torch.zeros_like(memory.values),
        mask=memory.mask,
    )


def counterfactual_documents(record: dict, swap_city: str) -> list[str]:
    target = record["target"]
    return [document.replace(target, swap_city) for document in record["source_documents"]]
