"""Evaluation helpers for reset-pair writable memory on bAbI."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import Tensor

from memwmlm.data.babi import BabiExample
from memwmlm.memory.embedding_memory import MemoryState
from memwmlm.models.memory_qwen import FrozenQwenWithMemory
from memwmlm.models.qwen import QwenChatPromptFormatter
from memwmlm.tasks.babi import INSTRUCTION
from memwmlm.tasks.babi_memory import render_application_episode, render_source_episode
from memwmlm.training.reset_pair import _pad_id


@dataclass(frozen=True)
class MemoryEvaluationResult:
    examples: int
    correct: int
    accuracy: float


@torch.inference_mode()
def evaluate_memory_babi(
    model: FrozenQwenWithMemory,
    tokenizer: object,
    examples: Sequence[BabiExample],
    candidates: Sequence[str],
    scoring: str,
    max_new_tokens: int,
    instruction: str = INSTRUCTION,
) -> MemoryEvaluationResult:
    """Evaluate one-example-at-a-time to keep runtime memory unambiguous."""

    if scoring not in {"candidate_likelihood", "free_generation"}:
        raise ValueError(f"Unknown scoring mode: {scoring}")
    formatter = QwenChatPromptFormatter(tokenizer)
    correct = 0
    for example in examples:
        memory = _write_one(model, tokenizer, formatter(render_source_episode(example)))
        prefix = formatter(render_application_episode(example, instruction))
        if scoring == "candidate_likelihood":
            prediction = _predict_candidate(model, tokenizer, prefix, candidates, memory)
        else:
            prediction = _generate(model, tokenizer, prefix, candidates, memory, max_new_tokens)
        correct += prediction == example.answer
    return MemoryEvaluationResult(len(examples), correct, correct / len(examples))


def _write_one(model: FrozenQwenWithMemory, tokenizer: object, source: str) -> MemoryState:
    encoded = tokenizer(source, add_special_tokens=True, return_tensors="pt")
    encoded = {name: value.to(model.device) for name, value in encoded.items()}
    return model.write_memory(encoded["input_ids"], encoded["attention_mask"])


def _predict_candidate(
    model: FrozenQwenWithMemory,
    tokenizer: object,
    prefix: str,
    candidates: Sequence[str],
    memory: MemoryState,
) -> str:
    prefix_ids = tokenizer(prefix, add_special_tokens=True).input_ids
    encoded = [(candidate, tokenizer(candidate, add_special_tokens=False).input_ids) for candidate in candidates]
    if not all(tokens for _, tokens in encoded):
        raise ValueError("At least one candidate tokenized to an empty sequence")
    pad_id = _pad_id(tokenizer)
    max_length = max(len(prefix_ids) + len(tokens) for _, tokens in encoded)
    input_ids = torch.full((len(encoded), max_length), pad_id, dtype=torch.long, device=model.device)
    attention_mask = torch.zeros_like(input_ids)
    labels = torch.full_like(input_ids, -100)
    for row, (_, suffix) in enumerate(encoded):
        sequence = prefix_ids + suffix
        input_ids[row, : len(sequence)] = torch.tensor(sequence, device=model.device)
        attention_mask[row, : len(sequence)] = 1
        labels[row, len(prefix_ids) : len(sequence)] = torch.tensor(suffix, device=model.device)

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
    scores = _candidate_log_likelihood(logits, labels)
    return encoded[int(scores.argmax())][0]


def _generate(
    model: FrozenQwenWithMemory,
    tokenizer: object,
    prefix: str,
    candidates: Sequence[str],
    memory: MemoryState,
    max_new_tokens: int,
) -> str:
    encoded = tokenizer(prefix, add_special_tokens=True, return_tensors="pt")
    encoded = {name: value.to(model.device) for name, value in encoded.items()}
    # Generation invokes multiple model forwards. Hooks need the same state for
    # every step, so keep the reader scope active around generate().
    with model.read_memory(memory), model.application_lora(True):
        output_ids = model.qwen.generate(
            **encoded,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    continuation = tokenizer.decode(
        output_ids[0, encoded["input_ids"].shape[1] :], skip_special_tokens=True
    )
    normalized = _normalize(continuation)
    lookup = {_normalize(candidate): candidate for candidate in candidates}
    return lookup.get(normalized, continuation.strip())


def _candidate_log_likelihood(logits: Tensor, labels: Tensor) -> Tensor:
    log_probs = logits[:, :-1].log_softmax(dim=-1)
    target_ids = labels[:, 1:]
    target_mask = target_ids.ne(-100)
    gathered = log_probs.gather(-1, target_ids.clamp_min(0).unsqueeze(-1)).squeeze(-1)
    return (gathered * target_mask).sum(dim=-1)


def _normalize(text: str) -> str:
    return " ".join(text.strip().strip(".,;:!?`'\"“”").casefold().split())
