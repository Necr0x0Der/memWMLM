"""Evaluation helpers for reset-pair writable memory on whole-text reproduction.

Unlike bAbI (a short answer from a fixed candidate vocabulary), reproduction has
no candidate list, so scoring is by generation against the original window:
exact match, token-level accuracy, and mean log-likelihood of the original text.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import Tensor

from memwmlm.data.repro import ReproExample
from memwmlm.memory.embedding_memory import MemoryState
from memwmlm.models.memory_qwen import FrozenQwenWithMemory
from memwmlm.models.qwen import QwenChatPromptFormatter
from memwmlm.tasks.repro_memory import render_application_episode, render_source_episode, target_text


def _index_key(index: int | None) -> str:
    """Stable per-position key: 'whole' for the whole passage, else the sentence number."""

    return "whole" if index is None else str(index)


@dataclass(frozen=True)
class ReproEvaluationResult:
    examples: int
    exact_correct: int
    accuracy: float          # exact match (normalized)
    mean_token_acc: float    # mean per-token accuracy vs the reproduced target
    mean_loglik: float       # mean per-token log-prob of the reproduced target
    per_position: dict[str, dict] | None = None  # {_index_key: metrics}


@torch.inference_mode()
def _write_one(model: FrozenQwenWithMemory, tokenizer: object, source: str) -> MemoryState:
    encoded = tokenizer(source, add_special_tokens=True, return_tensors="pt")
    encoded = {name: value.to(model.device) for name, value in encoded.items()}
    return model.write_memory(encoded["input_ids"], encoded["attention_mask"])


@torch.inference_mode()
def _generate(
    model: FrozenQwenWithMemory,
    tokenizer: object,
    prefix: str,
    memory: MemoryState,
    max_new_tokens: int,
) -> str:
    encoded = tokenizer(prefix, add_special_tokens=True, return_tensors="pt")
    encoded = {name: value.to(model.device) for name, value in encoded.items()}
    with model.read_memory(memory), model.application_lora(True):
        output_ids = model.qwen.generate(
            **encoded,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(
        output_ids[0, encoded["input_ids"].shape[1]:], skip_special_tokens=True
    )


def _original_log_likelihood(
    model: FrozenQwenWithMemory,
    tokenizer: object,
    prefix: str,
    text: str,
    memory: MemoryState,
) -> float:
    """Mean per-token log-prob of ``text`` given the prefix and committed memory."""

    prefix_ids = tokenizer(prefix, add_special_tokens=True).input_ids
    text_ids = tokenizer(text, add_special_tokens=False).input_ids
    sequence = prefix_ids + text_ids
    input_ids = torch.tensor([sequence], dtype=torch.long, device=model.device)
    attention_mask = torch.ones_like(input_ids)
    logits = model(
        input_ids=input_ids, attention_mask=attention_mask, memory=memory, use_cache=False, return_dict=True
    ).logits
    log_probs = logits[0, :-1].log_softmax(dim=-1)
    target = torch.tensor(sequence[1:], dtype=torch.long, device=model.device)
    gathered = log_probs.gather(-1, target.unsqueeze(-1)).squeeze(-1)
    return float(gathered.mean().item())


def _normalize(text: str) -> str:
    return " ".join(text.strip().strip(".,;:!?`'\"“”").casefold().split())


@torch.inference_mode()
def evaluate_memory_repro(
    model: FrozenQwenWithMemory,
    tokenizer: object,
    examples: Sequence[ReproExample],
    max_new_tokens: int = 256,
    sentence_index: int | None = None,
    sentence_indices: Sequence[int | None] | None = None,
    progress_every: int = 200,
) -> ReproEvaluationResult:
    """Evaluate one example at a time to keep runtime memory unambiguous.

    ``sentence_indices`` (when set) gives a per-example target (None = whole
    passage, else a 1-based sentence), so a single pass reports mixed whole-text
    and single-sentence reproduction. Scoring is against each example's own target
    (not the whole window), which matters for single-sentence targets."""

    formatter = QwenChatPromptFormatter(tokenizer)
    exact_correct = 0
    token_accs: list[float] = []
    logliks: list[float] = []
    pos: dict[str, list[float]] = {}  # index_key -> [cnt, exact, tokacc_sum, loglik_sum]

    def _emit(step: int) -> None:
        run_n = len(token_accs)
        if run_n == 0:
            return
        print(
            json.dumps(
                {
                    "event": "eval_progress",
                    "step": step,
                    "running_exact_acc": exact_correct / run_n,
                    "running_token_acc": sum(token_accs) / run_n,
                    "running_loglik": sum(logliks) / run_n,
                },
                sort_keys=True,
            ),
            flush=True,
        )

    indices = list(sentence_indices) if sentence_indices is not None else [sentence_index] * len(examples)

    for i, example in enumerate(examples, start=1):
        index = indices[i - 1]
        target = target_text(example.text, index)
        memory = _write_one(model, tokenizer, formatter(render_source_episode(example)))
        prefix = formatter(render_application_episode(example, sentence_index=index))
        generated = _generate(model, tokenizer, prefix, memory, max_new_tokens)

        exact = int(_normalize(generated) == _normalize(target))
        exact_correct += exact

        gen_ids = tokenizer(generated, add_special_tokens=False).input_ids
        ref_ids = tokenizer(target, add_special_tokens=False).input_ids
        n = min(len(gen_ids), len(ref_ids))
        token_acc = (sum(1 for a, b in zip(gen_ids[:n], ref_ids[:n]) if a == b) / n) if n else 0.0
        token_accs.append(token_acc)

        loglik = _original_log_likelihood(model, tokenizer, prefix, target, memory)
        logliks.append(loglik)

        key = _index_key(index)
        acc = pos.get(key)
        if acc is None:
            acc = [0.0, 0.0, 0.0, 0.0]
            pos[key] = acc
        acc[0] += 1.0
        acc[1] += exact
        acc[2] += token_acc
        acc[3] += loglik

        if progress_every and i % progress_every == 0:
            _emit(i)

    n = len(examples)
    per_position = {
        key: {
            "examples": int(acc[0]),
            "exact_correct": int(acc[1]),
            "accuracy": acc[1] / acc[0] if acc[0] else 0.0,
            "mean_token_acc": acc[2] / acc[0] if acc[0] else 0.0,
            "mean_loglik": acc[3] / acc[0] if acc[0] else 0.0,
        }
        for key, acc in pos.items()
    }
    return ReproEvaluationResult(
        examples=n,
        exact_correct=exact_correct,
        accuracy=(exact_correct / n) if n else 0.0,
        mean_token_acc=(sum(token_accs) / n) if n else 0.0,
        mean_loglik=(sum(logliks) / n) if n else 0.0,
        per_position=per_position,
    )


@dataclass(frozen=True)
class ReproChainResult:
    """Early-stopped reproduction: how far each window is copied before the first
    token mismatch with the original. Cheaper and more informative than full
    generation, since most windows diverge long before the token budget."""

    examples: int
    mean_chain_len: float      # mean tokens copied correctly before first error
    mean_ref_len: float        # mean reference (window) length in tokens
    exact_correct: int         # chains that reached the full reference length
    coverage: float            # mean(chain_len / ref_len)
    per_position: dict[str, dict] | None = None  # {_index_key: metrics}


@torch.inference_mode()
def _correct_chain_length(
    model: FrozenQwenWithMemory,
    prefix_ids: Sequence[int],
    ref_ids: Sequence[int],
    memory: MemoryState,
    max_new_tokens: int,
) -> int:
    """Greedy-generate and stop at the first token that differs from ``ref_ids``.

    Returns the number of leading tokens reproduced correctly (the "chain")."""

    input_ids = torch.tensor([list(prefix_ids)], dtype=torch.long, device=model.device)
    limit = min(max_new_tokens, len(ref_ids))
    correct = 0
    with model.read_memory(memory), model.application_lora(True):
        outputs = model.qwen(input_ids, use_cache=True, return_dict=True)
        logits_last = outputs.logits[0, -1]
        past = outputs.past_key_values
        for position in range(limit):
            next_token = int(logits_last.argmax(dim=-1))
            if next_token != ref_ids[position]:
                break
            correct += 1
            if position < limit - 1:
                new_token = torch.tensor([[next_token]], dtype=torch.long, device=model.device)
                outputs = model.qwen(new_token, past_key_values=past, use_cache=True, return_dict=True)
                logits_last = outputs.logits[0, -1]
                past = outputs.past_key_values
    return correct


@torch.inference_mode()
def evaluate_memory_repro_until_error(
    model: FrozenQwenWithMemory,
    tokenizer: object,
    examples: Sequence[ReproExample],
    max_new_tokens: int = 256,
    sentence_index: int | None = None,
    sentence_indices: Sequence[int | None] | None = None,
    progress_every: int = 200,
) -> ReproChainResult:
    """Evaluate reproduction by first-match chain length, one example at a time.

    ``sentence_indices`` (when set) gives a per-example target; the chain is then
    measured against that target's tokens (not the whole window), so single-sentence
    targets are scored correctly instead of diverging at the first later sentence."""

    formatter = QwenChatPromptFormatter(tokenizer)
    chains: list[int] = []
    ref_lens: list[int] = []
    exact_correct = 0
    pos: dict[str, list[float]] = {}  # index_key -> [cnt, chain_sum, refsum, exact, covsum]

    indices = list(sentence_indices) if sentence_indices is not None else [sentence_index] * len(examples)

    for i, example in enumerate(examples, start=1):
        index = indices[i - 1]
        target = target_text(example.text, index)
        memory = _write_one(model, tokenizer, formatter(render_source_episode(example)))
        prefix = formatter(render_application_episode(example, sentence_index=index))
        prefix_ids = tokenizer(prefix, add_special_tokens=True).input_ids
        ref_ids = tokenizer(target, add_special_tokens=False).input_ids
        chain = _correct_chain_length(model, prefix_ids, ref_ids, memory, max_new_tokens)

        chains.append(chain)
        ref_lens.append(len(ref_ids))
        exact = int(chain >= len(ref_ids))
        exact_correct += exact

        key = _index_key(index)
        acc = pos.get(key)
        if acc is None:
            acc = [0.0, 0.0, 0.0, 0.0, 0.0]
            pos[key] = acc
        cov = chain / len(ref_ids) if ref_ids else 0.0
        acc[0] += 1.0
        acc[1] += chain
        acc[2] += len(ref_ids)
        acc[3] += exact
        acc[4] += cov

        if progress_every and i % progress_every == 0:
            run_n = len(chains)
            print(
                json.dumps(
                    {
                        "event": "chain_progress",
                        "step": i,
                        "running_mean_chain_len": sum(chains) / run_n,
                        "running_mean_ref_len": sum(ref_lens) / run_n,
                        "running_exact": exact_correct / run_n,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    n = len(examples)
    per_position = {
        key: {
            "examples": int(acc[0]),
            "mean_chain_len": acc[1] / acc[0] if acc[0] else 0.0,
            "mean_ref_len": acc[2] / acc[0] if acc[0] else 0.0,
            "exact_correct": int(acc[3]),
            "coverage": acc[4] / acc[0] if acc[0] else 0.0,
        }
        for key, acc in pos.items()
    }
    return ReproChainResult(
        examples=n,
        mean_chain_len=(sum(chains) / n) if n else 0.0,
        mean_ref_len=(sum(ref_lens) / n) if n else 0.0,
        exact_correct=exact_correct,
        coverage=(sum(c / r for c, r in zip(chains, ref_lens)) / n) if n else 0.0,
        per_position=per_position,
    )
