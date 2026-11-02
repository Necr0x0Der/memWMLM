"""Frozen-base evaluation on the reproduction task.

This mirrors ``evaluation/repro_memory.py`` but uses an ordinary Qwen model with
no writer, reader, memory cells, or LoRA.  The application prefix is identical;
the only difference is source visibility:

  * ``full_context``     — the passage is placed in the model's normal context.
  * ``application_only`` — the passage is absent (lower-bound control).

Both ``chain`` and ``full`` scoring modes are supported.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from collections.abc import Sequence
from typing import Literal

import torch

from memwmlm.data.repro import ReproExample
from memwmlm.models.qwen import QwenChatPromptFormatter
from memwmlm.tasks.repro_memory import (
    REPRO_INSTRUCTION,
    render_application_episode,
    render_source_episode,
    sentence_target,
)

ReproVariant = Literal["full_context", "application_only"]


def _index_key(index: int | None) -> str:
    """Stable per-position key: 'whole' for the whole passage, else the sentence number."""

    return "whole" if index is None else str(index)


@dataclass(frozen=True)
class ReproBaselineResult:
    variant: ReproVariant
    examples: int
    exact_correct: int
    accuracy: float
    mean_token_acc: float
    mean_loglik: float
    per_position: dict[str, dict] | None = None  # {_index_key: metrics}


@dataclass(frozen=True)
class ReproBaselineChainResult:
    variant: ReproVariant
    examples: int
    mean_chain_len: float
    mean_ref_len: float
    exact_correct: int
    coverage: float
    per_position: dict[str, dict] | None = None  # {_index_key: metrics}


def render_baseline_prefix(
    example: ReproExample,
    variant: ReproVariant,
    sentence_index: int | None = None,
) -> str:
    """Render the application prefix with or without the source passage."""

    application = render_application_episode(example, REPRO_INSTRUCTION, sentence_index)
    if variant == "application_only":
        return application
    return "\n\n".join((render_source_episode(example), application))


def target_text(example: ReproExample, sentence_index: int | None = None) -> str:
    """The text the model is asked to reproduce."""

    if sentence_index is None:
        return example.text
    return sentence_target(example.text, sentence_index)


def _normalize(text: str) -> str:
    return " ".join(text.strip().strip(".,;:!?`'\"“”").casefold().split())


@torch.inference_mode()
def _generate(
    model: object,
    tokenizer: object,
    prefix: str,
    max_new_tokens: int,
) -> str:
    encoded = tokenizer(prefix, add_special_tokens=True, return_tensors="pt")
    encoded = {name: value.to(model.device) for name, value in encoded.items()}
    output_ids = model.generate(
        **encoded,
        do_sample=False,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    continuation = output_ids[0, encoded["input_ids"].shape[1]:]
    return tokenizer.decode(continuation, skip_special_tokens=True)


@torch.inference_mode()
def _target_log_likelihood(
    model: object,
    tokenizer: object,
    prefix: str,
    target: str,
) -> float:
    """Mean per-token log-probability of ``target`` after ``prefix``."""

    prefix_ids = tokenizer(prefix, add_special_tokens=True).input_ids
    target_ids = tokenizer(target, add_special_tokens=False).input_ids
    if not target_ids:
        raise ValueError("Empty target tokenization")

    sequence = prefix_ids + target_ids
    input_ids = torch.tensor([sequence], dtype=torch.long, device=model.device)
    attention_mask = torch.ones_like(input_ids)
    logits = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=False,
        return_dict=True,
    ).logits

    log_probs = logits[0, :-1].log_softmax(dim=-1)
    target_tensor = torch.tensor(sequence[1:], dtype=torch.long, device=model.device)
    gathered = log_probs.gather(-1, target_tensor.unsqueeze(-1)).squeeze(-1)
    return float(gathered.mean().item())


@torch.inference_mode()
def evaluate_repro_baseline(
    model: object,
    tokenizer: object,
    examples: Sequence[ReproExample],
    variant: ReproVariant = "full_context",
    max_new_tokens: int = 256,
    sentence_index: int | None = None,
    sentence_indices: Sequence[int | None] | None = None,
    progress_every: int = 200,
) -> ReproBaselineResult:
    """Generate the requested text from the base Qwen context and score it.

    ``sentence_indices`` (when set) gives a per-example target (None = whole
    passage, else a 1-based sentence), so a single pass reports mixed whole-text
    and single-sentence reproduction, matching the memory sidecar's selective mode.
    Scoring is against each example's own target (not the whole window)."""

    if not examples:
        raise ValueError("Cannot evaluate an empty example sequence")

    formatter = QwenChatPromptFormatter(tokenizer)
    exact_correct = 0
    token_accs: list[float] = []
    logliks: list[float] = []
    pos: dict[str, list[float]] = {}  # index_key -> [cnt, exact, tokacc_sum, loglik_sum]

    indices = list(sentence_indices) if sentence_indices is not None else [sentence_index] * len(examples)

    for i, example in enumerate(examples, start=1):
        index = indices[i - 1]
        prefix = formatter(render_baseline_prefix(example, variant, index))
        target = target_text(example, index)
        generated = _generate(model, tokenizer, prefix, max_new_tokens)

        exact = int(_normalize(generated) == _normalize(target))
        exact_correct += exact

        gen_ids = tokenizer(generated, add_special_tokens=False).input_ids
        ref_ids = tokenizer(target, add_special_tokens=False).input_ids
        n = min(len(gen_ids), len(ref_ids))
        token_acc = (sum(1 for a, b in zip(gen_ids[:n], ref_ids[:n]) if a == b) / n) if n else 0.0
        token_accs.append(token_acc)

        loglik = _target_log_likelihood(model, tokenizer, prefix, target)
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
            run_n = len(token_accs)
            print(
                json.dumps(
                    {
                        "event": "eval_progress",
                        "step": i,
                        "variant": variant,
                        "running_exact_acc": exact_correct / i,
                        "running_token_acc": (sum(token_accs) / run_n) if run_n else 0.0,
                        "running_loglik": sum(logliks) / i,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

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
    return ReproBaselineResult(
        variant=variant,
        examples=n,
        exact_correct=exact_correct,
        accuracy=(exact_correct / n) if n else 0.0,
        mean_token_acc=(sum(token_accs) / n) if n else 0.0,
        mean_loglik=(sum(logliks) / n) if n else 0.0,
        per_position=per_position,
    )


@torch.inference_mode()
def _correct_chain_length(
    model: object,
    prefix_ids: Sequence[int],
    ref_ids: Sequence[int],
    max_new_tokens: int,
) -> int:
    """Greedy-generate and stop at the first token that differs from the target."""

    input_ids = torch.tensor([list(prefix_ids)], dtype=torch.long, device=model.device)
    attention_mask = torch.ones_like(input_ids)
    limit = min(max_new_tokens, len(ref_ids))
    correct = 0

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=True,
        return_dict=True,
    )
    logits_last = outputs.logits[0, -1]
    past = outputs.past_key_values

    for position in range(limit):
        next_token = int(logits_last.argmax(dim=-1))
        if next_token != ref_ids[position]:
            break
        correct += 1
        if position < limit - 1:
            new_token = torch.tensor([[next_token]], dtype=torch.long, device=model.device)
            outputs = model(
                new_token,
                past_key_values=past,
                use_cache=True,
                return_dict=True,
            )
            logits_last = outputs.logits[0, -1]
            past = outputs.past_key_values

    return correct


@torch.inference_mode()
def evaluate_repro_baseline_until_error(
    model: object,
    tokenizer: object,
    examples: Sequence[ReproExample],
    variant: ReproVariant = "full_context",
    max_new_tokens: int = 256,
    sentence_index: int | None = None,
    sentence_indices: Sequence[int | None] | None = None,
    progress_every: int = 200,
) -> ReproBaselineChainResult:
    """Measure how many leading target tokens the base model reproduces exactly.

    ``sentence_indices`` (when set) gives a per-example target; the chain is then
    measured against that target's tokens (not the whole window), so single-sentence
    targets are scored correctly instead of diverging at the first later sentence."""

    if not examples:
        raise ValueError("Cannot evaluate an empty example sequence")

    formatter = QwenChatPromptFormatter(tokenizer)
    chains: list[int] = []
    ref_lens: list[int] = []
    exact_correct = 0
    pos: dict[str, list[float]] = {}  # index_key -> [cnt, chain_sum, refsum, exact, covsum]

    indices = list(sentence_indices) if sentence_indices is not None else [sentence_index] * len(examples)

    for i, example in enumerate(examples, start=1):
        index = indices[i - 1]
        target = target_text(example, index)
        prefix = formatter(render_baseline_prefix(example, variant, index))
        prefix_ids = tokenizer(prefix, add_special_tokens=True).input_ids
        ref_ids = tokenizer(target, add_special_tokens=False).input_ids

        chain = _correct_chain_length(model, prefix_ids, ref_ids, max_new_tokens)
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
                        "variant": variant,
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
    return ReproBaselineChainResult(
        variant=variant,
        examples=n,
        mean_chain_len=(sum(chains) / n) if n else 0.0,
        mean_ref_len=(sum(ref_lens) / n) if n else 0.0,
        exact_correct=exact_correct,
        coverage=(sum(c / r for c, r in zip(chains, ref_lens)) / n) if n else 0.0,
        per_position=per_position,
    )
