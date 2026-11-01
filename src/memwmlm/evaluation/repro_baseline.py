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


@dataclass(frozen=True)
class ReproBaselineResult:
    variant: ReproVariant
    examples: int
    exact_correct: int
    accuracy: float
    mean_token_acc: float
    mean_loglik: float


@dataclass(frozen=True)
class ReproBaselineChainResult:
    variant: ReproVariant
    examples: int
    mean_chain_len: float
    mean_ref_len: float
    exact_correct: int
    coverage: float


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
    progress_every: int = 200,
) -> ReproBaselineResult:
    """Generate the requested text from the base Qwen context and score it."""

    if not examples:
        raise ValueError("Cannot evaluate an empty example sequence")

    formatter = QwenChatPromptFormatter(tokenizer)
    exact_correct = 0
    token_accs: list[float] = []
    logliks: list[float] = []

    for index, example in enumerate(examples, start=1):
        prefix = formatter(render_baseline_prefix(example, variant, sentence_index))
        target = target_text(example, sentence_index)
        generated = _generate(model, tokenizer, prefix, max_new_tokens)

        exact_correct += int(_normalize(generated) == _normalize(target))

        gen_ids = tokenizer(generated, add_special_tokens=False).input_ids
        ref_ids = tokenizer(target, add_special_tokens=False).input_ids
        n = min(len(gen_ids), len(ref_ids))
        if n:
            correct = sum(1 for a, b in zip(gen_ids[:n], ref_ids[:n]) if a == b)
            token_accs.append(correct / n)

        logliks.append(_target_log_likelihood(model, tokenizer, prefix, target))

        if progress_every and index % progress_every == 0:
            run_n = len(token_accs)
            print(
                json.dumps(
                    {
                        "event": "eval_progress",
                        "step": index,
                        "variant": variant,
                        "running_exact_acc": exact_correct / index,
                        "running_token_acc": (sum(token_accs) / run_n) if run_n else 0.0,
                        "running_loglik": sum(logliks) / index,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    n = len(examples)
    return ReproBaselineResult(
        variant=variant,
        examples=n,
        exact_correct=exact_correct,
        accuracy=(exact_correct / n) if n else 0.0,
        mean_token_acc=(sum(token_accs) / n) if n else 0.0,
        mean_loglik=(sum(logliks) / n) if n else 0.0,
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
    progress_every: int = 200,
) -> ReproBaselineChainResult:
    """Measure how many leading target tokens the base model reproduces exactly."""

    if not examples:
        raise ValueError("Cannot evaluate an empty example sequence")

    formatter = QwenChatPromptFormatter(tokenizer)
    chains: list[int] = []
    ref_lens: list[int] = []
    exact_correct = 0

    for index, example in enumerate(examples, start=1):
        prefix = formatter(render_baseline_prefix(example, variant, sentence_index))
        target = target_text(example, sentence_index)
        prefix_ids = tokenizer(prefix, add_special_tokens=True).input_ids
        ref_ids = tokenizer(target, add_special_tokens=False).input_ids

        chain = _correct_chain_length(model, prefix_ids, ref_ids, max_new_tokens)
        chains.append(chain)
        ref_lens.append(len(ref_ids))
        exact_correct += int(chain >= len(ref_ids))

        if progress_every and index % progress_every == 0:
            run_n = len(chains)
            print(
                json.dumps(
                    {
                        "event": "chain_progress",
                        "step": index,
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
    return ReproBaselineChainResult(
        variant=variant,
        examples=n,
        mean_chain_len=(sum(chains) / n) if n else 0.0,
        mean_ref_len=(sum(ref_lens) / n) if n else 0.0,
        exact_correct=exact_correct,
        coverage=(sum(c / r for c, r in zip(chains, ref_lens)) / n) if n else 0.0,
    )
