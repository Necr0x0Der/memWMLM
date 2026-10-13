#!/usr/bin/env python3
"""Frozen Qwen: verify native source KV transfer against identical full-context tokens."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from memwmlm.data.babi import answer_vocabulary, load_babi_task
from memwmlm.evaluation.kv_boundary import split_at_question
from memwmlm.models.qwen import QwenChatPromptFormatter
from memwmlm.tasks.babi import BabiVariant, render_babi_prompt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--train-file", type=Path, required=True)
    parser.add_argument("--evaluation-file", type=Path, required=True)
    parser.add_argument("--max-examples", type=int, default=20)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--dtype", choices=("auto", "bfloat16", "float16", "float32"), default="auto")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


@torch.inference_mode()
def compare_candidate(model, source_ids: list[int], application_ids: list[int],
                      suffix_ids: list[int], source_cache: object, source_last_logits: torch.Tensor,
                      device: torch.device) -> tuple[float, float, float, float]:
    """Return (full LL, cached LL, max aligned logit difference, max target logprob difference)."""
    continuation = application_ids + suffix_ids
    full_ids = torch.tensor([source_ids + continuation], device=device)
    split = len(source_ids)
    full = model(input_ids=full_ids, attention_mask=torch.ones_like(full_ids), use_cache=False)
    rest_ids = torch.tensor([continuation], device=device)
    # A candidate is a fork of the committed source, not a mutation of the shared cache.
    cached = model(
        input_ids=rest_ids,
        attention_mask=torch.ones((1, len(source_ids) + len(continuation)), device=device, dtype=torch.long),
        past_key_values=copy.deepcopy(source_cache),
        cache_position=torch.arange(split, split + len(continuation), device=device),
        use_cache=True,
    )
    # The last source logit predicts the first application token. Later predictions
    # come from the continuation forward pass, identically aligned to the full run.
    cached_logits = torch.cat((source_last_logits, cached.logits[:, :-1]), dim=1).float()
    full_logits = full.logits[:, split - 1 : -1].float()
    if cached_logits.shape != full_logits.shape:
        raise RuntimeError("Full and cached logit shapes differ")
    max_logit = (cached_logits - full_logits).abs().max().item()
    target = rest_ids[:, 1:]
    full_logprob = full_logits[:, :-1].log_softmax(-1)
    cached_logprob = cached_logits[:, :-1].log_softmax(-1)
    full_token = full_logprob.gather(-1, target.unsqueeze(-1)).squeeze(-1)
    cached_token = cached_logprob.gather(-1, target.unsqueeze(-1)).squeeze(-1)
    # app length is nonzero, so suffix[0] is predicted by app[-1].
    start = len(application_ids) - 1
    full_answer = full_token[:, start:]
    cached_answer = cached_token[:, start:]
    return (full_answer.sum().item(), cached_answer.sum().item(), max_logit,
            (full_answer - cached_answer).abs().max().item())


def main() -> None:
    args = parse_args()
    if args.max_examples <= 0:
        raise ValueError("--max-examples must be positive")
    train = load_babi_task(args.train_file)
    examples = load_babi_task(args.evaluation_file)[: args.max_examples]
    candidates = answer_vocabulary(train)
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if not tokenizer.is_fast:
        raise ValueError("A fast tokenizer with character offsets is required")
    dtype = "auto" if args.dtype == "auto" else getattr(torch, args.dtype)
    kwargs: dict[str, object] = {"torch_dtype": dtype}
    if args.device_map != "none":
        kwargs["device_map"] = args.device_map
    model = AutoModelForCausalLM.from_pretrained(args.model, **kwargs).eval()
    device = model.get_input_embeddings().weight.device
    formatter = QwenChatPromptFormatter(tokenizer)
    results = []
    for example in examples:
        prompt = formatter(render_babi_prompt(example, BabiVariant.FULL_CONTEXT).prefix)
        source_ids, application_ids = split_at_question(tokenizer, prompt)
        input_ids = torch.tensor([source_ids], device=device)
        with torch.inference_mode():
            source = model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids), use_cache=True)
            full_scores = {}
            cache_scores = {}
            max_logit = 0.0
            max_target_logprob = 0.0
            for candidate in candidates:
                suffix = tokenizer(candidate, add_special_tokens=False).input_ids
                if not suffix:
                    raise ValueError(f"Empty candidate: {candidate!r}")
                a, b, c, d = compare_candidate(
                    model, source_ids, application_ids, suffix, source.past_key_values,
                    source.logits[:, -1:], device,
                )
                full_scores[candidate] = a
                cache_scores[candidate] = b
                max_logit = max(max_logit, c)
                max_target_logprob = max(max_target_logprob, d)
        results.append({
            "example_id": example.example_id,
            "target": example.answer,
            "source_tokens": len(source_ids),
            "application_tokens": len(application_ids),
            "full_prediction": max(full_scores, key=full_scores.get),
            "cached_prediction": max(cache_scores, key=cache_scores.get),
            "max_logit_abs_diff": max_logit,
            "max_answer_token_logprob_abs_diff": max_target_logprob,
            "max_candidate_score_abs_diff": max(abs(full_scores[c] - cache_scores[c]) for c in candidates),
        })
    payload = {
        "model": args.model, "train_file": str(args.train_file),
        "evaluation_file": str(args.evaluation_file), "candidate_vocabulary": list(candidates),
        "examples": len(results),
        "full_accuracy": sum(r["full_prediction"] == r["target"] for r in results) / len(results),
        "cached_accuracy": sum(r["cached_prediction"] == r["target"] for r in results) / len(results),
        "prediction_mismatches": sum(r["full_prediction"] != r["cached_prediction"] for r in results),
        "max_logit_abs_diff": max(r["max_logit_abs_diff"] for r in results),
        "max_answer_token_logprob_abs_diff": max(r["max_answer_token_logprob_abs_diff"] for r in results),
        "results": results,
    }
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    print(text, end="")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
