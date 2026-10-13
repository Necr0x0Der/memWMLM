#!/usr/bin/env python3
"""Evaluate frozen Qwen on full bAbI split with native source KV continuation."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from memwmlm.data.babi import answer_vocabulary, load_babi_task
from memwmlm.evaluation.kv_boundary import split_at_question
from memwmlm.models.candidate_scorer import CausalCandidateScorer
from memwmlm.models.qwen import QwenChatPromptFormatter
from memwmlm.tasks.babi import BabiVariant, render_babi_prompt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--train-file", type=Path, required=True)
    parser.add_argument("--evaluation-file", type=Path, required=True)
    parser.add_argument("--max-examples", type=int, default=None,
                        help="Optional subset for quick checks; default: entire evaluation split")
    parser.add_argument("--verify-examples", type=int, default=3,
                        help="Compare aligned logits against one-pass full context on first N examples (0 disables)")
    parser.add_argument("--verify-mismatches", type=int, default=3,
                        help="Additionally verify up to N examples whose candidate predictions disagree")
    parser.add_argument("--include-examples", action="store_true",
                        help="Include per-example predictions in output JSON")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--dtype", choices=("auto", "bfloat16", "float16", "float32"), default="auto")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


@torch.inference_mode()
def cached_candidate_scores(model, tokenizer, source_ids: list[int], application_ids: list[int],
                            candidates: tuple[str, ...], device: torch.device,
                            verify: bool = False) -> tuple[dict[str, float], dict[str, float] | None,
                                                           dict[str, float] | None]:
    """Prefill source and application once, then fork cache per candidate."""
    source_tensor = torch.tensor([source_ids], device=device)
    source = model(input_ids=source_tensor, attention_mask=torch.ones_like(source_tensor), use_cache=True)
    app_tensor = torch.tensor([application_ids], device=device)
    app = model(
        input_ids=app_tensor,
        attention_mask=torch.ones((1, len(source_ids) + len(application_ids)), device=device, dtype=torch.long),
        past_key_values=source.past_key_values,
        cache_position=torch.arange(len(source_ids), len(source_ids) + len(application_ids), device=device),
        use_cache=True,
    )
    prefix_len = len(source_ids) + len(application_ids)
    scores: dict[str, float] = {}
    single_full_scores: dict[str, float] | None = {} if verify else None
    diagnostics = ({"source_logit_abs_diff": 0.0, "application_logit_abs_diff": 0.0,
                    "answer_logit_abs_diff": 0.0, "single_full_vs_cached_score_abs_diff": 0.0}
                   if verify else None)
    for candidate in candidates:
        suffix = tokenizer(candidate, add_special_tokens=False).input_ids
        if not suffix:
            raise ValueError(f"Empty candidate: {candidate!r}")
        # app[-1] predicts the first candidate token. Feed only preceding
        # candidate tokens to predict the rest; never add EOS or a leading space.
        if len(suffix) > 1:
            preceding = torch.tensor([suffix[:-1]], device=device)
            branch = model(
                input_ids=preceding,
                attention_mask=torch.ones((1, prefix_len + len(suffix) - 1), device=device, dtype=torch.long),
                past_key_values=copy.deepcopy(app.past_key_values),
                cache_position=torch.arange(prefix_len, prefix_len + len(suffix) - 1, device=device),
                use_cache=True,
            )
            answer_logits = torch.cat((app.logits[:, -1:], branch.logits), dim=1)
        else:
            answer_logits = app.logits[:, -1:]
        target = torch.tensor([suffix], device=device)
        scores[candidate] = answer_logits.log_softmax(-1).gather(-1, target.unsqueeze(-1)).sum().item()
        if verify:
            # Expensive exact one-pass logit comparison only for selected examples.
            full_ids = torch.tensor([source_ids + application_ids + suffix], device=device)
            full = model(input_ids=full_ids, attention_mask=torch.ones_like(full_ids), use_cache=False)
            full_source_logits = full.logits[:, :len(source_ids)]
            diagnostics["source_logit_abs_diff"] = max(
                diagnostics["source_logit_abs_diff"],
                (full_source_logits.float() - source.logits.float()).abs().max().item(),
            )
            full_answer_logits = full.logits[:, prefix_len - 1 : -1]
            diagnostics["answer_logit_abs_diff"] = max(
                diagnostics["answer_logit_abs_diff"],
                (full_answer_logits.float() - answer_logits.float()).abs().max().item(),
            )
            full_answer_score = full_answer_logits.log_softmax(-1).gather(-1, target.unsqueeze(-1)).sum().item()
            single_full_scores[candidate] = full_answer_score
            diagnostics["single_full_vs_cached_score_abs_diff"] = max(
                diagnostics["single_full_vs_cached_score_abs_diff"],
                abs(full_answer_score - scores[candidate]),
            )
            # The first application token is predicted by the last source state;
            # later application logits come from the cache continuation.
            full_app_logits = full.logits[:, len(source_ids) - 1 : prefix_len - 1]
            cached_app_logits = torch.cat((source.logits[:, -1:], app.logits[:, :-1]), dim=1)
            diagnostics["application_logit_abs_diff"] = max(
                diagnostics["application_logit_abs_diff"],
                (full_app_logits.float() - cached_app_logits.float()).abs().max().item(),
            )
    return scores, diagnostics, single_full_scores


def main() -> None:
    args = parse_args()
    if args.max_examples is not None and args.max_examples <= 0:
        raise ValueError("--max-examples must be positive")
    if args.verify_examples < 0:
        raise ValueError("--verify-examples must be nonnegative")
    if args.verify_mismatches < 0:
        raise ValueError("--verify-mismatches must be nonnegative")
    train = load_babi_task(args.train_file)
    examples = load_babi_task(args.evaluation_file)
    if args.max_examples is not None:
        examples = examples[: args.max_examples]
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
    baseline = CausalCandidateScorer(model, tokenizer)
    full_correct = cached_correct = mismatches = 0
    max_score_diff = max_batch_vs_single_score_diff = 0.0
    batch_vs_single_mismatches = single_vs_cached_mismatches = 0
    verified_diagnostics: dict[str, float] = {}
    verified_mismatches = 0
    verification_count = 0
    mismatch_details = []
    details = []
    for index, example in enumerate(examples):
        prompt = formatter(render_babi_prompt(example, BabiVariant.FULL_CONTEXT).prefix)
        source_ids, application_ids = split_at_question(tokenizer, prompt)
        with torch.inference_mode():
            full_scores = {item.candidate: item.log_likelihood for item in baseline.score(prompt, candidates)}
            cache_scores, diagnostic, single_full_scores = cached_candidate_scores(
                model, tokenizer, source_ids, application_ids, candidates, device,
                verify=index < args.verify_examples,
            )
        full_prediction = max(full_scores, key=full_scores.get)
        cached_prediction = max(cache_scores, key=cache_scores.get)
        if (full_prediction != cached_prediction and diagnostic is None
                and verified_mismatches < args.verify_mismatches):
            with torch.inference_mode():
                _, diagnostic, single_full_scores = cached_candidate_scores(
                    model, tokenizer, source_ids, application_ids, candidates, device, verify=True,
                )
            verified_mismatches += 1
        if diagnostic is not None:
            verification_count += 1
        full_correct += full_prediction == example.answer
        cached_correct += cached_prediction == example.answer
        mismatches += full_prediction != cached_prediction
        score_diff = max(abs(full_scores[c] - cache_scores[c]) for c in candidates)
        max_score_diff = max(max_score_diff, score_diff)
        if single_full_scores is not None:
            single_prediction = max(single_full_scores, key=single_full_scores.get)
            batch_vs_single_mismatches += full_prediction != single_prediction
            single_vs_cached_mismatches += single_prediction != cached_prediction
            max_batch_vs_single_score_diff = max(
                max_batch_vs_single_score_diff,
                max(abs(full_scores[c] - single_full_scores[c]) for c in candidates),
            )
        if diagnostic is not None:
            for key, value in diagnostic.items():
                verified_diagnostics[key] = max(verified_diagnostics.get(key, 0.0), value)
        if full_prediction != cached_prediction:
            mismatch_details.append({
                "example_id": example.example_id, "target": example.answer,
                "full_prediction": full_prediction, "cached_prediction": cached_prediction,
                "single_full_prediction": (max(single_full_scores, key=single_full_scores.get)
                                           if single_full_scores is not None else None),
                "full_top_score": full_scores[full_prediction],
                "full_runner_up_score": full_scores[cached_prediction],
                "cached_top_score": cache_scores[cached_prediction],
                "cached_runner_up_score": cache_scores[full_prediction],
                "max_candidate_score_abs_diff": score_diff,
                "verified_diagnostics": diagnostic,
            })
        if args.include_examples:
            details.append({"example_id": example.example_id, "target": example.answer,
                            "full_prediction": full_prediction, "cached_prediction": cached_prediction,
                            "max_candidate_score_abs_diff": score_diff,
                            "single_full_prediction": (max(single_full_scores, key=single_full_scores.get)
                                                       if single_full_scores is not None else None),
                            "verified_diagnostics": diagnostic})
    payload = {
        "model": args.model, "train_file": str(args.train_file),
        "evaluation_file": str(args.evaluation_file), "candidate_vocabulary": list(candidates),
        "examples": len(examples), "verified_examples": verification_count,
        "full_correct": full_correct, "full_accuracy": full_correct / len(examples),
        "cached_correct": cached_correct, "cached_accuracy": cached_correct / len(examples),
        "prediction_mismatches": mismatches,
        "max_candidate_score_abs_diff": max_score_diff,
        "batch_vs_single_full_prediction_mismatches_verified": batch_vs_single_mismatches,
        "single_full_vs_cached_prediction_mismatches_verified": single_vs_cached_mismatches,
        "max_batch_vs_single_full_score_abs_diff_verified": (
            max_batch_vs_single_score_diff if verification_count else None),
        "verified_diagnostics": verified_diagnostics if verification_count else None,
        "mismatch_details": mismatch_details,
    }
    if args.include_examples:
        payload["results"] = details
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    print(text, end="")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
