#!/usr/bin/env python3
"""Evaluate frozen Qwen on bAbI Task 1 with explicit source visibility."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from memwmlm.data.babi import answer_vocabulary, load_babi_task
from memwmlm.evaluation.babi_baseline import evaluate_babi_variant
from memwmlm.models.candidate_scorer import CausalCandidateScorer
from memwmlm.models.generation_scorer import FreeGenerationScorer
from memwmlm.models.qwen import QwenChatPromptFormatter
from memwmlm.tasks.babi import PROMPT_VERSIONS, BabiVariant, prompt_instruction


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--prompt-version", choices=tuple(PROMPT_VERSIONS), default="general_one_word")
    parser.add_argument("--train-file", type=Path, required=True)
    parser.add_argument("--evaluation-file", type=Path, required=True)
    parser.add_argument(
        "--variant",
        choices=[variant.value for variant in BabiVariant],
        required=True,
        help="Whether application sees the source in Qwen's normal token context.",
    )
    parser.add_argument(
        "--split",
        choices=("train", "test", "both"),
        default="test",
        help="Evaluate train, test, or both; --max-examples applies to each selected split.",
    )
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument(
        "--scoring",
        choices=("candidate_likelihood", "free_generation"),
        default="candidate_likelihood",
        help="Use teacher-forced candidate likelihood or greedy free generation.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=16,
        help="Generation limit for --scoring free_generation.",
    )
    parser.add_argument(
        "--device-map",
        default="auto",
        help='Passed to from_pretrained; use "none" to leave placement to PyTorch.',
    )
    parser.add_argument(
        "--dtype",
        choices=("auto", "bfloat16", "float16", "float32"),
        default="auto",
    )
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def resolve_dtype(name: str) -> str | torch.dtype:
    return "auto" if name == "auto" else getattr(torch, name)


def main() -> None:
    args = parse_args()
    train_examples = load_babi_task(args.train_file)
    test_examples = load_babi_task(args.evaluation_file)
    if args.max_examples is not None and args.max_examples <= 0:
        raise ValueError("--max-examples must be positive")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs: dict[str, object] = {"torch_dtype": resolve_dtype(args.dtype)}
    if args.device_map != "none":
        model_kwargs["device_map"] = args.device_map
    model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)
    model.eval()

    scorer = (
        CausalCandidateScorer(model, tokenizer)
        if args.scoring == "candidate_likelihood"
        else FreeGenerationScorer(model, tokenizer, max_new_tokens=args.max_new_tokens)
    )
    selected_splits = ("train", "test") if args.split == "both" else (args.split,)
    split_examples = {"train": train_examples, "test": test_examples}
    split_results = {}
    for split in selected_splits:
        examples = split_examples[split]
        if args.max_examples is not None:
            examples = examples[: args.max_examples]
        result = evaluate_babi_variant(
            scorer=scorer,
            examples=examples,
            candidates=answer_vocabulary(train_examples),
            variant=BabiVariant(args.variant),
            prompt_formatter=QwenChatPromptFormatter(tokenizer),
            instruction=prompt_instruction(args.prompt_version),
        )
        split_results[split] = {
            "examples": result.examples,
            "correct": result.correct,
            "accuracy": result.accuracy,
        }
    payload = {
        "model": args.model,
        "prompt_version": args.prompt_version,
        "train_file": str(args.train_file),
        "evaluation_file": str(args.evaluation_file),
        "candidate_vocabulary": list(answer_vocabulary(train_examples)),
        "variant": result.variant.value,
        "scoring": args.scoring,
        "max_new_tokens": args.max_new_tokens if args.scoring == "free_generation" else None,
        "splits": split_results,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
