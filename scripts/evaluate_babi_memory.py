#!/usr/bin/env python3
"""Evaluate a trained reset-pair writable-memory sidecar on bAbI Task 1."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from memwmlm.data.babi import answer_vocabulary, load_babi_task
from memwmlm.evaluation.memory_babi import evaluate_memory_babi
from memwmlm.memory.embedding_memory import MemoryState
from memwmlm.models.memory_qwen import FrozenQwenWithMemory, MemoryQwenConfig
from memwmlm.models.qwen import QwenChatPromptFormatter
from memwmlm.tasks.babi_memory import render_application_episode, render_source_episode
from memwmlm.utils.config import config_path_from_argv, load_config_defaults


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None, help="Flat YAML defaults; CLI flags override it.")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--train-file", type=Path, default=None)
    parser.add_argument("--evaluation-file", type=Path, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--scoring", choices=("candidate_likelihood", "free_generation"), default="candidate_likelihood")
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--memory-variant", choices=("memory", "shuffled_memory", "null_memory"), default="memory")
    parser.add_argument(
        "--split",
        choices=("train", "test", "both"),
        default="both",
        help="Evaluate train, test, or both splits; --max-examples applies to each selected split.",
    )
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0, help="Only used for shuffled_memory.")
    parser.add_argument("--output", type=Path, default=None)
    parser.set_defaults(**load_config_defaults(config_path_from_argv()))
    args = parser.parse_args()
    for name in ("checkpoint", "train_file", "evaluation_file", "output"):
        value = getattr(args, name)
        if value is not None:
            setattr(args, name, Path(value))
    return args


def main() -> None:
    args = parse_args()
    if args.checkpoint is None or args.train_file is None or args.evaluation_file is None:
        raise ValueError("--checkpoint, --train-file, and --evaluation-file are required directly or through --config")
    if args.max_examples is not None and args.max_examples <= 0:
        raise ValueError("--max-examples must be positive")
    if args.max_new_tokens <= 0:
        raise ValueError("--max-new-tokens must be positive")
    if not torch.cuda.is_available() and args.device.startswith("cuda"):
        raise RuntimeError("CUDA is unavailable; pass --device cpu only for a small debug run")

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    backbone_name = checkpoint["model"]
    config = MemoryQwenConfig(**checkpoint["memory_config"])
    tokenizer = AutoTokenizer.from_pretrained(backbone_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    qwen = AutoModelForCausalLM.from_pretrained(backbone_name, torch_dtype=getattr(torch, args.dtype))
    qwen.to(args.device)
    model = FrozenQwenWithMemory(qwen, config).to(args.device)
    model.load_trainable_state_dict(checkpoint["sidecar_state_dict"])
    model.eval()

    train_examples = load_babi_task(args.train_file)
    test_examples = load_babi_task(args.evaluation_file)
    candidates = answer_vocabulary(train_examples)
    split_examples = {"train": train_examples, "test": test_examples}
    selected_splits = ("train", "test") if args.split == "both" else (args.split,)
    split_results = {}
    for split in selected_splits:
        examples = split_examples[split]
        if args.max_examples is not None:
            examples = examples[: args.max_examples]
        if args.memory_variant == "memory":
            result = evaluate_memory_babi(
                model, tokenizer, examples, candidates, args.scoring, args.max_new_tokens
            )
        else:
            result = evaluate_control_variant(
                model, tokenizer, examples, candidates, args.scoring, args.max_new_tokens,
                memory_variant=args.memory_variant, seed=args.seed,
            )
        split_results[split] = {
            "examples": result.examples,
            "correct": result.correct,
            "accuracy": result.accuracy,
        }
    payload = {
        "checkpoint": str(args.checkpoint),
        "model": backbone_name,
        "memory_variant": args.memory_variant,
        "scoring": args.scoring,
        "max_new_tokens": args.max_new_tokens if args.scoring == "free_generation" else None,
        "splits": split_results,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    model.close()


@torch.inference_mode()
def evaluate_control_variant(
    model: FrozenQwenWithMemory,
    tokenizer: object,
    examples: list,
    candidates: tuple[str, ...],
    scoring: str,
    max_new_tokens: int,
    memory_variant: str,
    seed: int,
):
    """Evaluate controls by replacing the per-example state before application."""

    # Reuse the public evaluator one pair at a time only for the ordinary path;
    # controls need to substitute memory state, so preserve its scoring semantics
    # with a small local implementation.
    from memwmlm.evaluation.memory_babi import (  # local to keep public API small
        MemoryEvaluationResult,
        _generate,
        _predict_candidate,
        _write_one,
    )

    formatter = QwenChatPromptFormatter(tokenizer)
    states = [_write_one(model, tokenizer, formatter(render_source_episode(example))) for example in examples]
    if memory_variant == "shuffled_memory":
        if len(states) > 1:
            # A cyclic shift guarantees that no example reads its own source.
            offset = random.Random(seed).randint(1, len(states) - 1)
            states = states[offset:] + states[:offset]
    elif memory_variant == "null_memory":
        states = [
            MemoryState(keys=torch.zeros_like(state.keys), values=torch.zeros_like(state.values))
            for state in states
        ]
    else:
        raise ValueError(f"Unknown control variant: {memory_variant}")

    correct = 0
    for example, memory in zip(examples, states, strict=True):
        prefix = formatter(render_application_episode(example))
        if scoring == "candidate_likelihood":
            answer = _predict_candidate(model, tokenizer, prefix, candidates, memory)
        else:
            answer = _generate(model, tokenizer, prefix, candidates, memory, max_new_tokens)
        correct += answer == example.answer
    return MemoryEvaluationResult(len(examples), correct, correct / len(examples))


if __name__ == "__main__":
    main()
