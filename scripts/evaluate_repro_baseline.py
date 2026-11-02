#!/usr/bin/env python3
"""Evaluate the frozen base Qwen model on the whole-text reproduction task.

This is the no-memory baseline for ``evaluate_repro_memory.py``: the same
source/application rendering is used, but the passage is either placed in the
model's ordinary context (``full_context``) or omitted (``application_only``).
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from memwmlm.data.repro import load_repro_task
from memwmlm.evaluation.repro_baseline import (
    evaluate_repro_baseline,
    evaluate_repro_baseline_until_error,
)
from memwmlm.tasks.repro_memory import choose_sentence_index
from memwmlm.utils.config import config_path_from_argv, load_config_defaults


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None, help="Flat YAML defaults; CLI flags override it.")
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B", help="Axial model identifier.")
    parser.add_argument("--evaluation-file", type=Path, default=None, help="Held-out windows to reproduce.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--mode", choices=("full", "chain"), default="chain",
                        help="full = generate whole window; chain = early-stop at first mismatch.")
    parser.add_argument("--variant", choices=("full_context", "application_only"), default="full_context",
                        help="full_context = passage in normal context; application_only = passage absent.")
    parser.add_argument("--sentence-index", type=int, default=None,
                        help="Which sentence to reproduce (None = whole passage). Used when a sentence is asked.")
    parser.add_argument("--sentence-mode", choices=("fixed", "random"), default="fixed",
                        help="How the sentence target is chosen (only matters when --whole-prob < 1.0): "
                             "fixed = the same --sentence-index every example; random = a fresh random "
                             "sentence per example.")
    parser.add_argument("--whole-prob", type=float, default=1.0,
                        help="Fraction of steps asking the WHOLE passage vs one sentence. Default 1.0 = "
                             "whole-passage (the original repro mode); <1.0 enables selective reproduction "
                             "(a mix). Only meaningful once a sentence target is possible (a --sentence-index "
                             "in fixed mode, or --sentence-mode random).")
    parser.add_argument("--exclude-last", dest="exclude_last", action="store_true", default=True,
                        help="In random mode, skip the final (possibly truncated) sentence (default: on).")
    parser.add_argument("--include-last", dest="exclude_last", action="store_false",
                        help="In random mode, include the final sentence in sampling.")
    parser.add_argument("--eval-seed", type=int, default=0,
                        help="Seed for per-example sentence sampling in random mode.")
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.set_defaults(**load_config_defaults(config_path_from_argv()))
    args = parser.parse_args()
    for name in ("evaluation_file", "output"):
        value = getattr(args, name)
        if value is not None:
            setattr(args, name, Path(value))
    return args


def main() -> None:
    args = parse_args()
    if args.evaluation_file is None:
        raise ValueError("--evaluation-file is required directly or through --config")
    if args.max_examples is not None and args.max_examples <= 0:
        raise ValueError("--max-examples must be positive")
    if args.max_new_tokens <= 0:
        raise ValueError("--max-new-tokens must be positive")
    if not torch.cuda.is_available() and args.device.startswith("cuda"):
        raise RuntimeError("CUDA is unavailable; pass --device cpu only for a small debug run")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=getattr(torch, args.dtype))
    model.to(args.device)
    model.eval()

    print(
        json.dumps(
            {
                "event": "model_loaded",
                "model": args.model,
                "variant": args.variant,
                "mode": args.mode,
                "trainable_parameters": 0,
                "parameters": sum(parameter.numel() for parameter in model.parameters()),
            },
            sort_keys=True,
        ),
        flush=True,
    )

    eval_examples = load_repro_task(args.evaluation_file)
    if args.max_examples is not None:
        eval_examples = eval_examples[: args.max_examples]

    # In random mode, sample a per-example target (reproducible via --eval-seed);
    # in fixed mode, still sample per example so --whole-prob mixes whole vs the fixed
    # --sentence-index. Either way sentence_indices is a list the evaluators consume.
    eval_rng = random.Random(args.eval_seed)
    if args.sentence_mode == "random":
        sentence_indices: list[int | None] = [
            choose_sentence_index(ex.text, eval_rng, args.whole_prob, args.exclude_last)
            for ex in eval_examples
        ]
    else:
        sentence_indices = [
            (None if eval_rng.random() < args.whole_prob else args.sentence_index)
            for ex in eval_examples
        ]

    if args.mode == "chain":
        result = evaluate_repro_baseline_until_error(
            model,
            tokenizer,
            eval_examples,
            variant=args.variant,
            max_new_tokens=args.max_new_tokens,
            sentence_index=args.sentence_index,
            sentence_indices=sentence_indices,
        )
        payload = {
            "model": args.model,
            "task": "reproduction",
            "variant": args.variant,
            "mode": "chain",
            "sentence_mode": args.sentence_mode,
            "sentence_index": args.sentence_index,
            "whole_prob": args.whole_prob if args.sentence_mode == "random" else None,
            "max_new_tokens": args.max_new_tokens,
            "num_examples": result.examples,
            "mean_chain_len": result.mean_chain_len,
            "mean_ref_len": result.mean_ref_len,
            "exact_correct": result.exact_correct,
            "coverage": result.coverage,
            "per_position": result.per_position,
        }
    else:
        result = evaluate_repro_baseline(
            model,
            tokenizer,
            eval_examples,
            variant=args.variant,
            max_new_tokens=args.max_new_tokens,
            sentence_index=args.sentence_index,
            sentence_indices=sentence_indices,
        )
        payload = {
            "model": args.model,
            "task": "reproduction",
            "variant": args.variant,
            "mode": "full",
            "sentence_mode": args.sentence_mode,
            "sentence_index": args.sentence_index,
            "whole_prob": args.whole_prob if args.sentence_mode == "random" else None,
            "max_new_tokens": args.max_new_tokens,
            "num_examples": result.examples,
            "exact_correct": result.exact_correct,
            "accuracy": result.accuracy,
            "mean_token_acc": result.mean_token_acc,
            "mean_loglik": result.mean_loglik,
            "per_position": result.per_position,
        }

    print(json.dumps(payload, indent=2, sort_keys=True))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
