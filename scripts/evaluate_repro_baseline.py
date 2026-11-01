#!/usr/bin/env python3
"""Evaluate the frozen base Qwen model on the whole-text reproduction task.

This is the no-memory baseline for ``evaluate_repro_memory.py``: the same
source/application rendering is used, but the passage is either placed in the
model's ordinary context (``full_context``) or omitted (``application_only``).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from memwmlm.data.repro import load_repro_task
from memwmlm.evaluation.repro_baseline import (
    evaluate_repro_baseline,
    evaluate_repro_baseline_until_error,
)
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
                        help="None = reproduce whole passage; n = print the n-th sentence.")
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

    if args.mode == "chain":
        result = evaluate_repro_baseline_until_error(
            model,
            tokenizer,
            eval_examples,
            variant=args.variant,
            max_new_tokens=args.max_new_tokens,
            sentence_index=args.sentence_index,
        )
        payload = {
            "model": args.model,
            "task": "reproduction",
            "variant": args.variant,
            "mode": "chain",
            "sentence_index": args.sentence_index,
            "max_new_tokens": args.max_new_tokens,
            "num_examples": result.examples,
            "mean_chain_len": result.mean_chain_len,
            "mean_ref_len": result.mean_ref_len,
            "exact_correct": result.exact_correct,
            "coverage": result.coverage,
        }
    else:
        result = evaluate_repro_baseline(
            model,
            tokenizer,
            eval_examples,
            variant=args.variant,
            max_new_tokens=args.max_new_tokens,
            sentence_index=args.sentence_index,
        )
        payload = {
            "model": args.model,
            "task": "reproduction",
            "variant": args.variant,
            "mode": "full",
            "sentence_index": args.sentence_index,
            "max_new_tokens": args.max_new_tokens,
            "num_examples": result.examples,
            "exact_correct": result.exact_correct,
            "accuracy": result.accuracy,
            "mean_token_acc": result.mean_token_acc,
            "mean_loglik": result.mean_loglik,
        }

    print(json.dumps(payload, indent=2, sort_keys=True))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
