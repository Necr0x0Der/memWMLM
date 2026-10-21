#!/usr/bin/env python3
"""Evaluate a task-adapted full-context LoRA control via answer candidate likelihood."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from memwmlm.data.babi import answer_vocabulary, load_babi_task
from memwmlm.evaluation.babi_baseline import evaluate_babi_variant
from memwmlm.models.candidate_scorer import CausalCandidateScorer
from memwmlm.models.qwen import QwenChatPromptFormatter
from memwmlm.tasks.babi import PROMPT_VERSIONS, prompt_instruction
from memwmlm.tasks.babi import BabiVariant
from memwmlm.training.full_context_lora import FullContextLoRA
from memwmlm.utils.config import config_path_from_argv, load_config_defaults


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--prompt-version", choices=tuple(PROMPT_VERSIONS), default=None)
    parser.add_argument("--train-file", type=Path, default=None)
    parser.add_argument("--evaluation-file", type=Path, default=None)
    parser.add_argument("--split", choices=("train", "test", "both"), default="both")
    parser.add_argument("--max-examples", type=int, default=None, help="Limit each selected split; default evaluates all")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--output", type=Path, default=None)
    parser.set_defaults(**load_config_defaults(config_path_from_argv()))
    args = parser.parse_args()
    for name in ("checkpoint", "train_file", "evaluation_file", "output"):
        if getattr(args, name) is not None:
            setattr(args, name, Path(getattr(args, name)))
    return args


def main() -> None:
    args = parse_args()
    if args.checkpoint is None or args.train_file is None or args.evaluation_file is None:
        raise ValueError("--checkpoint, --train-file, --evaluation-file are required")
    if args.max_examples is not None and args.max_examples <= 0:
        raise ValueError("--max-examples must be positive")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; use --device cpu for a small smoke test")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("format") != "babi_full_context_lora_v1":
        raise ValueError("Not a bAbI full-context LoRA v1 checkpoint")
    prompt_version = checkpoint.get("prompt_version", "legacy_location")
    if args.prompt_version is not None and args.prompt_version != prompt_version:
        raise ValueError(f"Prompt mismatch: checkpoint={prompt_version}, requested={args.prompt_version}")
    instruction = prompt_instruction(prompt_version)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint["model"])
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(checkpoint["model"], torch_dtype=getattr(torch, args.dtype)).to(args.device)
    model = FullContextLoRA(base, **checkpoint["adapter_config"])
    model.load_adapter_state_dict(checkpoint["adapter_state_dict"])
    model.eval()
    print(json.dumps({"event": "model_loaded", "step": checkpoint["step"], "lora": model.summary()}), flush=True)
    train_examples = load_babi_task(args.train_file)
    test_examples = load_babi_task(args.evaluation_file)
    candidates = answer_vocabulary(train_examples)
    scorer = CausalCandidateScorer(model, tokenizer)
    formatter = QwenChatPromptFormatter(tokenizer)
    splits = {"train": train_examples, "test": test_examples}
    chosen = ("train", "test") if args.split == "both" else (args.split,)
    results = {}
    for split in chosen:
        examples = splits[split][: args.max_examples]
        result = evaluate_babi_variant(scorer, examples, candidates, BabiVariant.FULL_CONTEXT, formatter, instruction)
        results[split] = {"examples": result.examples, "correct": result.correct, "accuracy": result.accuracy}
    payload = {
        "format": "babi_full_context_lora_eval_v1",
        "prompt_version": prompt_version,
        "model": checkpoint["model"], "checkpoint": str(args.checkpoint), "checkpoint_step": checkpoint["step"],
        "lora": model.summary(), "train_file": str(args.train_file), "evaluation_file": str(args.evaluation_file),
        "candidate_vocabulary": list(candidates), "variant": BabiVariant.FULL_CONTEXT.value,
        "scoring": "candidate_likelihood", "splits": results,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
