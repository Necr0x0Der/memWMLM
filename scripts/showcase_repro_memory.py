#!/usr/bin/env python3
"""Showcase a trained reset-pair memory sidecar on a few random repro windows.

For each selected window it prints what the model sees and produces:

  SOURCE      - the memorization prompt (the passage shown once, written to memory),
  APPLICATION - the question prompt ("Sentence N:" or the whole passage),
  EXPECTED    - the target the model was trained to reproduce,
  MODEL       - the model's greedy generation.

This is an eyeball check: it confirms the *right question* is asked for each
target and lets you see *what the model actually outputs*. Requires a checkpoint
(and, for a real run, a GPU). Uses the same model-building and generation helpers
as scripts/evaluate_repro_memory.py.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from memwmlm.data.repro import load_repro_task
from memwmlm.evaluation.repro_memory import _generate, _normalize, _write_one
from memwmlm.models.memory_qwen import FrozenQwenWithMemory, MemoryQwenConfig
from memwmlm.models.qwen import QwenChatPromptFormatter
from memwmlm.tasks.repro_memory import (
    REPRO_INSTRUCTION,
    choose_sentence_index,
    render_application_episode,
    render_source_episode,
    target_text,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None, help="Flat YAML defaults; CLI flags override it.")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Trained sidecar checkpoint (.pt).")
    parser.add_argument("--dataset", type=Path, default=None,
                        help="JSONL of windows to sample from (default: heldout.jsonl).")
    parser.add_argument("--num", type=int, default=8, help="How many random windows to show.")
    parser.add_argument("--seed", type=int, default=0, help="Seed for window selection + sentence sampling.")
    parser.add_argument("--sentence-index", type=int, default=None,
                        help="Fixed target: None = whole passage, n = the n-th sentence.")
    parser.add_argument("--sentence-mode", choices=("fixed", "random"), default="random",
                        help="fixed = one --sentence-index for all; random = per-example target.")
    parser.add_argument("--whole-prob", type=float, default=0.5,
                        help="In random mode, probability of asking for the whole passage.")
    parser.add_argument("--exclude-last", dest="exclude_last", action="store_true", default=True,
                        help="In random mode, skip the final (possibly truncated) sentence (default: on).")
    parser.add_argument("--include-last", dest="exclude_last", action="store_false",
                        help="In random mode, include the final sentence in sampling.")
    parser.add_argument("--max-new-tokens", type=int, default=256, help="Generation length.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.set_defaults(**_load_config_defaults())
    args = parser.parse_args()
    if args.checkpoint is not None:
        args.checkpoint = Path(args.checkpoint)
    if args.dataset is not None:
        args.dataset = Path(args.dataset)
    return args


def _load_config_defaults():
    """Load YAML defaults from the default config if present; empty dict otherwise."""
    from memwmlm.utils.config import config_path_from_argv, load_config_defaults

    try:
        return load_config_defaults(config_path_from_argv())
    except (FileNotFoundError, ValueError):
        return {}


def build_model(
    args: argparse.Namespace,
    checkpoint: dict,
) -> tuple[FrozenQwenWithMemory, object]:
    """Build the frozen-Qwen + memory sidecar from a checkpoint (mirrors eval)."""
    backbone_name = checkpoint["model"]
    config = MemoryQwenConfig.from_checkpoint(checkpoint["memory_config"])
    tokenizer = AutoTokenizer.from_pretrained(backbone_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    qwen = AutoModelForCausalLM.from_pretrained(backbone_name, torch_dtype=getattr(torch, args.dtype))
    qwen.to(args.device)
    model = FrozenQwenWithMemory(qwen, config).to(args.device)
    model.load_trainable_state_dict(checkpoint["sidecar_state_dict"])
    model.eval()
    return model, tokenizer


def _first_diff(expected: str, generated: str) -> int:
    """Length of the common prefix of two strings (0 = differs at the very start)."""
    n = min(len(expected), len(generated))
    i = 0
    while i < n and expected[i] == generated[i]:
        i += 1
    return i


def _print_block(title: str, body: str) -> None:
    print(f"  {title}:\n    {body.replace(chr(10), chr(10) + '    ')}")


def main() -> None:
    args = parse_args()
    if args.checkpoint is None:
        raise ValueError("--checkpoint is required")
    if not 0.0 <= args.whole_prob <= 1.0:
        raise ValueError("--whole-prob must be in [0, 1]")

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model, tokenizer = build_model(args, checkpoint)
    print(
        f"# checkpoint step={checkpoint.get('step', '?')}  "
        f"memory_width={MemoryQwenConfig.from_checkpoint(checkpoint['memory_config']).memory_width}  "
        f"lora={model.lora_summary()}"
    )

    examples = load_repro_task(args.dataset) if args.dataset else _load_default_dataset()
    if not examples:
        raise ValueError("No examples found in the dataset")
    sel_rng = random.Random(args.seed)
    draw_rng = random.Random(args.seed + 12345)
    selected = sel_rng.sample(examples, min(args.num, len(examples)))
    formatter = QwenChatPromptFormatter(tokenizer)

    print(f"# showing {len(selected)} random windows (seed={args.seed})\n")

    for k, ex in enumerate(selected, start=1):
        index = (
            choose_sentence_index(ex.text, draw_rng, args.whole_prob, args.exclude_last)
            if args.sentence_mode == "random"
            else args.sentence_index
        )
        target = target_text(ex.text, index)
        source = render_source_episode(ex)
        question = render_application_episode(ex, REPRO_INSTRUCTION, index)

        memory = _write_one(model, tokenizer, source)
        prefix = formatter(question)
        generated = _generate(model, tokenizer, prefix, memory, args.max_new_tokens)

        exact = _normalize(generated) == _normalize(target)
        diff_at = _first_diff(target.strip(), generated.strip())

        print("=" * 100)
        print(f"EXAMPLE {k}  (id={ex.example_id}, split={ex.split}, target={'whole' if index is None else f'Sentence {index}'})")
        print("=" * 100)
        _print_block("SOURCE (memorize)", source)
        _print_block("APPLICATION (question)", question)
        _print_block("EXPECTED (target)", target)
        _print_block("MODEL OUTPUT", generated)
        print(f"  ---  exact={ 'YES' if exact else 'no' }   "
              f"expected {len(target.strip())} chars, got {len(generated.strip())}   "
              f"first diff at char {diff_at}  ---")
        print()

    model.close()


def _load_default_dataset() -> list:
    """Default dataset: datasets/repro/heldout.jsonl (never seen during training).

    Anchored to the repo root (parent of scripts/) so it resolves from any cwd
    while keeping no hardcoded absolute path literal in the source.
    """
    default = Path(__file__).resolve().parent.parent / "datasets" / "repro" / "heldout.jsonl"
    if default.exists():
        return load_repro_task(default)
    raise ValueError("Pass --dataset explicitly (default heldout.jsonl not found)")


if __name__ == "__main__":
    main()
