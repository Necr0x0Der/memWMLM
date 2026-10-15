#!/usr/bin/env python3
"""Evaluate learned episodic KV reader on bAbI train/test and causal controls."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from memwmlm.data.babi import answer_vocabulary, load_babi_task
from memwmlm.models.episodic_qwen import EpisodicConfig, FrozenQwenWithEpisodicKV
from memwmlm.models.qwen import QwenChatPromptFormatter
from memwmlm.tasks.babi_memory import render_application_episode, render_source_episode
from memwmlm.utils.config import config_path_from_argv, load_config_defaults


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--train-file", type=Path)
    p.add_argument("--evaluation-file", type=Path)
    p.add_argument("--split", choices=("train", "test", "both"), default="test")
    p.add_argument("--variant", choices=("memory", "shuffled_memory", "null_memory"), default="memory")
    p.add_argument("--max-examples", type=int)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    p.add_argument("--output", type=Path)
    p.add_argument("--config", type=Path)
    p.set_defaults(**load_config_defaults(config_path_from_argv()))
    args = p.parse_args()
    for name in ("checkpoint", "train_file", "evaluation_file", "output"):
        if getattr(args, name) is not None:
            setattr(args, name, Path(getattr(args, name)))
    return args


@torch.inference_mode()
def score_candidates(model, tokenizer, application_ids, memory, candidates, device):
    if not candidates or not application_ids:
        raise ValueError("Application prefix and candidate vocabulary must be nonempty")
    scores = {}
    prefix_len = len(application_ids)
    for candidate in candidates:
        suffix = tokenizer(candidate, add_special_tokens=False).input_ids
        if not suffix:
            raise ValueError(f"Empty candidate tokenization: {candidate!r}")
        ids = torch.tensor([application_ids + suffix], device=device)
        logits = model(ids, memory=memory, use_cache=False).logits[:, prefix_len - 1 : -1].float()
        target = torch.tensor([suffix], device=device)
        scores[candidate] = logits.log_softmax(-1).gather(-1, target.unsqueeze(-1)).sum().item()
    return max(scores, key=scores.get)


def shuffled_source_indices(examples):
    """Pick a deterministic wrong source with a different target answer."""
    by_answer = {}
    for index, example in enumerate(examples):
        by_answer.setdefault(example.answer, []).append(index)
    if len(by_answer) < 2:
        raise ValueError("shuffled_memory requires at least two distinct answer locations")
    return [next(j for j in range(len(examples)) if examples[j].answer != example.answer)
            for example in examples]


def main() -> None:
    a = parse_args()
    if a.checkpoint is None or a.train_file is None or a.evaluation_file is None:
        raise ValueError("checkpoint, train-file and evaluation-file are required")
    if a.max_examples is not None and a.max_examples <= 0:
        raise ValueError("max-examples must be positive")
    checkpoint = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("architecture") != "episodic_kv_reader":
        raise ValueError("Not an episodic KV reader checkpoint")
    if Path(checkpoint["train_file"]).resolve() != a.train_file.resolve():
        raise ValueError("Training vocabulary source differs from checkpoint")
    train = load_babi_task(a.train_file)
    test = load_babi_task(a.evaluation_file)
    vocabulary = answer_vocabulary(train)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint["model"])
    if tokenizer.eos_token_id is None:
        raise ValueError("Tokenizer requires eos_token_id")
    device = torch.device(a.device)
    qwen = AutoModelForCausalLM.from_pretrained(checkpoint["model"], torch_dtype=getattr(torch, a.dtype)).to(device)
    model = FrozenQwenWithEpisodicKV(qwen, EpisodicConfig(**checkpoint["config"])).eval()
    model.load_trainable_state_dict(checkpoint["trainable_state_dict"])
    formatter = QwenChatPromptFormatter(tokenizer)
    output = {}
    for split in (("train", "test") if a.split == "both" else (a.split,)):
        selected = train if split == "train" else test
        if a.max_examples is not None:
            selected = selected[:a.max_examples]
        if not selected:
            raise ValueError(f"No examples selected for {split}")
        correct = 0
        wrong_indices = shuffled_source_indices(selected) if a.variant == "shuffled_memory" else None
        for i, example in enumerate(selected):
            source_example = selected[wrong_indices[i]] if wrong_indices is not None else example
            if a.variant == "null_memory":
                memory = None
            else:
                source = torch.tensor([tokenizer(formatter(render_source_episode(source_example)),
                                                 add_special_tokens=True).input_ids], device=device)
                memory = model.encode(source)
            app_ids = tokenizer(formatter(render_application_episode(example)), add_special_tokens=True).input_ids
            predicted = score_candidates(model, tokenizer, app_ids, memory, vocabulary, device)
            correct += predicted == example.answer
        output[split] = {"examples": len(selected), "correct": correct, "accuracy": correct / len(selected)}
    payload = {"model": checkpoint["model"], "checkpoint": str(a.checkpoint), "step": checkpoint["step"],
               "effective_injection_indices": list(model.indices),
               "variant": a.variant, "candidate_vocabulary": list(vocabulary), "splits": output}
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    print(text, end="")
    if a.output:
        a.output.parent.mkdir(parents=True, exist_ok=True)
        a.output.write_text(text, encoding="utf-8")
    model.close()


if __name__ == "__main__":
    main()
