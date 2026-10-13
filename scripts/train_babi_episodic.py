#!/usr/bin/env python3
"""Train a reader of uncompressed source-token layerwise KV on bAbI Task 1."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from memwmlm.data.babi import load_babi_task
from memwmlm.models.episodic_qwen import EpisodicConfig, FrozenQwenWithEpisodicKV
from memwmlm.models.qwen import QwenChatPromptFormatter
from memwmlm.tasks.babi_memory import render_application_episode, render_source_episode
from memwmlm.training.reset_pair import causal_answer_loss
from memwmlm.utils.config import config_path_from_argv, load_config_defaults


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--train-file", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--injection-layers", type=int, default=4)
    parser.add_argument("--reader-heads", type=int, default=8)
    parser.add_argument("--gate-bias-init", type=float, default=-6.0)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=float, default=16.0)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--save-every", type=int, default=500)
    parser.set_defaults(**load_config_defaults(config_path_from_argv()))
    args = parser.parse_args()
    for name in ("train_file", "output"):
        if getattr(args, name) is not None:
            setattr(args, name, Path(getattr(args, name)))
    return args


def make_training_ids(tokenizer, formatter, example, device):
    source_ids = tokenizer(formatter(render_source_episode(example)), add_special_tokens=True).input_ids
    prefix_ids = tokenizer(formatter(render_application_episode(example)), add_special_tokens=True).input_ids
    target_ids = tokenizer(example.answer, add_special_tokens=False).input_ids + [tokenizer.eos_token_id]
    if not prefix_ids or not target_ids:
        raise ValueError("Empty application prefix or target")
    application_ids = prefix_ids + target_ids
    labels = [-100] * len(prefix_ids) + target_ids
    return (torch.tensor([source_ids], device=device),
            torch.tensor([application_ids], device=device),
            torch.tensor([labels], device=device))


def main() -> None:
    args = parse_args()
    if args.train_file is None or args.output is None:
        raise ValueError("train-file and output are required")
    if min(args.steps, args.log_every, args.save_every) <= 0 or args.lora_rank < 0:
        raise ValueError("steps/log-every/save-every must be positive and LoRA rank >= 0")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    examples = load_babi_task(args.train_file)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.eos_token_id is None:
        raise ValueError("Tokenizer requires eos_token_id")
    qwen = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=getattr(torch, args.dtype)).to(device)
    config = EpisodicConfig(args.injection_layers, args.reader_heads, args.gate_bias_init,
                            args.lora_rank, args.lora_alpha)
    model = FrozenQwenWithEpisodicKV(qwen, config).train()
    formatter = QwenChatPromptFormatter(tokenizer)
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate)
    rng = random.Random(args.seed)
    print(json.dumps({"event": "model_initialized", "configuration": vars(args),
                      "trainable_parameters": sum(p.numel() for p in parameters)}, default=str), flush=True)
    for step in range(1, args.steps + 1):
        example = rng.choices(examples, k=1)[0]
        source_ids, application_ids, labels = make_training_ids(tokenizer, formatter, example, device)
        memory = model.encode(source_ids)
        outputs = model(application_ids, memory=memory, use_cache=False)
        loss = causal_answer_loss(outputs.logits, labels)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        optimizer.step()
        if step == 1 or step % args.log_every == 0:
            print(json.dumps({"step": step, "loss": float(loss.detach())}), flush=True)
        if step % args.save_every == 0 or step == args.steps:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"format_version": 1, "architecture": "episodic_kv_reader",
                        "model": args.model, "config": config.__dict__, "step": step,
                        "seed": args.seed, "train_file": str(args.train_file),
                        "trainable_state_dict": model.trainable_state_dict(),
                        "optimizer_state_dict": optimizer.state_dict()}, args.output)
    model.close()


if __name__ == "__main__":
    main()
