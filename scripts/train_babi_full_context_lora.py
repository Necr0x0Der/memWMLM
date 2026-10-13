#!/usr/bin/env python3
"""Train a task-adapted full-context LoRA control for bAbI Task 1."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from memwmlm.data.babi import load_babi_task
from memwmlm.models.qwen import QwenChatPromptFormatter
from memwmlm.training.full_context_lora import FullContextLoRA, causal_answer_loss, make_full_context_batch
from memwmlm.utils.config import config_path_from_argv, load_config_defaults


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--train-file", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None, help="LoRA-only checkpoint .pt")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=float, default=16.0)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument("--lora-layers", type=int, default=4)
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--adam-beta1", type=float, default=0.9)
    parser.add_argument("--adam-beta2", type=float, default=0.999)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--save-every", type=int, default=500)
    parser.set_defaults(**load_config_defaults(config_path_from_argv()))
    args = parser.parse_args()
    for field in ("train_file", "output"):
        if getattr(args, field) is not None:
            setattr(args, field, Path(getattr(args, field)))
    return args


def validate(args: argparse.Namespace) -> None:
    if args.train_file is None or args.output is None:
        raise ValueError("--train-file and --output are required")
    if min(args.steps, args.batch_size, args.log_every, args.save_every, args.lora_rank, args.lora_layers) <= 0:
        raise ValueError("steps, batch-size, intervals, rank and layers must be positive")
    if args.lora_alpha <= 0 or not 0 <= args.lora_dropout < 1:
        raise ValueError("LoRA alpha must be positive and dropout in [0, 1)")
    if args.learning_rate <= 0 or args.weight_decay < 0 or any(not 0 <= beta < 1 for beta in (args.adam_beta1, args.adam_beta2)):
        raise ValueError("Invalid AdamW hyperparameters")
    if args.max_grad_norm is not None and args.max_grad_norm <= 0:
        raise ValueError("--max-grad-norm must be positive when set")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; use --device cpu for a small smoke test")


def save(path: Path, model: FullContextLoRA, args: argparse.Namespace, step: int, optimizer: torch.optim.Optimizer) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "format": "babi_full_context_lora_v1",
        "model": args.model,
        "adapter_config": {"rank": args.lora_rank, "alpha": args.lora_alpha, "dropout": args.lora_dropout, "layers": args.lora_layers},
        "training_config": {"train_file": str(args.train_file), "dtype": args.dtype, "seed": args.seed,
                            "batch_size": args.batch_size, "learning_rate": args.learning_rate,
                            "adam_beta1": args.adam_beta1, "adam_beta2": args.adam_beta2,
                            "weight_decay": args.weight_decay, "max_grad_norm": args.max_grad_norm},
        "step": step,
        "adapter_state_dict": model.adapter_state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }, path)


def main() -> None:
    args = parse_args()
    validate(args)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    examples = load_babi_task(args.train_file)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    device = torch.device(args.device)
    base = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=getattr(torch, args.dtype)).to(device)
    model = FullContextLoRA(base, args.lora_rank, args.lora_alpha, args.lora_dropout, args.lora_layers).train()
    trainable = model.trainable_parameters()
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate, betas=(args.adam_beta1, args.adam_beta2), weight_decay=args.weight_decay)
    formatter = QwenChatPromptFormatter(tokenizer)
    rng = random.Random(args.seed)
    print(json.dumps({"event": "model_initialized", "lora": model.summary(), "train_examples": len(examples)}), flush=True)
    for step in range(1, args.steps + 1):
        batch = make_full_context_batch(rng.choices(examples, k=args.batch_size), tokenizer, formatter, device)
        outputs = model(input_ids=batch.input_ids, attention_mask=batch.attention_mask, use_cache=False)
        loss = causal_answer_loss(outputs.logits, batch.labels)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if args.max_grad_norm is None:
            norms = [p.grad.detach().float().norm() for p in trainable if p.grad is not None]
            grad_norm = torch.linalg.vector_norm(torch.stack(norms)) if norms else torch.zeros(())
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable, args.max_grad_norm)
        optimizer.step()
        if step == 1 or step % args.log_every == 0:
            print(json.dumps({"step": step, "loss": float(loss.detach()), "grad_norm": float(grad_norm)}), flush=True)
        if step % args.save_every == 0:
            save(args.output, model, args, step, optimizer)
    save(args.output, model, args, args.steps, optimizer)
    print(json.dumps({"event": "checkpoint_saved", "step": args.steps, "path": str(args.output)}), flush=True)


if __name__ == "__main__":
    main()
