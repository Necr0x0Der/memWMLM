#!/usr/bin/env python3
"""Train reset-pair writable embedding memory on bAbI Task 1."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from memwmlm.data.babi import load_babi_task
from memwmlm.models.memory_qwen import FrozenQwenWithMemory, MemoryQwenConfig
from memwmlm.models.qwen import QwenChatPromptFormatter
from memwmlm.training.reset_pair import causal_answer_loss, make_reset_pair_batch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--train-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="Output .pt checkpoint path.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--steps", type=int, default=2_000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--memory-cells", type=int, default=32)
    parser.add_argument("--memory-width", type=int, default=512)
    parser.add_argument("--writer-heads", type=int, default=8)
    parser.add_argument("--reader-heads", type=int, default=8)
    parser.add_argument("--injection-layers", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--save-every", type=int, default=500)
    return parser.parse_args()


def torch_dtype(name: str) -> torch.dtype:
    return getattr(torch, name)


def checkpoint_payload(
    model: FrozenQwenWithMemory,
    args: argparse.Namespace,
    step: int,
    optimizer: torch.optim.Optimizer,
) -> dict[str, object]:
    return {
        "format_version": 1,
        "model": args.model,
        "memory_config": {
            "memory_width": args.memory_width,
            "memory_cells": args.memory_cells,
            "writer_heads": args.writer_heads,
            "reader_heads": args.reader_heads,
            "injection_layers": args.injection_layers,
        },
        "training_config": {
            "seed": args.seed,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
        },
        "step": step,
        "sidecar_state_dict": model.trainable_state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }


def save_checkpoint(
    path: Path,
    model: FrozenQwenWithMemory,
    args: argparse.Namespace,
    step: int,
    optimizer: torch.optim.Optimizer,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint_payload(model, args, step, optimizer), path)


def main() -> None:
    args = parse_args()
    if args.steps <= 0 or args.batch_size <= 0 or args.log_every <= 0:
        raise ValueError("steps, batch-size, and log-every must be positive")
    if not torch.cuda.is_available() and args.device.startswith("cuda"):
        raise RuntimeError("CUDA is unavailable; pass --device cpu only for a small debug run")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    examples = load_babi_task(args.train_file)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    qwen = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch_dtype(args.dtype))
    qwen.to(device)
    model = FrozenQwenWithMemory(
        qwen,
        MemoryQwenConfig(
            memory_width=args.memory_width,
            memory_cells=args.memory_cells,
            writer_heads=args.writer_heads,
            reader_heads=args.reader_heads,
            injection_layers=args.injection_layers,
        ),
    ).to(device)
    model.train()
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.learning_rate,
    )
    formatter = QwenChatPromptFormatter(tokenizer)
    rng = random.Random(args.seed)

    for step in range(1, args.steps + 1):
        selected = rng.choices(examples, k=args.batch_size)
        batch = make_reset_pair_batch(selected, tokenizer, formatter, device)
        memory = model.write_memory(batch.source_input_ids, batch.source_attention_mask)
        outputs = model(
            input_ids=batch.application_input_ids,
            attention_mask=batch.application_attention_mask,
            memory=memory,
            use_cache=False,
            return_dict=True,
        )
        loss = causal_answer_loss(outputs.logits, batch.labels)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step == 1 or step % args.log_every == 0:
            print(json.dumps({"step": step, "loss": float(loss.detach())}, sort_keys=True), flush=True)
        if step % args.save_every == 0:
            save_checkpoint(args.output, model, args, step, optimizer)

    save_checkpoint(args.output, model, args, args.steps, optimizer)
    model.close()


if __name__ == "__main__":
    main()
