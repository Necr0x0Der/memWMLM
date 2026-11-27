#!/usr/bin/env python3
"""Train a frozen Qwen3-1.7B plus a position-aligned memory sidecar on
synthetic multi-commit versioned facts."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from memwmlm.models.memory_qwen import FrozenQwenWithMemory, MemoryQwenConfig

from common import (
    concat_memory,
    load_records,
    make_multicommit_batch,
    APPLICATION_INSTRUCTION,
)
from memwmlm.models.qwen import QwenChatPromptFormatter
from memwmlm.training.reset_pair import causal_answer_loss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-file", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--steps", type=int, default=2500)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--memory-width", type=int, default=1024)
    parser.add_argument("--writer-heads", type=int, default=8)
    parser.add_argument("--reader-heads", type=int, default=8)
    parser.add_argument("--injection-indices", type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument("--gate-bias-init", type=float, default=-6.0)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--source-lora-rank", type=int, default=8)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    return parser.parse_args()


def torch_dtype(name: str) -> torch.dtype:
    return getattr(torch, name)


def save_checkpoint(
    path: Path,
    model: FrozenQwenWithMemory,
    args: argparse.Namespace,
    step: int,
    optimizer: torch.optim.Optimizer,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": 1,
        "prompt_version": "multicommit_v1",
        "model": str(args.model_dir),
        "memory_config": {
            "memory_width": args.memory_width,
            "writer_heads": args.writer_heads,
            "reader_heads": args.reader_heads,
            "injection_layers": len(args.injection_indices),
            "injection_indices": list(model.layer_indices),
            "gate_bias_init": args.gate_bias_init,
            "lora_rank": args.lora_rank,
            "lora_alpha": 16.0,
            "lora_dropout": 0.0,
            "source_lora_rank": args.source_lora_rank,
            "source_lora_alpha": 16.0,
            "source_lora_dropout": 0.0,
        },
        "training_config": {
            "seed": args.seed,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "steps": args.steps,
        },
        "step": step,
        "sidecar_state_dict": model.trainable_state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    torch.save(payload, path)


def main() -> None:
    args = parse_args()
    if args.steps <= 0 or args.batch_size <= 0:
        raise ValueError("steps and batch-size must be positive")
    if not torch.cuda.is_available() and args.device.startswith("cuda"):
        raise RuntimeError("CUDA is unavailable")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    records = load_records(args.train_file)
    if not records:
        raise ValueError("No training records")

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    qwen = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        torch_dtype=torch_dtype(args.dtype),
        local_files_only=True,
    )
    qwen.to(device)

    model = FrozenQwenWithMemory(
        qwen,
        MemoryQwenConfig(
            memory_width=args.memory_width,
            writer_heads=args.writer_heads,
            reader_heads=args.reader_heads,
            injection_layers=len(args.injection_indices),
            injection_indices=tuple(args.injection_indices),
            gate_bias_init=args.gate_bias_init,
            lora_rank=args.lora_rank,
            lora_alpha=16.0,
            lora_dropout=0.0,
            source_lora_rank=args.source_lora_rank,
            source_lora_alpha=16.0,
            source_lora_dropout=0.0,
        ),
    ).to(device)
    model.train()

    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate, weight_decay=args.weight_decay)
    formatter = QwenChatPromptFormatter(tokenizer)
    rng = random.Random(args.seed)

    print(
        json.dumps(
            {
                "event": "model_initialized",
                "memory_config": model.config.__dict__,
                "effective_injection_indices": list(model.layer_indices),
                "trainable_parameters": model.trainable_parameter_summary(),
            },
            sort_keys=True,
        ),
        flush=True,
    )

    for step in range(1, args.steps + 1):
        selected = rng.choices(records, k=args.batch_size)
        batch = make_multicommit_batch(selected, tokenizer, formatter, device)
        source_states = [
            model.write_memory(ids, mask)
            for ids, mask in zip(batch.source_input_ids, batch.source_attention_masks, strict=True)
        ]
        memory = concat_memory(source_states)
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
        torch.nn.utils.clip_grad_norm_(trainable, args.max_grad_norm)
        optimizer.step()

        if step == 1 or step % args.log_every == 0:
            print(json.dumps({"step": step, "loss": float(loss.detach())}, sort_keys=True), flush=True)
        if step % args.save_every == 0:
            save_checkpoint(args.output, model, args, step, optimizer)

    save_checkpoint(args.output, model, args, args.steps, optimizer)
    model.close()


if __name__ == "__main__":
    main()
