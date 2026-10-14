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
from memwmlm.utils.config import config_path_from_argv, load_config_defaults


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None, help="Flat YAML defaults; CLI flags override it.")
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--train-file", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None, help="Output .pt checkpoint path.")
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
    parser.add_argument(
        "--injection-indices", type=int, nargs="+", default=None,
        help="0-based decoder blocks for post-block reader injection; overrides --injection-layers.",
    )
    parser.add_argument("--gate-bias-init", type=float, default=-6.0)
    parser.add_argument(
        "--lora-rank",
        type=int,
        default=0,
        help="Application-only LoRA rank on q_proj/o_proj of injection layers; 0 disables it.",
    )
    parser.add_argument("--lora-alpha", type=float, default=16.0)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument(
        "--source-lora-rank",
        type=int,
        default=0,
        help="Source-only LoRA rank on q_proj/o_proj; 0 disables source adaptation.",
    )
    parser.add_argument("--source-lora-alpha", type=float, default=16.0)
    parser.add_argument("--source-lora-dropout", type=float, default=0.0)
    parser.add_argument("--adam-beta1", type=float, default=0.9)
    parser.add_argument("--adam-beta2", type=float, default=0.999)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=None,
        help="Clip global sidecar gradient norm when positive; null disables clipping.",
    )
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--save-every", type=int, default=500)
    parser.set_defaults(**load_config_defaults(config_path_from_argv()))
    args = parser.parse_args()
    if args.train_file is not None:
        args.train_file = Path(args.train_file)
    if args.output is not None:
        args.output = Path(args.output)
    return args


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
            "injection_indices": list(model.layer_indices) if args.injection_indices is not None else None,
            "gate_bias_init": args.gate_bias_init,
            "lora_rank": args.lora_rank,
            "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout,
            "source_lora_rank": args.source_lora_rank,
            "source_lora_alpha": args.source_lora_alpha,
            "source_lora_dropout": args.source_lora_dropout,
        },
        "training_config": {
            "seed": args.seed,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "adam_beta1": args.adam_beta1,
            "adam_beta2": args.adam_beta2,
            "weight_decay": args.weight_decay,
            "max_grad_norm": args.max_grad_norm,
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
    if args.train_file is None or args.output is None:
        raise ValueError("--train-file and --output are required, directly or through --config")
    if args.steps <= 0 or args.batch_size <= 0 or args.log_every <= 0:
        raise ValueError("steps, batch-size, and log-every must be positive")
    if (
        args.lora_rank < 0
        or args.source_lora_rank < 0
        or args.lora_alpha <= 0
        or args.source_lora_alpha <= 0
        or not 0 <= args.lora_dropout < 1
        or not 0 <= args.source_lora_dropout < 1
    ):
        raise ValueError("LoRA requires ranks >= 0, alpha > 0, and dropout in [0, 1)")
    if not 0 <= args.adam_beta1 < 1 or not 0 <= args.adam_beta2 < 1:
        raise ValueError("Adam beta values must be in [0, 1)")
    if args.weight_decay < 0:
        raise ValueError("weight-decay must be non-negative")
    if args.max_grad_norm is not None and args.max_grad_norm <= 0:
        raise ValueError("max-grad-norm must be positive when set")
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
            injection_indices=tuple(args.injection_indices) if args.injection_indices is not None else None,
            gate_bias_init=args.gate_bias_init,
            lora_rank=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            source_lora_rank=args.source_lora_rank,
            source_lora_alpha=args.source_lora_alpha,
            source_lora_dropout=args.source_lora_dropout,
        ),
    ).to(device)
    model.train()
    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.weight_decay,
    )
    formatter = QwenChatPromptFormatter(tokenizer)
    rng = random.Random(args.seed)
    print(
        json.dumps(
            {
                "event": "model_initialized",
                "memory_config": model.config.__dict__,
                "effective_injection_indices": list(model.layer_indices),
                "lora": model.lora_summary(),
                "trainable_parameters": model.trainable_parameter_summary(),
            },
            sort_keys=True,
        ),
        flush=True,
    )

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
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable_parameters, args.max_grad_norm) if args.max_grad_norm is not None else _grad_norm(trainable_parameters)
        optimizer.step()

        if step == 1 or step % args.log_every == 0:
            print(
                json.dumps(
                    {"step": step, "loss": float(loss.detach()), "grad_norm": float(grad_norm)},
                    sort_keys=True,
                ),
                flush=True,
            )
        if step % args.save_every == 0:
            save_checkpoint(args.output, model, args, step, optimizer)

    save_checkpoint(args.output, model, args, args.steps, optimizer)
    model.close()


def _grad_norm(parameters: list[torch.nn.Parameter]) -> torch.Tensor:
    norms = [parameter.grad.detach().norm() for parameter in parameters if parameter.grad is not None]
    return torch.linalg.vector_norm(torch.stack(norms)) if norms else torch.zeros(())


if __name__ == "__main__":
    main()
