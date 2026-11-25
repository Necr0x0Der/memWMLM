#!/usr/bin/env python3
"""Evaluate multi-commit memory, controls, and context baselines."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from memwmlm.models.memory_qwen import FrozenQwenWithMemory, MemoryQwenConfig
from memwmlm.models.qwen import QwenChatPromptFormatter

from common import (
    APPLICATION_INSTRUCTION,
    counterfactual_documents,
    load_records,
    predict_raw,
    predict_with_memory,
    render_application,
    render_full_context,
    write_selected_documents,
    zero_memory,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--test-file", type=Path, required=True)
    parser.add_argument("--train-file", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "test"), default="test")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def torch_dtype(name: str) -> torch.dtype:
    return getattr(torch, name)


def summarize(predictions: list[str], records: list[dict]) -> dict:
    total = len(records)
    correct = sum(prediction == record["target"] for prediction, record in zip(predictions, records, strict=True))
    by_type: dict[str, dict[str, int]] = {}
    for prediction, record in zip(predictions, records, strict=True):
        qtype = record["question_type"]
        bucket = by_type.setdefault(qtype, {"examples": 0, "correct": 0})
        bucket["examples"] += 1
        bucket["correct"] += int(prediction == record["target"])
    return {
        "examples": total,
        "correct": correct,
        "accuracy": correct / total if total else 0.0,
        "by_question_type": {
            qtype: {"examples": value["examples"], "correct": value["correct"], "accuracy": value["correct"] / value["examples"]}
            for qtype, value in by_type.items()
        },
    }


def main() -> None:
    args = parse_args()
    if args.max_examples is not None and args.max_examples <= 0:
        raise ValueError("--max-examples must be positive")

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = MemoryQwenConfig.from_checkpoint(checkpoint["memory_config"])
    tokenizer = AutoTokenizer.from_pretrained(checkpoint["model"], local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    qwen = AutoModelForCausalLM.from_pretrained(
        checkpoint["model"],
        torch_dtype=torch_dtype(args.dtype),
        local_files_only=True,
    )
    qwen.to(args.device)
    model = FrozenQwenWithMemory(qwen, config).to(args.device)
    model.load_trainable_state_dict(checkpoint["sidecar_state_dict"])
    model.eval()
    formatter = QwenChatPromptFormatter(tokenizer)

    train_records = load_records(args.train_file)
    candidates = sorted({record["target"] for record in train_records})
    records = load_records(args.test_file if args.split == "test" else args.train_file)
    if args.max_examples is not None:
        records = records[: args.max_examples]

    print(
        json.dumps(
            {
                "event": "model_loaded",
                "checkpoint_step": checkpoint.get("step"),
                "memory_config": model.config.__dict__,
                "effective_injection_indices": list(model.layer_indices),
                "trainable_parameters": model.trainable_parameter_summary(),
                "split": args.split,
                "records": len(records),
                "candidates": candidates,
            },
            sort_keys=True,
        ),
        flush=True,
    )

    # Build full multi-commit memory for every record.
    memories = [
        write_selected_documents(model, tokenizer, formatter, record, [0, 1, 2])
        for record in records
    ]

    predictions_memory = [
        predict_with_memory(model, tokenizer, formatter(render_application(record)), candidates, memory)
        for record, memory in zip(records, memories, strict=True)
    ]

    predictions_null = []
    predictions_shuffled = []
    predictions_last_commit = []
    for index, record in enumerate(records):
        predictions_null.append(
            predict_with_memory(
                model, tokenizer, formatter(render_application(record)), candidates,
                zero_memory(memories[index]),
            )
        )
        shuffled_memory = memories[(index + 1) % len(memories)]
        predictions_shuffled.append(
            predict_with_memory(
                model, tokenizer, formatter(render_application(record)), candidates,
                shuffled_memory,
            )
        )
        last_memory = write_selected_documents(model, tokenizer, formatter, record, [2])
        predictions_last_commit.append(
            predict_with_memory(
                model, tokenizer, formatter(render_application(record)), candidates,
                last_memory,
            )
        )

    predictions_full_context = [
        predict_raw(qwen, tokenizer, formatter(render_full_context(record)), candidates)
        for record in records
    ]
    predictions_no_source = [
        predict_raw(qwen, tokenizer, formatter(render_application(record)), candidates)
        for record in records
    ]

    # Counterfactual memory: replace each record's target city in its source text.
    rng = random.Random(args.seed)
    counterfactual_follow = 0
    counterfactual_resist = 0
    counterfactual_other = 0
    for record in records:
        used = set(record["metadata"]["cities_used"])
        available = [city for city in candidates if city not in used]
        if not available:
            raise ValueError(f"No counterfactual city available for {record['example_id']}")
        swap = rng.choice(available)
        modified = dict(record)
        modified["source_documents"] = counterfactual_documents(record, swap)
        memory = write_selected_documents(model, tokenizer, formatter, modified, [0, 1, 2])
        prediction = predict_with_memory(
            model, tokenizer, formatter(render_application(modified)), candidates, memory
        )
        if prediction == swap:
            counterfactual_follow += 1
        elif prediction == record["target"]:
            counterfactual_resist += 1
        else:
            counterfactual_other += 1

    variants = {
        "memory": summarize(predictions_memory, records),
        "null_memory": summarize(predictions_null, records),
        "shuffled_memory": summarize(predictions_shuffled, records),
        "last_commit_only": summarize(predictions_last_commit, records),
        "full_context": summarize(predictions_full_context, records),
        "no_source": summarize(predictions_no_source, records),
    }
    counterfactual = {
        "examples": len(records),
        "follow": counterfactual_follow,
        "follow_rate": counterfactual_follow / len(records),
        "resist": counterfactual_resist,
        "resist_rate": counterfactual_resist / len(records),
        "other": counterfactual_other,
    }

    payload = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_step": checkpoint.get("step"),
        "split": args.split,
        "model": checkpoint["model"],
        "prompt_version": checkpoint.get("prompt_version"),
        "effective_injection_indices": list(model.layer_indices),
        "examples": len(records),
        "variants": variants,
        "counterfactual_memory": counterfactual,
    }

    print(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")

    model.close()


if __name__ == "__main__":
    main()
