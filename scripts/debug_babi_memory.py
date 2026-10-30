#!/usr/bin/env python3
"""Print source, application, generated continuation, and candidate scores for bAbI memory examples."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from memwmlm.data.babi import answer_vocabulary, load_babi_task
from memwmlm.evaluation.memory_babi import _candidate_log_likelihood, _write_one
from memwmlm.memory.embedding_memory import MemoryState
from memwmlm.models.memory_qwen import FrozenQwenWithMemory, MemoryQwenConfig
from memwmlm.models.qwen import QwenChatPromptFormatter
from memwmlm.tasks.babi import prompt_instruction
from memwmlm.tasks.babi_memory import render_application_episode, render_source_episode
from memwmlm.training.reset_pair import _pad_id
from memwmlm.utils.config import config_path_from_argv, load_config_defaults


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None, help="Flat YAML defaults; CLI flags override it.")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--train-file", type=Path, default=None)
    parser.add_argument("--evaluation-file", type=Path, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--memory-variant", choices=("memory", "null_memory"), default="memory")
    parser.add_argument("--examples", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.set_defaults(**load_config_defaults(config_path_from_argv()))
    args = parser.parse_args()
    for name in ("checkpoint", "train_file", "evaluation_file"):
        value = getattr(args, name)
        if value is not None:
            setattr(args, name, Path(value))
    return args


def load_model(args: argparse.Namespace) -> tuple[FrozenQwenWithMemory, object, dict[str, object]]:
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint["model"])
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    qwen = AutoModelForCausalLM.from_pretrained(
        checkpoint["model"], torch_dtype=getattr(torch, args.dtype)
    )
    qwen.to(args.device)
    model = FrozenQwenWithMemory(qwen, MemoryQwenConfig.from_checkpoint(checkpoint["memory_config"])).to(args.device)
    model.load_trainable_state_dict(checkpoint["sidecar_state_dict"])
    model.eval()
    print(
        json.dumps(
            {
                "event": "model_loaded",
                "checkpoint_step": checkpoint["step"],
                "memory_config": model.config.__dict__,
                "effective_injection_indices": list(model.layer_indices),
                "lora": model.lora_summary(),
                "trainable_parameters": model.trainable_parameter_summary(),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return model, tokenizer, checkpoint


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    if args.checkpoint is None or args.train_file is None or args.evaluation_file is None:
        raise ValueError("--checkpoint, --train-file, and --evaluation-file are required directly or through --config")
    if args.examples <= 0 or args.max_new_tokens <= 0:
        raise ValueError("--examples and --max-new-tokens must be positive")

    model, tokenizer, checkpoint = load_model(args)
    prompt_version = checkpoint.get("prompt_version", "legacy_location")
    instruction = prompt_instruction(prompt_version)
    candidates = answer_vocabulary(load_babi_task(args.train_file))
    examples = load_babi_task(args.evaluation_file)
    selected = random.Random(args.seed).sample(examples, k=min(args.examples, len(examples)))
    formatter = QwenChatPromptFormatter(tokenizer)

    for index, example in enumerate(selected, 1):
        source = render_source_episode(example)
        application = render_application_episode(example, instruction)
        memory = _write_one(model, tokenizer, formatter(source))
        if args.memory_variant == "null_memory":
            memory = MemoryState(torch.zeros_like(memory.keys), torch.zeros_like(memory.values), mask=memory.mask)
        prefix = formatter(application)
        generation = generate_continuation(model, tokenizer, prefix, memory, args.max_new_tokens)
        candidate_scores = score_candidates(model, tokenizer, prefix, candidates, memory)
        print(json.dumps({
            "sample": index,
            "example_id": example.example_id,
            "story": list(example.story),
            "question": example.question,
            "target": example.answer,
            "generated": generation.text,
            "generated_token_ids": generation.token_ids,
            "generated_tokens": len(generation.token_ids),
            "terminated_with_eos": generation.terminated_with_eos,
            "generated_correct": normalize(generation.text) == normalize(example.answer),
            "candidate_ranking": candidate_scores,
            "candidate_prediction": candidate_scores[0]["candidate"],
            "candidate_correct": candidate_scores[0]["candidate"] == example.answer,
            "memory_variant": args.memory_variant,
            "checkpoint_step": checkpoint["step"],
            "prompt_version": prompt_version,
        }, ensure_ascii=False, indent=2), flush=True)
    model.close()


class GenerationTrace:
    def __init__(self, text: str, token_ids: list[int], eos_token_id: int | None) -> None:
        self.text = text
        self.token_ids = token_ids
        self.terminated_with_eos = bool(token_ids) and eos_token_id is not None and token_ids[-1] == eos_token_id


def generate_continuation(
    model: FrozenQwenWithMemory,
    tokenizer: object,
    prefix: str,
    memory: MemoryState,
    max_new_tokens: int,
) -> GenerationTrace:
    encoded = tokenizer(prefix, add_special_tokens=True, return_tensors="pt")
    encoded = {name: value.to(model.device) for name, value in encoded.items()}
    with model.read_memory(memory):
        generated = model.qwen.generate(
            **encoded,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            return_dict_in_generate=True,
        )
    token_ids = generated.sequences[0, encoded["input_ids"].shape[1] :].tolist()
    return GenerationTrace(
        text=tokenizer.decode(token_ids, skip_special_tokens=True),
        token_ids=token_ids,
        eos_token_id=tokenizer.eos_token_id,
    )


def score_candidates(
    model: FrozenQwenWithMemory,
    tokenizer: object,
    prefix: str,
    candidates: tuple[str, ...],
    memory: MemoryState,
) -> list[dict[str, float | str]]:
    prefix_ids = tokenizer(prefix, add_special_tokens=True).input_ids
    suffixes = [(candidate, tokenizer(candidate, add_special_tokens=False).input_ids) for candidate in candidates]
    max_length = max(len(prefix_ids) + len(tokens) for _, tokens in suffixes)
    input_ids = torch.full(
        (len(suffixes), max_length), _pad_id(tokenizer), dtype=torch.long, device=model.device
    )
    attention_mask = torch.zeros_like(input_ids)
    labels = torch.full_like(input_ids, -100)
    for row, (_, suffix) in enumerate(suffixes):
        sequence = prefix_ids + suffix
        input_ids[row, : len(sequence)] = torch.tensor(sequence, device=model.device)
        attention_mask[row, : len(sequence)] = 1
        labels[row, len(prefix_ids) : len(sequence)] = torch.tensor(suffix, device=model.device)
    repeated = MemoryState(
        keys=memory.keys.expand(len(suffixes), -1, -1),
        values=memory.values.expand(len(suffixes), -1, -1),
        mask=memory.mask.expand(len(suffixes), -1) if memory.mask is not None else None,
    )
    logits = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        memory=repeated,
        use_cache=False,
        return_dict=True,
    ).logits
    scores = _candidate_log_likelihood(logits, labels)
    ranked = sorted(zip(candidates, scores.tolist(), strict=True), key=lambda item: item[1], reverse=True)
    return [{"candidate": candidate, "log_likelihood": score} for candidate, score in ranked]


def normalize(text: str) -> str:
    return " ".join(text.strip().strip(".,;:!?`'\"“”").casefold().split())


if __name__ == "__main__":
    main()
