#!/usr/bin/env python3
"""Write an arbitrary source prompt to a trained bAbI memory, then generate a reply."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from memwmlm.models.memory_qwen import FrozenQwenWithMemory, MemoryQwenConfig
from memwmlm.models.qwen import QwenChatPromptFormatter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True, help="Pooled-memory checkpoint from train_babi_memory.py")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--source", help="Text to encode with the writer")
    source.add_argument("--source-file", type=Path, help="UTF-8 text file to encode with the writer")
    application = parser.add_mutually_exclusive_group(required=True)
    application.add_argument("--application", help="New prompt answered with access to the written memory")
    application.add_argument("--application-file", type=Path, help="UTF-8 application prompt file")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    return parser.parse_args()


def prompt_text(value: str | None, path: Path | None) -> str:
    text = path.read_text(encoding="utf-8") if path is not None else value
    if text is None or not text.strip():
        raise ValueError("Prompt must contain non-whitespace text")
    return text


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    if args.max_new_tokens <= 0:
        raise ValueError("--max-new-tokens must be positive")
    source = prompt_text(args.source, args.source_file)
    application = prompt_text(args.application, args.application_file)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; use --device cpu for a small smoke test")

    # Checkpoints are torch pickles: load only files from a trusted source.
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("format_version") != 1 or "sidecar_state_dict" not in checkpoint:
        raise ValueError("Expected a train_babi_memory.py pooled-memory checkpoint")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint["model"])
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    qwen = AutoModelForCausalLM.from_pretrained(
        checkpoint["model"], torch_dtype=getattr(torch, args.dtype)
    ).to(args.device)
    model = FrozenQwenWithMemory(qwen, MemoryQwenConfig(**checkpoint["memory_config"]))
    try:
        model.load_trainable_state_dict(checkpoint["sidecar_state_dict"])
        model.eval()
        formatter = QwenChatPromptFormatter(tokenizer)
        source_inputs = tokenizer(formatter(source), add_special_tokens=True, return_tensors="pt")
        source_inputs = {key: value.to(model.device) for key, value in source_inputs.items()}
        memory = model.write_memory(source_inputs["input_ids"], source_inputs["attention_mask"])
        # Application starts its own context: source tokens are not passed to generate().
        inputs = tokenizer(formatter(application), add_special_tokens=True, return_tensors="pt")
        inputs = {key: value.to(model.device) for key, value in inputs.items()}
        with model.read_memory(memory), model.application_lora(True):
            output = model.qwen.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=args.max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        continuation = output[0, inputs["input_ids"].shape[1]:]
        print(tokenizer.decode(continuation, skip_special_tokens=True))
    finally:
        model.close()


if __name__ == "__main__":
    main()
