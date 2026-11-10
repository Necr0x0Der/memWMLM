#!/usr/bin/env python3
"""Run resumable no-context Qwen inference on TriviaQA."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from memwmlm.models.qwen import QwenChatPromptFormatter
from triviaqa import load_triviaqa, normalize_answer, score_prediction


PROMPT_VERSION = "triviaqa_short_answer_v1"
INSTRUCTION = "Answer the factual question with only the shortest answer. Do not explain."


def render_task(question: str) -> str:
    return f"{INSTRUCTION}\n\nQuestion: {question}\nAnswer:"


def read_completed(path: Path) -> set[str]:
    completed = set()
    if not path.exists():
        return completed
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        record = json.loads(line)
        question_id = record.get("question_id")
        if not isinstance(question_id, str):
            raise ValueError(f"Invalid question_id in {path}:{line_number}")
        if question_id in completed:
            raise ValueError(f"Duplicate question_id in {path}: {question_id}")
        completed.add(question_id)
    return completed


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True, help="Immutable Hugging Face revision SHA")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-examples", type=int)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("float16", "bfloat16", "float32"), default="bfloat16")
    return parser.parse_args()


def git_revision() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else None


def main() -> None:
    args = parse_args()
    if args.max_examples is not None and args.max_examples <= 0:
        raise ValueError("--max-examples must be positive")
    if args.batch_size <= 0 or args.max_new_tokens <= 0:
        raise ValueError("--batch-size and --max-new-tokens must be positive")

    examples = load_triviaqa(args.dataset)
    if args.max_examples is not None:
        examples = examples[: args.max_examples]
    completed = read_completed(args.output)
    selected_ids = {example.question_id for example in examples}
    if not completed <= selected_ids:
        raise ValueError("Output contains QuestionIds outside the selected evaluation slice")

    dtype = getattr(torch, args.dtype)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, revision=args.revision, local_files_only=True
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        revision=args.revision,
        dtype=dtype,
        local_files_only=True,
    ).to(args.device)
    model.eval()
    formatter = QwenChatPromptFormatter(tokenizer)

    metadata = {
        "model": args.model,
        "requested_revision": args.revision,
        "resolved_model_revision": getattr(model.config, "_commit_hash", None),
        "prompt_version": PROMPT_VERSION,
        "instruction": INSTRUCTION,
        "enable_thinking": False,
        "do_sample": False,
        "max_new_tokens": args.max_new_tokens,
        "batch_size": args.batch_size,
        "device": args.device,
        "dtype": args.dtype,
        "quantization": None,
        "dataset": str(args.dataset),
        "dataset_sha256": sha256(args.dataset),
        "selected_examples": len(examples),
        "repository_commit": git_revision(),
        "torch_version": torch.__version__,
    }
    metadata_path = args.output.with_suffix(args.output.suffix + ".meta.json")
    if metadata_path.exists():
        previous = json.loads(metadata_path.read_text(encoding="utf-8"))
        if previous != metadata:
            raise ValueError(f"Run metadata differs from existing {metadata_path}")
    else:
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    pending = [example for example in examples if example.question_id not in completed]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    with args.output.open("a", encoding="utf-8") as output:
        for start in range(0, len(pending), args.batch_size):
            batch = pending[start : start + args.batch_size]
            prompts = [formatter(render_task(example.question)) for example in batch]
            inputs = tokenizer(prompts, padding=True, return_tensors="pt").to(args.device)
            with torch.inference_mode():
                generated = model.generate(
                    **inputs,
                    do_sample=False,
                    max_new_tokens=args.max_new_tokens,
                    pad_token_id=tokenizer.pad_token_id,
                )
            prefix_length = inputs["input_ids"].shape[1]
            for example, token_ids in zip(batch, generated[:, prefix_length:], strict=True):
                prediction = tokenizer.decode(token_ids, skip_special_tokens=True).strip()
                exact, f1 = score_prediction(prediction, example.ground_truths)
                record = {
                    "question_id": example.question_id,
                    "question": example.question,
                    "prediction": prediction,
                    "normalized_prediction": normalize_answer(prediction),
                    "ground_truths": list(example.ground_truths),
                    "exact_match": exact,
                    "f1": f1,
                    "generated_tokens": int(token_ids.ne(tokenizer.pad_token_id).sum()),
                }
                output.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
                output.flush()

    elapsed = time.perf_counter() - started
    peak = torch.cuda.max_memory_allocated() if args.device.startswith("cuda") else None
    print(json.dumps({"new_examples": len(pending), "seconds": elapsed, "peak_vram_bytes": peak}))


if __name__ == "__main__":
    main()
