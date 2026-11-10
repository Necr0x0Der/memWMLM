#!/usr/bin/env python3
"""Run resumable five-shot MMLU candidate-likelihood evaluation for Qwen."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

from memwmlm.models.qwen import QwenChatPromptFormatter
from mmlu import CHOICES, load_few_shots, load_mmlu, render_task, write_summary


PROMPT_VERSION = "mmlu_official_5shot_chat_v1"
SCORING_VERSION = "candidate_log_likelihood_sum_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True, help="Immutable Hugging Face revision SHA")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--n-shot", type=int, default=5)
    parser.add_argument("--max-examples-per-subject", type=int)
    parser.add_argument("--batch-size", type=int, default=1, help="Questions per model forward pass")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("float16", "bfloat16", "float32"), default="bfloat16")
    return parser.parse_args()


def git_revision() -> str | None:
    result = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else None


def dataset_manifest(data_dir: Path) -> dict[str, str]:
    manifest = {}
    for path in sorted((*data_dir.glob("dev/*_dev.csv"), *data_dir.glob("test/*_test.csv"))):
        manifest[str(path.relative_to(data_dir))] = hashlib.sha256(path.read_bytes()).hexdigest()
    transform_manifest = data_dir / "permutation-manifest.json"
    if transform_manifest.exists():
        manifest[transform_manifest.name] = hashlib.sha256(transform_manifest.read_bytes()).hexdigest()
    if not manifest:
        raise ValueError(f"No MMLU dev/test files found under {data_dir}")
    return manifest


def read_completed(path: Path) -> set[str]:
    completed = set()
    if not path.exists():
        return completed
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        record = json.loads(line)
        example_id = record.get("example_id")
        if not isinstance(example_id, str) or example_id in completed:
            raise ValueError(f"Invalid or duplicate example_id in {path}:{line_number}")
        completed.add(example_id)
    return completed


def select_examples(examples: list, per_subject: int | None) -> list:
    if per_subject is None:
        return examples
    if per_subject <= 0:
        raise ValueError("--max-examples-per-subject must be positive")
    counts: dict[str, int] = {}
    selected = []
    for example in examples:
        count = counts.get(example.subject, 0)
        if count < per_subject:
            selected.append(example)
            counts[example.subject] = count + 1
    return selected


def score_candidates(model, tokenizer, prompts: list[str], device: str) -> list[list[float]]:
    """Return summed conditional log likelihood for A/B/C/D for every prompt."""
    rows = []
    spans = []
    for prompt in prompts:
        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        if not prompt_ids:
            raise ValueError("Tokenizer produced an empty prompt")
        for candidate in CHOICES:
            candidate_ids = tokenizer(candidate, add_special_tokens=False)["input_ids"]
            if not candidate_ids:
                raise ValueError(f"Tokenizer produced no tokens for candidate {candidate}")
            rows.append({"input_ids": prompt_ids + candidate_ids})
            spans.append((len(prompt_ids), len(candidate_ids)))

    batch = tokenizer.pad(rows, padding=True, return_tensors="pt").to(device)
    with torch.inference_mode():
        logits = model(**batch).logits.log_softmax(dim=-1)

    sequence_length = batch["input_ids"].shape[1]
    flat_scores = []
    for row_index, (prompt_length, candidate_length) in enumerate(spans):
        unpadded_length = prompt_length + candidate_length
        left_padding = sequence_length - unpadded_length
        score = logits.new_zeros(())
        for offset in range(candidate_length):
            token_position = left_padding + prompt_length + offset
            token_id = batch["input_ids"][row_index, token_position]
            score = score + logits[row_index, token_position - 1, token_id]
        flat_scores.append(score)
    host_scores = torch.stack(flat_scores).tolist()
    return [host_scores[index : index + 4] for index in range(0, len(host_scores), 4)]


def main() -> None:
    args = parse_args()
    if args.n_shot < 0 or args.n_shot > 5:
        raise ValueError("--n-shot must be between 0 and 5")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    torch.manual_seed(args.seed)

    examples = select_examples(load_mmlu(args.data_dir), args.max_examples_per_subject)
    selected_ids = {example.example_id for example in examples}
    completed = read_completed(args.output)
    if not completed <= selected_ids:
        raise ValueError("Output contains IDs outside the selected evaluation slice")

    dtype = getattr(torch, args.dtype)
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision, local_files_only=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, revision=args.revision, dtype=dtype, local_files_only=True
    ).to(args.device)
    model.eval()
    formatter = QwenChatPromptFormatter(tokenizer)

    metadata = {
        "model": args.model,
        "tokenizer": args.model,
        "requested_revision": args.revision,
        "requested_tokenizer_revision": args.revision,
        "resolved_model_revision": getattr(model.config, "_commit_hash", None),
        "resolved_tokenizer_revision": tokenizer.init_kwargs.get("_commit_hash"),
        "prompt_version": PROMPT_VERSION,
        "scoring_version": SCORING_VERSION,
        "candidate_strings": list(CHOICES),
        "enable_thinking": False,
        "n_shot": args.n_shot,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "stochastic_sampling": False,
        "device": args.device,
        "dtype": args.dtype,
        "quantization": None,
        "dataset": "cais/mmlu",
        "dataset_revision": "c30699e8356da336a370243923dbaf21066bb9fe",
        "dataset_files_sha256": dataset_manifest(args.data_dir),
        "split": "test",
        "selected_examples": len(examples),
        "repository_commit": git_revision(),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
    }
    metadata_path = args.output.with_suffix(args.output.suffix + ".meta.json")
    if metadata_path.exists():
        if json.loads(metadata_path.read_text(encoding="utf-8")) != metadata:
            raise ValueError(f"Run metadata differs from existing {metadata_path}")
    else:
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    pending = [example for example in examples if example.example_id not in completed]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    with args.output.open("a", encoding="utf-8") as output:
        for start in range(0, len(pending), args.batch_size):
            items = pending[start : start + args.batch_size]
            prompts = [
                formatter(render_task(item, load_few_shots(args.data_dir, item.subject, args.n_shot)))
                for item in items
            ]
            scores = score_candidates(model, tokenizer, prompts, args.device)
            for item, candidate_scores in zip(items, scores, strict=True):
                prediction = CHOICES[max(range(4), key=candidate_scores.__getitem__)]
                record = {
                    "example_id": item.example_id,
                    "subject": item.subject,
                    "domain": item.domain,
                    "question": item.question,
                    "choices": list(item.choices),
                    "answer": item.answer,
                    "prediction": prediction,
                    "candidate_log_likelihoods": dict(zip(CHOICES, candidate_scores)),
                    "correct": prediction == item.answer,
                }
                output.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
                output.flush()

    summary_path = write_summary(args.output)
    elapsed = time.perf_counter() - started
    peak = torch.cuda.max_memory_allocated() if args.device.startswith("cuda") else None
    print(json.dumps({"new_examples": len(pending), "seconds": elapsed, "peak_vram_bytes": peak, "summary": str(summary_path)}))


if __name__ == "__main__":
    main()
