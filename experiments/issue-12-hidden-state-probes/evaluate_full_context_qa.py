#!/usr/bin/env python3
"""Ask frozen Qwen issue #12 questions with the full situation context.

For every fixed question, the model receives the complete situation and an
instruction to answer with exactly one allowed short answer through Qwen's chat
template with thinking disabled by default. Greedy generation is parsed from an explicit
``FINAL: <answer>`` marker. No probe is trained; ground truth is used only to
compute metrics.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import tempfile
import time
from functools import partial
from pathlib import Path
from typing import Mapping, Sequence

import torch
from torch import Tensor

from extract_hidden_features import (
    DEFAULT_MODEL,
    DEFAULT_REVISION,
    git_revision,
    load_frozen_model,
    storage_dtype,
    validate_dataset_manifest,
)
from train_linear_probes import (
    SPLITS,
    condition_metrics,
    encode_targets,
    group_bootstrap_intervals,
    read_jsonl,
)


RESULT_SCHEMA_VERSION = "spatial-full-context-final-answer-v3"
DEFAULT_SPLITS = ("validation", "test")
MODEL_REVISIONS = {
    DEFAULT_MODEL: DEFAULT_REVISION,
    "Qwen/Qwen3-4B": "1cfa9a7208912126459214e8b04321603b3df60c",
}
PROMPT_TEMPLATE = (
    "{situation}\n\nQuestion: {question}\n"
    "Return exactly one of the following lines and nothing else:\n{valid_responses}"
)


def format_qa_chat(task_text: str, tokenizer: object, enable_thinking: bool = False) -> str:
    """Apply this experiment's thinking setting without changing shared formatters."""
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": task_text}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )


def answer_text(value: object) -> str:
    if value == 0:
        return "No"
    if value == 1:
        return "Yes"
    if isinstance(value, str) and value:
        return value
    raise ValueError(f"Unsupported answer class: {value!r}")


def build_prompt(
    situation: str,
    question: str,
    answers: Sequence[str],
) -> str:
    if not situation or not question or len(answers) < 2:
        raise ValueError("Situation, question, and at least two answers are required")
    return PROMPT_TEMPLATE.format(
        situation=situation,
        question=question,
        valid_responses="\n".join(f"FINAL: {answer}" for answer in answers),
    )


def parse_short_answer(
    response: str, answers: Sequence[str], enable_thinking: bool = False,
) -> int | None:
    """Prefer the last FINAL marker; accept an unambiguous leading short answer."""
    if enable_thinking:
        # The opening <think> is already in the prompt. Never score reasoning,
        # including a FINAL marker produced before an exhausted token budget.
        if "</think>" not in response:
            return None
        response = response.rsplit("</think>", 1)[1]
    allowed = {answer.casefold(): index for index, answer in enumerate(answers)}
    final_matches = re.findall(
        r"(?im)^\s*(?:\*\*)?FINAL\s*:\s*(?:\*\*)?([A-Za-z]+)\b",
        response,
    )
    if final_matches:
        return allowed.get(final_matches[-1].casefold())
    leading = re.match(
        r"(?is)^\s*(?:Answer\s*:\s*)?(?:\*\*)?([A-Za-z]+)\b",
        response,
    )
    if leading is None:
        return None
    return allowed.get(leading.group(1).casefold())


def generate_answers(
    rows: Sequence[Mapping[str, object]],
    question: str,
    answers: Sequence[str],
    tokenizer: object,
    formatter: object,
    model: object,
    batch_size: int,
    max_length: int,
    max_new_tokens: int,
    device: str,
    enable_thinking: bool = False,
) -> tuple[Tensor, list[str]]:
    if batch_size <= 0 or max_length <= 0 or max_new_tokens <= 0:
        raise ValueError("batch_size, max_length, and max_new_tokens must be positive")

    predictions: list[Tensor] = []
    responses: list[str] = []
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        prompts = [
            formatter(build_prompt(str(row["text"]), question, answers)) for row in batch
        ]
        encoded = tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=max_length,
            add_special_tokens=False,
            return_tensors="pt",
        )
        model_inputs = {name: tensor.to(device) for name, tensor in encoded.items()}
        attention_mask = model_inputs.get("attention_mask")
        if attention_mask is None:
            raise ValueError("Tokenizer output does not contain attention_mask")
        token_counts = attention_mask.sum(dim=1)
        if torch.any(token_counts >= max_length):
            raise ValueError(
                f"At least one full-context prompt reached --max-length={max_length}; "
                "increase the limit so prompts are never silently truncated"
            )
        with torch.inference_mode():
            generated = model.generate(
                **model_inputs,
                do_sample=False,
                num_beams=1,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
            )
        continuation = generated[:, model_inputs["input_ids"].shape[1] :]
        batch_responses = tokenizer.batch_decode(
            continuation,
            skip_special_tokens=not enable_thinking,
            clean_up_tokenization_spaces=False,
        )
        responses.extend(batch_responses)
        predictions.append(
            torch.tensor(
                [
                    parsed if (parsed := parse_short_answer(response, answers, enable_thinking)) is not None else -1
                    for response in batch_responses
                ],
                dtype=torch.long,
            )
        )
        del generated, continuation
    return torch.cat(predictions), responses


def parse_splits(spec: str) -> tuple[str, ...]:
    values = tuple(value.strip() for value in spec.split(",") if value.strip())
    if not values or len(set(values)) != len(values):
        raise ValueError("--splits must contain unique split names")
    unknown = set(values) - set(SPLITS)
    if unknown:
        raise ValueError(f"Unknown splits: {sorted(unknown)}")
    return values


def select_fact_order(rows: list[dict], fact_order: str) -> list[dict]:
    if fact_order not in ("all", "canonical", "shuffled"):
        raise ValueError(f"Unknown fact order: {fact_order}")
    selected = [row for row in rows if fact_order == "all" or row["fact_order"] == fact_order]
    if not selected:
        raise ValueError(f"No records for fact order: {fact_order}")
    return selected


def resolve_revision(model_name: str, revision: str | None) -> str:
    if revision:
        return revision
    if model_name not in MODEL_REVISIONS:
        raise ValueError("Specify --revision for a model without a known pinned revision")
    return MODEL_REVISIONS[model_name]


def validate_qa_rows(rows: Sequence[dict], manifest: dict) -> None:
    """Check the v2 QA export against the definitions actually used for scoring."""
    for row in rows:
        if manifest.get("schema_version") != "spatial-hidden-state-probes-v2":
            continue  # Older datasets have no exported QA or fact lists.
        expected_facts = row[f"{row['fact_order']}_facts"]
        instruction = manifest["instructions"][row["instruction_id"]]
        expected_text = "\n".join(([instruction] if instruction else []) + expected_facts)
        if row["facts"] != expected_facts or row["text"] != expected_text:
            raise ValueError(f"Fact order/text mismatch in {row['id']}")
        qa = {item["id"]: item for item in row["qwen_qa"]}
        if len(qa) != len(row["qwen_qa"]) or set(qa) != set(manifest["probes"]):
            raise ValueError(f"QA question IDs mismatch in {row['id']}")
        for name, definition in manifest["probes"].items():
            item = qa[name]
            if (
                item["question"] != definition["question"]
                or item["allowed_answers"] != [answer_text(v) for v in definition["classes"]]
                or item["answer"] != answer_text(row["targets"][name])
            ):
                raise ValueError(f"QA definition/target mismatch in {row['id']}: {name}")


def run_full_context_qa(
    dataset_dir: Path,
    output_dir: Path,
    model_name: str,
    revision: str,
    splits: Sequence[str],
    batch_size: int,
    max_length: int,
    max_new_tokens: int,
    device: str,
    model_dtype_name: str,
    local_files_only: bool,
    bootstrap_samples: int,
    bootstrap_seed: int,
    fact_order: str = "all",
    enable_thinking: bool = False,
) -> Path:
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output directory: {output_dir}")
    if bootstrap_samples <= 0:
        raise ValueError("bootstrap samples must be positive")
    if not splits:
        raise ValueError("At least one split is required")
    if fact_order not in ("all", "canonical", "shuffled"):
        raise ValueError(f"Unknown fact order: {fact_order}")
    if batch_size <= 0 or max_length <= 0 or max_new_tokens <= 0:
        raise ValueError("batch_size, max_length, and max_new_tokens must be positive")

    dataset_manifest, dataset_hashes = validate_dataset_manifest(dataset_dir)
    probes = dataset_manifest.get("probes")
    if not isinstance(probes, dict) or not probes:
        raise ValueError("Dataset manifest contains no probe definitions")
    class_counts = {
        name: len(definition["classes"]) for name, definition in probes.items()
    }
    rows = {split: read_jsonl(dataset_dir / f"{split}.jsonl") for split in splits}
    for split in splits:
        rows[split] = select_fact_order(rows[split], fact_order)
        validate_qa_rows(rows[split], dataset_manifest)
    targets = {split: encode_targets(rows[split], probes) for split in splits}

    model_dtype = storage_dtype(model_dtype_name)
    tokenizer, model = load_frozen_model(
        model_name,
        revision,
        model_dtype,
        device,
        local_files_only,
    )
    # Batched decoder-only generation must read logits after the actual prompt,
    # not after right-padding tokens.
    tokenizer.padding_side = "left"
    if not getattr(tokenizer, "chat_template", None):
        raise RuntimeError("Qwen tokenizer has no chat template")
    formatter = partial(format_qa_chat, tokenizer=tokenizer, enable_thinking=enable_thinking)
    choices: dict[str, dict[str, object]] = {}
    for probe_name, definition in probes.items():
        answers = [answer_text(value) for value in definition["classes"]]
        choices[probe_name] = {
            "question": definition["question"],
            "classes": definition["classes"],
            "answer_texts": answers,
        }

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}-", dir=output_dir.parent)
    )
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    try:
        evaluations: dict[str, object] = {}
        response_records: list[dict[str, object]] = []
        for split_offset, split in enumerate(splits):
            split_predictions: dict[str, Tensor] = {}
            for probe_name, choice in choices.items():
                predictions, raw_responses = generate_answers(
                    rows[split],
                    str(choice["question"]),
                    list(choice["answer_texts"]),
                    tokenizer,
                    formatter,
                    model,
                    batch_size,
                    max_length,
                    max_new_tokens,
                    device,
                    enable_thinking=enable_thinking,
                )
                split_predictions[probe_name] = predictions
                target_labels = targets[split][probe_name]
                for row, prediction, target, response in zip(
                    rows[split],
                    predictions.tolist(),
                    target_labels.tolist(),
                    raw_responses,
                ):
                    response_records.append(
                        {
                            "split": split,
                            "id": row["id"],
                            "group_id": row["group_id"],
                            "instruction_id": row["instruction_id"],
                            "fact_order": row["fact_order"],
                            "question_id": probe_name,
                            "question": choice["question"],
                            "allowed_answers": choice["answer_texts"],
                            "raw_response": response,
                            "situation": row["text"],
                            "expected_answer": choice["answer_texts"][target],
                            "predicted_class_index": prediction if prediction >= 0 else None,
                            "target_class_index": target,
                            "valid_format": prediction >= 0,
                            "correct": prediction == target,
                        }
                    )
                print(
                    json.dumps(
                        {
                            "split": split,
                            "probe": probe_name,
                            "examples": len(rows[split]),
                        }
                    ),
                    flush=True,
                )
            evaluation = condition_metrics(
                rows[split], split_predictions, targets[split], class_counts
            )
            evaluation["group_bootstrap_95ci"] = group_bootstrap_intervals(
                rows[split],
                split_predictions,
                targets[split],
                class_counts,
                bootstrap_samples,
                bootstrap_seed + split_offset,
            )
            invalid_by_probe = {
                probe_name: {
                    "count": int((predictions < 0).sum().item()),
                    "rate": float((predictions < 0).to(torch.float32).mean().item()),
                }
                for probe_name, predictions in split_predictions.items()
            }
            invalid_count = sum(item["count"] for item in invalid_by_probe.values())
            total_responses = len(rows[split]) * len(split_predictions)
            evaluation["invalid_answers"] = {
                "count": invalid_count,
                "rate": invalid_count / total_responses,
                "total_responses": total_responses,
                "per_question": invalid_by_probe,
            }
            evaluations[split] = evaluation

        with (temporary / "responses.jsonl").open("w", encoding="utf-8") as handle:
            for record in response_records:
                handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

        report = {
            "schema_version": RESULT_SCHEMA_VERSION,
            "protocol": {
                "task": "full situation plus fixed question",
                "decision_rule": (
                    "last FINAL: <answer> marker, with a leading short-answer fallback"
                    + ("; only after </think>, missing closing tag is invalid" if enable_thinking else "")
                ),
                "generation": True,
                "chat_template_applied": True,
                "enable_thinking": enable_thinking,
                "do_sample": False,
                "num_beams": 1,
                "max_new_tokens": max_new_tokens,
                "prompt_template": PROMPT_TEMPLATE,
                "confidence_intervals": (
                    "95% percentile bootstrap over abstract-layout groups"
                ),
            },
            "model": {
                "name": model_name,
                "requested_revision": revision,
                "resolved_revision": getattr(model.config, "_commit_hash", None),
                "model_type": model.config.model_type,
                "dtype": model_dtype_name,
                "device": device,
                "eval_mode": not model.training,
                "trainable_parameters": sum(
                    parameter.numel()
                    for parameter in model.parameters()
                    if parameter.requires_grad
                ),
            },
            "tokenization": {
                "padding_side": tokenizer.padding_side,
                "pad_token_id": tokenizer.pad_token_id,
                "add_special_tokens": False,
                "max_length": max_length,
                "truncation_policy": "fail if any prompt reaches max_length",
                "chat_template_applied": True,
                "enable_thinking": enable_thinking,
            },
            "choices": choices,
            "dataset": {
                "path": str(dataset_dir),
                "schema_version": dataset_manifest.get("schema_version"),
                "files": dataset_hashes,
            },
            "splits": list(splits),
            "fact_order": fact_order,
            "evaluation": evaluations,
            "batch_size": batch_size,
            "responses_file": "responses.jsonl",
            "bootstrap_samples": bootstrap_samples,
            "bootstrap_seed": bootstrap_seed,
            "elapsed_seconds": time.perf_counter() - started,
            "peak_vram_bytes": (
                torch.cuda.max_memory_allocated() if device.startswith("cuda") else None
            ),
            "repository_commit": git_revision(),
            "torch_version": torch.__version__,
        }
        report_path = temporary / "report.json"
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(output_dir)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return output_dir / "report.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--fact-order", choices=("all", "shuffled", "canonical"), default="all",
        help="Evaluate only the selected fact order; default: both orders.",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--revision", help="Model revision; defaults to a per-model pinned SHA for Qwen3-1.7B/4B.")
    parser.add_argument(
        "--splits",
        default=",".join(DEFAULT_SPLITS),
        help="Comma-separated dataset splits; default: validation,test.",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument(
        "--enable-thinking", action="store_true",
        help="Enable Qwen reasoning; use a larger --max-new-tokens budget (e.g. 2048).",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--model-dtype",
        choices=("float16", "bfloat16", "float32"),
        default="bfloat16",
    )
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=12013)
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help="Allow Hugging Face access; by default the pinned model must be cached.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = run_full_context_qa(
        fact_order=args.fact_order,
        enable_thinking=args.enable_thinking,
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        model_name=args.model,
        revision=resolve_revision(args.model, args.revision),
        splits=parse_splits(args.splits),
        batch_size=args.batch_size,
        max_length=args.max_length,
        max_new_tokens=args.max_new_tokens,
        device=args.device,
        model_dtype_name=args.model_dtype,
        local_files_only=not args.allow_download,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
    )
    print(json.dumps({"report": str(report)}))


if __name__ == "__main__":
    main()
