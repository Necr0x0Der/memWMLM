#!/usr/bin/env python3
"""Extract frozen Qwen hidden features for the issue #12 probe dataset.

This script never generates text and never supplies probe targets to the model.
It encodes the raw situation-only ``text`` field, then stores last-token,
attention-mask-aware mean-pooled, and fact-sentence-end-pooled features from
selected transformer layers.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Iterator, Mapping, Sequence

import torch
from torch import Tensor
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


DEFAULT_MODEL = "Qwen/Qwen3-1.7B"
DEFAULT_REVISION = "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
DEFAULT_LAYERS = "4,8,12,16,20,24,last"
SPLITS = ("train", "validation", "test")
FEATURE_SCHEMA_VERSION = "qwen-hidden-probe-features-v2"
FACTS_PER_EXAMPLE = 6


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_revision() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def parse_layer_spec(spec: str, num_hidden_layers: int) -> tuple[int, ...]:
    """Resolve one-based block numbers; hidden_states[0] is embeddings."""
    if num_hidden_layers <= 0:
        raise ValueError("num_hidden_layers must be positive")
    resolved: list[int] = []
    for raw_value in spec.split(","):
        value = raw_value.strip().lower()
        if not value:
            raise ValueError(f"Empty layer in --layers {spec!r}")
        layer = num_hidden_layers if value == "last" else int(value)
        if layer < 1 or layer > num_hidden_layers:
            raise ValueError(
                f"Layer {layer} is outside 1..{num_hidden_layers}; "
                "layer 0 is the embedding output, not a transformer block"
            )
        if layer not in resolved:
            resolved.append(layer)
    if not resolved:
        raise ValueError("At least one layer is required")
    return tuple(resolved)


def storage_dtype(name: str) -> torch.dtype:
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def iter_jsonl_batches(
    path: Path,
    batch_size: int,
    max_examples: int | None = None,
) -> Iterator[list[dict[str, object]]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if max_examples is not None and max_examples <= 0:
        raise ValueError("max_examples must be positive")
    batch: list[dict[str, object]] = []
    seen_ids: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON in {path}:{line_number}") from error
            record_id = record.get("id")
            text = record.get("text")
            if not isinstance(record_id, str) or not isinstance(text, str) or not text:
                raise ValueError(f"Expected non-empty string id/text in {path}:{line_number}")
            if record_id in seen_ids:
                raise ValueError(f"Duplicate id {record_id!r} in {path}")
            seen_ids.add(record_id)
            batch.append(record)
            if len(batch) == batch_size:
                yield batch
                batch = []
            if max_examples is not None and len(seen_ids) == max_examples:
                break
    if batch:
        yield batch


def pool_selected_hidden_states(
    hidden_states: Sequence[Tensor],
    attention_mask: Tensor,
    layer_indices: Sequence[int],
) -> tuple[Tensor, Tensor]:
    """Return [batch, layer, hidden] last-token and mean-pooled features."""
    if attention_mask.ndim != 2:
        raise ValueError("attention_mask must have shape [batch, sequence]")
    token_counts = attention_mask.sum(dim=1)
    if torch.any(token_counts <= 0):
        raise ValueError("Cannot pool an empty token sequence")
    positions = torch.arange(attention_mask.shape[1], device=attention_mask.device)
    last_indices = (attention_mask.to(torch.long) * positions.unsqueeze(0)).argmax(dim=1)
    batch_indices = torch.arange(attention_mask.shape[0], device=attention_mask.device)
    mask = attention_mask.unsqueeze(-1)

    last_features: list[Tensor] = []
    mean_features: list[Tensor] = []
    for layer in layer_indices:
        if layer >= len(hidden_states):
            raise ValueError(
                f"Requested layer {layer}, but model returned {len(hidden_states)} hidden-state tensors"
            )
        states = hidden_states[layer]
        if states.shape[:2] != attention_mask.shape:
            raise ValueError("Hidden-state and attention-mask shapes do not match")
        last_features.append(states[batch_indices, last_indices])
        summed = (states * mask.to(dtype=states.dtype)).sum(dim=1)
        mean_features.append(summed / token_counts.unsqueeze(-1).to(dtype=states.dtype))
    return torch.stack(last_features, dim=1), torch.stack(mean_features, dim=1)


def fact_sentence_end_characters(record: Mapping[str, object]) -> tuple[int, ...]:
    """Return final non-whitespace character offsets for situation fact lines."""
    text = record.get("text")
    if not isinstance(text, str) or not text:
        raise ValueError("Expected a non-empty text field")
    line_spans: list[tuple[int, int]] = []
    for match in re.finditer(r"[^\r\n]+", text):
        stripped = match.group().rstrip()
        if stripped:
            line_spans.append((match.start(), match.start() + len(stripped)))
    instruction_id = record.get("instruction_id")
    if instruction_id is not None:
        if not isinstance(instruction_id, str):
            raise ValueError("instruction_id must be a string when present")
        if instruction_id != "none":
            if len(line_spans) <= 1:
                raise ValueError("Instruction example contains no fact lines")
            line_spans = line_spans[1:]
    if not line_spans:
        raise ValueError("Example contains no fact lines")
    return tuple(end - 1 for _, end in line_spans)


def sentence_end_token_mask(
    records: Sequence[Mapping[str, object]],
    offset_mapping: Tensor,
    attention_mask: Tensor,
) -> Tensor:
    """Map fact-line terminal characters to their tokenizer positions."""
    if offset_mapping.ndim != 3 or offset_mapping.shape[-1] != 2:
        raise ValueError("offset_mapping must have shape [batch, sequence, 2]")
    if offset_mapping.shape[:2] != attention_mask.shape:
        raise ValueError("offset_mapping and attention_mask shapes do not match")
    if offset_mapping.shape[0] != len(records):
        raise ValueError("offset_mapping batch does not match records")
    result = torch.zeros_like(attention_mask)
    for batch_index, record in enumerate(records):
        starts = offset_mapping[batch_index, :, 0]
        ends = offset_mapping[batch_index, :, 1]
        valid = attention_mask[batch_index].to(torch.bool)
        for character in fact_sentence_end_characters(record):
            matches = torch.nonzero(
                valid & (starts <= character) & (ends > character), as_tuple=False
            ).flatten()
            if matches.numel() != 1:
                raise ValueError(
                    f"Expected one token for sentence-end character {character}, "
                    f"found {matches.numel()}"
                )
            result[batch_index, int(matches.item())] = 1
    return result


def pool_masked_hidden_states(
    hidden_states: Sequence[Tensor],
    pooling_mask: Tensor,
    layer_indices: Sequence[int],
) -> Tensor:
    """Return [batch, layer, hidden] means over an explicit token mask."""
    if pooling_mask.ndim != 2:
        raise ValueError("pooling_mask must have shape [batch, sequence]")
    token_counts = pooling_mask.sum(dim=1)
    if torch.any(token_counts <= 0):
        raise ValueError("Cannot pool an empty token selection")
    mask = pooling_mask.unsqueeze(-1)
    pooled: list[Tensor] = []
    for layer in layer_indices:
        if layer >= len(hidden_states):
            raise ValueError(
                f"Requested layer {layer}, but model returned {len(hidden_states)} hidden-state tensors"
            )
        states = hidden_states[layer]
        if states.shape[:2] != pooling_mask.shape:
            raise ValueError("Hidden-state and pooling-mask shapes do not match")
        summed = (states * mask.to(dtype=states.dtype)).sum(dim=1)
        pooled.append(summed / token_counts.unsqueeze(-1).to(dtype=states.dtype))
    return torch.stack(pooled, dim=1)


def extract_split(
    dataset_path: Path,
    output_dir: Path,
    tokenizer: object,
    model: object,
    layer_indices: Sequence[int],
    batch_size: int,
    device: str,
    output_dtype: torch.dtype,
    max_length: int,
    add_special_tokens: bool = False,
    max_examples: int | None = None,
    save_token_states: bool = False,
) -> dict[str, object]:
    """Extract one split into independently loadable, bounded-size shards."""
    output_dir.mkdir(parents=True, exist_ok=False)
    example_count = 0
    shard_records: list[dict[str, object]] = []
    for shard_index, records in enumerate(
        iter_jsonl_batches(dataset_path, batch_size, max_examples)
    ):
        texts = [str(record["text"]) for record in records]
        encoded = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            add_special_tokens=add_special_tokens,
            return_offsets_mapping=True,
            return_tensors="pt",
        )
        offset_mapping = encoded.pop("offset_mapping", None)
        if not isinstance(offset_mapping, Tensor):
            raise ValueError("Fast tokenizer did not return tensor offset mappings")
        if "attention_mask" not in encoded:
            raise ValueError("Tokenizer output does not contain attention_mask")
        sentence_end_mask = sentence_end_token_mask(
            records,
            offset_mapping,
            encoded["attention_mask"],
        )
        for record, count in zip(records, sentence_end_mask.sum(dim=1).tolist()):
            if "instruction_id" in record and count != FACTS_PER_EXAMPLE:
                raise ValueError(
                    f"Expected {FACTS_PER_EXAMPLE} fact sentence ends for "
                    f"{record.get('id')!r}, found {count}"
                )
        model_inputs = {name: tensor.to(device) for name, tensor in encoded.items()}
        if "attention_mask" not in model_inputs:
            raise ValueError("Tokenizer output does not contain attention_mask")
        original_token_counts = model_inputs["attention_mask"].sum(dim=1)
        if torch.any(original_token_counts >= max_length):
            raise ValueError(
                f"At least one input in {dataset_path} reached --max-length={max_length}; "
                "increase the limit so situation text is never silently truncated"
            )
        with torch.inference_mode():
            outputs = model(
                **model_inputs,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )
        if outputs.hidden_states is None:
            raise RuntimeError("Model did not return hidden states")
        last_token, mean_pool = pool_selected_hidden_states(
            outputs.hidden_states,
            model_inputs["attention_mask"],
            layer_indices,
        )
        sentence_end_mean = pool_masked_hidden_states(
            outputs.hidden_states,
            sentence_end_mask.to(device),
            layer_indices,
        )
        payload = {
            "ids": [str(record["id"]) for record in records],
            "layer_indices": torch.tensor(layer_indices, dtype=torch.int16),
            "token_counts": original_token_counts.to(device="cpu", dtype=torch.int32),
            "sentence_end_counts": sentence_end_mask.sum(dim=1).to(dtype=torch.int16),
            "last_token": last_token.to(device="cpu", dtype=output_dtype),
            "mean_pool": mean_pool.to(device="cpu", dtype=output_dtype),
            "sentence_end_mean": sentence_end_mean.to(device="cpu", dtype=output_dtype),
        }
        shard_path = output_dir / f"shard-{shard_index:05d}.pt"
        if save_token_states:
            # Ragged [layers, valid_tokens, hidden] tensors; padding is never saved.
            payload["token_states"] = [
                torch.stack([
                    outputs.hidden_states[layer][i, model_inputs["attention_mask"][i].bool()]
                    .to(device="cpu", dtype=output_dtype)
                    for layer in layer_indices
                ])
                for i in range(len(records))
            ]
        torch.save(payload, shard_path)
        shard_records.append(
            {
                "path": shard_path.name,
                "examples": len(records),
                "sha256": sha256(shard_path),
            }
        )
        example_count += len(records)
        print(
            json.dumps(
                {
                    "split": dataset_path.stem,
                    "shard": shard_index,
                    "processed": example_count,
                }
            ),
            flush=True,
        )
        del outputs, last_token, mean_pool, sentence_end_mean, payload
    return {"examples": example_count, "shards": shard_records}


def validate_dataset_manifest(dataset_dir: Path) -> tuple[dict[str, object], dict[str, str]]:
    manifest_path = dataset_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    file_hashes: dict[str, str] = {}
    for split in SPLITS:
        path = dataset_dir / f"{split}.jsonl"
        actual_hash = sha256(path)
        expected = manifest.get("files", {}).get(path.name, {}).get("sha256")
        if expected != actual_hash:
            raise ValueError(
                f"Dataset hash mismatch for {path}: manifest={expected!r}, actual={actual_hash}"
            )
        file_hashes[path.name] = actual_hash
    return manifest, file_hashes


def load_frozen_model(
    model_name: str,
    revision: str,
    dtype: torch.dtype,
    device: str,
    local_files_only: bool,
) -> tuple[object, object]:
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        revision=revision,
        local_files_only=local_files_only,
        use_fast=True,
    )
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Match the loading path used by the existing Qwen evaluation scripts.
    # Extraction calls only ``model.model``, so the LM head is never executed.
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        revision=revision,
        dtype=dtype,
        local_files_only=local_files_only,
    ).to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    trainable_parameters = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    if trainable_parameters:
        raise RuntimeError(f"Frozen model unexpectedly has {trainable_parameters} trainable parameters")
    return tokenizer, model


def build_features(
    dataset_dir: Path,
    output_dir: Path,
    model_name: str,
    revision: str,
    layer_spec: str,
    batch_size: int,
    device: str,
    model_dtype_name: str,
    output_dtype_name: str,
    max_length: int,
    local_files_only: bool,
    add_special_tokens: bool = False,
    max_examples_per_split: int | None = None,
    save_token_states: bool = False,
) -> Path:
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output directory: {output_dir}")
    if max_length <= 0:
        raise ValueError("max_length must be positive")
    if max_examples_per_split is not None and max_examples_per_split <= 0:
        raise ValueError("max_examples_per_split must be positive")
    dataset_manifest, dataset_hashes = validate_dataset_manifest(dataset_dir)
    config = AutoConfig.from_pretrained(
        model_name,
        revision=revision,
        local_files_only=local_files_only,
    )
    layer_indices = parse_layer_spec(layer_spec, config.num_hidden_layers)
    model_dtype = storage_dtype(model_dtype_name)
    output_dtype = storage_dtype(output_dtype_name)
    tokenizer, model = load_frozen_model(
        model_name,
        revision,
        model_dtype,
        device,
        local_files_only,
    )

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}-", dir=output_dir.parent))
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    try:
        split_metadata: dict[str, object] = {}
        backbone = getattr(model, "model", None)
        if backbone is None:
            raise RuntimeError("Expected the loaded causal LM to expose its base model as .model")
        for split in SPLITS:
            split_metadata[split] = extract_split(
                dataset_dir / f"{split}.jsonl",
                temporary / split,
                tokenizer,
                backbone,
                layer_indices,
                batch_size,
                device,
                output_dtype,
                max_length,
                add_special_tokens,
                max_examples_per_split,
                save_token_states,
            )
        manifest = {
            "schema_version": FEATURE_SCHEMA_VERSION,
            "issue": "https://github.com/Necr0x0Der/memWMLM/issues/12",
            "model": {
                "name": model_name,
                "requested_revision": revision,
                "resolved_revision": getattr(model.config, "_commit_hash", None),
                "model_type": model.config.model_type,
                "num_hidden_layers": model.config.num_hidden_layers,
                "hidden_size": model.config.hidden_size,
                "dtype": model_dtype_name,
                "device": device,
                "eval_mode": not model.training,
                "trainable_parameters": sum(
                    parameter.numel() for parameter in model.parameters() if parameter.requires_grad
                ),
            },
            "tokenization": {
                "padding_side": tokenizer.padding_side,
                "pad_token_id": tokenizer.pad_token_id,
                "add_special_tokens": add_special_tokens,
                "max_length": max_length,
                "truncation_policy": "fail if any example reaches max_length",
                "chat_template_applied": False,
            },
            "features": {
                "token_states": "ragged [layers, valid_tokens, hidden]; all input tokens including instruction" if save_token_states else None,
                "layer_indices": list(layer_indices),
                "layer_indexing": (
                    "hidden_states[0] is the embedding output; index N below the final "
                    "index is the output of transformer block N; the final index is the "
                    "last block output after the model final normalization"
                ),
                "pooling": {
                    "last_token": "last token selected by attention_mask",
                    "mean_pool": "arithmetic mean over all non-padding input tokens",
                    "sentence_end_mean": (
                        "arithmetic mean over the hidden state at the final token of each "
                        "of the six situation fact lines; an instruction line is excluded"
                    ),
                },
                "storage_dtype": output_dtype_name,
            },
            "dataset": {
                "path": str(dataset_dir),
                "schema_version": dataset_manifest.get("schema_version"),
                "files": dataset_hashes,
            },
            "splits": split_metadata,
            "batch_size": batch_size,
            "max_examples_per_split": max_examples_per_split,
            "elapsed_seconds": time.perf_counter() - started,
            "peak_vram_bytes": (
                torch.cuda.max_memory_allocated() if device.startswith("cuda") else None
            ),
            "repository_commit": git_revision(),
            "torch_version": torch.__version__,
        }
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(output_dir)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return output_dir / "manifest.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--layers", default=DEFAULT_LAYERS)
    parser.add_argument("--save-token-states", action="store_true",
                        help="Also save individual valid-token states for learned attention pooling.")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--model-dtype",
        choices=("float16", "bfloat16", "float32"),
        default="bfloat16",
    )
    parser.add_argument(
        "--output-dtype",
        choices=("float16", "bfloat16", "float32"),
        default="float16",
    )
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument(
        "--max-examples-per-split",
        type=int,
        help="Deterministic prefix limit for a runtime smoke test; omit for the full dataset.",
    )
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help="Allow Hugging Face network access; by default the pinned model must be cached.",
    )
    parser.add_argument(
        "--add-special-tokens",
        action="store_true",
        help="Add tokenizer special tokens. Default is raw situation tokens only.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest_path = build_features(
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        model_name=args.model,
        revision=args.revision,
        layer_spec=args.layers,
        batch_size=args.batch_size,
        device=args.device,
        model_dtype_name=args.model_dtype,
        output_dtype_name=args.output_dtype,
        max_length=args.max_length,
        local_files_only=not args.allow_download,
        add_special_tokens=args.add_special_tokens,
        max_examples_per_split=args.max_examples_per_split,
        save_token_states=args.save_token_states,
    )
    print(json.dumps({"manifest": str(manifest_path)}))


if __name__ == "__main__":
    main()
