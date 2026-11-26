#!/usr/bin/env python3
"""Evaluate issue #6 conditions on TriviaQA fact-passage banks.

Banks are consecutive chunks of size ``N`` of ``eval.jsonl`` (or of
``calibration.jsonl`` with ``--split calibration``), identical to the record in
``banks.json``. Every condition answers every question of the split once per
``N``; results are resumable JSONL, one file per (condition, N).

Conditions
  closed_book            question only (prompt identical to issue #5)
  oracle_passage         the question's own passage in context
  counterfactual_oracle  its answer-swapped passage in context
  full_context           all N bank passages in context
  rag_bm25@K             top-K BM25 bank passages in context
  memory                 delta-rule state of the bank, no passages in context
  null_memory            zero state, identical read path
  shuffled_memory        state of N *calibration* passages (facts absent)
  counterfactual_memory  state of the bank's answer-swapped passages
  qa_memory              state of the bank's question-answer pairs, each written as
                         the exact closed-book prompt followed by the gold answer
                         (sanity check, 2026-11-19 README amendment)
  shuffled_qa_memory     same for N calibration questions (facts absent)
  counterfactual_qa_memory  same with each answer swapped for the counterfactual one
  passage_query_memory   state as ``memory``; the usual question prompt, but every layer reads
                         the memory with the mean pre-RoPE query of the question's own passage
                         instead of the token's query (sanity check 6, 2026-11-23 amendment)
  shuffled_passage_query_memory  state as ``shuffled_memory`` (own passage absent), same read
  counterfactual_passage_query_memory  state as ``counterfactual_memory``; read with the
                         mean query of the stored (answer-swapped) passage
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "issue-5-qwen3-factual-knowledge"))
from triviaqa import normalize_answer, score_prediction  # noqa: E402

from memwmlm.models.delta_qwen import (
    DeltaMemoryConfig, DeltaMemoryQwen, layer_subset, load_feature_maps, null_state, prepend_sink,
)
from memwmlm.models.qwen import QwenChatPromptFormatter

CLOSED_PROMPT_VERSION = "triviaqa_short_answer_v1"  # identical to issue #5
CONTEXT_PROMPT_VERSION = "triviaqa_short_answer_context_v1"
INSTRUCTION = "Answer the factual question with only the shortest answer. Do not explain."
CONTEXT_INSTRUCTION = (
    "Use the passages below if they are relevant. "
    "Answer the factual question with only the shortest answer. Do not explain."
)
MEMORY_CONDITIONS = ("memory", "null_memory", "shuffled_memory", "counterfactual_memory",
                     "qa_memory", "shuffled_qa_memory", "counterfactual_qa_memory",
                     "passage_query_memory", "shuffled_passage_query_memory",
                     "counterfactual_passage_query_memory")


def render_closed(question: str) -> str:
    return f"{INSTRUCTION}\n\nQuestion: {question}\nAnswer:"


def render_context(question: str, passages: list[str]) -> str:
    body = "\n".join(f"[{i + 1}] {p}" for i, p in enumerate(passages))
    return f"{CONTEXT_INSTRUCTION}\n\nPassages:\n{body}\n\nQuestion: {question}\nAnswer:"


DONOR_QUESTIONS = Path("runs/delta-rule/global-v1/questions.jsonl")  # answer surfaces of counterfactual donors
_donor_answers: dict[str, str] = {}


def answer_surface(row: dict, counterfactual: bool = False) -> str:
    """The gold answer as written in the passage, or the donor's answer that the counterfactual swapped in."""

    if not counterfactual:
        start, end = row["answer_span"]
        return row["passage"][start:end]
    if not _donor_answers:
        _donor_answers.update({r["question_id"]: r["answer_surface"] for r in read_jsonl(DONOR_QUESTIONS)})
    return _donor_answers[row["counterfactual"]["donor_question_id"]]


def qa_text(row: dict, formatter, counterfactual: bool = False) -> str:
    return formatter(render_closed(row["question"])) + answer_surface(row, counterfactual)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").split("\n") if line]


def completed_ids(path: Path) -> set[str]:
    return {r["question_id"] for r in read_jsonl(path)} if path.exists() else set()


def git_revision() -> str | None:
    result = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else None


def git_dirty() -> bool:
    result = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, check=False)
    return bool(result.stdout.strip())


class BM25:
    def __init__(self, passages: list[str]) -> None:
        from rank_bm25 import BM25Okapi

        self.index = BM25Okapi([normalize_answer(p).split() or ["_"] for p in passages])

    def top(self, query: str, k: int) -> list[int]:
        scores = self.index.get_scores(normalize_answer(query).split())
        return sorted(range(len(scores)), key=lambda i: (-scores[i], i))[:k]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--banks-dir", type=Path, required=True)
    p.add_argument("--split", choices=("eval", "calibration"), default="eval")
    p.add_argument("--conditions", required=True, help="comma list, e.g. closed_book,memory,rag_bm25@3")
    p.add_argument("--ladder", default="1,4,16,64,256,1024")
    p.add_argument("--limit", type=int, default=None, help="first K questions only (smoke tests)")
    p.add_argument("--start", type=int, default=0, help="skip the first S questions of the split")
    p.add_argument("--model", default="Qwen/Qwen3-1.7B")
    p.add_argument("--revision", default="70d244cc86ccca08cf5af4e1e306ecf908b1ad5e")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--write-batch-size", type=int, default=64)
    p.add_argument("--max-new-tokens", type=int, default=32)
    p.add_argument("--max-context-tokens", type=int, default=30_000)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    # Variant 1 memory settings
    p.add_argument("--rule", choices=("delta", "hebbian"), default="delta")
    p.add_argument("--beta", type=float, default=1.0)
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--beta-norm", choices=("none", "tokens"), default="none",
                   help="tokens: divide beta by the number of tokens written into each state")
    p.add_argument("--gate", type=float, default=1.0)
    p.add_argument("--layers", default="all")
    p.add_argument("--center-keys", action="store_true")
    p.add_argument("--keep-first-token", action="store_true")
    p.add_argument("--sink-prefix", action="store_true",
                   help="prepend <|endoftext|> to every written passage so the skipped first position is "
                        "that token, not the first content token (2026-11-18 fix)")
    p.add_argument("--key-expand", type=int, default=0, help="DPFP order nu: key dim 128 -> 256*nu (0 = off)")
    p.add_argument("--center-queries", action="store_true", help="centre read queries with the key mean")
    p.add_argument("--whiten", action="store_true",
                   help="whiten keys and queries per head with calibration key statistics (needs centring)")
    p.add_argument("--whiten-eps", type=float, default=0.1, help="shrinkage, relative to the mean eigenvalue")
    p.add_argument("--read", choices=("linear", "knn"), default="linear",
                   help="knn = uncompressed stored keys/values, joint softmax (upper bound, no delta rule)")
    p.add_argument("--knn-top-k", type=int, default=32, help="stored keys per query head/token (0 = all)")
    p.add_argument("--feature-map", type=Path, default=None,
                   help="trained FeatureMaps checkpoint (Variant 1T): delta rule on phi_k(k), read with phi_q(q)")
    p.add_argument("--extra-lora", type=Path, default=None,
                   help="with --lora: also stack this Variant 1 LoRA's q/o_proj weights (gates/beta not used)")
    p.add_argument("--lora", type=Path, default=None,
                   help="train_lora_banks.py checkpoint: application LoRA + trained gates (writes use frozen Qwen)")
    p.add_argument("--head-probe", type=Path, default=None,
                   help="probe_head_retrieval.py JSON (calibration split) for head selection")
    p.add_argument("--top-heads", type=int, default=0, help="read only the K best probe heads (0 = all)")
    p.add_argument("--probe-rung", default="16")
    p.add_argument("--probe-score", choices=("maxsim", "last"), default="maxsim")
    p.add_argument("--tag", default="", help="suffix distinguishing memory settings in file names")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    split_rows = read_jsonl(args.banks_dir / f"{args.split}.jsonl")
    calibration_rows = read_jsonl(args.banks_dir / "calibration.jsonl")
    split_rows = split_rows[args.start :]
    if args.limit:
        split_rows = split_rows[: args.limit]
    if args.split == "calibration":
        # shuffled_memory must not contain the evaluated questions' passages
        used = {r["question_id"] for r in split_rows}
        calibration_rows = [r for r in calibration_rows if r["question_id"] not in used]
    conditions = args.conditions.split(",")
    ladder = [int(n) for n in args.ladder.split(",")]

    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, revision=args.revision, dtype=getattr(torch, args.dtype), local_files_only=True
    ).to(args.device).eval()
    formatter = QwenChatPromptFormatter(tokenizer)
    n_layers = len(model.model.layers)
    memory = None
    if any(c in MEMORY_CONDITIONS for c in conditions):
        memory = DeltaMemoryQwen(model, DeltaMemoryConfig(
            rule=args.rule, beta=args.beta, alpha=args.alpha, gate=args.gate, beta_norm=args.beta_norm,
            layers=layer_subset(n_layers, args.layers),
            skip_first_token=not args.keep_first_token, center_keys=args.center_keys,
            key_expand=args.key_expand, center_queries=args.center_queries,
            read=args.read, knn_top_k=args.knn_top_k, whiten=args.whiten, whiten_eps=args.whiten_eps,
        ))
        if args.feature_map:
            if args.center_keys or args.key_expand or args.whiten:
                raise SystemExit("--feature-map replaces centring / DPFP / whitening")
            load_feature_maps(memory, args.feature_map)
        if args.top_heads:
            select_heads(memory, args)
        if args.center_keys:
            k, _, m = encode_passages(memory, tokenizer, [r["passage"] for r in calibration_rows[:256]], args)
            memory.calibrate_key_mean(k, m)
    if args.lora:
        if memory is None:  # closed_book only: LoRA without memory
            memory_free = DeltaMemoryQwen(model, DeltaMemoryConfig())
            _install_lora(model, memory_free, args)
            memory_free.close()
        else:
            _install_lora(model, memory, args)

    memory_settings = {
        "rule": args.rule, "beta": args.beta, "beta_norm": args.beta_norm, "alpha": args.alpha, "gate": args.gate,
        "layers": args.layers, "center_keys": args.center_keys,
        "sink_prefix": args.sink_prefix, "key_expand": args.key_expand, "center_queries": args.center_queries,
        "feature_map": str(args.feature_map) if args.feature_map else None,
        "lora": str(args.lora) if args.lora else None,
        "extra_lora": str(args.extra_lora) if args.extra_lora else None,
        "whiten": args.whiten, "whiten_eps": args.whiten_eps if args.whiten else None,
        "read": args.read, "knn_top_k": args.knn_top_k if args.read == "knn" else None,
        "skip_first_token": not args.keep_first_token, "key_space": "pre_rope_k_norm_l2",
        "head_probe": str(args.head_probe) if args.head_probe else None, "top_heads": args.top_heads,
        "probe_rung": args.probe_rung, "probe_score": args.probe_score,
        "selected_heads": getattr(memory, "selected_heads", None) if memory else None,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_meta = {
        "model": args.model, "revision": args.revision, "dtype": args.dtype, "split": args.split,
        "banks_manifest_sha256": hashlib.sha256((args.banks_dir / "manifest.json").read_bytes()).hexdigest(),
        "prompt_versions": {"closed": CLOSED_PROMPT_VERSION, "context": CONTEXT_PROMPT_VERSION},
        "max_new_tokens": args.max_new_tokens, "do_sample": False, "enable_thinking": False,
        "repository_commit": git_revision(), "working_tree_dirty": git_dirty(),
        "torch": torch.__version__, "memory_settings": memory_settings, "limit": args.limit,
        "start": args.start,
    }

    for n in ladder:
        if len(split_rows) % n:
            print(f"skip N={n}: {len(split_rows)} questions not divisible", flush=True)
            continue
        banks = [split_rows[i : i + n] for i in range(0, len(split_rows), n)]
        for condition in conditions:
            if condition in MEMORY_CONDITIONS and memory is None:
                raise RuntimeError("memory condition without memory")
            if condition in ("closed_book", "oracle_passage", "counterfactual_oracle") and n != ladder[0]:
                continue  # independent of N: evaluate once
            if condition.startswith("counterfactual") and args.split != "eval":
                continue
            tag = f"-{args.tag}" if (args.tag and condition in MEMORY_CONDITIONS) else ""
            out = args.output_dir / f"{condition}{tag}-N{n}.jsonl"
            meta_path = out.with_suffix(".meta.json")
            meta = {**run_meta, "condition": condition, "N": n}
            if not meta_path.exists():
                meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
            done = completed_ids(out)
            started = time.perf_counter()
            with out.open("a", encoding="utf-8") as handle:
                for bank_index, bank in enumerate(banks):
                    todo = [r for r in bank if r["question_id"] not in done]
                    if not todo:
                        continue
                    run_bank(condition, n, bank_index, bank, todo, calibration_rows, model, tokenizer,
                             formatter, memory, args, handle)
            rows = read_jsonl(out)
            em = sum(r["exact_match"] for r in rows) / max(len(rows), 1)
            print(json.dumps({"condition": condition, "tag": args.tag, "N": n, "examples": len(rows),
                              "exact_match": round(100 * em, 2),
                              "seconds": round(time.perf_counter() - started, 1)}), flush=True)


def select_heads(memory: DeltaMemoryQwen, args) -> None:
    """Gate = args.gate on the K best calibration-probe heads, 0 elsewhere."""

    probe = json.loads(args.head_probe.read_text())
    acc = torch.tensor(probe["rungs"][args.probe_rung][args.probe_score]["per_head"])  # [L, H]
    flat = acc.flatten().topk(args.top_heads).indices
    chosen = sorted((int(i) // acc.shape[1], int(i) % acc.shape[1]) for i in flat)
    memory.selected_heads = [list(x) for x in chosen]
    device = next(memory.qwen.parameters()).device
    for layer in memory.layer_indices:
        gate = torch.zeros(memory.num_heads, device=device)
        for l, h in chosen:
            if l == layer:
                gate[h] = args.gate
        memory.gates[layer] = gate


@torch.no_grad()
def encode_passages(memory: DeltaMemoryQwen, tokenizer, passages: list[str], args):
    keys, values, masks, hidden = [], [], [], []
    for start in range(0, len(passages), args.write_batch_size):
        chunk = passages[start : start + args.write_batch_size]
        tokenizer.padding_side = "right"
        batch = tokenizer(chunk, padding=True, return_tensors="pt").to(args.device)
        ids, mask = batch["input_ids"], batch["attention_mask"]
        if args.sink_prefix:
            ids, mask = prepend_sink(ids, mask, tokenizer.convert_tokens_to_ids("<|endoftext|>"))
        k, v, m = memory.capture_passages(ids, mask)
        keys.append(k[:, :, m])
        values.append(v[:, :, m])
        masks.append(m[m])
        if memory.needs_hidden:  # per-token beta / writer need the written tokens' attention inputs
            hidden.append(memory.last_hidden[:, m])
    if hidden:
        memory.last_hidden = torch.cat(hidden, 1)
    return torch.cat(keys, 2), torch.cat(values, 2), torch.cat(masks)


_LORAS: list = []  # application LoRA modules (--lora); off while writing


def _install_lora(model, memory, args) -> None:
    from train_lora_banks import load_lora

    _LORAS.extend(load_lora(model, memory, args.lora))
    if args.extra_lora:  # 2026-11-25: a Variant 1 LoRA stacked on top (its q/o_proj weights only)
        from train_lora_banks import install_lora

        extra = torch.load(args.extra_lora, map_location="cpu")
        mods = install_lora(model, extra["rank"])
        for i, m in enumerate(mods):
            m.down.weight.data.copy_(extra["lora"][i][0])
            m.up.weight.data.copy_(extra["lora"][i][1])
        _LORAS.extend(mods)


def build_state(memory, tokenizer, passages, args):
    for m in _LORAS:
        m.enabled = False  # memory is always written by frozen Qwen
    try:
        keys, values, mask = encode_passages(memory, tokenizer, passages, args)
        return memory.build_state(keys, values, mask,
                                  hidden=memory.last_hidden if memory.needs_hidden else None)
    finally:
        for m in _LORAS:
            m.enabled = True


@torch.no_grad()
def passage_read_queries(memory: DeltaMemoryQwen, tokenizer, passages: list[str], args) -> dict:
    """Mean pre-RoPE query of each passage (same encoding as the write): layer -> [B, H, D]."""

    tokenizer.padding_side = "right"
    batch = tokenizer(passages, padding=True, return_tensors="pt").to(args.device)
    ids, mask = batch["input_ids"], batch["attention_mask"]
    if args.sink_prefix:
        ids, mask = prepend_sink(ids, mask, tokenizer.convert_tokens_to_ids("<|endoftext|>"))
    return memory.passage_queries(ids, mask)


def run_bank(condition, n, bank_index, bank, todo, calibration_rows, model, tokenizer, formatter,
             memory, args, handle):
    state = None
    query_source = None  # passage whose mean query reads the memory (passage_query conditions)
    if condition.endswith("passage_query_memory"):
        cf = condition.startswith("counterfactual")
        query_source = (lambda r: r["counterfactual"]["passage"]) if cf else (lambda r: r["passage"])
        condition_state = condition.replace("passage_query_memory", "memory")
    else:
        condition_state = condition
    if condition in MEMORY_CONDITIONS:
        condition, original = condition_state, condition
        if condition == "memory":
            passages = [r["passage"] for r in bank]
        elif condition == "qa_memory":
            passages = [qa_text(r, formatter) for r in bank]
        elif condition == "counterfactual_qa_memory":
            passages = [qa_text(r, formatter, counterfactual=True) for r in bank]
        elif condition == "shuffled_qa_memory":
            offset = (bank_index * n) % max(len(calibration_rows) - n + 1, 1)
            passages = [qa_text(r, formatter) for r in calibration_rows[offset : offset + n]]
            if len(passages) < n:
                raise RuntimeError("calibration set smaller than N")
        elif condition == "counterfactual_memory":
            passages = [r["counterfactual"]["passage"] for r in bank]
        elif condition == "shuffled_memory":
            offset = (bank_index * n) % max(len(calibration_rows) - n + 1, 1)
            passages = [r["passage"] for r in calibration_rows[offset : offset + n]]
            if len(passages) < n:
                raise RuntimeError("calibration set smaller than N")
        else:
            passages = [r["passage"] for r in bank[:1]]
        state = build_state(memory, tokenizer, passages, args)
        if condition == "null_memory":
            state = null_state(state)
        condition = original

    bm25 = BM25([r["passage"] for r in bank]) if condition.startswith("rag_bm25@") else None
    prompts = []
    for row in todo:
        if condition in ("closed_book",) + MEMORY_CONDITIONS:
            text = render_closed(row["question"])
        elif condition == "oracle_passage":
            text = render_context(row["question"], [row["passage"]])
        elif condition == "counterfactual_oracle":
            text = render_context(row["question"], [row["counterfactual"]["passage"]])
        elif condition == "full_context":
            text = render_context(row["question"], [r["passage"] for r in bank])
        elif bm25 is not None:
            k = int(condition.split("@")[1])
            text = render_context(row["question"], [bank[i]["passage"] for i in bm25.top(row["question"], k)])
        else:
            raise ValueError(f"Unknown condition {condition}")
        prompts.append(formatter(text))

    tokenizer.padding_side = "left"
    batch_size = args.batch_size if condition != "full_context" else max(1, args.batch_size * 16 // max(n, 16))
    for start in range(0, len(todo), batch_size):
        rows = todo[start : start + batch_size]
        inputs = tokenizer(prompts[start : start + batch_size], padding=True, return_tensors="pt").to(args.device)
        if inputs["input_ids"].shape[1] > args.max_context_tokens:
            raise RuntimeError(f"{condition} N={n}: prompt exceeds --max-context-tokens")
        context = memory.reading(state) if memory is not None else _nullcontext()
        if query_source is not None:
            memory.read_queries = passage_read_queries(memory, tokenizer, [query_source(r) for r in rows], args)
        try:
            with context:
                generated = model.generate(**inputs, do_sample=False, max_new_tokens=args.max_new_tokens,
                                           pad_token_id=tokenizer.pad_token_id, top_p=None, top_k=None,
                                           temperature=None)
        finally:
            if memory is not None:
                memory.read_queries = None
        for row, ids in zip(rows, generated[:, inputs["input_ids"].shape[1]:], strict=True):
            prediction = tokenizer.decode(ids, skip_special_tokens=True).strip()
            exact, f1 = score_prediction(prediction, row["aliases"])
            record = {"question_id": row["question_id"], "bank": bank_index, "prediction": prediction,
                      "exact_match": exact, "f1": f1}
            if "counterfactual" in row and condition.startswith("counterfactual"):
                cf_exact, cf_f1 = score_prediction(prediction, row["counterfactual"]["aliases"])
                record.update({"cf_follow_exact_match": cf_exact, "cf_follow_f1": cf_f1,
                               "cf_valid": row["counterfactual"]["valid"]})
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()


class _nullcontext:
    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False


if __name__ == "__main__":
    main()
