#!/usr/bin/env python3
"""Variant 1 with one global memory S (README amendment 2026-11-13).

The whole corpus of ``prepare_global.py`` is written once into per-layer/KV-head
delta-rule (or Hebbian) states using frozen Qwen3 (any LoRA disabled). Then:

  --stage frozen  scalar gate, nothing trained
  --stage gate    per-layer x per-query-head gates trained (448 parameters)
  --stage lora    gates + rank-r LoRA on q_proj and o_proj (application side)

Training uses the closed-book prompt and the answer surface form of every train
question except the ``train_heldout`` group, with S read during the forward
pass (or S = 0 with ``--train-null``, the fine-tuning control). Evaluation
generates answers for every group under: memory (S), null (S = 0) and, for
``dev_in``, counterfactual memory (S with the dev_in passages answer-swapped).
``--references`` additionally runs closed-book and BM25@1 over the corpus.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from run_banks import BM25, git_dirty, git_revision, read_jsonl, render_closed, render_context  # noqa: E402
from run_banks import score_prediction  # noqa: E402

from memwmlm.models.delta_qwen import DeltaMemoryConfig, DeltaMemoryQwen, null_state, prepend_sink
from memwmlm.models.qwen import QwenChatPromptFormatter

GROUPS = ("train_seen", "train_heldout", "dev_in", "dev_out")


class LoRALinear(nn.Module):
    """Frozen linear + fp32 low-rank update; ``enabled`` toggles the update."""

    def __init__(self, base: nn.Linear, rank: int, alpha: float) -> None:
        super().__init__()
        self.base = base
        self.down = nn.Linear(base.in_features, rank, bias=False, device=base.weight.device, dtype=torch.float32)
        self.up = nn.Linear(rank, base.out_features, bias=False, device=base.weight.device, dtype=torch.float32)
        nn.init.kaiming_uniform_(self.down.weight, a=5**0.5)
        nn.init.zeros_(self.up.weight)
        self.scale = alpha / rank
        self.enabled = True

    # expose the wrapped layer's shape so a second LoRA can be stacked on top (2026-11-25)
    in_features = property(lambda self: self.base.in_features)
    out_features = property(lambda self: self.base.out_features)
    weight = property(lambda self: self.base.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        if not self.enabled:
            return out
        return out + (self.up(self.down(x.float())) * self.scale).to(out.dtype)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=Path("runs/delta-rule/global-v1"))
    p.add_argument("--banks-dir", type=Path, default=Path("runs/delta-rule/banks-v2"))
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--stage", choices=("frozen", "gate", "lora"), required=True)
    p.add_argument("--rule", choices=("delta", "hebbian"), default="delta")
    p.add_argument("--beta", type=float, default=0.5)
    p.add_argument("--beta-norm", choices=("none", "tokens"), default="none")
    p.add_argument("--gate", type=float, required=True, help="frozen gate / initial gate")
    p.add_argument("--train-null", action="store_true", help="train with S = 0 (fine-tuning control)")
    p.add_argument("--key-expand", type=int, default=0, help="DPFP order nu: key dim 128 -> 256*nu (0 = off)")
    p.add_argument("--sink-prefix", action="store_true",
                   help="prepend <|endoftext|> so the skipped first position is not content (2026-11-18 fix)")
    p.add_argument("--center-queries", action="store_true", help="centre read queries with the key mean (2026-11-18 fix)")
    p.add_argument("--gates", default=None, help="frozen only: comma list of gates, each evaluated")
    p.add_argument("--eval-groups", default=",".join(GROUPS))
    p.add_argument("--lora-rank", type=int, default=8)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr-gate", type=float, default=1e-2)
    p.add_argument("--lr-lora", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--references", action="store_true", help="also run closed_book and rag_bm25@1")
    p.add_argument("--eval-limit", type=int, default=None, help="questions per group (smoke tests)")
    p.add_argument("--train-limit", type=int, default=None, help="training questions (smoke tests)")
    p.add_argument("--micro-batch", type=int, default=0, help="split each training batch (0 = whole batch)")
    p.add_argument("--gen-batch-size", type=int, default=32)
    p.add_argument("--write-batch-size", type=int, default=64)
    p.add_argument("--max-new-tokens", type=int, default=32)
    p.add_argument("--model", default="Qwen/Qwen3-1.7B")
    p.add_argument("--revision", default="70d244cc86ccca08cf5af4e1e306ecf908b1ad5e")
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def set_lora(loras: list[LoRALinear], enabled: bool) -> None:
    for module in loras:
        module.enabled = enabled


def passage_batch(tokenizer, texts: list[str], args) -> tuple[torch.Tensor, torch.Tensor]:
    tokenizer.padding_side = "right"
    batch = tokenizer(texts, padding=True, return_tensors="pt").to(args.device)
    ids, mask = batch["input_ids"], batch["attention_mask"]
    if args.sink_prefix:
        ids, mask = prepend_sink(ids, mask, tokenizer.convert_tokens_to_ids("<|endoftext|>"))
    return ids, mask


@torch.no_grad()
def write_corpus(memory, tokenizer, passages: list[str], args) -> dict[int, torch.Tensor]:
    # first position skipped: a content token, or the sink with --sink-prefix
    total = sum(max(len(tokenizer(p)["input_ids"]) - (0 if args.sink_prefix else 1), 0) for p in passages)
    state = None
    for start in range(0, len(passages), args.write_batch_size):
        ids, attention = passage_batch(tokenizer, passages[start : start + args.write_batch_size], args)
        keys, values, mask = memory.capture_passages(ids, attention)
        state = memory.build_state(keys, values, mask, initial=state, total_tokens=total)
    return state


def cached_state(memory, tokenizer, passages, args, name: str) -> dict[int, torch.Tensor]:
    extra = {"key_expand": args.key_expand} if args.key_expand else {}
    if args.sink_prefix:  # changes the written state; old cache keys unchanged without it
        extra["sink_prefix"] = True
    model = {} if args.model == "Qwen/Qwen3-1.7B" else {"model": args.model, "revision": args.revision}
    key = json.dumps({**model, "rule": args.rule, "beta": args.beta, "beta_norm": args.beta_norm, "center": True, **extra,
                      "passages": hashlib.sha256("\n".join(passages).encode()).hexdigest()}, sort_keys=True)
    path = args.data / "states" / f"{name}-{hashlib.sha256(key.encode()).hexdigest()[:16]}.pt"
    if path.exists():
        return {int(k): v.to(args.device) for k, v in torch.load(path).items()}
    started = time.perf_counter()
    state = write_corpus(memory, tokenizer, passages, args)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({k: v.cpu() for k, v in state.items()}, path)
    print(json.dumps({"wrote": name, "passages": len(passages), "seconds": round(time.perf_counter() - started, 1)}),
          flush=True)
    return state


def train(model, memory, tokenizer, formatter, rows, state, params, loras, args) -> list[float]:
    rng = random.Random(args.seed)
    order = list(range(len(rows)))
    optimizer = torch.optim.AdamW(params, weight_decay=0.0)
    end = tokenizer.convert_tokens_to_ids("<|im_end|>")
    losses = []
    model.train(False)
    set_lora(loras, True)
    for epoch in range(args.epochs):
        rng.shuffle(order)
        for start in range(0, len(order), args.batch_size):
            batch_rows = [rows[i] for i in order[start : start + args.batch_size]]
            ids, labels = [], []
            for r in batch_rows:
                prompt = tokenizer(formatter(render_closed(r["question"])))["input_ids"]
                answer = tokenizer(r["answer_surface"], add_special_tokens=False)["input_ids"] + [end]
                ids.append(prompt + answer)
                labels.append([-100] * len(prompt) + answer)
            pad = tokenizer.pad_token_id
            total = sum(len(a) - a.count(-100) - (a[0] != -100) for a in labels)  # target tokens in the batch
            optimizer.zero_grad(set_to_none=True)
            loss_sum = 0.0
            # micro-batches with token-weighted accumulation: same gradient as the whole batch (4B memory)
            for m in range(0, len(ids), args.micro_batch or len(ids)):
                mids, mlabels = ids[m : m + (args.micro_batch or len(ids))], labels[m : m + (args.micro_batch or len(ids))]
                width = max(map(len, mids))
                input_ids = torch.tensor([x + [pad] * (width - len(x)) for x in mids], device=args.device)
                label_ids = torch.tensor([x + [-100] * (width - len(x)) for x in mlabels], device=args.device)
                attention = torch.tensor([[1] * len(x) + [0] * (width - len(x)) for x in mids], device=args.device)
                with memory.reading(state):
                    hidden = model.model(input_ids=input_ids, attention_mask=attention,
                                         use_cache=False).last_hidden_state
                # lm_head only at answer positions (same loss as full logits; full logits OOM for 4B)
                targets = label_ids[:, 1:]
                keep = targets != -100
                logits = model.lm_head(hidden[:, :-1][keep])
                part = nn.functional.cross_entropy(logits.float(), targets[keep], reduction="sum")
                (part / total).backward()
                loss_sum += part.item()
            optimizer.step()
            loss = torch.tensor(loss_sum / total)
            losses.append(loss.item())
            step = len(losses)
            if step % 25 == 0:
                print(json.dumps({"epoch": epoch, "step": step, "loss": round(sum(losses[-25:]) / 25, 4)}),
                      flush=True)
    return losses


@torch.no_grad()
def generate(model, memory, tokenizer, prompts: list[str], state, args) -> list[str]:
    out = []
    tokenizer.padding_side = "left"
    for start in range(0, len(prompts), args.gen_batch_size):
        inputs = tokenizer(prompts[start : start + args.gen_batch_size], padding=True,
                           return_tensors="pt").to(args.device)
        with memory.reading(state):
            generated = model.generate(**inputs, do_sample=False, max_new_tokens=args.max_new_tokens,
                                       pad_token_id=tokenizer.pad_token_id, top_p=None, top_k=None,
                                       temperature=None)
        out += [tokenizer.decode(g, skip_special_tokens=True).strip()
                for g in generated[:, inputs["input_ids"].shape[1]:]]
    return out


def evaluate(name, rows, prompts, preds, args, counterfactual=False) -> dict:
    path = args.output_dir / f"{name}.jsonl"
    records = []
    for r, pred in zip(rows, preds, strict=True):
        exact, f1 = score_prediction(pred, r["aliases"])
        rec = {"question_id": r["question_id"], "group": r["group"], "prediction": pred, "exact_match": exact,
               "f1": f1}
        if counterfactual:
            cf_exact, _ = score_prediction(pred, r["counterfactual"]["aliases"])
            rec.update({"cf_follow_exact_match": cf_exact, "cf_valid": r["counterfactual"]["valid"]})
        records.append(rec)
    path.write_text("".join(json.dumps(x, ensure_ascii=False, sort_keys=True) + "\n" for x in records))
    summary = {"run": name, "n": len(records), "em": round(100 * sum(x["exact_match"] for x in records) / len(records), 2)}
    if counterfactual:
        valid = [x for x in records if x["cf_valid"]]
        summary["cf_follow"] = round(100 * sum(x["cf_follow_exact_match"] for x in valid) / max(len(valid), 1), 2)
    print(json.dumps(summary), flush=True)
    return summary


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    questions = read_jsonl(args.data / "questions.jsonl")
    corpus = read_jsonl(args.data / "corpus.jsonl")
    calibration = read_jsonl(args.banks_dir / "calibration.jsonl")

    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, revision=args.revision, dtype=torch.bfloat16,
                                                 local_files_only=True).to(args.device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    formatter = QwenChatPromptFormatter(tokenizer)

    loras: list[LoRALinear] = []
    if args.stage == "lora":
        for layer in model.model.layers:
            for name in ("q_proj", "o_proj"):
                wrapped = LoRALinear(getattr(layer.self_attn, name), args.lora_rank, 2.0 * args.lora_rank)
                setattr(layer.self_attn, name, wrapped)
                loras.append(wrapped)

    memory = DeltaMemoryQwen(model, DeltaMemoryConfig(rule=args.rule, beta=args.beta, beta_norm=args.beta_norm,
                                                     gate=args.gate, center_keys=True,
                                                     key_expand=args.key_expand, center_queries=args.center_queries))
    # Writing (and the key mean) always uses frozen Qwen.
    set_lora(loras, False)
    with torch.no_grad():
        tokenizer.padding_side = "right"
        cal = [r["passage"] for r in calibration[:256]]
        keys, masks = [], []
        for start in range(0, len(cal), args.write_batch_size):
            k, _, m = memory.capture_passages(*passage_batch(tokenizer, cal[start : start + args.write_batch_size], args))
            keys.append(k[:, :, m])
            masks.append(m[m])
        memory.calibrate_key_mean(torch.cat(keys, 2), torch.cat(masks))
    state = cached_state(memory, tokenizer, [c["passage"] for c in corpus], args, "memory")
    cf_state = cached_state(memory, tokenizer,
                            [c["counterfactual_passage"] or c["passage"] for c in corpus], args, "counterfactual")
    zero = null_state(state)

    params: list[nn.Parameter] = []
    if args.stage in ("gate", "lora"):
        for index in memory.layer_indices:
            gate = nn.Parameter(torch.full((memory.num_heads,), args.gate, device=args.device))
            memory.gates[index] = gate
            params.append(gate)
    lora_params = [p for m in loras for p in (m.down.weight, m.up.weight)]
    for p in lora_params:
        p.requires_grad_(True)

    losses = []
    started = time.perf_counter()
    if args.stage != "frozen":
        train_rows = [q for q in questions if q["group"] in ("train_seen", "train_other")][: args.train_limit]
        groups = [{"params": params, "lr": args.lr_gate}]
        if lora_params:
            groups.append({"params": lora_params, "lr": args.lr_lora})
        losses = train(model, memory, tokenizer, formatter, train_rows, zero if args.train_null else state,
                       groups, loras, args)
        torch.save({"gates": {k: v.detach().cpu() for k, v in memory.gates.items()},
                    "lora": {i: (m.down.weight.detach().cpu(), m.up.weight.detach().cpu()) for i, m in enumerate(loras)}},
                   args.output_dir / "trained.pt")
    train_seconds = time.perf_counter() - started

    set_lora(loras, True)
    eval_rows = {g: [q for q in questions if q["group"] == g][: args.eval_limit]
                 for g in args.eval_groups.split(",")}
    summaries = []
    gates = [float(g) for g in args.gates.split(",")] if args.gates else [args.gate]
    for group, rows in eval_rows.items():
        prompts = [formatter(render_closed(r["question"])) for r in rows]
        for gate in gates:
            memory.config.gate = gate
            tag = f"-g{gate}" if args.gates else ""
            summaries.append(evaluate(f"memory-{group}{tag}", rows, prompts,
                                      generate(model, memory, tokenizer, prompts, state, args), args))
            if group == "dev_in" and args.gates:
                summaries.append(evaluate(f"counterfactual-{group}{tag}", rows, prompts,
                                          generate(model, memory, tokenizer, prompts, cf_state, args), args,
                                          counterfactual=True))
        memory.config.gate = args.gate
        if args.stage != "frozen":  # frozen null == closed book
            summaries.append(evaluate(f"null-{group}", rows, prompts,
                                      generate(model, memory, tokenizer, prompts, zero, args), args))
        if group == "dev_in" and not args.gates:
            summaries.append(evaluate(f"counterfactual-{group}", rows, prompts,
                                      generate(model, memory, tokenizer, prompts, cf_state, args), args,
                                      counterfactual=True))
    if args.references:
        set_lora(loras, False)
        bm25 = BM25([c["passage"] for c in corpus])
        for group, rows in eval_rows.items():
            prompts = [formatter(render_closed(r["question"])) for r in rows]
            summaries.append(evaluate(f"closed_book-{group}", rows, prompts,
                                      generate(model, memory, tokenizer, prompts, None, args), args))
            rag = [formatter(render_context(r["question"], [corpus[bm25.top(r["question"], 1)[0]]["passage"]]))
                   for r in rows]
            summaries.append(evaluate(f"rag_bm25@1-{group}", rows, rag,
                                      generate(model, memory, tokenizer, rag, None, args), args))

    meta = {**{k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            "data_manifest_sha256": hashlib.sha256((args.data / "manifest.json").read_bytes()).hexdigest(),
            "corpus_passages": len(corpus), "trainable_parameters": sum(p.numel() for p in params + lora_params),
            "optimizer_steps": len(losses), "train_seconds": round(train_seconds, 1),
            "final_loss_25": round(sum(losses[-25:]) / max(len(losses[-25:]), 1), 4) if losses else None,
            "repository_commit": git_revision(), "working_tree_dirty": git_dirty(), "torch": torch.__version__,
            "prompt": "triviaqa_short_answer_v1", "summaries": summaries}
    (args.output_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")


if __name__ == "__main__":
    main()
