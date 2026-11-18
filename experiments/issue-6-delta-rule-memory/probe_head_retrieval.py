#!/usr/bin/env python3
"""Diagnostic for Variant 1: can frozen Qwen heads *address* the right passage?

For banks of N calibration passages, each passage is encoded independently and
its pre-RoPE, ``k_norm``-ed, L2-normalised keys are captured per layer/KV head.
Each question is rendered with the closed-book prompt and its pre-RoPE queries
are captured. A passage's score under head (l, h) is the maximum cosine
similarity between any query token and any of its key tokens (``maxsim``), or
that of the final prompt token only (``last``). Top-1 retrieval accuracy of the
question's own passage is reported per head, with BM25 and chance (1/N) as
references.

This measures content addressing only; it does not test the linear delta-rule
readout or answer generation. Uses the calibration (train) split only.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from run_banks import BM25, read_jsonl, render_closed  # noqa: E402

from memwmlm.models.delta_qwen import DeltaMemoryConfig, DeltaMemoryQwen
from memwmlm.models.qwen import QwenChatPromptFormatter


@torch.no_grad()
def capture(memory, tokenizer, texts, device, queries=False):
    """Return per-text lists of [L, H, T, D] (keys if not queries) with pads/first token removed."""

    out = []
    memory.capture_queries = queries
    for start in range(0, len(texts), 32):
        chunk = texts[start : start + 32]
        tokenizer.padding_side = "right"
        batch = tokenizer(chunk, padding=True, return_tensors="pt").to(device)
        with memory.capturing() as caps:
            memory.qwen(**batch, use_cache=False)
        stack = torch.stack([torch.cat(caps[i].queries if queries else caps[i].keys) for i in memory.layer_indices])
        for b in range(len(chunk)):
            length = int(batch["attention_mask"][b].sum())
            out.append(F.normalize(stack[:, b, :, 1:length].float(), dim=-1))
    memory.capture_queries = False
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--banks-dir", type=Path, required=True)
    p.add_argument("--ladder", default="16,64,256")
    p.add_argument("--questions", type=int, default=256)
    p.add_argument("--model", default="Qwen/Qwen3-1.7B")
    p.add_argument("--revision", default="70d244cc86ccca08cf5af4e1e306ecf908b1ad5e")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    rows = read_jsonl(args.banks_dir / "calibration.jsonl")[: args.questions]
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, revision=args.revision, dtype=torch.bfloat16, local_files_only=True
    ).to(args.device).eval()
    memory = DeltaMemoryQwen(model, DeltaMemoryConfig())
    formatter = QwenChatPromptFormatter(tokenizer)

    keys = capture(memory, tokenizer, [r["passage"] for r in rows], args.device)
    queries = capture(memory, tokenizer, [formatter(render_closed(r["question"])) for r in rows],
                      args.device, queries=True)
    groups = memory.groups
    L, H = queries[0].shape[:2]

    report = {"questions": len(rows), "layers": L, "query_heads": H, "rungs": {}}
    for n in (int(x) for x in args.ladder.split(",")):
        if len(rows) % n:
            continue
        hits = {"maxsim": torch.zeros(L, H), "last": torch.zeros(L, H)}
        bm25_hits = 0
        for start in range(0, len(rows), n):
            bank = list(range(start, start + n))
            bm25 = BM25([rows[i]["passage"] for i in bank])
            for local, qi in enumerate(bank):
                bm25_hits += int(bm25.top(rows[qi]["question"], 1)[0] == local)
                q = queries[qi].to(args.device)  # [L, H, Tq, D]
                scores_max, scores_last = [], []
                for pi in bank:
                    k = keys[pi].repeat_interleave(groups, dim=1).to(args.device)  # [L, H, Tp, D]
                    sim = torch.einsum("lhqd,lhkd->lhqk", q, k)
                    scores_max.append(sim.amax(dim=(2, 3)))
                    scores_last.append(sim[:, :, -1, :].amax(dim=-1))
                hits["maxsim"] += (torch.stack(scores_max).argmax(0) == local).float().cpu()
                hits["last"] += (torch.stack(scores_last).argmax(0) == local).float().cpu()
        rung = {"chance": 1 / n, "bm25_top1": bm25_hits / len(rows)}
        for name, h in hits.items():
            acc = h / len(rows)
            flat = acc.flatten()
            best = flat.topk(5)
            rung[name] = {
                "median_head": float(flat.median()),
                "mean_head": float(flat.mean()),
                "best5": [{"layer": int(i // H), "head": int(i % H), "top1": float(v)}
                          for v, i in zip(best.values, best.indices)],
                "per_layer_best": [float(acc[l].max()) for l in range(L)],
                "per_head": acc.tolist(),  # [layer][query_head]
            }
        report["rungs"][str(n)] = rung
        print(json.dumps({"N": n, "chance": rung["chance"], "bm25": rung["bm25_top1"],
                          "maxsim_best": rung["maxsim"]["best5"][0], "last_best": rung["last"]["best5"][0],
                          "maxsim_median": rung["maxsim"]["median_head"]}), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
