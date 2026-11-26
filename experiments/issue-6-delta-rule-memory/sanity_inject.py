#!/usr/bin/env python3
"""Sanity check, step 2: does frozen Qwen answer from what the memory returns?

N TriviaQA training facts are written into one delta-rule state (``D`` x
``slots * d_model``). Key = standardised, mean-pooled Qwen hidden state of the
question (or of the passage, the issue-#6 setting) at ``--layer`` (sink token
excluded), DPFP-expanded. Value = the input embeddings of the answer's tokens
(``" " + answer``), up to ``--slots`` tokens, zero-padded. A read with the
question key returns ``slots`` vectors; slots whose norm is below half the
mean token-embedding norm are dropped (end of answer). The kept vectors are put
into Qwen's input as soft tokens after "Memory:" and Qwen answers greedily.

Conditions (EM % on ``--eval`` questions of the bank):
  closed_book       plain closed-book prompt (no memory line)
  text_hint         the gold answer typed as text after "Memory:" (upper bound)
  oracle_soft       the exact stored embeddings as soft tokens (= text_hint if
                    the soft-token path is right)
  mem_soft_q        memory read, question keys written
  mem_snap_q        same read, each slot snapped to its nearest vocabulary token
                    and inserted as text
  mem_soft_p / mem_snap_p   passage keys written, question keys read
  shuffled_soft_q   values permuted across facts before writing: "follow" =
                    Qwen gives the answer that was written under its key
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from run_banks import INSTRUCTION, read_jsonl, render_closed, score_prediction  # noqa: E402
from sanity_kv import hidden_means  # noqa: E402

from memwmlm.memory import delta_rule
from memwmlm.models.delta_qwen import dpfp
from memwmlm.models.qwen import QwenChatPromptFormatter

MARK = "@@MEM@@"


def memory_prompt(question: str) -> str:
    return f"{INSTRUCTION}\n\nMemory:{MARK}\n\nQuestion: {question}\nAnswer:"


@torch.no_grad()
def generate(model, tokenizer, formatter, questions, soft, device, batch=32, max_new=16):
    """soft: per question None (closed-book), str (text after Memory:) or Tensor [m, d] (soft tokens)."""

    emb = model.get_input_embeddings()
    preds = []
    for i in range(0, len(questions), batch):
        seqs = []
        for q, s in zip(questions[i:i + batch], soft[i:i + batch], strict=True):
            if s is None:
                ids = tokenizer(formatter(render_closed(q)), add_special_tokens=False)["input_ids"]
                seqs.append(emb(torch.tensor(ids, device=device)))
                continue
            text = formatter(memory_prompt(q))
            pre, post = text.split(MARK)
            if isinstance(s, str):
                ids = tokenizer(pre + s + post, add_special_tokens=False)["input_ids"]
                seqs.append(emb(torch.tensor(ids, device=device)))
                continue
            a = emb(torch.tensor(tokenizer(pre, add_special_tokens=False)["input_ids"], device=device))
            b = emb(torch.tensor(tokenizer(post, add_special_tokens=False)["input_ids"], device=device))
            seqs.append(torch.cat([a, s.to(a.dtype), b]))
        width = max(x.shape[0] for x in seqs)
        pad = emb(torch.tensor([tokenizer.pad_token_id], device=device))[0]
        embeds = torch.stack([torch.cat([pad.expand(width - x.shape[0], -1), x]) for x in seqs])
        mask = torch.tensor([[0] * (width - x.shape[0]) + [1] * x.shape[0] for x in seqs], device=device)
        out = model.generate(inputs_embeds=embeds, attention_mask=mask, max_new_tokens=max_new, do_sample=False,
                             pad_token_id=tokenizer.pad_token_id)
        preds += [tokenizer.decode(g, skip_special_tokens=True).strip() for g in out]
    return preds


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=Path("runs/delta-rule/global-v1"))
    p.add_argument("--model", default="Qwen/Qwen3-1.7B")
    p.add_argument("--revision", default="70d244cc86ccca08cf5af4e1e306ecf908b1ad5e")
    p.add_argument("--layer", type=int, default=27)
    p.add_argument("--nu", type=int, default=2)
    p.add_argument("--sizes", default="256,2048")
    p.add_argument("--eval", type=int, default=256)
    p.add_argument("--slots", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, revision=args.revision, dtype=torch.bfloat16,
                                                 local_files_only=True).to(args.device).eval()
    formatter = QwenChatPromptFormatter(tokenizer)
    emb_w = model.get_input_embeddings().weight.float()
    emb_norm = emb_w.norm(dim=-1).mean()

    sizes = [int(x) for x in args.sizes.split(",")]
    pool = [q for q in read_jsonl(args.data / "questions.jsonl") if q["split"] == "train" and q.get("passage")]
    gen = torch.Generator().manual_seed(args.seed)
    pool = [pool[i] for i in torch.randperm(len(pool), generator=gen).tolist()][: max(sizes)]
    tokenizer.padding_side = "right"
    q_hid = hidden_means(model, tokenizer, [r["question"] for r in pool], args.layer, args.device)
    p_hid = hidden_means(model, tokenizer, [r["passage"] for r in pool], args.layer, args.device)
    tokenizer.padding_side = "left"

    d = emb_w.shape[1]
    values = torch.zeros(len(pool), args.slots, d, device=args.device)
    lengths = []
    for i, r in enumerate(pool):
        ids = tokenizer(" " + r["answer_surface"], add_special_tokens=False)["input_ids"][: args.slots]
        values[i, : len(ids)] = emb_w[ids]
        lengths.append(len(ids))

    def keys(raw, bank):
        mean, std = bank.mean(0, keepdim=True), bank.std(0, keepdim=True) + 1e-6
        k = F.normalize((raw - mean) / std, dim=-1)
        return F.normalize(dpfp(k, args.nu), dim=-1) if args.nu else k

    def read_soft(k_write, k_read, vals):
        state = delta_rule.empty_state((), k_write.shape[-1], vals[0].numel(), device=args.device)
        state = delta_rule.write_chunked(state, k_write, vals.flatten(1), torch.ones(len(k_write), device=args.device), 256)
        out = delta_rule.read(state, k_read).view(len(k_read), args.slots, d)
        soft, snap = [], []
        for slots in out:
            kept = slots[: max(1, int((slots.norm(dim=-1) > 0.5 * emb_norm).sum()))]
            soft.append(kept)
            ids = (F.normalize(kept, dim=-1) @ F.normalize(emb_w, dim=-1).T).argmax(-1)
            snap.append(tokenizer.decode(ids))
        return soft, snap

    rows = []
    for n in sizes:
        bank = pool[:n]
        ev = bank[: args.eval]
        qs = [r["question"] for r in ev]
        conds = {"closed_book": [None] * len(ev),
                 "text_hint": [" " + r["answer_surface"] for r in ev],
                 "oracle_soft": [values[i, : lengths[i]] for i in range(len(ev))]}
        k_read = keys(q_hid[:n], q_hid[:n])[: len(ev)]
        soft, snap = read_soft(keys(q_hid[:n], q_hid[:n]), k_read, values[:n])
        conds["mem_soft_q"], conds["mem_snap_q"] = soft, snap
        k_read_p = keys(q_hid[:n], p_hid[:n])[: len(ev)]  # question read with the passage-bank statistics
        soft, snap = read_soft(keys(p_hid[:n], p_hid[:n]), k_read_p, values[:n])
        conds["mem_soft_p"], conds["mem_snap_p"] = soft, snap
        perm = torch.roll(torch.arange(n), 1)  # fact i stores the answer of fact i-1
        soft, _ = read_soft(keys(q_hid[:n], q_hid[:n]), k_read, values[perm])
        conds["shuffled_soft_q"] = soft
        for name, s in conds.items():
            t0 = time.time()
            preds = generate(model, tokenizer, formatter, qs, s, args.device)
            em = [score_prediction(pr, r["aliases"])[0] for pr, r in zip(preds, ev, strict=True)]
            row = {"N": n, "condition": name, "n": len(ev), "em": round(100 * sum(em) / len(em), 1),
                   "seconds": round(time.time() - t0, 1)}
            if name == "shuffled_soft_q":
                src = [pool[int(perm[i])] for i in range(len(ev))]
                row["follow"] = round(100 * sum(score_prediction(pr, r["aliases"])[0]
                                                for pr, r in zip(preds, src, strict=True)) / len(ev), 1)
            if name.startswith("mem_snap"):
                row["snap_exact"] = round(100 * sum(sn.strip() == r["answer_surface"].strip()
                                                    for sn, r in zip(s, ev, strict=True)) / len(ev), 1)
            row["examples"] = [{"q": r["question"], "gold": r["answer_surface"], "pred": pr,
                                **({"snap": s[i]} if isinstance(s[i], str) and name.startswith("mem_snap") else {})}
                               for i, (r, pr) in enumerate(zip(ev[:4], preds[:4]))]
            print(json.dumps({k: v for k, v in row.items() if k != "examples"}), flush=True)
            rows.append(row)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, indent=1, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
