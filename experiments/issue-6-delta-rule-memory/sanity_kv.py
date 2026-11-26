#!/usr/bin/env python3
"""Sanity check: can the delta-rule memory store key->value pairs and return
the right value when read with a key?

Part 1 (synthetic, no LLM): N random (key, value) pairs, d_k = d_v = 128,
written with the same code as the experiments (``write_chunked`` for delta,
outer-product sum for Hebbian), read with the same keys. Keys are iid random,
or correlated (shared direction, like LLM keys), optionally centred and
DPFP-expanded (D = 256 nu). Metric: top-1 = the read-out's nearest stored value
(cosine) is the right one.

Part 2 (real facts): N TriviaQA facts from the training split. Key = frozen
Qwen hidden state (mean over tokens at ``--layer``, first/sink token excluded) of the *question* (exact
lookup) or of the *passage* (the issue-#6 setting: passage written, question
asks). Value = Qwen input embedding of the answer (mean over its tokens).
Keys are standardised per dimension with the written bank's mean and std, then
L2-normalised, optionally DPFP. Read
with the question's key; top-1 = nearest stored answer value is the gold
answer string. Chance = 1/N.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from run_banks import read_jsonl  # noqa: E402

from memwmlm.memory import delta_rule
from memwmlm.models.delta_qwen import dpfp

SIZES = (16, 64, 128, 256, 512, 1024, 2048)


def store_and_read(keys, values, rule, chunk=512):
    """keys [N, D] unit, values [N, V] -> read-out with the same keys [N, V]."""

    if rule == "delta":
        state = delta_rule.empty_state((), keys.shape[-1], values.shape[-1], device=keys.device)
        state = delta_rule.write_chunked(state, keys, values, torch.ones(keys.shape[0], device=keys.device), chunk)
    else:
        state = keys.T @ values
    return state


def top1(read, values, gold_index):
    """Fraction of rows whose nearest value (cosine) is the gold one; plus mean cosine to gold."""

    sim = F.normalize(read, dim=-1) @ F.normalize(values, dim=-1).T
    hit = (sim.argmax(-1) == gold_index).float().mean().item()
    cos = F.cosine_similarity(read, values[gold_index], dim=-1).mean().item()
    return hit, cos


def prep_keys(raw, center, nu):
    k = raw - raw.mean(0, keepdim=True) if center else raw
    k = F.normalize(k, dim=-1)
    if nu:
        k = F.normalize(dpfp(k, nu), dim=-1)
    return k


def synthetic(device, seed):
    g = torch.Generator(device="cpu").manual_seed(seed)
    rows = []
    for n in SIZES:
        values = torch.randn(n, 128, generator=g).to(device)
        iid = torch.randn(n, 128, generator=g).to(device)
        shared = torch.randn(1, 128, generator=g).to(device)
        corr = iid + 1.4 * shared * iid.norm(dim=-1, keepdim=True) / shared.norm()  # mean cos ~0.5
        variants = {"iid": (iid, False, 0), "correlated": (corr, False, 0),
                    "correlated+centred": (corr, True, 0), "correlated+centred+dpfp16": (corr, True, 16)}
        for name, (raw, center, nu) in variants.items():
            k = prep_keys(raw, center, nu)
            gram = k @ k.T
            mean_cos = ((gram.sum() - n) / (n * (n - 1))).item()
            for rule in ("delta", "hebbian"):
                state = store_and_read(k, values, rule)
                hit, cos = top1(delta_rule.read(state, k), values, torch.arange(n, device=device))
                rows.append({"part": "synthetic", "keys": name, "D": k.shape[-1], "N": n, "rule": rule,
                             "mean_key_cos": round(mean_cos, 3), "top1": round(hit, 3), "cos": round(cos, 3)})
                print(json.dumps(rows[-1]), flush=True)
    return rows


@torch.no_grad()
def hidden_means(model, tokenizer, texts, layer, device, batch=64):
    out = []
    for i in range(0, len(texts), batch):
        enc = tokenizer(texts[i:i + batch], return_tensors="pt", padding=True).to(device)
        hs = model(**enc, output_hidden_states=True).hidden_states[layer].float()
        m = enc["attention_mask"].unsqueeze(-1).float().clone()
        m[:, 0] = 0  # first token = attention sink with a massive activation; its 1/length weight swamps the mean
        out.append((hs * m).sum(1) / m.sum(1))
    return torch.cat(out)


@torch.no_grad()
def embed_means(model, tokenizer, texts, device):
    emb = model.get_input_embeddings()
    out = []
    for t in texts:
        ids = tokenizer(t, return_tensors="pt", add_special_tokens=False)["input_ids"].to(device)
        out.append(emb(ids)[0].float().mean(0))
    return torch.stack(out)


def facts(args):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision, local_files_only=True)
    tokenizer.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(args.model, revision=args.revision, dtype=torch.bfloat16,
                                                 local_files_only=True).to(args.device).eval()
    pool = [q for q in read_jsonl(args.data / "questions.jsonl") if q["split"] == "train" and q.get("passage")]
    gen = torch.Generator().manual_seed(args.seed)
    pool = [pool[i] for i in torch.randperm(len(pool), generator=gen).tolist()][: max(SIZES)]
    q_hid = hidden_means(model, tokenizer, [r["question"] for r in pool], args.layer, args.device)
    p_hid = hidden_means(model, tokenizer, [r["passage"] for r in pool], args.layer, args.device)
    answers = [r["answer_surface"] for r in pool]
    a_emb = embed_means(model, tokenizer, answers, args.device)
    rows = []
    for n in SIZES:
        if n > len(pool):
            break
        values = a_emb[:n]
        # gold = any stored row with the same answer string (duplicates count as correct)
        same = torch.tensor([[answers[i] == answers[j] for j in range(n)] for i in range(n)], device=args.device)
        for written, raw_w in (("question", q_hid[:n]), ("passage", p_hid[:n])):
            mean, std = raw_w.mean(0, keepdim=True), raw_w.std(0, keepdim=True) + 1e-6
            for nu in (0, 2):
                def prep(x):
                    k = F.normalize((x - mean) / std, dim=-1)  # standardise with the written bank's statistics
                    return F.normalize(dpfp(k, nu), dim=-1) if nu else k
                k_write, k_read = prep(raw_w), prep(q_hid[:n])
                for rule in ("delta", "hebbian"):
                    state = store_and_read(k_write, values, rule)
                    read = delta_rule.read(state, k_read)
                    sim = F.normalize(read, dim=-1) @ F.normalize(values, dim=-1).T
                    hit = same[torch.arange(n), sim.argmax(-1)].float().mean().item()
                    # addressing alone, no memory: is the question key nearest to its own written key?
                    addr = same[torch.arange(n), (k_read @ k_write.T).argmax(-1)].float().mean().item()
                    rows.append({"part": "facts", "written_key": written, "read_key": "question",
                                 "D": k_write.shape[-1], "N": n, "rule": rule, "layer": args.layer,
                                 "top1": round(hit, 3), "nearest_key_top1": round(addr, 3), "chance": round(1 / n, 4)})
                    print(json.dumps(rows[-1]), flush=True)
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--part", choices=("synthetic", "facts", "both"), default="both")
    p.add_argument("--data", type=Path, default=Path("runs/delta-rule/global-v1"))
    p.add_argument("--model", default="Qwen/Qwen3-1.7B")
    p.add_argument("--revision", default="70d244cc86ccca08cf5af4e1e306ecf908b1ad5e")
    p.add_argument("--layer", type=int, default=14)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    rows = []
    if args.part in ("synthetic", "both"):
        rows += synthetic(args.device, args.seed)
    if args.part in ("facts", "both"):
        rows += facts(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, indent=1) + "\n")


if __name__ == "__main__":
    main()
