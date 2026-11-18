#!/usr/bin/env python3
"""Diagnostic for Variant 1: how much does a 128x128 head state retain?

Writes N calibration passages (pre-RoPE ``k_norm`` keys, L2-normalised) into the
per-layer/KV-head state with the delta or Hebbian rule, then reads with the
*stored keys themselves* and reports the mean cosine similarity between read-out
and stored value (self-retrieval fidelity), plus the number of written tokens.
A content-addressable memory that cannot return values for its own keys cannot
answer questions from them. Calibration (train) split only.
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
from run_banks import encode_passages, read_jsonl  # noqa: E402

from memwmlm.memory import delta_rule
from memwmlm.models.delta_qwen import DeltaMemoryConfig, DeltaMemoryQwen


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--banks-dir", type=Path, required=True)
    p.add_argument("--ladder", default="1,4,16,64,256")
    p.add_argument("--model", default="Qwen/Qwen3-1.7B")
    p.add_argument("--revision", default="70d244cc86ccca08cf5af4e1e306ecf908b1ad5e")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--write-batch-size", type=int, default=64)
    p.add_argument("--center-keys", action="store_true", help="subtract the bank's mean key before normalising")
    args = p.parse_args()

    rows = read_jsonl(args.banks_dir / "calibration.jsonl")
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, revision=args.revision, dtype=torch.bfloat16, local_files_only=True
    ).to(args.device).eval()
    memory = DeltaMemoryQwen(model, DeltaMemoryConfig())

    report = []
    for n in (int(x) for x in args.ladder.split(",")):
        keys, values, _ = encode_passages(memory, tokenizer, [r["passage"] for r in rows[:n]], args)
        # Control: passages never written into the state (rows after the last bank).
        h_keys, h_values, _ = encode_passages(memory, tokenizer, [r["passage"] for r in rows[-16:]], args)
        keys, h_keys = keys.float(), h_keys.float()
        if args.center_keys:
            key_mean = keys.mean(dim=2, keepdim=True)
            keys, h_keys = keys - key_mean, h_keys - key_mean
        keys, h_keys = F.normalize(keys, dim=-1), F.normalize(h_keys, dim=-1)
        values, h_values = values.float(), h_values.float()
        value_mean = values.mean(dim=2, keepdim=True)
        # mean pairwise |cos| between distinct stored keys (key collision level)
        sample = keys[:, :, : min(keys.shape[2], 512)]
        gram = torch.einsum("lhtd,lhsd->lhts", sample, sample).abs()
        t = sample.shape[2]
        off_diag = (gram.sum((-1, -2)) - t) / max(t * (t - 1), 1)
        entry = {"center_keys": args.center_keys, "N": n, "written_tokens": int(keys.shape[2]), "mean_abs_key_cos": float(off_diag.mean())}
        for rule in ("delta", "hebbian"):
            state0 = delta_rule.empty_state(keys.shape[:2], keys.shape[-1], values.shape[-1], device=keys.device)
            if rule == "hebbian":
                state = torch.einsum("lhtk,lhtv->lhkv", keys, values)
            else:
                state = delta_rule.write(state0, keys, values, rule="delta")
            read = delta_rule.read(state, keys)
            # value-mean-removed cosine: specific content, not the shared component
            cos = F.cosine_similarity(read - value_mean, values - value_mean, dim=-1)  # [L, H, T]
            raw_cos = F.cosine_similarity(read, values, dim=-1)
            per_layer = cos.mean(dim=(1, 2))
            h_read = delta_rule.read(state, h_keys)
            h_cos = F.cosine_similarity(h_read - value_mean, h_values - value_mean, dim=-1)
            entry[rule] = {
                "mean_cos_centered": float(cos.mean()),
                "mean_cos_raw": float(raw_cos.mean()),
                "unwritten_mean_cos_centered": float(h_cos.mean()),
                "written_minus_unwritten_best_layer": float((cos.mean(dim=(1, 2)) - h_cos.mean(dim=(1, 2))).max()),
                "first_passage_mean_cos": float(cos[:, :, : max(1, keys.shape[2] // n)].mean()),
                "last_passage_mean_cos": float(cos[:, :, -max(1, keys.shape[2] // n):].mean()),
                "best_layer": int(per_layer.argmax()),
                "best_layer_cos": float(per_layer.max()),
            }
        report.append(entry)
        print(json.dumps(entry), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
