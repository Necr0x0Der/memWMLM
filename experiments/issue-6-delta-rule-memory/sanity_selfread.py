#!/usr/bin/env python3
"""Sanity check 3: does the real Variant 1 memory return a stored value when
read with that token's own key?

The memory is filled exactly as in ``run_banks.py`` (frozen Qwen3, every
layer and KV head, sink prefix, centred keys and the chosen config). It is then
read through the model's own read path (``DeltaMemoryQwen._read``, gate 1),
with the *keys* of the written passages used as queries: no question, no
matching problem, only the memory's own storage.

Per head and token: ``cos`` = cosine(read-out, stored value), both centred with
the bank's mean value (raw values share a large common direction); ``top1`` =
the stored token whose value is nearest to the read-out is the right token
(chance 1 / stored tokens). Control ``unwritten``: keys of passages that were
*not* written; their ``cos`` to their own (never stored) values is what a read
gives by chance. Reported as means over the 224 heads (and the median head
for ``top1``).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from run_banks import encode_passages, read_jsonl  # noqa: E402

from memwmlm.models.delta_qwen import DeltaMemoryConfig, DeltaMemoryQwen

CONFIGS = {
    "A": {},  # centred, D 128 (capacity test A)
    "B": {"key_expand": 16},  # + DPFP, D 4,096
    "W": {"whiten": True},  # + whitening, D 128
    "WB": {"whiten": True, "key_expand": 16},
}


@torch.no_grad()
def read_with(memory: DeltaMemoryQwen, layer: int, keys: torch.Tensor) -> torch.Tensor:
    """keys [Hkv, T, D] used as queries -> read-out [Hkv, T, D] (gate 1)."""

    q = keys.repeat_interleave(memory.groups, dim=0)[None]  # [1, H, T, D]
    out = memory._read(layer, q)[0].transpose(0, 1)  # [H, T, D]
    return out[:: memory.groups]  # heads of one KV group read the same state with the same key


def head_metrics(read, values, idx, centre):
    """read [Hkv, S, D] for stored tokens idx; values [Hkv, T, D] all stored."""

    r = read - centre
    v = values - centre
    cos = F.cosine_similarity(r, v[:, idx], dim=-1).mean(dim=1)  # [Hkv]
    top1 = []
    vn = F.normalize(v, dim=-1)
    for h in range(r.shape[0]):
        best = (F.normalize(r[h], dim=-1) @ vn[h].T).argmax(dim=-1)
        top1.append((best == idx).float().mean())
    return cos, torch.stack(top1)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--banks-dir", type=Path, default=Path("runs/delta-rule/banks-v2"))
    p.add_argument("--model", default="Qwen/Qwen3-1.7B")
    p.add_argument("--revision", default="70d244cc86ccca08cf5af4e1e306ecf908b1ad5e")
    p.add_argument("--configs", default="A,B,W,WB")
    p.add_argument("--ladder", default="1,4,16,64,256,1024")
    p.add_argument("--passages-per-rung", type=int, default=64, help="small N: average over banks up to this many passages")
    p.add_argument("--sample-tokens", type=int, default=1024)
    p.add_argument("--rule", choices=("delta", "hebbian"), default="delta")
    p.add_argument("--device", default="cuda")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, revision=args.revision, dtype=torch.bfloat16,
                                                 local_files_only=True).to(args.device).eval()
    eval_rows = read_jsonl(args.banks_dir / "eval.jsonl")
    calib = read_jsonl(args.banks_dir / "calibration.jsonl")
    enc_args = SimpleNamespace(write_batch_size=64, device=args.device, sink_prefix=True)
    gen = torch.Generator().manual_seed(0)
    rows = []
    for name in args.configs.split(","):
        memory = DeltaMemoryQwen(model, DeltaMemoryConfig(
            rule=args.rule, beta=1.0, alpha=1.0, gate=1.0, center_keys=True, center_queries=True,
            skip_first_token=True, **CONFIGS[name]))
        k, _, m = encode_passages(memory, tokenizer, [r["passage"] for r in calib[:256]], enc_args)
        memory.calibrate_key_mean(k, m)
        del k
        for n in [int(x) for x in args.ladder.split(",")]:
            n_banks = max(1, args.passages_per_rung // n)
            acc = {"cos": [], "top1": [], "cos_unwritten": []}
            tokens = 0
            for b in range(n_banks):
                bank = [r["passage"] for r in eval_rows[b * n:(b + 1) * n]]
                other = [r["passage"] for r in calib[256 + b * min(n, 64):256 + (b + 1) * min(n, 64)]]
                keys, values, mask = encode_passages(memory, tokenizer, bank, enc_args)
                ukeys, uvalues, _ = encode_passages(memory, tokenizer, other, enc_args)
                t = keys.shape[2]
                tokens += t
                idx = torch.randperm(t, generator=gen)[: args.sample_tokens].to(args.device)
                uidx = torch.randperm(ukeys.shape[2], generator=gen)[: args.sample_tokens].to(args.device)
                cos_l, top_l, ucos_l = [], [], []
                all_layers = memory.layer_indices
                for li, layer in enumerate(all_layers):
                    memory.layer_indices = (layer,)  # write one layer at a time (N 1024 does not fit at once)
                    memory.state = memory.build_state(keys[li:li + 1], values[li:li + 1], mask)
                    memory.layer_indices = all_layers
                    vals = values[li].float()
                    centre = vals.mean(dim=1, keepdim=True)
                    read = read_with(memory, layer, keys[li][:, idx])
                    c, t1 = head_metrics(read, vals, idx, centre)
                    uread = read_with(memory, layer, ukeys[li][:, uidx])
                    uc = F.cosine_similarity(uread - centre, uvalues[li][:, uidx].float() - centre, dim=-1).mean(1)
                    cos_l.append(c), top_l.append(t1), ucos_l.append(uc)
                acc["cos"].append(torch.stack(cos_l))
                acc["top1"].append(torch.stack(top_l))
                acc["cos_unwritten"].append(torch.stack(ucos_l))
                memory.state = None
                del keys, values, ukeys, uvalues
                torch.cuda.empty_cache()
            cos = torch.stack(acc["cos"]).mean(0)  # [L, Hkv]
            top1 = torch.stack(acc["top1"]).mean(0)
            ucos = torch.stack(acc["cos_unwritten"]).mean(0)
            row = {"config": name, "rule": args.rule, "N": n, "banks": n_banks,
                   "tokens_per_bank": round(tokens / n_banks), "chance_top1": round(n_banks / tokens, 5),
                   "cos": round(cos.mean().item(), 3), "cos_unwritten": round(ucos.mean().item(), 3),
                   "top1_mean": round(top1.mean().item(), 3), "top1_median_head": round(top1.median().item(), 3),
                   "top1_best_head": round(top1.max().item(), 3),
                   "top1_by_layer": [round(x, 3) for x in top1.mean(1).tolist()]}
            print(json.dumps({k: v for k, v in row.items() if k != "top1_by_layer"}), flush=True)
            rows.append(row)
        memory.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, indent=1) + "\n")


if __name__ == "__main__":
    main()
