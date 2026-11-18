#!/usr/bin/env python3
"""Cache frozen-Qwen layer-l hidden states of WikiText cloze passages (exact bf16).

Every passage is encoded alone from position 0 (as in training), so its
hidden states do not depend on the other passages of an episode; caching is
exact. Output per split: ``hidden-L{l}/{split}.bin`` (bf16 bits as int16,
``[tokens, hidden]``, first token of each passage excluded) and
``{split}.offsets.npy`` (``[units + 1]`` token offsets in jsonl order).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from run_banks import read_jsonl  # noqa: E402
from train_v2 import cache_dir_for, load_model  # noqa: E402

from memwmlm.models.delta_interface import DeltaInterface, InterfaceQwen


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=Path("runs/delta-rule/wikitext-cloze-v1"))
    p.add_argument("--layer", type=int, default=14)
    p.add_argument("--splits", default="val,train")
    p.add_argument("--batch-tokens", type=int, default=16384)
    p.add_argument("--model", default="Qwen/Qwen3-1.7B")
    p.add_argument("--revision", default="70d244cc86ccca08cf5af4e1e306ecf908b1ad5e")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    tokenizer, model = load_model(args)
    wrapped = InterfaceQwen(model, DeltaInterface(model.config.hidden_size, 8).to(args.device), args.layer)
    out = cache_dir_for(args.data, args.layer, args.model, args.revision)
    out.mkdir(parents=True, exist_ok=True)
    hidden = model.config.hidden_size
    for split in args.splits.split(","):
        rows = read_jsonl(args.data / f"{split}.jsonl")
        ids = [tokenizer(r["passage"])["input_ids"] for r in rows]
        lengths = np.array([len(x) - 1 for x in ids], dtype=np.int64)
        offsets = np.concatenate([[0], np.cumsum(lengths)])
        store = np.memmap(out / f"{split}.bin", dtype=np.int16, mode="w+", shape=(int(offsets[-1]), hidden))
        order = sorted(range(len(rows)), key=lambda i: len(ids[i]))
        start = 0
        while start < len(order):
            width = len(ids[order[start]])
            end = start
            while end < len(order) and (end - start + 1) * len(ids[order[end]]) <= args.batch_tokens:
                end += 1
            batch = order[start : max(end, start + 1)]
            width = max(len(ids[i]) for i in batch)
            input_ids = torch.full((len(batch), width), tokenizer.pad_token_id)
            mask = torch.zeros((len(batch), width), dtype=torch.long)
            for row, i in enumerate(batch):
                input_ids[row, : len(ids[i])] = torch.tensor(ids[i])
                mask[row, : len(ids[i])] = 1
            h = wrapped.layer_hidden(input_ids.to(args.device), mask.to(args.device))
            bits = h.view(torch.int16).cpu().numpy()
            for row, i in enumerate(batch):
                store[offsets[i] : offsets[i + 1]] = bits[row, 1 : len(ids[i])]
            start += len(batch)
        store.flush()
        np.save(out / f"{split}.offsets.npy", offsets)
        print(json.dumps({"split": split, "units": len(rows), "tokens": int(offsets[-1])}), flush=True)
    (out / "manifest.json").write_text(json.dumps({"layer": args.layer, "model": args.model,
                                                   "revision": args.revision, "dtype": "bfloat16-bits-int16",
                                                   "first_token_excluded": True}, indent=2) + "\n")


if __name__ == "__main__":
    main()
