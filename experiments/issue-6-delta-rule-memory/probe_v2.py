#!/usr/bin/env python3
"""Diagnose a trained Variant 2 interface on WikiText validation cloze.

For bank sizes N, writes N validation sentences and asks queries about them,
reporting teacher-forced loss and greedy EM for: the real state, a state of N
*other* validation sentences (wrong memory), and S = 0 (closed-book). Separates
"retrieval works but capacity runs out" from "the read is a generic bias".
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from run_banks import read_jsonl, render_closed  # noqa: E402
from train_v2 import cache_dir_for, ClozeFormatter, HiddenCache, cloze_question, load_shots, batch_loss, cloze_em, generate, load_model  # noqa: E402

from memwmlm.models.delta_interface import DeltaInterface, InterfaceQwen
from memwmlm.models.qwen import QwenChatPromptFormatter


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--sizes", default="1,8,64,512,2048")
    p.add_argument("--queries", type=int, default=256)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    targs = argparse.Namespace(**ckpt["args"])
    targs.device = args.device
    tokenizer, model = load_model(targs)
    formatter = QwenChatPromptFormatter(tokenizer)
    interface = DeltaInterface(model.config.hidden_size, targs.width,
                               shift_values=getattr(targs, "shift_values", False),
                               bias_state=getattr(targs, "bias_state", False)).to(args.device)
    interface.load_state_dict(ckpt["interface"])
    interface.eval()
    wrapped = InterfaceQwen(model, interface, targs.layer)
    data = Path(targs.data)
    val = read_jsonl(data / "val.jsonl")
    formatter = ClozeFormatter(tokenizer, load_shots(data, int(getattr(targs, "fewshot", 0))))
    if getattr(targs, "full_cloze", False):
        for r in val:
            r["cloze"] = r["passage"].replace(r["answer"], "___", 1)
    cache = HiddenCache(cache_dir_for(data, targs.layer, targs.model, targs.revision), "val", model.config.hidden_size)
    rng = random.Random(3)
    results = []
    for n in (int(x) for x in args.sizes.split(",")):
        # queries spread over several banks so small N still gets `queries` questions
        perm = list(range(len(val)))
        rng.shuffle(perm)
        asked_rows, losses = {"real": [], "wrong": [], "null": []}, {"real": [], "wrong": [], "null": []}
        preds = {"real": [], "wrong": [], "null": []}
        answers, start = [], 0
        while len(answers) < args.queries and start + 2 * n <= len(perm):
            bank, other = perm[start : start + n], perm[start + n : start + 2 * n]
            start += 2 * n
            q = rng.sample(bank, min(n, args.queries - len(answers), 64))
            prompts = [cloze_question(val[i]["cloze"]) for i in q]
            ans = [val[i]["answer"] for i in q]
            with torch.no_grad():
                states = {"real": interface.write(cache.tokens(bank, args.device), chunk_size=1024,
                                                  lengths=cache.lengths(bank)),
                          "wrong": interface.write(cache.tokens(other, args.device), chunk_size=1024,
                                                   lengths=cache.lengths(other)),
                          "null": None}
                for k, s in states.items():
                    losses[k].append(batch_loss(model, wrapped, tokenizer, formatter, prompts, ans, s,
                                                args.device).item() * len(q))
                    preds[k] += generate(model, wrapped, tokenizer, [formatter(x) for x in prompts], s, args.device)
            answers += ans
        row = {"N": n, "queries": len(answers)}
        for k in ("real", "wrong", "null"):
            row[f"{k}_loss"] = round(sum(losses[k]) / len(answers), 3)
            row[f"{k}_em"] = round(cloze_em(preds[k], answers), 2)
        print(json.dumps(row), flush=True)
        results.append(row)
    (args.checkpoint.parent / "probe_v2.json").write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
