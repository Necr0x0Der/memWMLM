#!/usr/bin/env python3
"""Variant 2 on small banks (README amendment 2026-11-25, step A).

A trained Variant 2 interface (``train_v2.py``; trained on WikiText only) is
evaluated with the small-bank protocol of ``run_banks.py``: banks-v2 eval
split, banks of N consecutive questions, each question asked closed-book
while reading the delta-rule state S written by the interface from its
bank's N passages. Conditions (as in ``run_banks.py``):

  closed_book            S = None (W_O read off: plain Qwen, or Qwen + LoRA)
  memory                 S of the bank's N passages
  shuffled_memory        S of N calibration passages (same offsets as run_banks)
  counterfactual_memory  S of the bank's answer-swapped passages (+ follow rate)

``--lora``: also install a Variant 1 LoRA checkpoint (``train_lora_banks.py``;
only its q_proj / o_proj LoRA weights; Variant 1 gates / beta are not used).
Passages are always written with the LoRA off (plain Qwen), as in Variant 1.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from run_banks import read_jsonl, render_closed, score_prediction  # noqa: E402
from train_lora_banks import install_lora  # noqa: E402
from train_v2 import encode, load_model  # noqa: E402
from train_v1_global import set_lora  # noqa: E402

from memwmlm.models.delta_interface import DeltaInterface, InterfaceQwen
from memwmlm.models.qwen import QwenChatPromptFormatter


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--banks-dir", type=Path, default=Path("runs/delta-rule/banks-v2"))
    p.add_argument("--ladder", default="1,4,16,64")
    p.add_argument("--conditions", default="closed_book,memory,shuffled_memory,counterfactual_memory")
    p.add_argument("--lora", type=Path, default=None)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--chunk-size", type=int, default=1024)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = True  # as eval_v2.py
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    targs = argparse.Namespace(**ckpt["args"])
    targs.device = args.device
    tokenizer, model = load_model(targs)
    formatter = QwenChatPromptFormatter(tokenizer)
    interface = DeltaInterface(model.config.hidden_size, targs.width,
                               shift_values=getattr(targs, "shift_values", False) or False,
                               bias_state=getattr(targs, "bias_state", False) or False).to(args.device)
    missing, unexpected = interface.load_state_dict(ckpt["interface"], strict=False)
    # pre-V2b checkpoints predate the standardisation buffers; their defaults (off) are how they were trained
    if unexpected or set(missing) - {"in_mean", "in_std", "standardise"}:
        raise RuntimeError(f"checkpoint mismatch: missing {missing}, unexpected {unexpected}")
    interface.eval()
    wrapped = InterfaceQwen(model, interface, targs.layer)
    loras = []
    if args.lora:
        lc = torch.load(args.lora, map_location="cpu")
        loras = install_lora(model, lc["rank"])
        for i, m in enumerate(loras):
            m.down.weight.data.copy_(lc["lora"][i][0])
            m.up.weight.data.copy_(lc["lora"][i][1])
        model.to(args.device)

    @torch.no_grad()
    def state_of(passages: list[str]) -> torch.Tensor:
        set_lora(loras, False)  # written by plain Qwen
        try:
            hidden, lengths = encode(wrapped, tokenizer, passages, 64, args.device, return_lengths=True)
            return interface.write(hidden, chunk_size=args.chunk_size, lengths=lengths)
        finally:
            set_lora(loras, True)

    @torch.no_grad()
    def generate(prompts: list[str], state) -> list[str]:
        tokenizer.padding_side = "left"
        enc = tokenizer(prompts, padding=True, return_tensors="pt").to(args.device)
        with wrapped.reading(state):
            gen = model.generate(**enc, do_sample=False, max_new_tokens=32, pad_token_id=tokenizer.pad_token_id,
                                 top_p=None, top_k=None, temperature=None)
        return [tokenizer.decode(g, skip_special_tokens=True).strip() for g in gen[:, enc["input_ids"].shape[1]:]]

    rows = read_jsonl(args.banks_dir / "eval.jsonl")[: args.limit]
    calib = read_jsonl(args.banks_dir / "calibration.jsonl")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    meta = {"checkpoint": str(args.checkpoint), "train_args": ckpt["args"], "lora": str(args.lora) if args.lora else None}
    (args.output_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")

    for n in (int(x) for x in args.ladder.split(",")):
        banks = [rows[i:i + n] for i in range(0, len(rows), n)]
        for cond in args.conditions.split(","):
            if cond == "closed_book" and n != int(args.ladder.split(",")[0]):
                continue
            out = args.output_dir / f"{cond}-N{n}.jsonl"
            if out.exists():
                continue

            def passages_of(b, bank):
                if cond == "memory":
                    return [r["passage"] for r in bank]
                if cond == "counterfactual_memory":
                    return [r["counterfactual"]["passage"] for r in bank]
                offset = (b * n) % max(len(calib) - n + 1, 1)  # shuffled_memory, as run_banks.py
                return [r["passage"] for r in calib[offset:offset + n]]

            items = [(b, r) for b, bank in enumerate(banks) for r in bank]
            recs = []
            for start in range(0, len(items), args.batch):
                chunk = items[start:start + args.batch]
                prompts = [formatter(render_closed(r["question"])) for _, r in chunk]
                if cond == "closed_book":
                    state = None
                else:
                    cache = {b: state_of(passages_of(b, banks[b])) for b in dict.fromkeys(b for b, _ in chunk)}
                    state = torch.stack([cache[b] for b, _ in chunk])  # one S per row
                for (b, r), pred in zip(chunk, generate(prompts, state), strict=True):
                    exact, f1 = score_prediction(pred, r["aliases"])
                    rec = {"question_id": r["question_id"], "bank": b, "prediction": pred, "exact_match": exact, "f1": f1}
                    if cond == "counterfactual_memory":
                        rec["cf_follow_exact_match"] = score_prediction(pred, r["counterfactual"]["aliases"])[0]
                        rec["cf_valid"] = r["counterfactual"]["valid"]
                    recs.append(rec)
            out.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in recs))
            em = 100 * sum(x["exact_match"] for x in recs) / len(recs)
            print(json.dumps({"cond": cond, "N": n, "em": round(em, 2)}), flush=True)


if __name__ == "__main__":
    main()
