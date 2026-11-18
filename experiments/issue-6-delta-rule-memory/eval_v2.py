#!/usr/bin/env python3
"""Evaluate a trained Variant 2 interface write-once on the global TriviaQA corpus.

The interface never saw TriviaQA. The global corpus of ``prepare_global.py``
(all train-split passages + the dev_in half) is written once, in its fixed
order, with the trained writer; every question group is answered closed-book
while reading S. Conditions:

  memory          S of the whole corpus
  counterfactual  S with the dev_in passages answer-swapped (dev_in only)
  size curve      S of corpus subsets that always contain all dev_in passages
                  (dev_in only), to locate the capacity limit
  in-domain       WikiText validation cloze with S of ``--indomain-sizes``
                  validation sentences (positive control for the interface)

``S = 0`` reproduces closed-book exactly (``W_O`` reads zero), so the
closed-book / BM25 references of ``train_v1_global.py --references`` apply.
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
from run_banks import git_dirty, git_revision, read_jsonl, render_closed, score_prediction  # noqa: E402
from train_v2 import ClozeFormatter, cloze_em, cloze_question, load_shots, encode, generate, load_model  # noqa: E402

from memwmlm.models.delta_interface import DeltaInterface, InterfaceQwen
from memwmlm.models.qwen import QwenChatPromptFormatter

GROUPS = ("train_seen", "train_heldout", "dev_in", "dev_out")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--data", type=Path, default=Path("runs/delta-rule/global-v1"))
    p.add_argument("--cloze", type=Path, default=Path("runs/delta-rule/wikitext-cloze-v1"))
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--sizes", default="512,1024,2048")
    p.add_argument("--indomain-sizes", default="1024,4342")
    p.add_argument("--indomain-queries", type=int, default=512)
    p.add_argument("--chunk-size", type=int, default=1024)
    p.add_argument("--encode-batch", type=int, default=128)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = True
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    train_args = argparse.Namespace(**ckpt["args"])
    train_args.device = args.device
    out = args.output_dir or args.checkpoint.parent / "eval"
    out.mkdir(parents=True, exist_ok=True)

    tokenizer, model = load_model(train_args)
    formatter = QwenChatPromptFormatter(tokenizer)
    interface = DeltaInterface(model.config.hidden_size, train_args.width,
                               shift_values=getattr(train_args, "shift_values", False),
                               bias_state=getattr(train_args, "bias_state", False)).to(args.device)
    interface.load_state_dict(ckpt["interface"])
    interface.eval()
    wrapped = InterfaceQwen(model, interface, train_args.layer)

    @torch.no_grad()
    def state_of(passages: list[str]) -> torch.Tensor:
        hidden, lengths = encode(wrapped, tokenizer, passages, args.encode_batch, args.device, return_lengths=True)
        return interface.write(hidden, chunk_size=args.chunk_size, lengths=lengths)

    questions = read_jsonl(args.data / "questions.jsonl")
    corpus = read_jsonl(args.data / "corpus.jsonl")
    summaries = []

    def record(name, rows, preds, counterfactual=False):
        recs = []
        for r, pred in zip(rows, preds, strict=True):
            exact, f1 = score_prediction(pred, r["aliases"])
            rec = {"question_id": r["question_id"], "group": r["group"], "prediction": pred,
                   "exact_match": exact, "f1": f1}
            if counterfactual:
                rec["cf_follow_exact_match"] = score_prediction(pred, r["counterfactual"]["aliases"])[0]
                rec["cf_valid"] = r["counterfactual"]["valid"]
            recs.append(rec)
        (out / f"{name}.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in recs))
        s = {"run": name, "n": len(recs), "em": round(100 * sum(x["exact_match"] for x in recs) / len(recs), 2)}
        if counterfactual:
            valid = [x for x in recs if x["cf_valid"]]
            s["cf_follow"] = round(100 * sum(x["cf_follow_exact_match"] for x in valid) / max(len(valid), 1), 2)
        print(json.dumps(s), flush=True)
        summaries.append(s)

    rows = {g: [q for q in questions if q["group"] == g] for g in GROUPS}
    prompts = {g: [formatter(render_closed(r["question"])) for r in rows[g]] for g in GROUPS}
    full = state_of([c["passage"] for c in corpus])
    for g in GROUPS:
        record(f"memory-{g}", rows[g], generate(model, wrapped, tokenizer, prompts[g], full, args.device))
    del full
    cf = state_of([c["counterfactual_passage"] or c["passage"] for c in corpus])
    record("counterfactual-dev_in", rows["dev_in"],
           generate(model, wrapped, tokenizer, prompts["dev_in"], cf, args.device), counterfactual=True)
    del cf
    dev_in = [c for c in corpus if c["group"] == "dev_in"]
    others = [c for c in corpus if c["group"] != "dev_in"]
    for size in (int(x) for x in args.sizes.split(",")):
        keep = {c["question_id"] for c in dev_in + others[: size - len(dev_in)]}
        subset = [c["passage"] for c in corpus if c["question_id"] in keep]
        record(f"memory-dev_in-corpus{size}", rows["dev_in"],
               generate(model, wrapped, tokenizer, prompts["dev_in"], state_of(subset), args.device))

    val = read_jsonl(args.cloze / "val.jsonl")
    cloze_formatter = ClozeFormatter(tokenizer, load_shots(args.cloze, int(getattr(train_args, "fewshot", 0))))
    if getattr(train_args, "full_cloze", False):
        for r in val:
            r["cloze"] = r["passage"].replace(r["answer"], "___", 1)
    for size in (int(x) for x in args.indomain_sizes.split(",")):
        bank = val[:size]
        asked = random.Random(2).sample(bank, min(args.indomain_queries, size))
        cp = [cloze_formatter(cloze_question(r["cloze"])) for r in asked]
        preds = generate(model, wrapped, tokenizer, cp, state_of([r["passage"] for r in bank]), args.device)
        s = {"run": f"indomain_cloze-N{size}", "n": len(asked), "em": round(cloze_em(preds, [r["answer"] for r in asked]), 2)}
        print(json.dumps(s), flush=True)
        summaries.append(s)
        if size == max(int(x) for x in args.indomain_sizes.split(",")):
            null = generate(model, wrapped, tokenizer, cp, None, args.device)
            s = {"run": "indomain_cloze-closed_book", "n": len(asked),
                 "em": round(cloze_em(null, [r["answer"] for r in asked]), 2)}
            print(json.dumps(s), flush=True)
            summaries.append(s)

    meta = {"checkpoint": str(args.checkpoint), "train_args": ckpt["args"], "step": ckpt["step"],
            "corpus_passages": len(corpus), "repository_commit": git_revision(),
            "working_tree_dirty": git_dirty(), "summaries": summaries}
    (out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")


if __name__ == "__main__":
    main()
