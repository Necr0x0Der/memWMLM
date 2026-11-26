#!/usr/bin/env python3
"""Sanity check 5: does memory help when it is read with the *same passage*?

Memory is filled exactly as in ``run_banks.py`` (banks of N passages, frozen
Qwen3, sink prefix). At test time Qwen does not get the question: it gets the
question's own passage cut right before the answer (``<|endoftext|>`` + the
passage text up to ``answer_span[0]``, the same tokens and sink as when it was
written) and continues it greedily. The answer is not in the prompt; it exists
only in memory. Hit = the normalised continuation starts with the gold answer
(any alias or the passage surface). Items whose prefix has < 3 words are
skipped (the answer starts the passage); banks still contain every passage.

Conditions: none (prefix only, no memory), memory (bank incl. the passage),
shuffled (N calibration passages), counterfactual (the bank's answer-swapped
passages; follow = continuation starts with the swapped answer), oracle (the
full passage in context before the prefix: copy upper bound).

Linear configs choose the gate per N on the calibration split (memory −
shuffled, ties -> smaller, gates where both are < 1% skipped). Config K is the
uncompressed softmax memory (upper-bound test, top-k 4), no gate.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "issue-5-qwen3-factual-knowledge"))
from run_banks import answer_surface, build_state, read_jsonl  # noqa: E402
from triviaqa import normalize_answer  # noqa: E402

from memwmlm.models.delta_qwen import DeltaMemoryConfig, DeltaMemoryQwen, load_feature_maps

CONFIGS = {
    "A": {"center_keys": True},
    "B": {"center_keys": True, "center_queries": True, "key_expand": 16},
    "WB": {"center_keys": True, "center_queries": True, "key_expand": 16, "whiten": True},
    "K": {"read": "knn", "knn_top_k": 4},
    "T": {},  # trained feature maps (--feature-map), gate fixed at 1
}
GATES = (0.05, 0.1, 0.2, 0.5, 1.0)


def starts_with(prediction: str, answers: list[str]) -> bool:
    p = normalize_answer(prediction)
    return any(a and (p == a or p.startswith(a + " ")) for a in answers)


def items_of(rows):
    out = []
    for i, r in enumerate(rows):
        prefix = r["passage"][: r["answer_span"][0]].rstrip()
        if len(prefix.split()) >= 3:
            out.append((i, prefix))
    return out


@torch.no_grad()
def generate(model, tokenizer, texts, sink, device, max_new=12):
    seqs = [[sink] + tokenizer(t, add_special_tokens=False)["input_ids"] for t in texts]
    width = max(map(len, seqs))
    ids = torch.tensor([[tokenizer.pad_token_id] * (width - len(s)) + s for s in seqs], device=device)
    mask = torch.tensor([[0] * (width - len(s)) + [1] * len(s) for s in seqs], device=device)
    out = model.generate(input_ids=ids, attention_mask=mask, max_new_tokens=max_new, do_sample=False,
                         pad_token_id=tokenizer.pad_token_id, top_p=None, top_k=None, temperature=None)
    return [tokenizer.decode(g, skip_special_tokens=True) for g in out[:, width:]]


def run(cond, n, rows, shuffle_rows, model, tokenizer, memory, gates, args, sink):
    """Return {gate: [record per item]} for one condition and bank size."""

    items = items_of(rows)
    results = {g: [] for g in gates}
    cache: dict[int, dict] = {}

    def bank_passages(b):
        bank = rows[b * n:(b + 1) * n]
        if cond == "memory":
            return [r["passage"] for r in bank]
        if cond == "counterfactual":
            return [r["counterfactual"]["passage"] for r in bank]
        offset = (b * n) % max(len(shuffle_rows) - n + 1, 1)
        return [r["passage"] for r in shuffle_rows[offset:offset + n]]

    def state_of(b):
        if b not in cache:
            cache[b] = build_state(memory, tokenizer, bank_passages(b), args)
        return cache[b]

    knn = memory is not None and memory.config.read == "knn"
    batch = 1 if knn else (args.batch // 2 if memory is not None and memory.config.key_expand else args.batch)
    for start in range(0, len(items), batch):
        chunk = items[start:start + batch]
        texts = [p if cond != "oracle" else rows[i]["passage"] + "\n\n" + p for i, p in chunk]
        if cond in ("none", "oracle"):
            preds = {g: generate(model, tokenizer, texts, sink, args.device) for g in gates}
        else:
            needed = {i // n for i, _ in chunk}
            for b in [b for b in cache if b not in needed]:
                del cache[b]  # keep only the banks of this batch (expanded states are ~0.5 GB each)
            states = [state_of(i // n) for i, _ in chunk]
            state = states[0] if knn else {l: torch.stack([s[l] for s in states]) for l in states[0]}
            preds = {}
            for g in gates:
                memory.config.gate = g
                with memory.reading(state):
                    preds[g] = generate(model, tokenizer, texts, sink, args.device)
        for g in gates:
            for (i, prefix), pred in zip(chunk, preds[g], strict=True):
                r = rows[i]
                rec = {"question_id": r["question_id"], "prediction": pred,
                       "hit": float(starts_with(pred, r["aliases"] + [normalize_answer(answer_surface(r))]))}
                if cond == "counterfactual":
                    cf = answer_surface(r, counterfactual=True)
                    rec["follow"] = float(starts_with(pred, r["counterfactual"]["aliases"] + [normalize_answer(cf)]))
                results[g].append(rec)
    return results


def save(path: Path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records))


def rate(records, key="hit"):
    return round(100 * sum(r[key] for r in records) / max(len(records), 1), 2)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--banks-dir", type=Path, default=Path("runs/delta-rule/banks-v2"))
    p.add_argument("--model", default="Qwen/Qwen3-1.7B")
    p.add_argument("--revision", default="70d244cc86ccca08cf5af4e1e306ecf908b1ad5e")
    p.add_argument("--configs", default="A,B,WB,K")
    p.add_argument("--ladder", default="1,4,16,64")
    p.add_argument("--calib-limit", type=int, default=512)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--feature-map", type=Path, default=None, help="checkpoint for config T")
    p.add_argument("--device", default="cuda")
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    args.write_batch_size, args.sink_prefix = 64, True
    torch.backends.cuda.matmul.allow_tf32 = False
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, revision=args.revision, dtype=torch.bfloat16,
                                                 local_files_only=True).to(args.device).eval()
    sink = tokenizer.convert_tokens_to_ids("<|endoftext|>")
    eval_rows = read_jsonl(args.banks_dir / "eval.jsonl")
    calib = read_jsonl(args.banks_dir / "calibration.jsonl")
    calib_eval, calib_shuffle = calib[: args.calib_limit], calib[args.calib_limit:]
    ladder = [int(x) for x in args.ladder.split(",")]
    out = args.output_dir
    print(json.dumps({"eval_items": len(items_of(eval_rows)), "calib_items": len(items_of(calib_eval))}), flush=True)

    for cond in ("none", "oracle"):
        if (out / f"eval/{cond}.jsonl").exists():
            continue  # memory-free references, identical across runs
        recs = run(cond, 1, eval_rows, calib, model, tokenizer, None, [None], args, sink)[None]
        save(out / f"eval/{cond}.jsonl", recs)
        print(json.dumps({"config": "-", "cond": cond, "hit": rate(recs)}), flush=True)

    for name in args.configs.split(","):
        memory = DeltaMemoryQwen(model, DeltaMemoryConfig(beta=1.0, alpha=1.0, **CONFIGS[name]))
        if CONFIGS[name].get("center_keys"):
            enc = SimpleNamespace(write_batch_size=64, device=args.device, sink_prefix=True)
            from run_banks import encode_passages
            k, _, m = encode_passages(memory, tokenizer, [r["passage"] for r in calib[:256]], enc)
            memory.calibrate_key_mean(k, m)
            del k
        knn = CONFIGS[name].get("read") == "knn"
        if name == "T":
            load_feature_maps(memory, args.feature_map)
        for n in ladder:
            gate = 1.0 if name == "T" else None
            if not knn and name != "T":
                scores = []
                cal = {c: run(c, n, calib_eval, calib_shuffle, model, tokenizer, memory, GATES, args, sink)
                       for c in ("memory", "shuffled")}
                for g in GATES:
                    m_, s_ = rate(cal["memory"][g]), rate(cal["shuffled"][g])
                    collapsed = m_ < 1 and s_ < 1
                    print(json.dumps({"config": name, "N": n, "calib_gate": g, "memory": m_, "shuffled": s_,
                                      "collapsed": collapsed}), flush=True)
                    if not collapsed:
                        scores.append((m_ - s_, -g, g))
                gate = max(scores)[2]
            gates = [gate]
            for cond in ("memory", "shuffled", "counterfactual"):
                recs = run(cond, n, eval_rows, calib, model, tokenizer, memory, gates, args, sink)[gate]
                save(out / f"eval/{name}-{cond}-N{n}.jsonl", recs)
                row = {"config": name, "N": n, "gate": gate, "cond": cond, "hit": rate(recs)}
                if cond == "counterfactual":
                    row["follow"] = rate(recs, "follow")
                print(json.dumps(row), flush=True)
            torch.cuda.empty_cache()
        memory.close()
    print("CLOZE DONE", flush=True)


if __name__ == "__main__":
    main()
