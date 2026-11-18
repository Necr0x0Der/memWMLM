#!/usr/bin/env python3
"""Build the global-memory corpus and question groups (README amendment 2026-11-13).

Corpus written into one state S: passages of all train questions (the 1,024
calibration questions of ``banks-v2`` plus ``--extra`` further train questions
drawn with the same seed and selection rule, minus any whose aliases intersect
an eval alias) and of the even-indexed half of ``eval.jsonl``. The odd-indexed
eval half is never written.

Outputs in ``--output``: ``questions.jsonl`` (every question with ``group`` in
train_seen / train_other / train_heldout / dev_in / dev_out and its answer
surface form), ``corpus.jsonl`` (write order), ``manifest.json``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from prepare_banks import load_split  # noqa: E402
from run_banks import read_jsonl  # noqa: E402


def dump(path: Path, rows: list[dict]) -> str:
    path.write_text("".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in rows),
                    encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--banks-dir", type=Path, default=Path("runs/delta-rule/banks-v2"))
    p.add_argument("--rc-dir", type=Path, default=Path("artifacts/triviaqa/triviaqa-rc"))
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--extra", type=int, default=3072)
    p.add_argument("--bank-seed", type=int, default=20261111)
    p.add_argument("--seed", type=int, default=20261113)
    p.add_argument("--group-size", type=int, default=512)
    args = p.parse_args()

    manifest_in = json.loads((args.banks_dir / "manifest.json").read_text())
    calibration = read_jsonl(args.banks_dir / "calibration.jsonl")
    evaluation = read_jsonl(args.banks_dir / "eval.jsonl")
    selection = "first" if manifest_in["selection_rule"].startswith("first") else "overlap"

    train_qa = args.rc_dir / "qa" / "wikipedia-train.json"
    evidence = args.rc_dir / "evidence" / "wikipedia"
    drawn, missing = load_split(train_qa, evidence, manifest_in["min_words"], manifest_in["max_words"],
                                len(calibration) + args.extra, args.bank_seed, selection)
    if [d["question_id"] for d in drawn[: len(calibration)]] != [c["question_id"] for c in calibration]:
        raise SystemExit("re-drawn train questions do not reproduce calibration.jsonl")
    eval_aliases = {a for e in evaluation for a in e["aliases"]}
    extra = [d for d in drawn[len(calibration):] if not eval_aliases & set(d["aliases"])]
    leaked = len(drawn) - len(calibration) - len(extra)

    train = calibration + extra
    rng = random.Random(args.seed)
    order = list(range(len(train)))
    rng.shuffle(order)
    g = args.group_size
    groups = {}
    for rank, i in enumerate(order):
        groups[i] = "train_heldout" if rank < g else "train_seen" if rank < 2 * g else "train_other"

    questions = []
    for i, row in enumerate(train):
        s, e = row["answer_span"]
        questions.append({**row, "group": groups[i], "answer_surface": row["passage"][s:e], "split": "train"})
    for i, row in enumerate(evaluation):
        s, e = row["answer_span"]
        questions.append({**row, "group": "dev_in" if i % 2 == 0 else "dev_out",
                          "answer_surface": row["passage"][s:e], "split": "dev"})

    written = [q for q in questions if q["group"] != "dev_out"]
    corpus = [{"question_id": q["question_id"], "group": q["group"], "passage": q["passage"],
               "counterfactual_passage": q["counterfactual"]["passage"] if q["group"] == "dev_in" else None}
              for q in written]
    rng.shuffle(corpus)

    args.output.mkdir(parents=True, exist_ok=True)
    counts = {name: sum(q["group"] == name for q in questions)
              for name in ("train_seen", "train_other", "train_heldout", "dev_in", "dev_out")}
    manifest = {
        "rule": "global_v1: calibration + extra train (alias-disjoint from eval) + even eval half; shuffled write order",
        "banks_manifest_sha256": hashlib.sha256((args.banks_dir / "manifest.json").read_bytes()).hexdigest(),
        "selection": selection, "bank_seed": args.bank_seed, "seed": args.seed,
        "extra_requested": args.extra, "extra_dropped_for_eval_alias_leakage": leaked,
        "train_without_passage_among_drawn": missing,
        "groups": counts, "corpus_passages": len(corpus),
        "outputs": {"questions.jsonl": dump(args.output / "questions.jsonl", questions),
                    "corpus.jsonl": dump(args.output / "corpus.jsonl", corpus)},
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
