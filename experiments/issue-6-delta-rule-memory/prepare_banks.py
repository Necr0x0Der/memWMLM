#!/usr/bin/env python3
"""Build deterministic TriviaQA fact-passage banks for issue #6.

For each question, a *fact passage* is one sentence from its Wikipedia evidence
documents that contains an answer alias (matched on word tokens, case- and
punctuation-insensitive). Among candidate sentences, the one with the largest
content-word overlap with the question is chosen (ties: earliest). The matched
answer surface span is stored for the counterfactual control.

One fixed evaluation set of ``--eval-size`` dev questions is drawn once and then
partitioned into banks of each ladder size ``N``, so every rung evaluates the
same questions (paired across ``N``). Calibration questions come from train,
and dev questions sharing any normalised alias with a calibration question are
excluded (answer-leakage control).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "issue-5-qwen3-factual-knowledge"))
from triviaqa import normalize_answer  # noqa: E402

SELECTION_RULES = {
    "overlap": "max_question_overlap_answer_sentence_or_window_v2+cf_all_aliases_v2",
    "first": "first_answer_sentence_question_independent_v1+cf_all_aliases_v2",
}
STOPWORDS = frozenset(
    "a an the of in on at to for by with from and or is was were are be been which what who whom "
    "whose where when how why this that these those it its as into his her their he she they".split()
)
SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")


def tokens(text: str) -> list[str]:
    return normalize_answer(text).split()


def alias_pattern(alias: str) -> re.Pattern[str] | None:
    parts = alias.split()
    if not parts:
        return None
    body = r"[\W_]+(?:(?:a|an|the)[\W_]+)?".join(re.escape(p) for p in parts)
    return re.compile(rf"(?<![^\W_]){body}(?![^\W_])", re.IGNORECASE)


def candidate_sentences(text: str, min_words: int, max_words: int):
    """Yield (passage, offset_of_answer_sentence) for single sentences and
    previous+current two-sentence windows (resolves leading pronouns)."""

    for paragraph in text.split("\n"):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        sentences = SENTENCE_SPLIT.split(paragraph)
        for i, sentence in enumerate(sentences):
            n = len(sentence.split())
            if min_words <= n <= max_words:
                yield sentence, 0
            if i and min_words <= n:
                window = sentences[i - 1] + " " + sentence
                if len(window.split()) <= max_words:
                    yield window, len(sentences[i - 1]) + 1


def select_passage(question: str, aliases: list[str], documents: list[str], min_words: int, max_words: int,
                   selection: str = "overlap"):
    patterns = [p for p in (alias_pattern(a) for a in sorted(aliases, key=len, reverse=True)) if p]
    question_words = {t for t in tokens(question) if t not in STOPWORDS}
    best = None
    order = 0
    for document in documents:
        for sentence, answer_offset in candidate_sentences(document, min_words, max_words):
            for pattern in patterns:
                # The answer must lie in the final (answer) sentence of a window.
                match = pattern.search(sentence, answer_offset)
                if match:
                    overlap = len(question_words & set(tokens(sentence)))
                    key = (overlap, -len(sentence.split()), -order) if selection == "overlap" else (0, 0, -order)
                    if best is None or key > best[0]:
                        best = (key, sentence, match.span())
                    break
            order += 1
    if best is None:
        return None
    _, sentence, span = best
    return {"passage": sentence, "answer_span": list(span), "question_overlap": best[0][0]}


def load_split(qa_path: Path, evidence_dir: Path, min_words: int, max_words: int, limit: int | None, rng_seed: int,
               selection: str = "overlap"):
    payload = json.loads(qa_path.read_text(encoding="utf-8"))
    records = payload["Data"]
    order = list(range(len(records)))
    random.Random(rng_seed).shuffle(order)
    selected, missing = [], 0
    for idx in order:
        if limit is not None and len(selected) >= limit:
            break
        record = records[idx]
        aliases = record["Answer"]["NormalizedAliases"]
        documents = []
        for page in record.get("EntityPages", []):
            path = evidence_dir / page["Filename"]
            if path.exists():
                documents.append(path.read_text(encoding="utf-8"))
        chosen = select_passage(record["Question"], aliases, documents, min_words, max_words, selection)
        if chosen is None:
            missing += 1
            continue
        selected.append({
            "question_id": record["QuestionId"],
            "question": record["Question"],
            "aliases": aliases,
            **chosen,
        })
    return selected, missing


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rc-dir", type=Path, default=Path("artifacts/triviaqa/triviaqa-rc"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--eval-size", type=int, default=1024)
    parser.add_argument("--calibration-size", type=int, default=1024)
    parser.add_argument("--ladder", default="1,4,16,64,256,1024")
    parser.add_argument("--min-words", type=int, default=6)
    parser.add_argument("--max-words", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20261111)
    parser.add_argument("--selection", choices=tuple(SELECTION_RULES), default="overlap",
                        help="overlap = most question-word overlap; first = first answer sentence (question-blind)")
    args = parser.parse_args()

    evidence = args.rc_dir / "evidence" / "wikipedia"
    train_qa = args.rc_dir / "qa" / "wikipedia-train.json"
    dev_qa = args.rc_dir / "qa" / "wikipedia-dev.json"

    calibration, cal_missing = load_split(
        train_qa, evidence, args.min_words, args.max_words, args.calibration_size, args.seed, args.selection
    )
    calibration_aliases = {a for item in calibration for a in item["aliases"]}
    # Over-draw dev, then drop answer-leaking questions before truncating.
    dev, dev_missing = load_split(
        dev_qa, evidence, args.min_words, args.max_words, 3 * args.eval_size, args.seed + 1, args.selection
    )
    leaked = [d for d in dev if calibration_aliases & set(d["aliases"])]
    dev = [d for d in dev if not calibration_aliases & set(d["aliases"])][: args.eval_size]
    if len(dev) < args.eval_size:
        raise SystemExit(f"Only {len(dev)} eligible dev questions")

    ladder = [int(x) for x in args.ladder.split(",")]
    banks = {}
    for n in ladder:
        if args.eval_size % n:
            raise SystemExit(f"eval-size {args.eval_size} not divisible by N={n}")
        banks[str(n)] = [[d["question_id"] for d in dev[i : i + n]] for i in range(0, args.eval_size, n)]

    # Counterfactual: a fixed derangement of the eval set; question i's passage gets
    # the answer surface of question perm[i]; scored against that question's aliases.
    rng = random.Random(args.seed + 2)
    perm = list(range(len(dev)))
    while any(i == p for i, p in enumerate(perm)):
        rng.shuffle(perm)
    for i, item in enumerate(dev):
        donor = dev[perm[i]]
        s, e = item["answer_span"]
        ds, de = donor["answer_span"]
        surface = donor["passage"][ds:de]
        passage = item["passage"][:s] + surface + item["passage"][e:]
        # Replace every remaining occurrence of every original alias too.
        own = [p for p in (alias_pattern(a) for a in sorted(item["aliases"], key=len, reverse=True)) if p]
        for pattern in own:
            passage = pattern.sub(lambda _m: surface, passage)
        donor_patterns = [p for p in (alias_pattern(a) for a in donor["aliases"]) if p]
        valid = (
            not any(p.search(passage) for p in own)  # original answer fully removed
            and not any(p.search(item["passage"]) for p in donor_patterns)  # donor not already present
        )
        item["counterfactual"] = {
            "donor_question_id": donor["question_id"],
            "passage": passage,
            "aliases": donor["aliases"],
            "valid": valid,
        }

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "eval.jsonl").write_text(
        "".join(json.dumps(d, ensure_ascii=False, sort_keys=True) + "\n" for d in dev), encoding="utf-8"
    )
    (args.output / "calibration.jsonl").write_text(
        "".join(json.dumps(d, ensure_ascii=False, sort_keys=True) + "\n" for d in calibration), encoding="utf-8"
    )
    (args.output / "banks.json").write_text(json.dumps(banks, indent=1) + "\n", encoding="utf-8")
    manifest = {
        "selection_rule": SELECTION_RULES[args.selection],
        "seed": args.seed,
        "min_words": args.min_words,
        "max_words": args.max_words,
        "ladder": ladder,
        "eval_size": len(dev),
        "calibration_size": len(calibration),
        "dev_without_passage_among_drawn": dev_missing,
        "train_without_passage_among_drawn": cal_missing,
        "dev_dropped_for_alias_leakage_among_overdraw": len(leaked),
        "counterfactual_valid": sum(d["counterfactual"]["valid"] for d in dev),
        "sources": {str(p): sha256(p) for p in (train_qa, dev_qa)},
        "outputs": {name: sha256(args.output / name) for name in ("eval.jsonl", "calibration.jsonl", "banks.json")},
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
