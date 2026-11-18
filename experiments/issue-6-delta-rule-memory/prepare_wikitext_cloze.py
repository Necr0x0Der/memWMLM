#!/usr/bin/env python3
"""General-text interface data for Variant 2: WikiText-103 sentence cloze.

Each unit is one WikiText-103 sentence of 6-64 words (the eval passage length
range) with one *answer-like* span chosen at random: a run of 1-4 capitalised
words not at the sentence start (inner "of/the/de/and/von" allowed), or a
number of 3-4 digits. The unit's full sentence is what gets written into
memory; the query is a cropped cloze: a contiguous window of 6-16 words around
the blank (span replaced by ``___``), so the query never contains the whole
stored sentence.

Answer-leakage control against TriviaQA (``--questions``: all questions of the
global corpus, which include every eval question): a unit is dropped if its
span normalises to any TriviaQA alias, or if the sentence contains any
multi-word TriviaQA alias.

Train units come from the WikiText-103 train split (article order shuffled
with ``--seed``); validation units from the validation split.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from pathlib import Path

import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from run_banks import normalize_answer, read_jsonl  # noqa: E402

INNER = {"of", "the", "de", "and", "von", "van", "la", "le", "du", "da"}
CAP = re.compile(r"^[A-Z][A-Za-z'.\-]*$")
NUM = re.compile(r"^\d{3,4}$")
PUNCT = ".,;:!?\")'"


def detok(text: str) -> str:
    text = text.replace(" @-@ ", "-").replace(" @,@ ", ",").replace(" @.@ ", ".")
    text = re.sub(r" ([.,;:!?%)\]])", r"\1", text)
    text = re.sub(r"([(\[$]) ", r"\1", text)
    return text.replace(" 's", "'s").replace(" n't", "n't").strip()


def articles(path_list: list[Path]):
    """Yield (title, [paragraph, ...]) from raw WikiText parquet files."""

    title, paragraphs = None, []
    for path in path_list:
        for line in pq.read_table(path).column("text").to_pylist():
            stripped = line.strip()
            if re.fullmatch(r"= [^=].* =", stripped):
                if title is not None:
                    yield title, paragraphs
                title, paragraphs = stripped.strip("= ").strip(), []
            elif stripped and not stripped.startswith("="):
                paragraphs.append(detok(stripped))
    if title is not None:
        yield title, paragraphs


ABBREV = re.compile(r"(?:\b(?:St|Mr|Mrs|Ms|Dr|Jr|Sr|No|Co|Mt|Ft|Lt|Gen|Col|Capt|Sgt|Rev|Fr|vs|Vol|pp|ca)|\b[A-Z])\.$")


def sentences(paragraph: str) -> list[str]:
    """Split on sentence punctuation, re-joining splits after abbreviations/initials."""

    out: list[str] = []
    for piece in re.split(r"(?<=[.!?])\s+(?=[A-Z\"'(])", paragraph):
        if out and ABBREV.search(out[-1]):
            out[-1] = out[-1] + " " + piece
        else:
            out.append(piece)
    return out


def spans(words: list[str]) -> list[tuple[int, int]]:
    core = [w.rstrip(PUNCT).lstrip("\"('") for w in words]
    found, i = [], 1
    while i < len(words):
        if CAP.match(core[i]):
            j = i + 1
            while j < len(words) and j - i < 4 and core[j - 1] == words[j - 1] and (
                CAP.match(core[j]) or (core[j] in INNER and j + 1 < len(words) and CAP.match(core[j + 1]))
            ):
                j += 1
            if core[j - 1] in INNER:
                j -= 1
            found.append((i, j))
            i = j
        elif NUM.match(core[i]):
            found.append((i, i + 1))
            i += 1
        else:
            i += 1
    return found


def make_unit(sentence: str, rng: random.Random, aliases: set[str], multi: set[str]):
    words = sentence.split()
    if not 6 <= len(words) <= 64:
        return None, "length"
    candidates = spans(words)
    if not candidates:
        return None, "no_span"
    i, j = rng.choice(candidates)
    lead = words[i][: len(words[i]) - len(words[i].lstrip("\"('"))]
    trail = words[j - 1][len(words[j - 1].rstrip(PUNCT)):]
    answer = " ".join(words[i:j])[len(lead):]
    answer = answer[: len(answer) - len(trail)] if trail else answer
    norm = normalize_answer(answer)
    if not norm:
        return None, "no_span"
    if norm in aliases:
        return None, "alias_answer"
    toks = normalize_answer(sentence).split()
    grams = {" ".join(toks[a:b]) for a in range(len(toks)) for b in range(a + 2, min(a + 7, len(toks) + 1))}
    if grams & multi:
        return None, "alias_in_sentence"
    blanked = words[:i] + [lead + "___" + trail] + words[j:]
    width = rng.randint(6, 16)
    lo = max(0, min(i - rng.randint(0, width - 1), len(blanked) - width))
    cloze = " ".join(blanked[lo : lo + width])
    return {"passage": sentence, "cloze": cloze, "answer": answer}, "ok"


def build(paths, limit, seed, aliases, multi, prefix):
    rng = random.Random(seed)
    docs = list(articles(paths))
    rng.shuffle(docs)
    units, stats = [], {}
    for title, paragraphs in docs:
        for paragraph in paragraphs:
            for sentence in sentences(paragraph):
                unit, why = make_unit(sentence.strip(), rng, aliases, multi)
                stats[why] = stats.get(why, 0) + 1
                if unit:
                    units.append({"id": f"{prefix}{len(units)}", "title": title, **unit})
                    if len(units) >= limit:
                        return units, stats
    return units, stats


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--wikitext-dir", type=Path, default=Path("artifacts/wikitext/wikitext-103-raw-v1"))
    p.add_argument("--questions", type=Path, default=Path("runs/delta-rule/global-v1/questions.jsonl"))
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--train-size", type=int, default=120_000)
    p.add_argument("--val-size", type=int, default=6_000)
    p.add_argument("--seed", type=int, default=20261113)
    args = p.parse_args()

    aliases = {a for q in read_jsonl(args.questions) for a in q["aliases"]}
    multi = {a for a in aliases if len(a.split()) >= 2}
    train_files = sorted(args.wikitext_dir.glob("train-*.parquet"))
    val_files = sorted(args.wikitext_dir.glob("validation-*.parquet"))
    train, train_stats = build(train_files, args.train_size, args.seed, aliases, multi, "t")
    val, val_stats = build(val_files, args.val_size, args.seed + 1, aliases, multi, "v")
    args.output.mkdir(parents=True, exist_ok=True)
    outputs = {}
    for name, rows in (("train.jsonl", train), ("val.jsonl", val)):
        path = args.output / name
        path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
        outputs[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = {
        "rule": "wikitext103_sentence_cloze_v1 (span: 1-4 capitalised words not sentence-initial, or 3-4 digit "
                "number; query: 6-16 word window around the blank)",
        "seed": args.seed, "train_units": len(train), "val_units": len(val),
        "train_filter_counts": train_stats, "val_filter_counts": val_stats,
        "triviaqa_aliases": len(aliases), "triviaqa_multiword_aliases": len(multi),
        "sources": {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in train_files + val_files},
        "outputs": outputs,
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
