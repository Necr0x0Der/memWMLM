#!/usr/bin/env python3
"""Prepare TriviaQA context containing a gold answer alias when evidence allows it."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from functools import lru_cache
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter

from triviaqa import normalize_answer


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def associated_files(example: dict, evidence_path: str | Path) -> list[dict]:
    """Return the Web and Wikipedia files associated with one question."""
    files: dict[str, dict] = {}
    for field, source in (("EntityPages", "wikipedia"), ("SearchResults", "web")):
        for item in example.get(field) or []:
            filename = item.get("Filename")
            if not filename:
                continue
            path = os.path.join(evidence_path, source, filename)
            files.setdefault(
                path,
                {
                    "path": path,
                    "source": source,
                    "title": item.get("Title") or Path(filename).stem,
                },
            )
    return list(files.values())


def contains_alias(text: str, normalized_aliases: list[str]) -> bool:
    normalized_text = f" {normalize_answer(text)} "
    return any(
        alias and f" {normalize_answer(alias)} " in normalized_text
        for alias in normalized_aliases
    )


def create_oracle_cache(
    dataset_path: str | Path,
    evidence_path: str | Path,
    output_path: str | Path,
    *,
    top_k: int = 5,
    chunk_words: int = 300,
    overlap_words: int = 50,
) -> dict:
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if chunk_words <= 0 or overlap_words < 0 or overlap_words >= chunk_words:
        raise ValueError("chunk_words must be positive and overlap_words must be smaller")

    with open(dataset_path, encoding="utf-8") as file:
        data = json.load(file)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_words,
        chunk_overlap=overlap_words,
        length_function=lambda text: len(text.split()),
    )

    @lru_cache(maxsize=128)
    def split_file(path: str) -> tuple[str, ...]:
        with open(path, encoding="utf-8", errors="replace") as file:
            return tuple(splitter.split_text(file.read()))

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    answer_found_count = 0
    fallback_count = 0

    with output_path.open("w", encoding="utf-8") as output:
        for number, example in enumerate(data["Data"], 1):
            answer = example.get("Answer") or {}
            aliases = answer.get("NormalizedAliases") or []
            if not aliases:
                raise ValueError(f"Question {example.get('QuestionId')} has no NormalizedAliases")

            all_passages = []
            answer_passages = []
            for file_info in associated_files(example, evidence_path):
                if not Path(file_info["path"]).is_file():
                    raise FileNotFoundError(file_info["path"])
                for chunk_index, text in enumerate(split_file(file_info["path"])):
                    passage = {
                        "passage_id": hashlib.sha256(
                            (file_info["path"] + "\0" + text).encode()
                        ).hexdigest(),
                        "source": file_info["path"],
                        "title": file_info["title"],
                        "text": text,
                        "chunk_index": chunk_index,
                    }
                    all_passages.append(passage)
                    if contains_alias(text, aliases):
                        answer_passages.append(passage)

            oracle_answer_found = bool(answer_passages)
            if oracle_answer_found:
                selected = answer_passages[:top_k]
                answer_found_count += 1
            else:
                selected = all_passages[:top_k]
                fallback_count += 1
                if not selected:
                    selected = [
                        {
                            "passage_id": f"no-evidence:{example['QuestionId']}",
                            "source": "no-associated-evidence",
                            "title": "No associated evidence",
                            "text": "No associated TriviaQA evidence was available.",
                            "chunk_index": None,
                        }
                    ]

            record = {
                "question_id": example["QuestionId"],
                "question": example["Question"],
                "mode": "oracle_answer_containing",
                "oracle_answer_found": oracle_answer_found,
                "oracle_matching_passages": len(answer_passages),
                "passages": selected,
            }
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
            output.flush()
            if number % 100 == 0:
                print(
                    f"Prepared {number}/{len(data['Data'])}; "
                    f"answer found for {answer_found_count}",
                    flush=True,
                )

    metadata = {
        "mode": "oracle_answer_containing",
        "top_k": top_k,
        "chunk_words": chunk_words,
        "overlap_words": overlap_words,
        "dataset": str(dataset_path),
        "dataset_sha256": sha256(dataset_path),
        "examples": len(data["Data"]),
        "answer_found_examples": answer_found_count,
        "fallback_examples": fallback_count,
        "gold_used_for_retrieval": True,
        "index_sha256": "not_applicable",
        "retrieval_cache_sha256": sha256(output_path),
    }
    metadata_path = output_path.with_suffix(output_path.suffix + ".meta.json")
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--evidence-path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--chunk-words", type=int, default=300)
    parser.add_argument("--overlap-words", type=int, default=50)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    create_oracle_cache(
        args.dataset,
        args.evidence_path,
        args.output,
        top_k=args.top_k,
        chunk_words=args.chunk_words,
        overlap_words=args.overlap_words,
    )


if __name__ == "__main__":
    main()
