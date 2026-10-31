#!/usr/bin/env python3
"""
make_repro_dataset.py — create the dataset for Phase 0 (Stage A: exact reproduction).

Conservative: does NOT modify memWMLM, only prepares the dataset.

Source: Salesforce/wikitext, config wikitext-103-raw-v1.
  - Primary path: load_dataset (requires `pip install datasets`).
  - Fallback: manual parquet download from the HF hub (if datasets is not installed).

Logic:
  1. Load train + validation + test (raw-v1).
  2. train -> train windows; validation+test -> heldout windows (never seen during training).
  3. Filter empty/short lines, split into sentences, group
     into windows (min..max sentences).
  4. Take --fraction of train windows (scale parameter, 100k+).
  5. Save train.jsonl and heldout.jsonl (id, split, text).

Examples:
  # Full dataset (~120k train windows):
  python scripts/make_repro_dataset.py --fraction 1.0 --max-windows 120000 --output-dir datasets/repro

  # Quick test (5% train):
  python scripts/make_repro_dataset.py --fraction 0.05 --output-dir datasets/repro

Dependencies: datasets (recommended), pyarrow, pandas, requests.
"""

import argparse
import json
import os
import re
import requests
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

warnings.filterwarnings("ignore")  # insecure-request warnings (public dataset)

REPO = "Salesforce/wikitext"
CFG = "wikitext-103-raw-v1"
HF_BASE = f"https://huggingface.co/datasets/{REPO}/resolve/main"
# files for the fallback download (if datasets is not installed)
WT103_FILES = {
    "train": [
        "wikitext-103-raw-v1/train-00000-of-00002.parquet",
        "wikitext-103-raw-v1/train-00001-of-00002.parquet",
    ],
    "heldout": [
        "wikitext-103-raw-v1/validation-00000-of-00001.parquet",
        "wikitext-103-raw-v1/test-00000-of-00001.parquet",
    ],
}


# ---------------------------------------------------------------------------
# Source loading
# ---------------------------------------------------------------------------

def _clean(texts: List[str], min_len: int = 15) -> List[str]:
    """Removes empty/short lines (blank lines in the raw format)."""
    return [t.strip() for t in texts if len(t.strip()) >= min_len]


def load_via_hf() -> Dict[str, List[str]]:
    """Primary path: load_dataset (Salesforce/wikitext, raw-v1)."""
    from datasets import load_dataset
    print("      load_dataset('Salesforce/wikitext', 'wikitext-103-raw-v1') ...")
    train = load_dataset(REPO, CFG, split="train")
    valid = load_dataset(REPO, CFG, split="validation")
    test = load_dataset(REPO, CFG, split="test")
    return {
        "train": _clean(train["text"]),
        "heldout": _clean(list(valid["text"]) + list(test["text"])),
    }


def _download_file(rel: str, cache_dir: str) -> str:
    os.makedirs(cache_dir, exist_ok=True)
    dest = os.path.join(cache_dir, os.path.basename(rel))
    if os.path.exists(dest):
        return dest
    url = f"{HF_BASE}/{rel}"
    print(f"      downloading {url} ...")
    r = requests.get(url, timeout=600, stream=True)
    r.raise_for_status()
    with open(dest, "wb") as f:
        for chunk in r.iter_content(65536):
            if chunk:
                f.write(chunk)
    return dest


def load_via_manual(cache_dir: str) -> Dict[str, List[str]]:
    """Fallback: manual parquet download + pyarrow."""
    import pyarrow.parquet as pq
    def read_split(rels: List[str]) -> List[str]:
        texts: List[str] = []
        for rel in rels:
            path = _download_file(rel, cache_dir)
            table = pq.read_table(path)
            col = table.column_names[0]  # 'text'
            texts.extend(table.select([col]).to_pandas()[col].tolist())
        return texts
    return {
        "train": _clean(read_split(WT103_FILES["train"])),
        "heldout": _clean(read_split(WT103_FILES["heldout"])),
    }


def load_texts(cache_dir: str) -> Dict[str, List[str]]:
    """Tries load_dataset, otherwise manual parquet."""
    try:
        data = load_via_hf()
        # check that we actually read non-empty text
        if len(data["train"]) == 0:
            raise RuntimeError("load_dataset returned empty train")
        return data
    except ImportError:
        print("      [fallback] datasets not installed -> manual parquet")
        return load_via_manual(cache_dir)


# ---------------------------------------------------------------------------
# Processing
# ---------------------------------------------------------------------------

def split_sentences(text: str, use_nltk: bool) -> List[str]:
    if use_nltk:
        try:
            from nltk.tokenize import sent_tokenize
            return sent_tokenize(text)
        except (ImportError, LookupError):
            pass
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sentences if len(s.strip()) > 8]


def make_windows(sentences: List[str], min_s: int, max_s: int, seed: int) -> List[str]:
    windows, i, n = [], 0, len(sentences)
    span = max_s - min_s
    while i < n:
        size = min_s + ((i * 7 + seed) % (span + 1)) if span > 0 else min_s
        size = min(size, n - i)
        windows.append(" ".join(sentences[i:i + size]))
        i += size
    return windows


def take_fraction(windows: List[str], fraction: float, max_windows: int) -> List[str]:
    k = int(len(windows) * fraction)
    if max_windows:
        k = min(k, max_windows)
    return windows[:k]


def save_jsonl(items: List[dict], path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for d in items:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")


def stats(items: List[str]) -> Tuple[int, float, float]:
    toks = [len(w.split()) for w in items]
    n = len(toks)
    return n, (sum(toks) / n if n else 0.0), (max(toks) if toks else 0)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Create the Phase 0 dataset (Stage A)")
    ap.add_argument("--source", default="wikitext-103",
                    help="'wikitext-103' (default, HF) OR a path to .txt")
    ap.add_argument("--cache-dir", default="datasets/repro/.cache", help="Cache for parquet (fallback)")
    ap.add_argument("--fraction", type=float, default=1.0,
                    help="Fraction of train windows (0..1); scale parameter (100k+)")
    ap.add_argument("--max-windows", type=int, default=0, help="Cap train windows (0 = no cap)")
    ap.add_argument("--min-sentences", type=int, default=4)
    ap.add_argument("--max-sentences", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--use-nltk", action="store_true", help="nltk for splitting (if available)")
    ap.add_argument("--output-dir", default="datasets/repro", help="Where to write train/heldout jsonl")
    args = ap.parse_args()

    # 1. Text
    print("[1/5] Loading WikiText-103-raw-v1 ...")
    if args.source != "wikitext-103":
        files = Path(args.source)
        train_texts = [f.read_text(encoding="utf-8") for f in
                       ([files] if files.is_file() else sorted(files.rglob("*.txt")))]
        heldout_texts = []
    else:
        data = load_texts(args.cache_dir)
        train_texts, heldout_texts = data["train"], data["heldout"]
    print(f"      train lines: {len(train_texts):,} | heldout lines: {len(heldout_texts):,}")

    # 2. Windows
    print("[2/5] Splitting into sentences and grouping into windows...")
    def windows_from(texts, seed):
        sent = []
        for t in texts:
            sent.extend(split_sentences(t, args.use_nltk))
        return make_windows(sent, args.min_sentences, args.max_sentences, seed)

    train_w = windows_from(train_texts, args.seed)
    heldout_w = windows_from(heldout_texts, args.seed + 1000) if heldout_texts else []

    # 3. Take the train fraction
    print(f"[3/5] Train windows: {len(train_w):,}  (fraction {args.fraction}, cap={args.max_windows or '—'})")
    train_sel = take_fraction(train_w, args.fraction, args.max_windows)

    # 4. Save
    print("[4/5] Saving...")
    out_dir = args.output_dir
    train_items = [{"id": i, "split": "train", "text": w} for i, w in enumerate(train_sel)]
    heldout_items = [{"id": i, "split": "heldout", "text": w} for i, w in enumerate(heldout_w)]
    save_jsonl(train_items, os.path.join(out_dir, "train.jsonl"))
    save_jsonl(heldout_items, os.path.join(out_dir, "heldout.jsonl"))

    # 5. Stats
    print("[5/5] Done.")
    nt, at, mt = stats([d["text"] for d in train_items])
    nh, ah, mh = stats([d["text"] for d in heldout_items])
    print(f"  train.jsonl   : {nt:,} windows | avg {at:.0f} tok | max {mt} tok")
    print(f"  heldout.jsonl : {nh:,} windows | avg {ah:.0f} tok | max {mh} tok")
    print(f"  -> {out_dir}/")


if __name__ == "__main__":
    main()
