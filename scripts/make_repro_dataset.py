#!/usr/bin/env python3
"""
make_repro_dataset.py — создание датасета для Phase 0 (Stage A: exact reproduction).

Консервативно: НЕ меняет memWMLM, только готовит датасет.

Источник: Salesforce/wikitext, config wikitext-103-raw-v1.
  - Основной путь: load_dataset (нужен `pip install datasets`).
  - Fallback: ручной даунлоад parquet с HF hub (если datasets не установлен).

Логика:
  1. Грузим train + validation + test (raw-v1).
  2. train → train окна; validation+test → heldout окна (never seen при обучении).
  3. Фильтруем пустые/короткие строки, разбиваем на предложения, группируем
     в окна (min..max предложений).
  4. Берём --fraction train-окон (параметр масштаба, 100k+).
  5. Сохраняем train.jsonl и heldout.jsonl (id, split, text).

Примеры:
  # Полный датасет (~120k train окон):
  python scripts/make_repro_dataset.py --fraction 1.0 --max-windows 120000 --output-dir datasets/repro

  # Быстрый тест (5% train):
  python scripts/make_repro_dataset.py --fraction 0.05 --output-dir datasets/repro

Зависимости: datasets (желательно), pyarrow, pandas, requests.
"""

import argparse
import json
import os
import re
import requests
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

warnings.filterwarnings("ignore")  # insecure-request warnings (публичный датасет)

REPO = "Salesforce/wikitext"
CFG = "wikitext-103-raw-v1"
HF_BASE = f"https://huggingface.co/datasets/{REPO}/resolve/main"
# файлы для fallback-даунлоада (если datasets не установлен)
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
# Загрузка источника
# ---------------------------------------------------------------------------

def _clean(texts: List[str], min_len: int = 15) -> List[str]:
    """Убирает пустые/короткие строки (blank lines в raw-формате)."""
    return [t.strip() for t in texts if len(t.strip()) >= min_len]


def load_via_hf() -> Dict[str, List[str]]:
    """Основной путь: load_dataset (Salesforce/wikitext, raw-v1)."""
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
    print(f"      скачиваем {url} ...")
    r = requests.get(url, timeout=600, stream=True)
    r.raise_for_status()
    with open(dest, "wb") as f:
        for chunk in r.iter_content(65536):
            if chunk:
                f.write(chunk)
    return dest


def load_via_manual(cache_dir: str) -> Dict[str, List[str]]:
    """Fallback: ручной даунлоад parquet + pyarrow."""
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
    """Пытается load_dataset, иначе — ручной parquet."""
    try:
        data = load_via_hf()
        # проверяем, что реально прочитали непустой текст
        if len(data["train"]) == 0:
            raise RuntimeError("load_dataset вернул пустой train")
        return data
    except ImportError:
        print("      [fallback] datasets не установлен → ручной parquet")
        return load_via_manual(cache_dir)


# ---------------------------------------------------------------------------
# Обработка
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
    ap = argparse.ArgumentParser(description="Создание датасета Phase 0 (Stage A)")
    ap.add_argument("--source", default="wikitext-103",
                    help="'wikitext-103' (по умолчанию, HF) ИЛИ путь к .txt")
    ap.add_argument("--cache-dir", default="datasets/repro/.cache", help="Кэш для parquet (fallback)")
    ap.add_argument("--fraction", type=float, default=1.0,
                    help="Доля train-окон (0..1); параметр масштаба (100k+)")
    ap.add_argument("--max-windows", type=int, default=0, help="Cap train-окон (0 = без cap)")
    ap.add_argument("--min-sentences", type=int, default=4)
    ap.add_argument("--max-sentences", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--use-nltk", action="store_true", help="nltk для split (если доступен)")
    ap.add_argument("--output-dir", default="datasets/repro", help="Куда писать train/heldout jsonl")
    args = ap.parse_args()

    # 1. Текст
    print("[1/5] Грузим WikiText-103-raw-v1 ...")
    if args.source != "wikitext-103":
        files = Path(args.source)
        train_texts = [f.read_text(encoding="utf-8") for f in
                       ([files] if files.is_file() else sorted(files.rglob("*.txt")))]
        heldout_texts = []
    else:
        data = load_texts(args.cache_dir)
        train_texts, heldout_texts = data["train"], data["heldout"]
    print(f"      train строк: {len(train_texts):,} | heldout строк: {len(heldout_texts):,}")

    # 2. Окна
    print("[2/5] Разбиваем на предложения и группируем в окна...")
    def windows_from(texts, seed):
        sent = []
        for t in texts:
            sent.extend(split_sentences(t, args.use_nltk))
        return make_windows(sent, args.min_sentences, args.max_sentences, seed)

    train_w = windows_from(train_texts, args.seed)
    heldout_w = windows_from(heldout_texts, args.seed + 1000) if heldout_texts else []

    # 3. Берём долю train
    print(f"[3/5] Train-окон: {len(train_w):,}  (доля {args.fraction}, cap={args.max_windows or '—'})")
    train_sel = take_fraction(train_w, args.fraction, args.max_windows)

    # 4. Сохраняем
    print("[4/5] Сохраняем...")
    out_dir = args.output_dir
    train_items = [{"id": i, "split": "train", "text": w} for i, w in enumerate(train_sel)]
    heldout_items = [{"id": i, "split": "heldout", "text": w} for i, w in enumerate(heldout_w)]
    save_jsonl(train_items, os.path.join(out_dir, "train.jsonl"))
    save_jsonl(heldout_items, os.path.join(out_dir, "heldout.jsonl"))

    # 5. Статистика
    print("[5/5] Готово.")
    nt, at, mt = stats([d["text"] for d in train_items])
    nh, ah, mh = stats([d["text"] for d in heldout_items])
    print(f"  train.jsonl   : {nt:,} окон | avg {at:.0f} ток | max {mt} ток")
    print(f"  heldout.jsonl : {nh:,} окон | avg {ah:.0f} ток | max {mh} ток")
    print(f"  -> {out_dir}/")


if __name__ == "__main__":
    main()
