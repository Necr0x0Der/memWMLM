#!/usr/bin/env python3
"""Create a deterministic answer-order-shuffled copy of the MMLU CSV data."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import tempfile
from pathlib import Path

from mmlu import CHOICES


ALGORITHM = "sha256_sort_by_seed_example_id_and_original_index_v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def permutation_for(seed: int, example_id: str) -> tuple[int, int, int, int]:
    """Return new-position -> original-position, stable across machines and runs."""
    prefix = f"{seed}\0{example_id}\0".encode()
    return tuple(
        sorted(range(4), key=lambda index: hashlib.sha256(prefix + str(index).encode()).digest())
    )  # type: ignore[return-value]


def shuffled_row(row: list[str], order: tuple[int, int, int, int]) -> tuple[list[str], str]:
    if len(row) != 6 or row[5] not in CHOICES:
        raise ValueError(f"Malformed MMLU row: expected question, four choices, and A/B/C/D; got {row!r}")
    original_answer_index = CHOICES.index(row[5])
    shifted_answer = CHOICES[order.index(original_answer_index)]
    return [row[0], *(row[1 + index] for index in order), shifted_answer], shifted_answer


def prepare_dataset(source_dir: Path, output_dir: Path, seed: int) -> Path:
    """Copy dev unchanged and deterministically shuffle every test row."""
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output directory: {output_dir}")
    dev_files = sorted((source_dir / "dev").glob("*_dev.csv"))
    test_files = sorted((source_dir / "test").glob("*_test.csv"))
    if len(dev_files) != 57 or len(test_files) != 57:
        raise ValueError(
            f"Expected 57 MMLU dev and test files, found {len(dev_files)} dev and {len(test_files)} test"
        )

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}-", dir=output_dir.parent))
    try:
        (temporary / "dev").mkdir()
        (temporary / "test").mkdir()
        manifest: dict = {
            "algorithm": ALGORITHM,
            "seed": seed,
            "source_dir": str(source_dir),
            "dev_policy": "copied byte-for-byte; five-shot demonstrations unchanged",
            "test_policy": "four choices deterministically permuted; answer letter remapped",
            "files": {},
            "examples": [],
        }

        for source in dev_files:
            destination = temporary / "dev" / source.name
            shutil.copyfile(source, destination)
            manifest["files"][f"dev/{source.name}"] = {
                "source_sha256": sha256(source),
                "output_sha256": sha256(destination),
                "transformation": "copied",
            }

        identity_count = 0
        for source in test_files:
            subject = source.name.removesuffix("_test.csv")
            destination = temporary / "test" / source.name
            with source.open(encoding="utf-8", newline="") as source_handle, destination.open(
                "w", encoding="utf-8", newline=""
            ) as destination_handle:
                reader = csv.reader(source_handle)
                writer = csv.writer(destination_handle)
                row_count = 0
                for row_index, row in enumerate(reader):
                    example_id = f"{subject}:test:{row_index}"
                    order = permutation_for(seed, example_id)
                    transformed, shifted_answer = shuffled_row(row, order)
                    writer.writerow(transformed)
                    identity_count += order == (0, 1, 2, 3)
                    manifest["examples"].append(
                        {
                            "example_id": example_id,
                            "new_to_original_choice_indices": list(order),
                            "original_answer": row[5],
                            "shifted_answer": shifted_answer,
                        }
                    )
                    row_count += 1
            manifest["files"][f"test/{source.name}"] = {
                "source_sha256": sha256(source),
                "output_sha256": sha256(destination),
                "transformation": "choice_order_shuffled",
                "examples": row_count,
            }

        manifest["example_count"] = len(manifest["examples"])
        manifest["identity_permutation_count"] = identity_count
        manifest_path = temporary / "permutation-manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(output_dir)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return output_dir / "permutation-manifest.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = prepare_dataset(args.source_dir, args.output_dir, args.seed)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "manifest": str(manifest),
                "examples": payload["example_count"],
                "identity_permutations": payload["identity_permutation_count"],
            }
        )
    )


if __name__ == "__main__":
    main()
