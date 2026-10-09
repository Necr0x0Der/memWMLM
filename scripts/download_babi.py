#!/usr/bin/env python3
"""Download and safely extract the original bAbI Tasks v1.2 archive."""

from __future__ import annotations

import argparse
import shutil
import tarfile
import urllib.request
from pathlib import Path


ARCHIVE_URLS = (
    "https://s3.amazonaws.com/text-datasets/babi_tasks_1-20_v1-2.tar.gz",
    "https://dl.fbaipublicfiles.com/babi/tasks_1-20_v1-2.tar.gz",
)
ROOT_DIR = "tasks_1-20_v1-2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--force", action="store_true", help="Replace an existing extracted dataset.")
    return parser.parse_args()


def safe_extract(archive: tarfile.TarFile, destination: Path) -> None:
    """Extract only archive members that remain within ``destination``."""

    root = destination.resolve()
    for member in archive.getmembers():
        member_path = (destination / member.name).resolve()
        if not member_path.is_relative_to(root):
            raise ValueError(f"Refusing unsafe archive member: {member.name!r}")
    archive.extractall(destination, filter="data")


def main() -> None:
    args = parse_args()
    output_dir: Path = args.output_dir
    dataset_root = output_dir / ROOT_DIR
    if dataset_root.exists() and not args.force:
        print(f"Dataset already exists: {dataset_root}")
        return
    if dataset_root.exists():
        shutil.rmtree(dataset_root)

    output_dir.mkdir(parents=True, exist_ok=True)
    archive_path = output_dir / "tasks_1-20_v1-2.tar.gz"
    download_error: Exception | None = None
    for archive_url in ARCHIVE_URLS:
        try:
            print(f"Downloading {archive_url}")
            urllib.request.urlretrieve(archive_url, archive_path)
            break
        except Exception as error:  # Try the public mirror if a host is unavailable.
            download_error = error
            archive_path.unlink(missing_ok=True)
    else:
        raise RuntimeError("Could not download bAbI archive from any configured mirror") from download_error

    print(f"Extracting {archive_path}")
    with tarfile.open(archive_path, "r:gz") as archive:
        safe_extract(archive, output_dir)
    required = dataset_root / "en-10k" / "qa1_single-supporting-fact_train.txt"
    if not required.is_file():
        raise RuntimeError(f"Archive did not contain expected file: {required}")
    print(f"Ready: {dataset_root}")


if __name__ == "__main__":
    main()
