"""Small YAML configuration helper for experiment entrypoints."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml


def load_config_defaults(path: str | Path | None) -> dict[str, Any]:
    """Load a flat YAML mapping suitable for ``ArgumentParser.set_defaults``."""

    if path is None:
        return {}
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file does not exist: {config_path}")
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if data is None:
        return {}
    if not isinstance(data, dict) or any(isinstance(value, dict) for value in data.values()):
        raise ValueError(f"Configuration must be a flat YAML mapping: {config_path}")
    return data


def config_path_from_argv() -> Path | None:
    """Read only ``--config`` before constructing the full experiment parser."""

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", type=Path, default=None)
    return parser.parse_known_args()[0].config
