from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence


EXPERIMENT_DIR = Path(__file__).resolve().parent
PATH_KEYS = {"data_root", "output_dir"}


def parse_args_with_config(
    parser: argparse.ArgumentParser, argv: Sequence[str] | None = None
) -> argparse.Namespace:
    """Load JSON defaults first, then let explicit CLI arguments override them."""

    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=Path, default=None)
    known, _ = config_parser.parse_known_args(argv)

    config_path: Path | None = None
    if known.config is not None:
        config_path = known.config.expanduser().resolve()
        defaults = _load_json_defaults(config_path, parser)
        parser.set_defaults(**defaults)

    args = parser.parse_args(argv)
    args.config = config_path
    args.data_root = _resolve_runtime_path(args.data_root)
    args.output_dir = _resolve_runtime_path(args.output_dir)
    return args


def _load_json_defaults(path: Path, parser: argparse.ArgumentParser) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Config file does not exist: {path}")
    try:
        with path.open("r", encoding="utf-8") as handle:
            values = json.load(handle)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON config {path}: {exc}") from exc
    if not isinstance(values, dict):
        raise ValueError(f"Config root must be a JSON object: {path}")

    allowed = {action.dest for action in parser._actions if action.dest != "help"}
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise ValueError(f"Unknown config keys in {path}: {', '.join(unknown)}")
    if "config" in values:
        raise ValueError("A config file cannot set the 'config' key.")

    resolved = dict(values)
    for key in PATH_KEYS:
        if key in resolved:
            value = resolved[key]
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"Config key '{key}' must be a non-empty path string.")
            candidate = Path(value).expanduser()
            if not candidate.is_absolute():
                candidate = path.parent / candidate
            resolved[key] = candidate.resolve()
    return resolved


def _resolve_runtime_path(value: Path | str) -> Path:
    path = Path(value).expanduser()
    return path.resolve()
