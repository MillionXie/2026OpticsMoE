"""Shared, model-free contracts for simulated-to-real CCD agreement.

The laboratory evaluator intentionally imports this module without importing
Torch, Transformers, or Qwen.  All paths in the manifests are relative to the
stage directory and every material payload is bound by SHA-256.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


STAGES = (
    "vision_expert",
    "vision_global",
    "language_expert",
    "language_global",
)
SCHEMA_VERSION = 1


def sha256_file(path: str | Path) -> str:
    source = Path(path)
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def stage_directory(session_dir: str | Path, stage: str) -> Path:
    if stage not in STAGES:
        raise ValueError(f"Unknown agreement stage {stage!r}")
    return Path(session_dir) / f"{STAGES.index(stage) + 1:02d}_{stage}"


def write_json(path: str | Path, value: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_csv(path: str | Path, rows: Iterable[dict[str, Any]], fields: Iterable[str]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(fields)
    values = list(rows)
    with destination.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        writer.writerows(values)


def read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def safe_relative_file(root: Path, value: str, *, label: str) -> Path:
    """Resolve a manifest-owned plain relative path without directory escape."""

    raw = Path(str(value))
    if raw.is_absolute() or ".." in raw.parts or not raw.parts:
        raise ValueError(f"{label} must be a safe relative path, got {value!r}")
    resolved_root = root.resolve()
    result = (resolved_root / raw).resolve()
    try:
        result.relative_to(resolved_root)
    except ValueError as error:
        raise ValueError(f"{label} escapes stage directory: {value!r}") from error
    return result


def require_sha256(value: str, *, label: str) -> str:
    digest = str(value).strip().lower()
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError(f"{label} is not a SHA-256 digest: {value!r}")
    return digest


def verify_file(path: Path, expected_sha256: str, *, label: str) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {label}: {path}")
    expected = require_sha256(expected_sha256, label=f"{label} expected digest")
    observed = sha256_file(path)
    if observed != expected:
        raise RuntimeError(
            f"{label} SHA-256 mismatch for {path.name}: expected={expected}, observed={observed}"
        )
    return observed


__all__ = [
    "SCHEMA_VERSION",
    "STAGES",
    "read_csv",
    "read_json",
    "require_sha256",
    "safe_relative_file",
    "sha256_file",
    "sha256_text",
    "stage_directory",
    "verify_file",
    "write_csv",
    "write_json",
]
