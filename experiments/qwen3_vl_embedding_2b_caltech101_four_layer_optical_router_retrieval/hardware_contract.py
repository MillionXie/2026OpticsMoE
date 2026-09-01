from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from PIL import Image


SIX_STAGES = (
    "vision_router",
    "vision_expert",
    "vision_global",
    "language_router",
    "language_expert",
    "language_global",
)
ROUTER_STAGES = ("vision_router", "language_router")
FEATURE_STAGES = (
    "vision_expert",
    "vision_global",
    "language_expert",
    "language_global",
)
IMAGE_SUFFIXES = (".png", ".bmp", ".tif", ".tiff")


def sha256_file(path: Path) -> str:
    path = path.expanduser().resolve()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_canonical_json(value: Any) -> str:
    """Hash an already JSON-compatible value without formatting ambiguity."""

    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def stage_directory(session_dir: Path, stage: str) -> Path:
    if stage not in SIX_STAGES:
        raise ValueError(f"Unknown six-stage optical stage {stage!r}")
    return session_dir.expanduser().resolve() / f"{SIX_STAGES.index(stage) + 1:02d}_{stage}"


def require_empty_directory(path: Path, *, label: str) -> Path:
    """Create a destination only when it cannot contain stale artifacts."""

    path = path.expanduser().resolve()
    if path.exists():
        if not path.is_dir():
            raise RuntimeError(f"{label} is not a directory: {path}")
        existing = sorted(item.name for item in path.iterdir())
        if existing:
            preview = ", ".join(existing[:8])
            suffix = " ..." if len(existing) > 8 else ""
            raise RuntimeError(
                f"{label} must be new or empty; refusing stale files in {path}: "
                f"{preview}{suffix}"
            )
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dict(value), indent=2, ensure_ascii=False), encoding="utf-8"
    )


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON contract must contain an object: {path}")
    return value


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write an empty CSV contract: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError(f"CSV contract is empty: {path}")
    return rows


def unique_image_files(path: Path) -> list[Path]:
    path = path.expanduser().resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"Image directory is missing: {path}")
    files = sorted(
        item
        for item in path.iterdir()
        if item.is_file() and item.suffix.lower() in IMAGE_SUFFIXES
    )
    if not files:
        raise FileNotFoundError(f"No supported image files found in {path}")
    stems: dict[str, list[str]] = {}
    for item in files:
        stems.setdefault(item.stem, []).append(item.name)
    duplicates = {key: names for key, names in stems.items() if len(names) != 1}
    if duplicates:
        raise RuntimeError(f"Image stems are not unique in {path}: {duplicates}")
    return files


def validate_grayscale_image(
    path: Path,
    *,
    expected_shape_hw: tuple[int, int],
    require_uint8: bool = True,
) -> dict[str, Any]:
    with Image.open(path) as image:
        mode = image.mode
        size = image.size
    expected_wh = (int(expected_shape_hw[1]), int(expected_shape_hw[0]))
    if size != expected_wh:
        raise RuntimeError(
            f"Image {path} is {size[0]}x{size[1]}; expected "
            f"{expected_wh[0]}x{expected_wh[1]}"
        )
    if require_uint8 and mode != "L":
        raise RuntimeError(
            f"Image {path} must be a single-channel uint8 image (PIL mode L), "
            f"got {mode!r}"
        )
    return {
        "filename": path.name,
        "sha256": sha256_file(path),
        "mode": mode,
        "width": size[0],
        "height": size[1],
    }


def expected_key_files(
    image_dir: Path,
    expected_keys: Iterable[str],
    *,
    expected_shape_hw: tuple[int, int],
    require_uint8: bool = True,
) -> list[dict[str, Any]]:
    """Require exactly one image per key and reject all unrelated images."""

    paths = unique_image_files(image_dir)
    by_stem = {path.stem: path for path in paths}
    keys = list(expected_keys)
    if len(keys) != len(set(keys)):
        raise RuntimeError("Expected sample keys are not unique")
    missing = sorted(set(keys).difference(by_stem))
    unexpected = sorted(set(by_stem).difference(keys))
    if missing or unexpected:
        raise RuntimeError(
            f"Image/key contract mismatch below {image_dir}: "
            f"missing={missing[:8]}, unexpected={unexpected[:8]}"
        )
    rows: list[dict[str, Any]] = []
    for order, key in enumerate(keys):
        row = validate_grayscale_image(
            by_stem[key],
            expected_shape_hw=expected_shape_hw,
            require_uint8=require_uint8,
        )
        rows.append({"order": order, "key": key, **row})
    return rows


def initialize_state(
    *,
    config: Path,
    checkpoint: Path,
    manifest: Path,
    architecture: str,
) -> dict[str, Any]:
    return {
        "schema_version": 2,
        "six_stage_order": list(SIX_STAGES),
        "config": str(config.expanduser().resolve()),
        "config_sha256": sha256_file(config),
        "initial_checkpoint": str(checkpoint.expanduser().resolve()),
        "initial_checkpoint_sha256": sha256_file(checkpoint),
        "current_checkpoint": str(checkpoint.expanduser().resolve()),
        "current_checkpoint_sha256": sha256_file(checkpoint),
        "checkpoint_architecture": architecture,
        "dataset_manifest": str(manifest.expanduser().resolve()),
        "dataset_manifest_sha256": sha256_file(manifest),
        "stages": {stage: {} for stage in SIX_STAGES},
        "events": [],
    }


def validate_state_identity(
    state: Mapping[str, Any], *, config: Path, checkpoint: Path, manifest: Path
) -> None:
    checks = {
        "config_sha256": sha256_file(config),
        "current_checkpoint_sha256": sha256_file(checkpoint),
        "dataset_manifest_sha256": sha256_file(manifest),
    }
    mismatches = {
        key: {"expected": state.get(key), "actual": value}
        for key, value in checks.items()
        if state.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"Six-stage session identity mismatch: {mismatches}")


def record_event(
    state: dict[str, Any],
    *,
    stage: str | None,
    action: str,
    payload: Mapping[str, Any],
) -> None:
    if stage is not None and stage not in SIX_STAGES:
        raise ValueError(f"Unknown six-stage optical stage {stage!r}")
    event = {"index": len(state["events"]), "stage": stage, "action": action, **payload}
    state["events"].append(event)
    if stage is not None:
        state["stages"][stage][action] = event


__all__ = [
    "FEATURE_STAGES",
    "IMAGE_SUFFIXES",
    "ROUTER_STAGES",
    "SIX_STAGES",
    "expected_key_files",
    "initialize_state",
    "read_csv",
    "read_json",
    "record_event",
    "require_empty_directory",
    "sha256_canonical_json",
    "sha256_file",
    "stage_directory",
    "unique_image_files",
    "validate_grayscale_image",
    "validate_state_identity",
    "write_csv",
    "write_json",
]
