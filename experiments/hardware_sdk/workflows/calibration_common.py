"""Small shared helpers for hardware calibration and offline processing."""

from __future__ import annotations

import json
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from PIL import Image


FRAME_SUFFIXES = {".npy", ".png", ".tif", ".tiff", ".bmp"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def expand_environment(value: Any) -> str:
    raw = str(value)
    raw = re.sub(
        r"%([^%]+)%",
        lambda match: os.environ.get(match.group(1), match.group(0)),
        raw,
    )
    return os.path.expandvars(raw)


def resolve_path(value: str | Path, base: Path) -> Path:
    path = Path(expand_environment(value)).expanduser()
    return (path if path.is_absolute() else base / path).resolve()


def load_yaml_config(path: str | Path) -> tuple[dict[str, Any], Path]:
    config_path = Path(path).expanduser().resolve()
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Calibration config must be a YAML mapping: {config_path}")
    return raw, config_path


def load_frame(path: str | Path) -> np.ndarray:
    path = Path(path)
    if path.suffix.lower() == ".npy":
        value = np.load(path, allow_pickle=False)
    else:
        value = np.asarray(Image.open(path))
    value = np.asarray(value).squeeze()
    if value.ndim != 2:
        raise ValueError(f"Expected a monochrome 2-D frame, got {value.shape}: {path}")
    if not np.isfinite(value).all():
        raise ValueError(f"Frame contains NaN/Inf: {path}")
    return value


def save_tiff(path: str | Path, value: np.ndarray, *, force_uint16: bool = False) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    array = np.asarray(value)
    if force_uint16:
        array = np.rint(np.clip(array, 0, 65535)).astype(np.uint16)
    elif array.dtype not in (np.uint8, np.uint16, np.float32):
        array = array.astype(np.float32)
    Image.fromarray(array).save(path, format="TIFF")


def preview_uint8(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float32)
    finite = value[np.isfinite(value)]
    if finite.size == 0:
        return np.zeros(value.shape, dtype=np.uint8)
    low, high = np.percentile(finite, [1.0, 99.8])
    if high <= low:
        high = low + 1.0
    return np.rint(np.clip((value - low) / (high - low), 0, 1) * 255).astype(np.uint8)


def save_preview(path: str | Path, value: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(preview_uint8(value), mode="L").save(path)


def corrected_frame(raw: np.ndarray, background: np.ndarray) -> np.ndarray:
    if raw.shape != background.shape:
        raise ValueError(
            f"Raw/background shape mismatch: raw={raw.shape}, background={background.shape}"
        )
    return np.maximum(raw.astype(np.float32) - background.astype(np.float32), 0.0)


def capture_array(camera: Any, temporary_dir: Path, stem: str) -> tuple[np.ndarray, dict[str, Any]]:
    """Capture losslessly through the existing CameraDriver interface."""
    temporary_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=f".{stem}_", suffix=".npy", dir=temporary_dir, delete=False
    ) as handle:
        capture_path = Path(handle.name)
    capture_path.unlink(missing_ok=True)
    try:
        camera.capture(capture_path)
        value = load_frame(capture_path)
        info = dict(camera.device_info().get("last_capture") or {})
        return value, info
    finally:
        capture_path.unlink(missing_ok=True)


def median_capture(
    camera: Any, temporary_dir: Path, stem: str, frame_count: int
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    if frame_count <= 0:
        raise ValueError("frame_count must be positive")
    frames: list[np.ndarray] = []
    metadata: list[dict[str, Any]] = []
    for index in range(frame_count):
        frame, info = capture_array(camera, temporary_dir, f"{stem}_{index:03d}")
        frames.append(frame)
        metadata.append(info)
    shapes = {frame.shape for frame in frames}
    if len(shapes) != 1:
        raise RuntimeError(f"Camera frame shape changed during capture: {sorted(shapes)}")
    return np.median(np.stack(frames).astype(np.float32), axis=0), metadata


def json_dump(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
