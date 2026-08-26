"""Export several checkpoint-compatible phase masks without rebuilding amplitudes."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from experiments.hardware_sdk.workflows.reconstruct_slm import (
    encode_active_phase,
    physical_pitch_nearest,
    place_at_center,
)

from .modeling import RobustRawCCDMNIST4D2NN
from .settings import load_settings
from .training import load_checkpoint


_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _parse_candidate(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError("Each --candidate must be NAME=CHECKPOINT")
    name, raw_path = value.split("=", 1)
    name = name.strip()
    if not _SAFE_NAME.fullmatch(name):
        raise ValueError(f"Unsafe candidate name: {name!r}")
    path = Path(raw_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Candidate checkpoint is missing: {path}")
    return name, path


def _metric(payload: dict[str, Any], name: str) -> float | None:
    value = payload.get("metrics", {}).get(name)
    return None if value is None else float(value)


@torch.no_grad()
def export_candidates(
    *,
    config: str | Path,
    candidates: list[str],
    output_dir: str | Path,
) -> dict[str, Any]:
    if not candidates:
        raise ValueError("At least one --candidate is required")
    settings = load_settings(config)
    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    parsed = [_parse_candidate(value) for value in candidates]
    if len({name for name, _ in parsed}) != len(parsed):
        raise ValueError("Candidate names must be unique")

    rows: list[dict[str, Any]] = []
    for name, checkpoint in parsed:
        model = RobustRawCCDMNIST4D2NN(settings).cpu().eval()
        payload = load_checkpoint(checkpoint, model, torch.device("cpu"))
        logical_phase = encode_active_phase(model.phase().cpu().numpy())
        if settings.phase_flip_vertical:
            logical_phase = np.flipud(logical_phase)
        if settings.phase_flip_horizontal:
            logical_phase = np.fliplr(logical_phase)
        native_phase = physical_pitch_nearest(
            np.ascontiguousarray(logical_phase),
            logical_pixel_pitch_um=settings.logical_pixel_pitch_um,
            slm_pixel_pitch_um=settings.phase_slm_pixel_pitch_um,
        )
        phase_image, bounds, actual_center = place_at_center(
            Image.fromarray(native_phase, mode="L"),
            slm_size_wh=settings.phase_slm_size_wh,
            center_xy=settings.phase_slm_center_xy,
        )
        epoch = int(payload.get("epoch", -1))
        filename = f"{name}_epoch{epoch:03d}_1920x1200.bmp"
        path = destination / filename
        phase_image.save(path, format="BMP")
        with Image.open(path) as check:
            if check.mode != "L" or check.size != tuple(settings.phase_slm_size_wh):
                raise RuntimeError(f"Invalid exported phase BMP: {path}")
        rows.append(
            {
                "name": name,
                "file": filename,
                "sha256": _sha256(path),
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": _sha256(checkpoint),
                "epoch": epoch,
                "validation_accuracy": _metric(payload, "validation_accuracy"),
                "validation_loss": _metric(payload, "validation_loss"),
                "phase_std_rad": _metric(payload, "phase_std_rad"),
                "phase_bounds_xyxy": list(bounds),
                "actual_phase_center_xy": list(actual_center),
            }
        )
    report = {
        "schema_version": 1,
        "config": str(Path(config).expanduser().resolve()),
        "logical_phase_shape": [settings.active_size, settings.active_size],
        "phase_slm_size_wh": list(settings.phase_slm_size_wh),
        "phase_slm_center_xy": list(settings.phase_slm_center_xy),
        "phase_flip_vertical": settings.phase_flip_vertical,
        "phase_flip_horizontal": settings.phase_flip_horizontal,
        "detector_bounds_xyxy": [list(value) for value in settings.detector_bounds()],
        "candidates": rows,
    }
    (destination / "mask_candidates.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Export multiple MNIST-4 phase-only BMP candidates"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--candidate",
        action="append",
        default=[],
        metavar="NAME=CHECKPOINT",
    )
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    report = export_candidates(
        config=args.config,
        candidates=args.candidate,
        output_dir=args.output_dir,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
