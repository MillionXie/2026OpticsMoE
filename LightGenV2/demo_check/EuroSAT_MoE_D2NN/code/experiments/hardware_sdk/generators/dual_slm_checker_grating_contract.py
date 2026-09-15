"""Validate the compact normal-polarity dual-SLM registration pair."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from PIL import Image


SOURCE_DIRECTORY_NAME = "recommended_checker_grating_pair"
ARCHIVE_ROOT = "payload/calibration/dual_slm_checker_grating"
AMPLITUDE_NAME = "amplitude_checker_255open_c64_1024x1024.bmp"
PHASE_NAME = "phase_grating_xy_in_255open_cells_c64_p8_1920x1200.bmp"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_file(path: Path, label: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"Required dual-SLM {label} is missing: {path}")
    return path


def _validate_bmp(path: Path, size: tuple[int, int]) -> None:
    with Image.open(path) as image:
        if image.format != "BMP" or image.mode != "L" or image.size != size:
            raise RuntimeError(
                f"Invalid dual-SLM BMP {path}: format={image.format}, mode={image.mode}, "
                f"size={image.size}, expected={size}"
            )


def validate_dual_slm_checker_grating_pair(
    pair_dir: str | Path,
) -> tuple[list[Path], dict[str, Any]]:
    """Validate source pair_manifest and return a path-portable bundle contract.

    The source manifest may contain absolute generation-machine paths.  Those
    fields are never copied into the laboratory bundle.  The returned contract
    uses archive-relative paths exclusively.
    """

    root = Path(pair_dir).expanduser().resolve()
    manifest_path = _require_file(root / "pair_manifest.json", "pair manifest")
    amplitude_path = _require_file(root / AMPLITUDE_NAME, "amplitude BMP")
    phase_path = _require_file(root / PHASE_NAME, "phase BMP")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if int(manifest.get("schema_version", -1)) != 1:
        raise RuntimeError("Dual-SLM pair manifest schema must be 1")
    if manifest.get("pair_id") != "recommended_checker_grating_pair":
        raise RuntimeError("Dual-SLM pair manifest has the wrong pair_id")
    if manifest.get("use_only_as_a_pair") is not True:
        raise RuntimeError("Dual-SLM checker and grating must be declared as one pair")
    command = manifest.get("amplitude_command_contract", {})
    if (
        command.get("white_open_value_uint8") != 255
        or command.get("black_closed_value_uint8") != 0
        or command.get("invert_in_player") is not False
    ):
        raise RuntimeError("Dual-SLM pair does not use the normal 255=open polarity")
    phase_rule = str(manifest.get("phase_rule", ""))
    if "phase=0" not in phase_rule or "amplitude-255" not in phase_rule:
        raise RuntimeError("Dual-SLM pair phase rule is not white-cell-only grating")
    transform = manifest.get("phase_transform", {})
    if (
        transform.get("flip_vertical_before_raster") is not True
        or transform.get("flip_horizontal_before_raster") is not False
    ):
        raise RuntimeError("Dual-SLM pair phase flip contract mismatch")
    if transform.get("center_xy") != [980.0, 590.0]:
        raise RuntimeError("Dual-SLM pair phase centre must be [980,590]")
    _validate_bmp(amplitude_path, (1024, 1024))
    _validate_bmp(phase_path, (1920, 1200))
    with Image.open(amplitude_path) as image:
        histogram = image.histogram()
        if sum(histogram[1:255]) != 0 or histogram[0] == 0 or histogram[255] == 0:
            raise RuntimeError("Dual-SLM checker amplitude must be non-empty binary 0/255")
    amplitude_sha = _sha256(amplitude_path)
    phase_sha = _sha256(phase_path)
    if str(manifest.get("amplitude", {}).get("sha256", "")).lower() != amplitude_sha:
        raise RuntimeError("Dual-SLM checker amplitude SHA-256 mismatch")
    if str(manifest.get("phase", {}).get("sha256", "")).lower() != phase_sha:
        raise RuntimeError("Dual-SLM grating phase SHA-256 mismatch")
    if manifest.get("amplitude", {}).get("active_bounds_xyxy") != [273, 273, 751, 751]:
        raise RuntimeError("Dual-SLM checker active bounds mismatch")
    if manifest.get("phase", {}).get("active_bounds_xyxy") != [472, 82, 1488, 1098]:
        raise RuntimeError("Dual-SLM grating active bounds mismatch")

    contract = {
        "pair_id": "recommended_checker_grating_pair",
        "use_only_as_a_pair": True,
        "recommended_first_alignment_step": True,
        "amplitude": {
            "archive_path": f"{ARCHIVE_ROOT}/{AMPLITUDE_NAME}",
            "sha256": amplitude_sha,
            "size_wh": [1024, 1024],
            "polarity": "255=open/transmissive; 0=closed/opaque",
            "invert_in_player": False,
            "active_bounds_xyxy": [273, 273, 751, 751],
        },
        "phase": {
            "archive_path": f"{ARCHIVE_ROOT}/{PHASE_NAME}",
            "sha256": phase_sha,
            "size_wh": [1920, 1200],
            "centre_xy": [980.0, 590.0],
            "vertical_flip_already_applied": True,
            "horizontal_flip_already_applied": False,
            "active_bounds_xyxy": [472, 82, 1488, 1098],
            "rule": "zero in amplitude-0 cells; alternating x/y grating only in amplitude-255 cells",
        },
        "source_pair_manifest_sha256": _sha256(manifest_path),
        "source_absolute_paths_copied_to_bundle": False,
        "vendor_sdk_examples_are_calibration_evidence": False,
    }
    return [amplitude_path, phase_path], contract
