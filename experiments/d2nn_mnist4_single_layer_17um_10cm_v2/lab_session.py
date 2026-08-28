"""Reconstruct one strict MNIST-4 laboratory session from the compact bundle."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image


PROJECT_PACKAGE = "d2nn_mnist4_single_layer_17um_10cm_v2"
PROFILES = {"quick40": (40, 10, False), "formal400": (400, 100, True)}
ACTIVE_BOUNDS = (273, 273, 751, 751)
ACTIVE_SIZE = (478, 478)
AMPLITUDE_SIZE = (1024, 1024)
PHASE_SIZE = (1920, 1200)
CLASS_IDS = (0, 1, 2, 3)
MARKER_NAME = ".mnist4_lab_session.json"


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected a JSON object: {path}")
    return value


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError("Cannot write an empty session manifest")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def _default_bundle_root() -> Path:
    return Path(__file__).resolve().parents[1] / "lab_qwen" / "mnist4"


def _safe_replace_target(target: Path, overwrite: bool) -> None:
    if not target.exists():
        return
    if not target.is_dir():
        raise FileExistsError(f"Session target is not a directory: {target}")
    marker = target / MARKER_NAME
    if not overwrite:
        raise FileExistsError(
            f"Session already exists: {target}. Use a new directory or --overwrite."
        )
    if not marker.is_file():
        raise RuntimeError(
            f"Refusing --overwrite because the ownership marker is missing: {marker}"
        )
    marker_value = _read_json(marker)
    if marker_value.get("owner") != PROJECT_PACKAGE:
        raise RuntimeError(f"Refusing to remove a foreign session directory: {target}")
    shutil.rmtree(target)


def _select_mask(payload: Path, mask_name: str) -> tuple[dict[str, Any], Path]:
    evidence = _read_json(payload / "phase_masks" / "phase_masks.json")
    candidates = evidence.get("candidates", [])
    row = next(
        (value for value in candidates if value.get("name") == mask_name), None
    )
    if not isinstance(row, dict):
        available = [value.get("name") for value in candidates]
        raise ValueError(f"Unknown mask {mask_name!r}; available={available}")
    filename = str(row.get("file", ""))
    source = payload / "phase_masks" / filename
    if not source.is_file() or _sha256(source) != row.get("sha256"):
        raise RuntimeError(f"Phase mask hash validation failed: {source}")
    with Image.open(source) as image:
        image.load()
        if image.format != "BMP" or image.mode != "L" or image.size != PHASE_SIZE:
            raise RuntimeError(f"Invalid phase BMP geometry: {source}")
    return row, source


def _validate_source_rows(rows: list[dict[str, str]], profile: str) -> None:
    expected_count, per_class, _ = PROFILES[profile]
    if len(rows) != expected_count:
        raise RuntimeError(
            f"{profile} must contain {expected_count} rows, got {len(rows)}"
        )
    counts = Counter(int(row["label"]) for row in rows)
    if counts != Counter({label: per_class for label in CLASS_IDS}):
        raise RuntimeError(f"{profile} class counts are wrong: {dict(counts)}")
    keys = [row["key"] for row in rows]
    if any(not key or Path(key).name != key for key in keys) or len(keys) != len(
        set(keys)
    ):
        raise RuntimeError(f"{profile} contains unsafe or duplicate keys")


def _reconstruct_amplitude(
    compact_path: Path,
    destination: Path,
    row: dict[str, str],
) -> dict[str, str]:
    if _sha256(compact_path) != row["compact_sha256"]:
        raise RuntimeError(f"Compact PNG SHA-256 mismatch: {compact_path.name}")
    with Image.open(compact_path) as opened:
        opened.load()
        if opened.format != "PNG" or opened.mode != "L" or opened.size != ACTIVE_SIZE:
            raise RuntimeError(
                f"Compact amplitude must be 478x478 L-mode PNG: {compact_path}"
            )
        compact = opened.copy()
    if _sha256_bytes(compact.tobytes()) != row["compact_pixel_sha256"]:
        raise RuntimeError(f"Compact pixel SHA-256 mismatch: {compact_path.name}")
    canvas = Image.new("L", AMPLITUDE_SIZE, color=0)
    canvas.paste(compact, ACTIVE_BOUNDS[:2])
    full_pixel_sha = _sha256_bytes(canvas.tobytes())
    if full_pixel_sha != row["source_full_pixel_sha256"]:
        raise RuntimeError(
            f"1:1 reconstruction changed full-frame pixels: {compact_path.name}"
        )
    canvas.save(destination, format="BMP")
    with Image.open(destination) as checked:
        checked.load()
        if checked.format != "BMP" or checked.mode != "L" or checked.size != AMPLITUDE_SIZE:
            raise RuntimeError(f"Reconstructed amplitude BMP is invalid: {destination}")
        if _sha256_bytes(checked.tobytes()) != full_pixel_sha:
            raise RuntimeError(f"BMP round-trip changed pixels: {destination}")
    return {
        "amplitude_sha256": _sha256(destination),
        "amplitude_pixel_sha256": full_pixel_sha,
    }


def validate_session(stage_dir: str | Path) -> dict[str, Any]:
    stage = Path(stage_dir).expanduser().resolve()
    marker = _read_json(stage / MARKER_NAME)
    if marker.get("owner") != PROJECT_PACKAGE:
        raise RuntimeError("Session ownership marker is invalid")
    profile = str(marker.get("profile"))
    if profile not in PROFILES:
        raise RuntimeError(f"Session profile is invalid: {profile!r}")
    rows = _read_csv(stage / "samples.csv")
    _validate_source_rows(rows, profile)
    phase_files = sorted((stage / "phase_to_play").glob("*.bmp"))
    if len(phase_files) != 1 or _sha256(phase_files[0]) != marker.get("phase_sha256"):
        raise RuntimeError("Session phase file/hash is invalid")
    phase_manifest = _read_csv(
        stage / "phase_to_play" / "reconstruction_manifest.csv"
    )
    if (
        len(phase_manifest) != 1
        or phase_manifest[0].get("output_bmp") != phase_files[0].name
        or phase_manifest[0].get("output_sha256") != marker.get("phase_sha256")
    ):
        raise RuntimeError("Session phase reconstruction manifest is invalid")
    amplitude_dir = stage / "amplitude_to_play"
    expected = {f"{row['key']}.bmp" for row in rows}
    actual = {path.name for path in amplitude_dir.glob("*.bmp")}
    if actual != expected:
        raise RuntimeError("Session amplitude file set does not exactly match samples.csv")
    for index, row in enumerate(rows):
        path = amplitude_dir / row["amplitude_file"]
        if _sha256(path) != row["amplitude_sha256"]:
            raise RuntimeError(f"Reconstructed amplitude file hash mismatch: {path.name}")
        with Image.open(path) as image:
            image.load()
            if image.format != "BMP" or image.mode != "L" or image.size != AMPLITUDE_SIZE:
                raise RuntimeError(f"Invalid amplitude BMP: {path}")
            if _sha256_bytes(image.tobytes()) != row["amplitude_pixel_sha256"]:
                raise RuntimeError(f"Amplitude pixel hash mismatch: {path.name}")
        if (index + 1) % 100 == 0:
            print(f"[validate_session] {index + 1}/{len(rows)} amplitudes")
    contract = _read_json(stage / "stage_contract.json")
    if (
        contract.get("profile") != profile
        or int(contract.get("samples", -1)) != len(rows)
        or contract.get("phase_sha256") != marker.get("phase_sha256")
    ):
        raise RuntimeError("stage_contract.json does not match the session")
    return {
        "stage_dir": str(stage),
        "profile": profile,
        "samples": len(rows),
        "mask_name": marker["mask_name"],
        "phase_sha256": marker["phase_sha256"],
        "suitable_for_accuracy_reporting": bool(
            contract["suitable_for_accuracy_reporting"]
        ),
        "hash_and_geometry_validation": "passed",
    }


def prepare_session(
    *,
    profile: str,
    mask_name: str,
    output_dir: str | Path,
    bundle_root: str | Path | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    if profile not in PROFILES:
        raise ValueError(f"profile must be one of {sorted(PROFILES)}")
    root = (
        Path(bundle_root).expanduser().resolve()
        if bundle_root is not None
        else _default_bundle_root()
    )
    payload = root / "payload"
    source_rows = _read_csv(payload / "samples" / f"{profile}.csv")
    _validate_source_rows(source_rows, profile)
    mask, phase_source = _select_mask(payload, mask_name)
    target = Path(output_dir).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    _safe_replace_target(target, overwrite)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{target.name}.building-", dir=target.parent)
    )
    try:
        amplitude_dir = temporary / "amplitude_to_play"
        phase_dir = temporary / "phase_to_play"
        amplitude_dir.mkdir()
        phase_dir.mkdir()
        (temporary / "ccd_captured").mkdir()
        (temporary / "acquisition_logs").mkdir()
        phase_destination = phase_dir / phase_source.name
        shutil.copy2(phase_source, phase_destination)
        if _sha256(phase_destination) != mask["sha256"]:
            raise RuntimeError("Phase mask changed while preparing session")
        _write_csv(
            phase_dir / "reconstruction_manifest.csv",
            [
                {
                    "source_file": phase_source.name,
                    "output_bmp": phase_destination.name,
                    "output_sha256": mask["sha256"],
                    "output_width": PHASE_SIZE[0],
                    "output_height": PHASE_SIZE[1],
                    "output_mode": "L",
                    "transform": "already_native_physical_pitch_nearest",
                }
            ],
        )

        stage_rows: list[dict[str, Any]] = []
        compact_dir = payload / "samples" / "compact_amplitude"
        for index, row in enumerate(source_rows):
            key = row["key"]
            output_name = f"{key}.bmp"
            hashes = _reconstruct_amplitude(
                compact_dir / row["compact_file"],
                amplitude_dir / output_name,
                row,
            )
            stage_rows.append(
                {
                    "key": key,
                    "profile": profile,
                    "selection_policy": row["selection_policy"],
                    "selection_seed": int(row["selection_seed"]),
                    "selection_rank_within_class": int(
                        row["selection_rank_within_class"]
                    ),
                    "dataset_index": int(row["dataset_index"]),
                    "label": int(row["label"]),
                    "amplitude_file": output_name,
                    **hashes,
                    "phase_file": phase_source.name,
                    "phase_sha256": mask["sha256"],
                }
            )
            if (index + 1) % 25 == 0 or index + 1 == len(source_rows):
                print(f"[prepare_session] {index + 1}/{len(source_rows)} amplitudes")
        _write_csv(temporary / "samples.csv", stage_rows)
        suitable = PROFILES[profile][2]
        contract = {
            "schema_version": 3,
            "profile": profile,
            "samples": len(stage_rows),
            "samples_per_class": {
                str(label): PROFILES[profile][1] for label in CLASS_IDS
            },
            "selection_policy": (
                "fixed_random_within_true_class_without_model_filtering"
                if suitable
                else "formal400 fixed-random rank 0..9 diagnostic subset"
            ),
            "selection_seed": 42,
            "suitable_for_accuracy_reporting": suitable,
            "report_metric_name": (
                "accuracy" if suitable else "diagnostic_success_rate"
            ),
            "diagnostic_warning": (
                None
                if suitable
                else "quick40 is for alignment/exposure only; never report as accuracy"
            ),
            "phase_mask_name": mask_name,
            "phase_file": f"phase_to_play/{phase_source.name}",
            "phase_sha256": mask["sha256"],
            "phase_export": {
                "logical_shape_hw": [478, 478],
                "logical_pixel_pitch_um": 17.0,
                "native_pixel_pitch_um": 8.0,
                "native_active_bounds_xyxy": list(mask["phase_bounds_xyxy"]),
                "rasterization": "physical_pitch_nearest",
                "encoding": "floor(mod(phase,2pi)/(2pi)*256)",
                "flip_vertical_before_rasterization": True,
                "flip_horizontal_before_rasterization": False,
            },
            "amplitude_directory": "amplitude_to_play",
            "capture_directory": "ccd_captured",
            "manifest": "samples.csv",
            "ccd_readout": "argmax of four untouched raw 59x59 ROI sums",
            "ccd_shape_hw": [478, 478],
            "detector_bounds_xyxy": [
                [162, 162, 221, 221],
                [257, 162, 316, 221],
                [162, 257, 221, 316],
                [257, 257, 316, 316],
            ],
        }
        _write_json(temporary / "stage_contract.json", contract)
        marker = {
            "schema_version": 1,
            "owner": PROJECT_PACKAGE,
            "profile": profile,
            "mask_name": mask_name,
            "phase_file": phase_source.name,
            "phase_sha256": mask["sha256"],
            "sample_count": len(stage_rows),
            "bundle_manifest_sha256": (
                _sha256(root / "bundle_manifest.json")
                if (root / "bundle_manifest.json").is_file()
                else None
            ),
        }
        _write_json(temporary / MARKER_NAME, marker)
        (temporary / "SESSION_README.md").write_text(
            f"# MNIST-4 {profile} / {mask_name}\n\n"
            "This directory was reconstructed 1:1 from the shared compact payload. "
            "Do not alter amplitude BMPs, the phase BMP, or samples.csv after "
            "validation. CCD frames and acquisition logs will be written here.\n",
            encoding="utf-8",
        )
        temporary.replace(target)
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    return validate_session(target)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=sorted(PROFILES), required=True)
    parser.add_argument("--mask", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bundle-root", default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)
    report = (
        validate_session(args.output_dir)
        if args.validate_only
        else prepare_session(
            profile=args.profile,
            mask_name=args.mask,
            output_dir=args.output_dir,
            bundle_root=args.bundle_root,
            overwrite=args.overwrite,
        )
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
