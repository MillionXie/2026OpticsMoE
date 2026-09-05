"""Build and verify the compact MNIST-4 four-mask laboratory bundle.

The server-side 1024x1024 amplitude BMPs are deliberately not copied into the
archive.  Their shared 478x478 active regions are losslessly stored as PNG and
are reconstructed 1:1 on the laboratory computer.  This keeps one fixed set of
400 inputs shared by all four phase masks without carrying roughly 400 MiB of
zero padding.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import zipfile
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from PIL import Image


PROJECT_PACKAGE = "d2nn_mnist4_single_layer_17um_10cm_v2"
FORMAL_PROFILE_SOURCE = "formal_fixed_random_100_per_class"
FORMAL_PROFILE = "formal400"
QUICK_PROFILE = "quick40"
FORMAL_COUNT = 400
QUICK_COUNT = 40
FORMAL_PER_CLASS = 100
QUICK_PER_CLASS = 10
CLASS_IDS = (0, 1, 2, 3)
ACTIVE_BOUNDS = (273, 273, 751, 751)
ACTIVE_SIZE = (478, 478)
AMPLITUDE_SLM_SIZE = (1024, 1024)
PHASE_SLM_SIZE = (1920, 1200)
PHASE_CENTER_XY = (980.0, 590.0)
DETECTOR_BOUNDS = (
    (162, 162, 221, 221),
    (257, 162, 316, 221),
    (162, 257, 221, 316),
    (257, 257, 316, 316),
)
EXPECTED_MASK_NAMES = (
    "pre_robust_best",
    "early_robust",
    "post_robust_best",
    "mid_robust_energy",
)
RECOMMENDED_MASK_ORDER = (
    "post_robust_best",
    "mid_robust_energy",
    "pre_robust_best",
    "early_robust",
)
DEFAULT_ZIP_NAME = "mnist4_angle_roi_four_mask_lab_bundle.zip"


@dataclass(frozen=True)
class BundleEntry:
    archive_path: str
    category: str
    source: Path | None = None
    data: bytes | None = None

    def read_bytes(self) -> bytes:
        if (self.source is None) == (self.data is None):
            raise RuntimeError("BundleEntry requires exactly one of source/data")
        return self.data if self.data is not None else self.source.read_bytes()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_file(path: Path, label: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"Required {label} is missing: {path}")
    return path


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(_require_file(path, "JSON file").read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected a JSON object in {path}")
    return value


def _read_csv(path: Path) -> list[dict[str, str]]:
    with _require_file(path, "CSV file").open(
        "r", encoding="utf-8-sig", newline=""
    ) as handle:
        return list(csv.DictReader(handle))


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n"
    ).encode("utf-8")


def _csv_bytes(rows: list[dict[str, Any]]) -> bytes:
    if not rows:
        raise RuntimeError("Cannot serialize an empty CSV")
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=list(rows[0]), extrasaction="raise")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue().encode("utf-8")


def _validate_image(path: Path, *, size: tuple[int, int], label: str) -> None:
    with Image.open(_require_file(path, label)) as image:
        image.load()
        if image.format != "BMP" or image.mode != "L" or image.size != size:
            raise RuntimeError(
                f"{label} must be native L-mode BMP {size[0]}x{size[1]}; "
                f"got format={image.format}, mode={image.mode}, size={image.size}: {path}"
            )


def _validate_export_contract(export_dir: Path) -> dict[str, Any]:
    contract = _read_json(export_dir / "hardware_contract.json")
    if contract.get("wavelength_nm") != 532.0:
        raise RuntimeError("Hardware export wavelength must be 532 nm")
    if abs(float(contract.get("phase_to_ccd_distance_cm", -1.0)) - 10.0) > 1e-9:
        raise RuntimeError("Hardware export must use 10 cm phase-to-CCD propagation")
    geometry = contract.get("logical_geometry", {})
    if geometry.get("active_ccd_size") != 478 or geometry.get(
        "detector_bounds_xyxy"
    ) != [list(value) for value in DETECTOR_BOUNDS]:
        raise RuntimeError("Hardware export detector geometry is not the angle-ROI v2")
    amplitude = contract.get("amplitude_slm", {})
    if (
        amplitude.get("size_wh") != list(AMPLITUDE_SLM_SIZE)
        or amplitude.get("active_bounds_xyxy") != list(ACTIVE_BOUNDS)
        or amplitude.get("invert_before_export") is not False
        or amplitude.get("bright_value_uint8") != 255
        or amplitude.get("dark_value_uint8") != 0
    ):
        raise RuntimeError("Amplitude export is not the corrected 255=bright contract")
    if contract.get("ccd_postprocess") != "none: raw region sums only":
        raise RuntimeError("CCD contract must remain raw four-region sums")
    if contract.get("background_subtraction") is not False:
        raise RuntimeError("An unmeasured background subtraction is forbidden")
    return contract


def _corrected_lab_model_config(export_dir: Path) -> bytes:
    """Pin the angle-preserving detector mapping in the standalone config.

    Early v2 exports recorded the correct detector_regions.csv/hardware contract
    but omitted three v2 mapping fields from lab_model_config.yaml, causing the
    standalone loader to fall back to the obsolete proportional ROIs.  The
    bundle refuses that ambiguity and writes the formal angle geometry.
    """

    source_path = _require_file(
        export_dir / "lab_model_config.yaml", "standalone model config"
    )
    try:
        value = json.loads(source_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise RuntimeError(
            "The exported lab_model_config.yaml must be the expected JSON-compatible YAML"
        ) from error
    if not isinstance(value, dict) or not isinstance(value.get("detector"), dict):
        raise RuntimeError("Standalone model config has no detector mapping")
    detector = value["detector"]
    detector.update(
        {
            "mapping_mode": "preserve_reference_angle",
            "reference_pixel_pitch_um": 16.0,
            "reference_distance_m": 0.20,
        }
    )
    detector["resolved_bounds_xyxy"] = [list(item) for item in DETECTOR_BOUNDS]
    return _json_bytes(value)


def _validate_masks(mask_dir: Path) -> tuple[list[BundleEntry], dict[str, Any]]:
    source = _read_json(mask_dir / "mask_candidates.json")
    required = {
        "logical_phase_shape": [478, 478],
        "phase_slm_size_wh": list(PHASE_SLM_SIZE),
        "phase_slm_center_xy": list(PHASE_CENTER_XY),
        "phase_flip_vertical": True,
        "phase_flip_horizontal": False,
        "detector_bounds_xyxy": [list(value) for value in DETECTOR_BOUNDS],
    }
    for key, expected in required.items():
        if source.get(key) != expected:
            raise RuntimeError(
                f"mask_candidates.json {key!r} must be {expected!r}, "
                f"got {source.get(key)!r}"
            )
    candidates = source.get("candidates")
    if not isinstance(candidates, list):
        raise RuntimeError("mask_candidates.json has no candidate list")
    by_name = {str(row.get("name")): row for row in candidates}
    if set(by_name) != set(EXPECTED_MASK_NAMES):
        raise RuntimeError(
            f"Expected masks {EXPECTED_MASK_NAMES}, got {sorted(by_name)}"
        )

    entries: list[BundleEntry] = []
    sanitized: list[dict[str, Any]] = []
    observed_hashes: set[str] = set()
    for name in EXPECTED_MASK_NAMES:
        row = by_name[name]
        filename = str(row.get("file", ""))
        if not filename or Path(filename).name != filename:
            raise RuntimeError(f"Unsafe phase filename for {name}: {filename!r}")
        bmp = _require_file(mask_dir / filename, f"phase mask {name}")
        _validate_image(bmp, size=PHASE_SLM_SIZE, label=f"phase mask {name}")
        digest = _sha256(bmp)
        if digest != str(row.get("sha256", "")):
            raise RuntimeError(f"Phase mask SHA-256 mismatch for {name}")
        if digest in observed_hashes:
            raise RuntimeError("The four formal phase masks must be byte-distinct")
        observed_hashes.add(digest)
        epoch = int(row.get("epoch", -1))
        accuracy = float(row.get("validation_accuracy", float("nan")))
        loss = float(row.get("validation_loss", float("nan")))
        phase_std = float(row.get("phase_std_rad", float("nan")))
        if epoch <= 0 or not all(math.isfinite(v) for v in (accuracy, loss, phase_std)):
            raise RuntimeError(f"Candidate metrics are invalid for {name}")
        if not 0.0 <= accuracy <= 1.0 or phase_std <= 0.0:
            raise RuntimeError(f"Candidate metrics are out of range for {name}")
        checkpoint_digest = str(row.get("checkpoint_sha256", "")).lower()
        if len(checkpoint_digest) != 64:
            raise RuntimeError(f"Candidate checkpoint hash is invalid for {name}")
        sanitized.append(
            {
                "name": name,
                "file": filename,
                "sha256": digest,
                "source_checkpoint_sha256": checkpoint_digest,
                "epoch": epoch,
                "validation_accuracy": accuracy,
                "validation_loss": loss,
                "phase_std_rad": phase_std,
                "phase_bounds_xyxy": row.get("phase_bounds_xyxy"),
                "actual_phase_center_xy": row.get("actual_phase_center_xy"),
                "metric_scope": "candidate checkpoint validation split only",
                "hardware_accuracy": None,
            }
        )
        entries.append(
            BundleEntry(
                f"payload/phase_masks/{filename}", "phase_mask", source=bmp
            )
        )
    evidence = {
        "schema_version": 2,
        "recommended_trial_order": list(RECOMMENDED_MASK_ORDER),
        "shared_input_policy": (
            "Every mask must be evaluated on the identical quick40 or formal400 keys"
        ),
        "validation_metric_warning": (
            "The listed validation metrics belong to each candidate checkpoint; "
            "they are not measured hardware accuracy."
        ),
        "logical_phase_shape": [478, 478],
        "phase_slm_size_wh": list(PHASE_SLM_SIZE),
        "phase_slm_center_xy": list(PHASE_CENTER_XY),
        "phase_flip_vertical": True,
        "phase_flip_horizontal": False,
        "detector_bounds_xyxy": [list(value) for value in DETECTOR_BOUNDS],
        "candidates": sanitized,
    }
    entries.append(
        BundleEntry(
            "payload/phase_masks/phase_masks.json",
            "phase_evidence",
            data=_json_bytes(evidence),
        )
    )
    return entries, evidence


def _compact_amplitudes(
    export_dir: Path,
) -> tuple[list[BundleEntry], list[dict[str, Any]], list[dict[str, Any]]]:
    stage = export_dir / FORMAL_PROFILE_SOURCE
    source_rows = _read_csv(stage / "samples.csv")
    required_fields = {
        "key",
        "label",
        "dataset_index",
        "selection_seed",
        "selection_rank_within_class",
        "selection_policy",
        "amplitude_file",
        "amplitude_sha256",
    }
    if len(source_rows) != FORMAL_COUNT or not source_rows or not required_fields.issubset(
        source_rows[0]
    ):
        raise RuntimeError("Source formal manifest must contain 400 fixed-random rows")
    counts: Counter[int] = Counter()
    ranks: dict[int, set[int]] = {label: set() for label in CLASS_IDS}
    keys: set[str] = set()
    formal_rows: list[dict[str, Any]] = []
    entries: list[BundleEntry] = []
    amplitude_dir = stage / "amplitude_to_play"
    for formal_order, row in enumerate(source_rows):
        key = str(row["key"])
        if not key or Path(key).name != key or key in keys:
            raise RuntimeError(f"Invalid or duplicate source key {key!r}")
        keys.add(key)
        label = int(row["label"])
        rank = int(row["selection_rank_within_class"])
        if label not in CLASS_IDS or rank in ranks[label]:
            raise RuntimeError(f"Invalid class/rank for {key}")
        counts[label] += 1
        ranks[label].add(rank)
        if row["selection_policy"] != "fixed_random_within_true_class_without_model_filtering":
            raise RuntimeError("Formal inputs must be fixed-random without model filtering")
        if int(row["selection_seed"]) != 42:
            raise RuntimeError("Formal input selection seed must remain 42")
        filename = str(row["amplitude_file"])
        if filename != f"{key}.bmp":
            raise RuntimeError(f"Amplitude filename/key mismatch for {key}")
        source_bmp = _require_file(amplitude_dir / filename, "formal amplitude BMP")
        _validate_image(source_bmp, size=AMPLITUDE_SLM_SIZE, label="formal amplitude")
        source_sha = _sha256(source_bmp)
        if source_sha != row["amplitude_sha256"]:
            raise RuntimeError(f"Source amplitude SHA-256 mismatch for {key}")
        with Image.open(source_bmp) as opened:
            image = opened.copy()
        left, top, right, bottom = ACTIVE_BOUNDS
        outside_boxes = (
            (0, 0, AMPLITUDE_SLM_SIZE[0], top),
            (0, bottom, AMPLITUDE_SLM_SIZE[0], AMPLITUDE_SLM_SIZE[1]),
            (0, top, left, bottom),
            (right, top, AMPLITUDE_SLM_SIZE[0], bottom),
        )
        if any(image.crop(box).getbbox() is not None for box in outside_boxes):
            raise RuntimeError(f"Nonzero pixels outside active 478x478 area for {key}")
        compact = image.crop(ACTIVE_BOUNDS)
        if compact.size != ACTIVE_SIZE or compact.mode != "L":
            raise RuntimeError(f"Compact amplitude geometry changed for {key}")
        buffer = io.BytesIO()
        compact.save(buffer, format="PNG", optimize=True)
        compact_bytes = buffer.getvalue()
        compact_sha = _sha256_bytes(compact_bytes)
        compact_name = f"{key}.png"
        entries.append(
            BundleEntry(
                f"payload/samples/compact_amplitude/{compact_name}",
                "compact_amplitude",
                data=compact_bytes,
            )
        )
        formal_rows.append(
            {
                "formal_order": formal_order,
                "key": key,
                "label": label,
                "dataset_index": int(row["dataset_index"]),
                "selection_seed": 42,
                "selection_rank_within_class": rank,
                "selection_policy": row["selection_policy"],
                "compact_file": compact_name,
                "compact_sha256": compact_sha,
                "compact_pixel_sha256": _sha256_bytes(compact.tobytes()),
                "source_full_bmp_sha256": source_sha,
                "source_full_pixel_sha256": _sha256_bytes(image.tobytes()),
            }
        )
    if counts != Counter({label: FORMAL_PER_CLASS for label in CLASS_IDS}):
        raise RuntimeError(f"Formal class counts are wrong: {dict(counts)}")
    if any(ranks[label] != set(range(FORMAL_PER_CLASS)) for label in CLASS_IDS):
        raise RuntimeError("Formal selection ranks must be exactly 0..99 per class")
    disk_files = {path.name for path in amplitude_dir.glob("*.bmp")}
    expected_files = {f"{key}.bmp" for key in keys}
    if disk_files != expected_files:
        raise RuntimeError("Formal source amplitude directory has undeclared/missing BMPs")

    quick_rows = [
        {**row, "quick_order": index}
        for index, row in enumerate(
            row
            for row in formal_rows
            if int(row["selection_rank_within_class"]) < QUICK_PER_CLASS
        )
    ]
    if len(quick_rows) != QUICK_COUNT or Counter(
        int(row["label"]) for row in quick_rows
    ) != Counter({label: QUICK_PER_CLASS for label in CLASS_IDS}):
        raise RuntimeError("quick40 must be the first 10 fixed-random ranks per class")
    entries.extend(
        (
            BundleEntry(
                "payload/samples/formal400.csv",
                "sample_manifest",
                data=_csv_bytes(formal_rows),
            ),
            BundleEntry(
                "payload/samples/quick40.csv",
                "sample_manifest",
                data=_csv_bytes(quick_rows),
            ),
        )
    )
    return entries, formal_rows, quick_rows


def _runtime_files(repo_root: Path, include_vendor_sdk: bool) -> Iterable[BundleEntry]:
    project = f"experiments/{PROJECT_PACKAGE}"
    relative_files = (
        "experiments/__init__.py",
        f"{project}/__init__.py",
        f"{project}/ccd_evaluate.py",
        f"{project}/base_settings.py",
        f"{project}/io_utils.py",
        f"{project}/settings.py",
        f"{project}/lab_session.py",
        f"{project}/lab_pipeline.py",
        f"{project}/paper_evaluation.py",
        f"{project}/lab_hardware_config.yaml",
        f"{project}/requirements-lab.txt",
        "experiments/hardware_sdk/__init__.py",
        "experiments/hardware_sdk/devices.py",
        "experiments/hardware_sdk/workflows/__init__.py",
        "experiments/hardware_sdk/workflows/acquire_folder.py",
        "experiments/hardware_sdk/workflows/calibration_common.py",
        "experiments/hardware_sdk/drivers/__init__.py",
        "experiments/hardware_sdk/drivers/meadowlark_pcie_slm.py",
        "experiments/hardware_sdk/drivers/tucam_camera.py",
    )
    for relative in relative_files:
        path = _require_file(repo_root / relative, "laboratory runtime file")
        yield BundleEntry(relative, "lab_runtime", source=path)
    if not include_vendor_sdk:
        return
    for relative in (
        "experiments/hardware_sdk/vendor_sdk/amplitude_meadowlark",
        "experiments/hardware_sdk/vendor_sdk/camera_tucam_mosaic",
    ):
        directory = repo_root / relative
        if not directory.is_dir():
            raise FileNotFoundError(f"Required vendor SDK directory is missing: {directory}")
        for path in sorted(directory.rglob("*")):
            if not path.is_file() or "__pycache__" in path.parts:
                continue
            if path.suffix.lower() in {".pyc", ".pyo"}:
                continue
            yield BundleEntry(
                path.relative_to(repo_root).as_posix(), "vendor_sdk", source=path
            )


def _reference_files(repo_root: Path) -> Iterable[BundleEntry]:
    project_root = repo_root / "experiments" / PROJECT_PACKAGE
    names = (
        "LAB_BUNDLE.md",
        "RUN_COMMANDS.md",
        "README.md",
        "CORRECTED_TRAINING.md",
        "NOTEBOOK_AUDIT.md",
        "lab_package.py",
        "configs/release/mnist4_single_layer_17um_10cm_v2_notebook_mse_angle_roi.yaml",
    )
    for name in names:
        path = _require_file(project_root / name, "reference file")
        yield BundleEntry(
            (Path("reference") / "mnist4_project" / name).as_posix(),
            "reference",
            source=path,
        )


def _deduplicate(entries: Iterable[BundleEntry]) -> list[BundleEntry]:
    result: list[BundleEntry] = []
    names: set[str] = set()
    for entry in entries:
        name = Path(entry.archive_path).as_posix().lstrip("/")
        if not name or name.startswith("../") or "/../" in f"/{name}/":
            raise RuntimeError(f"Unsafe archive path {entry.archive_path!r}")
        if name in names:
            raise RuntimeError(f"Duplicate archive path {name}")
        names.add(name)
        result.append(
            BundleEntry(name, entry.category, source=entry.source, data=entry.data)
        )
    return result


def _verify_zip(path: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    expected = {row["archive_path"]: row for row in manifest["archive_files"]}
    expected["bundle_manifest.json"] = None
    with zipfile.ZipFile(path, "r") as archive:
        names = archive.namelist()
        if len(names) != len(set(names)) or set(names) != set(expected):
            raise RuntimeError("ZIP entry set does not exactly match bundle manifest")
        if archive.testzip() is not None:
            raise RuntimeError("ZIP CRC validation failed")
        embedded = json.loads(archive.read("bundle_manifest.json"))
        if embedded != manifest:
            raise RuntimeError("Embedded bundle manifest changed during serialization")
        for name, row in expected.items():
            if row is None:
                continue
            data = archive.read(name)
            if len(data) != row["size_bytes"] or _sha256_bytes(data) != row["sha256"]:
                raise RuntimeError(f"ZIP content hash mismatch: {name}")
    names_lower = {f"/{name.lower()}" for name in expected}
    forbidden = ("/checkpoints/", "/ccd_captured/", "/amplitude_to_play/", "/cache/")
    if any(fragment in name for name in names_lower for fragment in forbidden):
        raise RuntimeError("Bundle contains a forbidden large/generated directory")
    tensor_entries = [
        name
        for name in expected
        if Path(name).suffix.lower() in {".pt", ".pth", ".ckpt", ".safetensors"}
    ]
    if tensor_entries:
        raise RuntimeError(f"Laboratory bundle must contain no checkpoints: {tensor_entries}")
    compact = [
        name
        for name in expected
        if name.startswith("payload/samples/compact_amplitude/")
        and name.endswith(".png")
    ]
    phase = [
        name
        for name in expected
        if name.startswith("payload/phase_masks/") and name.endswith(".bmp")
    ]
    if len(compact) != FORMAL_COUNT or len(phase) != len(EXPECTED_MASK_NAMES):
        raise RuntimeError("Bundle compact-amplitude or phase-mask count is wrong")
    return {
        "entry_count": len(expected),
        "crc_and_sha256_validation": "passed",
        "compact_amplitude_count": len(compact),
        "phase_mask_count": len(phase),
    }


def create_lab_bundle(
    *,
    export_dir: str | Path,
    mask_dir: str | Path,
    output_path: str | Path,
    include_vendor_sdk: bool = True,
    overwrite: bool = False,
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    root = (
        Path(repo_root).expanduser().resolve()
        if repo_root is not None
        else Path(__file__).resolve().parents[2]
    )
    export_root = Path(export_dir).expanduser().resolve()
    mask_root = Path(mask_dir).expanduser().resolve()
    export_contract = _validate_export_contract(export_root)
    mask_entries, mask_evidence = _validate_masks(mask_root)
    sample_entries, formal_rows, quick_rows = _compact_amplitudes(export_root)
    model_files = ("detector_regions.csv", "detector_roi_478.png")
    entries: list[BundleEntry] = [
        BundleEntry("README_FIRST.md", "documentation", source=_require_file(
            root / "experiments" / PROJECT_PACKAGE / "LAB_BUNDLE.md",
            "laboratory README",
        )),
        *mask_entries,
        *sample_entries,
        BundleEntry(
            "payload/model/lab_model_config.yaml",
            "model_geometry",
            data=_corrected_lab_model_config(export_root),
        ),
    ]
    entries.extend(
        BundleEntry(
            f"payload/model/{name}",
            "model_geometry",
            source=_require_file(export_root / name, f"model payload {name}"),
        )
        for name in model_files
    )
    entries.extend(_runtime_files(root, include_vendor_sdk))
    entries.extend(_reference_files(root))
    entries = _deduplicate(entries)

    rows: list[dict[str, Any]] = []
    entry_data: dict[str, bytes] = {}
    for entry in entries:
        data = entry.read_bytes()
        entry_data[entry.archive_path] = data
        rows.append(
            {
                "archive_path": entry.archive_path,
                "category": entry.category,
                "size_bytes": len(data),
                "sha256": _sha256_bytes(data),
            }
        )
    category_counts = dict(sorted(Counter(row["category"] for row in rows).items()))
    manifest: dict[str, Any] = {
        "schema_version": 3,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "project": PROJECT_PACKAGE,
        "hardware_contract": {
            "wavelength_nm": 532.0,
            "phase_to_ccd_distance_cm": 10.0,
            "logical_active_shape_hw": [478, 478],
            "amplitude_slm": {
                "size_wh": list(AMPLITUDE_SLM_SIZE),
                "pixel_pitch_um": 17.0,
                "active_bounds_xyxy": list(ACTIVE_BOUNDS),
                "reconstruction": "lossless 1:1 center paste; no resize",
                "polarity": "255=bright/transmissive, 0=dark/blocking",
            },
            "phase_slm": {
                "size_wh": list(PHASE_SLM_SIZE),
                "pixel_pitch_um": 8.0,
                "center_xy": list(PHASE_CENTER_XY),
                "flip_vertical_before_export": True,
                "flip_horizontal_before_export": False,
            },
            "detector_bounds_xyxy": [list(value) for value in DETECTOR_BOUNDS],
            "ccd_contract": "478x478 uint8 grayscale; raw ROI sums only",
            "background_subtraction": False,
            "normalization": False,
            "nonlinearity": False,
        },
        "shared_samples": {
            "formal400": {
                "count": len(formal_rows),
                "per_class": FORMAL_PER_CLASS,
                "selection_seed": 42,
                "suitable_for_accuracy_reporting": True,
            },
            "quick40": {
                "count": len(quick_rows),
                "per_class": QUICK_PER_CLASS,
                "derivation": "formal400 selection_rank_within_class 0..9",
                "suitable_for_accuracy_reporting": False,
                "purpose": "alignment and exposure diagnostic only",
            },
            "same_keys_for_every_phase_mask": True,
        },
        "phase_masks": {
            "count": len(EXPECTED_MASK_NAMES),
            "names": list(EXPECTED_MASK_NAMES),
            "recommended_trial_order": list(RECOMMENDED_MASK_ORDER),
            "candidate_evidence": mask_evidence,
        },
        "paper_evaluation": {
            "module": f"experiments.{PROJECT_PACKAGE}.paper_evaluation",
            "formal_comparison_requires": "formal400, identical 400 keys per mask",
            "quick40_excluded_from_formal_comparison": True,
            "figure_style": "Arial 7 pt, 5 cm high, PDF/SVG/600-dpi PNG",
        },
        "source_export_contract_sha256": _sha256(
            export_root / "hardware_contract.json"
        ),
        "include_vendor_sdk": bool(include_vendor_sdk),
        "exclusion_contract": [
            "no training checkpoint",
            "no MNIST dataset/cache",
            "no CCD captures",
            "no padded 1024x1024 server amplitude BMPs",
            "no simulation prediction/energy columns from unrelated best.pt",
        ],
        "category_file_counts": category_counts,
        "archive_files": rows,
    }
    output = Path(output_path).expanduser().resolve()
    if output.exists() and not overwrite:
        raise FileExistsError(f"Bundle exists; pass --overwrite to replace it: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest_bytes = _json_bytes(manifest)
    with zipfile.ZipFile(
        output,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=6,
        allowZip64=True,
    ) as archive:
        for entry in entries:
            archive.writestr(entry.archive_path, entry_data[entry.archive_path])
        archive.writestr("bundle_manifest.json", manifest_bytes)
    verification = _verify_zip(output, manifest)
    report = {
        "zip": str(output),
        "zip_sha256": _sha256(output),
        "zip_size_bytes": output.stat().st_size,
        "formal_samples": FORMAL_COUNT,
        "quick_samples": QUICK_COUNT,
        "phase_masks": len(EXPECTED_MASK_NAMES),
        "vendor_sdk_included": bool(include_vendor_sdk),
        "zip_validation": verification,
    }
    output.with_suffix(output.suffix + ".json").write_bytes(_json_bytes(report))
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-dir", required=True)
    parser.add_argument("--mask-dir", required=True)
    parser.add_argument("--output", default=DEFAULT_ZIP_NAME)
    parser.add_argument("--omit-vendor-sdk", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    report = create_lab_bundle(
        export_dir=args.export_dir,
        mask_dir=args.mask_dir,
        output_path=args.output,
        include_vendor_sdk=not args.omit_vendor_sdk,
        overwrite=args.overwrite,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
