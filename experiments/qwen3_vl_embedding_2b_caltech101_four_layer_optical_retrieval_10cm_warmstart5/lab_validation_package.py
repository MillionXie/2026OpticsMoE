"""Build the independent optical-calibration and sim-to-real laboratory ZIP.

This packager is intentionally separate from :mod:`lab_package`.  The latter
seals the formal warmstart5 result and pins a particular checkpoint, phase
export, and quick210 payload.  This module never reads, validates, copies, or
rewrites those sealed artifacts.  It only packages reusable calibration,
capture, geometry, and simulated-to-measured CCD agreement tools.

The archive preserves the repository-relative ``experiments/...`` layout so it
can be extracted directly into ``E:\\code\\guest\\2026OpticsMoE``.  Every
material file is bound by an embedded SHA-256 manifest and the ZIP itself gets
an adjacent ``.sha256`` file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import zipfile
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping


PROJECT_PACKAGE = (
    "qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_"
    "warmstart5"
)
ARCHIVE_TYPE = "qwen_warmstart5_optical_validation_lab_tools"
DEFAULT_ZIP_NAME = "qwen_warmstart5_optical_validation_lab_tools.zip"
EXPECTED_FRESNEL_V3_MANIFEST_SHA256 = (
    "07914c0d92d4d3f772a423a987c0c71e8d65d3a7d2a6f345ac1f07079230a568"
)
FRESNEL_ARCHIVE_ROOT = (
    "experiments/hardware_sdk/generators/slm_patterns/generated/"
    "fresnel_square_aperture_array_532nm_17um_8um_v3"
)
K1_ARCHIVE_ROOT = (
    "experiments/hardware_sdk/generators/slm_patterns/generated/"
    "dual_slm_k1_ready_to_play"
)
FRESNEL_GENERATE_COMMAND = (
    "python -m experiments.hardware_sdk.generators.fresnel_square_aperture_array "
    "--config experiments/hardware_sdk/generators/slm_patterns/configs/"
    "fresnel_square_aperture_array_17um_8um.yaml"
)
K1_GENERATE_COMMAND = (
    "python -m experiments.hardware_sdk.generators.dual_slm_registration_sweep "
    "--config experiments/hardware_sdk/generators/slm_patterns/configs/"
    "dual_slm_17um_8um_normal_scale_sweep.yaml"
)


@dataclass(frozen=True)
class PackageFile:
    source: Path
    archive_path: str
    category: str


LAB_RUNTIME_FILES = (
    "experiments/__init__.py",
    "experiments/hardware_sdk/__init__.py",
    "experiments/hardware_sdk/devices.py",
    "experiments/hardware_sdk/README.md",
    "experiments/hardware_sdk/RUN_COMMANDS.md",
    "experiments/hardware_sdk/GEOMETRY_AND_BRIGHTNESS.md",
    "experiments/hardware_sdk/LAB_WINDOWS_QUICKSTART.md",
    "experiments/hardware_sdk/requirements-light.txt",
    "experiments/hardware_sdk/configs/tucam_meadowlark_1024_windows.yaml",
    "experiments/hardware_sdk/configs/tucam_meadowlark_calibration_windows.yaml",
    "experiments/hardware_sdk/configs/tucam_windows.yaml",
    "experiments/hardware_sdk/configs/detector_homography_478.example.yaml",
    "experiments/hardware_sdk/configs/phase_slm_demo.yaml",
    "experiments/hardware_sdk/drivers/__init__.py",
    "experiments/hardware_sdk/drivers/meadowlark_pcie_slm.py",
    "experiments/hardware_sdk/drivers/tucam_camera.py",
    "experiments/hardware_sdk/drivers/README.md",
    "experiments/hardware_sdk/workflows/__init__.py",
    "experiments/hardware_sdk/workflows/acquire_folder.py",
    "experiments/hardware_sdk/workflows/calibration_common.py",
    "experiments/hardware_sdk/workflows/detector_homography.py",
    "experiments/hardware_sdk/workflows/optional_background.py",
    "experiments/hardware_sdk/workflows/reconstruct_slm.py",
    "experiments/hardware_sdk/workflows/roi_calibration.py",
    "experiments/hardware_sdk/demos/__init__.py",
    "experiments/hardware_sdk/demos/amplitude_camera_demo.py",
    "experiments/hardware_sdk/demos/phase_slm_demo.py",
    "experiments/hardware_sdk/tools/__init__.py",
    "experiments/hardware_sdk/tools/camera_smoke_test.py",
    "experiments/hardware_sdk/generators/__init__.py",
    "experiments/hardware_sdk/generators/dual_slm_alignment.py",
    "experiments/hardware_sdk/generators/dual_slm_registration_sweep.py",
    "experiments/hardware_sdk/generators/fresnel_square_aperture_array.py",
    "experiments/hardware_sdk/generators/slm_patterns/__init__.py",
    "experiments/hardware_sdk/generators/slm_patterns/settings.py",
    "experiments/hardware_sdk/generators/slm_patterns/generate.py",
    "experiments/hardware_sdk/generators/slm_patterns/README.md",
    "experiments/hardware_sdk/generators/slm_patterns/V3_CALIBRATION_COMMANDS.md",
    "experiments/hardware_sdk/generators/slm_patterns/configs/"
    "dual_slm_17um_8um_normal_scale_sweep.yaml",
    "experiments/hardware_sdk/generators/slm_patterns/configs/"
    "fresnel_square_aperture_array_17um_8um.yaml",
)

PROJECT_RUNTIME_FILES = tuple(
    f"experiments/{PROJECT_PACKAGE}/{name}"
    for name in (
        "__init__.py",
        "agreement_common.py",
        "agreement_export.py",
        "agreement_evaluate.py",
        "agreement_report.py",
        "settings.py",
        "modeling.py",
        "SIM_TO_REAL_AGREEMENT.md",
        "requirements-lab.txt",
        "configs/release/agreement_quick_language_global.yaml",
        "configs/release/stage2_joint_sealed_test.yaml",
        "configs/release/stage2_joint_hardware_canonical_ccd.yaml",
        "configs/release/quick_last_stage_10x10.yaml",
        "configs/release/quick_last_stage_10x10_canonical_ccd.yaml",
        "LAB_VALIDATION_BUNDLE.md",
        "lab_validation_package.py",
    )
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _require_file(path: Path, label: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"Required {label} is missing: {path}")
    return path


def _read_json(path: Path, label: str) -> dict[str, Any]:
    value = json.loads(_require_file(path, label).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} must contain a JSON object: {path}")
    return value


def _safe_archive_path(value: str) -> str:
    path = PurePosixPath(str(value).replace("\\", "/"))
    if (
        not path.parts
        or path.is_absolute()
        or ".." in path.parts
        or any(part in {"", "."} for part in path.parts)
    ):
        raise ValueError(f"Unsafe archive path: {value!r}")
    return path.as_posix()


def _safe_relative(root: Path, value: str, *, label: str) -> Path:
    raw = Path(str(value))
    if raw.is_absolute() or not raw.parts or ".." in raw.parts:
        raise RuntimeError(f"{label} must be a safe relative path: {value!r}")
    resolved_root = root.resolve()
    resolved = (resolved_root / raw).resolve()
    try:
        resolved.relative_to(resolved_root)
    except ValueError as error:
        raise RuntimeError(f"{label} escapes its generated directory: {value!r}") from error
    return resolved


def _validate_u8_image(path: Path, size_wh: tuple[int, int], label: str) -> None:
    try:
        from PIL import Image
    except ImportError as error:  # pragma: no cover - packaging environment guard
        raise RuntimeError("Pillow is required to validate laboratory BMP/PNG files") from error
    with Image.open(_require_file(path, label)) as image:
        image.load()
        if image.size != size_wh:
            raise RuntimeError(
                f"{label} must be {size_wh[0]}x{size_wh[1]}, got {image.size}: {path}"
            )
        if image.mode not in {"L", "P"}:
            raise RuntimeError(f"{label} must be 8-bit grayscale/indexed, got {image.mode}")


def _verify_declared_file(
    root: Path,
    record: Mapping[str, Any],
    *,
    label: str,
    expected_size_wh: tuple[int, int] | None = None,
) -> Path:
    path = _safe_relative(root, str(record.get("path", "")), label=f"{label} path")
    _require_file(path, label)
    observed = sha256_file(path)
    expected = str(record.get("sha256", "")).strip().lower()
    if observed != expected:
        raise RuntimeError(
            f"{label} SHA-256 mismatch: expected={expected!r}, observed={observed}"
        )
    if expected_size_wh is not None:
        _validate_u8_image(path, expected_size_wh, label)
    return path


def validate_fresnel_v3(directory: str | Path) -> dict[str, Any]:
    root = Path(directory).expanduser().resolve()
    manifest_path = root / "fresnel_square_aperture_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            "Fresnel v3 generated payload is missing. Run from the repository root:\n"
            f"  {FRESNEL_GENERATE_COMMAND}\n"
            f"Expected manifest: {manifest_path}"
        )
    manifest = _read_json(manifest_path, "Fresnel v3 manifest")
    manifest_sha256 = sha256_file(manifest_path)
    if manifest_sha256 != EXPECTED_FRESNEL_V3_MANIFEST_SHA256:
        raise RuntimeError(
            "Fresnel v3 manifest is stale or unreviewed: expected the center-preserving "
            f"release {EXPECTED_FRESNEL_V3_MANIFEST_SHA256}, got {manifest_sha256}"
        )
    required_scalar = {
        "schema_version": 3,
        "wavelength_nm": 532.0,
        "propagation_cm": 10.0,
    }
    for key, expected in required_scalar.items():
        if manifest.get(key) != expected:
            raise RuntimeError(
                f"Fresnel v3 {key} must be {expected!r}, got {manifest.get(key)!r}"
            )
    amplitude = manifest.get("amplitude_polarity", {})
    if (
        amplitude.get("open_uint8") != 255
        or amplitude.get("closed_uint8") != 0
        or amplitude.get("all_zero_forbidden") is not True
    ):
        raise RuntimeError("Fresnel v3 must declare 255=open, 0=closed, and forbid all-zero input")
    amplitude_slm = manifest.get("amplitude_slm", {})
    phase_slm = manifest.get("phase_slm", {})
    geometry = manifest.get("roi_geometry", {})
    if amplitude_slm.get("size_wh") != [1024, 1024] or amplitude_slm.get(
        "pixel_pitch_um"
    ) != 17.0:
        raise RuntimeError("Fresnel v3 amplitude SLM contract must be 1024x1024 at 17 um")
    if (
        phase_slm.get("size_wh") != [1920, 1200]
        or phase_slm.get("pixel_pitch_um") != 8.0
        or phase_slm.get("center_edge_xy") != [980.0, 590.0]
        or phase_slm.get("flip_vertical_on_export") is not True
        or phase_slm.get("flip_horizontal_on_export") is not False
        or phase_slm.get("flip_axis_edge_y") != 590.0
        or phase_slm.get("configured_center_preserved_in_exported_bmp") is not True
    ):
        raise RuntimeError("Fresnel v3 phase SLM geometry/orientation contract is wrong")
    if geometry.get("amplitude_active_size_px") != 478:
        raise RuntimeError("Fresnel v3 logical optical support must be 478 pixels")

    pairs = manifest.get("pairs")
    if not isinstance(pairs, list) or len(pairs) != 12:
        raise RuntimeError("Fresnel v3 must contain 12 n1/n4/n9 x aperture-width pairs")
    combinations: Counter[tuple[int, int]] = Counter()
    declared_paths: set[Path] = {manifest_path}
    for pair in pairs:
        if not isinstance(pair, dict):
            raise RuntimeError("Fresnel v3 pair entries must be JSON objects")
        array_count = int(pair.get("array_count", -1))
        aperture = int(pair.get("requested_aperture_width_phase_px", -1))
        combinations[(array_count, aperture)] += 1
        if pair.get("numerical_validation", {}).get("passed") is not True:
            raise RuntimeError(f"Fresnel numerical validation failed for {pair.get('pair_id')}")
        lenslets = pair.get("lenslets", [])
        if len(lenslets) != array_count or any(
            lenslet.get("aperture_kind") != "full_independent_square"
            or lenslet.get("full_aperture_not_clipped") is not True
            for lenslet in lenslets
        ):
            raise RuntimeError(f"Fresnel pair {pair.get('pair_id')} contains clipped/non-full pupils")
        for lenslet in lenslets:
            before = lenslet.get("target_phase_edge_xy_before_export_flip")
            exported = lenslet.get("target_phase_edge_xy_in_exported_bmp")
            if (
                not isinstance(before, list)
                or not isinstance(exported, list)
                or len(before) != 2
                or len(exported) != 2
                or not math.isclose(float(exported[0]), float(before[0]), abs_tol=1.0e-9)
                or not math.isclose(
                    float(exported[1]),
                    2.0 * float(phase_slm["flip_axis_edge_y"]) - float(before[1]),
                    abs_tol=1.0e-9,
                )
            ):
                raise RuntimeError(
                    f"Fresnel pair {pair.get('pair_id')} does not reflect around configured y=590"
                )
        declared_paths.add(
            _verify_declared_file(
                root,
                pair.get("amplitude_file", {}),
                label=f"Fresnel amplitude {pair.get('pair_id')}",
                expected_size_wh=(1024, 1024),
            )
        )
        declared_paths.add(
            _verify_declared_file(
                root,
                pair.get("phase_file", {}),
                label=f"Fresnel phase {pair.get('pair_id')}",
                expected_size_wh=(1920, 1200),
            )
        )
        for key in ("phase_support_preview", "ideal_ccd_linear", "ideal_ccd_log"):
            declared_paths.add(
                _verify_declared_file(root, pair.get(key, {}), label=f"Fresnel {key}")
            )
    expected_combinations = {
        (array_count, aperture)
        for array_count in (1, 4, 9)
        for aperture in (48, 64, 96, 128)
    }
    if set(combinations) != expected_combinations or any(value != 1 for value in combinations.values()):
        raise RuntimeError("Fresnel v3 does not contain exactly one expected array/aperture combination")
    ideal = manifest.get("ideal_simulation", {})
    for key in ("metrics_json", "metrics_csv"):
        declared_paths.add(
            _verify_declared_file(root, ideal.get(key, {}), label=f"Fresnel {key}")
        )
    _require_file(root / "README.md", "Fresnel v3 README")
    files = sorted(path for path in root.rglob("*") if path.is_file())
    if not files:
        raise RuntimeError("Fresnel v3 directory is empty")
    return {
        "directory": root,
        "manifest": manifest,
        "manifest_sha256": manifest_sha256,
        "files": files,
        "declared_material_files": len(declared_paths),
        "pair_ids": [str(pair["pair_id"]) for pair in pairs],
    }


def validate_k1_suite(directory: str | Path) -> dict[str, Any]:
    root = Path(directory).expanduser().resolve()
    manifest_path = root / "k1_pair_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            "Dual-SLM k=1 ready-to-play suite is missing. Run from the repository root:\n"
            f"  {K1_GENERATE_COMMAND}\n"
            "The required suite will be written below the generated sweep as "
            "00_k1_ready_to_play.\n"
            f"Expected manifest: {manifest_path}"
        )
    manifest = _read_json(manifest_path, "dual-SLM k=1 manifest")
    if manifest.get("schema_version") != 1 or float(manifest.get("scale_k", -1)) != 1.0:
        raise RuntimeError("Dual-SLM ready-to-play suite must be schema 1 at exactly k=1")
    if manifest.get("amplitude_polarity") != "255=white/open, 0=black/closed":
        raise RuntimeError("Dual-SLM k=1 suite has the wrong amplitude polarity")
    pairs = manifest.get("pairs")
    expected = (
        ("checker_c64", "legacy_xy"),
        ("large_blocks_c48_x", "x"),
        ("large_blocks_c48_y", "y"),
    )
    if not isinstance(pairs, list) or len(pairs) != len(expected):
        raise RuntimeError("Dual-SLM k=1 suite must contain checker, irregular-X and irregular-Y")
    for order, (pair, (name, axis)) in enumerate(zip(pairs, expected), start=1):
        if (
            pair.get("order") != order
            or pair.get("name") != name
            or pair.get("grating_axis") != axis
            or float(pair.get("k", -1)) != 1.0
        ):
            raise RuntimeError(f"Dual-SLM k=1 pair {order} has the wrong identity/order")
        amplitude = _safe_relative(root, str(pair.get("amplitude_bmp", "")), label="k1 amplitude")
        phase = _safe_relative(root, str(pair.get("phase_bmp", "")), label="k1 phase")
        preview = _safe_relative(root, str(pair.get("preview_png", "")), label="k1 preview")
        _validate_u8_image(amplitude, (1024, 1024), f"k1 amplitude {name}")
        _validate_u8_image(phase, (1920, 1200), f"k1 phase {name}")
        _require_file(preview, f"k1 preview {name}")
        for path, key in ((amplitude, "amplitude_sha256"), (phase, "phase_sha256")):
            if sha256_file(path) != str(pair.get(key, "")).strip().lower():
                raise RuntimeError(f"Dual-SLM k=1 {name} {key} mismatch")
    _require_file(root / "README.md", "dual-SLM k=1 README")
    files = sorted(path for path in root.rglob("*") if path.is_file())
    return {
        "directory": root,
        "manifest": manifest,
        "manifest_sha256": sha256_file(manifest_path),
        "files": files,
        "pairs": [pair["name"] for pair in pairs],
    }


def _repository_files(repo_root: Path) -> list[PackageFile]:
    result: list[PackageFile] = []
    for relative in (*LAB_RUNTIME_FILES, *PROJECT_RUNTIME_FILES):
        source = _require_file(repo_root / relative, "laboratory validation source")
        result.append(PackageFile(source, _safe_archive_path(relative), "lab_runtime"))
    readme_source = _require_file(
        repo_root
        / "experiments"
        / PROJECT_PACKAGE
        / "LAB_VALIDATION_BUNDLE.md",
        "laboratory validation README",
    )
    result.append(
        PackageFile(
            readme_source,
            "README_SIM_TO_REAL_LAB.md",
            "documentation",
        )
    )
    return result


def _vendor_files(repo_root: Path) -> list[PackageFile]:
    amplitude_root = (
        repo_root
        / "experiments"
        / "hardware_sdk"
        / "vendor_sdk"
        / "amplitude_meadowlark"
    )
    camera_root = (
        repo_root
        / "experiments"
        / "hardware_sdk"
        / "vendor_sdk"
        / "camera_tucam_mosaic"
    )
    required = (
        amplitude_root / "SDK" / "Blink_C_wrapper.dll",
        amplitude_root / "SDK" / "Blink_SDK.dll",
        amplitude_root / "LUT Files" / "slm7930_at532_30C.lut",
        amplitude_root / "LUT Files" / "slm7930_at532_70C.lut",
        camera_root / "TUCam.py",
        camera_root / "lib" / "x64" / "TUCam.dll",
    )
    for path in required:
        _require_file(path, "x64 vendor runtime")
    selected = {
        *amplitude_root.joinpath("SDK").glob("*.dll"),
        *amplitude_root.joinpath("LUT Files").glob("*.lut"),
        camera_root / "TUCam.py",
        *camera_root.joinpath("lib", "x64").glob("*"),
        *camera_root.glob("*.xml"),
    }
    files: list[PackageFile] = []
    for source in sorted(path for path in selected if path.is_file()):
        relative = source.relative_to(repo_root).as_posix()
        lowered = relative.lower()
        if "/x86/" in lowered or lowered.endswith(".pdf"):
            raise RuntimeError(f"Forbidden vendor payload selected: {relative}")
        files.append(PackageFile(source, relative, "vendor_runtime_x64"))
    return files


def _generated_files(
    report: Mapping[str, Any], archive_root: str, category: str
) -> list[PackageFile]:
    root = Path(report["directory"])
    return [
        PackageFile(
            path,
            (PurePosixPath(archive_root) / path.relative_to(root).as_posix()).as_posix(),
            category,
        )
        for path in report["files"]
    ]


def _deduplicate(files: Iterable[PackageFile]) -> list[PackageFile]:
    by_name: dict[str, PackageFile] = {}
    for item in files:
        name = _safe_archive_path(item.archive_path)
        source = item.source.expanduser().resolve()
        _require_file(source, f"archive input {name}")
        if name in by_name and by_name[name].source != source:
            raise RuntimeError(f"Archive path collision for {name}")
        by_name[name] = PackageFile(source, name, item.category)
    return [by_name[name] for name in sorted(by_name)]


def _archive_rows(files: Iterable[PackageFile]) -> list[dict[str, Any]]:
    return [
        {
            "archive_path": item.archive_path,
            "category": item.category,
            "size_bytes": item.source.stat().st_size,
            "sha256": sha256_file(item.source),
        }
        for item in files
    ]


def _checksum_text(rows: Iterable[Mapping[str, Any]]) -> bytes:
    return (
        "".join(f"{row['sha256']}  {row['archive_path']}\n" for row in rows)
    ).encode("utf-8")


def _manifest_bytes(manifest: Mapping[str, Any]) -> bytes:
    return json.dumps(
        manifest, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False
    ).encode("utf-8")


def _build_manifest(
    *,
    rows: list[dict[str, Any]],
    fresnel: Mapping[str, Any],
    k1: Mapping[str, Any],
    include_vendor_sdk: bool,
) -> tuple[dict[str, Any], bytes]:
    checksum_bytes = _checksum_text(rows)
    categories = Counter(str(row["category"]) for row in rows)
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "type": ARCHIVE_TYPE,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "target_extract_root": r"E:\code\guest\2026OpticsMoE",
        "layout_contract": "extract over the repository root; all operational files remain below experiments/",
        "sealed_formal_warmstart5_bundle_modified": False,
        "archive_files": rows,
        "file_count": len(rows),
        "total_uncompressed_bytes": sum(int(row["size_bytes"]) for row in rows),
        "category_counts": dict(sorted(categories.items())),
        "sha256sums": {
            "archive_path": "SHA256SUMS.txt",
            "sha256": sha256_bytes(checksum_bytes),
            "scope": "all archive_files; excludes the manifest and SHA256SUMS.txt themselves",
        },
        "fresnel_v3": {
            "archive_root": FRESNEL_ARCHIVE_ROOT,
            "source_manifest_sha256": fresnel["manifest_sha256"],
            "pair_count": len(fresnel["pair_ids"]),
            "pair_ids": list(fresnel["pair_ids"]),
            "ordinary_quadratic_fresnel_only": True,
            "matched_amplitude_required": True,
            "recommended_first_aperture": "a64px",
        },
        "dual_slm_k1": {
            "archive_root": K1_ARCHIVE_ROOT,
            "source_manifest_sha256": k1["manifest_sha256"],
            "scale_k": 1.0,
            "pairs": list(k1["pairs"]),
            "amplitude_polarity": "255=open, 0=closed",
        },
        "hardware": {
            "amplitude_slm": "Meadowlark PCIe 1024x1024, 17 um, 8-bit",
            "phase_slm": "1920x1200, 8 um; phase mask is loaded manually",
            "camera": "TUCam/Mosaic x64 runtime",
            "vendor_runtime_included": bool(include_vendor_sdk),
            "brightness_contract": "32 gray values including 0/255; 3 frames per gray; fixed exposure",
            "geometry_contract": "one session-fixed TL/TR/BR/BL homography to canonical 478x478",
        },
        "agreement": {
            "tools": [
                "agreement_export (server/model environment)",
                "agreement_evaluate (laboratory, model-free)",
                "agreement_report (laboratory, model-free)",
            ],
            "probe_families": ["designed calibration", "held-out model", "repeatability"],
            "primary_metrics": [
                "PCC",
                "signal-support PCC",
                "SSIM",
                "shape NRMSE",
                "calibrated energy ratio",
                "saturation fraction",
                "centroid error",
                "outside-support energy",
            ],
            "network_input_normalization": [
                "clamp nonnegative",
                "divide by per-frame mean",
                "relative clip",
                "log1p",
                "adaptive average pool 478x478 to 224x224",
            ],
        },
        "explicit_exclusions": [
            "the sealed formal warmstart5 ZIP and its sidecars",
            "model checkpoints and datasets",
            "Caltech101/Hugging Face caches",
            "CCD experiment sessions and reconstructed bulk payloads",
            "x86 vendor libraries, PDFs, IDE projects, and vendor example images",
        ],
    }
    return manifest, checksum_bytes


def _write_zip(
    output: Path,
    files: list[PackageFile],
    manifest: Mapping[str, Any],
    checksum_bytes: bytes,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(
        output,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=6,
        allowZip64=True,
    ) as archive:
        for item in files:
            archive.write(item.source, item.archive_path)
        archive.writestr("SHA256SUMS.txt", checksum_bytes)
        archive.writestr("validation_bundle_manifest.json", _manifest_bytes(manifest))


def verify_validation_zip(path: str | Path) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    _require_file(source, "validation ZIP")
    with zipfile.ZipFile(source, "r") as archive:
        infos = archive.infolist()
        names = [info.filename for info in infos]
        if len(names) != len(set(names)):
            raise RuntimeError("Validation ZIP contains duplicate member names")
        if any(_safe_archive_path(name) != name for name in names):
            raise RuntimeError("Validation ZIP contains unsafe/non-canonical member names")
        required_meta = {"validation_bundle_manifest.json", "SHA256SUMS.txt"}
        if not required_meta.issubset(names):
            raise RuntimeError("Validation ZIP is missing its manifest or SHA256SUMS.txt")
        manifest = json.loads(archive.read("validation_bundle_manifest.json"))
        if not isinstance(manifest, dict) or manifest.get("type") != ARCHIVE_TYPE:
            raise RuntimeError("Validation ZIP contains the wrong manifest type")
        rows = manifest.get("archive_files")
        if not isinstance(rows, list):
            raise RuntimeError("Validation bundle manifest has no archive_files list")
        expected_names = {str(row["archive_path"]) for row in rows} | required_meta
        if set(names) != expected_names:
            missing = sorted(expected_names - set(names))
            extra = sorted(set(names) - expected_names)
            raise RuntimeError(f"Validation ZIP entry mismatch: missing={missing}, extra={extra}")
        for row in rows:
            name = _safe_archive_path(str(row["archive_path"]))
            value = archive.read(name)
            if len(value) != int(row["size_bytes"]) or sha256_bytes(value) != row["sha256"]:
                raise RuntimeError(f"Validation ZIP content mismatch for {name}")
        checksum_bytes = archive.read("SHA256SUMS.txt")
        if checksum_bytes != _checksum_text(rows):
            raise RuntimeError("Validation ZIP SHA256SUMS.txt does not match archive_files")
        if sha256_bytes(checksum_bytes) != manifest.get("sha256sums", {}).get("sha256"):
            raise RuntimeError("Validation ZIP SHA256SUMS.txt digest mismatch")
        if manifest.get("sealed_formal_warmstart5_bundle_modified") is not False:
            raise RuntimeError("Validation bundle must explicitly leave the formal bundle untouched")
        forbidden = (
            "/lab_bundles/",
            "/runs/",
            "/hardware_sessions/",
            "/x86/",
        )
        lowered_names = [f"/{name.lower()}" for name in names]
        if any(fragment in name for fragment in forbidden for name in lowered_names):
            raise RuntimeError("Validation ZIP contains a forbidden formal/session/x86 payload")
        if any(name.endswith((".pt", ".pth", ".ckpt", ".pdf")) for name in lowered_names):
            raise RuntimeError("Validation ZIP contains a checkpoint or PDF")
    return {
        "path": str(source),
        "sha256": sha256_file(source),
        "size_bytes": source.stat().st_size,
        "file_count": len(rows),
        "manifest_sha256": sha256_bytes(_manifest_bytes(manifest)),
        "vendor_runtime_included": bool(
            manifest.get("hardware", {}).get("vendor_runtime_included")
        ),
    }


def verify_extracted_tree(root: str | Path) -> dict[str, Any]:
    directory = Path(root).expanduser().resolve()
    manifest_path = _require_file(
        directory / "validation_bundle_manifest.json", "extracted validation manifest"
    )
    manifest = _read_json(manifest_path, "extracted validation manifest")
    if manifest.get("type") != ARCHIVE_TYPE:
        raise RuntimeError("Extracted tree contains the wrong validation manifest type")
    rows = manifest.get("archive_files")
    if not isinstance(rows, list):
        raise RuntimeError("Extracted validation manifest has no archive_files list")
    for row in rows:
        relative = _safe_archive_path(str(row["archive_path"]))
        path = (directory / Path(*PurePosixPath(relative).parts)).resolve()
        try:
            path.relative_to(directory)
        except ValueError as error:
            raise RuntimeError(f"Extracted path escapes root: {relative}") from error
        _require_file(path, f"extracted file {relative}")
        if path.stat().st_size != int(row["size_bytes"]) or sha256_file(path) != row["sha256"]:
            raise RuntimeError(f"Extracted validation file mismatch: {relative}")
    checksum_path = _require_file(directory / "SHA256SUMS.txt", "extracted SHA256SUMS")
    checksum_bytes = checksum_path.read_bytes()
    if checksum_bytes != _checksum_text(rows):
        raise RuntimeError("Extracted SHA256SUMS.txt does not match the manifest")
    return {
        "root": str(directory),
        "file_count": len(rows),
        "manifest_sha256": sha256_file(manifest_path),
        "status": "verified",
    }


def create_validation_bundle(
    *,
    repo_root: str | Path,
    output_path: str | Path,
    fresnel_dir: str | Path,
    k1_dir: str | Path,
    include_vendor_sdk: bool = True,
    overwrite: bool = False,
    check_only: bool = False,
) -> dict[str, Any]:
    root = Path(repo_root).expanduser().resolve()
    output = Path(output_path).expanduser().resolve()
    fresnel = validate_fresnel_v3(fresnel_dir)
    k1 = validate_k1_suite(k1_dir)
    files = _repository_files(root)
    files.extend(_generated_files(fresnel, FRESNEL_ARCHIVE_ROOT, "fresnel_v3"))
    files.extend(_generated_files(k1, K1_ARCHIVE_ROOT, "dual_slm_k1"))
    if include_vendor_sdk:
        files.extend(_vendor_files(root))
    files = _deduplicate(files)
    rows = _archive_rows(files)
    manifest, checksum_bytes = _build_manifest(
        rows=rows,
        fresnel=fresnel,
        k1=k1,
        include_vendor_sdk=include_vendor_sdk,
    )
    summary: dict[str, Any] = {
        "status": "source_validation_passed" if check_only else "created",
        "output": str(output),
        "archive_file_count": len(rows),
        "uncompressed_bytes": sum(int(row["size_bytes"]) for row in rows),
        "fresnel_pairs": len(fresnel["pair_ids"]),
        "k1_pairs": list(k1["pairs"]),
        "vendor_runtime_included": bool(include_vendor_sdk),
        "sealed_formal_bundle_modified": False,
    }
    if check_only:
        return summary
    if output.exists() and not overwrite:
        raise FileExistsError(f"Output already exists; pass --overwrite to replace it: {output}")
    if output.exists():
        output.unlink()
    _write_zip(output, files, manifest, checksum_bytes)
    verified = verify_validation_zip(output)
    zip_digest = verified["sha256"]
    sha_path = output.with_suffix(output.suffix + ".sha256")
    sha_path.write_text(f"{zip_digest}  {output.name}\n", encoding="ascii")
    sidecar_path = output.with_suffix(output.suffix + ".json")
    sidecar = {
        **summary,
        **verified,
        "sha256_file": str(sha_path),
        "manifest_type": ARCHIVE_TYPE,
    }
    sidecar_path.write_text(
        json.dumps(sidecar, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
        newline="\n",
    )
    return {**sidecar, "sidecar": str(sidecar_path)}


def _default_repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _default_paths(repo_root: Path) -> tuple[Path, Path, Path]:
    generated = repo_root / "experiments" / "hardware_sdk" / "generators" / "slm_patterns" / "generated"
    output = (
        repo_root
        / "experiments"
        / PROJECT_PACKAGE
        / "validation_bundles"
        / DEFAULT_ZIP_NAME
    )
    fresnel = generated / "fresnel_square_aperture_array_532nm_17um_8um_v3"
    k1 = generated / "dual_slm_17um_8um_normal_large_blocks_k0p1" / "00_k1_ready_to_play"
    return output, fresnel, k1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--verify-zip", default=None, help="verify an existing validation ZIP")
    mode.add_argument(
        "--verify-tree",
        default=None,
        help="verify files after extracting the bundle over a repository root",
    )
    parser.add_argument("--repo-root", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--fresnel-dir", default=None)
    parser.add_argument("--k1-dir", default=None)
    parser.add_argument("--omit-vendor-sdk", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args(argv)

    if args.verify_zip:
        print(json.dumps(verify_validation_zip(args.verify_zip), ensure_ascii=False, indent=2))
        return 0
    if args.verify_tree:
        print(json.dumps(verify_extracted_tree(args.verify_tree), ensure_ascii=False, indent=2))
        return 0

    repo_root = (
        Path(args.repo_root).expanduser().resolve()
        if args.repo_root
        else _default_repo_root()
    )
    default_output, default_fresnel, default_k1 = _default_paths(repo_root)
    report = create_validation_bundle(
        repo_root=repo_root,
        output_path=args.output or default_output,
        fresnel_dir=args.fresnel_dir or default_fresnel,
        k1_dir=args.k1_dir or default_k1,
        include_vendor_sdk=not args.omit_vendor_sdk,
        overwrite=args.overwrite,
        check_only=args.check_only,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
