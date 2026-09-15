"""Build a self-contained laboratory transfer bundle for the 10 cm Qwen run.

The archive deliberately contains only compact, reproducible payloads.  In
particular, Caltech101 images, Hugging Face caches, teacher caches, generated
full-size amplitude BMPs, and CCD captures are never selected by this module.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import zipfile
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


PROJECT_PACKAGE = (
    "qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_robust"
)
STAGES = (
    "vision_expert",
    "vision_global",
    "language_expert",
    "language_global",
)
QUICK_SAMPLE_COUNT = 210
DEFAULT_ZIP_NAME = "qwen_caltech101_10cm_quick210_lab_bundle.zip"


@dataclass(frozen=True)
class BundleFile:
    source: Path
    archive_path: str
    category: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _require_file(path: Path, label: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"Required {label} is missing: {path}")
    return path


def _source_label(path: Path, repo_root: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return path.name


def _runtime_files(repo_root: Path, include_vendor_sdk: bool) -> Iterable[BundleFile]:
    project = f"experiments/{PROJECT_PACKAGE}"
    relative_files = (
        "experiments/__init__.py",
        "experiments/hardware_sdk/__init__.py",
        "experiments/hardware_sdk/devices.py",
        "experiments/hardware_sdk/configs/tucam_meadowlark_1024_windows.yaml",
        "experiments/hardware_sdk/workflows/__init__.py",
        "experiments/hardware_sdk/workflows/acquire_folder.py",
        "experiments/hardware_sdk/workflows/calibration_common.py",
        "experiments/hardware_sdk/workflows/reconstruct_slm.py",
        "experiments/hardware_sdk/drivers/__init__.py",
        "experiments/hardware_sdk/drivers/meadowlark_pcie_slm.py",
        "experiments/hardware_sdk/drivers/tucam_camera.py",
        f"{project}/requirements-lab.txt",
    )
    for relative in relative_files:
        path = _require_file(repo_root / relative, "laboratory runtime file")
        yield BundleFile(path, relative, "lab_runtime")

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
            if not path.is_file():
                continue
            if "__pycache__" in path.parts or path.suffix.lower() in {".pyc", ".pyo"}:
                continue
            yield BundleFile(
                path,
                path.relative_to(repo_root).as_posix(),
                "vendor_sdk",
            )


def _reference_source_files(repo_root: Path) -> Iterable[BundleFile]:
    project_root = repo_root / "experiments" / PROJECT_PACKAGE
    relative_names = (
        "__init__.py",
        "__main__.py",
        "settings.py",
        "optical_blocks.py",
        "modeling.py",
        "run.py",
        "hardware_bridge.py",
        "export_phase_bmps.py",
        "README.md",
        "ARCHITECTURE_AUDIT.md",
        "DATA_PIPELINE.md",
        "RUN_COMMANDS.md",
        "environment_server_and_lab.yml",
        "configs/release/caltech101_four_layer_optical_joint_17um_10cm_robust.yaml",
        "configs/release/caltech101_four_layer_optical_quick_last_stage_10x10.yaml",
    )
    for relative in relative_names:
        path = _require_file(project_root / relative, "Qwen reference source")
        archive_path = (
            Path("reference")
            / "qwen_project_source"
            / relative
        ).as_posix()
        yield BundleFile(path, archive_path, "qwen_reference_source")


def _phase_export_files(phase_export_dir: Path) -> Iterable[BundleFile]:
    report_path = _require_file(
        phase_export_dir / "phase_export_report.json", "four-phase export report"
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if tuple(report.get("stages", ())) != STAGES:
        raise ValueError(
            "phase_export_report.json does not contain the ordered four-stage contract"
        )
    distance = float(report.get("propagation_distance_m", -1.0))
    if abs(distance - 0.1) > 1.0e-9:
        raise ValueError(f"Four-phase export is not the 10 cm model: distance={distance}")

    native_paths = [phase_export_dir / "phase_bmp" / f"{stage}.bmp" for stage in STAGES]
    compact_paths = [
        phase_export_dir / "compact_phase" / f"{stage}.png" for stage in STAGES
    ]
    for path in (*native_paths, *compact_paths):
        _require_file(path, "four-phase payload")

    selected = [report_path, *compact_paths, *native_paths]
    for optional in (
        phase_export_dir / "phase_preview.png",
        phase_export_dir / "phase_bmp" / "reconstruction_manifest.csv",
        phase_export_dir / "phase_bmp" / "reconstruction_report.json",
    ):
        if optional.is_file():
            selected.append(optional)
    for path in selected:
        yield BundleFile(
            path,
            (Path("payload") / "four_phase_export" / path.relative_to(phase_export_dir)).as_posix(),
            "four_phase_export",
        )


def _quick210_files(quick_session_dir: Path) -> Iterable[BundleFile]:
    session_manifest = _require_file(
        quick_session_dir / "manifest.csv", "quick210 session manifest"
    )
    session_rows = _read_csv(session_manifest)
    if len(session_rows) != QUICK_SAMPLE_COUNT:
        raise ValueError(
            f"Quick session must contain {QUICK_SAMPLE_COUNT} samples; "
            f"manifest contains {len(session_rows)}"
        )
    keys = [row.get("key", "") for row in session_rows]
    if any(not key for key in keys) or len(set(keys)) != QUICK_SAMPLE_COUNT:
        raise ValueError("Quick session manifest keys must be non-empty and unique")

    stage_dir = quick_session_dir / "04_language_global"
    transport_path = _require_file(stage_dir / "transport_spec.json", "quick transport spec")
    transport = json.loads(transport_path.read_text(encoding="utf-8"))
    if transport.get("stage") != "language_global":
        raise ValueError("Quick payload must be the language_global stage")
    if transport.get("upstream_source") != "simulation":
        raise ValueError("Quick payload must use simulated upstream stages")
    if int(transport.get("samples", -1)) != QUICK_SAMPLE_COUNT:
        raise ValueError("Quick transport spec does not declare 210 samples")

    amplitude_manifest = _require_file(
        stage_dir / "compact_amplitude_manifest.csv", "quick amplitude manifest"
    )
    amplitude_rows = _read_csv(amplitude_manifest)
    if len(amplitude_rows) != QUICK_SAMPLE_COUNT:
        raise ValueError(
            f"Quick compact amplitude manifest must contain {QUICK_SAMPLE_COUNT} rows"
        )
    declared_names = [row.get("filename", "") for row in amplitude_rows]
    if any(not name for name in declared_names) or len(set(declared_names)) != QUICK_SAMPLE_COUNT:
        raise ValueError("Quick amplitude filenames must be non-empty and unique")
    amplitude_dir = stage_dir / "compact_amplitude"
    amplitude_paths = sorted(amplitude_dir.glob("*.png"))
    if {path.name for path in amplitude_paths} != set(declared_names):
        raise ValueError(
            "Quick compact_amplitude PNG set does not exactly match its manifest"
        )

    compact_phase_paths = sorted((stage_dir / "compact_phase").glob("*.png"))
    native_phase_paths = sorted((stage_dir / "phase_to_play").glob("*.bmp"))
    if [path.name for path in compact_phase_paths] != ["language_global.png"]:
        raise ValueError("Quick stage requires exactly compact_phase/language_global.png")
    if [path.name for path in native_phase_paths] != ["language_global.bmp"]:
        raise ValueError("Quick stage requires exactly phase_to_play/language_global.bmp")

    selected = [
        session_manifest,
        transport_path,
        amplitude_manifest,
        *amplitude_paths,
        *compact_phase_paths,
        *native_phase_paths,
    ]
    for optional in (
        stage_dir / "phase_to_play" / "reconstruction_manifest.csv",
        stage_dir / "phase_to_play" / "reconstruction_report.json",
    ):
        if optional.is_file():
            selected.append(optional)
    for path in selected:
        yield BundleFile(
            path,
            (Path("payload") / "quick210" / path.relative_to(quick_session_dir)).as_posix(),
            "quick210_compact_payload",
        )


def _evidence_files(run_dir: Path) -> Iterable[BundleFile]:
    root_names = (
        "config.yaml",
        "dataset.json",
        "environment.json",
        "model.json",
        "train_log.csv",
        "formal_train.log",
        "fast_2h_train.log",
        "router_recovery_train.log",
    )
    selected = [run_dir / name for name in root_names if (run_dir / name).is_file()]
    metrics_dir = run_dir / "metrics"
    if metrics_dir.is_dir():
        selected.extend(sorted(metrics_dir.glob("*.json")))
    if not selected:
        raise FileNotFoundError(f"No whitelisted training evidence found under {run_dir}")
    for path in selected:
        yield BundleFile(
            path,
            (Path("reference") / "training_evidence" / path.relative_to(run_dir)).as_posix(),
            "training_evidence",
        )


def _deduplicate(files: Iterable[BundleFile]) -> list[BundleFile]:
    result: list[BundleFile] = []
    archive_paths: set[str] = set()
    for item in files:
        normalized = Path(item.archive_path).as_posix().lstrip("/")
        if not normalized or normalized.startswith("../"):
            raise ValueError(f"Unsafe archive path: {item.archive_path!r}")
        if normalized in archive_paths:
            raise ValueError(f"Duplicate archive path: {normalized}")
        archive_paths.add(normalized)
        result.append(BundleFile(item.source.resolve(), normalized, item.category))
    return result


def create_lab_bundle(
    *,
    checkpoint: str | Path,
    phase_export_dir: str | Path,
    quick_session_dir: str | Path,
    output_path: str | Path,
    include_evidence: bool = False,
    run_dir: str | Path | None = None,
    include_vendor_sdk: bool = True,
    overwrite: bool = False,
    repo_root: str | Path | None = None,
) -> dict[str, object]:
    root = (
        Path(repo_root).expanduser().resolve()
        if repo_root is not None
        else Path(__file__).resolve().parents[2]
    )
    checkpoint_path = _require_file(
        Path(checkpoint).expanduser().resolve(), "selected Qwen checkpoint"
    )
    phase_root = Path(phase_export_dir).expanduser().resolve()
    quick_root = Path(quick_session_dir).expanduser().resolve()
    output = Path(output_path).expanduser().resolve()
    if output.exists() and not overwrite:
        raise FileExistsError(f"Bundle already exists; pass --overwrite to replace it: {output}")
    if include_evidence and run_dir is None:
        run_dir = checkpoint_path.parent

    project_root = root / "experiments" / PROJECT_PACKAGE
    readme_path = _require_file(project_root / "LAB_BUNDLE.md", "bundle README")
    files: list[BundleFile] = [
        BundleFile(readme_path, "README_LAB_AND_SERVER.md", "documentation"),
        BundleFile(
            checkpoint_path,
            f"payload/checkpoint/{checkpoint_path.name}",
            "selected_checkpoint",
        ),
    ]
    files.extend(_runtime_files(root, include_vendor_sdk))
    files.extend(_reference_source_files(root))
    files.extend(_phase_export_files(phase_root))
    files.extend(_quick210_files(quick_root))
    if include_evidence:
        files.extend(_evidence_files(Path(run_dir).expanduser().resolve()))
    files = _deduplicate(files)

    entries = []
    for item in files:
        entries.append(
            {
                "archive_path": item.archive_path,
                "category": item.category,
                "source": _source_label(item.source, root),
                "size_bytes": item.source.stat().st_size,
                "sha256": _sha256(item.source),
            }
        )
    category_counts = dict(sorted(Counter(row["category"] for row in entries).items()))
    manifest = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "project": PROJECT_PACKAGE,
        "hardware_contract": {
            "propagation_distance_m": 0.1,
            "logical_pixel_pitch_um": 17.0,
            "phase_slm_pixel_pitch_um": 8.0,
            "optical_fusion_minimum": 0.1,
            "amplitude_command_polarity": "255=bright/transmissive, 0=dark/blocking",
            "quick_stage": "language_global",
            "quick_samples": QUICK_SAMPLE_COUNT,
            "quick_upstream_source": "simulation",
        },
        "selected_checkpoint": {
            "archive_path": f"payload/checkpoint/{checkpoint_path.name}",
            "sha256": _sha256(checkpoint_path),
        },
        "include_vendor_sdk": bool(include_vendor_sdk),
        "include_training_evidence": bool(include_evidence),
        "exclusion_contract": [
            "no Caltech101 image dataset",
            "no Hugging Face/model/download cache",
            "no teacher/feature cache",
            "no CCD captures",
            "no generated full-size amplitude_to_play BMPs",
            "no extra checkpoints beyond the explicitly selected checkpoint",
        ],
        "category_file_counts": category_counts,
        "archive_files": entries,
    }
    manifest_bytes = json.dumps(manifest, ensure_ascii=False, indent=2).encode("utf-8")

    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(
        output,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=6,
        allowZip64=True,
    ) as archive:
        for item in files:
            archive.write(item.source, item.archive_path)
        archive.writestr("bundle_manifest.json", manifest_bytes)

    report = {
        "zip": str(output),
        "zip_sha256": _sha256(output),
        "zip_size_bytes": output.stat().st_size,
        "manifest_entries": len(entries),
        "category_file_counts": category_counts,
        "selected_checkpoint_sha256": _sha256(checkpoint_path),
        "quick_samples": QUICK_SAMPLE_COUNT,
        "vendor_sdk_included": bool(include_vendor_sdk),
        "training_evidence_included": bool(include_evidence),
    }
    sidecar = output.with_suffix(output.suffix + ".json")
    sidecar.write_text(
        json.dumps({**report, "bundle_manifest": manifest}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Package the Qwen 10 cm quick210 laboratory payload and runtime"
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--phase-export-dir", required=True)
    parser.add_argument("--quick-session-dir", required=True)
    parser.add_argument("--output", default=DEFAULT_ZIP_NAME)
    parser.add_argument(
        "--include-evidence",
        action="store_true",
        help="Include only the small whitelisted run logs/JSON/CSV evidence",
    )
    parser.add_argument(
        "--run-dir",
        help="Run evidence directory; defaults to the selected checkpoint directory",
    )
    parser.add_argument(
        "--omit-vendor-sdk",
        action="store_true",
        help="Developer-only small archive; formal lab transfer should include SDKs",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    report = create_lab_bundle(
        checkpoint=args.checkpoint,
        phase_export_dir=args.phase_export_dir,
        quick_session_dir=args.quick_session_dir,
        output_path=args.output,
        include_evidence=args.include_evidence,
        run_dir=args.run_dir,
        include_vendor_sdk=not args.omit_vendor_sdk,
        overwrite=args.overwrite,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
