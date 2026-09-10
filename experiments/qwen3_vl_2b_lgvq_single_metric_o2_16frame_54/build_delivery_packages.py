"""Build hash-audited laboratory and phase-evolution delivery ZIPs."""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import yaml

from .export_hardware_masks import export_hardware_masks
from .settings import load_settings


PROJECT = "experiments/qwen3_vl_2b_lgvq_single_metric_o2_16frame_54"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tree(root: Path, relative: str, excluded: set[str] | None = None) -> Iterable[Path]:
    base = root / relative
    excluded = excluded or set()
    if not base.exists():
        return ()
    return (
        path
        for path in base.rglob("*")
        if path.is_file() and not any(part in excluded for part in path.relative_to(base).parts)
    )


def _add_tree(selected: dict[str, Path], root: Path, relative: str, excluded: set[str] | None = None) -> None:
    for path in _tree(root, relative, excluded):
        selected[path.relative_to(root).as_posix()] = path


_LAB_RUNTIME_FILES = {
    "__init__.py",
    "__main__.py",
    "data.py",
    "export_hardware_masks.py",
    "hardware_bridge.py",
    "hardware_contract.py",
    "metrics.py",
    "modeling.py",
    "phase_snapshots.py",
    "plot_results.py",
    "preflight.py",
    "run.py",
    "settings.py",
    "training.py",
    "VERIFY_BUNDLE.py",
}


def _add_config_chain(selected: dict[str, Path], root: Path, config: Path) -> None:
    """Copy only the selected config and its explicit base chain."""

    current = config.resolve()
    seen: set[Path] = set()
    while current not in seen:
        seen.add(current)
        try:
            relative = current.relative_to(root)
        except ValueError as error:
            raise ValueError(f"Config is outside repository root: {current}") from error
        selected[relative.as_posix()] = current
        raw = yaml.safe_load(current.read_text(encoding="utf-8")) or {}
        base = raw.get("base_config")
        if base is None:
            return
        base_path = Path(str(base)).expanduser()
        current = (
            base_path.resolve()
            if base_path.is_absolute()
            else (current.parent / base_path).resolve()
        )
    raise ValueError(f"Cyclic config chain while packaging {config}")


def _project_code(
    selected: dict[str, Path],
    root: Path,
    *,
    config: Path,
    runtime_only: bool,
) -> None:
    base = root / PROJECT
    for path in base.glob("*.py"):
        if not runtime_only or path.name in _LAB_RUNTIME_FILES:
            selected[path.relative_to(root).as_posix()] = path
    if runtime_only:
        _add_config_chain(selected, root, config)
        deployment = base / "configs/deployment/spatial4_custom_conv_lab.yaml"
        if deployment.is_file():
            _add_config_chain(selected, root, deployment)
    else:
        _add_tree(selected, root, f"{PROJECT}/configs/release")
        _add_tree(selected, root, f"{PROJECT}/configs/deployment")
    if runtime_only:
        for name in (
            "__init__.py",
            "test_hardware_contract.py",
            "test_hardware_mask_export.py",
            "test_phase_snapshots.py",
        ):
            path = base / "tests" / name
            if path.is_file():
                selected[path.relative_to(root).as_posix()] = path
    else:
        _add_tree(selected, root, f"{PROJECT}/tests", {"__pycache__"})
    for relative in ("experiments/__init__.py",):
        selected[relative] = root / relative


def _hardware_code(
    selected: dict[str, Path], root: Path, *, current_bench_only: bool
) -> None:
    excluded = {"__pycache__", "artifacts"}
    if current_bench_only:
        # The current bench is Meadowlark PCIe + manual 8 um phase SLM + TUCam.
        # Do not ship historical display-SLM or legacy camera SDK trees.
        excluded.update({"amplitude_holoeye", "legacy"})
    _add_tree(selected, root, "experiments/hardware_sdk", excluded)
    _add_tree(
        selected,
        root,
        "experiments/lab_lgvq",
        {"__pycache__", "generated", "work", "results", "sessions"},
    )
    # The clean template deliberately replaces any bench-specific/legacy file.
    template = root / "experiments/lab_lgvq/LAB_CONFIG_TEMPLATE_CLEAN.yaml"
    selected["experiments/lab_lgvq/LAB_CONFIG.yaml"] = template
    for relative in ("experiments/lab_qwen/__init__.py", "experiments/lab_qwen/prepare_lab.py"):
        path = root / relative
        if path.is_file():
            selected[relative] = path


def _data_files(settings, selected: dict[str, Path]) -> None:
    mapping = {
        "manifest_path": "lgvq_train2250_test558.csv",
        "vision_cache_path": Path(settings.vision_cache_path).name,
        "language_cache_path": Path(settings.language_cache_path).name,
        "quality_feature_cache_path": None if settings.quality_feature_cache_path is None else Path(settings.quality_feature_cache_path).name,
        "raw_frame_cache_path": None if settings.raw_frame_cache_path is None else Path(settings.raw_frame_cache_path).name,
        "vgg_feature_cache_path": None if settings.vgg_feature_cache_path is None else Path(settings.vgg_feature_cache_path).name,
        "training_soft_targets_path": None if settings.training_soft_targets_path is None else "training_only_teacher_predictions.pt",
    }
    for attribute, filename in mapping.items():
        path = getattr(settings, attribute)
        if path is not None and filename is not None:
            if not Path(path).is_file():
                raise FileNotFoundError(path)
            selected[f"{PROJECT}/deployment/data/{filename}"] = Path(path)


def _write_zip(
    selected: dict[str, Path],
    output: Path,
    *,
    purpose: str,
    checkpoint_sha256: str,
    root_readme: Path,
) -> dict:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.unlink(missing_ok=True)
    records = []
    with zipfile.ZipFile(temporary, "w", allowZip64=True) as archive:
        for arcname, source in sorted(selected.items()):
            if not source.is_file():
                raise FileNotFoundError(source)
            compression = zipfile.ZIP_STORED if source.suffix.lower() == ".pt" else zipfile.ZIP_DEFLATED
            archive.write(source, arcname, compress_type=compression, compresslevel=None if compression == zipfile.ZIP_STORED else 3)
            records.append({"path": arcname, "bytes": source.stat().st_size, "sha256": sha256(source)})
        for arcname, source in (("README_FIRST.md", root_readme), ("VERIFY_BUNDLE.py", root_readme.parent / "VERIFY_BUNDLE.py")):
            archive.write(source, arcname, compress_type=zipfile.ZIP_DEFLATED, compresslevel=3)
            records.append({"path": arcname, "bytes": source.stat().st_size, "sha256": sha256(source)})
        manifest = {
            "schema_version": 1,
            "built_utc": datetime.now(timezone.utc).isoformat(),
            "purpose": purpose,
            "checkpoint_sha256": checkpoint_sha256,
            "files": records,
            "total_bytes": sum(int(row["bytes"]) for row in records),
        }
        archive.writestr("BUNDLE_MANIFEST.json", json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(output)
    report = {
        **manifest,
        "zip": str(output.resolve()),
        "zip_bytes": output.stat().st_size,
        "zip_sha256": sha256(output),
    }
    output.with_suffix(output.suffix + ".sha256").write_text(f"{report['zip_sha256']}  {output.name}\n", encoding="ascii")
    output.with_suffix(".report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def build_lab(root: Path, config: Path, checkpoint: Path, output: Path, guide: Path) -> dict:
    settings = load_settings(config)
    checkpoint = checkpoint.resolve()
    selected: dict[str, Path] = {}
    custom_runtime = bool(
        settings.custom_conv_electronic_enabled
        and settings.vgg_feature_cache_path is None
        and settings.resnet_feature_cache_path is None
        and settings.mobilenet_feature_cache_path is None
    )
    _project_code(
        selected,
        root,
        config=config,
        runtime_only=custom_runtime,
    )
    _hardware_code(selected, root, current_bench_only=custom_runtime)
    _data_files(settings, selected)
    selected[f"{PROJECT}/deployment/checkpoints/best_observed_test_checkpoint.pt"] = checkpoint
    document_names = (
        ("LAB_SPATIAL4_CUSTOM_CONV_GUIDE.md",)
        if custom_runtime
        else (
            "TEMPORAL9_COMPACT_STUDY.md",
            "temporal9_compact_result.json",
            "SPATIAL_OPTIMIZATION_RESULT.md",
            "spatial_optimization_result.json",
            "SPATIAL_BALANCED_FORMAL_RESULT.md",
            "spatial_balanced_formal_result.json",
            "LAB_TEMPORAL9_GUIDE.md",
            "LAB_SPATIAL4_GUIDE.md",
        )
    )
    for name in document_names:
        path = root / PROJECT / name
        if path.is_file():
            selected[path.relative_to(root).as_posix()] = path
    if custom_runtime:
        architecture = (
            root
            / "LightGenV2/tasks/t06_video_quality_assessment/"
            "SPATIAL_CUSTOM_OEO_ARCHITECTURE.md"
        )
        result_root = (
            root
            / "LightGenV2/tasks/t06_video_quality_assessment/reports/"
            "paper_results/spatial_custom_conv_s749_20260910"
        )
        if architecture.is_file():
            selected["documentation/SPATIAL_CUSTOM_OEO_ARCHITECTURE.md"] = architecture
        for path in result_root.glob("*"):
            if path.is_file():
                selected[f"documentation/formal_result/{path.name}"] = path
    study_document = root / PROJECT / "TEMPORAL_16_36_SPEED_QUALITY_STUDY.md"
    if settings.target_name == "temporal" and study_document.is_file():
        selected[study_document.relative_to(root).as_posix()] = study_document
    if settings.target_name == "temporal" and settings.frame_count in (16, 36):
        study_root = root / PROJECT / "artifacts/temporal16_36_study"
        report = study_root / "models" / (
            f"temporal{settings.frame_count}_balanced_calibrated_report.json"
        )
        if report.is_file():
            selected[
                f"{PROJECT}/deployment/evidence/formal_metrics_report.json"
            ] = report
        for relative in (
            "tradeoff",
            f"temporal{settings.frame_count}_optical_fields",
        ):
            directory = study_root / relative
            if directory.is_dir():
                for path in directory.rglob("*"):
                    if path.is_file():
                        selected[
                            f"{PROJECT}/deployment/evidence/{relative}/"
                            f"{path.relative_to(directory).as_posix()}"
                        ] = path
    figures = root / PROJECT / "artifacts/temporal9_final_figures"
    if settings.target_name == "temporal" and figures.is_dir():
        _add_tree(selected, root, f"{PROJECT}/artifacts/temporal9_final_figures")
    spatial_figures = root / PROJECT / "artifacts/spatial_balanced_final_figures"
    if settings.target_name == "spatial" and spatial_figures.is_dir():
        _add_tree(selected, root, f"{PROJECT}/artifacts/spatial_balanced_final_figures")
    with tempfile.TemporaryDirectory(prefix="lgvq_masks_") as temporary:
        mask_root = Path(temporary) / "hardware_masks"
        export_hardware_masks(settings, checkpoint, mask_root)
        for path in mask_root.rglob("*"):
            if path.is_file():
                selected[f"{PROJECT}/deployment/hardware_masks/{path.relative_to(mask_root).as_posix()}"] = path
        return _write_zip(
            selected,
            output,
            purpose=f"{settings.target_name} laboratory fine-tuning and six-pass hardware control",
            checkpoint_sha256=sha256(checkpoint),
            root_readme=guide,
        )


def build_evolution(root: Path, config: Path, checkpoint: Path, snapshots: Path, output: Path) -> dict:
    settings = load_settings(config)
    selected: dict[str, Path] = {}
    _project_code(selected, root, config=config, runtime_only=False)
    selected["best_observed_test_checkpoint.pt"] = checkpoint
    for path in snapshots.rglob("*"):
        if path.is_file() and path.suffix.lower() in {
            ".pt", ".json", ".csv", ".npy", ".png", ".pdf"
        }:
            selected[f"phase_snapshots/{path.relative_to(snapshots).as_posix()}"] = path
    if not any(name.startswith("phase_snapshots/epoch_") for name in selected):
        raise FileNotFoundError(f"No epoch snapshots under {snapshots}")
    readme = root / PROJECT / "MASK_EVOLUTION_HANDOFF.md"
    selected["MASK_EVOLUTION_HANDOFF.md"] = readme
    return _write_zip(
        selected,
        output,
        purpose=f"{settings.target_name} five-epoch optical-mask evolution analysis",
        checkpoint_sha256=sha256(checkpoint),
        root_readme=readme,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=("lab", "evolution"))
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--guide", default=f"{PROJECT}/LAB_TEMPORAL9_GUIDE.md")
    parser.add_argument("--snapshot-dir")
    args = parser.parse_args()
    root = Path(args.repo_root).resolve()
    if args.kind == "lab":
        report = build_lab(root, Path(args.config), Path(args.checkpoint), Path(args.output), Path(args.guide))
    else:
        if args.snapshot_dir is None:
            parser.error("evolution requires --snapshot-dir")
        report = build_evolution(root, Path(args.config), Path(args.checkpoint), Path(args.snapshot_dir), Path(args.output))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
