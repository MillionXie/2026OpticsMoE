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


def _project_code(selected: dict[str, Path], root: Path) -> None:
    base = root / PROJECT
    for path in base.glob("*.py"):
        selected[path.relative_to(root).as_posix()] = path
    _add_tree(selected, root, f"{PROJECT}/configs/release")
    _add_tree(selected, root, f"{PROJECT}/configs/deployment")
    _add_tree(selected, root, f"{PROJECT}/tests", {"__pycache__"})
    for relative in ("experiments/__init__.py",):
        selected[relative] = root / relative


def _hardware_code(selected: dict[str, Path], root: Path) -> None:
    _add_tree(selected, root, "experiments/hardware_sdk", {"__pycache__", "artifacts"})
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
    _project_code(selected, root)
    _hardware_code(selected, root)
    _data_files(settings, selected)
    selected[f"{PROJECT}/deployment/checkpoints/best_observed_test_checkpoint.pt"] = checkpoint
    for name in (
        "TEMPORAL9_COMPACT_STUDY.md",
        "temporal9_compact_result.json",
        "SPATIAL_OPTIMIZATION_RESULT.md",
        "spatial_optimization_result.json",
        "SPATIAL_BALANCED_FORMAL_RESULT.md",
        "spatial_balanced_formal_result.json",
        "LAB_TEMPORAL9_GUIDE.md",
        "LAB_SPATIAL4_GUIDE.md",
    ):
        path = root / PROJECT / name
        if path.is_file():
            selected[path.relative_to(root).as_posix()] = path
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
    _project_code(selected, root)
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
