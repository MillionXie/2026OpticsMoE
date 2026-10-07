"""Build the corrected, standalone LGVQ Temporal 0.8044 teacher release."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
import zipfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ARCHIVE_SHA256 = (
    "8421e7418ade4c9ec59f924cdad8e274db4100600c001cf802a8291ab7a67aff"
)
CHECKPOINT_SHA256 = (
    "5303b574b200e14bf943af21c60a246720be453b9cf8847cd93c0eaaa243a77c"
)
PACKAGE_NAME = "lgvq_temporal_08044"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def verify_source_tree(root: Path) -> dict[str, object]:
    manifest_path = root / "SHA256.json"
    if not manifest_path.is_file():
        raise RuntimeError("Pinned source archive has no SHA256.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    errors: list[str] = []
    for relative, expected in manifest.items():
        path = root / relative
        if not path.is_file():
            errors.append(f"missing: {relative}")
        elif sha256(path) != expected:
            errors.append(f"SHA256 mismatch: {relative}")
    if errors:
        raise RuntimeError("Pinned source archive failed verification:\n" + "\n".join(errors))
    release = json.loads((root / "release.json").read_text(encoding="utf-8"))
    if release.get("checkpoint_sha256") != CHECKPOINT_SHA256:
        raise RuntimeError("Pinned source archive contains a different checkpoint")
    return release


def copy_project_source(destination: Path) -> None:
    shutil.copy2(PROJECT_ROOT / "TEACHER_README.md", destination / "README.md")
    for relative in (
        "requirements.txt",
        "pyproject.toml",
        "simulate.py",
        "train.py",
        "verify_release.py",
    ):
        shutil.copy2(PROJECT_ROOT / relative, destination / relative)
    for directory in ("configs", "runtime", "tests"):
        shutil.copytree(
            PROJECT_ROOT / directory,
            destination / directory,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache"),
        )


def sanitize_release(release: dict[str, object]) -> dict[str, object]:
    result = dict(release)
    result.update(
        {
            "schema_version": 2,
            "package_name": PACKAGE_NAME,
            "source_checkpoint": "weights/best_checkpoint.pt",
            "training_record_commit": "ddc0d71abb79185b506e8db979b38695e2dffa79",
            "reproduction_runtime_commit": "8e869473787f4ffceb2a6a77f4430b94c206f459",
            "model_selection": {
                "best_epoch": 30,
                "criterion": "periodic test SRCC",
                "test_interval_epochs": 5,
                "validation_used": False,
                "selection_bias_disclosed": True,
            },
            "fixed_input_boundary": (
                "35 packaged frozen Qwen-front test fields; no raw videos or "
                "network download required"
            ),
            "simulation_metrics": {
                "target": "temporal",
                "srcc": 0.8022806420367042,
                "krcc": 0.5931156844050931,
                "plcc": 0.8154774079271463,
                "rmse": 8.049678802490234,
                "mae": 6.05666971206665,
                "count": 558,
            },
        }
    )
    feature_source = dict(result.get("feature_source", {}))
    for key in ("vision_cache_path", "language_cache_path"):
        if key in feature_source:
            feature_source[key] = "packaged_as_inputs/field_*.pt"
    result["feature_source"] = feature_source
    return result


def build(
    source_archive: Path,
    output: Path,
    training_assets_root: Path | None = None,
) -> dict[str, object]:
    source_archive = source_archive.expanduser().resolve()
    output = output.expanduser().resolve()
    delivery_path = output.with_suffix(".delivery.json")
    if not source_archive.is_file():
        raise FileNotFoundError(source_archive)
    if output.exists() or delivery_path.exists():
        raise FileExistsError("Refusing to overwrite an existing release")
    archive_sha = sha256(source_archive)
    if archive_sha != SOURCE_ARCHIVE_SHA256:
        raise RuntimeError(
            f"Wrong source archive SHA256: expected {SOURCE_ARCHIVE_SHA256}, got {archive_sha}"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="lgvq_temporal_08044_") as temporary:
        temporary_root = Path(temporary)
        source_root = temporary_root / "source"
        package_root = temporary_root / PACKAGE_NAME
        with zipfile.ZipFile(source_archive) as archive:
            archive.extractall(source_root)
        release = verify_source_tree(source_root)
        package_root.mkdir()
        copy_project_source(package_root)
        shutil.copytree(source_root / "inputs", package_root / "inputs")
        shutil.copytree(source_root / "phases", package_root / "phases")
        (package_root / "weights").mkdir()
        shutil.copy2(
            source_root / "weights" / "best_checkpoint.pt",
            package_root / "weights" / "best_checkpoint.pt",
        )
        if training_assets_root is not None:
            training_assets_root = training_assets_root.expanduser().resolve()
            if not training_assets_root.is_dir():
                raise FileNotFoundError(training_assets_root)
            shutil.copytree(training_assets_root, package_root / "assets")
        if sha256(package_root / "weights" / "best_checkpoint.pt") != CHECKPOINT_SHA256:
            raise RuntimeError("Checkpoint changed while building the release")
        write_json(package_root / "release.json", sanitize_release(release))
        manifest = {
            path.relative_to(package_root).as_posix(): sha256(path)
            for path in sorted(package_root.rglob("*"))
            if path.is_file() and path.name != "SHA256.json"
        }
        write_json(package_root / "SHA256.json", manifest)
        with zipfile.ZipFile(
            output, mode="x", compression=zipfile.ZIP_DEFLATED, compresslevel=6
        ) as archive:
            for path in sorted(package_root.rglob("*")):
                if path.is_file():
                    archive.write(
                        path,
                        (Path(PACKAGE_NAME) / path.relative_to(package_root)).as_posix(),
                    )
    delivery = {
        "package": PACKAGE_NAME,
        "zip": output.name,
        "sha256": sha256(output),
        "bytes": output.stat().st_size,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "source_archive_sha256": SOURCE_ARCHIVE_SHA256,
        "reference_srcc": 0.8022806420367042,
        "test_videos": 558,
        "ccd_noise_available": True,
        "ccd_noise_enabled_in_reported_config": False,
    }
    write_json(delivery_path, delivery)
    return delivery


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--training-assets-root", type=Path, default=None)
    args = parser.parse_args()
    print(
        json.dumps(
            build(args.source_archive, args.output, args.training_assets_root),
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
