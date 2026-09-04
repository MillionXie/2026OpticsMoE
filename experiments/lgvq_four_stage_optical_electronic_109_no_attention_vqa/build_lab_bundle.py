"""Build an allowlisted, hash-audited LGVQ simulation and hardware ZIP."""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


PROJECT_NAME = "lgvq_four_stage_optical_electronic_109_no_attention_vqa"
EXPECTED_CHECKPOINT_SHA256 = (
    "d357fe51b888ecace74c050096febebc09c08abc21d6b42f533ecc3cf1f4de55"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_archive(path: Path) -> dict[str, object]:
    """Stream every archived payload and compare it with BUNDLE_MANIFEST.json."""

    failures: list[str] = []
    with zipfile.ZipFile(path, "r", allowZip64=True) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            failures.append("duplicate archive member name")
        manifest = json.loads(archive.read("BUNDLE_MANIFEST.json").decode("utf-8"))
        for record in manifest["files"]:
            arcname = str(record["path"])
            try:
                info = archive.getinfo(arcname)
            except KeyError:
                failures.append(f"missing: {arcname}")
                continue
            if info.file_size != int(record["bytes"]):
                failures.append(f"size mismatch: {arcname}")
                continue
            digest = hashlib.sha256()
            with archive.open(info, "r") as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(block)
            if digest.hexdigest() != record["sha256"]:
                failures.append(f"SHA256 mismatch: {arcname}")
    if failures:
        raise RuntimeError("Archive verification failed: " + "; ".join(failures[:20]))
    return {
        "archive_verified": True,
        "verified_file_count": len(manifest["files"]),
        "verified_payload_bytes": int(manifest["total_bytes"]),
    }


def _files(root: Path, relative: str, *, suffixes: set[str] | None = None) -> Iterable[Path]:
    base = root / relative
    if not base.exists():
        return ()
    values = (path for path in base.rglob("*") if path.is_file())
    if suffixes is not None:
        values = (path for path in values if path.suffix.lower() in suffixes)
    return values


def _add_tree(
    selected: dict[str, Path],
    repo: Path,
    relative: str,
    *,
    suffixes: set[str] | None = None,
    excluded_parts: set[str] | None = None,
) -> None:
    excluded_parts = excluded_parts or set()
    for path in _files(repo, relative, suffixes=suffixes):
        rel = path.relative_to(repo).as_posix()
        if any(part in excluded_parts for part in path.relative_to(repo).parts):
            continue
        selected[rel] = path


def collect_files(
    repo: Path,
    checkpoint: Path,
    frame_cache: Path | None,
    dataset_manifest: Path | None,
) -> dict[str, Path]:
    selected: dict[str, Path] = {}
    project = repo / "experiments" / PROJECT_NAME
    for name in (
        "__init__.py",
        "__main__.py",
        "settings.py",
        "data.py",
        "metrics.py",
        "modeling.py",
        "training.py",
        "run.py",
        "preflight.py",
        "prepare_manifest.py",
        "cache_frames.py",
        "hardware_contract.py",
        "hardware_bridge.py",
        "agreement_evaluate.py",
        "result_report.py",
        "build_lab_bundle.py",
        "VERIFY_BUNDLE.py",
        "ARCHITECTURE.md",
        "README.md",
        "RESULTS.md",
        "RUN_COMMANDS.md",
        "SIMULATION_REPORT.md",
        "LAB_DEPLOYMENT.md",
        "AI_HANDOFF.md",
        "BUNDLE_CONTENTS.md",
        "requirements-lab.txt",
    ):
        path = project / name
        if not path.is_file():
            raise FileNotFoundError(path)
        selected[path.relative_to(repo).as_posix()] = path
    _add_tree(selected, repo, f"experiments/{PROJECT_NAME}/configs/deployment")
    # Keep the compact simulation-config archive so the included architecture
    # tests remain runnable.  BUNDLE_CONTENTS names the sole selected config;
    # none of these server configs is a laboratory execution entry point.
    _add_tree(selected, repo, f"experiments/{PROJECT_NAME}/configs/release")
    _add_tree(selected, repo, f"experiments/{PROJECT_NAME}/evidence/recommended")
    _add_tree(selected, repo, f"experiments/{PROJECT_NAME}/deployment/hardware_assets")
    _add_tree(selected, repo, f"experiments/{PROJECT_NAME}/deployment/simulation_report")
    _add_tree(selected, repo, f"experiments/{PROJECT_NAME}/deployment/demo_session")
    _add_tree(selected, repo, f"experiments/{PROJECT_NAME}/tests", suffixes={".py"})

    for path in (repo / "experiments" / "__init__.py",):
        selected[path.relative_to(repo).as_posix()] = path
    for relative in (
        "experiments/hardware_sdk/__init__.py",
        "experiments/hardware_sdk/devices.py",
        "experiments/hardware_sdk/requirements-light.txt",
    ):
        path = repo / relative
        selected[relative] = path
    # The calibration workflows import helpers from both packages at module
    # import time.  Keep the complete Python surface so a clean extraction can
    # run ROI/LUT/Fresnel calibration without reaching back into the source repo.
    _add_tree(selected, repo, "experiments/hardware_sdk/generators", suffixes={".py"})
    _add_tree(selected, repo, "experiments/hardware_sdk/demos", suffixes={".py"})
    _add_tree(selected, repo, "experiments/hardware_sdk/drivers", suffixes={".py"})
    _add_tree(selected, repo, "experiments/hardware_sdk/workflows", suffixes={".py"})
    # Runtime binaries are deliberately limited to the two devices in this experiment.
    _add_tree(selected, repo, "experiments/hardware_sdk/vendor_sdk/amplitude_meadowlark")
    _add_tree(selected, repo, "experiments/hardware_sdk/vendor_sdk/camera_tucam_mosaic")

    _add_tree(
        selected,
        repo,
        "experiments/lab_lgvq",
        excluded_parts={"generated", "work", "results", "sessions", "__pycache__"},
    )
    for relative in (
        "experiments/lab_qwen/__init__.py",
        "experiments/lab_qwen/prepare_lab.py",
    ):
        path = repo / relative
        selected[relative] = path

    checkpoint_arc = (
        f"experiments/{PROJECT_NAME}/deployment/checkpoints/"
        "best_observed_test_checkpoint.pt"
    )
    selected[checkpoint_arc] = checkpoint
    if frame_cache is not None:
        selected[
            f"experiments/{PROJECT_NAME}/deployment/data/"
            "lgvq_four_frames_center100_224_uint8.pt"
        ] = frame_cache
    if dataset_manifest is not None:
        selected[
            f"experiments/{PROJECT_NAME}/deployment/data/lgvq_train2250_test558.csv"
        ] = dataset_manifest
    return selected


def build(
    *,
    repo: Path,
    checkpoint: Path,
    frame_cache: Path | None,
    dataset_manifest: Path | None,
    output: Path,
) -> dict[str, object]:
    repo = repo.expanduser().resolve()
    checkpoint = checkpoint.expanduser().resolve()
    if sha256(checkpoint) != EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError("Formal checkpoint SHA256 mismatch; refusing to package")
    selected = collect_files(repo, checkpoint, frame_cache, dataset_manifest)
    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.unlink(missing_ok=True)
    records: list[dict[str, object]] = []
    with zipfile.ZipFile(
        temporary,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=3,
        allowZip64=True,
    ) as archive:
        for arcname, source in sorted(selected.items()):
            if not source.is_file():
                raise FileNotFoundError(source)
            size = source.stat().st_size
            digest = sha256(source)
            archive.write(source, arcname)
            records.append({"path": arcname, "bytes": size, "sha256": digest})
        # Convenience copies at archive root.
        archive.write(
            repo / "experiments" / PROJECT_NAME / "LAB_DEPLOYMENT.md",
            "README_FIRST.md",
        )
        archive.write(
            repo / "experiments" / PROJECT_NAME / "VERIFY_BUNDLE.py",
            "VERIFY_BUNDLE.py",
        )
        archive.write(
            repo / "experiments" / PROJECT_NAME / "BUNDLE_CONTENTS.md",
            "PACKAGE_INDEX.md",
        )
        for arcname in ("README_FIRST.md", "VERIFY_BUNDLE.py", "PACKAGE_INDEX.md"):
            source = (
                repo / "experiments" / PROJECT_NAME /
                (
                    "LAB_DEPLOYMENT.md"
                    if arcname == "README_FIRST.md"
                    else "VERIFY_BUNDLE.py"
                    if arcname == "VERIFY_BUNDLE.py"
                    else "BUNDLE_CONTENTS.md"
                )
            )
            records.append(
                {"path": arcname, "bytes": source.stat().st_size, "sha256": sha256(source)}
            )
        manifest: dict[str, object] = {
            "schema_version": 1,
            "built_utc": datetime.now(timezone.utc).isoformat(),
            "project": PROJECT_NAME,
            "purpose": "simulation evidence plus six-pass four-stage laboratory deployment",
            "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
            "full_frame_cache_included": frame_cache is not None,
            "dataset_manifest_included": dataset_manifest is not None,
            "files": records,
            "total_bytes": sum(int(row["bytes"]) for row in records),
            "legacy_holoeye_or_dvp_included": False,
            "phase_slm_driver": "manual",
            "amplitude_slm_driver": "meadowlark_pcie",
            "camera_driver": "tucam",
        }
        archive.writestr(
            "BUNDLE_MANIFEST.json",
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        )
    temporary.replace(output)
    verification = verify_archive(output)
    result = {
        **manifest,
        "zip": str(output),
        "zip_bytes": output.stat().st_size,
        "zip_sha256": sha256(output),
        **verification,
    }
    output.with_suffix(output.suffix + ".sha256").write_text(
        f"{result['zip_sha256']}  {output.name}\n", encoding="ascii"
    )
    output.with_suffix(".report.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--frame-cache")
    parser.add_argument("--dataset-manifest")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = build(
        repo=Path(args.repo_root),
        checkpoint=Path(args.checkpoint),
        frame_cache=None if args.frame_cache is None else Path(args.frame_cache),
        dataset_manifest=(
            None if args.dataset_manifest is None else Path(args.dataset_manifest)
        ),
        output=Path(args.output),
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
