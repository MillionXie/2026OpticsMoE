"""Package an auditable T04 checkpoint for review (not a full dataset bundle)."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import zipfile
from pathlib import Path

from .build_lab_package import REPO, source_closure


def digest(path: Path) -> str:
    hash_ = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            hash_.update(chunk)
    return hash_.hexdigest()


def copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def build(run: Path, output: Path) -> dict:
    run = run.resolve()
    output = output.resolve()
    zip_path = output.with_suffix(".zip")
    if output.exists() or zip_path.exists():
        raise FileExistsError("Choose a new output path; do not overwrite a review release")
    checkpoint = run / "tested_checkpoints" / "epoch_070.pt"
    audit = json.loads((run / "audit_epoch070" / "audit.json").read_text())
    if audit["epoch"] != 70 or abs(audit["normal"]["overall"]["changed_cell_accuracy"] - .94) > 1e-8:
        raise ValueError("Unexpected selected epoch or score")
    if digest(checkpoint) != audit["checkpoint_sha256"]:
        raise ValueError("Audited checkpoint SHA256 mismatch")
    report = json.loads((run / "training_report.json").read_text())
    if report["selected_epoch"] != 70:
        raise ValueError("Training selection and independent audit disagree")

    output.mkdir(parents=True)
    sources = source_closure()
    for source in sources:
        copy(source, output / "source" / source.relative_to(REPO))
    copy(Path(__file__), output / "source" / Path(__file__).relative_to(REPO))
    copy(Path(__file__).parent / "configs" / "layered_scene_exp05_dc30_ccdsmall.yaml",
         output / "source" / "LightGenV2/tasks/t04_semantic_interaction/configs/layered_scene_exp05_dc30_ccdsmall.yaml")
    copy(Path(__file__).parent / "reports/reproduction/DC30_CCD_NOISE_RETRAIN_20260924.md",
         output / "README.md")
    copy(checkpoint, output / "weights" / "epoch_070_0p9400.pt")
    for relative in [
        "run_manifest.json", "resolved_config.json", "split_contract.json",
        "student_architecture.json", "training_report.json",
        "metrics/training_history.csv", "metrics/phase_training_audit.json",
        "audit_epoch070/audit.json", "audit_epoch070/normal_predictions.jsonl",
        "audit_epoch070/remove_optical_predictions.jsonl",
        "best_visualization/best_phase_overview.png",
        "best_visualization/phase_statistics.json",
    ]:
        copy(run / relative, output / "evidence" / relative)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    files = [
        {"path": item.relative_to(output).as_posix(), "bytes": item.stat().st_size, "sha256": digest(item)}
        for item in sorted(output.rglob("*")) if item.is_file()
    ]
    manifest = {
        "kind": "t04_dc30_ccdsmall_review",
        "source_commit": commit,
        "training_commit": json.loads((run / "run_manifest.json").read_text())["git_commit"],
        "selected_epoch": 70,
        "changed_cell_accuracy": .94,
        "checkpoint_sha256": audit["checkpoint_sha256"],
        "full_dataset_included": False,
        "qwen_model_included": False,
        "purpose": "Review real checkpoint, source, audit and predictions; not a standalone lab-control release",
        "files": files,
    }
    (output / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for item in sorted(output.rglob("*")):
            if item.is_file():
                archive.write(item, Path(output.name) / item.relative_to(output))
    return {"zip": str(zip_path), "bytes": zip_path.stat().st_size, "sha256": digest(zip_path),
            "files": len(files) + 1, "checkpoint_sha256": audit["checkpoint_sha256"]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.run_dir, args.output_dir), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
