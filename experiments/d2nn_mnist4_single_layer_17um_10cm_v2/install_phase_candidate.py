"""Install one audited MNIST-4 phase BMP into an existing lab payload."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

from PIL import Image


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def install_candidate(
    *,
    bundle_root: str | Path,
    phase_bmp: str | Path,
    name: str,
    training_summary: str | Path | None = None,
) -> dict[str, Any]:
    if not name or Path(name).name != name or any(char in name for char in " /\\"):
        raise ValueError("Candidate name must be one safe filename component")
    root = Path(bundle_root).expanduser().resolve()
    source = Path(phase_bmp).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Phase BMP is missing: {source}")
    with Image.open(source) as opened:
        opened.load()
        if opened.format != "BMP" or opened.mode != "L" or opened.size != (1920, 1200):
            raise RuntimeError(
                f"Phase candidate must be a 1920x1200 L-mode BMP: {source}"
            )
    phase_directory = root / "payload" / "phase_masks"
    manifest_path = phase_directory / "phase_masks.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Lab phase manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    candidates = manifest.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise RuntimeError("Lab phase manifest has no baseline candidates")
    digest = _sha256(source)
    filename = f"{name}_1920x1200.bmp"
    destination = phase_directory / filename
    existing = next((row for row in candidates if row.get("name") == name), None)
    if existing is not None:
        if existing.get("sha256") != digest or not destination.is_file():
            raise RuntimeError(
                f"Candidate {name!r} already exists with different evidence"
            )
        return {"status": "already_installed", "candidate": existing}
    if destination.exists():
        raise FileExistsError(f"Refusing to replace an unregistered phase BMP: {destination}")
    reference = candidates[0]
    summary: dict[str, Any] = {}
    if training_summary is not None:
        summary_path = Path(training_summary).expanduser().resolve()
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    shutil.copy2(source, destination)
    if _sha256(destination) != digest:
        destination.unlink(missing_ok=True)
        raise RuntimeError("Phase BMP changed while copying into the lab payload")
    row = {
        "name": name,
        "file": filename,
        "sha256": digest,
        "source": str(source),
        "epoch": summary.get("best_epoch"),
        "validation_accuracy": summary.get("best_validation_accuracy"),
        "robust_validation_accuracy": summary.get(
            "best_robust_validation_accuracy"
        ),
        "selection_metric": summary.get("selection_metric"),
        "phase_std_rad": (summary.get("phase_statistics") or {}).get(
            "phase_std_rad"
        ),
        "phase_bounds_xyxy": reference["phase_bounds_xyxy"],
        "actual_phase_center_xy": reference["actual_phase_center_xy"],
    }
    candidates.append(row)
    temporary = manifest_path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(manifest_path)
    return {"status": "installed", "candidate": row, "manifest": str(manifest_path)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-root", required=True)
    parser.add_argument("--phase-bmp", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--training-summary")
    args = parser.parse_args()
    report = install_candidate(
        bundle_root=args.bundle_root,
        phase_bmp=args.phase_bmp,
        name=args.name,
        training_summary=args.training_summary,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
