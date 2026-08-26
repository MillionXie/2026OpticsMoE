"""Build one transferable ZIP containing payload, light runtime, and vendor SDKs."""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path
from typing import Iterable

PROJECT_PACKAGE = "d2nn_mnist4_single_layer_17um_10cm"
DEFAULT_ZIP_NAME = "mnist4_single_layer_17um_10cm_lab_bundle.zip"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _payload_files(export_dir: Path) -> Iterable[Path]:
    root_names = {
        "README_LAB.md",
        "detector_regions.csv",
        "detector_roi_478.png",
        "hardware_contract.json",
        "lab_model_config.yaml",
    }
    for name in sorted(root_names):
        path = export_dir / name
        if not path.is_file():
            raise FileNotFoundError(f"Required exported payload is missing: {path}")
        yield path
    canonical_phase = export_dir / "phase_to_play"
    for path in sorted(canonical_phase.glob("*.bmp")):
        yield path
    contract = json.loads(
        (export_dir / "hardware_contract.json").read_text(encoding="utf-8")
    )
    profiles = contract.get("profiles")
    if not isinstance(profiles, dict) or len(profiles) != 2:
        raise ValueError("hardware_contract.json must define two export profiles")
    for profile in sorted(profiles):
        stage = export_dir / profile
        for name in ("README.md", "samples.csv", "stage_contract.json"):
            path = stage / name
            if not path.is_file():
                raise FileNotFoundError(f"Required stage file is missing: {path}")
            yield path
        for folder in ("amplitude_to_play", "phase_to_play"):
            files = sorted((stage / folder).glob("*.bmp"))
            if not files:
                raise FileNotFoundError(f"No BMP files found under {stage / folder}")
            yield from files


def _runtime_files(repo_root: Path, include_vendor_sdk: bool) -> Iterable[Path]:
    relative_files = [
        "experiments/__init__.py",
        f"experiments/{PROJECT_PACKAGE}/__init__.py",
        f"experiments/{PROJECT_PACKAGE}/ccd_evaluate.py",
        f"experiments/{PROJECT_PACKAGE}/lab_pipeline.py",
        f"experiments/{PROJECT_PACKAGE}/settings.py",
        f"experiments/{PROJECT_PACKAGE}/lab_hardware_config.yaml",
        f"experiments/{PROJECT_PACKAGE}/requirements-lab.txt",
        f"experiments/{PROJECT_PACKAGE}/HARDWARE_LAB.md",
        f"experiments/{PROJECT_PACKAGE}/RUN_COMMANDS.md",
        "experiments/hardware_sdk/__init__.py",
        "experiments/hardware_sdk/devices.py",
        "experiments/hardware_sdk/workflows/__init__.py",
        "experiments/hardware_sdk/workflows/acquire_folder.py",
        "experiments/hardware_sdk/workflows/calibration_common.py",
        "experiments/hardware_sdk/drivers/__init__.py",
        "experiments/hardware_sdk/drivers/meadowlark_pcie_slm.py",
        "experiments/hardware_sdk/drivers/tucam_camera.py",
    ]
    for relative in relative_files:
        path = repo_root / relative
        if not path.is_file():
            raise FileNotFoundError(f"Required laboratory runtime file is missing: {path}")
        yield path
    if include_vendor_sdk:
        for relative in (
            "experiments/hardware_sdk/vendor_sdk/amplitude_meadowlark",
            "experiments/hardware_sdk/vendor_sdk/camera_tucam_mosaic",
        ):
            directory = repo_root / relative
            if not directory.is_dir():
                raise FileNotFoundError(f"Vendor SDK directory is missing: {directory}")
            for path in sorted(directory.rglob("*")):
                if path.is_file():
                    yield path


def create_lab_zip(
    *,
    export_dir: str | Path,
    output_path: str | Path,
    include_vendor_sdk: bool = True,
    repo_root: str | Path | None = None,
) -> dict[str, object]:
    export_root = Path(export_dir).expanduser().resolve()
    output = Path(output_path).expanduser().resolve()
    root = (
        Path(repo_root).expanduser().resolve()
        if repo_root is not None
        else Path(__file__).resolve().parents[2]
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = list(dict.fromkeys(_payload_files(export_root)))
    runtime = list(dict.fromkeys(_runtime_files(root, include_vendor_sdk)))
    with zipfile.ZipFile(
        output,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=6,
    ) as archive:
        for path in payload:
            archive.write(path, (Path("payload") / path.relative_to(export_root)).as_posix())
        for path in runtime:
            archive.write(path, path.relative_to(root).as_posix())
    report = {
        "zip": str(output),
        "sha256": _sha256(output),
        "size_bytes": output.stat().st_size,
        "payload_files": len(payload),
        "runtime_files": len(runtime),
        "vendor_sdk_included": bool(include_vendor_sdk),
    }
    report_path = output.with_suffix(output.suffix + ".json")
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Package the MNIST-4 10 cm laboratory payload and runtime"
    )
    parser.add_argument("--export-dir", required=True)
    parser.add_argument("--output", default=DEFAULT_ZIP_NAME)
    parser.add_argument(
        "--omit-vendor-sdk",
        action="store_true",
        help="Create a small developer archive; formal lab delivery should include SDKs",
    )
    args = parser.parse_args(argv)
    report = create_lab_zip(
        export_dir=args.export_dir,
        output_path=args.output,
        include_vendor_sdk=not args.omit_vendor_sdk,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
