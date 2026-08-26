"""One-command laboratory validation, acquisition, and MNIST-4 evaluation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

from experiments.hardware_sdk.workflows.acquire_folder import run as acquire_folder

from .ccd_evaluate import (
    DEFAULT_QC_MAX_SATURATION_FRACTION,
    DEFAULT_QC_MIN_MEAN_UINT8,
    DEFAULT_QC_MIN_REGION_RELATIVE_SPREAD,
    evaluate_directory,
    parse_roi,
)


PHASES = {"validate", "acquire", "evaluate", "all"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_stage(stage_dir: str | Path) -> dict[str, Any]:
    stage = Path(stage_dir).expanduser().resolve()
    manifest = stage / "samples.csv"
    if not manifest.is_file():
        raise FileNotFoundError(f"Stage manifest is missing: {manifest}")
    with manifest.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Stage manifest is empty: {manifest}")
    keys = [row.get("key", "") for row in rows]
    if any(not key for key in keys) or len(keys) != len(set(keys)):
        raise ValueError("samples.csv must contain unique, non-empty keys")
    amplitude_dir = stage / "amplitude_to_play"
    phase_dir = stage / "phase_to_play"
    amplitude_files = sorted(amplitude_dir.glob("*.bmp"))
    phase_files = sorted(phase_dir.glob("*.bmp"))
    expected_amplitudes = {f"{key}.bmp" for key in keys}
    actual_amplitudes = {path.name for path in amplitude_files}
    if actual_amplitudes != expected_amplitudes:
        missing = sorted(expected_amplitudes - actual_amplitudes)
        extra = sorted(actual_amplitudes - expected_amplitudes)
        raise RuntimeError(
            f"Stage amplitude/manifest mismatch; missing={missing[:3]}, "
            f"extra={extra[:3]}"
        )
    if len(phase_files) != 1:
        raise RuntimeError(
            f"Stage must contain exactly one phase BMP, found {len(phase_files)}"
        )
    profiles = sorted({row.get("profile", "unspecified") for row in rows})
    if len(profiles) != 1:
        raise RuntimeError(f"One stage must contain exactly one profile: {profiles}")
    contract_path = stage / "stage_contract.json"
    if not contract_path.is_file():
        raise FileNotFoundError(f"Stage contract is missing: {contract_path}")
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    if contract.get("profile") != profiles[0] or int(
        contract.get("samples", -1)
    ) != len(rows):
        raise RuntimeError("stage_contract.json does not match samples.csv")
    if not isinstance(contract.get("suitable_for_accuracy_reporting"), bool):
        raise RuntimeError(
            "stage_contract.json must contain boolean "
            "suitable_for_accuracy_reporting"
        )
    phase_sha256 = _sha256(phase_files[0])
    if contract.get("phase_sha256") != phase_sha256:
        raise RuntimeError("Phase BMP SHA-256 does not match stage_contract.json")
    for row in rows:
        amplitude_path = amplitude_dir / f"{row['key']}.bmp"
        if row.get("amplitude_sha256") != _sha256(amplitude_path):
            raise RuntimeError(
                f"Amplitude BMP SHA-256 mismatch: {amplitude_path.name}"
            )
        if row.get("phase_sha256") != phase_sha256:
            raise RuntimeError(
                f"Manifest phase SHA-256 mismatch for key={row['key']}"
            )
    return {
        "stage_dir": str(stage),
        "profile": profiles[0],
        "samples": len(rows),
        "manifest": str(manifest),
        "phase_mask": str(phase_files[0]),
        "phase_sha256": phase_sha256,
        "suitable_for_accuracy_reporting": contract[
            "suitable_for_accuracy_reporting"
        ],
        "amplitude_dir": str(amplitude_dir),
        "ccd_dir": str(stage / "ccd_captured"),
    }


def run_pipeline(
    *,
    phase: str,
    stage_dir: str | Path,
    hardware_config: str | Path,
    model_config: str | Path | None = None,
    roi: tuple[int, int, int, int] | None = None,
    flip_vertical: bool = False,
    flip_horizontal: bool = False,
    clear_output: bool = False,
    assume_yes: bool = False,
    allow_biased_demo_metric: bool = False,
    allow_invalid_formal: bool = False,
    qc_min_mean_uint8: float = DEFAULT_QC_MIN_MEAN_UINT8,
    qc_max_saturation_fraction: float = DEFAULT_QC_MAX_SATURATION_FRACTION,
    qc_min_region_relative_spread: float = DEFAULT_QC_MIN_REGION_RELATIVE_SPREAD,
) -> dict[str, Any]:
    if phase not in PHASES:
        raise ValueError(f"phase must be one of {sorted(PHASES)}, got {phase!r}")
    stage = Path(stage_dir).expanduser().resolve()
    hardware_config_path = Path(hardware_config).expanduser().resolve()
    stage_report = validate_stage(stage)
    if (
        phase in {"evaluate", "all"}
        and not stage_report["suitable_for_accuracy_reporting"]
        and not allow_biased_demo_metric
    ):
        raise PermissionError(
            "The demo profile is deliberately preselected and cannot produce an "
            "accuracy metric. Use --allow-biased-demo-metric only for a diagnostic "
            "demo_success_rate."
        )
    results: dict[str, Any] = {"stage": stage_report}

    if phase in {"validate", "all"}:
        results["device_validation"] = acquire_folder(
            hardware_config_path,
            stage_override=stage,
            validate_only=True,
        )
    if phase in {"acquire", "all"}:
        results["acquisition"] = acquire_folder(
            hardware_config_path,
            stage_override=stage,
            clear_output=clear_output,
            assume_yes=assume_yes,
            validate_only=False,
        )
    if phase in {"evaluate", "all"}:
        resolved_model_config = (
            Path(model_config).expanduser().resolve()
            if model_config is not None
            else (stage.parent / "lab_model_config.yaml").resolve()
        )
        if not resolved_model_config.is_file():
            raise FileNotFoundError(
                "Standalone model geometry config is missing: "
                f"{resolved_model_config}"
            )
        results["evaluation"] = evaluate_directory(
            config=resolved_model_config,
            manifest=stage / "samples.csv",
            ccd_dir=stage / "ccd_captured",
            output_dir=stage / "hardware_evaluation",
            roi=roi,
            flip_vertical=flip_vertical,
            flip_horizontal=flip_horizontal,
            stage_contract=stage / "stage_contract.json",
            capture_manifest=stage / "acquisition_logs" / "capture_manifest.csv",
            allow_biased_demo_metric=allow_biased_demo_metric,
            allow_invalid_formal=allow_invalid_formal,
            qc_min_mean_uint8=qc_min_mean_uint8,
            qc_max_saturation_fraction=qc_max_saturation_fraction,
            qc_min_region_relative_spread=qc_min_region_relative_spread,
        )
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Validate the fixed phase/stage, play Meadowlark amplitude BMPs, "
            "capture TUCam frames, and evaluate MNIST-4 accuracy"
        )
    )
    parser.add_argument("--phase", choices=sorted(PHASES), default="all")
    parser.add_argument("--stage-dir", required=True)
    parser.add_argument(
        "--hardware-config",
        default=str(Path(__file__).with_name("lab_hardware_config.yaml")),
    )
    parser.add_argument(
        "--model-config",
        default=None,
        help="Defaults to <stage-parent>/lab_model_config.yaml",
    )
    parser.add_argument("--roi", default=None, help="left,top,right,bottom")
    parser.add_argument("--flip-vertical", action="store_true")
    parser.add_argument("--flip-horizontal", action="store_true")
    parser.add_argument("--clear-output", action="store_true")
    parser.add_argument("--yes", action="store_true")
    parser.add_argument("--allow-biased-demo-metric", action="store_true")
    parser.add_argument("--allow-invalid-formal", action="store_true")
    parser.add_argument(
        "--qc-min-mean-uint8", type=float, default=DEFAULT_QC_MIN_MEAN_UINT8
    )
    parser.add_argument(
        "--qc-max-saturation-fraction",
        type=float,
        default=DEFAULT_QC_MAX_SATURATION_FRACTION,
    )
    parser.add_argument(
        "--qc-min-region-relative-spread",
        type=float,
        default=DEFAULT_QC_MIN_REGION_RELATIVE_SPREAD,
    )
    args = parser.parse_args(argv)
    report = run_pipeline(
        phase=args.phase,
        stage_dir=args.stage_dir,
        hardware_config=args.hardware_config,
        model_config=args.model_config,
        roi=parse_roi(args.roi),
        flip_vertical=args.flip_vertical,
        flip_horizontal=args.flip_horizontal,
        clear_output=args.clear_output,
        assume_yes=args.yes,
        allow_biased_demo_metric=args.allow_biased_demo_metric,
        allow_invalid_formal=args.allow_invalid_formal,
        qc_min_mean_uint8=args.qc_min_mean_uint8,
        qc_max_saturation_fraction=args.qc_max_saturation_fraction,
        qc_min_region_relative_spread=args.qc_min_region_relative_spread,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
