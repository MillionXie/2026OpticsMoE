"""Validate, acquire, evaluate, and plot one prepared MNIST-4 lab session."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

from experiments.hardware_sdk.workflows.acquire_folder import run as acquire_folder

from .ccd_evaluate import evaluate_directory
from .lab_session import _default_bundle_root, validate_session
from .simulation_agreement import evaluate_agreement


PHASES = {"validate", "acquire", "evaluate", "agreement", "all"}


def run_pipeline(
    *,
    phase: str,
    stage_dir: str | Path,
    hardware_config: str | Path,
    bundle_root: str | Path | None = None,
    clear_output: bool = False,
    assume_yes: bool = False,
    allow_quick40_diagnostic: bool = False,
    allow_invalid_formal: bool = False,
    flip_vertical: bool = False,
    flip_horizontal: bool = False,
    device: str = "auto",
    batch_size: int = 4,
) -> dict[str, Any]:
    if phase not in PHASES:
        raise ValueError(f"phase must be one of {sorted(PHASES)}")
    stage = Path(stage_dir).expanduser().resolve()
    stage_report = validate_session(stage)
    if (
        phase in {"evaluate", "all"}
        and stage_report["profile"] == "quick40"
        and not allow_quick40_diagnostic
    ):
        raise PermissionError(
            "quick40 is an alignment/exposure diagnostic, never formal accuracy. "
            "Pass --allow-quick40-diagnostic to generate its clearly labelled report."
        )
    root = (
        Path(bundle_root).expanduser().resolve()
        if bundle_root is not None
        else _default_bundle_root()
    )
    results: dict[str, Any] = {"stage": stage_report}
    config = Path(hardware_config).expanduser().resolve()
    if phase in {"validate", "all"}:
        results["device_validation"] = acquire_folder(
            config, stage_override=stage, validate_only=True
        )
    if phase in {"acquire", "all"}:
        results["acquisition"] = acquire_folder(
            config,
            stage_override=stage,
            clear_output=clear_output,
            assume_yes=assume_yes,
            validate_only=False,
        )
    if phase in {"evaluate", "all"}:
        evaluation_output = stage / "hardware_evaluation"
        if clear_output and evaluation_output.exists():
            # ``stage`` has already passed the ownership/hash validation above;
            # this removes generated evaluation artifacts only, never captures,
            # amplitudes, masks, or the session contract.
            shutil.rmtree(evaluation_output)
        results["evaluation"] = evaluate_directory(
            config=root / "payload" / "model" / "lab_model_config.yaml",
            manifest=stage / "samples.csv",
            ccd_dir=stage / "ccd_captured",
            output_dir=evaluation_output,
            flip_vertical=flip_vertical,
            flip_horizontal=flip_horizontal,
            allow_biased_demo_metric=allow_quick40_diagnostic,
            allow_invalid_formal=allow_invalid_formal,
            generate_paper_report=True,
        )
    if phase in {"agreement", "all"}:
        results["agreement"] = evaluate_agreement(
            stage_dir=stage,
            bundle_root=root,
            device=device,
            batch_size=batch_size,
        )
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=sorted(PHASES), default="all")
    parser.add_argument("--stage-dir", required=True)
    parser.add_argument(
        "--hardware-config",
        default=str(
            Path(__file__).resolve().parents[1]
            / "lab_qwen"
            / "generated"
            / "formal_hardware.yaml"
        ),
    )
    parser.add_argument("--bundle-root", default=None)
    parser.add_argument("--clear-output", action="store_true")
    parser.add_argument("--yes", action="store_true")
    parser.add_argument("--allow-quick40-diagnostic", action="store_true")
    parser.add_argument(
        "--allow-invalid-formal",
        action="store_true",
        help="Developer diagnosis only; failed-QC formal accuracy remains non-reportable",
    )
    parser.add_argument("--flip-vertical", action="store_true")
    parser.add_argument("--flip-horizontal", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args(argv)
    report = run_pipeline(
        phase=args.phase,
        stage_dir=args.stage_dir,
        hardware_config=args.hardware_config,
        bundle_root=args.bundle_root,
        clear_output=args.clear_output,
        assume_yes=args.yes,
        allow_quick40_diagnostic=args.allow_quick40_diagnostic,
        allow_invalid_formal=args.allow_invalid_formal,
        flip_vertical=args.flip_vertical,
        flip_horizontal=args.flip_horizontal,
        device=args.device,
        batch_size=args.batch_size,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
