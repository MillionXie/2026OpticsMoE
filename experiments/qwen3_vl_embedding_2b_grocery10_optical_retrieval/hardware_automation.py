"""Interactive plane-first SLM -> CCD -> electronic bridge automation."""

from __future__ import annotations

import argparse
import csv
import json
import time
from contextlib import ExitStack
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import yaml

from .hardware_devices import build_camera, build_slm
from .hardware_pipeline import (
    STAGES,
    Runtime,
    build_runtime,
    close_runtime,
    load_captured_intensity,
    load_hardware_config,
    prepare,
    process_language_expert,
    process_language_global,
    process_vision_expert,
    process_vision_global,
)
from .io_utils import seed_everything, write_csv, write_json


STAGE_ORDER = ("vision_expert", "vision_global", "language_expert", "language_global")
PROCESSORS: dict[str, Callable[..., Any]] = {
    "vision_expert": process_vision_expert,
    "vision_global": process_vision_global,
    "language_expert": process_language_expert,
    "language_global": process_language_global,
}


@dataclass(frozen=True)
class AutomationConfig:
    config_path: Path
    settle_delay_seconds: float
    confirm_each_phase_mask: bool
    resume_existing_captures: bool
    comparison_normalization: str
    comparison_preview_count: int
    phase_slm: dict[str, Any]
    amplitude_slm: dict[str, Any]
    camera: dict[str, Any]
    output_extension: str


def load_automation_config(path: str | Path) -> AutomationConfig:
    config_path = Path(path).expanduser().resolve()
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    automation = raw.get("automation", {})
    devices = automation.get("devices", {})
    camera = dict(devices.get("camera", {}))
    extension = str(camera.pop("output_extension", ".npy")).lower()
    if extension not in {".npy", ".tif", ".tiff", ".png"}:
        raise ValueError("automation camera output_extension must be lossless .npy/.tif/.tiff/.png")
    normalization = str(automation.get("comparison_normalization", "mean"))
    if normalization not in {"mean", "energy", "none"}:
        raise ValueError("comparison_normalization must be mean, energy, or none")
    delay_ms = float(automation.get("settle_delay_ms", 40.0))
    if delay_ms < 0:
        raise ValueError("settle_delay_ms cannot be negative")
    return AutomationConfig(
        config_path=config_path,
        settle_delay_seconds=delay_ms / 1000.0,
        confirm_each_phase_mask=bool(automation.get("confirm_each_phase_mask", True)),
        resume_existing_captures=bool(automation.get("resume_existing_captures", True)),
        comparison_normalization=normalization,
        comparison_preview_count=int(automation.get("comparison_preview_count", 8)),
        phase_slm=dict(devices.get("phase_slm", {"driver": "manual"})),
        amplitude_slm=dict(devices.get("amplitude_slm", {"driver": "manual"})),
        camera=camera,
        output_extension=extension,
    )


def normalized_ccd_comparison(
    measured: torch.Tensor,
    theoretical: torch.Tensor,
    normalization: str = "mean",
) -> dict[str, float]:
    measured = measured.detach().cpu().float()
    theoretical = theoretical.detach().cpu().float()
    if measured.shape != theoretical.shape:
        raise ValueError(f"CCD comparison shape mismatch: {tuple(measured.shape)} vs {tuple(theoretical.shape)}")
    eps = 1.0e-12
    if normalization == "mean":
        measured_norm = measured / measured.mean().clamp_min(eps)
        theoretical_norm = theoretical / theoretical.mean().clamp_min(eps)
    elif normalization == "energy":
        measured_norm = measured / measured.square().sum().sqrt().clamp_min(eps)
        theoretical_norm = theoretical / theoretical.square().sum().sqrt().clamp_min(eps)
    elif normalization == "none":
        measured_norm, theoretical_norm = measured, theoretical
    else:
        raise ValueError(f"Unknown normalization {normalization!r}")
    difference = measured_norm - theoretical_norm
    left = measured.flatten() - measured.mean()
    right = theoretical.flatten() - theoretical.mean()
    pcc_denominator = left.square().sum().sqrt() * right.square().sum().sqrt()
    pcc = float((left * right).sum() / pcc_denominator.clamp_min(eps))
    return {
        "normalized_mse": float(difference.square().mean()),
        "normalized_mae": float(difference.abs().mean()),
        "pcc": pcc,
        "measured_mean": float(measured.mean()),
        "theoretical_mean": float(theoretical.mean()),
        "measured_max": float(measured.max()),
        "theoretical_max": float(theoretical.max()),
    }


def _manifest_keys(root: Path) -> list[str]:
    with (root / "00_manifest" / "play_order.csv").open("r", encoding="utf-8", newline="") as handle:
        return [row["sample_key"] for row in csv.DictReader(handle)]


def _single_bmp(directory: Path) -> Path:
    values = sorted(directory.glob("*.bmp"))
    if len(values) != 1:
        raise RuntimeError(f"Expected exactly one shared phase BMP in {directory}, found {len(values)}")
    return values[0]


def _confirm_phase(stage: str, mask: Path) -> None:
    while True:
        answer = input(
            f"\n[{stage}] 请准备并确认相位 mask：\n  {mask}\n"
            "输入 y 开始播放本层全部振幅；输入 q 安全退出： "
        ).strip().lower()
        if answer in {"y", "yes"}:
            return
        if answer in {"q", "quit", "n", "no"}:
            raise KeyboardInterrupt("operator stopped before exposure")


def _existing_capture(directory: Path, key: str) -> Path | None:
    matches = [candidate for suffix in (".npy", ".tif", ".tiff", ".png", ".pt") if (candidate := directory / f"{key}{suffix}").is_file()]
    if len(matches) > 1:
        raise RuntimeError(f"Multiple captured files exist for {key}: {matches}")
    return matches[0] if matches else None


def _capture_stage(
    runtime: Runtime,
    stage: str,
    automation: AutomationConfig,
    phase_slm: Any,
    amplitude_slm: Any,
    camera: Any,
    *,
    assume_yes: bool,
) -> None:
    stage_dir = runtime.hardware.output_dir / STAGES[stage]
    phase = _single_bmp(runtime.hardware.output_dir / "00_masks" / STAGES[stage])
    phase_slm.display_file(phase)
    if automation.confirm_each_phase_mask and not assume_yes:
        _confirm_phase(stage, phase)
    amplitudes = stage_dir / "amplitude_to_play"
    captures = stage_dir / "ccd_captured"
    keys = _manifest_keys(runtime.hardware.output_dir)
    amplitude_files = [amplitudes / f"{key}.bmp" for key in keys]
    missing = [path for path in amplitude_files if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"{stage} is missing {len(missing)} amplitude frames; first: {missing[0]}"
        )
    amplitude_slm.preload_files(amplitude_files)
    for index, key in enumerate(keys, 1):
        amplitude = amplitudes / f"{key}.bmp"
        if not amplitude.is_file():
            raise FileNotFoundError(
                f"Stage input is missing: {amplitude}. The preceding electronic bridge did not finish."
            )
        existing = _existing_capture(captures, key)
        if existing is not None and automation.resume_existing_captures:
            print(f"[{stage}] capture {index}/{len(keys)} already exists: {existing.name}")
            continue
        amplitude_slm.display_file(amplitude)
        time.sleep(automation.settle_delay_seconds)
        destination = captures / f"{key}{automation.output_extension}"
        camera.capture(destination)
        print(f"[{stage}] captured {index}/{len(keys)} -> {destination.name}")


def _comparison_preview(measured: torch.Tensor, theoretical: torch.Tensor, path: Path, title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 3, figsize=(14, 4.6), constrained_layout=True)
    difference = measured.float() / measured.mean().clamp_min(1e-12) - theoretical.float() / theoretical.mean().clamp_min(1e-12)
    for axis, value, name, cmap in (
        (axes[0], measured, "measured intensity", "viridis"),
        (axes[1], theoretical, "theoretical intensity", "viridis"),
        (axes[2], difference, "mean-normalized difference", "coolwarm"),
    ):
        shown = axis.imshow(value.cpu().numpy(), cmap=cmap)
        figure.colorbar(shown, ax=axis)
        axis.set_title(name); axis.set_xlabel("x pixel"); axis.set_ylabel("y pixel")
    figure.suptitle(title)
    figure.savefig(path, dpi=140)
    plt.close(figure)


def _compare_stage(runtime: Runtime, stage: str, automation: AutomationConfig, *, use_simulation: bool) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for index, key in enumerate(_manifest_keys(runtime.hardware.output_dir)):
        theoretical = load_captured_intensity(runtime, stage, key, use_simulation=True)
        measured = theoretical if use_simulation else load_captured_intensity(runtime, stage, key, use_simulation=False)
        metrics = normalized_ccd_comparison(measured, theoretical, automation.comparison_normalization)
        rows.append({"sample_key": key, **metrics})
        if index < automation.comparison_preview_count:
            _comparison_preview(
                measured,
                theoretical,
                runtime.hardware.output_dir / STAGES[stage] / "comparison" / f"{key}.png",
                f"{stage}: {key}",
            )
    summary = {
        "stage": stage,
        "sample_count": len(rows),
        "comparison_normalization": automation.comparison_normalization,
        "mean_normalized_mse": sum(row["normalized_mse"] for row in rows) / len(rows),
        "mean_pcc": sum(row["pcc"] for row in rows) / len(rows),
    }
    write_csv(
        runtime.hardware.output_dir / STAGES[stage] / "comparison" / "ccd_vs_theory.csv",
        rows,
        list(rows[0]),
    )
    write_json(runtime.hardware.output_dir / STAGES[stage] / "comparison" / "summary.json", summary)
    print(f"[{stage}] CCD/theory normalized MSE={summary['mean_normalized_mse']:.6g}, PCC={summary['mean_pcc']:.5f}")
    return summary


def _plot_confusion(root: Path, class_names: list[str]) -> Path:
    source = root / "05_retrieval" / "confusion_matrix.csv"
    with source.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    matrix = torch.tensor([[int(row[name]) for name in class_names] for row in rows])
    output = root / "05_retrieval" / "confusion_matrix.png"
    figure, axis = plt.subplots(figsize=(10, 9), constrained_layout=True)
    shown = axis.imshow(matrix.numpy(), cmap="Blues")
    figure.colorbar(shown, ax=axis, label="query count")
    axis.set_xticks(range(len(class_names)), class_names, rotation=55, ha="right")
    axis.set_yticks(range(len(class_names)), class_names)
    axis.set_xlabel("retrieved SKU"); axis.set_ylabel("true SKU")
    axis.set_title("Physical optical retrieval confusion matrix")
    for row in range(len(class_names)):
        for col in range(len(class_names)):
            axis.text(col, row, str(int(matrix[row, col])), ha="center", va="center", fontsize=7)
    figure.savefig(output, dpi=160)
    plt.close(figure)
    return output


def run(
    config_path: str | Path,
    *,
    session_dir: Path | None,
    simulate: bool,
    skip_prepare: bool,
    assume_yes: bool,
    sample_limit: int | None,
) -> dict[str, Any]:
    automation = load_automation_config(config_path)
    hardware = load_hardware_config(config_path)
    hardware = replace(
        hardware,
        output_dir=(session_dir.expanduser().resolve() if session_dir is not None else hardware.output_dir),
        minimal_artifacts=False,
        sample_limit=sample_limit,
        clean_output_before_prepare=(hardware.clean_output_before_prepare and not skip_prepare),
    )
    seed_everything(42)
    runtime = build_runtime(hardware)
    stage_summaries: list[dict[str, Any]] = []
    try:
        deployment = hardware.output_dir / "00_manifest" / "deployment.json"
        if not skip_prepare or not deployment.is_file():
            prepare(runtime)
        if simulate:
            for stage in STAGE_ORDER:
                stage_summaries.append(_compare_stage(runtime, stage, automation, use_simulation=True))
                result = PROCESSORS[stage](runtime, use_simulation=True)
        else:
            with ExitStack() as stack:
                phase_slm = stack.enter_context(build_slm(automation.phase_slm, automation.config_path.parent))
                amplitude_slm = stack.enter_context(build_slm(automation.amplitude_slm, automation.config_path.parent))
                camera = stack.enter_context(build_camera(automation.camera, automation.config_path.parent))
                write_json(
                    hardware.output_dir / "00_manifest" / "resolved_hardware_devices.json",
                    {
                        "phase_slm": phase_slm.device_info(),
                        "amplitude_slm": amplitude_slm.device_info(),
                        "camera": camera.device_info(),
                        "settle_delay_ms": automation.settle_delay_seconds * 1000.0,
                        "note": (
                            "Explicit YAML camera settings override a loaded vendor config. "
                            "A stale frame is discarded before every saved frame when configured."
                        ),
                    },
                )
                result = None
                for stage in STAGE_ORDER:
                    _capture_stage(runtime, stage, automation, phase_slm, amplitude_slm, camera, assume_yes=assume_yes)
                    stage_summaries.append(_compare_stage(runtime, stage, automation, use_simulation=False))
                    result = PROCESSORS[stage](runtime, use_simulation=False)
        if not isinstance(result, dict):
            raise RuntimeError("Final language-global processing did not return retrieval metrics")
        confusion = _plot_confusion(hardware.output_dir, list(runtime.bundle.class_names))
        summary = {
            "mode": "simulation" if simulate else "physical_hardware",
            "output_dir": str(hardware.output_dir),
            "settle_delay_ms": automation.settle_delay_seconds * 1000.0,
            "stage_comparisons": stage_summaries,
            "retrieval": result,
            "confusion_matrix_png": str(confusion),
        }
        write_json(hardware.output_dir / "05_retrieval" / "automation_summary.json", summary)
        return summary
    finally:
        close_runtime(runtime)


def main() -> int:
    parser = argparse.ArgumentParser(description="Automated four-plane Grocery optical acquisition and electronic replay")
    parser.add_argument("--config", required=True)
    parser.add_argument("--session-dir", type=Path, default=None)
    parser.add_argument("--simulate", action="store_true", help="Use theoretical CCD tensors; no SDK or operator prompt")
    parser.add_argument("--skip-prepare", action="store_true", help="Reuse an already prepared session directory")
    parser.add_argument("--yes", action="store_true", help="Skip phase-mask confirmation (not recommended for physical runs)")
    parser.add_argument("--sample-limit", type=int, default=None)
    args = parser.parse_args()
    if args.sample_limit is not None and args.sample_limit <= 0:
        parser.error("--sample-limit must be positive")
    summary = run(
        args.config,
        session_dir=args.session_dir,
        simulate=args.simulate,
        skip_prepare=args.skip_prepare,
        assume_yes=args.yes,
        sample_limit=args.sample_limit,
    )
    metrics = summary["retrieval"]
    print(
        "Complete: "
        f"Top-1={metrics['top1_retrieval_accuracy']:.4f} "
        f"Top-3={metrics['top3_retrieval_accuracy']:.4f} MRR={metrics['mrr']:.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
