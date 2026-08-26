from __future__ import annotations

import argparse
import platform
import sys
from pathlib import Path

import torch

from .data import build_datasets, build_loaders
from .hardware_export import export_hardware_bundle
from .io_utils import set_seed, write_json
from .modeling import RobustRawCCDMNIST4D2NN
from .settings import load_settings
from .training import evaluate, load_checkpoint, train_model


PHASES = {"train", "test", "export", "all"}


def _device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return torch.device(requested)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Train/test/export the robust raw-CCD MNIST-4 D2NN v2"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--phase", choices=sorted(PHASES), default="all")
    parser.add_argument("--checkpoint", default=None)
    args = parser.parse_args(argv)

    settings = load_settings(args.config)
    set_seed(settings.random_seed)
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(settings.output_dir / "resolved_config.json", settings.to_dict())
    write_json(
        settings.output_dir / "environment.json",
        {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device": (
                torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
            ),
        },
    )
    bundle = build_datasets(settings)
    write_json(settings.output_dir / "dataset.json", bundle.metadata)
    train_loader, validation_loader, test_loader = build_loaders(bundle, settings)
    device = _device(settings.device)
    model = RobustRawCCDMNIST4D2NN(settings).to(device)
    write_json(
        settings.output_dir / "model.json",
        {
            "architecture": (
                "co-planar amplitude + one phase mask + k-space cutoff + "
                "10 cm padded ASM + untouched 478x478 raw CCD"
            ),
            "electronic_trainable_parameters": 0,
            "optical_trainable_parameters": model.raw_phase.numel(),
            "raw_phase_initialization": 0.0,
            "initial_actual_phase_rad": float(torch.pi),
            "phase_parameterization": "2*pi*sigmoid(raw_phase)",
            "detector_bounds_xyxy": [
                list(value) for value in settings.detector_bounds()
            ],
            "notebook_geometry_scale": settings.active_size
            / settings.detector_reference_grid_size,
            "loss": {
                "mode": settings.loss_mode,
                "notebook_full_plane_mse_scale": (
                    settings.notebook_full_plane_mse_scale
                ),
                "detector_ce_weight": settings.detector_ce_loss_weight,
                "detector_ce_scope": (
                    "training objective only; raw ROI-energy argmax at inference"
                ),
                "formal_objective": (
                    "scale * mean((raw_ccd_intensity - binary_target_template)^2)"
                    if settings.loss_mode == "notebook_full_plane_mse"
                    else "legacy separately averaged target/background MSE"
                ),
                "target_region_mse": "diagnostic only in notebook mode",
                "background_mse": "diagnostic only in notebook mode",
            },
            "ccd_postprocess": {
                "normalization": False,
                "nonlinearity": False,
                "background_subtraction": False,
                "classification": "four raw region sums then argmax",
            },
            "k_space_pass_fraction": model.propagator.pass_fraction,
            "robustness": {
                "warmup_epochs": settings.robustness_warmup_epochs,
                "probability_after_warmup": settings.robustness_probability,
                "input_shift_max_px": settings.input_shift_max_px,
                "phase_shift_max_px": settings.phase_shift_max_px,
                "pre_ccd_shift_max_px": settings.pre_ccd_shift_max_px,
            },
        },
    )
    print(
        f"device={device} train/val/test={len(bundle.train)}/"
        f"{len(bundle.validation)}/{len(bundle.test)} "
        f"raw_phase={tuple(model.raw_phase.shape)} "
        f"detectors={settings.detector_bounds()} "
        f"k_pass={model.propagator.pass_fraction:.4f}",
        flush=True,
    )
    checkpoint = (
        Path(args.checkpoint).expanduser().resolve()
        if args.checkpoint
        else settings.output_dir / "checkpoints" / "best.pt"
    )
    if args.phase in {"train", "all"}:
        train_model(model, train_loader, validation_loader, settings, device)
    elif args.phase in {"test", "export"}:
        load_checkpoint(checkpoint, model, device)

    if args.phase in {"test", "all"}:
        test_metrics = evaluate(model, test_loader, device)
        test_metrics["checkpoint"] = str(checkpoint)
        test_metrics["phase_statistics"] = model.phase_statistics()
        write_json(settings.output_dir / "metrics" / "test_metrics.json", test_metrics)
        print(
            f"test accuracy={test_metrics['accuracy']:.4f} "
            f"loss={test_metrics['loss']:.5f} "
            f"target_energy={test_metrics['target_energy']:.5f} "
            f"background_energy={test_metrics['background_energy']:.5f}",
            flush=True,
        )
    if args.phase in {"export", "all"}:
        export_dir = settings.output_dir / settings.hardware_export_subdir
        contract = export_hardware_bundle(
            model, bundle.test, settings, device, export_dir
        )
        print(
            f"hardware BMPs exported to {export_dir}; "
            f"samples={contract['sample_count']}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
