from __future__ import annotations

import argparse
import os
import platform
import random
import sys
from pathlib import Path
from typing import Any

from .settings import Settings, load_settings, resolved_dict


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="BDD100K Weather-4 no-Qwen baseline: grayscale -> five optical layers -> MLP"
    )
    parser.add_argument("--config", required=True, help="Path to a JSON configuration file")
    parser.add_argument("--phase", choices=["prepare_data", "train", "test", "all"], default="all")
    parser.add_argument("--device", help="Override device, for example cuda, cuda:0, or cpu")
    parser.add_argument("--epochs", type=int, help="Override training epochs")
    parser.add_argument("--output-dir", help="Override output directory")
    parser.add_argument("--smoke-test", action="store_true", help="Use 64 train images, 32 test images, one epoch, batch size 8, and no workers")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    import torch

    from .data import load_weather4, make_loader
    from .metrics import write_confusion_csv, write_json
    from .training import evaluate, load_best_checkpoint, train_model
    from .visualization import save_confusion_matrix

    settings, config_path = load_settings(args.config)
    _apply_overrides(settings, args)
    settings.validate()
    seed_everything(settings.seed)
    output_dir = Path(settings.output_dir)
    _create_output_tree(output_dir)
    write_json(output_dir / "config_resolved.json", resolved_dict(settings))
    write_json(output_dir / "environment.json", _environment())
    if args.phase == "prepare_data":
        data = load_weather4(settings)
        write_json(output_dir / "dataset.json", data.metadata)
        print(f"[dataset] ready at {settings.data_root}: train={len(data.train)} validation={len(data.validation)} test={len(data.test)}")
        return 0

    data = load_weather4(settings)
    write_json(output_dir / "dataset.json", data.metadata)
    device = resolve_device(settings.device)
    model = build_model(settings).to(device)
    write_json(output_dir / "model.json", _model_metadata(model, settings))
    print(f"[run] phase={args.phase} device={device} output={output_dir}")
    if args.phase in {"train", "all"}:
        train_model(model, data, settings, device, output_dir)
    if args.phase in {"test", "all"}:
        load_best_checkpoint(model, output_dir, device)
        test_loader = make_loader(data.test, settings.batch_size, settings.num_workers, False, settings.seed + 20)
        metrics = evaluate(model, test_loader, device, data.class_names)
        write_json(output_dir / "metrics" / "test_metrics.json", metrics)
        write_json(output_dir / "metrics" / "per_class_metrics.json", metrics["per_class"])
        write_confusion_csv(output_dir / "metrics" / "confusion_matrix.csv", metrics["confusion_matrix"], data.class_names)
        save_confusion_matrix(metrics["confusion_matrix"], data.class_names, output_dir / "figures" / "confusion_matrix.png")
        print(
            f"[test] top1={metrics['top1_accuracy']:.4f} macro_f1={metrics['macro_f1']:.4f} "
            f"balanced_accuracy={metrics['balanced_accuracy']:.4f}"
        )
    return 0


def build_model(settings: Settings):
    from .model import Optical5MLPWeatherClassifier

    return Optical5MLPWeatherClassifier(
        input_size=settings.input_size,
        optical_field_size=settings.optical_field_size,
        optical_padding_size=settings.optical_padding_size,
        wavelength_nm=settings.wavelength_nm,
        pixel_pitch_um=settings.pixel_pitch_um,
        mask_distance_cm=settings.mask_distance_cm,
        phase_init=settings.phase_init,
        amplitude_mask_enabled=settings.amplitude_mask_enabled,
        detector_pool_size=settings.detector_pool_size,
        mlp_hidden_dim=settings.mlp_hidden_dim,
        dropout=settings.dropout,
        num_classes=settings.num_classes,
        optical_layers=settings.optical_layers,
        phase_dropout=settings.phase_dropout,
    )


def _apply_overrides(settings: Settings, args: argparse.Namespace) -> None:
    if args.device:
        settings.device = args.device
    if args.epochs is not None:
        settings.epochs = args.epochs
    if args.output_dir:
        settings.output_dir = str(Path(os.path.expandvars(os.path.expanduser(args.output_dir))).resolve())
    if args.smoke_test:
        settings.train_limit = 64
        settings.test_limit = 32
        settings.epochs = 1
        settings.batch_size = 8
        settings.num_workers = 0
        settings.save_interval_epochs = 1


def resolve_device(requested: str):
    import torch

    if requested.startswith("cuda") and not torch.cuda.is_available():
        print("[device] CUDA requested but unavailable; using CPU", file=sys.stderr)
        return torch.device("cpu")
    return torch.device(requested)


def seed_everything(seed: int) -> None:
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _create_output_tree(root: Path) -> None:
    for relative in (
        "metrics", "figures/phase_masks", "figures/light_fields",
        "figures/detector_outputs", "checkpoints",
    ):
        (root / relative).mkdir(parents=True, exist_ok=True)


def _model_metadata(model: Any, settings: Settings) -> dict[str, Any]:
    return {
        "model_name": "Optical5MLPWeatherClassifier",
        "pipeline": "RGB image -> grayscale amplitude -> 5 optical detection layers -> MLP readout",
        "uses_qwen": False,
        "uses_transformer": False,
        "uses_moe": False,
        "optical_layers": settings.optical_layers,
        "optical_field_size": settings.optical_field_size,
        "optical_padding_size": settings.optical_padding_size,
        "wavelength_nm": settings.wavelength_nm,
        "pixel_pitch_um": settings.pixel_pitch_um,
        "mask_distance_cm": settings.mask_distance_cm,
        "amplitude_mask_enabled": settings.amplitude_mask_enabled,
        "detector_pool_size": settings.detector_pool_size,
        "phase_dropout": settings.regularization.get("phase_dropout", {}),
        **model.parameter_summary(),
    }


def _environment() -> dict[str, Any]:
    import torch

    try:
        import torchvision
        torchvision_version = torchvision.__version__
    except Exception as exc:
        torchvision_version = f"unavailable: {exc}"
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "pytorch": torch.__version__,
        "torchvision": torchvision_version,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "gpu_names": [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())],
    }


if __name__ == "__main__":
    raise SystemExit(main())
