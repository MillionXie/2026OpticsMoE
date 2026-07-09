from __future__ import annotations

import argparse
import json
import platform
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .data import load_data
from .metrics import write_json
from .models import build_model, parameter_report
from .settings import Settings, load_settings, resolve_path
from .training import test_model, train_model


PHASES = ("prepare_data", "train", "test", "all", "compare")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="BDD100K TimeOfDay-3 standard baseline runner")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--phase", choices=PHASES, default="all")
    parser.add_argument("--device")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--baseline-output-dir", type=Path, action="append", default=[])
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.phase == "compare":
        return _compare(args.baseline_output_dir)
    if args.config is None:
        raise SystemExit("--config is required except for --phase compare")
    settings = load_settings(args.config)
    _overrides(settings, args)
    set_seed(settings.seed)
    _make_dirs(settings.output_dir)
    write_json(settings.output_dir / "config_resolved.json", settings.to_dict())
    write_json(settings.output_dir / "environment.json", runtime_metadata())
    data = load_data(settings)
    write_json(settings.output_dir / "dataset.json", data.metadata)
    if args.phase == "prepare_data":
        print(
            f"BDD100K TimeOfDay-3 ready: train={len(data.train)} "
            f"validation={len(data.validation)} test={len(data.test)}"
        )
        return 0
    device = resolve_device(settings.device)
    model = build_model(settings).to(device)
    report = parameter_report(model)
    report.update(
        {
            "model_type": settings.model_type,
            "class_names": settings.class_names,
            "image_size": settings.image_size,
            "image_normalization": settings.image_normalization,
            "pretrained": settings.pretrained,
            "dataset_alignment_reference": settings.reference_experiment,
        }
    )
    write_json(settings.output_dir / "model.json", report)
    _log(f"model={settings.model_type} parameters={report['parameters']} output={settings.output_dir}")
    if args.phase in {"train", "all"}:
        train_model(model, data, settings, device)
    if args.phase in {"test", "all"}:
        metrics = test_model(model, data, settings, device)
        print(
            f"[test] top1={metrics['top1_accuracy']:.4f} "
            f"macro_f1={metrics['macro_f1']:.4f} balanced={metrics['balanced_accuracy']:.4f}"
        )
    return 0


def _overrides(settings: Settings, args: argparse.Namespace) -> None:
    if args.device:
        settings.device = args.device
    if args.epochs:
        settings.epochs = args.epochs
    if args.data_root:
        settings.data_root = resolve_path(args.data_root, Path.cwd(), "data_root")
    if args.output_dir:
        settings.output_dir = resolve_path(args.output_dir, Path.cwd(), "output_dir")
    settings.validate()


def _compare(output_dirs: list[Path]) -> int:
    if not output_dirs:
        raise SystemExit("compare requires one or more --baseline-output-dir")
    rows = []
    for raw in output_dirs:
        root = resolve_path(raw, Path.cwd(), "baseline_output_dir")
        metrics = _read_json(root / "metrics" / "test_metrics.json")
        model = _read_json(root / "model.json")
        rows.append(
            {
                "name": root.name,
                "model_type": model.get("model_type"),
                "top1_accuracy": metrics.get("top1_accuracy"),
                "macro_f1": metrics.get("macro_f1"),
                "balanced_accuracy": metrics.get("balanced_accuracy"),
                "parameters": model.get("parameters"),
                "trainable_parameters": model.get("trainable_parameters"),
                "training_time_sec": _training_time(root),
                "output_dir": str(root),
            }
        )
    destination = resolve_path(output_dirs[0], Path.cwd(), "baseline_output_dir") / "metrics" / "baseline_compare.json"
    write_json(destination, {"created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "baselines": rows})
    print(json.dumps(rows, indent=2))
    return 0


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(value: str) -> torch.device:
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return device


def runtime_metadata() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpus": [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())],
    }


def _make_dirs(root: Path) -> None:
    for name in (
        "metrics",
        "checkpoints",
        "figures/phase_masks",
        "figures/light_fields",
        "figures/detector_outputs",
        "figures/detector_regions",
    ):
        (root / name).mkdir(parents=True, exist_ok=True)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _training_time(root: Path) -> float | None:
    path = root / "metrics" / "training_history.csv"
    if not path.is_file():
        return None
    import csv

    with path.open(encoding="utf-8") as handle:
        return sum(float(row["epoch_time_sec"]) for row in csv.DictReader(handle))


def _log(message: str) -> None:
    print(f"[{datetime.now(timezone.utc).isoformat(timespec='seconds')}] {message}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
