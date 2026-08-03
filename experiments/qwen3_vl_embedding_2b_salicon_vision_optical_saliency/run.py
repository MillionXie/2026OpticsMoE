from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch

from .datasets import prepare_salicon
from .io_utils import environment_report, seed_everything, write_json
from .modeling import load_vision_backbone
from .settings import load_settings, save_resolved_config
from .training import (
    cache_teacher_maps,
    evaluate_checkpoint,
    train_student,
    train_teacher,
)


PHASES = {
    "prepare_data",
    "teacher_train",
    "teacher_evaluate",
    "cache_teacher_maps",
    "student_train",
    "student_evaluate",
    "all",
}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="SALICON fixation-density prediction with Qwen Vision Optical MoE16"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--phase", choices=sorted(PHASES), default="all")
    parser.add_argument("--checkpoint", default=None)
    args = parser.parse_args()

    settings = load_settings(args.config)
    seed_everything(settings.random_seed)
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    settings.artifact_cache_dir.mkdir(parents=True, exist_ok=True)
    save_resolved_config(settings)
    write_json(settings.output_dir / "environment.json", environment_report())
    bundle = prepare_salicon(settings, persist=True)
    if args.phase == "prepare_data":
        print(
            f"SALICON train={len(bundle.train_records):,} "
            f"validation={len(bundle.validation_records):,}",
            flush=True,
        )
        return 0

    device = torch.device(
        settings.device
        if settings.device != "cuda" or torch.cuda.is_available()
        else "cpu"
    )
    loaded = load_vision_backbone(settings, device)
    save_resolved_config(settings)
    checkpoint = (
        Path(args.checkpoint).expanduser().resolve()
        if args.checkpoint
        else None
    )
    if args.phase in {"teacher_train", "all"}:
        train_teacher(loaded, bundle, settings)
        if args.phase == "teacher_train":
            return 0
    teacher_metrics = None
    if args.phase in {"teacher_evaluate", "all"}:
        teacher_metrics = evaluate_checkpoint(
            "teacher", loaded, bundle, settings, checkpoint
        )
        print(_metric_line("Teacher", teacher_metrics), flush=True)
        if args.phase == "teacher_evaluate":
            return 0
    if args.phase == "cache_teacher_maps":
        cache_teacher_maps(loaded, bundle, settings)
        return 0
    if args.phase in {"student_train", "all"}:
        if settings.map_kd_weight > 0:
            cache_teacher_maps(loaded, bundle, settings)
        train_student(loaded, bundle, settings)
        if args.phase == "student_train":
            return 0
    if args.phase in {"student_evaluate", "all"}:
        student_metrics = evaluate_checkpoint(
            "student", loaded, bundle, settings, checkpoint
        )
        print(_metric_line("Optical Student", student_metrics), flush=True)
        if teacher_metrics is not None:
            keys = ("cc", "sim", "nss", "auc_judd", "kld", "mae")
            write_json(
                settings.output_dir / "metrics" / "comparison.json",
                {
                    "split": "official_validation",
                    "teacher": teacher_metrics,
                    "optical_student": student_metrics,
                    "optical_minus_teacher": {
                        key: float(student_metrics[key]) - float(teacher_metrics[key])
                        for key in keys
                    },
                    "note": (
                        "Higher is better for CC/SIM/NSS/AUC-Judd; lower is "
                        "better for KLD/MAE."
                    ),
                },
            )
    return 0


def _metric_line(name: str, values: dict[str, Any]) -> str:
    return (
        f"{name}: CC={values['cc']:.4f} SIM={values['sim']:.4f} "
        f"NSS={values['nss']:.4f} AUC-J={values['auc_judd']:.4f} "
        f"KLD={values['kld']:.4f} MAE={values['mae']:.4f}"
    )


if __name__ == "__main__":
    raise SystemExit(main())
