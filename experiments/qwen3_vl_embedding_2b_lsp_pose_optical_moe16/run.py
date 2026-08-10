from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .datasets import prepare_lsp
from .modeling import load_vision_backbone
from .settings import load_settings, save_resolved_config
from .training import (
    cache_teacher_heatmaps,
    infer_student,
    infer_teacher,
    save_comparison,
    seed_everything,
    train_student,
    train_teacher,
)


PHASES = (
    "prepare_data", "teacher_train", "teacher_inference",
    "teacher_cache", "student_train", "student_inference", "compare", "all",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LSP/LSPET Qwen Vision optical pose experiment")
    parser.add_argument("--config", required=True)
    parser.add_argument("--phase", choices=PHASES, default="all")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    settings = load_settings(args.config)
    save_resolved_config(settings)
    seed_everything(settings.random_seed)
    bundle = prepare_lsp(settings, persist=True)
    print(
        f"LSP pose train={len(bundle.train):,} test={len(bundle.test):,} "
        f"protocol={bundle.metadata['protocol']}", flush=True,
    )
    if args.phase == "prepare_data":
        return 0
    if args.phase == "compare":
        _compare_from_disk(settings)
        return 0
    device = _device(settings.device)
    print(f"loading {settings.model_id} on {device}", flush=True)
    loaded = load_vision_backbone(settings, device)
    if args.phase in {"teacher_train", "all"}:
        train_teacher(loaded, bundle, settings)
    if args.phase in {"teacher_inference", "all"}:
        infer_teacher(loaded, bundle, settings)
    if args.phase in {"teacher_cache", "all"}:
        cache_teacher_heatmaps(loaded, bundle, settings)
    if args.phase in {"student_train", "all"}:
        train_student(loaded, bundle, settings)
    if args.phase in {"student_inference", "all"}:
        infer_student(loaded, bundle, settings)
    if args.phase == "all":
        _compare_from_disk(settings)
    return 0


def _device(requested: str) -> torch.device:
    if requested.startswith("cuda") and not torch.cuda.is_available():
        print("WARNING: CUDA requested but unavailable; falling back to CPU", flush=True)
        return torch.device("cpu")
    return torch.device(requested)


def _compare_from_disk(settings: object) -> None:
    metrics = settings.output_dir / "metrics"
    teacher_path = metrics / "teacher_inference.json"
    student_path = metrics / "student_inference.json"
    missing = [str(path) for path in (teacher_path, student_path) if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Run teacher_inference and student_inference first; missing {missing}")
    teacher = json.loads(teacher_path.read_text(encoding="utf-8"))
    student = json.loads(student_path.read_text(encoding="utf-8"))
    save_comparison(settings, teacher, student)


if __name__ == "__main__":
    raise SystemExit(main())

