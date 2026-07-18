from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from experiments.qwen3_vl_2b_spaq_multitask_iqa.data_prepare import ensure_spaq_dataset
from experiments.qwen3_vl_2b_spaq_multitask_iqa.datasets import load_spaq, make_loader
from experiments.qwen3_vl_2b_spaq_multitask_iqa.io_utils import (
    resolve_device,
    resolve_dtype,
    runtime_metadata,
    set_seed,
    write_json,
)
from experiments.qwen3_vl_2b_spaq_multitask_iqa.modeling import load_backbone
from experiments.qwen3_vl_2b_spaq_multitask_iqa.settings import resolve_model_id, resolve_path

from .generation import evaluate_zeroshot
from .settings import Settings, load_settings, normalize_hub_cache_dir


PHASES = ("download", "prepare_data", "evaluate", "all")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Qwen3-VL-2B zero-shot direct numeric scoring on SPAQ without training"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--phase", choices=PHASES, default="all")
    parser.add_argument("--device")
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--model-id")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--local-files-only", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = load_settings(args.config)
    _apply_overrides(settings, args, args.config.resolve().parent)
    settings.validate()
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(settings.seed)
    device = resolve_device(settings.device)
    write_json(settings.output_dir / "resolved_config.json", settings.to_dict())
    write_json(settings.output_dir / "environment.json", runtime_metadata(device))
    preparation = ensure_spaq_dataset(settings)
    write_json(settings.output_dir / "data_preparation.json", preparation)
    if args.phase == "download":
        return 0
    data = load_spaq(settings, persist_split=True)
    write_json(settings.output_dir / "dataset.json", data.metadata)
    print(
        f"SPAQ zero-shot ready: test_images={len(data.test_records)} "
        f"test_pairs={len(data.test)} training=none"
    )
    if args.phase == "prepare_data":
        return 0

    loaded = load_backbone(
        model_id=settings.model_id,
        cache_dir=normalize_hub_cache_dir(settings.cache_dir, settings.model_id),
        local_files_only=settings.local_files_only,
        dtype=resolve_dtype(settings.dtype),
        device=device,
        attn_implementation=settings.attn_implementation,
        min_pixels=settings.processor_min_pixels,
        max_pixels=settings.processor_max_pixels,
    )
    _write_model_report(settings, loaded.model, loaded.load_time_sec)
    loader = make_loader(
        data.test,
        batch_size=settings.generation_batch_size,
        num_workers=settings.num_workers,
        shuffle=False,
        seed=settings.seed,
    )
    metadata = {
        "cache_schema_version": 1,
        "dataset": "spaq",
        "split_digest": data.cache_identity["split_digest"],
        "test_samples": len(data.test),
        "model_id": settings.model_id,
        "processor_min_pixels": settings.processor_min_pixels,
        "processor_max_pixels": settings.processor_max_pixels,
        "dtype": settings.dtype,
        "attn_implementation": settings.attn_implementation,
        "max_new_tokens": settings.max_new_tokens,
        "task_prompts": dict(settings.task_prompts or {}),
        "generation_mode": "deterministic_zero_shot_numeric_score",
    }
    _, metrics = evaluate_zeroshot(
        loaded.model, loaded.processor, loader, settings, metadata, device
    )
    print(
        "SPAQ zero-shot macro: "
        f"MAE={_fmt(metrics['macro_average']['mae'])} "
        f"SRCC={_fmt(metrics['macro_average']['srcc'])} "
        f"PLCC={_fmt(metrics['macro_average']['plcc'])} "
        f"parse_rate={metrics['parse_rate']:.2%}"
    )
    _write_comparison(settings, metrics)
    return 0


def _write_model_report(settings: Settings, model: torch.nn.Module, load_time: float) -> None:
    config = getattr(model, "config", None)
    text_config = getattr(config, "text_config", config)
    vision_config = getattr(config, "vision_config", None)
    write_json(
        settings.output_dir / "model.json",
        {
            "model_id": settings.model_id,
            "model_class": type(model).__name__,
            "load_time_sec": load_time,
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "trainable_parameters": sum(
                parameter.numel() for parameter in model.parameters() if parameter.requires_grad
            ),
            "mode": "eval" if not model.training else "train",
            "training_or_finetuning_used": False,
            "generation_used": True,
            "vision_hidden_size": getattr(vision_config, "hidden_size", None),
            "text_hidden_size": getattr(text_config, "hidden_size", None),
        },
    )


def _write_comparison(settings: Settings, zero_shot: dict[str, Any]) -> None:
    path = settings.supervised_reference_metrics
    if path is None or not path.is_file():
        print(f"warning: supervised reference metrics unavailable: {path}")
        return
    supervised = json.loads(path.read_text(encoding="utf-8"))
    write_json(
        settings.output_dir / "zeroshot_vs_supervised_head.json",
        {
            "warning": "The supervised head is trained; zero-shot uses direct text generation.",
            "zero_shot": zero_shot,
            "frozen_qwen_sigmoid_head": supervised,
        },
    )


def _apply_overrides(settings: Settings, args: argparse.Namespace, config_dir: Path) -> None:
    if args.device:
        settings.device = args.device
    if args.cache_dir is not None:
        settings.cache_dir = resolve_path(args.cache_dir, Path.cwd(), "cache_dir")
    if args.output_dir is not None:
        settings.output_dir = resolve_path(args.output_dir, Path.cwd(), "output_dir")
    if args.model_id:
        settings.model_id = resolve_model_id(args.model_id, config_dir)
    if args.local_files_only:
        settings.local_files_only = True


def _fmt(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):.4f}"


if __name__ == "__main__":
    raise SystemExit(main())

