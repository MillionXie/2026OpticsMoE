from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch

from .datasets import load_timeofday3
from .feature_probe import build_source_encoder, extract_split, source_parameter_report
from .io_utils import resolve_device, resolve_dtype, runtime_metadata, set_seed, write_json
from .modeling import load_backbone
from .settings import Settings, load_settings, resolve_path
from .training import probe_inference, train_probe


PHASES = ("prepare_data", "extract_features", "train_probe", "probe_inference", "all")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="BDD100K TimeOfDay-3 trained vision optical-input-field MLP probe"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--phase", choices=PHASES, default="all")
    parser.add_argument("--device")
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--model-id")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = load_settings(args.config); _overrides(settings, args); set_seed(settings.seed)
    _make_dirs(settings.output_dir)
    write_json(settings.output_dir / "config_resolved.json", settings.to_dict())
    data = load_timeofday3(settings)
    write_json(settings.output_dir / "dataset.json", data.metadata)
    if args.phase == "prepare_data":
        print(f"BDD100K TimeOfDay-3 ready: train={len(data.train)} test={len(data.test)}")
        return 0
    device = resolve_device(settings.device)
    write_json(settings.output_dir / "environment.json", runtime_metadata(device))
    source_parameters = _expected_source_parameters(settings)
    if args.phase in {"extract_features", "all"}:
        loaded = load_backbone(
            settings.model_id, settings.cache_dir, settings.local_files_only,
            resolve_dtype(settings.dtype), device, settings.attn_implementation,
            settings.processor_min_pixels, settings.processor_max_pixels,
        )
        encoder = build_source_encoder(loaded, settings)
        source_parameters = source_parameter_report(encoder)
        write_json(settings.output_dir / "metrics" / "probe_model.json", {
            **source_parameters, "feature_type": "vision_optical_input_field", "feature_dim": 4096,
            "probe_head_type": settings.probe_head_type, "probe_hidden_dim": settings.probe_hidden_dim,
            "finetune_vision_input_adapter": False,
        })
        extract_split(loaded, encoder, data.train, "train", settings, data.class_names)
        extract_split(loaded, encoder, data.test, "test", settings, data.class_names)
        if args.phase == "extract_features":
            return 0
        del encoder, loaded
        if device.type == "cuda":
            torch.cuda.empty_cache()
    if args.phase in {"train_probe", "all"}:
        train_probe(settings, data.class_names, device, source_parameters)
        if args.phase == "train_probe":
            return 0
    if args.phase in {"probe_inference", "all"}:
        report = probe_inference(settings, data.class_names, device)
        print(
            f"Vision-field probe: top1={report['top1_accuracy']:.4f} "
            f"macro_f1={report['macro_f1']:.4f} balanced_accuracy={report['balanced_accuracy']:.4f}",
            flush=True,
        )
    return 0


def _expected_source_parameters(settings: Settings) -> dict[str, int]:
    source_config = settings.source_experiment_dir / "config_resolved.json"
    hidden_size = 1024
    if source_config.is_file():
        import json
        hidden_size = int(json.loads(source_config.read_text(encoding="utf-8")).get("vision_hidden_size", hidden_size))
    input_parameters = hidden_size * settings.optical_dim + settings.optical_dim
    norm_parameters = 2 * settings.optical_dim
    return {
        "source_vision_input_adapter_parameters": input_parameters,
        "source_vision_adapter_norm_parameters": norm_parameters,
        "source_frozen_encoder_parameters": input_parameters + norm_parameters,
        "source_trainable_parameters_during_probe": 0,
    }


def _overrides(settings: Settings, args: argparse.Namespace) -> None:
    if args.device:
        settings.device = args.device
    if args.cache_dir:
        settings.cache_dir = resolve_path(args.cache_dir, Path.cwd(), "cache_dir")
    if args.output_dir:
        settings.output_dir = resolve_path(args.output_dir, Path.cwd(), "output_dir")
    if args.model_id:
        settings.model_id = args.model_id if args.model_id == "Qwen/Qwen3-VL-2B-Instruct" else str(resolve_path(args.model_id, Path.cwd(), "model_id"))
    if args.local_files_only:
        settings.local_files_only = True
    settings.validate()


def _make_dirs(root: Path) -> None:
    for name in ("features", "metrics", "checkpoints", "figures/vision_input_fields"):
        (root / name).mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())

