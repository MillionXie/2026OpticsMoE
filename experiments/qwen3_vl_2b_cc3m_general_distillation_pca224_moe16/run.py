from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from .calibration import fit_shared_pca, run_pca_oracle_check
from .cache_paths import teacher_cache_root
from .datasets import DatasetBundle, load_jsonl_dataset
from .io_utils import (
    configure_cpu_runtime,
    resolve_device,
    resolve_dtype,
    runtime_metadata,
    set_seed,
    write_json,
)
from .modeling import LoadedBackbone, load_backbone, module_parameters
from .optics import (
    LanguagePCAOpticalMoE,
    PCAMultimodalReplacement,
    VisionPCAOpticalMoE,
)
from .pca import load_projection, projection_paths
from .settings import Settings, load_settings, resolve_path
from .teacher_cache import build_projected_teacher_cache, load_cache_stores
from .training import train_phase


PHASES = (
    "fit_pca",
    "pca_oracle_check",
    "precompute_teacher",
    "train_vision",
    "train_language",
    "train_joint",
    "all",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Unlabeled CC3M general Qwen3-VL distillation into PCA224 Optical MoE16"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--phase", choices=PHASES, default="all")
    parser.add_argument("--device")
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--local-files-only", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = load_settings(args.config)
    _apply_overrides(settings, args)
    cpu = configure_cpu_runtime(settings.cpu_threads, settings.cpu_interop_threads)
    set_seed(settings.seed)
    _make_directories(settings)
    data = load_jsonl_dataset(settings, persist_split=True)
    write_json(settings.output_dir / "dataset.json", data.metadata)
    write_json(settings.output_dir / "config_resolved.json", settings.to_dict())
    print(
        f"[runtime] cpu_threads={cpu['torch_cpu_threads']} "
        f"cpu_interop_threads={cpu['torch_cpu_interop_threads']} "
        f"precompute_workers={settings.num_workers} student_cache_workers=0",
        flush=True,
    )
    device = resolve_device(settings.device)
    write_json(settings.output_dir / "environment.json", runtime_metadata(device))
    loaded = _load_model(settings, device)
    settings.resolve_architecture(loaded.model)
    settings.validate()
    write_json(settings.output_dir / "config_resolved.json", settings.to_dict())
    write_json(
        settings.output_dir / "teacher_model.json",
        {
            "model_id": settings.model_id,
            "vision_depth": settings.vision_depth,
            "vision_hidden_size": settings.vision_hidden_size,
            "text_depth": settings.text_depth,
            "text_hidden_size": settings.text_hidden_size,
            "deepstack_visual_indexes": list(settings.deepstack_visual_indexes),
            "language_tap_indexes": list(settings.language_tap_indexes),
            "total_parameters": module_parameters(loaded.model),
            "trainable_parameters": module_parameters(loaded.model, True),
            "frozen": True,
        },
    )
    replacement: PCAMultimodalReplacement | None = None
    try:
        if args.phase in {"fit_pca", "all"}:
            fit_shared_pca(loaded.model, loaded.processor, data, settings, device)
            if args.phase == "fit_pca":
                return 0
        if args.phase in {"pca_oracle_check", "all"}:
            run_pca_oracle_check(loaded.model, loaded.processor, data, settings, device)
            if args.phase == "pca_oracle_check":
                return 0
        if args.phase in {"precompute_teacher", "all"}:
            build_projected_teacher_cache(
                loaded.model, loaded.processor, data, settings, device
            )
            if args.phase == "precompute_teacher":
                return 0
        if args.phase in {"train_vision", "train_language", "train_joint", "all"}:
            replacement = _build_replacement(loaded, settings, device)
            _write_student_model_report(loaded.model, replacement, settings)
            stores = load_cache_stores(settings)
            pad_token_id = getattr(loaded.processor.tokenizer, "pad_token_id", 0)
            if pad_token_id is None:
                pad_token_id = 0
            padding_side = getattr(loaded.processor.tokenizer, "padding_side", "left")
            modes = (
                ("vision", "language", "joint")
                if args.phase == "all"
                else (args.phase.removeprefix("train_"),)
            )
            for mode in modes:
                train_phase(
                    mode,
                    loaded.model,
                    replacement,
                    stores["train"],
                    stores["validation"],
                    settings,
                    device,
                    pad_token_id=int(pad_token_id),
                    padding_side=padding_side,
                )
        return 0
    finally:
        if replacement is not None:
            replacement.close()


def _load_model(settings: Settings, device: torch.device) -> LoadedBackbone:
    _log(f"loading frozen teacher {settings.model_id}")
    return load_backbone(
        settings.model_id,
        settings.cache_dir,
        settings.local_files_only,
        resolve_dtype(settings.dtype),
        device,
        settings.attn_implementation,
        settings.processor_min_pixels,
        settings.processor_max_pixels,
    )


def _build_replacement(
    loaded: LoadedBackbone,
    settings: Settings,
    device: torch.device,
) -> PCAMultimodalReplacement:
    vision_path, language_path = projection_paths(settings)
    vision_pca = load_projection(vision_path, settings.vision_hidden_size)
    language_pca = load_projection(language_path, settings.text_hidden_size)
    vision = VisionPCAOpticalMoE(vision_pca, settings).to(device)
    language = LanguagePCAOpticalMoE(language_pca, settings).to(device)
    return PCAMultimodalReplacement(loaded.model, vision, language, settings)


def _write_student_model_report(
    model: torch.nn.Module,
    replacement: PCAMultimodalReplacement,
    settings: Settings,
) -> None:
    vision = replacement.vision_surrogate.parameter_breakdown()
    language = replacement.language_surrogate.parameter_breakdown()
    trainable = {
        id(parameter): parameter
        for parameter in replacement.trainable_parameters("joint")
        if parameter.requires_grad
    }
    suspicious_linears = []
    for stack_name, surrogate, hidden_dim in (
        ("vision", replacement.vision_surrogate, settings.vision_hidden_size),
        ("language", replacement.language_surrogate, settings.text_hidden_size),
    ):
        for name, module in surrogate.named_modules():
            if isinstance(module, torch.nn.Linear) and {
                module.in_features,
                module.out_features,
            } == {settings.latent_dim, hidden_dim}:
                suspicious_linears.append(f"{stack_name}.{name}")
    if suspicious_linears:
        raise RuntimeError(
            "Trainable hidden/PCA adapter Linear modules are forbidden: "
            + ", ".join(suspicious_linears)
        )
    report = {
        "experiment": settings.experiment_name,
        "teacher_model_id": settings.model_id,
        "teacher_frozen": True,
        "task_head": None,
        "task_labels_used": False,
        "teacher_logits_used": False,
        "generation_used": False,
        "pca": {
            "latent_dim": settings.latent_dim,
            "vision_shared_across_input_and_stages": True,
            "language_shared_across_input_and_stages": True,
            "mean_and_components_are_buffers": True,
            "trainable_parameters": 0,
            "trainable_hidden_to_latent_linear_parameters": 0,
            "trainable_latent_to_hidden_linear_parameters": 0,
        },
        "vision": vision,
        "language": language,
        "student_total_trainable_parameters": sum(
            parameter.numel() for parameter in trainable.values()
        ),
        "qwen_total_parameters": module_parameters(model),
        "qwen_trainable_parameters": module_parameters(model, True),
        "stage_representation": "[valid_tokens,224] signed detector readout",
        "reload_representation": "relu(signed_readout), used only by the next optical stage",
        "vision_tap_indexes": [
            *settings.deepstack_visual_indexes,
            settings.vision_depth - 1,
        ],
        "language_tap_indexes": list(settings.language_tap_indexes),
        "cache_root": str(teacher_cache_root(settings)),
    }
    write_json(settings.output_dir / "model.json", report)


def _apply_overrides(settings: Settings, args: argparse.Namespace) -> None:
    if args.device:
        settings.device = args.device
    if args.cache_dir:
        settings.cache_dir = resolve_path(args.cache_dir, Path.cwd(), "cache_dir")
    if args.output_dir:
        settings.output_dir = resolve_path(args.output_dir, Path.cwd(), "output_dir")
    if args.local_files_only:
        settings.local_files_only = True
    settings.validate()


def _make_directories(settings: Settings) -> None:
    for name in ("pca", "metrics", "checkpoints", "figures"):
        (settings.output_dir / name).mkdir(parents=True, exist_ok=True)
    settings.precompute_cache_dir.mkdir(parents=True, exist_ok=True)


def _log(message: str) -> None:
    print(
        f"[{datetime.now(timezone.utc).isoformat(timespec='seconds')}] {message}",
        flush=True,
    )


if __name__ == "__main__":
    raise SystemExit(main())
