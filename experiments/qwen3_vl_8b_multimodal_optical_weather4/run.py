from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import random_split

from .datasets import DatasetBundle, WEATHER4_CLASSES, load_weather4, make_loader
from .download import download_checkpoint
from .features import extract_and_cache, load_feature_cache
from .io_utils import resolve_device, resolve_dtype, runtime_metadata, set_seed, write_json
from .modeling import MLPHead, LoadedBackbone, load_backbone, parameter_report
from .optics import OpticalVisionBlockSurrogate, VisionBlockReplacement
from .results import run_and_save_inference, write_comparison
from .settings import (
    Settings,
    load_settings,
    normalize_hub_cache_dir,
    resolve_model_id,
    resolve_path,
)
from .student_training import train_optical_student
from .training import train_head


PHASES = (
    "download",
    "prepare_data",
    "teacher_train",
    "teacher_inference",
    "student_train",
    "student_inference",
    "compare",
    "all",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Qwen3-VL-8B multimodal BDD100K Weather-4 teacher/student experiment "
            "with one optical vision-block surrogate"
        )
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--phase", choices=PHASES, default="all")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--local-files-only", action=argparse.BooleanOptionalAction, default=None
    )
    parser.add_argument("--download-workers", type=int, default=2)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = load_settings(args.config)
    config_path = resolve_path(args.config, Path.cwd(), "config")
    for name in ("output_dir", "data_root", "cache_dir"):
        value = getattr(args, name)
        if value is not None:
            setattr(settings, name, resolve_path(value, Path.cwd(), name))
    if args.model_id is not None:
        settings.model_id = resolve_model_id(args.model_id, config_path.parent)
    settings.cache_dir = normalize_hub_cache_dir(settings.cache_dir, settings.model_id)
    if args.device is not None:
        settings.device = args.device
    if args.local_files_only is not None:
        settings.local_files_only = args.local_files_only
    settings.validate()
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    _restore_download_cache(settings, args.phase)
    write_json(settings.output_dir / "config_resolved.json", settings.to_dict())

    if args.phase == "download":
        snapshot = download_checkpoint(
            settings.model_id,
            settings.cache_dir,
            max_workers=args.download_workers,
            disable_xet=True,
        )
        write_json(
            settings.output_dir / "download.json",
            {
                "model_id": settings.model_id,
                "snapshot": str(snapshot),
                "cache_dir": str(settings.cache_dir) if settings.cache_dir else None,
            },
        )
        _log(f"checkpoint downloaded: {snapshot}")
        return 0
    if args.phase == "compare":
        _compare(settings, WEATHER4_CLASSES)
        return 0

    if args.phase == "prepare_data":
        data = _load_data(settings)
        write_json(settings.output_dir / "dataset.json", data.metadata)
        _log(
            f"dataset ready: train={len(data.train)} test={len(data.test)} "
            f"classes={data.class_names}"
        )
        return 0

    set_seed(settings.seed)
    device = resolve_device(settings.device)
    write_json(settings.output_dir / "environment.json", runtime_metadata(device))
    data = _load_data(settings)
    write_json(settings.output_dir / "dataset.json", data.metadata)
    _log(
        f"dataset ready: train={len(data.train)} test={len(data.test)} "
        f"classes={data.class_names}"
    )
    loaded = _load_model(settings, device)
    replacement = _build_replacement(loaded, settings, device)
    _write_model_report(loaded.model, replacement, settings)

    try:
        teacher_head: MLPHead | None = None
        student_head: MLPHead | None = None

        if args.phase in {"teacher_train", "all"}:
            teacher_head = _train_teacher(loaded, replacement, data, settings, device)
            if args.phase == "teacher_train":
                return 0

        if args.phase in {"teacher_inference", "student_train", "all"}:
            if teacher_head is None:
                teacher_head = _load_named_head(
                    settings.output_dir / "checkpoints" / "teacher_mlp.pt",
                    settings.dropout,
                    device,
                    "Teacher MLP checkpoint is missing. Run --phase teacher_train first.",
                )

        if args.phase in {"teacher_inference", "all"}:
            test_loader = _test_loader(data, settings)
            run_and_save_inference(
                model=loaded.model,
                processor=loaded.processor,
                replacement=replacement,
                head=teacher_head,
                loader=test_loader,
                class_names=data.class_names,
                prompt=settings.classification_prompt,
                device=device,
                student=False,
                max_batches=settings.benchmark_batches,
                output_dir=settings.output_dir,
            )
            if args.phase == "teacher_inference":
                return 0

        if args.phase in {"student_train", "all"}:
            assert teacher_head is not None
            student_head = MLPHead(
                teacher_head.feature_dim,
                settings.hidden_dim,
                len(data.class_names),
                settings.dropout,
            ).to(device)
            if settings.initialize_student_mlp_from_teacher:
                student_head.load_state_dict(teacher_head.state_dict())
            train_subset, validation_subset = _split_train_dataset(data, settings)
            train_loader = make_loader(
                train_subset,
                settings.feature_batch_size,
                settings.num_workers,
                True,
                settings.seed,
            )
            validation_loader = make_loader(
                validation_subset,
                settings.inference_batch_size,
                settings.num_workers,
                False,
                settings.seed,
            )
            train_optical_student(
                loaded.model,
                loaded.processor,
                replacement,
                teacher_head,
                student_head,
                train_loader,
                validation_loader,
                data.class_names,
                settings.classification_prompt,
                device,
                settings.output_dir,
                settings.epochs,
                settings.learning_rate,
                settings.weight_decay,
                settings.distill_temperature,
                settings.loss_hidden_weight,
                settings.loss_kd_weight,
                settings.loss_ce_weight,
                settings.progress,
            )
            if args.phase == "student_train":
                return 0

        if args.phase == "student_inference":
            student_head = _load_named_head(
                settings.output_dir / "checkpoints" / "student_mlp.pt",
                settings.dropout,
                device,
                "Student MLP checkpoint is missing. Run --phase student_train first.",
            )
            _load_optical_checkpoint(replacement, settings.output_dir, device)

        if args.phase in {"student_inference", "all"}:
            assert student_head is not None
            test_loader = _test_loader(data, settings)
            run_and_save_inference(
                model=loaded.model,
                processor=loaded.processor,
                replacement=replacement,
                head=student_head,
                loader=test_loader,
                class_names=data.class_names,
                prompt=settings.classification_prompt,
                device=device,
                student=True,
                max_batches=settings.benchmark_batches,
                output_dir=settings.output_dir,
            )
            if args.phase == "student_inference":
                return 0

        if args.phase == "all":
            _compare(settings, data.class_names)
        return 0
    finally:
        replacement.close()


def _load_data(settings: Settings) -> DatasetBundle:
    started = time.perf_counter()
    data = load_weather4(
        settings.data_root,
        settings.resize_to,
        settings.train_limit,
        settings.test_limit,
        settings.train_limit_per_class,
        settings.test_limit_per_class,
        settings.imagefolder_train,
        settings.imagefolder_test,
        settings.seed,
        settings.download,
    )
    data.metadata["initialization_time_sec"] = time.perf_counter() - started
    return data


def _load_model(settings: Settings, device: torch.device) -> LoadedBackbone:
    _log(f"loading frozen Qwen3-VL backbone: {settings.model_id}")
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
    loaded: LoadedBackbone, settings: Settings, device: torch.device
) -> VisionBlockReplacement:
    vision_config = loaded.model.config.vision_config
    surrogate = OpticalVisionBlockSurrogate(
        hidden_size=int(vision_config.hidden_size),
        optical_dim=settings.optical_dim,
        optical_layers=settings.optical_layers,
        optical_field_size=settings.optical_field_size,
        optical_padding_size=settings.optical_padding_size,
        wavelength_nm=settings.wavelength_nm,
        pixel_pitch_um=settings.pixel_pitch_um,
        mask_distance_cm=settings.mask_distance_cm,
    ).to(device=device, dtype=resolve_dtype(settings.dtype))
    return VisionBlockReplacement(
        loaded.model, settings.replace_vision_block_start, surrogate
    )


def _train_teacher(
    loaded: LoadedBackbone,
    replacement: VisionBlockReplacement,
    data: DatasetBundle,
    settings: Settings,
    device: torch.device,
) -> MLPHead:
    replacement.use_teacher()
    train_features, train_labels = _teacher_features(
        "train", data, loaded, settings, device
    )
    test_features, test_labels = _teacher_features(
        "test", data, loaded, settings, device
    )
    replacement.capture.clear()
    head = MLPHead(
        train_features.shape[-1],
        settings.hidden_dim,
        len(data.class_names),
        settings.dropout,
    )
    head, _ = train_head(
        head,
        train_features,
        train_labels,
        test_features,
        test_labels,
        data.class_names,
        device,
        settings.output_dir,
        settings.head_batch_size,
        settings.epochs,
        settings.validation_fraction,
        settings.learning_rate,
        settings.weight_decay,
        settings.seed,
        settings.progress,
    )
    source = settings.output_dir / "checkpoints" / "best_mlp.pt"
    checkpoint = torch.load(source, map_location="cpu", weights_only=True)
    torch.save(checkpoint, settings.output_dir / "checkpoints" / "teacher_mlp.pt")
    return head


def _teacher_features(
    split: str,
    data: DatasetBundle,
    loaded: LoadedBackbone,
    settings: Settings,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    path = settings.output_dir / "features" / f"{split}.pt"
    dataset = data.train if split == "train" else data.test
    metadata = {
        "dataset": "bdd100k_weather4",
        "split": split,
        "samples": len(dataset),
        "model_id": settings.model_id,
        "classification_prompt": settings.classification_prompt,
        "feature_source": "electronic_teacher_answer_position_hidden_state",
    }
    if settings.cache_features and path.is_file():
        features, labels, cached = load_feature_cache(path)
        if cached == metadata:
            return features, labels
    loader = make_loader(
        dataset,
        settings.feature_batch_size,
        settings.num_workers,
        False,
        settings.seed,
    )
    features, labels, _ = extract_and_cache(
        loaded.model,
        loaded.processor,
        loader,
        device,
        split,
        settings.output_dir,
        metadata,
        settings.cache_dtype,
        settings.progress,
        settings.classification_prompt,
    )
    return features, labels


def _load_named_head(
    path: Path, dropout: float, device: torch.device, missing_message: str
) -> MLPHead:
    if not path.is_file():
        raise FileNotFoundError(f"{missing_message} Expected: {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    head = MLPHead(
        checkpoint["feature_dim"],
        checkpoint["hidden_dim"],
        checkpoint["num_classes"],
        dropout,
    )
    head.load_state_dict(checkpoint["state_dict"])
    return head.to(device)


def _load_optical_checkpoint(
    replacement: VisionBlockReplacement, output_dir: Path, device: torch.device
) -> None:
    path = output_dir / "checkpoints" / "optical_surrogate.pt"
    if not path.is_file():
        raise FileNotFoundError(
            f"Optical surrogate checkpoint is missing: {path}. Run --phase student_train first."
        )
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if int(checkpoint["vision_block_index"]) != replacement.block_index:
        raise ValueError("Optical checkpoint vision block index does not match config")
    replacement.surrogate.load_state_dict(checkpoint["state_dict"])


def _split_train_dataset(data: DatasetBundle, settings: Settings) -> tuple[Any, Any]:
    validation_size = max(1, int(round(len(data.train) * settings.validation_fraction)))
    train_size = len(data.train) - validation_size
    if train_size <= 0:
        raise ValueError("Training dataset is too small for the requested validation_fraction")
    return random_split(
        data.train,
        [train_size, validation_size],
        generator=torch.Generator().manual_seed(settings.seed),
    )


def _test_loader(data: DatasetBundle, settings: Settings) -> Any:
    return make_loader(
        data.test,
        settings.inference_batch_size,
        settings.num_workers,
        False,
        settings.seed,
    )


def _write_model_report(
    model: nn.Module, replacement: VisionBlockReplacement, settings: Settings
) -> None:
    report = parameter_report(model)
    optical_parameters = sum(
        parameter.numel() for parameter in replacement.surrogate.parameters()
    )
    report.update(
        {
            "model_id": settings.model_id,
            "teacher": "full_electronic_qwen3_vl_multimodal_mlp",
            "student": f"optical_surrogate_vision_block_{replacement.block_index}",
            "replaced_vision_blocks": [replacement.block_index],
            "optical_trainable_parameters": optical_parameters,
        }
    )
    write_json(settings.output_dir / "model.json", report)


def _compare(settings: Settings, class_names: list[str]) -> None:
    write_comparison(
        settings.output_dir,
        "bdd100k_weather4",
        class_names,
        settings.classification_prompt,
        {
            "vision_blocks": [settings.replace_vision_block_start],
            "optical_dim": settings.optical_dim,
            "optical_layers": settings.optical_layers,
            "optical_field_size": settings.optical_field_size,
            "optical_padding_size": settings.optical_padding_size,
            "wavelength_nm": settings.wavelength_nm,
            "pixel_pitch_um": settings.pixel_pitch_um,
            "mask_distance_cm": settings.mask_distance_cm,
        },
        {
            "hidden": settings.loss_hidden_weight,
            "kd": settings.loss_kd_weight,
            "ce": settings.loss_ce_weight,
            "temperature": settings.distill_temperature,
        },
    )


def _restore_download_cache(settings: Settings, phase: str) -> None:
    if phase == "download" or not settings.local_files_only or settings.cache_dir is not None:
        return
    record_path = settings.output_dir / "download.json"
    if not record_path.is_file():
        return
    with record_path.open("r", encoding="utf-8") as handle:
        record = json.load(handle)
    cache_dir = record.get("cache_dir")
    if cache_dir and Path(cache_dir).is_dir():
        settings.cache_dir = Path(cache_dir).resolve()


def _log(message: str) -> None:
    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    print(f"[{timestamp}] {message}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
