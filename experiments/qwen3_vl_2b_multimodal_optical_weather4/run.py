from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import Subset

from .datasets import (
    DatasetBundle,
    WEATHER4_CLASSES,
    class_counts,
    load_weather4,
    make_loader,
    stratified_split_indices,
)
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
from .teacher_cache import (
    CachedTeacherDataset,
    TeacherCacheStore,
    build_teacher_cache,
    expected_teacher_cache_metadata,
    make_cached_teacher_loader,
)
from .training import train_head


PHASES = (
    "download",
    "prepare_data",
    "teacher_train",
    "teacher_cache",
    "teacher_inference",
    "student_train",
    "student_inference",
    "compare",
    "all",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Qwen3-VL-2B multimodal BDD100K Weather-4 teacher/student experiment "
            "with five single-mask optical conversions replacing 20 vision blocks"
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
    resolved_config = settings.to_dict()
    resolved_config["runtime_vision_depth"] = int(loaded.model.config.vision_config.depth)
    resolved_config["runtime_vision_hidden_size"] = int(
        loaded.model.config.vision_config.hidden_size
    )
    resolved_config["runtime_text_hidden_size"] = int(
        loaded.model.config.text_config.hidden_size
    )
    resolved_config["resolved_vision_block_groups"] = [
        list(group) for group in replacement.block_groups
    ]
    write_json(settings.output_dir / "config_resolved.json", resolved_config)
    _write_model_report(loaded.model, replacement, settings)

    try:
        teacher_head: MLPHead | None = None
        student_head: MLPHead | None = None

        if args.phase in {"teacher_train", "all"}:
            teacher_head = _train_teacher(loaded, replacement, data, settings, device)
            if args.phase == "teacher_train":
                return 0

        if args.phase in {"teacher_inference", "teacher_cache", "student_train", "all"}:
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

        if args.phase in {"teacher_cache", "all"}:
            assert teacher_head is not None
            _cache_teacher_splits(
                loaded, replacement, teacher_head, data, settings, device
            )
            if args.phase == "teacher_cache":
                return 0

        if args.phase in {"student_train", "all"}:
            assert teacher_head is not None
            teacher_head.requires_grad_(False).eval()
            student_head = MLPHead(
                teacher_head.feature_dim,
                settings.hidden_dim,
                len(data.class_names),
                settings.dropout,
            ).to(device)
            if settings.initialize_student_mlp_from_teacher:
                student_head.load_state_dict(teacher_head.state_dict())
            train_indices, validation_indices = stratified_split_indices(
                data.train, settings.validation_fraction, settings.seed
            )
            cache_path = settings.output_dir / "teacher_cache" / "train.pt"
            expected_cache = expected_teacher_cache_metadata(
                split="train",
                samples=len(data.train),
                settings=settings,
                model=loaded.model,
                replacement=replacement,
                class_names=data.class_names,
            )
            cache_store = TeacherCacheStore(cache_path, expected_cache)
            cached_dataset = CachedTeacherDataset(data.train, cache_store)
            train_loader = make_cached_teacher_loader(
                cached_dataset,
                train_indices,
                settings.feature_batch_size,
                min(settings.num_workers, 2),
                settings.seed,
            )
            validation_subset = Subset(data.train, validation_indices)
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
    train_indices, validation_indices = stratified_split_indices(
        data.train, settings.validation_fraction, settings.seed
    )
    student_train = Subset(data.train, train_indices)
    validation = Subset(data.train, validation_indices)
    data.metadata.update(
        {
            "full_train_samples": len(data.train),
            "train_samples": len(student_train),
            "validation_samples": len(validation),
            "test_samples": len(data.test),
            "validation_fraction": settings.validation_fraction,
            "per_class_train_counts": class_counts(student_train),
            "per_class_validation_counts": class_counts(validation),
            "per_class_test_counts": class_counts(data.test),
        }
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
    vision_depth = int(vision_config.depth)
    vision_hidden_size = int(vision_config.hidden_size)
    if settings.replace_last_n_vision_blocks > vision_depth:
        raise ValueError(
            f"Cannot replace the last {settings.replace_last_n_vision_blocks} vision blocks; "
            f"runtime vision depth is {vision_depth}"
        )
    replace_start = vision_depth - settings.replace_last_n_vision_blocks
    groups: list[tuple[int, int]] = []
    for conversion in range(settings.optical_conversions):
        start = (
            replace_start
            + conversion * settings.teacher_blocks_per_conversion
        )
        end = start + settings.teacher_blocks_per_conversion - 1
        groups.append((start, end))
    surrogates = [
        OpticalVisionBlockSurrogate(
            hidden_size=vision_hidden_size,
            optical_dim=settings.optical_dim,
            optical_layers=settings.optical_layers,
            optical_field_size=settings.optical_field_size,
            optical_padding_size=settings.optical_padding_size,
            wavelength_nm=settings.wavelength_nm,
            pixel_pitch_um=settings.pixel_pitch_um,
            mask_distance_cm=settings.mask_distance_cm,
        ).to(device=device)
        for _ in groups
    ]
    return VisionBlockReplacement(loaded.model, groups, surrogates)


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
    replacement.clear_captures()
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
    metadata = _teacher_feature_metadata(split, data, settings)
    if settings.cache_features and path.is_file():
        features, labels, cached = load_feature_cache(path)
        if cached == metadata:
            _log(f"reusing validated teacher feature cache: split={split} path={path}")
            return features, labels
        changed = sorted(
            key
            for key in set(cached) | set(metadata)
            if cached.get(key) != metadata.get(key)
        )
        _log(
            f"teacher feature cache invalidated: split={split} "
            f"changed_fields={changed}"
        )
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


def _teacher_feature_metadata(
    split: str, data: DatasetBundle, settings: Settings
) -> dict[str, Any]:
    dataset = data.train if split == "train" else data.test
    return {
        "cache_schema_version": 2,
        "dataset": "bdd100k_weather4",
        "split": split,
        "samples": len(dataset),
        "num_classes": len(data.class_names),
        "class_names": list(data.class_names),
        "data_root": str(settings.data_root),
        "imagefolder_train": settings.imagefolder_train,
        "imagefolder_test": settings.imagefolder_test,
        "resize_to": settings.resize_to,
        "processor_min_pixels": settings.processor_min_pixels,
        "processor_max_pixels": settings.processor_max_pixels,
        "train_limit": settings.train_limit,
        "test_limit": settings.test_limit,
        "train_limit_per_class": settings.train_limit_per_class,
        "test_limit_per_class": settings.test_limit_per_class,
        "dataset_seed": settings.seed,
        "model_id": settings.model_id,
        "classification_prompt": settings.classification_prompt,
        "dtype": settings.dtype,
        "attn_implementation": settings.attn_implementation,
        "feature_source": "electronic_teacher_answer_position_hidden_state",
    }


def _cache_teacher_splits(
    loaded: LoadedBackbone,
    replacement: VisionBlockReplacement,
    teacher_head: MLPHead,
    data: DatasetBundle,
    settings: Settings,
    device: torch.device,
) -> None:
    replacement.use_teacher()
    for split, dataset in (("train", data.train), ("test", data.test)):
        loader = make_loader(
            dataset,
            settings.feature_batch_size,
            settings.num_workers,
            False,
            settings.seed,
        )
        build_teacher_cache(
            split=split,
            model=loaded.model,
            processor=loaded.processor,
            replacement=replacement,
            teacher_head=teacher_head,
            loader=loader,
            dataset_size=len(dataset),
            class_names=data.class_names,
            settings=settings,
            device=device,
            log=_log,
        )
        replacement.clear_captures()


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
    if "block_groups" not in checkpoint or "state_dicts" not in checkpoint:
        raise ValueError(
            "Optical checkpoint uses the obsolete single-block/multi-mask format. "
            "Run --phase student_train to create a five-conversion single-mask checkpoint."
        )
    checkpoint_groups = [tuple(group) for group in checkpoint["block_groups"]]
    if checkpoint_groups != replacement.block_groups:
        raise ValueError(
            f"Optical checkpoint block groups {checkpoint_groups} do not match "
            f"config {replacement.block_groups}"
        )
    replacement.load_state_dicts(checkpoint["state_dicts"])


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
        parameter.numel()
        for surrogate in replacement.surrogates
        for parameter in surrogate.parameters()
    )
    report.update(
        {
            "model_id": settings.model_id,
            "teacher": "full_electronic_qwen3_vl_2b_multimodal_mlp",
            "student": "qwen3_vl_2b_with_five_single_mask_optical_conversions",
            "replaced_vision_blocks": replacement.replaced_block_indices,
            "distillation_block_groups": [list(group) for group in replacement.block_groups],
            "optical_conversions": len(replacement.surrogates),
            "phase_masks_per_conversion": 1,
            "electronic_residual_bypass": False,
            "teacher_targets_cached": True,
            "optical_trainable_parameters": optical_parameters,
            "backbone_compute_dtype": settings.dtype,
            "attention_implementation": settings.attn_implementation,
            "optical_real_compute_dtype": "float32",
            "optical_complex_compute_dtype": "complex64",
            "optical_boundary_dtype": "matches_backbone_hidden_state",
        }
    )
    write_json(settings.output_dir / "model.json", report)


def _compare(settings: Settings, class_names: list[str]) -> None:
    model_report_path = settings.output_dir / "model.json"
    dataset_report_path = settings.output_dir / "dataset.json"
    if not model_report_path.is_file() or not dataset_report_path.is_file():
        raise FileNotFoundError("model.json and dataset.json are required for comparison")
    with model_report_path.open("r", encoding="utf-8") as handle:
        model_report = json.load(handle)
    with dataset_report_path.open("r", encoding="utf-8") as handle:
        dataset_report = json.load(handle)
    write_comparison(
        settings.output_dir,
        "bdd100k_weather4",
        class_names,
        settings.classification_prompt,
        {
            "vision_blocks": model_report["replaced_vision_blocks"],
            "distillation_block_groups": model_report["distillation_block_groups"],
            "optical_conversions": settings.optical_conversions,
            "teacher_blocks_per_conversion": settings.teacher_blocks_per_conversion,
            "phase_masks_per_conversion": 1,
            "optical_dim": settings.optical_dim,
            "optical_layers": settings.optical_layers,
            "optical_field_size": settings.optical_field_size,
            "optical_padding_size": settings.optical_padding_size,
            "wavelength_nm": settings.wavelength_nm,
            "pixel_pitch_um": settings.pixel_pitch_um,
            "mask_distance_cm": settings.mask_distance_cm,
            "electronic_residual_bypass": False,
        },
        {
            "hidden": settings.loss_hidden_weight,
            "kd": settings.loss_kd_weight,
            "ce": settings.loss_ce_weight,
            "temperature": settings.distill_temperature,
        },
        dataset_report.get("class_imbalance", {}),
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
