from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from .datasets import (
    CocoImageDataset,
    DUTSSaliencyDataset,
    collate_coco,
    collate_duts,
    prepare_coco,
    prepare_datasets,
    prepare_duts,
)
from .io_utils import environment_report, seed_everything, write_json
from .modeling import (
    DUTSSaliencyModel,
    NativeVisionFeatureExtractor,
    build_optical_backbone,
    load_vision_backbone,
    preprocess_vision,
    restore_packed_spatial,
)
from .pca import (
    fit_vision_pca,
    load_pca,
    pca_oracle_metrics,
)
from .settings import load_settings, save_resolved_config
from .teacher_cache import build_teacher_target_cache
from .training import (
    build_coco_cache_loader,
    test_duts_checkpoint,
    train_coco_backbone,
    train_duts,
)


PHASES = {
    "prepare_data",
    "fit_pca",
    "pca_oracle_check",
    "precompute_teacher",
    "shape_smoke",
    "coco_pretrain",
    "duts_train",
    "duts_test",
    "coco_all",
    "duts_all",
    "all",
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Qwen Vision PCA224 general feature distillation on COCO, followed "
            "by DUTS saliency pretraining of a three-stage Optical MoE16"
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--phase", choices=sorted(PHASES), default="all")
    args = parser.parse_args(argv)
    settings = load_settings(args.config)
    seed_everything(settings.random_seed)
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    settings.cache_root.mkdir(parents=True, exist_ok=True)
    save_resolved_config(settings)
    write_json(settings.output_dir / "environment.json", environment_report())

    if args.phase == "prepare_data":
        bundle = prepare_datasets(settings, persist=True)
        _print_dataset(bundle.coco.metadata, bundle.duts.metadata)
        return 0

    needs_coco = args.phase in {
        "fit_pca",
        "pca_oracle_check",
        "precompute_teacher",
        "shape_smoke",
        "coco_pretrain",
        "coco_all",
        "all",
    }
    needs_duts = args.phase in {
        "shape_smoke",
        "duts_train",
        "duts_test",
        "duts_all",
        "all",
    }
    coco = prepare_coco(settings, persist=True) if needs_coco else None
    duts = prepare_duts(settings, persist=True) if needs_duts else None

    device = _device(settings.device)
    loaded = load_vision_backbone(settings, device)
    save_resolved_config(settings)
    write_json(
        settings.output_dir / "qwen_vision.json",
        {
            "model_id": settings.model_id,
            "vision_depth": settings.vision_depth,
            "vision_hidden_size": settings.vision_hidden_size,
            "processor_min_pixels": settings.processor_min_pixels,
            "processor_max_pixels": settings.processor_max_pixels,
            "language_model_forward_used": False,
            "native_vision_blocks_used_for_teacher_cache": True,
            "native_vision_blocks_used_for_student": False,
            "load_time_sec": loaded.load_time_sec,
        },
    )

    if args.phase == "fit_pca":
        assert coco is not None
        _fit_pca(loaded, coco, settings, force=True)
        return 0
    if args.phase == "pca_oracle_check":
        assert coco is not None
        _run_pca_oracle(loaded, coco, settings)
        return 0
    if args.phase == "precompute_teacher":
        assert coco is not None
        _precompute_teacher(loaded, coco, settings)
        return 0
    if args.phase == "shape_smoke":
        assert coco is not None and duts is not None
        _shape_smoke(loaded, coco, duts, settings)
        return 0
    if args.phase == "coco_pretrain":
        assert coco is not None
        pca_metadata = _require_pca_metadata(settings, coco)
        train_coco_backbone(
            loaded,
            coco,
            settings,
            pca_metadata=pca_metadata,
        )
        return 0
    if args.phase == "duts_train":
        assert duts is not None
        train_duts(loaded, duts, settings)
        return 0
    if args.phase == "duts_test":
        assert duts is not None
        checkpoint = (
            settings.output_dir
            / "checkpoints"
            / "duts_student_best_train_loss.pt"
        )
        metrics = test_duts_checkpoint(
            loaded,
            duts,
            settings,
            checkpoint=checkpoint,
            save_predictions=True,
            save_examples=True,
        )
        _print_duts(metrics)
        return 0

    if args.phase in {"coco_all", "all"}:
        assert coco is not None
        if not settings.pca_path.is_file():
            _fit_pca(loaded, coco, settings, force=False)
        _run_pca_oracle(loaded, coco, settings)
        _precompute_teacher(loaded, coco, settings)
        pca_metadata = _require_pca_metadata(settings, coco)
        train_coco_backbone(
            loaded,
            coco,
            settings,
            pca_metadata=pca_metadata,
        )
        if args.phase == "coco_all":
            return 0

    if args.phase in {"duts_all", "all"}:
        assert duts is not None
        result = train_duts(loaded, duts, settings)
        _print_duts(result["test_metrics"])
        return 0
    raise RuntimeError(f"Unhandled phase: {args.phase}")


def _fit_pca(
    loaded: Any,
    coco: Any,
    settings: Any,
    *,
    force: bool,
) -> dict[str, Any]:
    if settings.pca_path.is_file() and not force:
        return _require_pca_metadata(settings, coco)
    count = min(
        len(coco.train_records),
        int(settings.pca_calibration_images),
    )
    generator = torch.Generator()
    generator.manual_seed(int(settings.random_seed))
    indices = torch.randperm(len(coco.train_records), generator=generator)[:count]
    records = tuple(coco.train_records[int(index)] for index in indices)
    dataset = CocoImageDataset(records, settings)
    loader = DataLoader(
        dataset,
        batch_size=settings.pca_batch_size,
        shuffle=False,
        num_workers=settings.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=settings.num_workers > 0,
        collate_fn=collate_coco,
    )
    teacher = NativeVisionFeatureExtractor(loaded)
    try:
        return fit_vision_pca(
            teacher,
            loaded.processor,
            loader,
            settings,
            coco_manifest_digest=coco.metadata["manifest_sha256"],
        )
    finally:
        teacher.close()


def _run_pca_oracle(
    loaded: Any,
    coco: Any,
    settings: Any,
) -> dict[str, Any]:
    projection, metadata = _load_verified_pca(settings, coco, loaded.device)
    teacher = NativeVisionFeatureExtractor(loaded)
    dataset = CocoImageDataset(coco.val_records, settings)
    loader = DataLoader(
        dataset,
        batch_size=settings.pca_batch_size,
        shuffle=False,
        num_workers=settings.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=settings.num_workers > 0,
        collate_fn=collate_coco,
    )
    try:
        result = pca_oracle_metrics(
            teacher,
            projection,
            loaded.processor,
            loader,
            settings,
            max_batches=16,
        )
    finally:
        teacher.close()
    print(
        f"PCA oracle: cosine={result['hidden_cosine']:.4f} "
        f"MSE={result['hidden_mse']:.6f} "
        f"relative_error={result['relative_reconstruction_error_mean_token']:.4f}",
        flush=True,
    )
    return result


def _precompute_teacher(
    loaded: Any,
    coco: Any,
    settings: Any,
) -> None:
    projection, metadata = _load_verified_pca(settings, coco, loaded.device)
    # Runtime-only identity attributes. They are metadata, not parameters or
    # buffers and are never attached to the student.
    projection.metadata = metadata
    projection.projection_sha256 = metadata["projection_sha256"]
    teacher = NativeVisionFeatureExtractor(loaded)
    try:
        for split in ("train", "val"):
            loader = build_coco_cache_loader(
                coco,
                settings,
                split=split,
            )
            build_teacher_target_cache(
                split,
                teacher,
                projection,
                loaded.processor,
                loader,
                settings,
                dataset_manifest_digest=coco.metadata["manifest_sha256"],
            )
    finally:
        teacher.close()


@torch.no_grad()
def _shape_smoke(
    loaded: Any,
    coco: Any,
    duts: Any,
    settings: Any,
) -> dict[str, Any]:
    teacher = NativeVisionFeatureExtractor(loaded)
    coco_dataset = CocoImageDataset(coco.train_records[:1], settings)
    coco_batch = collate_coco([coco_dataset[0]])
    coco_inputs = preprocess_vision(
        loaded.processor,
        coco_batch["images"],
        loaded.device,
    )
    teacher_hidden, teacher_lengths = teacher.extract_packed(
        coco_inputs["pixel_values"],
        coco_inputs["image_grid_thw"],
    )
    teacher.close()
    if max(teacher_lengths) > settings.max_visual_tokens:
        raise RuntimeError(
            f"visual token count {max(teacher_lengths)} exceeds "
            f"max_visual_tokens={settings.max_visual_tokens}"
        )

    backbone = build_optical_backbone(
        loaded,
        settings,
        release_native_to_cpu=True,
    )
    optical_packed, optical_lengths, ccd = backbone(
        coco_inputs["pixel_values"],
        coco_inputs["image_grid_thw"],
    )
    duts_dataset = DUTSSaliencyDataset(
        duts.train_records[:1],
        settings,
        training=False,
    )
    duts_batch = collate_duts([duts_dataset[0]])
    duts_inputs = preprocess_vision(
        loaded.processor,
        duts_batch["images"],
        loaded.device,
    )
    saliency = DUTSSaliencyModel(backbone, settings)
    logits, spatial, duts_ccd = saliency(
        duts_inputs["pixel_values"],
        duts_inputs["image_grid_thw"],
    )
    report = {
        "input_rgb_shape": [1, 3, 224, 224],
        "coco_image_grid_thw": coco_inputs["image_grid_thw"]
        .cpu()
        .long()
        .tolist(),
        "teacher_final_pre_merger_hidden_shape": list(teacher_hidden.shape),
        "teacher_token_lengths": teacher_lengths,
        "student_packed_feature_shape": list(optical_packed.shape),
        "student_token_lengths": optical_lengths,
        "student_ccd_shape": list(ccd.shape),
        "expert_stages": len(backbone.core.expert_layers),
        "experts_per_stage": settings.num_experts,
        "recombiner_alpha": float(backbone.recombiner.alpha),
        "duts_image_grid_thw": duts_inputs["image_grid_thw"]
        .cpu()
        .long()
        .tolist(),
        "duts_spatial_feature_shape": list(spatial.shape),
        "duts_ccd_shape": list(duts_ccd.shape),
        "duts_mask_logits_shape": list(logits.shape),
        "pca_present_in_student": False,
        "unused_hidden_restore_adapter_present": hasattr(
            backbone.core, "output_adapter"
        ),
    }
    expected = {
        "student_packed_feature_shape": [teacher_lengths[0], 224],
        "student_ccd_shape": [1, 224, 224],
        "duts_mask_logits_shape": [1, 1, 224, 224],
    }
    for key, value in expected.items():
        if report[key] != value:
            raise RuntimeError(f"Shape smoke mismatch for {key}: {report}")
    if report["expert_stages"] != 3:
        raise RuntimeError(f"Expected three optical stages: {report}")
    if report["unused_hidden_restore_adapter_present"]:
        raise RuntimeError("Unused trainable 224->1024 adapter was not removed")
    write_json(settings.output_dir / "shape_smoke.json", report)
    for name, value in report.items():
        print(f"{name}: {value}", flush=True)
    return report


def _load_verified_pca(
    settings: Any,
    coco: Any,
    device: torch.device,
) -> tuple[Any, dict[str, Any]]:
    projection, metadata = load_pca(settings.pca_path, device)
    expected = {
        "model_id": settings.model_id,
        "coco_manifest_sha256": coco.metadata["manifest_sha256"],
        "processor_min_pixels": settings.processor_min_pixels,
        "processor_max_pixels": settings.processor_max_pixels,
        "rank": settings.pca_rank,
        "teacher_dim": 1024,
    }
    mismatches = {
        key: (metadata.get(key), value)
        for key, value in expected.items()
        if metadata.get(key) != value
    }
    if mismatches:
        raise RuntimeError(
            f"PCA checkpoint is incompatible with current data/config: "
            f"{mismatches}. Delete {settings.pca_path} and rerun fit_pca."
        )
    return projection, metadata


def _require_pca_metadata(settings: Any, coco: Any) -> dict[str, Any]:
    _, metadata = _load_verified_pca(settings, coco, torch.device("cpu"))
    return metadata


def _device(value: str) -> torch.device:
    if value == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "Configuration requests CUDA but torch.cuda.is_available() is false"
        )
    return torch.device(value)


def _print_dataset(coco: dict[str, Any], duts: dict[str, Any]) -> None:
    print(
        f"COCO train={coco['train_images']:,} val={coco['val_images']:,}; "
        f"DUTS train={duts['train_images']:,} test={duts['test_images']:,}",
        flush=True,
    )


def _print_duts(metrics: dict[str, Any]) -> None:
    print(
        f"DUTS test: mIoU={metrics['mean_iou']:.4f} "
        f"Dice={metrics['mean_dice']:.4f} MAE={metrics['mae']:.4f} "
        f"PixelAcc={metrics['pixel_accuracy']:.4f}",
        flush=True,
    )


if __name__ == "__main__":
    raise SystemExit(main())
