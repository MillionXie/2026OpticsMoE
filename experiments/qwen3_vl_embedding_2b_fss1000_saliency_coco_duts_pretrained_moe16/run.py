from __future__ import annotations

import argparse
from pathlib import Path

import torch

from experiments.qwen3_vl_embedding_2b_coco_duts_vision_optical_moe16_pretrain.io_utils import (
    environment_report,
    seed_everything,
    write_json,
)
from experiments.qwen3_vl_embedding_2b_coco_duts_vision_optical_moe16_pretrain.modeling import (
    build_duts_model,
    load_vision_backbone,
    preprocess_vision,
)

from .datasets import FSSSaliencyDataset, collate_saliency, prepare_fss1000
from .settings import load_settings
from .training import (
    evaluate_checkpoint,
    load_duts_initialization,
    train_fss,
)


PHASES = {"prepare_data", "shape_smoke", "train", "test", "all"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Fine-tune the COCO/DUTS-pretrained three-stage Optical MoE16 on "
            "the official class-disjoint FSS-1000 saliency split"
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--phase", choices=sorted(PHASES), default="all")
    args = parser.parse_args(argv)

    settings = load_settings(args.config)
    seed_everything(settings.random_seed)
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(settings.output_dir / "resolved_config.json", settings.to_dict())
    write_json(settings.output_dir / "environment.json", environment_report())
    bundle = prepare_fss1000(settings, persist=True)
    _print_dataset(bundle.metadata)
    if args.phase == "prepare_data":
        return 0

    device = _device(settings.device)
    loaded = load_vision_backbone(settings, device)
    write_json(settings.output_dir / "resolved_config.json", settings.to_dict())
    write_json(
        settings.output_dir / "qwen_vision.json",
        {
            "model_id": settings.model_id,
            "vision_depth": settings.vision_depth,
            "vision_hidden_size": settings.vision_hidden_size,
            "processor_min_pixels": settings.processor_min_pixels,
            "processor_max_pixels": settings.processor_max_pixels,
            "language_model_forward_used": False,
            "native_vision_blocks_used_for_student": False,
            "expert_stages": settings.expert_layers,
        },
    )

    if args.phase == "shape_smoke":
        _shape_smoke(loaded, bundle, settings)
        return 0
    if args.phase in {"train", "all"}:
        result = train_fss(loaded, bundle, settings)
        _print_metrics(result["final_test_metrics"])
        return 0
    if args.phase == "test":
        checkpoint = (
            settings.output_dir
            / "checkpoints"
            / "fss_student_best_train_loss.pt"
        )
        metrics = evaluate_checkpoint(
            loaded,
            bundle,
            settings,
            checkpoint=checkpoint,
            save_visualizations=True,
        )
        _print_metrics(metrics)
        return 0
    raise RuntimeError(f"Unhandled phase: {args.phase}")


def _shape_smoke(loaded: object, bundle: object, settings: object) -> None:
    dataset = FSSSaliencyDataset(
        bundle.train_records[:1],
        settings,
        training=False,
    )
    batch = collate_saliency([dataset[0]])
    model = build_duts_model(loaded, settings)
    transfer = load_duts_initialization(model, settings.source_checkpoint)
    inputs = preprocess_vision(
        loaded.processor,
        batch["images"],
        loaded.device,
    )
    with torch.no_grad():
        logits, spatial, ccd = model(
            inputs["pixel_values"],
            inputs["image_grid_thw"],
        )
    expected_tokens = int(inputs["image_grid_thw"][0].prod())
    if logits.shape != (1, 1, 224, 224):
        raise RuntimeError(f"Unexpected FSS logits shape: {tuple(logits.shape)}")
    if ccd.shape != (1, 224, 224):
        raise RuntimeError(f"Unexpected CCD shape: {tuple(ccd.shape)}")
    if spatial.shape != (1, 224, *inputs["image_grid_thw"][0, 1:].tolist()):
        raise RuntimeError(
            f"Unexpected restored spatial shape: {tuple(spatial.shape)}"
        )
    report = {
        "input_image_shape": [1, 3, 224, 224],
        "image_grid_thw": inputs["image_grid_thw"].cpu().tolist(),
        "visual_token_count": expected_tokens,
        "ccd_shape": list(ccd.shape),
        "restored_spatial_shape": list(spatial.shape),
        "mask_logits_shape": list(logits.shape),
        "source_transfer": transfer,
    }
    write_json(settings.output_dir / "shape_smoke.json", report)
    print(f"shape smoke passed: {report}", flush=True)


def _device(value: str) -> torch.device:
    if value.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return torch.device(value)


def _print_dataset(metadata: dict[str, object]) -> None:
    print(
        "FSS-1000 transfer split: "
        f"train_classes={metadata['train_classes']} "
        f"test_classes={metadata['test_classes']} "
        f"train_images={metadata['train_images']} "
        f"test_images={metadata['test_images']}",
        flush=True,
    )


def _print_metrics(metrics: dict[str, object]) -> None:
    print(
        "FSS-1000 final: "
        f"mIoU={float(metrics['mean_iou']):.4f} "
        f"Dice={float(metrics['mean_dice']):.4f} "
        f"MAE={float(metrics['mae']):.4f} "
        f"PixelAcc={float(metrics['pixel_accuracy']):.4f}",
        flush=True,
    )


if __name__ == "__main__":
    raise SystemExit(main())

