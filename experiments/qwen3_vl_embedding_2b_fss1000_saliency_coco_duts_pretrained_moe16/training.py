from __future__ import annotations

import contextlib
import time
from pathlib import Path
from typing import Any

import torch
from torch import nn

from experiments.qwen3_vl_embedding_2b_coco_duts_vision_optical_moe16_pretrain.io_utils import (
    atomic_torch_save,
    torch_load,
    write_csv,
    write_json,
)
from experiments.qwen3_vl_embedding_2b_coco_duts_vision_optical_moe16_pretrain.modeling import (
    DUTSSaliencyModel,
    LoadedVisionBackbone,
    build_duts_model,
    preprocess_vision,
    trainable_parameter_report,
)
from experiments.qwen3_vl_embedding_2b_coco_duts_vision_optical_moe16_pretrain.objectives import (
    SegmentationAccumulator,
    segmentation_loss,
)
from experiments.qwen3_vl_embedding_2b_coco_duts_vision_optical_moe16_pretrain.training import (
    evaluate_duts,
)
from experiments.qwen3_vl_embedding_2b_coco_duts_vision_optical_moe16_pretrain.visualization import (
    save_phase_masks,
    save_segmentation_examples,
    save_training_curves,
)
from experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval.optics.physical import (
    phase_dc_loss,
)

from .datasets import DatasetBundle, build_loaders


HISTORY_FIELDS = [
    "epoch",
    "training_phase",
    "learning_rate_optical",
    "learning_rate_recombiner",
    "learning_rate_head",
    "train_loss",
    "train_segmentation_loss",
    "train_bce",
    "train_dice_loss",
    "train_soft_iou_loss",
    "train_boundary_loss",
    "train_router_balance",
    "train_router_importance",
    "train_mean_iou",
    "train_mean_dice",
    "train_mae",
    "train_pixel_accuracy",
    "test_loss",
    "test_mean_iou",
    "test_mean_dice",
    "test_mae",
    "test_pixel_accuracy",
    "alpha",
    "epoch_time_sec",
]


def load_duts_initialization(
    model: DUTSSaliencyModel,
    checkpoint: Path,
) -> dict[str, Any]:
    """Strictly restore backbone, recombiner, and segmentation head."""

    if not checkpoint.is_file():
        raise FileNotFoundError(
            "DUTS-pretrained checkpoint is missing: "
            f"{checkpoint}. Finish the COCO/DUTS pretraining first or update "
            "source.duts_checkpoint."
        )
    payload = torch_load(checkpoint)
    required = {"checkpoint_type", "backbone", "head_state_dict", "epoch"}
    if not isinstance(payload, dict) or not required.issubset(payload):
        missing = sorted(required - set(payload if isinstance(payload, dict) else {}))
        raise RuntimeError(
            f"Invalid DUTS transfer checkpoint {checkpoint}; missing {missing}"
        )
    if payload["checkpoint_type"] != "duts_saliency_pretraining":
        raise RuntimeError(
            f"Expected duts_saliency_pretraining checkpoint, got "
            f"{payload['checkpoint_type']!r}"
        )
    backbone = payload["backbone"]
    architecture = backbone.get("architecture", {})
    expected = model.backbone.specification()
    for key in (
        "expert_stages",
        "experts_per_stage",
        "top_k",
        "expert_size",
        "active_size",
        "canvas_size",
        "ccd_shape",
    ):
        if architecture.get(key) != expected.get(key):
            raise RuntimeError(
                f"Source checkpoint architecture mismatch for {key}: "
                f"saved={architecture.get(key)!r}, current={expected.get(key)!r}"
            )
    model.backbone.core.load_state_dict(
        backbone["core_state_dict"],
        strict=True,
    )
    model.backbone.recombiner.load_state_dict(
        backbone["recombiner_state_dict"],
        strict=True,
    )
    model.head.load_state_dict(payload["head_state_dict"], strict=True)
    return {
        "checkpoint": str(checkpoint),
        "source_epoch": int(payload["epoch"]),
        "source_train_metrics": payload.get("train_metrics"),
        "source_test_metrics_observation_only": payload.get(
            "test_metrics_observation_only"
        ),
        "loaded_components": [
            "optical_core",
            "ccd_residual_recombiner",
            "segmentation_head",
        ],
        "optimizer_restored": False,
    }


def train_fss(
    loaded: LoadedVisionBackbone,
    bundle: DatasetBundle,
    settings: Any,
) -> dict[str, Any]:
    model = build_duts_model(loaded, settings)
    transfer = load_duts_initialization(model, settings.source_checkpoint)
    train_loader, test_loader = build_loaders(bundle, settings)

    report = trainable_parameter_report(model, prefix="fss_transfer_saliency")
    report["backbone"] = model.backbone.specification()
    report["segmentation_head"] = model.head.specification()
    report["transfer_initialization"] = transfer
    report["schedule"] = {
        "head_warmup_epochs": settings.duts_head_warmup_epochs,
        "joint_finetune_epochs": settings.duts_finetune_epochs,
        "optical_learning_rate": settings.duts_optical_learning_rate,
        "recombiner_learning_rate": settings.duts_recombiner_learning_rate,
        "head_learning_rate": settings.duts_head_learning_rate,
        "optimizer_state_restored": False,
    }
    write_json(settings.output_dir / "model.json", report)
    _print_trainable(report)

    initial_metrics, _ = evaluate_duts(
        model,
        loaded.processor,
        test_loader,
        settings,
        save_predictions=False,
    )
    initial_metrics.update(
        {
            "stage": "before_fss_finetuning",
            "source_checkpoint": str(settings.source_checkpoint),
            "test_used_for_checkpoint_selection": False,
        }
    )
    write_json(
        settings.output_dir / "metrics" / "initial_transfer_test.json",
        initial_metrics,
    )
    print(
        "FSS transfer before fine-tuning: "
        f"mIoU={initial_metrics['mean_iou']:.4f} "
        f"Dice={initial_metrics['mean_dice']:.4f}",
        flush=True,
    )

    best_train_loss = float("inf")
    history: list[dict[str, Any]] = []
    checkpoint_dir = settings.output_dir / "checkpoints"
    best_path = checkpoint_dir / "fss_student_best_train_loss.pt"
    last_path = checkpoint_dir / "fss_student_last.pt"
    optimizer: torch.optim.Optimizer | None = None
    current_phase = ""

    for epoch in range(1, settings.total_epochs + 1):
        warmup = epoch <= settings.duts_head_warmup_epochs
        phase = "head_warmup" if warmup else "joint_finetune"
        if phase != current_phase:
            _configure_trainability(model, warmup=warmup)
            optimizer = _build_optimizer(model, settings, warmup=warmup)
            current_phase = phase
            print(
                f"[FSS] entering {phase}; optimizer starts fresh with "
                f"{len(optimizer.param_groups)} group(s)",
                flush=True,
            )
        assert optimizer is not None
        started = time.perf_counter()
        train_metrics, parts = _run_epoch(
            model,
            loaded.processor,
            train_loader,
            settings,
            optimizer=optimizer,
            epoch=epoch,
            phase=phase,
            detach_backbone=warmup,
        )
        if settings.evaluate_duts_test_each_epoch:
            test_metrics, _ = evaluate_duts(
                model,
                loaded.processor,
                test_loader,
                settings,
                save_predictions=False,
            )
        else:
            test_metrics = _empty_metrics()

        improved = float(train_metrics["loss"]) < best_train_loss
        if improved:
            best_train_loss = float(train_metrics["loss"])
        payload = {
            "checkpoint_type": "fss1000_coco_duts_transfer_finetune",
            "epoch": epoch,
            "training_phase": phase,
            "backbone": model.backbone.checkpoint_state(),
            "head_state_dict": model.head.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_metrics": train_metrics,
            "test_metrics_observation_only": test_metrics,
            "selection_criterion": "minimum_train_loss",
            "test_used_for_selection": False,
            "source_duts_checkpoint": str(settings.source_checkpoint),
            "source_duts_epoch": transfer["source_epoch"],
        }
        atomic_torch_save(last_path, payload)
        if improved:
            atomic_torch_save(best_path, payload)
        if (
            settings.checkpoint_interval_epochs > 0
            and epoch % settings.checkpoint_interval_epochs == 0
        ):
            atomic_torch_save(
                checkpoint_dir / f"fss_student_epoch_{epoch:04d}.pt",
                payload,
            )

        row = {
            "epoch": epoch,
            "training_phase": phase,
            "learning_rate_optical": _optional_lr(optimizer, "optical"),
            "learning_rate_recombiner": _optional_lr(
                optimizer, "recombiner"
            ),
            "learning_rate_head": _optional_lr(optimizer, "head"),
            "train_loss": train_metrics["loss"],
            "train_segmentation_loss": parts["segmentation_loss"],
            "train_bce": parts["bce"],
            "train_dice_loss": parts["dice_loss"],
            "train_soft_iou_loss": parts["soft_iou_loss"],
            "train_boundary_loss": parts["boundary_loss"],
            "train_router_balance": parts["router_balance"],
            "train_router_importance": parts["router_importance"],
            "train_mean_iou": train_metrics["mean_iou"],
            "train_mean_dice": train_metrics["mean_dice"],
            "train_mae": train_metrics["mae"],
            "train_pixel_accuracy": train_metrics["pixel_accuracy"],
            "test_loss": test_metrics["loss"],
            "test_mean_iou": test_metrics["mean_iou"],
            "test_mean_dice": test_metrics["mean_dice"],
            "test_mae": test_metrics["mae"],
            "test_pixel_accuracy": test_metrics["pixel_accuracy"],
            "alpha": float(model.backbone.recombiner.alpha.detach()),
            "epoch_time_sec": time.perf_counter() - started,
        }
        history.append(row)
        write_csv(
            settings.output_dir / "metrics" / "training_history.csv",
            history,
            HISTORY_FIELDS,
        )
        write_json(settings.output_dir / "metrics" / "latest_epoch.json", row)
        save_training_curves(
            settings.output_dir / "figures" / "training_curves.png",
            history,
            stage="FSS-1000 transfer fine-tuning",
        )
        if (
            settings.visualization_interval_epochs > 0
            and epoch % settings.visualization_interval_epochs == 0
        ):
            save_segmentation_examples(
                model,
                loaded.processor,
                test_loader,
                settings,
                epoch=epoch,
                phase="test_observation",
                dataset_slug="fss1000",
            )
        print(
            f"FSS epoch {epoch:03d}/{settings.total_epochs} phase={phase} "
            f"train_loss={train_metrics['loss']:.5f} "
            f"train_mIoU={train_metrics['mean_iou']:.4f} "
            f"test_mIoU(observation)={test_metrics['mean_iou']:.4f} "
            f"test_Dice={test_metrics['mean_dice']:.4f} "
            f"balance={parts['router_balance']:.5f}",
            flush=True,
        )

    final = evaluate_checkpoint(
        loaded,
        bundle,
        settings,
        checkpoint=best_path,
        save_visualizations=True,
    )
    return {
        "initial_test_metrics": initial_metrics,
        "final_test_metrics": final,
        "best_checkpoint": str(best_path),
        "last_checkpoint": str(last_path),
        "best_train_loss": best_train_loss,
    }


def evaluate_checkpoint(
    loaded: LoadedVisionBackbone,
    bundle: DatasetBundle,
    settings: Any,
    *,
    checkpoint: Path,
    save_visualizations: bool,
) -> dict[str, Any]:
    if not checkpoint.is_file():
        raise FileNotFoundError(f"FSS checkpoint is missing: {checkpoint}")
    payload = torch_load(checkpoint)
    if (
        not isinstance(payload, dict)
        or payload.get("checkpoint_type")
        != "fss1000_coco_duts_transfer_finetune"
    ):
        raise RuntimeError(f"Invalid FSS transfer checkpoint: {checkpoint}")
    model = build_duts_model(loaded, settings)
    model.backbone.core.load_state_dict(
        payload["backbone"]["core_state_dict"],
        strict=True,
    )
    model.backbone.recombiner.load_state_dict(
        payload["backbone"]["recombiner_state_dict"],
        strict=True,
    )
    model.head.load_state_dict(payload["head_state_dict"], strict=True)
    _, test_loader = build_loaders(bundle, settings)
    metrics, rows = evaluate_duts(
        model,
        loaded.processor,
        test_loader,
        settings,
        save_predictions=True,
    )
    metrics.update(
        {
            "checkpoint": str(checkpoint),
            "checkpoint_epoch": int(payload["epoch"]),
            "checkpoint_selection": "minimum_train_loss",
            "test_used_for_selection": False,
            "source_duts_checkpoint": payload["source_duts_checkpoint"],
        }
    )
    write_json(settings.output_dir / "metrics" / "test_metrics.json", metrics)
    write_csv(
        settings.output_dir / "metrics" / "test_predictions.csv",
        rows,
        [
            "sample_id",
            "image_path",
            "mask_path",
            "mean_iou",
            "mean_dice",
            "mae",
        ],
    )
    if save_visualizations:
        save_segmentation_examples(
            model,
            loaded.processor,
            test_loader,
            settings,
            epoch=int(payload["epoch"]),
            phase="final_test",
            dataset_slug="fss1000",
        )
        save_phase_masks(
            model.backbone.core,
            settings.output_dir / "figures" / "final_phase_masks",
        )
    return metrics


def _run_epoch(
    model: DUTSSaliencyModel,
    processor: Any,
    loader: Any,
    settings: Any,
    *,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    phase: str,
    detach_backbone: bool,
) -> tuple[dict[str, Any], dict[str, float]]:
    model.train(True)
    accumulator = SegmentationAccumulator()
    sums = {
        "segmentation_loss": 0.0,
        "bce": 0.0,
        "dice_loss": 0.0,
        "soft_iou_loss": 0.0,
        "boundary_loss": 0.0,
        "router_balance": 0.0,
        "router_importance": 0.0,
        "phase_dc": 0.0,
    }
    samples = 0
    for batch_index, batch in enumerate(loader, start=1):
        inputs = preprocess_vision(
            processor,
            batch["images"],
            model.backbone.device,
        )
        masks = batch["masks"].to(model.backbone.device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with _autocast(settings, model.backbone.device):
            logits, _, _ = model(
                inputs["pixel_values"],
                inputs["image_grid_thw"],
                detach_backbone=detach_backbone,
            )
            segmentation, parts = segmentation_loss(
                logits,
                masks,
                bce_weight=settings.bce_weight,
                dice_weight=settings.dice_weight,
                soft_iou_weight=settings.soft_iou_weight,
                boundary_weight=settings.boundary_weight,
            )
            balance, importance = model.router_losses()
            dc = (
                phase_dc_loss(model.backbone)
                if not detach_backbone and settings.phase_dc_weight > 0.0
                else segmentation.new_zeros(())
            )
            total = (
                segmentation
                + settings.router_balance_weight * balance.float()
                + settings.router_importance_weight * importance.float()
                + settings.phase_dc_weight * dc
            )
        if not torch.isfinite(total):
            raise RuntimeError(
                f"Non-finite FSS loss at epoch={epoch}, batch={batch_index}"
            )
        total.backward()
        optimizer.step()

        batch_size = int(masks.shape[0])
        accumulator.update(logits.detach(), masks, loss=float(total.detach()))
        values = {
            "segmentation_loss": segmentation,
            **parts,
            "router_balance": balance,
            "router_importance": importance,
            "phase_dc": dc,
        }
        for name in sums:
            sums[name] += float(values[name].detach()) * batch_size
        samples += batch_size
        if batch_index % settings.log_interval_batches == 0:
            current = accumulator.compute()
            print(
                f"[FSS {phase}] epoch={epoch} batch={batch_index:,}/"
                f"{len(loader):,} loss={current['loss']:.5f} "
                f"mIoU={current['mean_iou']:.4f} "
                f"Dice={current['mean_dice']:.4f} "
                f"balance={float(balance):.5f} phase_dc={float(dc):.5f}",
                flush=True,
            )
    return accumulator.compute(), {
        name: value / max(1, samples) for name, value in sums.items()
    }


def _configure_trainability(
    model: DUTSSaliencyModel,
    *,
    warmup: bool,
) -> None:
    model.backbone.core.requires_grad_(not warmup)
    model.backbone.recombiner.requires_grad_(not warmup)
    model.head.requires_grad_(True)
    model.backbone.visual.requires_grad_(False)


def _build_optimizer(
    model: DUTSSaliencyModel,
    settings: Any,
    *,
    warmup: bool,
) -> torch.optim.Optimizer:
    groups: list[dict[str, Any]] = []
    if not warmup:
        phase_parameters = [
            parameter
            for name, parameter in model.backbone.core.named_parameters()
            if parameter.requires_grad and "raw_phase" in name
        ]
        phase_ids = {id(parameter) for parameter in phase_parameters}
        groups.extend(
            [
                {
                    "params": [
                        parameter
                        for parameter in model.backbone.core.parameters()
                        if parameter.requires_grad and id(parameter) not in phase_ids
                    ],
                    "lr": settings.duts_optical_learning_rate,
                    "name": "optical",
                },
                {
                    "params": phase_parameters,
                    "lr": settings.duts_phase_learning_rate,
                    "name": "phase",
                },
                {
                    "params": [
                        parameter
                        for parameter in model.backbone.recombiner.parameters()
                        if parameter.requires_grad
                    ],
                    "lr": settings.duts_recombiner_learning_rate,
                    "name": "recombiner",
                },
            ]
        )
    groups.append(
        {
            "params": [
                parameter
                for parameter in model.head.parameters()
                if parameter.requires_grad
            ],
            "lr": settings.duts_head_learning_rate,
            "name": "head",
        }
    )
    return torch.optim.AdamW(groups, weight_decay=settings.duts_weight_decay)


def _optional_lr(optimizer: torch.optim.Optimizer, name: str) -> float:
    for group in optimizer.param_groups:
        if group.get("name") == name:
            return float(group["lr"])
    return 0.0


def _autocast(settings: Any, device: torch.device) -> Any:
    if not settings.amp_enabled or device.type != "cuda":
        return contextlib.nullcontext()
    dtype = torch.bfloat16 if settings.dtype == "bfloat16" else torch.float16
    return torch.autocast("cuda", dtype=dtype)


def _empty_metrics() -> dict[str, float | int]:
    return {
        "loss": 0.0,
        "mean_iou": 0.0,
        "mean_dice": 0.0,
        "mean_f1": 0.0,
        "mae": 0.0,
        "pixel_accuracy": 0.0,
        "samples": 0,
        "pixels": 0,
    }


def _print_trainable(report: dict[str, Any]) -> None:
    print(
        f"FSS trainable parameters={report['trainable_parameters']:,} "
        f"tensors={report['trainable_tensors']}",
        flush=True,
    )
    for row in report["trainable_parameter_list"]:
        print(
            f"  {row['name']} shape={row['shape']} "
            f"params={row['parameters']:,}",
            flush=True,
        )
