from __future__ import annotations

import contextlib
import time
from pathlib import Path
from typing import Any

import torch

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
    segmentation_loss,
)
from experiments.qwen3_vl_embedding_2b_coco_duts_vision_optical_moe16_pretrain.visualization import (
    save_phase_masks,
    save_segmentation_examples,
    save_training_curves,
)

from .datasets import DatasetBundle, build_loaders
from .metrics import ISICSegmentationAccumulator, per_sample_metrics


HISTORY_FIELDS = [
    "epoch",
    "training_phase",
    "learning_rate_optical",
    "learning_rate_router",
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
    "train_sensitivity",
    "train_specificity",
    "test_loss",
    "test_mean_iou",
    "test_mean_dice",
    "test_mae",
    "test_pixel_accuracy",
    "test_sensitivity",
    "test_specificity",
    "alpha",
    "epoch_time_sec",
]


def initialize_model(
    model: DUTSSaliencyModel,
    settings: Any,
) -> dict[str, Any]:
    if settings.initialization_mode == "scratch_end_to_end":
        return {
            "mode": "scratch_end_to_end",
            "checkpoint": None,
            "loaded_components": [],
            "optimizer_restored": False,
            "all_task_modules_train_from_epoch_1": True,
        }
    checkpoint = settings.source_checkpoint
    if checkpoint is None or not checkpoint.is_file():
        raise FileNotFoundError(
            "Initialization checkpoint is missing: "
            f"{checkpoint}. Run the source pretraining or update the config."
        )
    payload = torch_load(checkpoint)
    if settings.initialization_mode == "isic_checkpoint_finetune":
        if (
            not isinstance(payload, dict)
            or payload.get("checkpoint_type")
            != "isic2016_skin_lesion_segmentation"
        ):
            raise RuntimeError(
                "Expected an isic2016_skin_lesion_segmentation checkpoint, "
                f"got {checkpoint}"
            )
        model.backbone.core.load_state_dict(
            payload["backbone"]["core_state_dict"],
            strict=True,
        )
        model.backbone.recombiner.load_state_dict(
            payload["backbone"]["recombiner_state_dict"],
            strict=True,
        )
        model.head.load_state_dict(payload["head_state_dict"], strict=True)
        return {
            "mode": "isic_checkpoint_finetune",
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
            "purpose": (
                "continue the previous head-only run after correcting optical "
                "trainability"
            ),
        }
    required = {"checkpoint_type", "backbone", "head_state_dict", "epoch"}
    if not isinstance(payload, dict) or not required.issubset(payload):
        missing = sorted(
            required - set(payload if isinstance(payload, dict) else {})
        )
        raise RuntimeError(
            f"Invalid pretrained checkpoint {checkpoint}; missing {missing}"
        )
    if payload["checkpoint_type"] != "duts_saliency_pretraining":
        raise RuntimeError(
            "Expected duts_saliency_pretraining checkpoint, got "
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
                f"Pretrained architecture mismatch for {key}: "
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
    loaded_components = ["optical_core", "ccd_residual_recombiner"]
    if settings.load_pretrained_segmentation_head:
        model.head.load_state_dict(payload["head_state_dict"], strict=True)
        loaded_components.append("segmentation_head")
    return {
        "mode": "coco_duts_pretrained",
        "checkpoint": str(checkpoint),
        "source_epoch": int(payload["epoch"]),
        "source_train_metrics": payload.get("train_metrics"),
        "source_test_metrics_observation_only": payload.get(
            "test_metrics_observation_only"
        ),
        "loaded_components": loaded_components,
        "optimizer_restored": False,
    }


def train_isic(
    loaded: LoadedVisionBackbone,
    bundle: DatasetBundle,
    settings: Any,
) -> dict[str, Any]:
    model = build_duts_model(loaded, settings)
    initialization = initialize_model(model, settings)
    train_loader, test_loader = build_loaders(bundle, settings)

    report = trainable_parameter_report(model, prefix="isic2016")
    report["backbone"] = model.backbone.specification()
    report["segmentation_head"] = model.head.specification()
    report["initialization"] = initialization
    report["schedule"] = {
        "head_warmup_epochs": settings.head_warmup_epochs,
        "joint_finetune_epochs": settings.joint_finetune_epochs,
        "optical_learning_rate": settings.optical_learning_rate,
        "router_learning_rate": settings.router_learning_rate,
        "recombiner_learning_rate": settings.recombiner_learning_rate,
        "head_learning_rate": settings.head_learning_rate,
        "test_used_for_checkpoint_selection": False,
    }
    write_json(settings.output_dir / "model.json", report)
    _print_trainable(report)

    initial_metrics, _ = evaluate_model(
        model,
        loaded.processor,
        test_loader,
        settings,
        save_predictions=False,
    )
    initial_metrics.update(
        {
            "stage": "before_isic_training",
            "initialization_mode": settings.initialization_mode,
            "test_used_for_checkpoint_selection": False,
        }
    )
    write_json(
        settings.output_dir / "metrics" / "initial_test_observation.json",
        initial_metrics,
    )

    history: list[dict[str, Any]] = []
    best_train_loss = float("inf")
    checkpoint_dir = settings.output_dir / "checkpoints"
    best_path = checkpoint_dir / "isic_student_best_train_loss.pt"
    last_path = checkpoint_dir / "isic_student_last.pt"
    optimizer: torch.optim.Optimizer | None = None
    current_phase = ""

    for epoch in range(1, settings.total_epochs + 1):
        warmup = epoch <= settings.head_warmup_epochs
        phase = "head_warmup" if warmup else "joint_end_to_end"
        if phase != current_phase:
            _configure_trainability(model, warmup=warmup)
            optimizer = _build_optimizer(model, settings, warmup=warmup)
            effective_counts = _optimizer_parameter_counts(optimizer)
            current_phase = phase
            print(
                f"[ISIC] entering {phase}; fresh optimizer with "
                f"{len(optimizer.param_groups)} parameter group(s): "
                f"{effective_counts}",
                flush=True,
            )
            write_json(
                settings.output_dir
                / "metrics"
                / f"optimizer_{phase}.json",
                {
                    "phase": phase,
                    "parameter_counts": effective_counts,
                    "total_parameters": sum(effective_counts.values()),
                },
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
        if settings.evaluate_test_each_epoch:
            test_metrics, _ = evaluate_model(
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
            "checkpoint_type": "isic2016_skin_lesion_segmentation",
            "epoch": epoch,
            "training_phase": phase,
            "initialization": initialization,
            "backbone": model.backbone.checkpoint_state(),
            "head_state_dict": model.head.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_metrics": train_metrics,
            "test_metrics_observation_only": test_metrics,
            "selection_criterion": "minimum_train_loss",
            "test_used_for_selection": False,
        }
        atomic_torch_save(last_path, payload)
        if improved:
            atomic_torch_save(best_path, payload)
        if (
            settings.checkpoint_interval_epochs > 0
            and epoch % settings.checkpoint_interval_epochs == 0
        ):
            atomic_torch_save(
                checkpoint_dir / f"isic_student_epoch_{epoch:04d}.pt",
                payload,
            )

        row = _history_row(
            epoch,
            phase,
            optimizer,
            train_metrics,
            test_metrics,
            parts,
            model,
            time.perf_counter() - started,
        )
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
            stage="ISIC 2016 lesion segmentation",
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
                dataset_slug="isic2016",
            )
        print(
            f"ISIC epoch {epoch:03d}/{settings.total_epochs} phase={phase} "
            f"train_loss={train_metrics['loss']:.5f} "
            f"train_Jaccard={train_metrics['mean_iou']:.4f} "
            f"test_Jaccard(observation)={test_metrics['mean_iou']:.4f} "
            f"test_Dice={test_metrics['mean_dice']:.4f} "
            f"sensitivity={test_metrics['sensitivity']:.4f} "
            f"specificity={test_metrics['specificity']:.4f}",
            flush=True,
        )

    final_metrics = evaluate_checkpoint(
        loaded,
        bundle,
        settings,
        checkpoint=best_path,
        save_visualizations=True,
    )
    return {
        "initial_test_metrics": initial_metrics,
        "final_test_metrics": final_metrics,
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
        raise FileNotFoundError(f"ISIC checkpoint is missing: {checkpoint}")
    payload = torch_load(checkpoint)
    if (
        not isinstance(payload, dict)
        or payload.get("checkpoint_type")
        != "isic2016_skin_lesion_segmentation"
    ):
        raise RuntimeError(f"Invalid ISIC checkpoint: {checkpoint}")
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
    metrics, rows = evaluate_model(
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
            "initialization_mode": settings.initialization_mode,
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
            "sensitivity",
            "specificity",
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
            dataset_slug="isic2016",
        )
        save_phase_masks(
            model.backbone.core,
            settings.output_dir / "figures" / "final_phase_masks",
        )
    return metrics


@torch.no_grad()
def evaluate_model(
    model: DUTSSaliencyModel,
    processor: Any,
    loader: Any,
    settings: Any,
    *,
    save_predictions: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    model.eval()
    accumulator = ISICSegmentationAccumulator()
    rows: list[dict[str, Any]] = []
    for batch in loader:
        inputs = preprocess_vision(
            processor,
            batch["images"],
            model.backbone.device,
        )
        masks = batch["masks"].to(
            model.backbone.device,
            non_blocking=True,
        )
        logits, _, _ = model(
            inputs["pixel_values"],
            inputs["image_grid_thw"],
        )
        loss, _ = segmentation_loss(
            logits,
            masks,
            bce_weight=settings.bce_weight,
            dice_weight=settings.dice_weight,
            soft_iou_weight=settings.soft_iou_weight,
            boundary_weight=settings.boundary_weight,
        )
        accumulator.update(logits, masks, loss=float(loss))
        if save_predictions:
            batch_metrics = per_sample_metrics(logits, masks)
            for index, sample_id in enumerate(batch["sample_ids"]):
                rows.append(
                    {
                        "sample_id": sample_id,
                        "image_path": batch["image_paths"][index],
                        "mask_path": batch["mask_paths"][index],
                        **batch_metrics[index],
                    }
                )
    return accumulator.compute(), rows


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
    accumulator = ISICSegmentationAccumulator()
    sums = {
        "segmentation_loss": 0.0,
        "bce": 0.0,
        "dice_loss": 0.0,
        "soft_iou_loss": 0.0,
        "boundary_loss": 0.0,
        "router_balance": 0.0,
        "router_importance": 0.0,
    }
    samples = 0
    for batch_index, batch in enumerate(loader, start=1):
        inputs = preprocess_vision(
            processor,
            batch["images"],
            model.backbone.device,
        )
        masks = batch["masks"].to(
            model.backbone.device,
            non_blocking=True,
        )
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
            total = (
                segmentation
                + settings.router_balance_weight * balance.float()
                + settings.router_importance_weight * importance.float()
            )
        if not torch.isfinite(total):
            raise RuntimeError(
                f"Non-finite ISIC loss at epoch={epoch}, batch={batch_index}"
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
        }
        for name in sums:
            sums[name] += float(values[name].detach()) * batch_size
        samples += batch_size
        if batch_index % settings.log_interval_batches == 0:
            current = accumulator.compute()
            print(
                f"[ISIC {phase}] epoch={epoch} batch={batch_index:,}/"
                f"{len(loader):,} loss={current['loss']:.5f} "
                f"Jaccard={current['mean_iou']:.4f} "
                f"Dice={current['mean_dice']:.4f} "
                f"balance={float(balance):.5f}",
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
    # Optical modules are registered inside visual.blocks[0] so that they can
    # replace the native Qwen stack. Freeze the complete visual tree first,
    # then explicitly re-enable the inserted student modules. Reversing this
    # order silently freezes the optical core and leaves empty optimizer
    # groups.
    model.backbone.visual.requires_grad_(False)
    model.backbone.core.requires_grad_(not warmup)
    model.backbone.recombiner.requires_grad_(not warmup)
    model.head.requires_grad_(True)


def _build_optimizer(
    model: DUTSSaliencyModel,
    settings: Any,
    *,
    warmup: bool,
) -> torch.optim.Optimizer:
    groups: list[dict[str, Any]] = []
    if not warmup:
        optical = []
        router = []
        for name, parameter in model.backbone.core.named_parameters():
            if not parameter.requires_grad:
                continue
            (router if name.startswith("router.") else optical).append(parameter)
        if not optical:
            raise RuntimeError(
                "Optical optimizer group is empty. The inserted optical core "
                "was probably frozen recursively through Qwen visual."
            )
        if not router:
            raise RuntimeError(
                "Router optimizer group is empty. Check optical trainability."
            )
        recombiner = [
            parameter
            for parameter in model.backbone.recombiner.parameters()
            if parameter.requires_grad
        ]
        if not recombiner:
            raise RuntimeError(
                "Recombiner optimizer group is empty. Check module freeze order."
            )
        groups.extend(
            [
                {
                    "params": optical,
                    "lr": settings.optical_learning_rate,
                    "name": "optical",
                },
                {
                    "params": router,
                    "lr": settings.router_learning_rate,
                    "name": "router",
                },
                {
                    "params": recombiner,
                    "lr": settings.recombiner_learning_rate,
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
            "lr": settings.head_learning_rate,
            "name": "head",
        }
    )
    return torch.optim.AdamW(groups, weight_decay=settings.weight_decay)


def _optimizer_parameter_counts(
    optimizer: torch.optim.Optimizer,
) -> dict[str, int]:
    return {
        str(group.get("name", f"group_{index}")): sum(
            parameter.numel() for parameter in group["params"]
        )
        for index, group in enumerate(optimizer.param_groups)
    }


def _history_row(
    epoch: int,
    phase: str,
    optimizer: torch.optim.Optimizer,
    train: dict[str, Any],
    test: dict[str, Any],
    parts: dict[str, float],
    model: DUTSSaliencyModel,
    elapsed: float,
) -> dict[str, Any]:
    return {
        "epoch": epoch,
        "training_phase": phase,
        "learning_rate_optical": _optional_lr(optimizer, "optical"),
        "learning_rate_router": _optional_lr(optimizer, "router"),
        "learning_rate_recombiner": _optional_lr(optimizer, "recombiner"),
        "learning_rate_head": _optional_lr(optimizer, "head"),
        "train_loss": train["loss"],
        "train_segmentation_loss": parts["segmentation_loss"],
        "train_bce": parts["bce"],
        "train_dice_loss": parts["dice_loss"],
        "train_soft_iou_loss": parts["soft_iou_loss"],
        "train_boundary_loss": parts["boundary_loss"],
        "train_router_balance": parts["router_balance"],
        "train_router_importance": parts["router_importance"],
        "train_mean_iou": train["mean_iou"],
        "train_mean_dice": train["mean_dice"],
        "train_mae": train["mae"],
        "train_pixel_accuracy": train["pixel_accuracy"],
        "train_sensitivity": train["sensitivity"],
        "train_specificity": train["specificity"],
        "test_loss": test["loss"],
        "test_mean_iou": test["mean_iou"],
        "test_mean_dice": test["mean_dice"],
        "test_mae": test["mae"],
        "test_pixel_accuracy": test["pixel_accuracy"],
        "test_sensitivity": test["sensitivity"],
        "test_specificity": test["specificity"],
        "alpha": float(model.backbone.recombiner.alpha.detach()),
        "epoch_time_sec": elapsed,
    }


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
        "sensitivity": 0.0,
        "specificity": 0.0,
        "samples": 0,
        "pixels": 0,
    }


def _print_trainable(report: dict[str, Any]) -> None:
    print(
        f"ISIC trainable parameters={report['trainable_parameters']:,} "
        f"tensors={report['trainable_tensors']}",
        flush=True,
    )
    for row in report["trainable_parameter_list"]:
        print(
            f"  {row['name']} shape={row['shape']} "
            f"params={row['parameters']:,}",
            flush=True,
        )


__all__ = [
    "evaluate_checkpoint",
    "evaluate_model",
    "initialize_model",
    "train_isic",
]
