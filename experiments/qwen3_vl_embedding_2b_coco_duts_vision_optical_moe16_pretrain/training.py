from __future__ import annotations

import contextlib
import time
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval.optics.physical import (
    phase_dc_loss,
)

from .datasets import (
    CocoBundle,
    CocoImageDataset,
    DUTSSaliencyDataset,
    DutsBundle,
    collate_coco,
    collate_duts,
)
from .io_utils import atomic_torch_save, torch_load, write_csv, write_json
from .modeling import (
    DUTSSaliencyModel,
    LoadedVisionBackbone,
    OpticalVisionBackbone,
    build_duts_model,
    build_optical_backbone,
    optical_parameter_breakdown,
    preprocess_vision,
    trainable_parameter_report,
)
from .objectives import (
    FeatureAccumulator,
    SegmentationAccumulator,
    feature_distillation_loss,
    segmentation_loss,
)
from .teacher_cache import (
    CocoTeacherTargetDataset,
    collate_coco_targets,
    expected_cache_identity,
)
from .visualization import (
    save_phase_masks,
    save_segmentation_examples,
    save_training_curves,
)


COCO_HISTORY_FIELDS = [
    "epoch",
    "learning_rate_optical",
    "learning_rate_router",
    "learning_rate_recombiner",
    "train_loss",
    "train_cosine_similarity",
    "train_smooth_l1",
    "train_mse",
    "train_router_balance",
    "train_router_importance",
    "observation_loss",
    "observation_cosine_similarity",
    "observation_smooth_l1",
    "observation_mse",
    "alpha",
    "epoch_time_sec",
]

DUTS_HISTORY_FIELDS = [
    "epoch",
    "training_phase",
    "learning_rate_optical",
    "learning_rate_recombiner",
    "learning_rate_head",
    "train_loss",
    "train_bce",
    "train_dice_loss",
    "train_soft_iou_loss",
    "train_boundary_loss",
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


def build_coco_cache_loader(
    bundle: CocoBundle,
    settings: Any,
    *,
    split: str,
) -> DataLoader:
    records = bundle.train_records if split == "train" else bundle.val_records
    dataset = CocoImageDataset(records, settings)
    return DataLoader(
        dataset,
        batch_size=settings.teacher_cache_batch_size,
        shuffle=False,
        drop_last=False,
        **_loader_kwargs(settings, collate_coco),
    )


def build_coco_training_loaders(
    bundle: CocoBundle,
    settings: Any,
    *,
    pca_metadata: dict[str, Any],
) -> tuple[DataLoader, DataLoader]:
    train_images = CocoImageDataset(bundle.train_records, settings)
    val_images = CocoImageDataset(bundle.val_records, settings)
    train_identity = expected_cache_identity(
        settings,
        split="train",
        dataset_manifest_digest=bundle.metadata["manifest_sha256"],
        pca_metadata=pca_metadata,
        sample_count=len(train_images),
    )
    val_identity = expected_cache_identity(
        settings,
        split="val",
        dataset_manifest_digest=bundle.metadata["manifest_sha256"],
        pca_metadata=pca_metadata,
        sample_count=len(val_images),
    )
    train = CocoTeacherTargetDataset(
        train_images,
        settings.teacher_cache_root / "train",
        lru_shards=settings.teacher_cache_lru_shards,
        expected_identity=train_identity,
    )
    val = CocoTeacherTargetDataset(
        val_images,
        settings.teacher_cache_root / "val",
        lru_shards=settings.teacher_cache_lru_shards,
        expected_identity=val_identity,
    )
    common = _loader_kwargs(settings, collate_coco_targets)
    return (
        DataLoader(
            train,
            batch_size=settings.coco_batch_size,
            shuffle=True,
            drop_last=False,
            **common,
        ),
        DataLoader(
            val,
            batch_size=settings.inference_batch_size,
            shuffle=False,
            drop_last=False,
            **common,
        ),
    )


def build_duts_loaders(
    bundle: DutsBundle,
    settings: Any,
) -> tuple[DataLoader, DataLoader]:
    train = DUTSSaliencyDataset(
        bundle.train_records,
        settings,
        training=True,
    )
    test = DUTSSaliencyDataset(
        bundle.test_records,
        settings,
        training=False,
    )
    common = _loader_kwargs(settings, collate_duts)
    return (
        DataLoader(
            train,
            batch_size=settings.duts_batch_size,
            shuffle=True,
            drop_last=False,
            **common,
        ),
        DataLoader(
            test,
            batch_size=settings.inference_batch_size,
            shuffle=False,
            drop_last=False,
            **common,
        ),
    )


def train_coco_backbone(
    loaded: LoadedVisionBackbone,
    bundle: CocoBundle,
    settings: Any,
    *,
    pca_metadata: dict[str, Any],
) -> dict[str, Any]:
    train_loader, val_loader = build_coco_training_loaders(
        bundle,
        settings,
        pca_metadata=pca_metadata,
    )
    backbone = build_optical_backbone(
        loaded,
        settings,
        release_native_to_cpu=True,
    )
    backbone.requires_grad_(True)
    optimizer = _coco_optimizer(backbone, settings)
    report = trainable_parameter_report(backbone, prefix="optical_backbone")
    report["architecture"] = backbone.specification()
    report["parameter_breakdown"] = optical_parameter_breakdown(backbone)
    report["pca_in_student"] = False
    report["teacher_target"] = {
        "source": "Qwen final pre-merger spatial hidden",
        "projection": "offline fixed PCA 1024->224",
        "pca_projection_sha256": pca_metadata.get("projection_sha256"),
    }
    write_json(settings.output_dir / "model_coco_backbone.json", report)
    _print_trainable_report(report)

    history: list[dict[str, Any]] = []
    best_train_loss = float("inf")
    checkpoint_dir = settings.output_dir / "checkpoints"
    best_path = checkpoint_dir / "coco_backbone_best_train_loss.pt"
    last_path = checkpoint_dir / "coco_backbone_last.pt"
    for epoch in range(1, settings.coco_epochs + 1):
        started = time.perf_counter()
        train_metrics, train_parts = _run_coco_epoch(
            backbone,
            loaded.processor,
            train_loader,
            settings,
            optimizer=optimizer,
            epoch=epoch,
            phase="train",
        )
        observation, _ = _run_coco_epoch(
            backbone,
            loaded.processor,
            val_loader,
            settings,
            optimizer=None,
            epoch=epoch,
            phase="val_observation",
        )
        improved = float(train_metrics["loss"]) < best_train_loss
        if improved:
            best_train_loss = float(train_metrics["loss"])
        payload = {
            "checkpoint_type": "coco_general_feature_pretraining",
            "epoch": epoch,
            "backbone": backbone.checkpoint_state(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_metrics": train_metrics,
            "val_metrics_observation_only": observation,
            "selection_criterion": "minimum_train_loss",
            "coco_val_used_for_selection": False,
            "pca_metadata": pca_metadata,
        }
        atomic_torch_save(last_path, payload)
        if improved:
            atomic_torch_save(best_path, payload)
        row = {
            "epoch": epoch,
            "learning_rate_optical": _group_lr(optimizer, "optical"),
            "learning_rate_router": _group_lr(optimizer, "router"),
            "learning_rate_recombiner": _group_lr(optimizer, "recombiner"),
            "train_loss": train_metrics["loss"],
            "train_cosine_similarity": train_metrics["cosine_similarity"],
            "train_smooth_l1": train_metrics["smooth_l1"],
            "train_mse": train_metrics["mse"],
            "train_router_balance": train_parts["router_balance"],
            "train_router_importance": train_parts["router_importance"],
            "observation_loss": observation["loss"],
            "observation_cosine_similarity": observation["cosine_similarity"],
            "observation_smooth_l1": observation["smooth_l1"],
            "observation_mse": observation["mse"],
            "alpha": float(backbone.recombiner.alpha.detach()),
            "epoch_time_sec": time.perf_counter() - started,
        }
        history.append(row)
        write_csv(
            settings.output_dir / "metrics" / "coco_training_history.csv",
            history,
            COCO_HISTORY_FIELDS,
        )
        write_json(
            settings.output_dir / "metrics" / "coco_training_latest.json",
            row,
        )
        save_training_curves(
            settings.output_dir / "figures" / "coco_training_curves.png",
            history,
            stage="COCO feature distillation",
        )
        print(
            f"COCO epoch {epoch:03d}/{settings.coco_epochs} "
            f"train_loss={train_metrics['loss']:.5f} "
            f"train_cos={train_metrics['cosine_similarity']:.4f} "
            f"val_cos(observation)={observation['cosine_similarity']:.4f} "
            f"alpha={row['alpha']:.4f} best_train_loss={best_train_loss:.5f}",
            flush=True,
        )
    save_phase_masks(
        backbone.core,
        settings.output_dir / "figures" / "coco_final_phase_masks",
    )
    # The same Python process may continue into DUTS. Put the original native
    # Vision blocks back before this student goes out of scope.
    backbone.core.to("cpu")
    backbone.recombiner.to("cpu")
    backbone.restore_native()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {
        "best_checkpoint": str(best_path),
        "last_checkpoint": str(last_path),
        "best_train_loss": best_train_loss,
    }


def train_duts(
    loaded: LoadedVisionBackbone,
    bundle: DutsBundle,
    settings: Any,
) -> dict[str, Any]:
    if not settings.coco_checkpoint.is_file():
        raise FileNotFoundError(
            f"COCO-pretrained optical backbone is missing: "
            f"{settings.coco_checkpoint}. Run --phase coco_pretrain first."
        )
    model = build_duts_model(
        loaded,
        settings,
        checkpoint=settings.coco_checkpoint,
    )
    train_loader, test_loader = build_duts_loaders(bundle, settings)
    report = trainable_parameter_report(model, prefix="duts_optical_saliency")
    report["backbone"] = model.backbone.specification()
    report["segmentation_head"] = model.head.specification()
    report["initial_backbone_checkpoint"] = str(settings.coco_checkpoint)
    report["training_schedule"] = {
        "head_only_epochs": settings.duts_head_warmup_epochs,
        "joint_finetune_epochs": settings.duts_finetune_epochs,
        "optical_learning_rate": settings.duts_optical_learning_rate,
        "recombiner_learning_rate": settings.duts_recombiner_learning_rate,
        "head_learning_rate": settings.duts_head_learning_rate,
    }
    write_json(settings.output_dir / "model_duts.json", report)

    history: list[dict[str, Any]] = []
    best_train_loss = float("inf")
    best_path = (
        settings.output_dir
        / "checkpoints"
        / "duts_student_best_train_loss.pt"
    )
    last_path = settings.output_dir / "checkpoints" / "duts_student_last.pt"
    optimizer: torch.optim.Optimizer | None = None
    current_phase = ""
    for epoch in range(1, settings.duts_total_epochs + 1):
        warmup = epoch <= settings.duts_head_warmup_epochs
        phase = "head_warmup" if warmup else "joint_finetune"
        if phase != current_phase:
            _configure_duts_trainability(model, warmup=warmup)
            optimizer = _duts_optimizer(model, settings, warmup=warmup)
            current_phase = phase
            print(
                f"[DUTS] entering {phase}; optimizer initialized with "
                f"{len(optimizer.param_groups)} parameter groups",
                flush=True,
            )
        assert optimizer is not None
        started = time.perf_counter()
        train_metrics, parts = _run_duts_epoch(
            model,
            loaded.processor,
            train_loader,
            settings,
            optimizer=optimizer,
            epoch=epoch,
            phase=phase,
            detach_backbone=warmup,
        )
        test_metrics = (
            evaluate_duts(
                model,
                loaded.processor,
                test_loader,
                settings,
                save_predictions=False,
            )[0]
            if settings.evaluate_duts_test_each_epoch
            else _empty_segmentation_metrics()
        )
        improved = float(train_metrics["loss"]) < best_train_loss
        if improved:
            best_train_loss = float(train_metrics["loss"])
        payload = {
            "checkpoint_type": "duts_saliency_pretraining",
            "epoch": epoch,
            "training_phase": phase,
            "backbone": model.backbone.checkpoint_state(),
            "head_state_dict": model.head.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_metrics": train_metrics,
            "test_metrics_observation_only": test_metrics,
            "selection_criterion": "minimum_train_loss",
            "duts_test_used_for_selection": False,
            "initial_coco_checkpoint": str(settings.coco_checkpoint),
        }
        atomic_torch_save(last_path, payload)
        if improved:
            atomic_torch_save(best_path, payload)
        if (
            settings.checkpoint_interval_epochs > 0
            and epoch % settings.checkpoint_interval_epochs == 0
        ):
            atomic_torch_save(
                settings.output_dir
                / "checkpoints"
                / f"duts_student_epoch_{epoch:04d}.pt",
                payload,
            )
        row = {
            "epoch": epoch,
            "training_phase": phase,
            "learning_rate_optical": _optional_group_lr(optimizer, "optical"),
            "learning_rate_recombiner": _optional_group_lr(
                optimizer, "recombiner"
            ),
            "learning_rate_head": _group_lr(optimizer, "head"),
            "train_loss": train_metrics["loss"],
            "train_bce": parts["bce"],
            "train_dice_loss": parts["dice_loss"],
            "train_soft_iou_loss": parts["soft_iou_loss"],
            "train_boundary_loss": parts["boundary_loss"],
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
            settings.output_dir / "metrics" / "duts_training_history.csv",
            history,
            DUTS_HISTORY_FIELDS,
        )
        write_json(
            settings.output_dir / "metrics" / "duts_training_latest.json",
            row,
        )
        save_training_curves(
            settings.output_dir / "figures" / "duts_training_curves.png",
            history,
            stage="DUTS saliency",
        )
        if epoch % settings.visualization_interval_epochs == 0:
            save_segmentation_examples(
                model,
                loaded.processor,
                test_loader,
                settings,
                epoch=epoch,
                phase="test_observation",
            )
        print(
            f"DUTS epoch {epoch:03d}/{settings.duts_total_epochs} "
            f"phase={phase} train_loss={train_metrics['loss']:.5f} "
            f"train_mIoU={train_metrics['mean_iou']:.4f} "
            f"test_mIoU(observation)={test_metrics['mean_iou']:.4f} "
            f"test_Dice={test_metrics['mean_dice']:.4f} "
            f"best_train_loss={best_train_loss:.5f}",
            flush=True,
        )
    # Report the train-loss-selected checkpoint once after training. DUTS-TE
    # has never influenced optimizer state or checkpoint selection.
    final_metrics = test_duts_checkpoint(
        loaded,
        bundle,
        settings,
        checkpoint=best_path,
        save_predictions=True,
        save_examples=True,
        model=model,
    )
    return {
        "best_checkpoint": str(best_path),
        "last_checkpoint": str(last_path),
        "best_train_loss": best_train_loss,
        "test_metrics": final_metrics,
    }


def test_duts_checkpoint(
    loaded: LoadedVisionBackbone,
    bundle: DutsBundle,
    settings: Any,
    *,
    checkpoint: Path,
    save_predictions: bool,
    save_examples: bool,
    model: DUTSSaliencyModel | None = None,
) -> dict[str, Any]:
    payload = torch_load(checkpoint)
    if not isinstance(payload, dict) or "backbone" not in payload:
        raise RuntimeError(f"Invalid DUTS checkpoint: {checkpoint}")
    if model is None:
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
    _, test_loader = build_duts_loaders(bundle, settings)
    metrics, rows = evaluate_duts(
        model,
        loaded.processor,
        test_loader,
        settings,
        save_predictions=save_predictions,
    )
    metrics.update(
        {
            "checkpoint": str(checkpoint),
            "checkpoint_epoch": int(payload["epoch"]),
            "checkpoint_selection": "minimum_train_loss",
            "test_used_for_selection": False,
        }
    )
    write_json(settings.output_dir / "metrics" / "duts_test.json", metrics)
    if save_predictions:
        write_csv(
            settings.output_dir / "metrics" / "duts_test_predictions.csv",
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
    if save_examples:
        save_segmentation_examples(
            model,
            loaded.processor,
            test_loader,
            settings,
            epoch=int(payload["epoch"]),
            phase="final_test",
        )
        save_phase_masks(
            model.backbone.core,
            settings.output_dir / "figures" / "duts_final_phase_masks",
        )
    return metrics


def _run_coco_epoch(
    backbone: OpticalVisionBackbone,
    processor: Any,
    loader: DataLoader,
    settings: Any,
    *,
    optimizer: torch.optim.Optimizer | None,
    epoch: int,
    phase: str,
) -> tuple[dict[str, Any], dict[str, float]]:
    training = optimizer is not None
    backbone.train(training)
    accumulator = FeatureAccumulator(
        smooth_l1_beta=settings.smooth_l1_beta,
    )
    part_sums = {
        "cosine_loss": 0.0,
        "smooth_l1_loss": 0.0,
        "router_balance": 0.0,
        "router_importance": 0.0,
        "phase_dc": 0.0,
    }
    part_weight = 0
    context = contextlib.nullcontext() if training else torch.no_grad()
    with context:
        for batch_index, batch in enumerate(loader, start=1):
            inputs = preprocess_vision(
                processor,
                batch["images"],
                backbone.device,
            )
            cached_grid = batch["teacher_image_grid_thw"].long()
            runtime_grid = inputs["image_grid_thw"].detach().cpu().long()
            if not torch.equal(cached_grid, runtime_grid):
                raise RuntimeError(
                    "Cached teacher token grid differs from current Qwen processor "
                    f"output: cache={cached_grid.tolist()}, runtime={runtime_grid.tolist()}"
                )
            targets = torch.cat(batch["teacher_targets"], dim=0).to(
                backbone.device,
                non_blocking=True,
            )
            if training:
                optimizer.zero_grad(set_to_none=True)
            with _autocast(settings, backbone.device):
                student, lengths, _ = backbone(
                    inputs["pixel_values"],
                    inputs["image_grid_thw"],
                )
                if lengths != [
                    int(value.prod()) for value in runtime_grid
                ]:
                    raise RuntimeError(
                        f"Student token lengths {lengths} do not match runtime grid"
                    )
                balance, importance = backbone.router_losses()
                loss, parts = feature_distillation_loss(
                    student,
                    targets,
                    cosine_weight=settings.cosine_loss_weight,
                    smooth_l1_weight=settings.smooth_l1_loss_weight,
                    smooth_l1_beta=settings.smooth_l1_beta,
                    router_balance=balance,
                    router_balance_weight=settings.router_balance_weight,
                    router_importance=importance,
                    router_importance_weight=settings.router_importance_weight,
                )
                dc = (
                    phase_dc_loss(backbone)
                    if training and settings.phase_dc_weight > 0.0
                    else loss.new_zeros(())
                )
                loss = loss + settings.phase_dc_weight * dc
                parts["phase_dc"] = dc
            if training:
                loss.backward()
                optimizer.step()
            token_count = int(student.shape[0])
            accumulator.update(
                student.detach(),
                targets.detach(),
                loss=float(loss.detach()),
                samples=len(batch["images"]),
            )
            for name in part_sums:
                part_sums[name] += float(parts[name].detach()) * token_count
            part_weight += token_count
            if training and batch_index % settings.log_interval_batches == 0:
                current = accumulator.compute()
                print(
                    f"[COCO {phase}] epoch={epoch} batch={batch_index:,}/"
                    f"{len(loader):,} loss={current['loss']:.5f} "
                    f"cos={current['cosine_similarity']:.4f} "
                    f"balance={float(parts['router_balance']):.4f}",
                    flush=True,
                )
    metrics = accumulator.compute()
    averaged = {
        name: value / max(1, part_weight)
        for name, value in part_sums.items()
    }
    return metrics, averaged


def _run_duts_epoch(
    model: DUTSSaliencyModel,
    processor: Any,
    loader: DataLoader,
    settings: Any,
    *,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    phase: str,
    detach_backbone: bool,
) -> tuple[dict[str, Any], dict[str, float]]:
    model.train(True)
    accumulator = SegmentationAccumulator()
    part_sums = {
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
            loss, parts = segmentation_loss(
                logits,
                masks,
                bce_weight=settings.bce_weight,
                dice_weight=settings.dice_weight,
                soft_iou_weight=settings.soft_iou_weight,
                boundary_weight=settings.boundary_weight,
            )
            if not detach_backbone:
                balance, importance = model.backbone.router_losses()
                dc = (
                    phase_dc_loss(model.backbone)
                    if settings.phase_dc_weight > 0.0
                    else loss.new_zeros(())
                )
                loss = (
                    loss
                    + settings.router_balance_weight * balance
                    + settings.router_importance_weight * importance
                    + settings.phase_dc_weight * dc
                )
            else:
                balance = loss.new_zeros(())
                importance = loss.new_zeros(())
                dc = loss.new_zeros(())
            parts["router_balance"] = balance
            parts["router_importance"] = importance
            parts["phase_dc"] = dc
        loss.backward()
        optimizer.step()
        batch_size = int(masks.shape[0])
        accumulator.update(logits.detach(), masks, loss=float(loss.detach()))
        for name in part_sums:
            part_sums[name] += float(parts[name].detach()) * batch_size
        samples += batch_size
        if batch_index % settings.log_interval_batches == 0:
            current = accumulator.compute()
            print(
                f"[DUTS {phase}] epoch={epoch} batch={batch_index:,}/"
                f"{len(loader):,} loss={current['loss']:.5f} "
                f"mIoU={current['mean_iou']:.4f} "
                f"Dice={current['mean_dice']:.4f}",
                flush=True,
            )
    return accumulator.compute(), {
        name: value / max(1, samples)
        for name, value in part_sums.items()
    }


@torch.no_grad()
def evaluate_duts(
    model: DUTSSaliencyModel,
    processor: Any,
    loader: DataLoader,
    settings: Any,
    *,
    save_predictions: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    model.eval()
    accumulator = SegmentationAccumulator()
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
            probability = logits.float().sigmoid()
            prediction = probability >= 0.5
            truth = masks >= 0.5
            intersection = (prediction & truth).flatten(1).sum(1).float()
            union = (prediction | truth).flatten(1).sum(1).float()
            prediction_sum = prediction.flatten(1).sum(1).float()
            truth_sum = truth.flatten(1).sum(1).float()
            iou = torch.where(
                union > 0,
                intersection / union,
                torch.ones_like(union),
            )
            dice = torch.where(
                prediction_sum + truth_sum > 0,
                2.0 * intersection / (prediction_sum + truth_sum),
                torch.ones_like(intersection),
            )
            mae = (probability - masks).abs().flatten(1).mean(1)
            for row_index, sample_id in enumerate(batch["sample_ids"]):
                rows.append(
                    {
                        "sample_id": sample_id,
                        "image_path": batch["image_paths"][row_index],
                        "mask_path": batch["mask_paths"][row_index],
                        "mean_iou": float(iou[row_index]),
                        "mean_dice": float(dice[row_index]),
                        "mae": float(mae[row_index]),
                    }
                )
    return accumulator.compute(), rows


def _coco_optimizer(
    backbone: OpticalVisionBackbone,
    settings: Any,
) -> torch.optim.Optimizer:
    optical: list[nn.Parameter] = []
    phases: list[nn.Parameter] = []
    router: list[nn.Parameter] = []
    for name, parameter in backbone.core.named_parameters():
        if not parameter.requires_grad:
            continue
        if "raw_phase" in name:
            phases.append(parameter)
        elif name.startswith("router."):
            router.append(parameter)
        else:
            optical.append(parameter)
    return torch.optim.AdamW(
        [
            {
                "params": optical,
                "lr": settings.coco_optical_learning_rate,
                "name": "optical",
            },
            {
                "params": phases,
                "lr": settings.coco_phase_learning_rate,
                "name": "phase",
            },
            {
                "params": router,
                "lr": settings.coco_router_learning_rate,
                "name": "router",
            },
            {
                "params": list(backbone.recombiner.parameters()),
                "lr": settings.coco_recombiner_learning_rate,
                "name": "recombiner",
            },
        ],
        weight_decay=settings.coco_weight_decay,
    )


def _configure_duts_trainability(
    model: DUTSSaliencyModel,
    *,
    warmup: bool,
) -> None:
    model.backbone.core.requires_grad_(not warmup)
    model.backbone.recombiner.requires_grad_(not warmup)
    model.head.requires_grad_(True)
    # Frozen Qwen patch/position stem and all other original parameters remain
    # excluded regardless of stage.
    model.backbone.visual.patch_embed.requires_grad_(False)


def _duts_optimizer(
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


def _loader_kwargs(settings: Any, collate_fn: Any) -> dict[str, Any]:
    values: dict[str, Any] = {
        "num_workers": settings.num_workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": settings.num_workers > 0,
        "collate_fn": collate_fn,
    }
    if settings.num_workers > 0:
        values["prefetch_factor"] = 2
    return values


def _autocast(settings: Any, device: torch.device) -> Any:
    enabled = bool(settings.amp_enabled and device.type == "cuda")
    if not enabled:
        return contextlib.nullcontext()
    dtype = torch.bfloat16 if settings.dtype == "bfloat16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype, enabled=True)


def _group_lr(optimizer: torch.optim.Optimizer, name: str) -> float:
    for group in optimizer.param_groups:
        if group.get("name") == name:
            return float(group["lr"])
    raise KeyError(f"Optimizer has no parameter group {name!r}")


def _optional_group_lr(optimizer: torch.optim.Optimizer, name: str) -> float:
    try:
        return _group_lr(optimizer, name)
    except KeyError:
        return 0.0


def _empty_segmentation_metrics() -> dict[str, float | int]:
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


def _print_trainable_report(report: dict[str, Any]) -> None:
    print(
        f"trainable parameters={report['trainable_parameters']:,} "
        f"tensors={report['trainable_tensors']}",
        flush=True,
    )
    for row in report["trainable_parameter_list"]:
        print(
            f"  {row['name']} shape={row['shape']} "
            f"params={row['parameters']:,}",
            flush=True,
        )
