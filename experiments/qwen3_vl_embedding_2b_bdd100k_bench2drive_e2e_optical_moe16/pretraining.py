from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval.optics.physical import (
    phase_dc_loss,
)

from .datasets_bdd100k import BDD100KSpatialDataset, collate_bdd
from .io_utils import append_csv, atomic_torch_save, write_json
from .modeling import (
    BDDPretrainModel,
    NativeVisionFeatureExtractor,
    OpticalDrivingBackbone,
    preprocess_vision,
    trainable_parameter_report,
)
from .objectives import auxiliary_structure_loss, normalized_feature_loss


def bdd_loader(
    records: list[Any], settings: Any, *, training: bool
) -> DataLoader:
    dataset = BDD100KSpatialDataset(
        records,
        settings.image_size,
        settings.bdd_lane_width,
        settings.road_participant_categories,
    )
    generator = torch.Generator()
    generator.manual_seed(settings.random_seed)
    return DataLoader(
        dataset,
        batch_size=settings.pretrain_batch_size,
        shuffle=training,
        num_workers=settings.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=settings.num_workers > 0,
        collate_fn=collate_bdd,
        generator=generator if training else None,
    )


def train_bdd_backbone(
    model: BDDPretrainModel,
    teacher: NativeVisionFeatureExtractor,
    projection: Any,
    processor: Any,
    train_records: list[Any],
    test_records: list[Any],
    settings: Any,
    device: torch.device,
) -> dict[str, Any]:
    train_loader = bdd_loader(train_records, settings, training=True)
    test_loader = bdd_loader(test_records, settings, training=False)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        parameters,
        lr=settings.pretrain_learning_rate,
        weight_decay=settings.pretrain_weight_decay,
    )
    scaler = _scaler(settings.amp_enabled, device)
    report = {
        "backbone": model.backbone.specification(),
        "training": trainable_parameter_report(model),
        "auxiliary_head_removed_from_export": True,
    }
    write_json(settings.output_dir / "metrics" / "bdd_pretrain_model.json", report)
    history_path = settings.output_dir / "metrics" / "bdd_pretrain_history.csv"
    best = float("inf")
    best_epoch = 0
    for epoch in range(1, settings.pretrain_epochs + 1):
        started = time.perf_counter()
        train_metrics = _run_pretrain_epoch(
            model,
            teacher,
            projection,
            processor,
            train_loader,
            settings,
            device,
            optimizer=optimizer,
            scaler=scaler,
            epoch=epoch,
        )
        test_metrics = _run_pretrain_epoch(
            model,
            teacher,
            projection,
            processor,
            test_loader,
            settings,
            device,
            optimizer=None,
            scaler=None,
            epoch=epoch,
        )
        row = {
            "epoch": epoch,
            **{f"train_{key}": value for key, value in train_metrics.items()},
            **{f"test_{key}": value for key, value in test_metrics.items()},
            "epoch_time_sec": time.perf_counter() - started,
        }
        append_csv(history_path, row)
        payload = {
            "epoch": epoch,
            "backbone": model.backbone.checkpoint_state(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": row,
            "auxiliary_head_state_dict": model.auxiliary_head.state_dict(),
        }
        atomic_torch_save(
            settings.output_dir / "checkpoints" / "bdd_pretrain_last.pt", payload
        )
        if test_metrics["total_loss"] < best:
            best = test_metrics["total_loss"]
            best_epoch = epoch
            # Export intentionally excludes the pretraining-only auxiliary head.
            atomic_torch_save(settings.pretrained_backbone_checkpoint, payload["backbone"])
            atomic_torch_save(
                settings.output_dir / "checkpoints" / "bdd_pretrain_best_full.pt",
                payload,
            )
        print(
            f"[bdd_pretrain] epoch={epoch:03d} "
            f"train={train_metrics['total_loss']:.5f} "
            f"test={test_metrics['total_loss']:.5f} "
            f"feature_cos={test_metrics['feature_cosine_similarity']:.4f} "
            f"best_epoch={best_epoch}",
            flush=True,
        )
    summary = {
        "best_epoch": best_epoch,
        "best_test_loss": best,
        "deployment_checkpoint": str(settings.pretrained_backbone_checkpoint),
        "deployment_contents": [
            "input_adapter",
            "input_norm",
            "router",
            "one expert phase plane",
            "OEO parameters",
            "global phase",
            "CCD LayerNorm",
            "Linear(224,224)",
        ],
        "excluded": ["BDD auxiliary segmentation head", "PCA", "Qwen teacher"],
    }
    write_json(settings.output_dir / "metrics" / "bdd_pretrain_summary.json", summary)
    return summary


def export_backbone(
    backbone: OpticalDrivingBackbone, settings: Any, source: Path | None = None
) -> dict[str, Any]:
    if source is not None:
        backbone.load_checkpoint(source)
    payload = backbone.checkpoint_state()
    atomic_torch_save(settings.pretrained_backbone_checkpoint, payload)
    report = {
        "checkpoint": str(settings.pretrained_backbone_checkpoint),
        "architecture": backbone.specification(),
        "training_only_auxiliary_heads_included": False,
    }
    write_json(settings.output_dir / "exports" / "optical_backbone.json", report)
    return report


def _run_pretrain_epoch(
    model: BDDPretrainModel,
    teacher: NativeVisionFeatureExtractor,
    projection: Any,
    processor: Any,
    loader: DataLoader,
    settings: Any,
    device: torch.device,
    *,
    optimizer: torch.optim.Optimizer | None,
    scaler: Any,
    epoch: int,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals: dict[str, float] = {}
    samples = 0
    for batch_index, batch in enumerate(loader, 1):
        inputs = preprocess_vision(processor, batch["images"], device)
        model.backbone.restore_native()
        with torch.no_grad():
            teacher_hidden, _ = teacher.extract(
                inputs["pixel_values"], inputs["image_grid_thw"]
            )
            teacher_target = projection.encode(teacher_hidden).detach()
        model.backbone.activate()
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training), _autocast(settings.amp_enabled, device):
            student, auxiliary, _ = model(
                inputs["pixel_values"], inputs["image_grid_thw"]
            )
            feature_loss, feature_parts = normalized_feature_loss(
                student,
                teacher_target,
                cosine_weight=settings.lambda_feature_cosine,
                smooth_l1_weight=settings.lambda_feature_smooth_l1,
            )
            auxiliary_loss, auxiliary_parts = auxiliary_structure_loss(
                auxiliary.float(),
                batch["targets"].to(device, non_blocking=True),
                weights=(
                    settings.lambda_drivable,
                    settings.lambda_lane,
                    settings.lambda_participant,
                ),
            )
            balance, importance = model.backbone.router_losses()
            dc = (
                phase_dc_loss(model.backbone)
                if training and settings.pretrain_phase_dc_weight > 0.0
                else student.new_zeros(())
            )
            loss = (
                feature_loss
                + auxiliary_loss
                + settings.lambda_router_balance * balance.float()
                + settings.lambda_router_importance * importance.float()
                + settings.pretrain_phase_dc_weight * dc
            )
        if training:
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
        batch_size = len(batch["images"])
        samples += batch_size
        values = {
            "total_loss": loss,
            **feature_parts,
            **auxiliary_parts,
            "router_balance": balance,
            "router_importance": importance,
            "phase_dc": dc,
            "feature_cosine_similarity": 1.0 - feature_parts["feature_cosine"],
        }
        for key, value in values.items():
            totals[key] = totals.get(key, 0.0) + float(value.detach()) * batch_size
        if training and batch_index % settings.log_interval_batches == 0:
            print(
                f"[bdd_pretrain] epoch={epoch} batch={batch_index}/{len(loader)} "
                f"loss={float(loss):.5f} balance={float(balance):.5f}",
                flush=True,
            )
    if not samples:
        raise RuntimeError("BDD pretraining loader is empty")
    return {key: value / samples for key, value in totals.items()}


def _autocast(enabled: bool, device: torch.device):
    return torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16 if device.type == "cuda" else torch.bfloat16,
        enabled=bool(enabled and device.type == "cuda"),
    )


def _scaler(enabled: bool, device: torch.device):
    if not enabled or device.type != "cuda":
        return None
    try:
        return torch.amp.GradScaler("cuda")
    except TypeError:
        return torch.cuda.amp.GradScaler()
