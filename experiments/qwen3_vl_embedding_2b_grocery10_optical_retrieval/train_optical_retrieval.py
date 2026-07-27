from __future__ import annotations

import argparse
import csv
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterator, Sequence

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Sampler

from .cache_teacher_embeddings import TeacherEmbeddingStore
from .features import (
    move_inputs,
    preprocess_images,
    student_embeddings,
    validate_token_budgets,
)
from .io_utils import write_json
from .modeling import (
    LoadedBackbone,
    OpticalRetrievalReadout,
    build_optical_student,
    load_backbone,
    trainable_parameter_report,
    unique_trainable_parameters,
)
from .optics.replacement import DeepStackMultimodalReplacement
from .prepare_grocery_retrieval_subset import (
    GroceryRetrievalBundle,
    GroceryRetrievalDataset,
    GrocerySample,
    collate_grocery,
    prepare_grocery_subset,
)
from .retrieval_metrics import evaluate_embeddings
from .settings import Settings, load_settings


class PKBatchSampler(Sampler[list[int]]):
    """Deterministic epoch-aware P-SKU x K-image batch sampler."""

    def __init__(
        self,
        samples: Sequence[GrocerySample],
        p: int,
        k: int,
        seed: int,
    ) -> None:
        self.p = int(p)
        self.k = int(k)
        self.seed = int(seed)
        self.epoch = 0
        grouped: dict[int, list[int]] = defaultdict(list)
        for index, sample in enumerate(samples):
            grouped[sample.sku_index].append(index)
        self.grouped = {key: tuple(values) for key, values in sorted(grouped.items())}
        if len(self.grouped) < self.p:
            raise ValueError(f"PK sampler needs P={self.p} SKUs, found {len(self.grouped)}")
        if any(not values for values in self.grouped.values()):
            raise ValueError("PK sampler encountered an empty SKU")
        self.batch_count = max(
            1, math.ceil(len(samples) / (self.p * self.k))
        )

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.batch_count

    def __iter__(self) -> Iterator[list[int]]:
        generator = torch.Generator().manual_seed(self.seed + self.epoch * 1_000_003)
        sku_ids = list(self.grouped)
        pools: dict[int, list[int]] = {}
        positions: dict[int, int] = {}
        for sku in sku_ids:
            order = torch.randperm(len(self.grouped[sku]), generator=generator).tolist()
            pools[sku] = [self.grouped[sku][index] for index in order]
            positions[sku] = 0
        sku_order = torch.randperm(len(sku_ids), generator=generator).tolist()
        sku_cursor = 0
        for _ in range(self.batch_count):
            if self.p == len(sku_ids):
                selected_skus = sku_ids
            else:
                if sku_cursor + self.p > len(sku_order):
                    sku_order = torch.randperm(len(sku_ids), generator=generator).tolist()
                    sku_cursor = 0
                selected_skus = [sku_ids[index] for index in sku_order[sku_cursor:sku_cursor + self.p]]
                sku_cursor += self.p
            batch: list[int] = []
            for sku in selected_skus:
                for _ in range(self.k):
                    if positions[sku] >= len(pools[sku]):
                        order = torch.randperm(
                            len(self.grouped[sku]), generator=generator
                        ).tolist()
                        pools[sku] = [self.grouped[sku][index] for index in order]
                        positions[sku] = 0
                    batch.append(pools[sku][positions[sku]])
                    positions[sku] += 1
            yield batch


def supervised_contrastive_loss(
    embeddings: torch.Tensor, labels: torch.Tensor, temperature: float
) -> torch.Tensor:
    if embeddings.ndim != 2 or labels.ndim != 1 or len(embeddings) != len(labels):
        raise ValueError("Supervised contrastive inputs must be [B,D] and [B]")
    embeddings = F.normalize(embeddings.float(), dim=-1)
    logits = embeddings @ embeddings.T / float(temperature)
    identity = torch.eye(len(embeddings), dtype=torch.bool, device=embeddings.device)
    positive = labels[:, None].eq(labels[None, :]) & ~identity
    valid = positive.any(dim=1)
    if not torch.all(valid):
        missing = torch.nonzero(~valid, as_tuple=False).flatten().tolist()
        raise RuntimeError(
            f"Every contrastive anchor needs a same-SKU positive; missing anchors={missing}"
        )
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()
    denominator_mask = ~identity
    log_denominator = torch.logsumexp(
        logits.masked_fill(~denominator_mask, -torch.inf), dim=1
    )
    log_probability = logits - log_denominator[:, None]
    mean_positive_log_probability = (
        (log_probability * positive).sum(dim=1) / positive.sum(dim=1)
    )
    return -mean_positive_log_probability.mean()


def embedding_distillation_loss(
    student: torch.Tensor, teacher: torch.Tensor
) -> torch.Tensor:
    if student.shape != teacher.shape:
        raise RuntimeError(
            f"Student/teacher embedding shapes differ: {student.shape} vs {teacher.shape}"
        )
    return (1.0 - F.cosine_similarity(student.float(), teacher.float(), dim=-1)).mean()


def save_checkpoint(
    path: Path,
    replacement: DeepStackMultimodalReplacement,
    readout: OpticalRetrievalReadout,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    train_loss: float,
    settings: Settings,
) -> None:
    payload = {
        "checkpoint_version": 2,
        "epoch": int(epoch),
        "train_loss": float(train_loss),
        "vision_optical": replacement.vision_surrogate.state_dict(),
        "language_optical": replacement.language_surrogate.state_dict(),
        "retrieval_readout": readout.state_dict(),
        "optimizer": optimizer.state_dict(),
        "metadata": {
            "embedding_dim": settings.embedding_dim,
            "detector_dim": settings.detector_output_size,
            "instruction": settings.instruction,
            "model_id": settings.model_id,
            "expert_stages_per_stack": settings.expert_layers,
            "vision_tap_stages": list(settings.vision_tap_stages),
            "student_deepstack_auxiliary_count": len(settings.vision_tap_stages),
            "language_optical_layer_indexes": list(
                replacement.language_optical_layer_indexes
            ),
            "optical_architecture": "one_expert_stage_plus_one_global_phase",
            "selection_criterion": "minimum_training_total_loss",
            "test_metrics_used_for_selection": False,
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_checkpoint(
    path: Path,
    replacement: DeepStackMultimodalReplacement,
    readout: OpticalRetrievalReadout,
    *,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Optical retrieval checkpoint is missing: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    metadata = payload.get("metadata", {})
    expected_layers = len(replacement.vision_surrogate.core.expert_layers)
    saved_layers = metadata.get("expert_stages_per_stack")
    if saved_layers is not None and int(saved_layers) != expected_layers:
        raise RuntimeError(
            "Optical retrieval checkpoint architecture mismatch: "
            f"saved expert stages={saved_layers}, current expert stages={expected_layers}. "
            "The corrected baseline uses one expert phase stage plus one global "
            "phase plane; do not reuse a four-stage Student checkpoint. The "
            "frozen Teacher embedding cache remains reusable."
        )
    replacement.vision_surrogate.load_state_dict(payload["vision_optical"])
    replacement.language_surrogate.load_state_dict(payload["language_optical"])
    readout.load_state_dict(payload["retrieval_readout"])
    if optimizer is not None and "optimizer" in payload:
        optimizer.load_state_dict(payload["optimizer"])
    return payload


def train_optical_retrieval(
    loaded: LoadedBackbone,
    replacement: DeepStackMultimodalReplacement,
    readout: OpticalRetrievalReadout,
    bundle: GroceryRetrievalBundle,
    teacher_store: TeacherEmbeddingStore,
    settings: Settings,
) -> dict[str, Any]:
    train_dataset = GroceryRetrievalDataset(
        bundle.train_samples,
        settings.image_size,
        augment=settings.augmentation_enabled,
        crop_scale_min=settings.crop_scale_min,
        brightness_jitter=settings.brightness_jitter,
        contrast_jitter=settings.contrast_jitter,
        rotation_degrees=settings.rotation_degrees,
    )
    sampler = PKBatchSampler(
        bundle.train_samples,
        settings.pk_skus_per_batch,
        settings.pk_images_per_sku,
        settings.random_seed,
    )
    loader = DataLoader(
        train_dataset,
        batch_sampler=sampler,
        num_workers=settings.num_workers,
        pin_memory=loaded.device.type == "cuda",
        persistent_workers=settings.num_workers > 0,
        collate_fn=collate_grocery,
    )
    parameters = unique_trainable_parameters(replacement, readout)
    optimizer = torch.optim.AdamW(
        parameters,
        lr=settings.learning_rate,
        weight_decay=settings.weight_decay,
    )
    report = trainable_parameter_report(
        loaded.model, replacement, readout
    )
    write_json(settings.output_dir / "model.json", report)
    loaded.model.eval()
    replacement.use_student()
    best_train_loss = math.inf
    history_path = settings.output_dir / "train_log.csv"
    history_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "epoch",
        "learning_rate",
        "total_loss",
        "kd_loss",
        "retrieval_loss",
        "samples",
        "epoch_time_sec",
        "test_top1",
        "test_top3",
        "test_mrr",
        "checkpoint_selected_by",
    ]
    rows: list[dict[str, Any]] = []
    if history_path.is_file():
        history_path.unlink()
    amp_dtype = torch.bfloat16 if settings.dtype == "bfloat16" else torch.float16
    use_amp = settings.amp_enabled and loaded.device.type == "cuda"
    for epoch in range(1, settings.epochs + 1):
        sampler.set_epoch(epoch)
        replacement.set_student_train_mode()
        readout.train()
        totals = {"total": 0.0, "kd": 0.0, "ret": 0.0, "samples": 0}
        started = time.perf_counter()
        for batch_index, batch in enumerate(loader, 1):
            inputs = preprocess_images(
                loaded.processor, batch["images"], settings.instruction
            )
            validate_token_budgets(inputs, settings)
            inputs = move_inputs(inputs, loaded.device)
            labels = torch.tensor(
                [sample.sku_index for sample in batch["samples"]],
                device=loaded.device,
                dtype=torch.long,
            )
            teacher = teacher_store.lookup(batch["samples"]).to(
                loaded.device, non_blocking=True
            )
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=loaded.device.type,
                dtype=amp_dtype,
                enabled=use_amp,
            ):
                student, detector = student_embeddings(
                    loaded.model, replacement, readout, inputs
                )
                if detector.shape != (
                    len(batch["samples"]),
                    settings.detector_output_size,
                ):
                    raise RuntimeError(
                        f"Student detector output shape {tuple(detector.shape)} is invalid"
                    )
                kd = embedding_distillation_loss(student, teacher)
                retrieval = supervised_contrastive_loss(
                    student, labels, settings.temperature
                )
                total = settings.lambda_kd * kd + settings.lambda_ret * retrieval
            if not torch.isfinite(total):
                raise RuntimeError(
                    f"Non-finite loss at epoch={epoch} batch={batch_index}: "
                    f"total={total}, kd={kd}, retrieval={retrieval}"
                )
            total.backward()
            bad_gradients = [
                index
                for index, parameter in enumerate(parameters)
                if parameter.grad is not None and not torch.isfinite(parameter.grad).all()
            ]
            if bad_gradients:
                raise RuntimeError(f"Non-finite gradients in trainable tensors {bad_gradients}")
            optimizer.step()
            count = len(batch["samples"])
            totals["total"] += float(total.detach()) * count
            totals["kd"] += float(kd.detach()) * count
            totals["ret"] += float(retrieval.detach()) * count
            totals["samples"] += count
            if batch_index % settings.log_interval_batches == 0 or batch_index == len(loader):
                print(
                    f"epoch {epoch:03d}/{settings.epochs:03d} "
                    f"batch {batch_index:04d}/{len(loader):04d} "
                    f"loss={totals['total']/totals['samples']:.5f} "
                    f"kd={totals['kd']/totals['samples']:.5f} "
                    f"ret={totals['ret']/totals['samples']:.5f}"
                )
        sample_count = int(totals["samples"])
        average_total = totals["total"] / sample_count
        test_metrics: dict[str, Any] = {}
        if settings.evaluate_test_each_epoch:
            test_metrics = evaluate_student_split(
                loaded,
                replacement,
                readout,
                bundle.test_samples,
                bundle.gallery_samples,
                bundle.class_names,
                settings,
            )
        row = {
            "epoch": epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "total_loss": average_total,
            "kd_loss": totals["kd"] / sample_count,
            "retrieval_loss": totals["ret"] / sample_count,
            "samples": sample_count,
            "epoch_time_sec": time.perf_counter() - started,
            "test_top1": test_metrics.get("top1_retrieval_accuracy"),
            "test_top3": test_metrics.get("top3_retrieval_accuracy"),
            "test_mrr": test_metrics.get("mrr"),
            "checkpoint_selected_by": "training_total_loss",
        }
        rows.append(row)
        _write_history(history_path, rows, fieldnames)
        write_json(settings.output_dir / "metrics" / "training_latest.json", row)
        save_checkpoint(
            settings.output_dir / "last_checkpoint.pt",
            replacement,
            readout,
            optimizer,
            epoch,
            average_total,
            settings,
        )
        if average_total < best_train_loss:
            best_train_loss = average_total
            save_checkpoint(
                settings.output_dir / "best_train_loss_checkpoint.pt",
                replacement,
                readout,
                optimizer,
                epoch,
                average_total,
                settings,
            )
            write_json(
                settings.output_dir / "metrics" / "best_train_loss.json",
                {
                    "epoch": epoch,
                    "train_total_loss": average_total,
                    "selection_criterion": "minimum_training_total_loss",
                    "test_was_not_used_for_selection": True,
                },
            )
        print(
            f"epoch {epoch:03d} complete train_loss={average_total:.5f} "
            f"test_top1={test_metrics.get('top1_retrieval_accuracy', float('nan')):.4f} "
            f"best_train_loss={best_train_loss:.5f}"
        )
    return {
        "epochs": settings.epochs,
        "best_train_loss": best_train_loss,
        "last_train_loss": rows[-1]["total_loss"],
        "checkpoint_selection": "minimum training total loss (test not used)",
    }


@torch.no_grad()
def encode_student_samples(
    loaded: LoadedBackbone,
    replacement: DeepStackMultimodalReplacement,
    readout: OpticalRetrievalReadout,
    samples: Sequence[GrocerySample],
    settings: Settings,
) -> torch.Tensor:
    dataset = GroceryRetrievalDataset(samples, settings.image_size, augment=False)
    loader = DataLoader(
        dataset,
        batch_size=settings.inference_batch_size,
        shuffle=False,
        num_workers=settings.num_workers,
        pin_memory=loaded.device.type == "cuda",
        persistent_workers=settings.num_workers > 0,
        collate_fn=collate_grocery,
    )
    loaded.model.eval()
    replacement.use_student()
    replacement.vision_surrogate.eval()
    replacement.language_surrogate.eval()
    readout.eval()
    chunks: list[torch.Tensor] = []
    amp_dtype = torch.bfloat16 if settings.dtype == "bfloat16" else torch.float16
    use_amp = settings.amp_enabled and loaded.device.type == "cuda"
    for batch in loader:
        inputs = preprocess_images(
            loaded.processor, batch["images"], settings.instruction
        )
        validate_token_budgets(inputs, settings)
        inputs = move_inputs(inputs, loaded.device)
        with torch.autocast(
            device_type=loaded.device.type, dtype=amp_dtype, enabled=use_amp
        ):
            embeddings, _ = student_embeddings(
                loaded.model, replacement, readout, inputs
            )
        chunks.append(embeddings.detach().cpu())
    output = torch.cat(chunks, dim=0)
    if output.shape != (len(samples), settings.embedding_dim):
        raise RuntimeError(f"Encoded student embedding shape is {tuple(output.shape)}")
    return output


@torch.no_grad()
def evaluate_student_split(
    loaded: LoadedBackbone,
    replacement: DeepStackMultimodalReplacement,
    readout: OpticalRetrievalReadout,
    query_samples: Sequence[GrocerySample],
    gallery_samples: Sequence[GrocerySample],
    class_names: Sequence[str],
    settings: Settings,
) -> dict[str, Any]:
    gallery = encode_student_samples(
        loaded, replacement, readout, gallery_samples, settings
    )
    query = encode_student_samples(
        loaded, replacement, readout, query_samples, settings
    )
    return evaluate_embeddings(
        query,
        query_samples,
        gallery,
        gallery_samples,
        class_names,
        settings.gallery_aggregation,
        system_name="optical_student_query_vs_optical_student_gallery",
    ).metrics


def _write_history(
    path: Path, rows: list[dict[str, Any]], fieldnames: list[str]
) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    settings = load_settings(args.config)
    bundle = prepare_grocery_subset(settings, persist=True)
    teacher_store = TeacherEmbeddingStore(settings.teacher_cache_path, bundle, settings)
    device = torch.device(
        settings.device if settings.device != "cuda" or torch.cuda.is_available() else "cpu"
    )
    loaded = load_backbone(settings, device)
    replacement, readout = build_optical_student(loaded, settings)
    train_optical_retrieval(
        loaded, replacement, readout, bundle, teacher_store, settings
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
