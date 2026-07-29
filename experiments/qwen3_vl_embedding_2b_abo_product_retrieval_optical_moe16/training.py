from __future__ import annotations

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

from .datasets import (
    ABOBundle,
    ABORetrievalDataset,
    ABOSample,
    collate_abo,
)
from .io_utils import write_json
from .losses import cosine_distillation_loss, supervised_contrastive_loss
from .modeling import (
    LoadedVisionBackbone,
    TrainingIdentityHead,
    VisionOpticalRetrievalEncoder,
    parameter_report,
    preprocess_vision,
    unique_trainable_parameters,
)
from .teacher_cache import TeacherEmbeddingStore


class PKBatchSampler(Sampler[list[int]]):
    """Epoch-aware deterministic P-item x K-view sampler."""

    def __init__(
        self,
        samples: Sequence[ABOSample],
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
            grouped[sample.item_index].append(index)
        self.grouped = {
            key: tuple(values) for key, values in sorted(grouped.items())
        }
        if len(self.grouped) < self.p:
            raise ValueError(
                f"PK sampler needs P={self.p} items; found {len(self.grouped)}"
            )
        if any(len(values) < 2 for values in self.grouped.values()):
            raise ValueError("Every PK-sampled item must have at least two views")
        self.batch_count = max(1, math.ceil(len(samples) / (self.p * self.k)))

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.batch_count

    def __iter__(self) -> Iterator[list[int]]:
        generator = torch.Generator().manual_seed(
            self.seed + self.epoch * 1_000_003
        )
        item_ids = list(self.grouped)
        pools: dict[int, list[int]] = {}
        positions: dict[int, int] = {}
        for item_id in item_ids:
            order = torch.randperm(
                len(self.grouped[item_id]), generator=generator
            ).tolist()
            pools[item_id] = [
                self.grouped[item_id][position] for position in order
            ]
            positions[item_id] = 0
        item_order = torch.randperm(len(item_ids), generator=generator).tolist()
        cursor = 0
        for _ in range(self.batch_count):
            if cursor + self.p > len(item_order):
                item_order = torch.randperm(
                    len(item_ids), generator=generator
                ).tolist()
                cursor = 0
            selected = [
                item_ids[position]
                for position in item_order[cursor : cursor + self.p]
            ]
            cursor += self.p
            batch: list[int] = []
            for item_id in selected:
                for _ in range(self.k):
                    if positions[item_id] >= len(pools[item_id]):
                        order = torch.randperm(
                            len(self.grouped[item_id]), generator=generator
                        ).tolist()
                        pools[item_id] = [
                            self.grouped[item_id][position] for position in order
                        ]
                        positions[item_id] = 0
                    batch.append(pools[item_id][positions[item_id]])
                    positions[item_id] += 1
            yield batch


def save_encoder_checkpoint(
    path: Path,
    encoder: VisionOpticalRetrievalEncoder,
    *,
    stage: str,
    epoch: int,
    manifest_digest: str,
    optimizer: torch.optim.Optimizer | None = None,
    identity_head: TrainingIdentityHead | None = None,
    training_loss: float | None = None,
) -> None:
    payload: dict[str, Any] = {
        "format_version": 1,
        "stage": stage,
        "epoch": int(epoch),
        "manifest_sha256": manifest_digest,
        "training_loss": training_loss,
        "optical_core": encoder.core.state_dict(),
        "detector_projection": encoder.readout.state_dict(),
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if identity_head is not None:
        payload["training_only_identity_head"] = identity_head.state_dict()
        payload["identity_head_classes"] = identity_head.classifier.out_features
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_encoder_checkpoint(
    path: Path,
    encoder: VisionOpticalRetrievalEncoder,
    *,
    expected_manifest_digest: str | None = None,
    identity_head: TrainingIdentityHead | None = None,
) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Encoder checkpoint is missing: {path}")
    payload = torch.load(path, map_location="cpu")
    if (
        expected_manifest_digest is not None
        and payload.get("manifest_sha256") != expected_manifest_digest
    ):
        raise RuntimeError(
            "Checkpoint manifest digest differs from the fixed ABO split"
        )
    encoder.core.load_state_dict(payload["optical_core"])
    encoder.readout.load_state_dict(payload["detector_projection"])
    if identity_head is not None:
        if "training_only_identity_head" not in payload:
            raise RuntimeError("Checkpoint has no Stage-2 identity head")
        identity_head.load_state_dict(payload["training_only_identity_head"])
    return payload


def train_stage1(
    loaded: LoadedVisionBackbone,
    encoder: VisionOpticalRetrievalEncoder,
    bundle: ABOBundle,
    teacher_store: TeacherEmbeddingStore,
    settings: Any,
) -> Path:
    return _train(
        stage="stage1",
        loaded=loaded,
        encoder=encoder,
        dataset=bundle.stage1_train,
        bundle=bundle,
        teacher_store=teacher_store,
        settings=settings,
        epochs=settings.stage1_epochs,
        p=settings.stage1_pk_items,
        k=settings.stage1_pk_images,
        learning_rate=settings.stage1_learning_rate,
        identity_head=None,
    )


def train_stage2(
    loaded: LoadedVisionBackbone,
    encoder: VisionOpticalRetrievalEncoder,
    bundle: ABOBundle,
    teacher_store: TeacherEmbeddingStore,
    settings: Any,
    stage1_checkpoint: Path | None = None,
) -> Path:
    stage1_checkpoint = (
        stage1_checkpoint
        or settings.output_dir / "checkpoints" / "stage1_best_train_loss.pt"
    )
    load_encoder_checkpoint(
        stage1_checkpoint,
        encoder,
        expected_manifest_digest=bundle.manifest_digest,
    )
    identity_head = TrainingIdentityHead(
        settings.embedding_dim, len(bundle.stage2_item_ids)
    ).to(loaded.device)
    path = _train(
        stage="stage2",
        loaded=loaded,
        encoder=encoder,
        dataset=bundle.stage2_train,
        bundle=bundle,
        teacher_store=teacher_store,
        settings=settings,
        epochs=settings.stage2_epochs,
        p=settings.stage2_pk_items,
        k=settings.stage2_pk_images,
        learning_rate=settings.stage2_learning_rate,
        identity_head=identity_head,
    )
    # Deployment artifact deliberately omits the training-only identity head.
    payload = torch.load(path, map_location="cpu")
    deployment_path = settings.output_dir / "checkpoints" / "deployment_encoder.pt"
    torch.save(
        {
            "format_version": 1,
            "stage": "deployment",
            "source_checkpoint": str(path),
            "manifest_sha256": bundle.manifest_digest,
            "optical_core": payload["optical_core"],
            "detector_projection": payload["detector_projection"],
            "identity_head_included": False,
        },
        deployment_path,
    )
    return deployment_path


def _train(
    *,
    stage: str,
    loaded: LoadedVisionBackbone,
    encoder: VisionOpticalRetrievalEncoder,
    dataset: ABORetrievalDataset,
    bundle: ABOBundle,
    teacher_store: TeacherEmbeddingStore,
    settings: Any,
    epochs: int,
    p: int,
    k: int,
    learning_rate: float,
    identity_head: TrainingIdentityHead | None,
) -> Path:
    if stage not in {"stage1", "stage2"}:
        raise ValueError(stage)
    sampler = PKBatchSampler(
        dataset.samples, p, k, settings.random_seed + (0 if stage == "stage1" else 17)
    )
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=settings.num_workers,
        pin_memory=loaded.device.type == "cuda",
        persistent_workers=settings.num_workers > 0,
        collate_fn=collate_abo,
    )
    trainable_modules: list[nn.Module] = [encoder]
    if identity_head is not None:
        trainable_modules.append(identity_head)
    parameters = unique_trainable_parameters(*trainable_modules)
    optimizer = torch.optim.AdamW(
        parameters, lr=learning_rate, weight_decay=settings.weight_decay
    )
    report = parameter_report(encoder, identity_head)
    report.update(
        {
            "stage": stage,
            "manifest_sha256": bundle.manifest_digest,
            "training_samples": len(dataset),
            "training_items": len(
                {sample.item_id for sample in dataset.samples}
            ),
            "checkpoint_selection": "lowest training loss; held-out Query/Gallery are never used",
        }
    )
    write_json(settings.output_dir / f"{stage}_model.json", report)
    print(
        f"[{stage}] trainable parameters={report['total_trainable_parameters']:,} "
        f"tensors={report['trainable_tensors']}"
    )
    for row in report["trainable_parameter_list"]:
        print(
            f"  {row['name']} shape={row['shape']} params={row['parameters']:,}"
        )

    encoder.train()
    if identity_head is not None:
        identity_head.train()
    best_loss = math.inf
    rows: list[dict[str, Any]] = []
    history_path = settings.output_dir / "metrics" / f"{stage}_training_history.csv"
    checkpoint_dir = settings.output_dir / "checkpoints"
    amp_dtype = torch.bfloat16 if settings.dtype == "bfloat16" else torch.float16
    use_amp = settings.amp_enabled and loaded.device.type == "cuda"
    for epoch in range(1, epochs + 1):
        sampler.set_epoch(epoch)
        encoder.train()
        if identity_head is not None:
            identity_head.train()
        totals = defaultdict(float)
        started = time.perf_counter()
        for batch_index, batch in enumerate(loader, start=1):
            labels = batch["item_indices"].to(loaded.device, non_blocking=True)
            teacher = teacher_store.get(batch["image_ids"], loaded.device)
            inputs = preprocess_vision(
                loaded.processor, batch["images"], loaded.device
            )
            visual_counts = inputs["image_grid_thw"].long().prod(dim=-1)
            if int(visual_counts.max()) > settings.max_visual_tokens:
                raise RuntimeError(
                    f"visual token count {int(visual_counts.max())} exceeds "
                    f"max_visual_tokens={settings.max_visual_tokens}; no truncation allowed"
                )
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=loaded.device.type,
                dtype=amp_dtype,
                enabled=use_amp,
            ):
                student = encoder(
                    inputs["pixel_values"], inputs["image_grid_thw"]
                )
                kd = cosine_distillation_loss(student, teacher)
                supcon = supervised_contrastive_loss(
                    student, labels, settings.temperature
                )
                balance, importance = encoder.router_losses()
                if stage == "stage1":
                    identity = student.new_zeros(())
                    total = (
                        settings.lambda_stage1_kd * kd
                        + settings.lambda_stage1_supcon * supcon
                        + settings.lambda_router_balance * balance
                        + settings.lambda_router_importance * importance
                    )
                else:
                    assert identity_head is not None
                    logits = identity_head(student)
                    identity = F.cross_entropy(logits, labels)
                    total = (
                        settings.lambda_stage2_supcon * supcon
                        + settings.lambda_stage2_id * identity
                        + settings.lambda_stage2_kd * kd
                        + settings.lambda_router_balance * balance
                        + settings.lambda_router_importance * importance
                    )
            if not torch.isfinite(total):
                raise RuntimeError(
                    f"Non-finite {stage} loss at epoch={epoch}, batch={batch_index}"
                )
            total.backward()
            bad = [
                index
                for index, parameter in enumerate(parameters)
                if parameter.grad is not None
                and not torch.isfinite(parameter.grad).all()
            ]
            if bad:
                raise RuntimeError(f"Non-finite gradients in tensors {bad}")
            optimizer.step()
            count = len(batch["image_ids"])
            totals["samples"] += count
            totals["total"] += float(total.detach()) * count
            totals["kd"] += float(kd.detach()) * count
            totals["supcon"] += float(supcon.detach()) * count
            totals["identity"] += float(identity.detach()) * count
            totals["balance"] += float(balance.detach()) * count
            totals["importance"] += float(importance.detach()) * count
            if identity_head is not None:
                totals["id_correct"] += float(
                    logits.argmax(dim=-1).eq(labels).sum()
                )
            if (
                batch_index % settings.log_interval_batches == 0
                or batch_index == len(loader)
            ):
                denominator = max(1.0, totals["samples"])
                print(
                    f"[{stage}] epoch={epoch:03d}/{epochs:03d} "
                    f"batch={batch_index:,}/{len(loader):,} "
                    f"loss={totals['total']/denominator:.5f} "
                    f"kd={totals['kd']/denominator:.5f} "
                    f"supcon={totals['supcon']/denominator:.5f} "
                    f"id={totals['identity']/denominator:.5f} "
                    f"balance={totals['balance']/denominator:.5f}"
                )
        denominator = totals["samples"]
        average = totals["total"] / denominator
        row = {
            "epoch": epoch,
            "total_loss": average,
            "kd_loss": totals["kd"] / denominator,
            "supcon_loss": totals["supcon"] / denominator,
            "identity_loss": totals["identity"] / denominator,
            "identity_train_accuracy": (
                totals["id_correct"] / denominator
                if identity_head is not None
                else None
            ),
            "router_balance_loss": totals["balance"] / denominator,
            "router_importance_loss": totals["importance"] / denominator,
            "samples": int(denominator),
            "epoch_time_sec": time.perf_counter() - started,
            "checkpoint_selected_by": "training_loss",
            "held_out_evaluation_performed": False,
        }
        rows.append(row)
        _write_history(history_path, rows)
        last = checkpoint_dir / f"{stage}_last.pt"
        save_encoder_checkpoint(
            last,
            encoder,
            stage=stage,
            epoch=epoch,
            manifest_digest=bundle.manifest_digest,
            optimizer=optimizer,
            identity_head=identity_head,
            training_loss=average,
        )
        if average < best_loss:
            best_loss = average
            save_encoder_checkpoint(
                checkpoint_dir / f"{stage}_best_train_loss.pt",
                encoder,
                stage=stage,
                epoch=epoch,
                manifest_digest=bundle.manifest_digest,
                optimizer=optimizer,
                identity_head=identity_head,
                training_loss=average,
            )
        print(
            f"[{stage}] epoch={epoch:03d} complete loss={average:.5f} "
            f"best_train_loss={best_loss:.5f}"
        )
    return checkpoint_dir / f"{stage}_best_train_loss.pt"


def _write_history(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

