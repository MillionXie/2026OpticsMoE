from __future__ import annotations

import csv
import json
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterator, Sequence

import torch
from torch import nn
from torch.nn import functional as F
from experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval.optics.physical import (
    phase_dc_loss,
    phase_dc_statistics,
)
from torch.utils.data import DataLoader, Sampler

from .datasets import (
    ABOBundle,
    ABORetrievalDataset,
    ABOSample,
    AugmentedABORetrievalDataset,
    collate_abo,
)
from .io_utils import write_json
from .losses import (
    CrossBatchMemory,
    cosine_distillation_loss,
    relational_similarity_loss,
    supervised_contrastive_loss_with_memory,
)
from .modeling import (
    LoadedBackbone,
    LoadedVisionBackbone,
    MultimodalOpticalRetrievalEncoder,
    VisionOpticalRetrievalEncoder,
    encode_product_images,
    parameter_report,
    unique_trainable_parameters,
)
from .teacher_adapter import CosineIdentityHead
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
    encoder: VisionOpticalRetrievalEncoder | MultimodalOpticalRetrievalEncoder,
    *,
    stage: str,
    epoch: int,
    manifest_digest: str,
    optimizer: torch.optim.Optimizer | None = None,
    identity_head: nn.Module | None = None,
    training_loss: float | None = None,
) -> None:
    payload: dict[str, Any] = {
        "format_version": 2,
        "stage": stage,
        "epoch": int(epoch),
        "manifest_sha256": manifest_digest,
        "training_loss": training_loss,
        "student_architecture": encoder.settings.student_architecture,
        "encoder_state": encoder.deployment_state_dict(),
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if identity_head is not None:
        payload["training_only_identity_head"] = identity_head.state_dict()
        payload["identity_head_classes"] = int(
            next(identity_head.parameters()).shape[0]
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_encoder_checkpoint(
    path: Path,
    encoder: VisionOpticalRetrievalEncoder | MultimodalOpticalRetrievalEncoder,
    *,
    expected_manifest_digest: str | None = None,
    identity_head: nn.Module | None = None,
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
    if "encoder_state" in payload:
        saved_architecture = payload.get("student_architecture")
        if saved_architecture != encoder.settings.student_architecture:
            raise RuntimeError(
                f"Checkpoint Student architecture={saved_architecture!r}, "
                f"current={encoder.settings.student_architecture!r}"
            )
        encoder.load_deployment_state_dict(payload["encoder_state"])
    else:
        if not isinstance(encoder, VisionOpticalRetrievalEncoder):
            raise RuntimeError(
                "Legacy vision-only checkpoint cannot initialize multimodal optical Student"
            )
        encoder.core.load_state_dict(payload["optical_core"])
        encoder.readout.load_state_dict(payload["detector_projection"])
    if identity_head is not None:
        if "training_only_identity_head" not in payload:
            raise RuntimeError("Checkpoint has no Stage-2 identity head")
        identity_head.load_state_dict(payload["training_only_identity_head"])
    return payload


def train_stage1(
    loaded: LoadedVisionBackbone | LoadedBackbone,
    encoder: VisionOpticalRetrievalEncoder | MultimodalOpticalRetrievalEncoder,
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
    loaded: LoadedVisionBackbone | LoadedBackbone,
    encoder: VisionOpticalRetrievalEncoder | MultimodalOpticalRetrievalEncoder,
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
    identity_head = CosineIdentityHead(
        settings.embedding_dim,
        len(bundle.stage2_item_ids),
        scale=settings.identity_scale,
        margin=settings.identity_margin,
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
            "format_version": 2,
            "stage": "deployment",
            "source_checkpoint": str(path),
            "manifest_sha256": bundle.manifest_digest,
            "student_architecture": settings.student_architecture,
            "encoder_state": payload["encoder_state"],
            "identity_head_included": False,
        },
        deployment_path,
    )
    return deployment_path


def _train(
    *,
    stage: str,
    loaded: LoadedVisionBackbone | LoadedBackbone,
    encoder: VisionOpticalRetrievalEncoder | MultimodalOpticalRetrievalEncoder,
    dataset: ABORetrievalDataset,
    bundle: ABOBundle,
    teacher_store: TeacherEmbeddingStore,
    settings: Any,
    epochs: int,
    p: int,
    k: int,
    learning_rate: float,
    identity_head: nn.Module | None,
) -> Path:
    if stage not in {"stage1", "stage2"}:
        raise ValueError(stage)
    sampler = PKBatchSampler(
        dataset.samples, p, k, settings.random_seed + (0 if stage == "stage1" else 17)
    )
    training_dataset: ABORetrievalDataset | AugmentedABORetrievalDataset = (
        AugmentedABORetrievalDataset(dataset, settings)
        if settings.augmentation_enabled
        else dataset
    )
    loader = DataLoader(
        training_dataset,
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
    optimizer = _build_optimizer(encoder, identity_head, settings, learning_rate)
    memory = CrossBatchMemory(
        settings.contrastive_memory_size, settings.embedding_dim
    ).to(loaded.device)
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
        router_temperature = _router_temperature(epoch, epochs, settings)
        encoder.set_router_temperature(router_temperature)
        encoder.train()
        if identity_head is not None:
            identity_head.train()
        totals = defaultdict(float)
        router_selection_sums: dict[str, torch.Tensor] = {}
        router_weight_sums: dict[str, torch.Tensor] = {}
        router_sample_counts: dict[str, int] = defaultdict(int)
        started = time.perf_counter()
        for batch_index, batch in enumerate(loader, start=1):
            labels = batch["item_indices"].to(loaded.device, non_blocking=True)
            teacher = teacher_store.get(batch["image_ids"], loaded.device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=loaded.device.type,
                dtype=amp_dtype,
                enabled=use_amp,
            ):
                student = encode_product_images(
                    loaded, encoder, batch["images"], settings
                )
                kd = cosine_distillation_loss(student, teacher)
                relational = relational_similarity_loss(student, teacher)
                supcon = supervised_contrastive_loss_with_memory(
                    student, labels, settings.temperature, memory
                )
                router = encoder.router_loss_components()
                balance = (
                    router["vision_balance"] + router["language_balance"]
                )
                importance = (
                    router["vision_importance"] + router["language_importance"]
                )
                entropy_penalty = (
                    router["vision_entropy_penalty"]
                    + router["language_entropy_penalty"]
                )
                dc = (
                    phase_dc_loss(encoder)
                    if settings.lambda_phase_dc > 0.0
                    else student.new_zeros(())
                )
                if stage == "stage1":
                    identity = student.new_zeros(())
                    total = (
                        settings.lambda_stage1_kd * kd
                        + settings.lambda_stage1_supcon * supcon
                        + settings.lambda_relational_kd * relational
                        + settings.lambda_router_balance * balance
                        + settings.lambda_router_importance * importance
                        + settings.lambda_router_entropy * entropy_penalty
                        + settings.lambda_phase_dc * dc
                    )
                else:
                    assert identity_head is not None
                    logits = identity_head(student, labels)
                    identity = F.cross_entropy(logits, labels)
                    total = (
                        settings.lambda_stage2_supcon * supcon
                        + settings.lambda_stage2_id * identity
                        + settings.lambda_stage2_kd * kd
                        + settings.lambda_relational_kd * relational
                        + settings.lambda_router_balance * balance
                        + settings.lambda_router_importance * importance
                        + settings.lambda_router_entropy * entropy_penalty
                        + settings.lambda_phase_dc * dc
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
            if settings.gradient_clip_norm:
                torch.nn.utils.clip_grad_norm_(
                    parameters, settings.gradient_clip_norm
                )
            optimizer.step()
            memory.enqueue(student, labels)
            for router_name, diagnostics in encoder.router_diagnostics().items():
                selected = diagnostics["selected_mask"].detach().float().sum(dim=0).cpu()
                weights = diagnostics["weights"].detach().float().sum(dim=0).cpu()
                if router_name not in router_selection_sums:
                    router_selection_sums[router_name] = torch.zeros_like(selected)
                    router_weight_sums[router_name] = torch.zeros_like(weights)
                router_selection_sums[router_name] += selected
                router_weight_sums[router_name] += weights
                router_sample_counts[router_name] += diagnostics["weights"].shape[0]
            count = len(batch["image_ids"])
            totals["samples"] += count
            totals["total"] += float(total.detach()) * count
            totals["kd"] += float(kd.detach()) * count
            totals["supcon"] += float(supcon.detach()) * count
            totals["relational"] += float(relational.detach()) * count
            totals["identity"] += float(identity.detach()) * count
            totals["balance"] += float(balance.detach()) * count
            totals["importance"] += float(importance.detach()) * count
            totals["entropy"] += float(entropy_penalty.detach()) * count
            totals["phase_dc"] += float(dc.detach()) * count
            totals["vision_balance"] += float(
                router["vision_balance"].detach()
            ) * count
            totals["vision_importance"] += float(
                router["vision_importance"].detach()
            ) * count
            totals["language_balance"] += float(
                router["language_balance"].detach()
            ) * count
            totals["language_importance"] += float(
                router["language_importance"].detach()
            ) * count
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
                    f"balance={totals['balance']/denominator:.5f} "
                    f"importance={totals['importance']/denominator:.5f} "
                    f"phase_dc={totals['phase_dc']/denominator:.5f} "
                    f"router_T={router_temperature:.3f}"
                )
        denominator = totals["samples"]
        average = totals["total"] / denominator
        row = {
            "epoch": epoch,
            "total_loss": average,
            "kd_loss": totals["kd"] / denominator,
            "supcon_loss": totals["supcon"] / denominator,
            "relational_kd_loss": totals["relational"] / denominator,
            "identity_loss": totals["identity"] / denominator,
            "identity_train_accuracy": (
                totals["id_correct"] / denominator
                if identity_head is not None
                else None
            ),
            "router_balance_loss": totals["balance"] / denominator,
            "router_importance_loss": totals["importance"] / denominator,
            "router_entropy_penalty": totals["entropy"] / denominator,
            "phase_dc_loss": totals["phase_dc"] / denominator,
            "phase_dc_weighted_loss": (
                settings.lambda_phase_dc * totals["phase_dc"] / denominator
            ),
            **phase_dc_statistics(encoder),
            "vision_router_balance_loss": totals["vision_balance"] / denominator,
            "vision_router_importance_loss": (
                totals["vision_importance"] / denominator
            ),
            "language_router_balance_loss": (
                totals["language_balance"] / denominator
            ),
            "language_router_importance_loss": (
                totals["language_importance"] / denominator
            ),
            "router_temperature": router_temperature,
            "contrastive_memory_size": int(memory.size),
            "samples": int(denominator),
            "epoch_time_sec": time.perf_counter() - started,
            "checkpoint_selected_by": "training_loss",
            "held_out_evaluation_performed": False,
        }
        for router_name in ("vision", "language"):
            samples_seen = router_sample_counts.get(router_name, 0)
            if samples_seen:
                row[f"{router_name}_expert_selection_rates"] = json.dumps(
                    (
                        router_selection_sums[router_name] / samples_seen
                    ).tolist()
                )
                row[f"{router_name}_expert_mean_weights"] = json.dumps(
                    (
                        router_weight_sums[router_name] / samples_seen
                    ).tolist()
                )
            else:
                row[f"{router_name}_expert_selection_rates"] = ""
                row[f"{router_name}_expert_mean_weights"] = ""
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


def _build_optimizer(
    encoder: nn.Module,
    identity_head: nn.Module | None,
    settings: Any,
    fallback_learning_rate: float,
) -> torch.optim.Optimizer:
    if not settings.use_grouped_learning_rates:
        parameters = unique_trainable_parameters(
            *([encoder] + ([identity_head] if identity_head is not None else []))
        )
        return torch.optim.AdamW(
            parameters,
            lr=fallback_learning_rate,
            weight_decay=settings.weight_decay,
        )
    groups: dict[str, list[nn.Parameter]] = {
        "phase": [],
        "router": [],
        "electronic": [],
        "head": [],
    }
    seen: set[int] = set()
    for name, parameter in encoder.named_parameters():
        if not parameter.requires_grad or id(parameter) in seen:
            continue
        seen.add(id(parameter))
        if "raw_phase" in name:
            groups["phase"].append(parameter)
        elif "router" in name:
            groups["router"].append(parameter)
        else:
            groups["electronic"].append(parameter)
    if identity_head is not None:
        for parameter in identity_head.parameters():
            if parameter.requires_grad and id(parameter) not in seen:
                seen.add(id(parameter))
                groups["head"].append(parameter)
    expected = unique_trainable_parameters(
        *([encoder] + ([identity_head] if identity_head is not None else []))
    )
    if {id(parameter) for parameter in expected} != seen:
        raise RuntimeError("Optimizer parameter grouping lost or duplicated tensors")
    values = [
        {
            "name": "phase",
            "params": groups["phase"],
            "lr": settings.phase_learning_rate or fallback_learning_rate,
            "weight_decay": 0.0,
        },
        {
            "name": "router",
            "params": groups["router"],
            "lr": settings.router_learning_rate or fallback_learning_rate,
            "weight_decay": settings.electronic_weight_decay,
        },
        {
            "name": "electronic",
            "params": groups["electronic"],
            "lr": settings.adapter_learning_rate or fallback_learning_rate,
            "weight_decay": settings.electronic_weight_decay,
        },
        {
            "name": "identity_head",
            "params": groups["head"],
            "lr": settings.head_learning_rate or fallback_learning_rate,
            "weight_decay": settings.electronic_weight_decay,
        },
    ]
    return torch.optim.AdamW(
        [value for value in values if value["params"]]
    )


def _router_temperature(epoch: int, epochs: int, settings: Any) -> float:
    warmup = min(int(settings.router_temperature_warmup_epochs), int(epochs))
    if warmup <= 1 or epoch >= warmup:
        return float(settings.router_temperature_final)
    fraction = float(epoch - 1) / float(warmup - 1)
    return (
        float(settings.router_temperature) * (1.0 - fraction)
        + float(settings.router_temperature_final) * fraction
    )
