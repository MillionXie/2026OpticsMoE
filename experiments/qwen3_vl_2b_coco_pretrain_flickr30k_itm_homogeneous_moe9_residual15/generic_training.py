from __future__ import annotations

import time
import json
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from .features import move_inputs, multimodal_forward_features, pool_answer_hidden_state
from .io_utils import write_csv, write_json
from .processor_cache import ProcessorCacheStore
from .sam import SAMController
from .sampling import EpochRotatingSampler
from .teacher_cache import TeacherCacheStore
from .training import CachedStudentDataset, cached_student_collate, normalized_packed_group_mse


def train_generic_distillation(model: torch.nn.Module, replacement: Any, dataset: Dataset[Any],
                               teacher_store: TeacherCacheStore, input_store: ProcessorCacheStore,
                               settings: Any, device: torch.device) -> None:
    if settings.student_language_mode != "optical_moe":
        raise RuntimeError("Generic joint multimodal pretraining requires student_language_mode=optical_moe")
    dummy_logits = torch.zeros(len(dataset), dtype=torch.float32)
    cached = CachedStudentDataset(dataset, teacher_store, input_store, dummy_logits)
    sampler = EpochRotatingSampler(len(dataset), settings.generic_train_samples_per_epoch,
                                   settings.seed, settings.teacher_cache_shard_size)
    loader = DataLoader(cached, batch_size=settings.student_batch_size, sampler=sampler, num_workers=0,
                        collate_fn=lambda batch: cached_student_collate(batch, input_store.metadata), pin_memory=True)
    replacement.use_student(); model.requires_grad_(False).eval(); replacement.configure_student_trainability()
    routers = [*replacement.vision_surrogate.core.prompt.router.parameters(),
               *replacement.language_surrogate.core.prompt.router.parameters()]
    router_ids = {id(parameter) for parameter in routers}
    optical = [parameter for parameter in replacement.trainable_parameters() if id(parameter) not in router_ids]
    optimizer_cls = torch.optim.AdamW if settings.optimizer_type == "adamw" else torch.optim.Adam
    optimizer = optimizer_cls([
        {"params": optical, "lr": settings.generic_learning_rate, "group_name": "generic_optical"},
        {"params": routers, "lr": settings.generic_router_learning_rate, "group_name": "generic_routers"},
    ], weight_decay=settings.weight_decay)
    all_trainable = optical + routers
    use_sam = bool(settings.sam_enabled and settings.sam_apply_to_generic_pretrain)
    sam = SAMController(optimizer, all_trainable, settings.sam_rho, settings.sam_adaptive) if use_sam else None
    scheduler = (torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=settings.generic_epochs)
                 if settings.scheduler_type == "cosine" else None)
    history: list[dict[str, Any]] = []
    for epoch in range(1, settings.generic_epochs + 1):
        started = time.perf_counter(); sampler.set_epoch(epoch); replacement.set_phase_dropout_active(False)
        replacement.set_student_train_mode(); totals = torch.zeros(7, device=device, dtype=torch.float64); seen = 0
        names = ("total", "vision", "answer", "vision_balance", "language_balance",
                 "vision_importance", "language_importance")
        for batch_index, (cpu_inputs, _labels, _indices, teacher_targets, _dummy_logits) in enumerate(loader, 1):
            inputs = move_inputs(cpu_inputs, device)
            teacher_targets = _targets_to_device(teacher_targets, device)

            def forward_loss() -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
                replacement.prepare_student_batch(cpu_inputs["attention_mask"])
                hidden = multimodal_forward_features(model, inputs)
                answer, _ = pool_answer_hidden_state(hidden, inputs["attention_mask"])
                taps = [*replacement.vision_surrogate.tap_outputs, replacement.vision_surrogate.last_output]
                vision = torch.stack([
                    normalized_packed_group_mse(student, teacher, teacher_targets["visual_token_counts"])
                    for student, teacher in zip(taps, teacher_targets["vision_taps"])
                ]).mean()
                teacher_answer = teacher_targets["answer_hidden"]
                answer_loss = F.mse_loss(F.layer_norm(answer.float(), (answer.shape[-1],)),
                                         F.layer_norm(teacher_answer, (teacher_answer.shape[-1],)))
                router = replacement.router_losses()
                total = (settings.generic_loss_vision_weight * vision +
                         settings.generic_loss_answer_weight * answer_loss +
                         settings.generic_router_balance_weight *
                         (router["vision_balance"] + router["language_balance"]) +
                         settings.generic_router_importance_weight *
                         (router["vision_importance"] + router["language_importance"]))
                values = (total, vision, answer_loss, router["vision_balance"], router["language_balance"],
                          router["vision_importance"], router["language_importance"])
                return total, values

            optimizer.zero_grad(set_to_none=True)
            loss, values = forward_loss(); loss.backward()
            if sam is not None:
                sam.first_step(zero_grad=True)
                second_loss, _ = forward_loss(); second_loss.backward(); sam.second_step(zero_grad=True)
            else:
                optimizer.step()
            batch_size = len(cpu_inputs["input_ids"]); seen += batch_size
            totals += torch.stack([value.detach().double() for value in values]) * batch_size
            if batch_index % settings.log_interval_batches == 0 or batch_index == len(loader):
                means = (totals / seen).cpu().tolist()
                print(f"generic epoch {epoch}/{settings.generic_epochs} batch {batch_index}/{len(loader)} "
                      f"total={means[0]:.5f} vision={means[1]:.5f} answer={means[2]:.5f} "
                      f"v_bal={means[3]:.4f} l_bal={means[4]:.4f} sam={use_sam}", flush=True)
        if scheduler is not None: scheduler.step()
        means = dict(zip(names, (totals / seen).cpu().tolist()))
        row = {"epoch": epoch, **{f"loss_{key}": value for key, value in means.items()},
               "samples_this_epoch": len(sampler), "epoch_time_sec": time.perf_counter() - started,
               "sam_enabled": use_sam, "sam_rho": settings.sam_rho if use_sam else None,
               "teacher_fine_tuned": False, "downstream_task_labels_used": False}
        history.append(row)
        root = settings.output_dir / "generic_pretrain"
        write_csv(root / "metrics" / "training_history.csv", history, list(row))
        write_json(root / "metrics" / "training_latest.json", row)
        save_generic_checkpoint(root, replacement, "last", epoch, row, settings)
        print(f"generic epoch {epoch:03d} complete loss={means['total']:.5f} samples={len(sampler)}", flush=True)
    save_generic_checkpoint(settings.output_dir / "generic_pretrain", replacement, "final",
                            settings.generic_epochs, history[-1], settings)
    write_json(settings.output_dir / "generic_pretrain" / "metrics" / "training.json", {
        "epochs": settings.generic_epochs, "checkpoint_selection": "fixed final epoch",
        "losses": ["normalized_vision_hidden_mse", "normalized_answer_hidden_mse", "router_regularization"],
        "task_head_used": False, "teacher_fine_tuned": False, "sam_enabled": use_sam,
    })


def _targets_to_device(targets: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        "visual_token_counts": targets["visual_token_counts"].to(device, non_blocking=True),
        "vision_taps": [value.to(device=device, dtype=torch.float32, non_blocking=True)
                        for value in targets["vision_taps"]],
        "answer_hidden": targets["answer_hidden"].to(device=device, dtype=torch.float32, non_blocking=True),
    }


def save_generic_checkpoint(root: Path, replacement: Any, tag: str, epoch: int,
                            metrics: dict[str, Any], settings: Any) -> None:
    checkpoint = root / "checkpoints"; checkpoint.mkdir(parents=True, exist_ok=True)
    metadata = {"epoch": epoch, "metrics": metrics, "generic_manifest_digest": settings.generic_manifest_digest,
                "architecture": replacement.alignment_specification(), "student_language_mode": replacement.language_mode,
                "physical_depth": replacement.vision_surrogate.core.total_physical_layers}
    torch.save({"state_dict": replacement.vision_surrogate.state_dict(), **metadata}, checkpoint / f"vision_moe_{tag}.pt")
    torch.save({"state_dict": replacement.language_surrogate.state_dict(), **metadata}, checkpoint / f"language_moe_{tag}.pt")


def load_generic_checkpoint(root: Path, replacement: Any, tag: str = "final") -> None:
    checkpoint = root / "generic_pretrain" / "checkpoints"
    vision_path = checkpoint / f"vision_moe_{tag}.pt"; language_path = checkpoint / f"language_moe_{tag}.pt"
    if not vision_path.is_file() or not language_path.is_file():
        raise FileNotFoundError(
            f"Generic pretraining checkpoint is incomplete under {checkpoint}. Run --phase generic_pretrain first."
        )
    vision = torch.load(vision_path, map_location="cpu", weights_only=True)
    language = torch.load(language_path, map_location="cpu", weights_only=True)
    manifest_metadata_path = root / "generic_pretrain" / "manifests" / "train_metadata.json"
    if not manifest_metadata_path.is_file():
        raise FileNotFoundError(f"Generic checkpoint provenance manifest is missing: {manifest_metadata_path}")
    manifest_digest = json.loads(manifest_metadata_path.read_text(encoding="utf-8")).get("sha256")
    for path, payload in ((vision_path, vision), (language_path, language)):
        if payload.get("generic_manifest_digest") != manifest_digest:
            raise RuntimeError(f"Generic checkpoint/manifest digest mismatch in {path}")
    current = replacement.alignment_specification()
    for path, payload in ((vision_path, vision), (language_path, language)):
        saved = payload.get("architecture", {})
        mismatched = [key for key in ("native_pre_attention_enabled", "native_pre_norm_enabled",
                                      "transformer_residual_enabled", "logical_optical_stages",
                                      "physical_layers_per_logical_stage", "total_physical_layers")
                      if saved.get(key) != current.get(key)]
        if mismatched: raise RuntimeError(f"Generic checkpoint architecture mismatch in {path}: {mismatched}")
    replacement.vision_surrogate.load_state_dict(vision["state_dict"])
    replacement.language_surrogate.load_state_dict(language["state_dict"])
