from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
import math
import os
import time
from pathlib import Path
from typing import Any, Iterable

import torch
from torch import nn
from torch.nn import functional as F

from experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval.features import (
    move_inputs,
    validate_token_budgets,
)
from experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval.modeling import (
    LoadedBackbone,
    unique_trainable_parameters,
)
from experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval.optics.physical import (
    phase_dc_loss,
)

from .data import CIFAR10DataBundle, build_eval_loader, build_train_loader
from .modeling import (
    CIFAR10ClassificationHead,
    RouterAblationReplacement,
    classification_logits,
)
from .preprocessing import cached_preprocess_images as preprocess_images


def _unique(values: Iterable[nn.Parameter]) -> list[nn.Parameter]:
    output: list[nn.Parameter] = []
    seen: set[int] = set()
    for value in values:
        if value.requires_grad and id(value) not in seen:
            seen.add(id(value))
            output.append(value)
    return output


def build_optimizer(
    replacement: RouterAblationReplacement,
    head: CIFAR10ClassificationHead,
    settings: Any,
) -> tuple[torch.optim.Optimizer, list[nn.Parameter]]:
    parameters = unique_trainable_parameters(replacement, head)
    router_parameters = _unique(replacement.router_parameters())
    phase_parameters = _unique(
        parameter
        for group in replacement.phase_parameter_groups().values()
        for parameter in group
    )
    head_parameters = _unique(head.parameters())
    adapter_parameters: list[nn.Parameter] = []
    for surrogate in (replacement.vision_surrogate, replacement.language_surrogate):
        for name in ("input_adapter", "input_norm", "output_adapter"):
            module = getattr(surrogate.core, name, None)
            if module is not None:
                adapter_parameters.extend(module.parameters())
    adapter_parameters = _unique(adapter_parameters)

    groups = {
        "routers": router_parameters,
        "optical_phases": phase_parameters,
        "classification_head": head_parameters,
        "optical_adapters": adapter_parameters,
    }
    owner: dict[int, str] = {}
    for name, group in groups.items():
        for parameter in group:
            previous = owner.setdefault(id(parameter), name)
            if previous != name:
                raise RuntimeError(
                    f"Optimizer parameter belongs to both {previous} and {name}"
                )
    base_parameters = [parameter for parameter in parameters if id(parameter) not in owner]
    groups["student_base"] = base_parameters
    expected = {id(parameter) for parameter in parameters}
    actual = {
        id(parameter) for group in groups.values() for parameter in group
    }
    if expected != actual:
        raise RuntimeError("Optimizer groups do not exactly cover trainable parameters")

    learning_rates = {
        "student_base": float(settings.learning_rate),
        "optical_adapters": float(
            settings.adapter_learning_rate
            if settings.adapter_learning_rate is not None
            else settings.learning_rate
        ),
        "classification_head": float(settings.classification_head_learning_rate),
        "optical_phases": float(
            settings.phase_learning_rate
            if settings.phase_learning_rate is not None
            else settings.learning_rate
        ),
        "routers": float(
            settings.router_learning_rate
            if settings.router_learning_rate is not None
            else settings.learning_rate
        ),
    }
    specs = []
    for name in (
        "student_base",
        "optical_adapters",
        "classification_head",
        "optical_phases",
        "routers",
    ):
        if groups[name]:
            specs.append(
                {
                    "params": groups[name],
                    "lr": learning_rates[name],
                    "initial_lr": learning_rates[name],
                    "group_name": name,
                    "weight_decay": 0.0,
                }
            )
    optimizer = torch.optim.AdamW(specs, weight_decay=0.0)
    return optimizer, parameters


def _learning_rate_scale(step: int, total_steps: int, warmup_ratio: float) -> float:
    warmup_steps = max(1, int(round(total_steps * warmup_ratio)))
    if step < warmup_steps:
        return float(step + 1) / float(warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


def _apply_lr_scale(optimizer: torch.optim.Optimizer, scale: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = float(group["initial_lr"]) * scale


def _amp(settings: Any, device: torch.device) -> tuple[bool, torch.dtype]:
    enabled = bool(settings.amp_enabled and device.type == "cuda")
    dtype = torch.bfloat16 if settings.dtype == "bfloat16" else torch.float16
    return enabled, dtype


def _auxiliary_loss(
    replacement: RouterAblationReplacement,
    settings: Any,
    reference: torch.Tensor,
    epoch: int,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    routing = replacement.router_losses()
    balance = 0.5 * (routing["vision_balance"] + routing["language_balance"])
    importance = 0.5 * (
        routing["vision_importance"] + routing["language_importance"]
    )
    response = (
        replacement.router_response_consistency_loss()
        if float(getattr(settings, "lambda_router_response_consistency", 0.0)) > 0.0
        else reference.new_zeros(())
    )
    dc_active = bool(
        getattr(settings, "phase_dc_enabled", False)
        and float(getattr(settings, "lambda_phase_dc", 0.0)) > 0.0
        and epoch >= int(getattr(settings, "phase_dc_start_epoch", 1))
    )
    dc = phase_dc_loss(replacement) if dc_active else reference.new_zeros(())
    auxiliary = replacement.auxiliary_losses()
    ccd = auxiliary.get("ccd_operating_point", reference.new_zeros(()))
    weighted = (
        float(settings.lambda_router_balance) * balance
        + float(settings.lambda_router_importance) * importance
        + float(getattr(settings, "lambda_router_response_consistency", 0.0))
        * response
        + float(getattr(settings, "lambda_phase_dc", 0.0)) * dc
        + float(getattr(settings, "lambda_ccd_operating_point", 0.0)) * ccd
    )
    return weighted, {
        "balance": balance,
        "importance": importance,
        "router_response": response,
        "phase_dc": dc,
        "ccd_operating_point": ccd,
    }


def _router_snapshot(replacement: RouterAblationReplacement) -> dict[str, float]:
    output: dict[str, float] = {}
    for prefix, surrogate in (
        ("vision", replacement.vision_surrogate),
        ("language", replacement.language_surrogate),
    ):
        routing = surrogate.core.last_routing
        selected = routing["selected_mask"].detach()
        importance = routing["importance"].detach()
        output[f"{prefix}_router_entropy"] = float(
            routing["normalized_entropy"].detach()
        )
        output[f"{prefix}_active_experts"] = float(selected.any(dim=0).sum())
        output[f"{prefix}_max_importance"] = float(importance.max())
    return output


def _prepare(
    loaded: LoadedBackbone, images: list[Any], settings: Any
) -> dict[str, torch.Tensor]:
    inputs = _prepare_cpu(loaded, images, settings)
    return move_inputs(inputs, loaded.device)


def _prepare_cpu(
    loaded: LoadedBackbone, images: list[Any], settings: Any
) -> dict[str, torch.Tensor]:
    """Prepare one CPU batch without touching CUDA.

    Keeping CPU preprocessing separate lets the next batch be prepared while
    the current batch runs on the GPU.  The same processor and validation path
    are used, so the tensors and training objective are unchanged.
    """
    inputs = preprocess_images(loaded.processor, images, settings.instruction)
    validate_token_budgets(inputs, settings)
    return inputs


def _prepared_batches(
    loader: Any,
    loaded: LoadedBackbone,
    settings: Any,
) -> Iterable[tuple[dict[str, torch.Tensor], torch.Tensor]]:
    """Overlap CPU preprocessing for batch N+1 with GPU work for batch N."""
    iterator = iter(loader)
    try:
        images, labels = next(iterator)
    except StopIteration:
        return

    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="qwen-preprocess") as pool:
        pending = pool.submit(_prepare_cpu, loaded, images, settings)
        while True:
            inputs_cpu = pending.result()
            try:
                next_images, next_labels = next(iterator)
            except StopIteration:
                next_images = None
                next_labels = None
            if next_images is not None:
                pending = pool.submit(_prepare_cpu, loaded, next_images, settings)

            yield move_inputs(inputs_cpu, loaded.device), labels

            if next_images is None:
                break
            labels = next_labels


@torch.inference_mode()
def evaluate(
    loaded: LoadedBackbone,
    replacement: RouterAblationReplacement,
    head: CIFAR10ClassificationHead,
    dataset: Any,
    settings: Any,
    *,
    split: str,
) -> dict[str, Any]:
    loaded.model.eval()
    replacement.use_student()
    replacement.vision_surrogate.eval()
    replacement.language_surrogate.eval()
    replacement.set_phase_dropout_active(False)
    head.eval()
    loader = build_eval_loader(dataset, settings)
    use_amp, amp_dtype = _amp(settings, loaded.device)
    total_loss = 0.0
    correct1 = 0
    correct5 = 0
    count = 0
    confusion = torch.zeros(settings.num_classes, settings.num_classes, dtype=torch.long)
    if loaded.device.type == "cuda":
        torch.cuda.synchronize(loaded.device)
    started = time.perf_counter()
    for inputs, labels_cpu in _prepared_batches(loader, loaded, settings):
        labels = labels_cpu.to(loaded.device, non_blocking=True)
        with torch.autocast(
            device_type=loaded.device.type, dtype=amp_dtype, enabled=use_amp
        ):
            logits, _ = classification_logits(
                loaded.model, replacement, head, inputs
            )
            loss = F.cross_entropy(logits, labels)
        predictions = logits.argmax(dim=1)
        top5 = logits.topk(min(5, settings.num_classes), dim=1).indices
        batch_count = labels.numel()
        total_loss += float(loss) * batch_count
        correct1 += int(predictions.eq(labels).sum())
        correct5 += int(top5.eq(labels[:, None]).any(dim=1).sum())
        count += batch_count
        encoded = (
            labels.detach().cpu() * settings.num_classes
            + predictions.detach().cpu()
        )
        confusion += torch.bincount(
            encoded, minlength=settings.num_classes**2
        ).reshape(settings.num_classes, settings.num_classes)
    if loaded.device.type == "cuda":
        torch.cuda.synchronize(loaded.device)
    elapsed = time.perf_counter() - started
    per_class = []
    for index, name in enumerate(settings.class_names):
        denominator = int(confusion[index].sum())
        per_class.append(
            {
                "class_index": index,
                "class_name": name,
                "accuracy": (
                    float(confusion[index, index]) / denominator if denominator else 0.0
                ),
                "samples": denominator,
            }
        )
    metrics = {
        "split": split,
        "loss": total_loss / max(1, count),
        "accuracy": correct1 / max(1, count),
        "top5_accuracy": correct5 / max(1, count),
        "samples": count,
        "elapsed_sec": elapsed,
        "images_per_sec": count / max(elapsed, 1.0e-9),
        "per_class": per_class,
        "confusion_matrix": confusion.tolist(),
    }
    print(
        f"[{split}] loss={metrics['loss']:.6f} "
        f"acc={metrics['accuracy']:.4%} top5={metrics['top5_accuracy']:.4%} "
        f"samples={count} images_per_sec={metrics['images_per_sec']:.2f}",
        flush=True,
    )
    return metrics


def _checkpoint_metadata(
    replacement: RouterAblationReplacement,
    bundle: CIFAR10DataBundle,
    settings: Any,
) -> dict[str, Any]:
    return {
        "task": "cifar10_classification",
        "num_classes": int(settings.num_classes),
        "class_names": list(settings.class_names),
        "detector_dim": int(settings.detector_output_size),
        "model_id": settings.model_id,
        "split_seed": int(settings.split_seed),
        "split_sha256": bundle.metadata["split_sha256"],
        "optical_architecture": (
            f"{replacement.checkpoint_architecture}_cifar10_cls10_v1"
        ),
        "router_backend": settings.router_backend,
        "top_k": int(settings.top_k),
        "router_contract_sha256": settings.router_contract_sha256,
        "selection_criterion": "maximum_validation_accuracy_then_minimum_val_loss",
        "test_metrics_used_for_selection": False,
        "initialization": settings.classification_initialization,
    }


def save_checkpoint(
    path: Path,
    replacement: RouterAblationReplacement,
    head: CIFAR10ClassificationHead,
    optimizer: torch.optim.Optimizer,
    *,
    epoch: int,
    global_step: int,
    train_metrics: dict[str, Any],
    validation_metrics: dict[str, Any],
    best_validation_accuracy: float,
    best_validation_loss: float,
    highest_train_accuracy: float,
    no_improvement_epochs: int,
    bundle: CIFAR10DataBundle,
    settings: Any,
) -> None:
    payload = {
        "checkpoint_version": 1,
        "task": "cifar10_classification",
        "epoch": int(epoch),
        "global_step": int(global_step),
        "train_loss": float(train_metrics["loss"]),
        "train_accuracy": float(train_metrics["accuracy"]),
        "val_loss": float(validation_metrics["loss"]),
        "val_accuracy": float(validation_metrics["accuracy"]),
        "best_val_accuracy": float(best_validation_accuracy),
        "best_val_loss": float(best_validation_loss),
        "highest_train_accuracy": float(highest_train_accuracy),
        "no_improvement_epochs": int(no_improvement_epochs),
        "vision_optical": replacement.vision_surrogate.state_dict(),
        "language_optical": replacement.language_surrogate.state_dict(),
        "classification_head": head.state_dict(),
        "optimizer": optimizer.state_dict(),
        "metadata": _checkpoint_metadata(replacement, bundle, settings),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def load_checkpoint(
    path: Path,
    replacement: RouterAblationReplacement,
    head: CIFAR10ClassificationHead,
    bundle: CIFAR10DataBundle,
    settings: Any,
    *,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Classification checkpoint is missing: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("task") != "cifar10_classification":
        raise RuntimeError("Checkpoint is not a CIFAR-10 classification checkpoint")
    metadata = payload.get("metadata", {})
    expected = _checkpoint_metadata(replacement, bundle, settings)
    for key in (
        "task",
        "num_classes",
        "class_names",
        "detector_dim",
        "model_id",
        "split_sha256",
        "optical_architecture",
        "router_backend",
        "top_k",
        "router_contract_sha256",
    ):
        if metadata.get(key) != expected[key]:
            raise RuntimeError(
                f"Classification checkpoint metadata mismatch for {key}: "
                f"saved={metadata.get(key)!r} expected={expected[key]!r}"
            )
    replacement.vision_surrogate.load_state_dict(payload["vision_optical"], strict=True)
    replacement.language_surrogate.load_state_dict(
        payload["language_optical"], strict=True
    )
    head.load_state_dict(payload["classification_head"], strict=True)
    if optimizer is not None:
        optimizer.load_state_dict(payload["optimizer"])
    return payload


def _append_jsonl(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False) + "\n")


def gradient_check(
    loaded: LoadedBackbone,
    replacement: RouterAblationReplacement,
    head: CIFAR10ClassificationHead,
    bundle: CIFAR10DataBundle,
    settings: Any,
    *,
    batch_size: int = 2,
) -> dict[str, Any]:
    loaded.model.eval()
    replacement.set_student_train_mode()
    head.train()
    images: list[Any] = []
    labels_list: list[int] = []
    for index in range(batch_size):
        image, label = bundle.train[index]
        images.append(image)
        labels_list.append(int(label))
    labels = torch.tensor(labels_list, dtype=torch.long, device=loaded.device)
    inputs = _prepare(loaded, images, settings)
    for parameter in replacement.trainable_parameters():
        parameter.grad = None
    head.zero_grad(set_to_none=True)
    use_amp, amp_dtype = _amp(settings, loaded.device)
    if loaded.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(loaded.device)
    started = time.perf_counter()
    with torch.autocast(
        device_type=loaded.device.type, dtype=amp_dtype, enabled=use_amp
    ):
        logits, features = classification_logits(
            loaded.model, replacement, head, inputs
        )
        cross_entropy = F.cross_entropy(logits, labels)
        auxiliary, components = _auxiliary_loss(
            replacement, settings, logits, epoch=1
        )
        total = cross_entropy + auxiliary
    total.backward()
    if loaded.device.type == "cuda":
        torch.cuda.synchronize(loaded.device)
    elapsed = time.perf_counter() - started

    def group_has_finite_nonzero(group: Iterable[nn.Parameter]) -> bool:
        gradients = [
            parameter.grad
            for parameter in group
            if parameter.grad is not None
        ]
        return bool(gradients) and all(torch.isfinite(value).all() for value in gradients) and any(
            bool(torch.count_nonzero(value)) for value in gradients
        )

    checks = {
        "classification_head": group_has_finite_nonzero(head.parameters()),
        "feature_phases": group_has_finite_nonzero(
            parameter
            for group in replacement.phase_parameter_groups().values()
            for parameter in group
        ),
        "routers": group_has_finite_nonzero(replacement.router_parameters()),
    }
    if not all(checks.values()):
        raise RuntimeError(f"Gradient check failed: {checks}")
    report = {
        "batch_size": batch_size,
        "detector_shape": list(features.shape),
        "logits_shape": list(logits.shape),
        "cross_entropy": float(cross_entropy.detach()),
        "total_loss": float(total.detach()),
        "elapsed_sec": elapsed,
        "images_per_sec": batch_size / max(elapsed, 1.0e-9),
        "gradient_checks": checks,
        "router": _router_snapshot(replacement),
        "auxiliary": {
            name: float(value.detach()) for name, value in components.items()
        },
        "gpu_max_memory_allocated_gib": (
            torch.cuda.max_memory_allocated(loaded.device) / 2**30
            if loaded.device.type == "cuda"
            else 0.0
        ),
        "gpu_max_memory_reserved_gib": (
            torch.cuda.max_memory_reserved(loaded.device) / 2**30
            if loaded.device.type == "cuda"
            else 0.0
        ),
    }
    print(f"[check] {json.dumps(report, ensure_ascii=False)}", flush=True)
    return report


def train(
    loaded: LoadedBackbone,
    replacement: RouterAblationReplacement,
    head: CIFAR10ClassificationHead,
    bundle: CIFAR10DataBundle,
    settings: Any,
    *,
    resume_checkpoint: Path | None = None,
) -> dict[str, Any]:
    optimizer, trainable_parameters = build_optimizer(replacement, head, settings)
    start_epoch = 1
    global_step = 0
    best_accuracy = -1.0
    best_loss = float("inf")
    highest_train_accuracy = 0.0
    no_improvement = 0
    if resume_checkpoint is not None:
        payload = load_checkpoint(
            resume_checkpoint,
            replacement,
            head,
            bundle,
            settings,
            optimizer=optimizer,
        )
        start_epoch = int(payload["epoch"]) + 1
        global_step = int(payload["global_step"])
        best_accuracy = float(payload["best_val_accuracy"])
        best_loss = float(payload.get("best_val_loss", payload["val_loss"]))
        highest_train_accuracy = float(
            payload.get("highest_train_accuracy", payload.get("train_accuracy", 0.0))
        )
        no_improvement = int(payload.get("no_improvement_epochs", 0))

    steps_per_epoch = settings.max_train_steps_per_epoch
    if steps_per_epoch is None:
        steps_per_epoch = math.ceil(len(bundle.train) / settings.batch_size)
    total_steps = int(settings.epochs) * int(steps_per_epoch)
    use_amp, amp_dtype = _amp(settings, loaded.device)
    metrics_path = settings.output_dir / "metrics.jsonl"
    best_path = settings.output_dir / "best_validation_accuracy_checkpoint.pt"
    latest_path = settings.output_dir / "latest_checkpoint.pt"
    history: list[dict[str, Any]] = []

    for epoch in range(start_epoch, int(settings.epochs) + 1):
        loader = build_train_loader(bundle, settings, epoch=epoch)
        loaded.model.eval()
        replacement.set_student_train_mode()
        head.train()
        totals = {
            "loss": 0.0,
            "cross_entropy": 0.0,
            "correct": 0,
            "samples": 0,
        }
        epoch_started = time.perf_counter()
        for step_in_epoch, (inputs, labels_cpu) in enumerate(
            _prepared_batches(loader, loaded, settings), start=1
        ):
            labels = labels_cpu.to(loaded.device, non_blocking=True)
            scale = _learning_rate_scale(
                global_step,
                total_steps,
                float(getattr(settings, "learning_rate_warmup_ratio", 0.05)),
            )
            _apply_lr_scale(optimizer, scale)
            optimizer.zero_grad(set_to_none=True)
            step_started = time.perf_counter()
            with torch.autocast(
                device_type=loaded.device.type, dtype=amp_dtype, enabled=use_amp
            ):
                logits, _ = classification_logits(
                    loaded.model, replacement, head, inputs
                )
                cross_entropy = F.cross_entropy(
                    logits,
                    labels,
                    label_smoothing=float(settings.label_smoothing),
                )
                auxiliary, components = _auxiliary_loss(
                    replacement, settings, logits, epoch
                )
                total = cross_entropy + auxiliary
            if not torch.isfinite(total):
                raise RuntimeError(
                    f"Non-finite classification loss at epoch={epoch} "
                    f"step={step_in_epoch}: {total}"
                )
            total.backward()
            if settings.gradient_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    trainable_parameters, float(settings.gradient_clip_norm)
                )
            optimizer.step()
            if loaded.device.type == "cuda":
                torch.cuda.synchronize(loaded.device)
            step_elapsed = time.perf_counter() - step_started
            batch_count = labels.numel()
            batch_correct = int(logits.detach().argmax(dim=1).eq(labels).sum())
            totals["loss"] += float(total.detach()) * batch_count
            totals["cross_entropy"] += float(cross_entropy.detach()) * batch_count
            totals["correct"] += batch_correct
            totals["samples"] += batch_count
            global_step += 1
            if step_in_epoch % settings.log_every_steps == 0:
                router = _router_snapshot(replacement)
                memory = (
                    torch.cuda.max_memory_allocated(loaded.device) / 2**30
                    if loaded.device.type == "cuda"
                    else 0.0
                )
                print(
                    f"[train] epoch={epoch}/{settings.epochs} "
                    f"step={step_in_epoch}/{len(loader)} global_step={global_step} "
                    f"loss={float(total.detach()):.6f} "
                    f"ce={float(cross_entropy.detach()):.6f} "
                    f"acc={batch_correct/max(1,batch_count):.4%} "
                    f"img_s={batch_count/max(step_elapsed,1e-9):.2f} "
                    f"lr_scale={scale:.4f} gpu_gib={memory:.2f} "
                    f"v_active={router['vision_active_experts']:.0f} "
                    f"l_active={router['language_active_experts']:.0f}",
                    flush=True,
                )

        epoch_elapsed = time.perf_counter() - epoch_started
        train_metrics = {
            "split": "train_online",
            "loss": totals["loss"] / totals["samples"],
            "cross_entropy": totals["cross_entropy"] / totals["samples"],
            "accuracy": totals["correct"] / totals["samples"],
            "samples": totals["samples"],
            "elapsed_sec": epoch_elapsed,
            "images_per_sec": totals["samples"] / max(epoch_elapsed, 1.0e-9),
        }
        highest_train_accuracy = max(highest_train_accuracy, train_metrics["accuracy"])
        should_validate = (
            epoch % settings.eval_every_epochs == 0 or epoch == settings.epochs
        )
        if not should_validate:
            continue
        validation_metrics = evaluate(
            loaded,
            replacement,
            head,
            bundle.validation,
            settings,
            split="validation",
        )
        improved = validation_metrics["accuracy"] > best_accuracy or (
            validation_metrics["accuracy"] == best_accuracy
            and validation_metrics["loss"] < best_loss
        )
        if improved:
            best_accuracy = float(validation_metrics["accuracy"])
            best_loss = float(validation_metrics["loss"])
            no_improvement = 0
        else:
            no_improvement += 1
        record = {
            "epoch": epoch,
            "global_step": global_step,
            "train": train_metrics,
            "validation": validation_metrics,
            "overfit_gap": train_metrics["accuracy"]
            - validation_metrics["accuracy"],
            "best_validation_accuracy": best_accuracy,
            "improved": improved,
        }
        history.append(record)
        _append_jsonl(metrics_path, record)
        save_checkpoint(
            latest_path,
            replacement,
            head,
            optimizer,
            epoch=epoch,
            global_step=global_step,
            train_metrics=train_metrics,
            validation_metrics=validation_metrics,
            best_validation_accuracy=best_accuracy,
            best_validation_loss=best_loss,
            highest_train_accuracy=highest_train_accuracy,
            no_improvement_epochs=no_improvement,
            bundle=bundle,
            settings=settings,
        )
        if improved:
            save_checkpoint(
                best_path,
                replacement,
                head,
                optimizer,
                epoch=epoch,
                global_step=global_step,
                train_metrics=train_metrics,
                validation_metrics=validation_metrics,
                best_validation_accuracy=best_accuracy,
                best_validation_loss=best_loss,
                highest_train_accuracy=highest_train_accuracy,
                no_improvement_epochs=no_improvement,
                bundle=bundle,
                settings=settings,
            )
        print(
            f"[epoch] epoch={epoch} train_loss={train_metrics['loss']:.6f} "
            f"train_acc={train_metrics['accuracy']:.4%} "
            f"val_loss={validation_metrics['loss']:.6f} "
            f"val_acc={validation_metrics['accuracy']:.4%} "
            f"gap={record['overfit_gap']:.4%} best_val={best_accuracy:.4%}",
            flush=True,
        )
        if settings.early_stopping_patience > 0 and (
            no_improvement >= settings.early_stopping_patience
        ):
            print(
                f"[early_stop] patience={settings.early_stopping_patience} "
                f"epoch={epoch}",
                flush=True,
            )
            break

    if not best_path.is_file():
        raise RuntimeError("Training completed without a validation checkpoint")
    best_payload = load_checkpoint(
        best_path, replacement, head, bundle, settings, optimizer=None
    )
    test_metrics = None
    if settings.run_final_test:
        print(
            "[sealed_test] evaluating the official test split once with the "
            "best-validation checkpoint",
            flush=True,
        )
        test_metrics = evaluate(
            loaded,
            replacement,
            head,
            bundle.test,
            settings,
            split="test",
        )
    final = {
        "task": "cifar10_classification",
        "router_backend": settings.router_backend,
        "top_k": settings.top_k,
        "highest_online_train_accuracy": highest_train_accuracy,
        "best_validation_accuracy": float(best_payload["val_accuracy"]),
        "best_validation_loss": float(best_payload["val_loss"]),
        "best_epoch": int(best_payload["epoch"]),
        "test": test_metrics,
        "best_checkpoint": str(best_path),
        "latest_checkpoint": str(latest_path),
    }
    output = settings.output_dir / "final_metrics.json"
    output.write_text(
        json.dumps(final, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"[final] {json.dumps(final, ensure_ascii=False)}", flush=True)
    return final


__all__ = [
    "build_optimizer",
    "evaluate",
    "gradient_check",
    "load_checkpoint",
    "save_checkpoint",
    "train",
]
