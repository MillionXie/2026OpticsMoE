from __future__ import annotations

import argparse
import json
import random
import shutil
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import torch

from .cache_teacher_embeddings import TeacherEmbeddingStore
from .features import student_embeddings
from .hardware_pipeline import (
    STAGES,
    Runtime,
    _clear_replay,
    _forward,
    _input_for_sample,
    _intensity_active_to_canvas,
    _manifest_rows,
    _replay_vision,
    _samples_by_id,
    _save_phase_mask,
    build_runtime,
    close_runtime,
    has_captured_intensity,
    load_captured_intensity,
    load_hardware_config,
    process_language_expert,
    process_language_global,
    process_vision_expert,
    process_vision_global,
)
from .io_utils import seed_everything, write_csv, write_json
from .optical_artifacts import phase_tensors
from .train_optical_retrieval import (
    embedding_distillation_loss,
    load_checkpoint,
    save_checkpoint,
    supervised_contrastive_loss,
)


CAPTURE_STAGES = tuple(STAGES)
STAGE_NUMBERS = {name: index + 1 for index, name in enumerate(CAPTURE_STAGES)}


@dataclass(frozen=True)
class AdaptationConfig:
    epochs: int
    learning_rate: float
    weight_decay: float
    skus_per_batch: int
    samples_per_sku: int
    lambda_kd: float
    lambda_retrieval: float
    lambda_prototype: float
    temperature: float
    prototype_temperature: float
    gradient_clip_norm: float
    run_tag: str | None


def load_adaptation_config(path: Path) -> AdaptationConfig:
    import yaml

    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    values = raw.get("adaptation", {})
    result = AdaptationConfig(
        epochs=int(values.get("epochs", 20)),
        learning_rate=float(values.get("learning_rate", 2.0e-4)),
        weight_decay=float(values.get("weight_decay", 0.0)),
        skus_per_batch=int(values.get("skus_per_batch", 2)),
        samples_per_sku=int(values.get("samples_per_sku", 2)),
        lambda_kd=float(values.get("lambda_kd", 1.0)),
        lambda_retrieval=float(values.get("lambda_retrieval", 1.0)),
        lambda_prototype=float(values.get("lambda_prototype", 0.0)),
        temperature=float(values.get("temperature", 0.07)),
        prototype_temperature=float(values.get("prototype_temperature", 0.07)),
        gradient_clip_norm=float(values.get("gradient_clip_norm", 1.0)),
        run_tag=(str(values["run_tag"]).strip() if values.get("run_tag") else None),
    )
    if result.epochs <= 0 or result.learning_rate <= 0:
        raise ValueError("adaptation epochs and learning_rate must be positive")
    if result.skus_per_batch < 2 or result.samples_per_sku < 2:
        raise ValueError(
            "Hardware SupCon batches need at least two SKUs and two samples per SKU"
        )
    if (
        result.temperature <= 0
        or result.prototype_temperature <= 0
        or result.gradient_clip_norm <= 0
    ):
        raise ValueError("adaptation temperatures and gradient_clip_norm must be positive")
    if result.lambda_kd < 0 or result.lambda_retrieval < 0 or result.lambda_prototype < 0:
        raise ValueError("adaptation loss weights must be nonnegative")
    return result


def _enable(module: torch.nn.Module, values: list[torch.nn.Parameter]) -> None:
    module.requires_grad_(True)
    values.extend(parameter for parameter in module.parameters() if parameter.requires_grad)


def configure_downstream_trainability(
    runtime: Runtime, capture_stage: str
) -> tuple[list[torch.nn.Parameter], list[dict[str, Any]]]:
    """Enable only modules physically downstream of the measured CCD plane."""
    if capture_stage not in CAPTURE_STAGES:
        raise ValueError(f"Unknown capture stage {capture_stage!r}")
    runtime.loaded.model.requires_grad_(False)
    runtime.replacement.vision_surrogate.requires_grad_(False)
    runtime.replacement.language_surrogate.requires_grad_(False)
    runtime.readout.requires_grad_(False)
    vision = runtime.replacement.vision_surrogate.core
    language = runtime.replacement.language_surrogate.core
    parameters: list[torch.nn.Parameter] = []
    modules: list[tuple[str, torch.nn.Module]] = []

    if capture_stage == "vision_expert":
        modules.extend(
            [
                ("vision.interlayer_electronics", vision.interlayer_conversions[0]),
                ("vision.global_phase", vision.global_phase),
                ("vision.final_detector_readout", vision.readout),
                ("vision.output_adapter", vision.output_adapter),
                ("language.input_adapter", language.input_adapter),
                ("language.input_norm", language.input_norm),
                ("language.router", language.router),
                ("language.expert_phase", language.expert_layers[0]),
                ("language.interlayer_electronics", language.interlayer_conversions[0]),
                ("language.global_phase", language.global_phase),
                ("language.final_detector_readout", language.readout),
            ]
        )
    elif capture_stage == "vision_global":
        modules.extend(
            [
                ("vision.final_detector_readout", vision.readout),
                ("vision.output_adapter", vision.output_adapter),
                ("language.input_adapter", language.input_adapter),
                ("language.input_norm", language.input_norm),
                ("language.router", language.router),
                ("language.expert_phase", language.expert_layers[0]),
                ("language.interlayer_electronics", language.interlayer_conversions[0]),
                ("language.global_phase", language.global_phase),
                ("language.final_detector_readout", language.readout),
            ]
        )
    elif capture_stage == "language_expert":
        modules.extend(
            [
                ("language.interlayer_electronics", language.interlayer_conversions[0]),
                ("language.global_phase", language.global_phase),
                ("language.final_detector_readout", language.readout),
            ]
        )
    else:
        modules.append(("language.final_detector_readout", language.readout))
    modules.append(("retrieval_readout", runtime.readout))

    report: list[dict[str, Any]] = []
    seen: set[int] = set()
    unique: list[torch.nn.Parameter] = []
    for prefix, module in modules:
        local: list[torch.nn.Parameter] = []
        _enable(module, local)
        for name, parameter in module.named_parameters():
            if id(parameter) in seen:
                continue
            seen.add(id(parameter))
            unique.append(parameter)
            report.append(
                {
                    "name": f"{prefix}.{name}",
                    "shape": list(parameter.shape),
                    "parameters": int(parameter.numel()),
                }
            )
    return unique, report


def _set_optional_measured_vision(runtime: Runtime, key: str, use_simulation: bool) -> None:
    simulation = runtime.hardware.output_dir / STAGES["vision_global"] / "simulation_reference" / "ccd_intensity" / f"{key}.pt"
    if use_simulation or has_captured_intensity(runtime, "vision_global", key):
        if use_simulation and not simulation.is_file():
            return
        _replay_vision(runtime, key, use_simulation)


def _prepare_training_forward(
    runtime: Runtime,
    capture_stage: str,
    key: str,
    sample: Any,
    *,
    use_simulation: bool,
) -> dict[str, torch.Tensor]:
    """Install a constant measured boundary while preserving downstream autograd."""
    _clear_replay(runtime)
    vision = runtime.replacement.vision_surrogate.core
    language = runtime.replacement.language_surrogate.core

    if capture_stage == "vision_expert":
        _, inputs, _, _ = _forward(runtime, sample)
        measured = load_captured_intensity(
            runtime, capture_stage, key, use_simulation=use_simulation
        ).to(runtime.loaded.device)
        full = _intensity_active_to_canvas(vision, measured).unsqueeze(0)
        reload_field = vision.interlayer_conversions[0].forward_intensity(
            full,
            selected_experts=vision.last_routing["selected_mask"],
            routing_weights=vision.last_routing["weights"],
        )
        vision.set_hardware_replay(stage_reload_fields={0: reload_field})
        return inputs

    if capture_stage == "vision_global":
        measured = load_captured_intensity(
            runtime, capture_stage, key, use_simulation=use_simulation
        ).to(runtime.loaded.device)
        vision.set_hardware_replay(final_detector_intensity=measured.unsqueeze(0))
        inputs, _, _ = _input_for_sample(runtime, sample)
        return inputs

    _set_optional_measured_vision(runtime, key, use_simulation)
    if capture_stage == "language_expert":
        _, inputs, _, _ = _forward(runtime, sample)
        measured = load_captured_intensity(
            runtime, capture_stage, key, use_simulation=use_simulation
        ).to(runtime.loaded.device)
        full = _intensity_active_to_canvas(language, measured).unsqueeze(0)
        reload_field = language.interlayer_conversions[0].forward_intensity(
            full,
            selected_experts=language.last_routing["selected_mask"],
            routing_weights=language.last_routing["weights"],
        )
        language.set_hardware_replay(stage_reload_fields={0: reload_field})
        return inputs

    measured = load_captured_intensity(
        runtime, capture_stage, key, use_simulation=use_simulation
    ).to(runtime.loaded.device)
    language.set_hardware_replay(final_detector_intensity=measured.unsqueeze(0))
    inputs, _, _ = _input_for_sample(runtime, sample)
    return inputs


def _epoch_batches(
    rows: list[dict[str, str]],
    samples: dict[str, Any],
    config: AdaptationConfig,
    epoch: int,
) -> list[list[dict[str, str]]]:
    grouped: dict[int, dict[str, list[dict[str, str]]]] = {}
    for row in rows:
        role = str(row.get("role", "query"))
        grouped.setdefault(
            samples[row["sample_id"]].sku_index, {"gallery": [], "query": []}
        ).setdefault(role, []).append(row)
    invalid = {
        sku: {role: len(values) for role, values in roles.items()}
        for sku, roles in grouped.items()
        if not roles["gallery"] or not roles["query"]
    }
    if invalid:
        raise RuntimeError(
            "Hardware prototype adaptation needs gallery and query views for every SKU; "
            f"invalid session groups={invalid}"
        )
    rng = random.Random(42 + epoch)
    sku_ids = sorted(grouped)
    rng.shuffle(sku_ids)
    batches: list[list[dict[str, str]]] = []
    for start in range(0, len(sku_ids), config.skus_per_batch):
        selected = sku_ids[start : start + config.skus_per_batch]
        if len(selected) < 2:
            selected = selected + sku_ids[: 2 - len(selected)]
        shuffled_queries: dict[int, list[dict[str, str]]] = {}
        for sku in selected:
            shuffled_queries[sku] = list(grouped[sku]["query"])
            rng.shuffle(shuffled_queries[sku])
        rounds = max(
            (len(shuffled_queries[sku]) + config.samples_per_sku - 1)
            // config.samples_per_sku
            for sku in selected
        )
        for round_index in range(rounds):
            # Every batch contains the registered gallery for every selected SKU.
            # Query slices are disjoint, so one epoch covers every measured query.
            batch = [row for sku in selected for row in grouped[sku]["gallery"]]
            begin = round_index * config.samples_per_sku
            end = begin + config.samples_per_sku
            batch.extend(
                row
                for sku in selected
                for row in shuffled_queries[sku][begin:end]
            )
            if len(batch) > len(selected):
                batches.append(batch)
    return batches


def prototype_retrieval_objective(
    embeddings: torch.Tensor,
    rows: list[dict[str, str]],
    samples: list[Any],
    temperature: float,
) -> tuple[torch.Tensor, dict[str, float | int]]:
    """Classify measured queries against measured gallery prototypes.

    This is not an electronic class head: prototypes are computed from the
    normalized gallery embeddings and inference remains cosine retrieval.
    """
    if embeddings.ndim != 2 or len(embeddings) != len(rows) or len(rows) != len(samples):
        raise ValueError("Prototype retrieval inputs must have matching leading dimensions")
    gallery_by_sku: dict[int, list[torch.Tensor]] = {}
    query_indexes: list[int] = []
    for index, (row, sample) in enumerate(zip(rows, samples)):
        if row.get("role") == "gallery":
            gallery_by_sku.setdefault(int(sample.sku_index), []).append(embeddings[index])
        elif row.get("role") == "query":
            query_indexes.append(index)
    if not gallery_by_sku or not query_indexes:
        raise RuntimeError("Prototype retrieval requires both gallery and query samples")
    sku_order = sorted(gallery_by_sku)
    sku_to_column = {sku: index for index, sku in enumerate(sku_order)}
    prototypes = torch.stack(
        [
            torch.nn.functional.normalize(
                torch.stack(gallery_by_sku[sku]).mean(dim=0), dim=0
            )
            for sku in sku_order
        ]
    )
    queries = embeddings[query_indexes]
    targets = torch.tensor(
        [sku_to_column[int(samples[index].sku_index)] for index in query_indexes],
        device=embeddings.device,
        dtype=torch.long,
    )
    logits = queries @ prototypes.T / float(temperature)
    loss = torch.nn.functional.cross_entropy(logits.float(), targets)
    order = logits.argsort(dim=1, descending=True)
    positions = (order == targets[:, None]).nonzero(as_tuple=False)[:, 1]
    count = int(len(targets))
    metrics: dict[str, float | int] = {
        "query_count": count,
        "top1": float((positions < 1).float().mean()),
        "top3": float((positions < min(3, len(sku_order))).float().mean()),
        "mrr": float((1.0 / (positions.float() + 1.0)).mean()),
    }
    return loss, metrics


def _build_final_ccd_cache(
    runtime: Runtime,
    rows: list[dict[str, str]],
    samples: dict[str, Any],
    *,
    use_simulation: bool,
) -> dict[str, tuple[torch.Tensor, int]]:
    """Cache parameter-free CCD pooling and token lengths for fast adaptation."""
    result: dict[str, tuple[torch.Tensor, int]] = {}
    readout = runtime.replacement.language_surrogate.core.readout
    for index, row in enumerate(rows, start=1):
        sample = samples[row["sample_id"]]
        inputs, _, _ = _input_for_sample(runtime, sample)
        length = int(inputs["attention_mask"].long().sum().item())
        measured = load_captured_intensity(
            runtime, "language_global", row["sample_key"], use_simulation=use_simulation
        ).float()
        pooled = readout.pool(measured.unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)
        if not 0 < length <= pooled.shape[0]:
            raise RuntimeError(
                f"Language length {length} is invalid for CCD rows={pooled.shape[0]}"
            )
        result[row["sample_key"]] = (pooled.contiguous(), length)
        if index % 25 == 0 or index == len(rows):
            print(f"[hardware_finetune:cache] {index}/{len(rows)} final CCD frames", flush=True)
    return result


def _embedding_from_final_ccd_cache(
    runtime: Runtime,
    row: dict[str, str],
    cache: dict[str, tuple[torch.Tensor, int]],
) -> torch.Tensor:
    pooled, length = cache[row["sample_key"]]
    detector = runtime.replacement.language_surrogate.core.readout
    normalized = detector.norm(pooled.to(runtime.loaded.device))
    readout = (
        torch.nn.functional.relu(normalized)
        if detector.nonlinearity == "relu"
        else torch.nn.functional.softplus(normalized)
    )
    return runtime.readout(readout[length - 1].unsqueeze(0))[0]


def _export_downstream_masks(runtime: Runtime, capture_stage: str, root: Path) -> list[str]:
    vision = phase_tensors(runtime.replacement.vision_surrogate.core)
    language = phase_tensors(runtime.replacement.language_surrogate.core)
    definitions = {
        "vision_global": ("vision", "global", vision["physical_global_phase_rad"]),
        "language_expert": ("language", "expert", language["physical_expert_mosaic_rad"]),
        "language_global": ("language", "global", language["physical_global_phase_rad"]),
    }
    downstream = {
        "vision_expert": ["vision_global", "language_expert", "language_global"],
        "vision_global": ["language_expert", "language_global"],
        "language_expert": ["language_global"],
        "language_global": [],
    }[capture_stage]
    exported: list[str] = []
    for stage in downstream:
        stack, plane, tensor = definitions[stage]
        _save_phase_mask(runtime, stack, plane, tensor, stage)
        source = runtime.hardware.output_dir / "00_masks" / STAGES[stage]
        target = root / "exported_downstream_masks" / STAGES[stage]
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(source, target)
        exported.extend(str(path) for path in sorted(target.glob("*.bmp")))
    return exported


def _regenerate_next_stage(
    runtime: Runtime, capture_stage: str, use_simulation: bool, root: Path
) -> dict[str, Any]:
    next_stage = {
        "vision_expert": "vision_global",
        "vision_global": "language_expert",
        "language_expert": "language_global",
        "language_global": None,
    }[capture_stage]
    if capture_stage == "vision_expert":
        process_vision_expert(runtime, use_simulation=use_simulation)
    elif capture_stage == "vision_global":
        process_vision_global(runtime, use_simulation=use_simulation)
    elif capture_stage == "language_expert":
        process_language_expert(runtime, use_simulation=use_simulation)
    else:
        metrics = process_language_global(runtime, use_simulation=use_simulation)
        return {"next_stage": None, "final_metrics": metrics}
    source = runtime.hardware.output_dir / STAGES[next_stage] / "amplitude_to_play"
    target = root / "next_amplitude_bmp" / STAGES[next_stage]
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(source, target)
    return {
        "next_stage": next_stage,
        "canonical_amplitude_dir": str(source),
        "snapshot_amplitude_dir": str(target),
    }


def _adaptation_root(
    runtime: Runtime, capture_stage: str, run_tag: str | None = None
) -> Path:
    vertical = runtime.hardware.capture_flip_vertical
    horizontal = runtime.hardware.capture_flip_horizontal
    if vertical and horizontal:
        transform_tag = "ccd_vhflip"
    elif vertical:
        transform_tag = "ccd_vflip"
    elif horizontal:
        transform_tag = "ccd_hflip"
    else:
        transform_tag = "ccd_native"
    name = (
        f"after_{STAGE_NUMBERS[capture_stage]:02d}_{capture_stage}"
        f"__{transform_tag}"
    )
    if run_tag:
        safe_tag = "".join(character if character.isalnum() or character in "-_" else "_" for character in run_tag)
        name += f"__{safe_tag}"
    return (
        runtime.hardware.output_dir
        / "06_hardware_finetune"
        / name
    )


def finalize_existing_adaptation(
    runtime: Runtime,
    capture_stage: str,
    *,
    use_simulation: bool,
    run_tag: str | None = None,
) -> dict[str, Any]:
    """Finish export/evaluation after training succeeded but finalization failed."""
    root = _adaptation_root(runtime, capture_stage, run_tag)
    best_path = root / "checkpoints" / "best_train_loss.pt"
    if not best_path.is_file():
        raise FileNotFoundError(
            f"Completed hardware-adaptation checkpoint is missing: {best_path}"
        )
    payload = load_checkpoint(
        best_path, runtime.replacement, runtime.readout
    )
    runtime.hardware = replace(runtime.hardware, checkpoint=best_path)
    exported_masks = _export_downstream_masks(runtime, capture_stage, root)
    regenerated = _regenerate_next_stage(
        runtime, capture_stage, use_simulation, root
    )
    summary = {
        "capture_stage": capture_stage,
        "finalize_only": True,
        "best_checkpoint": str(best_path),
        "best_checkpoint_epoch": int(payload.get("epoch", -1)),
        "best_training_loss": float(payload.get("train_loss", float("nan"))),
        "exported_downstream_phase_bmps": exported_masks,
        "regenerated": regenerated,
    }
    write_json(root / "metrics" / "summary.json", summary)
    return summary


def adapt_after_capture(
    runtime: Runtime,
    capture_stage: str,
    config: AdaptationConfig,
    *,
    use_simulation: bool,
) -> dict[str, Any]:
    source_checkpoint = runtime.hardware.checkpoint
    root = _adaptation_root(runtime, capture_stage, config.run_tag)
    root.mkdir(parents=True, exist_ok=True)
    parameters, parameter_report = configure_downstream_trainability(runtime, capture_stage)
    if not parameters:
        raise RuntimeError("No downstream trainable parameters were selected")
    write_json(
        root / "trainable_parameters.json",
        {
            "capture_stage": capture_stage,
            "causal_rule": "Only modules downstream of the measured CCD are trainable",
            "total_trainable_parameters": sum(value.numel() for value in parameters),
            "parameters": parameter_report,
        },
    )
    optimizer = torch.optim.AdamW(
        parameters, lr=config.learning_rate, weight_decay=config.weight_decay
    )
    teacher_store = TeacherEmbeddingStore(
        runtime.settings.teacher_cache_path, runtime.bundle, runtime.settings
    )
    by_id = _samples_by_id(runtime.bundle)
    rows = _manifest_rows(runtime)
    final_ccd_cache = (
        _build_final_ccd_cache(
            runtime, rows, by_id, use_simulation=use_simulation
        )
        if capture_stage == "language_global"
        else None
    )
    best_loss = float("inf")
    best_train_top1 = -1.0
    history: list[dict[str, Any]] = []
    best_path = root / "checkpoints" / "best_train_loss.pt"
    last_path = root / "checkpoints" / "last.pt"

    runtime.loaded.model.eval()
    runtime.replacement.set_phase_dropout_active(False)
    for epoch in range(1, config.epochs + 1):
        batches = _epoch_batches(rows, by_id, config, epoch)
        total_sum = kd_sum = retrieval_sum = prototype_sum = 0.0
        for batch_rows in batches:
            optimizer.zero_grad(set_to_none=True)
            embeddings: list[torch.Tensor] = []
            batch_samples = [by_id[row["sample_id"]] for row in batch_rows]
            for row, sample in zip(batch_rows, batch_samples):
                if final_ccd_cache is not None:
                    embeddings.append(
                        _embedding_from_final_ccd_cache(runtime, row, final_ccd_cache)
                    )
                else:
                    inputs = _prepare_training_forward(
                        runtime,
                        capture_stage,
                        row["sample_key"],
                        sample,
                        use_simulation=use_simulation,
                    )
                    embedding, _ = student_embeddings(
                        runtime.loaded.model,
                        runtime.replacement,
                        runtime.readout,
                        inputs,
                    )
                    embeddings.append(embedding[0])
                    _clear_replay(runtime)
            student = torch.stack(embeddings)
            teacher = teacher_store.lookup(batch_samples).to(student.device)
            labels = torch.tensor(
                [sample.sku_index for sample in batch_samples],
                device=student.device,
                dtype=torch.long,
            )
            kd = embedding_distillation_loss(student, teacher)
            retrieval = supervised_contrastive_loss(
                student, labels, config.temperature
            )
            prototype, _ = prototype_retrieval_objective(
                student, batch_rows, batch_samples, config.prototype_temperature
            )
            loss = (
                config.lambda_kd * kd
                + config.lambda_retrieval * retrieval
                + config.lambda_prototype * prototype
            )
            if not torch.isfinite(loss):
                raise RuntimeError("Hardware adaptation loss became NaN or Inf")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, config.gradient_clip_norm)
            optimizer.step()
            total_sum += float(loss.detach())
            kd_sum += float(kd.detach())
            retrieval_sum += float(retrieval.detach())
            prototype_sum += float(prototype.detach())
        count = max(1, len(batches))
        with torch.no_grad():
            evaluation_embeddings: list[torch.Tensor] = []
            for row in rows:
                sample = by_id[row["sample_id"]]
                if final_ccd_cache is not None:
                    evaluation_embeddings.append(
                        _embedding_from_final_ccd_cache(runtime, row, final_ccd_cache)
                    )
                else:
                    inputs = _prepare_training_forward(
                        runtime, capture_stage, row["sample_key"], sample,
                        use_simulation=use_simulation,
                    )
                    embedding, _ = student_embeddings(
                        runtime.loaded.model, runtime.replacement, runtime.readout, inputs
                    )
                    evaluation_embeddings.append(embedding[0])
                    _clear_replay(runtime)
            _, train_metrics = prototype_retrieval_objective(
                torch.stack(evaluation_embeddings),
                rows,
                [by_id[row["sample_id"]] for row in rows],
                config.prototype_temperature,
            )
        record = {
            "epoch": epoch,
            "total_loss": total_sum / count,
            "kd_loss": kd_sum / count,
            "retrieval_loss": retrieval_sum / count,
            "prototype_loss": prototype_sum / count,
            "train_top1": train_metrics["top1"],
            "train_top3": train_metrics["top3"],
            "train_mrr": train_metrics["mrr"],
            "train_query_count": train_metrics["query_count"],
            "capture_stage": capture_stage,
            "batches": len(batches),
        }
        history.append(record)
        print(
            f"[hardware_finetune:{capture_stage}] epoch={epoch}/{config.epochs} "
            f"loss={record['total_loss']:.6f} kd={record['kd_loss']:.6f} "
            f"retrieval={record['retrieval_loss']:.6f} "
            f"prototype={record['prototype_loss']:.6f} "
            f"train_top1={record['train_top1']:.4f}",
            flush=True,
        )
        save_checkpoint(
            last_path,
            runtime.replacement,
            runtime.readout,
            optimizer,
            epoch,
            record["total_loss"],
            runtime.settings,
        )
        accuracy_improved = float(record["train_top1"]) > best_train_top1 + 1.0e-12
        tied_with_lower_loss = (
            abs(float(record["train_top1"]) - best_train_top1) <= 1.0e-12
            and record["total_loss"] < best_loss
        )
        if accuracy_improved or tied_with_lower_loss:
            best_train_top1 = float(record["train_top1"])
            best_loss = record["total_loss"]
            shutil.copy2(last_path, best_path)
    write_csv(root / "metrics" / "history.csv", history, list(history[0]))
    load_checkpoint(best_path, runtime.replacement, runtime.readout)
    runtime.hardware = replace(runtime.hardware, checkpoint=best_path)
    exported_masks = _export_downstream_masks(runtime, capture_stage, root)
    regenerated = _regenerate_next_stage(runtime, capture_stage, use_simulation, root)
    summary = {
        "capture_stage": capture_stage,
        "source_checkpoint": str(source_checkpoint),
        "best_checkpoint": str(best_path),
        "best_training_loss": best_loss,
        "best_training_top1": best_train_top1,
        "checkpoint_selection": "highest measured-query vs measured-gallery train Top-1, then lower total loss",
        "trainable_parameters": sum(value.numel() for value in parameters),
        "exported_downstream_phase_bmps": exported_masks,
        "regenerated": regenerated,
        "physical_causality": (
            "The measured plane and all upstream masks are frozen. Only downstream "
            "optical/electronic parameters are optimized."
        ),
    }
    write_json(root / "metrics" / "summary.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Causal layer-by-layer fine-tuning from measured Grocery CCD frames"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--capture-stage", required=True, choices=CAPTURE_STAGES)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--use-simulation", action="store_true")
    parser.add_argument(
        "--finalize-only",
        action="store_true",
        help=(
            "Do not retrain; load this stage's saved best checkpoint and only "
            "export masks/regenerate the next input or evaluate the final CCD"
        ),
    )
    args = parser.parse_args()
    hardware = load_hardware_config(args.config)
    if hardware.minimal_artifacts:
        hardware = replace(hardware, minimal_artifacts=False)
    if args.checkpoint:
        hardware = replace(
            hardware, checkpoint=Path(args.checkpoint).expanduser().resolve()
        )
    if args.output_dir:
        hardware = replace(
            hardware, output_dir=Path(args.output_dir).expanduser().resolve()
        )
    seed_everything(42)
    runtime = build_runtime(hardware)
    try:
        adaptation_config = load_adaptation_config(hardware.config_path)
        if args.finalize_only:
            summary = finalize_existing_adaptation(
                runtime,
                args.capture_stage,
                use_simulation=args.use_simulation,
                run_tag=adaptation_config.run_tag,
            )
        else:
            summary = adapt_after_capture(
                runtime,
                args.capture_stage,
                adaptation_config,
                use_simulation=args.use_simulation,
            )
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    finally:
        close_runtime(runtime)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
