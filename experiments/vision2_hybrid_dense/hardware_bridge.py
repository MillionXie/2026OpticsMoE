from __future__ import annotations

import argparse
import csv
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from experiments.hardware_sdk.workflows.reconstruct_slm import (
    encode_active_amplitude_with_metadata,
    encode_active_phase,
    save_active_png,
)
from experiments.qwen3_vl_embedding_2b_caltech101_language2_optical_retrieval.hardware_bridge import (
    load_ccd,
)
from experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval.optical_artifacts import (
    phase_tensors,
)


STAGES = ("vision_expert", "vision_global")


@dataclass(frozen=True)
class SampleRef:
    split: str
    index: int
    key: str
    sample_id: str
    image_path: str


@dataclass
class TaskContext:
    task: str
    settings: Any
    loaded: Any
    model: torch.nn.Module
    datasets: dict[str, Any]
    collate: Callable[[list[dict[str, Any]]], dict[str, Any]]
    preprocess: Callable[..., dict[str, torch.Tensor]]
    task_loss: Callable[[torch.Tensor, dict[str, Any]], torch.Tensor]
    metric_factory: Callable[[], Any]
    metric_update: Callable[[Any, torch.Tensor, dict[str, Any], torch.Tensor], None]
    checkpoint_payload: dict[str, Any]


def _stage_dir(session: Path, stage: str) -> Path:
    return session / f"{STAGES.index(stage) + 1:02d}_{stage}"


def _safe_key(split: str, sample_id: str) -> str:
    digest = hashlib.sha1(sample_id.encode("utf-8")).hexdigest()[:16]
    return f"{split}__{digest}"


def _select(records: list[Any], limit: int, seed: int) -> list[Any]:
    if limit >= len(records):
        return records
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(records), generator=generator)[:limit].tolist()
    return [records[index] for index in sorted(indices)]


def _load_task(
    task: str,
    config: str,
    checkpoint: Path,
    train_limit: int | None,
    eval_limit: int | None,
) -> TaskContext:
    if task == "salicon":
        from experiments.qwen3_vl_embedding_2b_salicon_vision_optical_saliency.datasets import (
            SALICONSaliencyDataset,
            collate_salicon,
            prepare_salicon,
        )
        from experiments.qwen3_vl_embedding_2b_salicon_vision_optical_saliency.modeling import (
            build_student,
            load_vision_backbone,
            preprocess_vision,
        )
        from experiments.qwen3_vl_embedding_2b_salicon_vision_optical_saliency.objectives import (
            SaliencyAccumulator,
            saliency_loss,
        )
        from experiments.qwen3_vl_embedding_2b_salicon_vision_optical_saliency.settings import (
            load_settings,
        )

        settings = load_settings(config)
        bundle = prepare_salicon(settings, persist=True)
        train_records = list(bundle.train_records)
        eval_records = list(bundle.validation_records)
        dataset_type = SALICONSaliencyDataset
        collate = collate_salicon

        def loss(logits: torch.Tensor, batch: dict[str, Any]) -> torch.Tensor:
            value, _ = saliency_loss(
                logits,
                batch["density"].to(logits.device),
                batch["fixation"].to(logits.device),
                settings,
                teacher_logits=None,
            )
            return value

        def metric_update(
            accumulator: Any,
            logits: torch.Tensor,
            batch: dict[str, Any],
            _loss: torch.Tensor,
        ) -> None:
            accumulator.update(
                logits,
                batch["density"].to(logits.device),
                batch["fixation"].to(logits.device),
            )

        metric_factory = SaliencyAccumulator

        core_key, head_key = "optical_core", "head"
    elif task == "isic":
        from experiments.qwen3_vl_embedding_2b_coco_duts_vision_optical_moe16_pretrain.objectives import (
            segmentation_loss,
        )
        from experiments.qwen3_vl_embedding_2b_isic2016_skin_lesion_optical_segmentation.datasets import (
            ISICSegmentationDataset,
            collate_segmentation,
            prepare_isic2016,
        )
        from experiments.qwen3_vl_embedding_2b_isic2016_skin_lesion_optical_segmentation.modeling import (
            build_duts_model as build_student,
            load_vision_backbone,
            preprocess_vision,
        )
        from experiments.qwen3_vl_embedding_2b_isic2016_skin_lesion_optical_segmentation.metrics import (
            ISICSegmentationAccumulator,
        )
        from experiments.qwen3_vl_embedding_2b_isic2016_skin_lesion_optical_segmentation.settings import (
            load_settings,
        )

        settings = load_settings(config)
        bundle = prepare_isic2016(settings, persist=True)
        train_records = list(bundle.train_records)
        eval_records = list(bundle.test_records)
        dataset_type = ISICSegmentationDataset
        collate = collate_segmentation

        def loss(logits: torch.Tensor, batch: dict[str, Any]) -> torch.Tensor:
            value, _ = segmentation_loss(
                logits,
                batch["masks"].to(logits.device),
                bce_weight=settings.bce_weight,
                dice_weight=settings.dice_weight,
                soft_iou_weight=settings.soft_iou_weight,
                boundary_weight=settings.boundary_weight,
            )
            return value

        def metric_update(
            accumulator: Any,
            logits: torch.Tensor,
            batch: dict[str, Any],
            value: torch.Tensor,
        ) -> None:
            accumulator.update(
                logits,
                batch["masks"].to(logits.device),
                loss=value,
            )

        metric_factory = ISICSegmentationAccumulator

        core_key, head_key = "backbone.core_state_dict", "head_state_dict"
    elif task == "lsp":
        from experiments.qwen3_vl_embedding_2b_lsp_pose_optical_moe16.datasets import (
            LSPPoseDataset,
            pose_collate,
            prepare_lsp,
        )
        from experiments.qwen3_vl_embedding_2b_lsp_pose_optical_moe16.losses import (
            masked_coordinate_loss,
            masked_heatmap_mse,
        )
        from experiments.qwen3_vl_embedding_2b_lsp_pose_optical_moe16.modeling import (
            build_student,
            load_vision_backbone,
            preprocess_vision,
        )
        from experiments.qwen3_vl_embedding_2b_lsp_pose_optical_moe16.metrics import (
            PoseMetricAccumulator,
        )
        from experiments.qwen3_vl_embedding_2b_lsp_pose_optical_moe16.settings import (
            load_settings,
        )

        settings = load_settings(config)
        bundle = prepare_lsp(settings, persist=True)
        train_records = list(bundle.train)
        eval_records = list(bundle.test)
        dataset_type = LSPPoseDataset
        collate = pose_collate

        def loss(logits: torch.Tensor, batch: dict[str, Any]) -> torch.Tensor:
            visible = batch["visible"].to(logits.device)
            return (
                settings.heatmap_loss_weight
                * masked_heatmap_mse(
                    logits, batch["heatmaps"].to(logits.device), visible
                )
                + settings.coordinate_loss_weight
                * masked_coordinate_loss(
                    logits,
                    batch["keypoints"].to(logits.device),
                    visible,
                    settings.image_size,
                )
            )

        def metric_update(
            accumulator: Any,
            logits: torch.Tensor,
            batch: dict[str, Any],
            _loss: torch.Tensor,
        ) -> None:
            accumulator.update(
                logits,
                batch["keypoints"].to(logits.device),
                batch["visible"].to(logits.device),
                batch["torso_scale"],
                batch["head_scale"],
                settings.image_size,
            )

        metric_factory = PoseMetricAccumulator

        core_key, head_key = "core", "head"
    else:
        raise ValueError(f"Unsupported dense task {task!r}")

    if not settings.vision2_hybrid_enabled:
        raise RuntimeError("Hardware bridge requires vision2_hybrid.enabled=true")
    actual_train_limit = int(
        train_limit or settings.hardware_capture_train_limit
    )
    actual_eval_limit = int(eval_limit or settings.hardware_capture_eval_limit)
    train_records = _select(train_records, actual_train_limit, settings.random_seed)
    eval_records = _select(eval_records, actual_eval_limit, settings.random_seed + 1)
    datasets = {
        "train": dataset_type(train_records, settings, training=False),
        "eval": dataset_type(eval_records, settings, training=False),
    }
    device = torch.device(
        settings.device
        if settings.device != "cuda" or torch.cuda.is_available()
        else "cpu"
    )
    loaded = load_vision_backbone(settings, device)
    model = build_student(loaded, settings)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if core_key == "backbone.core_state_dict":
        core_state = payload["backbone"]["core_state_dict"]
    else:
        core_state = payload[core_key]
    model.core.load_state_dict(core_state, strict=True)
    model.head.load_state_dict(payload[head_key], strict=True)
    model.core.set_phase_dropout_active(False)
    model.eval()
    return TaskContext(
        task,
        settings,
        loaded,
        model,
        datasets,
        collate,
        preprocess_vision,
        loss,
        metric_factory,
        metric_update,
        payload,
    )


def _refs(context: TaskContext) -> list[SampleRef]:
    result: list[SampleRef] = []
    for split, dataset in context.datasets.items():
        for index, record in enumerate(dataset.records):
            sample_id = str(record.sample_id)
            result.append(
                SampleRef(
                    split,
                    index,
                    _safe_key(split, sample_id),
                    sample_id,
                    str(record.image_path),
                )
            )
    return result


def _batch(context: TaskContext, refs: list[SampleRef]) -> dict[str, Any]:
    return context.collate(
        [context.datasets[ref.split][ref.index] for ref in refs]
    )


def _forward(context: TaskContext, batch: dict[str, Any]):
    inputs = context.preprocess(
        context.loaded.processor, batch["images"], context.loaded.device
    )
    return context.model(
        inputs["pixel_values"], inputs["image_grid_thw"]
    )


def _load_stage_ccd(
    context: TaskContext, session: Path, stage: str, refs: list[SampleRef]
) -> torch.Tensor:
    return torch.stack(
        [
            load_ccd(
                _stage_dir(session, stage),
                ref.key,
                use_simulation=False,
                settings=context.settings,
                persist_registered=False,
                reuse_registered=False,
            )
            for ref in refs
        ]
    ).to(context.loaded.device)


def _install_measurements(
    context: TaskContext, session: Path, refs: list[SampleRef], through: int
) -> None:
    expert = (
        _load_stage_ccd(context, session, STAGES[0], refs)
        if through >= 0
        else None
    )
    global_ccd = (
        _load_stage_ccd(context, session, STAGES[1], refs)
        if through >= 1
        else None
    )
    context.model.core.optical_branch.set_measured_ccd(
        expert=expert, global_=global_ccd
    )


def _clear_measurements(context: TaskContext) -> None:
    context.model.core.optical_branch.clear_measured_ccd()


def _phase(context: TaskContext, stage: str) -> np.ndarray:
    values = phase_tensors(context.model.core.optical_branch.core)
    phase = (
        values["physical_expert_mosaic_rad"]
        if stage == "vision_expert"
        else values["physical_global_phase_rad"]
    )
    if context.settings.hardware_phase_flip_vertical:
        phase = torch.flip(phase, (-2,))
    if context.settings.hardware_phase_flip_horizontal:
        phase = torch.flip(phase, (-1,))
    return encode_active_phase(phase.detach().cpu().numpy())


def _amplitude(context: TaskContext, stage: str) -> torch.Tensor:
    branch = context.model.core.optical_branch
    value = (
        branch.last_expert_input_amplitude
        if stage == "vision_expert"
        else branch.last_global_input_amplitude
    )
    if value is None:
        raise RuntimeError(f"No captured optical input for {stage}")
    return value.detach().cpu()


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, value: dict[str, Any]) -> None:
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8"
    )


@torch.no_grad()
def export_stage(context: TaskContext, session: Path, stage: str) -> None:
    stage_index = STAGES.index(stage)
    refs = _refs(context)
    session.mkdir(parents=True, exist_ok=True)
    manifest = [vars(ref) for ref in refs]
    manifest_path = session / "manifest.csv"
    if manifest_path.is_file():
        with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
            existing = list(csv.DictReader(handle))
        if [row["key"] for row in existing] != [ref.key for ref in refs]:
            raise RuntimeError("Existing hardware manifest uses another subset")
    else:
        _write_csv(manifest_path, manifest)
    destination = _stage_dir(session, stage)
    compact = destination / "compact_amplitude"
    compact.mkdir(parents=True, exist_ok=True)
    (destination / "ccd_captured").mkdir(exist_ok=True)
    save_active_png(
        _phase(context, stage), destination / "compact_phase" / f"{stage}.png"
    )
    rows: list[dict[str, Any]] = []
    try:
        for start in range(0, len(refs), 8):
            current = refs[start : start + 8]
            _install_measurements(context, session, current, stage_index - 1)
            _forward(context, _batch(context, current))
            for ref, value in zip(current, _amplitude(context, stage)):
                encoded, metadata = encode_active_amplitude_with_metadata(
                    value.numpy()
                )
                path = compact / f"{ref.key}.png"
                save_active_png(encoded, path)
                rows.append(
                    {
                        "key": ref.key,
                        "filename": path.name,
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        **metadata,
                    }
                )
            print(
                f"[export_{stage}] {min(start + len(current), len(refs))}/{len(refs)}",
                flush=True,
            )
    finally:
        _clear_measurements(context)
    _write_csv(destination / "compact_amplitude_manifest.csv", rows)
    _write_json(
        destination / "transport_spec.json",
        {
            "schema_version": 1,
            "task": context.task,
            "stage": stage,
            "samples": len(refs),
            "train_samples": sum(ref.split == "train" for ref in refs),
            "eval_samples": sum(ref.split == "eval" for ref in refs),
            "compact_amplitude": "478x478 uint8 PNG in model coordinates",
            "compact_phase": "478x478 uint8 PNG in configured orientation",
            "expected_ccd_upload": "478x478 uint8 grayscale PNG; laboratory performs no flip",
            "normalization": "no background subtraction; model applies per-frame normalization",
            "server_cache": "no simulated CCD and no registered float tensor cache",
        },
    )


def _enable(module: torch.nn.Module) -> None:
    module.requires_grad_(True)


def _downstream_parameters(context: TaskContext, stage: str) -> list[torch.nn.Parameter]:
    model = context.model
    model.requires_grad_(False)
    hybrid = model.core.hybrid
    branch = hybrid.optical_branch
    if stage == "vision_expert":
        for module in (
            hybrid.blocks,
            branch.expert_readout,
            branch.expert_output_adapter,
            branch.core.global_phase,
            branch.core.readout,
            branch.core.output_adapter,
            hybrid.output_norm,
            model.head,
        ):
            _enable(module)
        hybrid.block1_optical_fusion_logit.requires_grad_(True)
        hybrid.block2_optical_fusion_logit.requires_grad_(True)
    else:
        for module in (
            hybrid.blocks[1],
            branch.core.readout,
            branch.core.output_adapter,
            hybrid.output_norm,
            model.head,
        ):
            _enable(module)
        hybrid.block2_optical_fusion_logit.requires_grad_(True)
    return [parameter for parameter in model.parameters() if parameter.requires_grad]


def _optimizer(
    context: TaskContext, parameters: list[torch.nn.Parameter]
) -> torch.optim.Optimizer:
    phase, readout, head, electronic = [], [], [], []
    head_ids = {id(parameter) for parameter in context.model.head.parameters()}
    allowed = {id(parameter) for parameter in parameters}
    for name, parameter in context.model.named_parameters():
        if id(parameter) not in allowed:
            continue
        if "raw_phase" in name:
            phase.append(parameter)
        elif id(parameter) in head_ids:
            head.append(parameter)
        elif "readout" in name or "output_adapter" in name:
            readout.append(parameter)
        else:
            electronic.append(parameter)
    settings = context.settings
    groups = [
        (electronic, settings.hardware_finetune_learning_rate, "electronic"),
        (phase, settings.phase_learning_rate, "phase"),
        (readout, settings.dense_readout_learning_rate, "readout"),
        (head, settings.dense_head_learning_rate, "head"),
    ]
    return torch.optim.AdamW(
        [
            {"params": values, "lr": lr, "name": name}
            for values, lr, name in groups
            if values
        ],
        weight_decay=settings.weight_decay,
    )


def _save_payload(
    context: TaskContext, path: Path, stage: str, epoch: int, loss: float
) -> None:
    payload = dict(context.checkpoint_payload)
    if context.task == "salicon":
        payload["optical_core"] = context.model.core.state_dict()
        payload["head"] = context.model.head.state_dict()
    elif context.task == "lsp":
        payload["core"] = context.model.core.state_dict()
        payload["head"] = context.model.head.state_dict()
    else:
        payload["backbone"] = dict(payload["backbone"])
        payload["backbone"]["core_state_dict"] = context.model.core.state_dict()
        payload["head_state_dict"] = context.model.head.state_dict()
    payload["hardware_finetune"] = {
        "stage": stage,
        "epoch": epoch,
        "train_loss": loss,
        "measured_stages": list(STAGES[: STAGES.index(stage) + 1]),
        "rule": "only train modules downstream of the newest measured CCD",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def _load_payload_into_model(context: TaskContext, path: Path) -> None:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if context.task == "salicon":
        core_state, head_state = payload["optical_core"], payload["head"]
    elif context.task == "lsp":
        core_state, head_state = payload["core"], payload["head"]
    else:
        core_state = payload["backbone"]["core_state_dict"]
        head_state = payload["head_state_dict"]
    context.model.core.load_state_dict(core_state, strict=True)
    context.model.head.load_state_dict(head_state, strict=True)


@torch.no_grad()
def _evaluate(
    context: TaskContext, session: Path, refs: list[SampleRef], through: int
) -> dict[str, Any]:
    context.model.eval()
    context.model.core.set_phase_dropout_active(False)
    total, samples = 0.0, 0
    accumulator = context.metric_factory()
    for start in range(0, len(refs), context.settings.hardware_finetune_batch_size):
        current = refs[start : start + context.settings.hardware_finetune_batch_size]
        batch = _batch(context, current)
        _install_measurements(context, session, current, through)
        logits = _forward(context, batch)[0]
        value = context.task_loss(logits, batch)
        total += float(value) * len(current)
        context.metric_update(accumulator, logits, batch, value)
        samples += len(current)
    return {
        "loss": total / max(1, samples),
        **accumulator.compute(),
    }


def finetune_stage(
    context: TaskContext, session: Path, stage: str, epochs: int, checkpoint: Path
) -> None:
    stage_index = STAGES.index(stage)
    refs = _refs(context)
    train_refs = [ref for ref in refs if ref.split == "train"]
    eval_refs = [ref for ref in refs if ref.split == "eval"]
    parameters = _downstream_parameters(context, stage)
    optimizer = _optimizer(context, parameters)
    generator = torch.Generator().manual_seed(context.settings.random_seed)
    best = float("inf")
    output = session / "checkpoints" / f"after_{stage}.pt"
    try:
        for epoch in range(1, epochs + 1):
            context.model.train(True)
            context.model.core.set_phase_dropout_active(stage == "vision_expert")
            order = torch.randperm(len(train_refs), generator=generator).tolist()
            total, samples = 0.0, 0
            size = context.settings.hardware_finetune_batch_size
            for start in range(0, len(order), size):
                current = [train_refs[index] for index in order[start : start + size]]
                batch = _batch(context, current)
                _install_measurements(context, session, current, stage_index)
                logits = _forward(context, batch)[0]
                loss = context.task_loss(logits, batch)
                if not torch.isfinite(loss):
                    raise RuntimeError(f"Non-finite hardware loss at epoch {epoch}")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    parameters, 1.0, error_if_nonfinite=True
                )
                optimizer.step()
                total += float(loss.detach()) * len(current)
                samples += len(current)
            average = total / max(1, samples)
            print(
                f"[finetune_{stage}] epoch={epoch:03d}/{epochs:03d} loss={average:.6f}",
                flush=True,
            )
            if average < best:
                best = average
                _save_payload(context, output, stage, epoch, average)
        _load_payload_into_model(context, output)
        eval_metrics = _evaluate(context, session, eval_refs, stage_index)
        _write_json(
            _stage_dir(session, stage) / "finetune_metrics.json",
            {
                "task": context.task,
                "stage": stage,
                "checkpoint": str(output),
                "train_loss": best,
                "eval_metrics": eval_metrics,
                "eval_samples": len(eval_refs),
                "background_subtraction": False,
            },
        )
    finally:
        _clear_measurements(context)


def _close(context: TaskContext) -> None:
    target = getattr(context.model, "student", context.model)
    if hasattr(target, "restore_native"):
        target.restore_native()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Two-stage Vision2 dense-task optical export/fine-tuning"
    )
    parser.add_argument("--task", choices=("salicon", "isic", "lsp"), required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--session-dir", required=True)
    parser.add_argument("--stage", choices=STAGES, required=True)
    parser.add_argument("--phase", choices=("export", "finetune"), required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--eval-limit", type=int)
    args = parser.parse_args()
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    session = Path(args.session_dir).expanduser().resolve()
    context = _load_task(
        args.task,
        args.config,
        checkpoint,
        args.train_limit,
        args.eval_limit,
    )
    try:
        if args.phase == "export":
            export_stage(context, session, args.stage)
        else:
            finetune_stage(context, session, args.stage, args.epochs, checkpoint)
    finally:
        _close(context)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
