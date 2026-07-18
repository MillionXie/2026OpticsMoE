from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from .data_prepare import ensure_spaq_dataset
from .datasets import DatasetBundle, load_spaq, make_indexed_loader
from .io_utils import resolve_device, resolve_dtype, runtime_metadata, set_seed, write_json
from .modeling import LoadedBackbone, build_head, load_backbone, load_processor, module_parameters
from .optics.moe import VisionHomogeneousMoESurrogate
from .optics.replacement import VisionStackReplacement
from .processor_cache import (ProcessorCacheStore, build_processor_cache,
                              validate_processor_cache)
from .settings import Settings, load_settings, resolve_path
from .teacher_cache import TeacherCacheStore, build_teacher_cache
from .training import (evaluate_student, generate_teacher_predictions, load_head, load_student_parts,
                       make_cached_evaluation_loader, save_student_inference, teacher_inference,
                       train_student, train_teacher_head)


PHASES = ("download", "prepare_data", "input_precompute", "teacher_precompute", "teacher_train", "teacher_predictions", "teacher_logits",
          "teacher_inference", "student_train", "student_inference", "compare", "all")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Qwen3-VL-2B SPAQ single-attribute electronic-teacher to homogeneous optical-MoE vision distillation"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--phase", choices=PHASES, default="all")
    parser.add_argument("--device")
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--model-id")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", "--student-batch-size", dest="student_batch_size", type=int)
    parser.add_argument("--train-samples-per-epoch", type=int)
    parser.add_argument("--log-interval-batches", type=int)
    parser.add_argument("--checkpoint-interval-epochs", type=int)
    parser.add_argument("--visualization-interval-epochs", type=int)
    parser.add_argument("--disable-visualization", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = load_settings(args.config); _apply_overrides(settings, args); set_seed(settings.seed)
    _make_dirs(settings.output_dir)
    preparation = ensure_spaq_dataset(settings)
    write_json(settings.output_dir / "data_preparation.json", preparation)
    data = load_spaq(settings, persist_split=True)
    settings.resolved_annotations_file = data.metadata["annotation_file"]
    settings.split_digest = data.metadata["split_digest"]
    write_json(settings.output_dir / "config_resolved.json", settings.to_dict())
    write_json(settings.output_dir / "dataset.json", data.metadata)
    if args.phase in {"download", "prepare_data"}:
        print(f"SPAQ RGB ready: train={len(data.train)} test={len(data.test)} task={settings.task_name}", flush=True)
        return 0
    if args.phase == "input_precompute":
        processor = load_processor(settings.model_id, settings.cache_dir, settings.local_files_only,
                                   settings.processor_min_pixels, settings.processor_max_pixels)
        _precompute_inputs(processor, data, settings)
        return 0
    device = resolve_device(settings.device)
    write_json(settings.output_dir / "environment.json", runtime_metadata(device))
    if args.phase in {"teacher_train", "teacher_predictions", "teacher_logits", "teacher_inference", "compare"}:
        stores = _load_stores(settings, data); _resolve_architecture_from_cache(settings, stores["train"])
        write_json(settings.output_dir / "config_resolved.json", settings.to_dict())
        if args.phase == "teacher_train":
            train_teacher_head(stores["train"], stores["test"], data.train, data.test, settings, device); return 0
        head = load_head(settings.output_dir / "checkpoints" / "teacher_head.pt", settings, device)
        if args.phase in {"teacher_predictions", "teacher_logits"}:
            if args.phase == "teacher_logits":
                print(f"[compatibility] teacher_logits means cached scalar teacher {settings.task_name} predictions", flush=True)
            generate_teacher_predictions(head, stores, _datasets(data), settings, device); return 0
        if args.phase == "teacher_inference":
            teacher_inference(head, stores["test"], data.test, settings, device); return 0
        _compare(settings); return 0
    loaded = _load_model(settings, device); settings.resolve_architecture(loaded.model)
    replacement = _build_replacement(loaded, settings, device)
    write_json(settings.output_dir / "config_resolved.json", settings.to_dict())
    _write_model_report(loaded.model, replacement, settings)
    try:
        if args.phase in {"teacher_precompute", "all"}:
            _precompute(loaded, replacement, data, settings, device)
            if args.phase == "teacher_precompute": return 0
        stores = _load_stores(settings, data)
        input_stores: dict[str, ProcessorCacheStore] | None = None
        if args.phase == "all":
            teacher_head = train_teacher_head(stores["train"], stores["test"], data.train, data.test, settings, device)
            generate_teacher_predictions(teacher_head, stores, _datasets(data), settings, device)
            teacher_inference(teacher_head, stores["test"], data.test, settings, device)
        if args.phase in {"student_train", "all"}:
            input_stores = _ensure_input_stores(loaded.processor, data, settings)
            student_head = build_head(settings, settings.vision_hidden_size).to(device)
            train_student(loaded.model, loaded.processor, replacement, student_head, data.train, data.test,
                          stores["train"], stores["test"], input_stores["train"], input_stores["test"],
                          settings, device)
            if args.phase == "student_train": return 0
        if args.phase in {"student_inference", "all"}:
            if input_stores is None:
                input_stores = _ensure_input_stores(loaded.processor, data, settings)
            student_head = build_head(settings, settings.vision_hidden_size).to(device)
            load_student_parts(settings.output_dir, replacement, student_head, "best")
            loader = make_cached_evaluation_loader(data.test, input_stores["test"], settings.inference_batch_size)
            predictions_path = settings.output_dir / "metrics" / "student_predictions.csv" if settings.save_predictions else None
            report = evaluate_student(loaded.model, loaded.processor, replacement, student_head, loader,
                                      device, data.test, predictions_path, inputs_are_cached=True)
            save_student_inference(report, settings, replacement, predictions_path)
            if args.phase == "student_inference": return 0
        if args.phase == "all": _compare(settings)
        return 0
    finally:
        replacement.close()


def _apply_overrides(settings: Settings, args: argparse.Namespace) -> None:
    if args.device: settings.device = args.device
    if args.cache_dir: settings.cache_dir = resolve_path(args.cache_dir, Path.cwd(), "cache_dir")
    if args.output_dir: settings.output_dir = resolve_path(args.output_dir, Path.cwd(), "output_dir")
    if args.model_id:
        settings.model_id = args.model_id if args.model_id == "Qwen/Qwen3-VL-2B-Instruct" else str(resolve_path(args.model_id, Path.cwd(), "model_id"))
    if args.local_files_only: settings.local_files_only = True
    for argument, attribute in (("epochs", "epochs"), ("student_batch_size", "student_batch_size"),
                                ("train_samples_per_epoch", "train_samples_per_epoch"),
                                ("log_interval_batches", "log_interval_batches"),
                                ("checkpoint_interval_epochs", "checkpoint_interval_epochs"),
                                ("visualization_interval_epochs", "visualization_interval_epochs")):
        value = getattr(args, argument)
        if value is not None: setattr(settings, attribute, value)
    if args.disable_visualization: settings.visualization_enabled = False
    settings.validate()


def _load_model(settings: Settings, device: torch.device) -> LoadedBackbone:
    _log(f"loading {settings.model_id}")
    return load_backbone(settings.model_id, settings.cache_dir, settings.local_files_only, resolve_dtype(settings.dtype),
                         device, settings.attn_implementation, settings.processor_min_pixels, settings.processor_max_pixels)


def _build_replacement(loaded: LoadedBackbone, settings: Settings, device: torch.device) -> VisionStackReplacement:
    return VisionStackReplacement(loaded.model, VisionHomogeneousMoESurrogate(settings.vision_hidden_size, settings).to(device))


def _precompute(loaded: LoadedBackbone, replacement: VisionStackReplacement, data: DatasetBundle,
                settings: Settings, device: torch.device) -> None:
    if _external_cache_complete(settings, "teacher_cache"):
        print(f"[teacher_precompute] reusing task-independent visual cache from {settings.source_cache_run_dir}", flush=True)
    else:
        for split, dataset in (("train", data.train), ("test", data.test)):
            loader = make_indexed_loader(dataset, settings.feature_batch_size, settings.num_workers, False, settings.seed)
            build_teacher_cache(split, loaded.model, loaded.processor, replacement, loader, len(dataset), settings, device)
    _precompute_inputs(loaded.processor, data, settings)


def _precompute_inputs(processor: Any, data: DatasetBundle, settings: Settings) -> None:
    if _external_cache_complete(settings, "processor_cache"):
        print(f"[processor_cache] reusing task-independent processor cache from {settings.source_cache_run_dir}", flush=True)
        return
    for split, dataset in (("train", data.train), ("test", data.test)):
        # This phase has no Qwen model forward and is not constrained by teacher
        # feature memory. Reuse the tested student batch size instead of the
        # deliberately conservative feature_batch_size=1.
        loader = make_indexed_loader(dataset, settings.student_batch_size, settings.num_workers, False, settings.seed)
        build_processor_cache(split, processor, loader, len(dataset), settings)


def _ensure_input_stores(processor: Any, data: DatasetBundle,
                         settings: Settings) -> dict[str, ProcessorCacheStore]:
    cache_root = _selected_cache_root(settings, "processor_cache")
    manifests = [cache_root / f"{split}.pt" for split in ("train", "test")]
    if not all(path.is_file() for path in manifests):
        print("[processor_cache] missing cache detected; building it once before student execution", flush=True)
        _precompute_inputs(processor, data, settings)
    stores = {
        split: ProcessorCacheStore(_selected_cache_root(settings, "processor_cache") / f"{split}.pt",
                                   settings.teacher_cache_lru_shards)
        for split in ("train", "test")
    }
    for split, dataset in (("train", data.train), ("test", data.test)):
        _validate_reusable_cache(stores[split].metadata, split, len(dataset), settings, "processor")
    return stores


def _load_stores(settings: Settings, data: DatasetBundle) -> dict[str, TeacherCacheStore]:
    stores = {split: TeacherCacheStore(_selected_cache_root(settings, "teacher_cache") / f"{split}.pt",
                                       settings.teacher_cache_lru_shards) for split in ("train", "test")}
    for split, dataset in (("train", data.train), ("test", data.test)):
        metadata = stores[split].metadata
        expected = {"split": split, "sample_count": len(dataset), "data_root": str(settings.data_root),
                    "model_id": settings.model_id,
                    "processor_min_pixels": settings.processor_min_pixels,
                    "processor_max_pixels": settings.processor_max_pixels,
                    "cache_dtype": settings.cache_dtype, "dtype": settings.dtype,
                    "attention_implementation": settings.attn_implementation,
                    "replacement_mode": "complete_vision_stack_homogeneous_moe9x5", "input_color_mode": "RGB",
                    "split_digest": settings.split_digest}
        changed = [key for key, value in expected.items() if metadata.get(key) != value]
        if changed:
            raise RuntimeError(f"Teacher cache metadata mismatch for {split}: {changed}. Delete it and rerun teacher_precompute.")
    return stores


def _resolve_architecture_from_cache(settings: Settings, store: TeacherCacheStore) -> None:
    settings.vision_depth = int(store.metadata["vision_depth"])
    settings.vision_hidden_size = int(store.metadata["vision_hidden_size"])


def _write_model_report(model: torch.nn.Module, replacement: VisionStackReplacement, settings: Settings) -> None:
    head = build_head(settings, settings.vision_hidden_size)
    breakdown = replacement.surrogate.parameter_breakdown(); head_parameters = module_parameters(head)
    student_total = breakdown["surrogate_trainable_parameters"] + head_parameters
    write_json(settings.output_dir / "model.json", {
        "model_id": settings.model_id, "dataset": "SPAQ", "task": settings.task_name, "input_color_mode": "RGB",
        "language_model_used": False, "prompt_used": False,
        "teacher": f"complete electronic Qwen3-VL-2B vision stack + normalized linear {settings.task_name} head",
        "student": f"frozen Qwen vision stem + homogeneous optical MoE9x5 + identical {settings.task_name} head",
        "vision_depth": settings.vision_depth, "vision_hidden_size": settings.vision_hidden_size,
        "token_mapping": "Linear(1024,120) -> LayerNorm -> Softplus -> strict zero-row padding",
        "detector": {"plane_shape": [480, 480], "class_regions": False, "pool": "AvgPool2d(4,4)",
                     "pooled_shape": [120, 120], "layernorm_affine": False,
                     "readout": "ReLU -> first T rows -> Linear(120,1024)"},
        "teacher_student_head_structure_identical": True,
        "teacher_student_head_weights_shared": False,
        "task_isolation": {
            "single_task": True,
            "shared_regression_head_across_tasks": False,
            "shared_optical_weights_across_tasks": False,
            "source_visual_cache_run_dir": str(settings.source_cache_run_dir) if settings.source_cache_run_dir else None,
            "legacy_cached_targets_ignored": settings.source_cache_run_dir is not None,
        },
        "regression_head": head.specification(),
        "target": {"name": settings.task_name, "source_scale": [0.0, 100.0], "training_scale": [0.0, 1.0],
                   "loss": f"SmoothL1(beta={settings.smooth_l1_beta})"},
        "losses": {"hidden": settings.loss_hidden_weight,
                   "teacher_prediction_distillation": settings.loss_prediction_distill_weight,
                   "ground_truth_regression": settings.loss_regression_weight,
                   "router_balance": settings.router_balance_weight,
                   "router_importance": settings.router_importance_weight},
        "optimizer": {"type": settings.optimizer_type, "learning_rate": settings.learning_rate,
                      "student_head_learning_rate": settings.student_head_learning_rate,
                      "router_learning_rate": settings.router_learning_rate,
                      "weight_decay": settings.weight_decay, "scheduler": settings.scheduler_type},
        "sampling": {"train_samples_per_epoch": settings.train_samples_per_epoch,
                     "rotating_epoch_windows": True, "full_train_pool_retained": True,
                     "shard_local_shuffle": True, "shard_size": settings.teacher_cache_shard_size},
        "student_input_pipeline": {
            "source": "persistent Qwen image_processor tensor cache",
            "cache_directory": str(_selected_cache_root(settings, "processor_cache")),
            "storage_dtype": settings.cache_dtype,
            "repeated_jpeg_decode_per_epoch": False,
            "repeated_processor_resize_per_epoch": False,
        },
        "checkpoint_selection": {"split": settings.student_selection_split,
                                 "metric": settings.student_selection_metric,
                                 "warning": "Best-on-test is selection-biased; last checkpoint is also retained."},
        **breakdown, "regression_head_parameters": head_parameters,
        "student_total_trainable_parameters": student_total,
        "optical_phase_ratio": breakdown["optical_phase_parameters"] / student_total,
        "adapter_ratio": breakdown["adapter_parameters"] / student_total,
        "regression_head_ratio": head_parameters / student_total,
        "qwen_total_parameters": module_parameters(model),
        "qwen_trainable_parameters": module_parameters(model, trainable_only=True),
    })


def _compare(settings: Settings) -> None:
    teacher_path = settings.output_dir / "metrics" / "teacher_inference.json"
    student_path = settings.output_dir / "metrics" / "student_inference.json"
    if not teacher_path.is_file() or not student_path.is_file():
        raise FileNotFoundError("teacher_inference.json and student_inference.json are required")
    teacher = json.loads(teacher_path.read_text(encoding="utf-8")); student = json.loads(student_path.read_text(encoding="utf-8"))
    write_json(settings.output_dir / "metrics" / "comparison.json", {
        "dataset": "SPAQ", "task": settings.task_name, "teacher": teacher, "student": student,
        "student_minus_teacher": {"mae": student["mae"] - teacher["mae"],
                                  "srcc": student["srcc"] - teacher["srcc"],
                                  "plcc": student["plcc"] - teacher["plcc"]},
    })


def _make_dirs(root: Path) -> None:
    for relative in ("teacher_cache", "processor_cache", "metrics", "checkpoints", "figures"):
        (root / relative).mkdir(parents=True, exist_ok=True)


def _log(message: str) -> None:
    print(f"[{datetime.now(timezone.utc).isoformat(timespec='seconds')}] {message}", flush=True)


def _datasets(data: DatasetBundle) -> dict[str, Any]:
    return {"train": data.train, "test": data.test}


def _external_cache_complete(settings: Settings, kind: str) -> bool:
    if settings.source_cache_run_dir is None:
        return False
    root = settings.source_cache_run_dir / kind
    present = [(root / f"{split}.pt").is_file() for split in ("train", "test")]
    if any(present) and not all(present):
        raise RuntimeError(f"Incomplete shared {kind} under {root}; both train.pt and test.pt are required")
    return all(present)


def _selected_cache_root(settings: Settings, kind: str) -> Path:
    if _external_cache_complete(settings, kind):
        return settings.source_cache_run_dir / kind  # type: ignore[operator]
    return settings.output_dir / kind


def _validate_reusable_cache(metadata: dict[str, Any], split: str, samples: int,
                             settings: Settings, kind: str) -> None:
    expected = {
        "split": split,
        "sample_count": samples,
        "data_root": str(settings.data_root),
        "split_digest": settings.split_digest,
        "model_id": settings.model_id,
        "processor_min_pixels": settings.processor_min_pixels,
        "processor_max_pixels": settings.processor_max_pixels,
        "input_color_mode": "RGB",
    }
    if kind == "processor":
        expected["storage_dtype"] = settings.cache_dtype
    changed = [key for key, value in expected.items() if metadata.get(key) != value]
    if changed:
        raise RuntimeError(
            f"Reusable {kind} cache metadata mismatch for {split}: {changed}. "
            "Use caches made from the same SPAQ split/model/pixel budget, or set source_cache_run_dir to null."
        )


if __name__ == "__main__":
    raise SystemExit(main())
