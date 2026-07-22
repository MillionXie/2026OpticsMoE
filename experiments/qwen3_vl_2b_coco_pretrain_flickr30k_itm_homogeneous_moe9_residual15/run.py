from __future__ import annotations

import argparse
import copy
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from .datasets import DatasetBundle, load_flickr30k, make_indexed_loader, targets_of
from .generic_datasets import GenericDatasetBundle, load_generic_coco
from .generic_training import load_generic_checkpoint, train_generic_distillation
from .io_utils import resolve_device, resolve_dtype, runtime_metadata, set_seed, write_json
from .modeling import LoadedBackbone, build_head, load_backbone, load_processor, module_parameters
from .optics import DeepStackMultimodalReplacement, LanguageDeepStackHomogeneousMoE, VisionDeepStackHomogeneousMoE
from .processor_cache import ProcessorCacheStore, build_processor_cache, validate_processor_cache
from .settings import Settings, load_settings, resolve_path
from .teacher_cache import TeacherCacheStore, build_teacher_cache
from .training import (evaluate_student, generate_teacher_logits, load_head, load_student_parts,
                       make_evaluation_loader, save_student_inference, teacher_inference,
                       train_student, train_teacher_head)


PHASES = ("download", "prepare_data", "generic_prepare_data", "generic_input_precompute",
          "generic_teacher_precompute", "generic_pretrain", "flickr_input_precompute",
          "flickr_teacher_precompute", "flickr_teacher_train", "flickr_teacher_logits",
          "flickr_teacher_inference", "flickr_finetune", "flickr_inference", "compare", "all",
          # Backward-compatible target-stage aliases inherited from the source experiment.
          "input_precompute", "teacher_precompute", "teacher_train", "teacher_logits",
          "teacher_inference", "student_train", "student_inference")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="COCO generic hidden distillation -> Flickr30k Qwen3-VL optical-MoE fine-tuning"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--phase", choices=PHASES, default="all")
    parser.add_argument("--device"); parser.add_argument("--cache-dir", type=Path); parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--local-files-only", action="store_true"); parser.add_argument("--epochs", type=int)
    parser.add_argument("--student-batch-size", type=int); parser.add_argument("--train-samples-per-epoch", type=int)
    parser.add_argument("--train-samples-per-class-per-epoch", type=int)
    parser.add_argument("--log-interval-batches", type=int)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv); settings = load_settings(args.config); _overrides(settings, args)
    phase = _canonical_phase(args.phase); set_seed(settings.seed); _dirs(settings.output_dir)
    generic_root = settings.output_dir / "generic_pretrain"; _dirs(generic_root)
    generic_needed = settings.generic_pretrain_enabled and (phase.startswith("generic_") or phase in {"prepare_data", "download", "all"})
    if phase.startswith("generic_") and not settings.generic_pretrain_enabled:
        raise RuntimeError("This phase requires generic_pretrain.enabled=true")
    flickr_needed = phase.startswith("flickr_") or phase in {"prepare_data", "download", "compare", "all"}
    generic: GenericDatasetBundle | None = load_generic_coco(settings, True) if generic_needed else None
    flickr: DatasetBundle | None = load_flickr30k(settings, True) if flickr_needed else None
    if generic is not None: write_json(generic_root / "dataset.json", generic.metadata)
    if flickr is not None: write_json(settings.output_dir / "dataset.json", flickr.metadata)
    write_json(settings.output_dir / "config_resolved.json", settings.to_dict())
    if phase in {"generic_prepare_data", "prepare_data", "download"}:
        if generic is not None: print(f"Generic COCO ready: samples={len(generic.train)} manifest={generic.manifest_digest}", flush=True)
        if flickr is not None: print(f"Flickr30k ready: train_pairs={len(flickr.train)} test_pairs={len(flickr.test)}", flush=True)
        return 0
    device = resolve_device(settings.device); write_json(settings.output_dir / "environment.json", runtime_metadata(device))

    # Cache-only target-head phases do not need to allocate Qwen.
    if phase in {"flickr_teacher_train", "flickr_teacher_logits", "flickr_teacher_inference", "compare"}:
        if phase == "compare": _compare(settings); return 0
        assert flickr is not None
        stores = _stores(settings, flickr); _architecture_from_cache(settings, stores["train"])
        if phase == "flickr_teacher_train":
            train_teacher_head(stores["train"], stores["test"], flickr.train, flickr.test, settings, device); return 0
        head = load_head(settings.output_dir / "checkpoints" / "teacher_head.pt", settings, device)
        if phase == "flickr_teacher_logits": generate_teacher_logits(head, stores, settings, device)
        else: teacher_inference(head, stores["test"], flickr.test, settings, device)
        return 0

    processor_phases = {"generic_input_precompute", "generic_teacher_precompute", "generic_pretrain",
                        "flickr_input_precompute", "flickr_teacher_precompute", "flickr_finetune", "all"}
    if phase in processor_phases:
        processor = load_processor(settings.model_id, settings.cache_dir, settings.local_files_only,
                                   settings.processor_min_pixels, settings.processor_max_pixels)
        if generic is not None and phase in {"generic_input_precompute", "generic_teacher_precompute", "generic_pretrain", "all"}:
            _generic_input_precompute(processor, generic, settings)
        if flickr is not None and phase in {"flickr_input_precompute", "flickr_teacher_precompute", "flickr_finetune", "all"}:
            _input_precompute(processor, flickr, settings)
        if phase in {"generic_input_precompute", "flickr_input_precompute"}: return 0
        del processor

    loaded = _load_model(settings, device); settings.resolve_architecture(loaded.model)
    replacement = _replacement(loaded, settings, device)
    write_json(settings.output_dir / "config_resolved.json", settings.to_dict()); _model_report(loaded.model, replacement, settings)
    try:
        if generic is not None and phase in {"generic_teacher_precompute", "generic_pretrain", "all"}:
            generic_settings = _generic_runtime_settings(settings, generic)
            _generic_teacher_precompute(loaded, replacement, generic, generic_settings, device)
            if phase == "generic_teacher_precompute": return 0
            generic_teacher = TeacherCacheStore(generic_root / "teacher_cache" / "train.pt",
                                                settings.teacher_cache_lru_shards)
            generic_inputs = ProcessorCacheStore(generic_root / "processor_cache" / "train.pt",
                                                 settings.teacher_cache_lru_shards)
            train_generic_distillation(loaded.model, replacement, generic.train, generic_teacher,
                                       generic_inputs, settings, device)
            if phase == "generic_pretrain": return 0

        assert flickr is not None
        if phase in {"flickr_teacher_precompute", "all"}:
            _teacher_precompute(loaded, replacement, flickr, settings, device)
            if phase == "flickr_teacher_precompute": return 0
        stores = _stores(settings, flickr); inputs = _input_stores(settings, flickr)
        if phase == "all":
            teacher_head = train_teacher_head(stores["train"], stores["test"], flickr.train, flickr.test, settings, device)
            generate_teacher_logits(teacher_head, stores, settings, device)
            teacher_inference(teacher_head, stores["test"], flickr.test, settings, device)
        if phase in {"flickr_finetune", "all"}:
            if settings.finetune_from_generic_pretrain:
                load_generic_checkpoint(settings.output_dir, replacement, "final")
            head = build_head(settings, settings.text_hidden_size).to(device)
            train_student(loaded.model, replacement, head, flickr.train, flickr.test, stores["train"], stores["test"],
                          inputs["train"], inputs["test"], settings, device)
            if phase == "flickr_finetune": return 0
        if phase in {"flickr_inference", "all"}:
            head = build_head(settings, settings.text_hidden_size).to(device)
            load_student_parts(settings.output_dir, replacement, head, "best")
            loader = make_evaluation_loader(flickr.test, inputs["test"], settings.inference_batch_size)
            predictions = settings.output_dir / "metrics" / "student_predictions.csv"
            report = evaluate_student(loaded.model, replacement, head, loader, settings, device, flickr.test, predictions)
            save_student_inference(report, settings, replacement, predictions)
            if phase == "flickr_inference": return 0
        if phase == "all": _compare(settings)
        return 0
    finally: replacement.close()


def _overrides(settings: Settings, args: argparse.Namespace) -> None:
    if args.device: settings.device = args.device
    if args.cache_dir: settings.cache_dir = resolve_path(args.cache_dir, Path.cwd(), "cache_dir")
    if args.output_dir: settings.output_dir = resolve_path(args.output_dir, Path.cwd(), "output_dir")
    if args.local_files_only: settings.local_files_only = True
    for name in ("epochs", "student_batch_size", "train_samples_per_epoch",
                 "train_samples_per_class_per_epoch", "log_interval_batches"):
        value = getattr(args, name)
        if value is not None:
            setattr(settings, name, value)
            if name == "train_samples_per_epoch": settings.train_samples_per_class_per_epoch = None
            if name == "train_samples_per_class_per_epoch": settings.train_samples_per_epoch = None
    settings.validate()


def _canonical_phase(phase: str) -> str:
    aliases = {"input_precompute": "flickr_input_precompute", "teacher_precompute": "flickr_teacher_precompute",
               "teacher_train": "flickr_teacher_train", "teacher_logits": "flickr_teacher_logits",
               "teacher_inference": "flickr_teacher_inference", "student_train": "flickr_finetune",
               "student_inference": "flickr_inference"}
    return aliases.get(phase, phase)


def _generic_runtime_settings(settings: Settings, data: GenericDatasetBundle) -> Settings:
    """Present generic COCO identity to the existing strict processor/teacher caches."""
    runtime = copy.copy(settings)
    runtime.output_dir = settings.output_dir / "generic_pretrain"
    runtime.dataset = settings.generic_dataset
    runtime.dataset_repo_id = settings.generic_dataset_repo_id
    runtime.dataset_revision = settings.generic_dataset_revision
    runtime.resolved_dataset_fingerprints = {"generic": data.dataset_fingerprint}
    runtime.pair_manifest_digests = {"train": data.manifest_digest}
    runtime.prompt_template = settings.generic_prompt_template
    runtime.negative_sampling_algorithm = "stable_sha256_one_caption_per_image_v1"
    runtime.captions_per_image = 1; runtime.negatives_per_positive = 0
    runtime.cache_purpose = "generic_multimodal_hidden_distillation"
    return runtime


def _load_model(settings: Settings, device: torch.device) -> LoadedBackbone:
    _log(f"loading {settings.model_id}")
    return load_backbone(settings.model_id, settings.cache_dir, settings.local_files_only, resolve_dtype(settings.dtype),
                         device, settings.attn_implementation, settings.processor_min_pixels, settings.processor_max_pixels)


def _replacement(loaded: LoadedBackbone, settings: Settings, device: torch.device) -> DeepStackMultimodalReplacement:
    vision = VisionDeepStackHomogeneousMoE(settings.vision_hidden_size, settings).to(device)
    language = LanguageDeepStackHomogeneousMoE(settings.text_hidden_size, settings).to(device)
    return DeepStackMultimodalReplacement(loaded.model, vision, language, settings)


def _teacher_precompute(loaded: LoadedBackbone, replacement: Any, data: DatasetBundle,
                        settings: Settings, device: torch.device) -> None:
    input_stores = _input_stores(settings, data)
    for split, dataset in (("train", data.train), ("test", data.test)):
        build_teacher_cache(split, loaded.model, replacement, input_stores[split], targets_of(dataset),
                            len(dataset), settings, device)


def _generic_teacher_precompute(loaded: LoadedBackbone, replacement: Any, data: GenericDatasetBundle,
                                runtime: Settings, device: torch.device) -> None:
    inputs = ProcessorCacheStore(runtime.output_dir / "processor_cache" / "train.pt",
                                 runtime.teacher_cache_lru_shards)
    validate_processor_cache(inputs, "train", len(data.train), runtime)
    build_teacher_cache("train", loaded.model, replacement, inputs, targets_of(data.train),
                        len(data.train), runtime, device)


def _input_precompute(processor: Any, data: DatasetBundle, settings: Settings) -> None:
    for split, dataset in (("train", data.train), ("test", data.test)):
        loader = make_indexed_loader(dataset, settings.feature_batch_size, settings.num_workers, False, settings.seed)
        build_processor_cache(split, processor, loader, len(dataset), settings)


def _generic_input_precompute(processor: Any, data: GenericDatasetBundle, settings: Settings) -> None:
    runtime = _generic_runtime_settings(settings, data)
    loader = make_indexed_loader(data.train, runtime.feature_batch_size, runtime.num_workers, False, runtime.seed)
    build_processor_cache("train", processor, loader, len(data.train), runtime)


def _stores(settings: Settings, data: DatasetBundle) -> dict[str, TeacherCacheStore]:
    stores = {split: TeacherCacheStore(settings.output_dir / "teacher_cache" / f"{split}.pt",
                                       settings.teacher_cache_lru_shards) for split in ("train", "test")}
    for split, dataset in (("train", data.train), ("test", data.test)):
        expected = {"split": split, "sample_count": len(dataset), "dataset": settings.dataset,
                    "pair_manifest_digest": (settings.pair_manifest_digests or {}).get(split),
                    "prompt_template": settings.prompt_template,
                    "negative_sampling_algorithm": settings.negative_sampling_algorithm,
                    "captions_per_image": settings.captions_per_image, "seed": settings.seed,
                    "processor_min_pixels": settings.processor_min_pixels,
                    "processor_max_pixels": settings.processor_max_pixels,
                    "replacement_mode": "qwen3_vl_native_deepstack_teacher_targets"}
        changed = [key for key, value in expected.items() if stores[split].metadata.get(key) != value]
        if changed: raise RuntimeError(f"Teacher cache metadata mismatch for {split}: {changed}. Rebuild it.")
    return stores


def _input_stores(settings: Settings, data: DatasetBundle) -> dict[str, ProcessorCacheStore]:
    stores = {split: ProcessorCacheStore(settings.output_dir / "processor_cache" / f"{split}.pt",
                                         settings.teacher_cache_lru_shards) for split in ("train", "test")}
    for split, dataset in (("train", data.train), ("test", data.test)):
        validate_processor_cache(stores[split], split, len(dataset), settings)
    return stores


def _architecture_from_cache(settings: Settings, store: TeacherCacheStore) -> None:
    for name in ("vision_depth", "vision_hidden_size", "text_depth", "text_hidden_size"):
        setattr(settings, name, int(store.metadata[name]))
    settings.deepstack_visual_indexes = tuple(store.metadata["deepstack_visual_indexes"])


def _model_report(model: torch.nn.Module, replacement: Any, settings: Settings) -> None:
    vision = replacement.vision_surrogate.parameter_breakdown(); language = replacement.language_surrogate.parameter_breakdown()
    head = build_head(settings, settings.text_hidden_size); head_params = module_parameters(head)
    trainable = vision["trainable_parameters"] + head_params
    if settings.student_language_mode == "optical_moe": trainable += language["trainable_parameters"]
    alignment = replacement.alignment_specification()
    trainable += alignment["vision_attention_trainable_parameters"]
    trainable += alignment["language_attention_trainable_parameters"]
    write_json(settings.output_dir / "model.json", {
        "model_id": settings.model_id, "task": "binary image-text matching",
        "training_protocol": "generic_COCO_hidden_distillation_then_Flickr30k_fine_tuning",
        "generic_teacher_fine_tuned": False,
        "student_language_mode": settings.student_language_mode,
        "vision_depth": settings.vision_depth, "vision_hidden_size": settings.vision_hidden_size,
        "text_depth": settings.text_depth, "text_hidden_size": settings.text_hidden_size,
        "deepstack_visual_indexes": list(settings.deepstack_visual_indexes or []),
        "vision_tap_stages": list(settings.vision_tap_stages), "native_deepstack_preserved": True,
        "optical_depth": {"logical_stages": settings.logical_optical_stages,
                          "physical_layers_per_logical_stage": settings.physical_layers_per_logical_stage,
                          "total_physical_layers": settings.expert_layers,
                          "grouping": "five Qwen-facing stages, three physical phase/OEO layers per stage"},
        "detector_layernorm_scope": settings.detector_layernorm_scope,
        "vision": vision, "language": language if settings.student_language_mode == "optical_moe" else {"frozen_electronic": True},
        "transformer_block_alignment": alignment,
        "head": head.specification(), "student_trainable_parameters": trainable,
        "qwen_total_parameters": module_parameters(model), "qwen_original_trainable_parameters": 0,
    })


def _compare(settings: Settings) -> None:
    teacher_path = settings.output_dir / "metrics" / "teacher_inference.json"
    student_path = settings.output_dir / "metrics" / "student_inference.json"
    if not teacher_path.is_file() or not student_path.is_file():
        raise FileNotFoundError("compare requires metrics/teacher_inference.json and metrics/student_inference.json")
    teacher = json.loads(teacher_path.read_text(encoding="utf-8")); student = json.loads(student_path.read_text(encoding="utf-8"))
    names = ("bce_loss", "accuracy", "balanced_accuracy", "auroc", "average_precision", "precision", "recall", "f1")
    model_report = json.loads((settings.output_dir / "model.json").read_text(encoding="utf-8")) if (settings.output_dir / "model.json").is_file() else {}
    write_json(settings.output_dir / "metrics" / "comparison.json", {
        "electronic_test_metrics": teacher, "optical_test_metrics": student,
        "optical_minus_electronic": {name: float(student[name]) - float(teacher[name]) for name in names},
        "accuracy_retention_ratio": float(student["accuracy"]) / float(teacher["accuracy"]) if teacher["accuracy"] else None,
        "electronic_manifest_digest": teacher.get("pair_manifest_digest"),
        "optical_manifest_digest": student.get("pair_manifest_digest"),
        "same_manifest": teacher.get("pair_manifest_digest") == student.get("pair_manifest_digest"),
        "model_and_head_parameters": {"head": model_report.get("head"),
                                      "student_trainable_parameters": model_report.get("student_trainable_parameters")},
        "student_language_mode": settings.student_language_mode,
        "protocol_warning": "test was evaluated each epoch and selected best checkpoints at user request",
    })


def _dirs(root: Path) -> None:
    for name in ("pair_manifests", "teacher_cache", "processor_cache", "metrics", "checkpoints", "figures"):
        (root / name).mkdir(parents=True, exist_ok=True)


def _log(message: str) -> None:
    print(f"[{datetime.now(timezone.utc).isoformat(timespec='seconds')}] {message}", flush=True)


if __name__ == "__main__": raise SystemExit(main())
