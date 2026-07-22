from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from .datasets import DatasetBundle, load_flickr30k, make_indexed_loader
from .io_utils import resolve_device, resolve_dtype, runtime_metadata, set_seed, write_json
from .modeling import LoadedBackbone, build_head, load_backbone, load_processor, module_parameters
from .optics import DeepStackMultimodalReplacement, LanguageDeepStackHomogeneousMoE, VisionDeepStackHomogeneousMoE
from .processor_cache import ProcessorCacheStore, build_processor_cache, validate_processor_cache
from .settings import Settings, load_settings, resolve_path
from .teacher_cache import TeacherCacheStore, build_teacher_cache
from .training import (evaluate_student, generate_teacher_logits, load_head, load_student_parts,
                       make_evaluation_loader, save_student_inference, teacher_inference,
                       train_student, train_teacher_head)


PHASES = ("download", "prepare_data", "input_precompute", "teacher_precompute", "teacher_train",
          "teacher_logits", "teacher_inference", "student_train", "student_inference", "compare", "all")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Flickr30k binary image-text matching with frozen Qwen3-VL-2B and homogeneous optical MoE9"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--phase", choices=PHASES, default="all")
    parser.add_argument("--device"); parser.add_argument("--cache-dir", type=Path); parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--local-files-only", action="store_true"); parser.add_argument("--epochs", type=int)
    parser.add_argument("--student-batch-size", type=int); parser.add_argument("--train-samples-per-epoch", type=int)
    parser.add_argument("--log-interval-batches", type=int)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv); settings = load_settings(args.config); _overrides(settings, args)
    set_seed(settings.seed); _dirs(settings.output_dir)
    data = load_flickr30k(settings, persist_manifest=True)
    write_json(settings.output_dir / "dataset.json", data.metadata)
    write_json(settings.output_dir / "config_resolved.json", settings.to_dict())
    if args.phase in {"download", "prepare_data"}:
        print(f"Flickr30k ready: train_pairs={len(data.train)} test_pairs={len(data.test)} "
              f"manifest={settings.pair_manifest_digests}", flush=True)
        return 0
    device = resolve_device(settings.device); write_json(settings.output_dir / "environment.json", runtime_metadata(device))
    if args.phase in {"teacher_train", "teacher_logits", "teacher_inference", "compare"}:
        stores = _stores(settings, data); _architecture_from_cache(settings, stores["train"])
        write_json(settings.output_dir / "config_resolved.json", settings.to_dict())
        if args.phase == "teacher_train":
            train_teacher_head(stores["train"], stores["test"], data.train, data.test, settings, device); return 0
        if args.phase == "compare": _compare(settings); return 0
        head = load_head(settings.output_dir / "checkpoints" / "teacher_head.pt", settings, device)
        if args.phase == "teacher_logits": generate_teacher_logits(head, stores, settings, device)
        else: teacher_inference(head, stores["test"], data.test, settings, device)
        return 0
    if args.phase == "input_precompute":
        processor = load_processor(settings.model_id, settings.cache_dir, settings.local_files_only,
                                   settings.processor_min_pixels, settings.processor_max_pixels)
        _input_precompute(processor, data, settings); return 0
    loaded = _load_model(settings, device); settings.resolve_architecture(loaded.model)
    replacement = _replacement(loaded, settings, device)
    write_json(settings.output_dir / "config_resolved.json", settings.to_dict()); _model_report(loaded.model, replacement, settings)
    try:
        if args.phase in {"teacher_precompute", "all"}:
            # Dynamic captions must pass the strict token-row budget before an
            # expensive frozen-teacher forward is allowed to begin.
            _input_precompute(loaded.processor, data, settings)
            _teacher_precompute(loaded, replacement, data, settings, device)
            if args.phase == "teacher_precompute": return 0
        stores = _stores(settings, data)
        if args.phase == "all":
            teacher_head = train_teacher_head(stores["train"], stores["test"], data.train, data.test, settings, device)
            generate_teacher_logits(teacher_head, stores, settings, device)
            teacher_inference(teacher_head, stores["test"], data.test, settings, device)
        inputs = _input_stores(settings, data)
        if args.phase in {"student_train", "all"}:
            head = build_head(settings, settings.text_hidden_size).to(device)
            train_student(loaded.model, replacement, head, data.train, data.test, stores["train"], stores["test"],
                          inputs["train"], inputs["test"], settings, device)
            if args.phase == "student_train": return 0
        if args.phase in {"student_inference", "all"}:
            head = build_head(settings, settings.text_hidden_size).to(device)
            load_student_parts(settings.output_dir, replacement, head, "best")
            loader = make_evaluation_loader(data.test, inputs["test"], settings.inference_batch_size)
            predictions = settings.output_dir / "metrics" / "student_predictions.csv"
            report = evaluate_student(loaded.model, replacement, head, loader, settings, device, data.test, predictions)
            save_student_inference(report, settings, replacement, predictions)
            if args.phase == "student_inference": return 0
        if args.phase == "all": _compare(settings)
        return 0
    finally:
        replacement.close()


def _overrides(settings: Settings, args: argparse.Namespace) -> None:
    if args.device: settings.device = args.device
    if args.cache_dir: settings.cache_dir = resolve_path(args.cache_dir, Path.cwd(), "cache_dir")
    if args.output_dir: settings.output_dir = resolve_path(args.output_dir, Path.cwd(), "output_dir")
    if args.local_files_only: settings.local_files_only = True
    for name in ("epochs", "student_batch_size", "train_samples_per_epoch", "log_interval_batches"):
        value = getattr(args, name)
        if value is not None: setattr(settings, name, value)
    settings.validate()


def _load_model(settings: Settings, device: torch.device) -> LoadedBackbone:
    _log(f"loading {settings.model_id}")
    return load_backbone(settings.model_id, settings.cache_dir, settings.local_files_only, resolve_dtype(settings.dtype),
                         device, settings.attn_implementation, settings.processor_min_pixels, settings.processor_max_pixels)


def _replacement(loaded: LoadedBackbone, settings: Settings, device: torch.device) -> DeepStackMultimodalReplacement:
    vision = VisionDeepStackHomogeneousMoE(settings.vision_hidden_size, settings).to(device)
    language = LanguageDeepStackHomogeneousMoE(settings.text_hidden_size, settings).to(device)
    return DeepStackMultimodalReplacement(loaded.model, vision, language, settings.student_language_mode)


def _teacher_precompute(loaded: LoadedBackbone, replacement: Any, data: DatasetBundle,
                        settings: Settings, device: torch.device) -> None:
    for split, dataset in (("train", data.train), ("test", data.test)):
        loader = make_indexed_loader(dataset, settings.feature_batch_size, settings.num_workers, False, settings.seed)
        build_teacher_cache(split, loaded.model, loaded.processor, replacement, loader, len(dataset), settings, device)


def _input_precompute(processor: Any, data: DatasetBundle, settings: Settings) -> None:
    for split, dataset in (("train", data.train), ("test", data.test)):
        loader = make_indexed_loader(dataset, settings.feature_batch_size, settings.num_workers, False, settings.seed)
        build_processor_cache(split, processor, loader, len(dataset), settings)


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
    write_json(settings.output_dir / "model.json", {
        "model_id": settings.model_id, "task": "binary image-text matching",
        "student_language_mode": settings.student_language_mode,
        "vision_depth": settings.vision_depth, "vision_hidden_size": settings.vision_hidden_size,
        "text_depth": settings.text_depth, "text_hidden_size": settings.text_hidden_size,
        "deepstack_visual_indexes": list(settings.deepstack_visual_indexes or []),
        "vision_tap_stages": list(settings.vision_tap_stages), "native_deepstack_preserved": True,
        "vision": vision, "language": language if settings.student_language_mode == "optical_moe" else {"frozen_electronic": True},
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
