from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from .data_prepare import ensure_spaq_dataset
from .datasets import DatasetBundle, load_spaq, make_indexed_loader
from .io_utils import (configure_cpu_runtime, resolve_device, resolve_dtype,
                       runtime_metadata, set_seed, write_json)
from .modeling import LoadedBackbone, build_head, load_backbone, load_processor, module_parameters
from .optics import (DeepStackMultimodalReplacement, LanguageDeepStackHomogeneousMoE,
                     VisionDeepStackHomogeneousMoE)
from .processor_cache import ProcessorCacheStore, build_processor_cache, validate_processor_cache
from .settings import Settings, load_settings, resolve_path
from .teacher_cache import TeacherCacheStore, build_teacher_cache
from .training import (evaluate_student, generate_teacher_predictions, load_head, load_student_parts,
                       make_evaluation_loader, save_student_inference, teacher_inference,
                       train_student, train_teacher_head)


PHASES = ("download", "prepare_data", "input_precompute", "teacher_precompute", "teacher_train",
          "teacher_predictions", "teacher_inference", "student_train", "student_inference", "compare", "all")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "SPAQ four-task single-attribute Qwen3-VL-2B distillation with "
            "electronic top-k amplitude routing and homogeneous optical MoE9"
        )
    )
    parser.add_argument("--config", type=Path, required=True); parser.add_argument("--phase", choices=PHASES, default="all")
    parser.add_argument("--device"); parser.add_argument("--cache-dir", type=Path); parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--local-files-only", action="store_true"); parser.add_argument("--epochs", type=int)
    parser.add_argument("--student-batch-size", type=int); parser.add_argument("--train-samples-per-epoch", type=int)
    parser.add_argument("--log-interval-batches", type=int)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv); settings = load_settings(args.config); _overrides(settings, args)
    cpu_runtime = configure_cpu_runtime(settings.cpu_threads, settings.cpu_interop_threads)
    print(
        f"[runtime] precompute_num_workers={settings.num_workers} "
        f"student_cache_workers=0 cpu_threads={cpu_runtime['torch_cpu_threads']} "
        f"cpu_interop_threads={cpu_runtime['torch_cpu_interop_threads']}",
        flush=True,
    )
    set_seed(settings.seed)
    _dirs(settings.output_dir); preparation = ensure_spaq_dataset(settings); write_json(settings.output_dir / "data_preparation.json", preparation)
    data = load_spaq(settings, persist_split=True); settings.resolved_annotations_file = data.metadata["annotation_file"]
    settings.split_digest = data.metadata["split_digest"]; write_json(settings.output_dir / "dataset.json", data.metadata)
    write_json(settings.output_dir / "config_resolved.json", settings.to_dict())
    if args.phase in {"download", "prepare_data"}:
        print(f"SPAQ ready: task={settings.task_name} train={len(data.train)} test={len(data.test)}", flush=True); return 0
    device = resolve_device(settings.device); write_json(settings.output_dir / "environment.json", runtime_metadata(device))
    if args.phase in {"teacher_train", "teacher_predictions", "teacher_inference", "compare"}:
        stores = _stores(settings, data); _architecture_from_cache(settings, stores["train"])
        if args.phase == "teacher_train": train_teacher_head(stores["train"], stores["test"], data.train, data.test, settings, device); return 0
        if args.phase == "compare": _compare(settings); return 0
        head = load_head(settings.output_dir / "checkpoints" / "teacher_head.pt", settings, device)
        if args.phase == "teacher_predictions": generate_teacher_predictions(head, stores, settings, device)
        else: teacher_inference(head, stores["test"], data.test, settings, device)
        return 0
    if args.phase == "input_precompute":
        processor = load_processor(settings.model_id, settings.cache_dir, settings.local_files_only,
                                   settings.processor_min_pixels, settings.processor_max_pixels)
        _input_precompute(processor, data, settings); return 0
    loaded = _load_model(settings, device); settings.resolve_architecture(loaded.model)
    replacement = _replacement(loaded, settings, device); write_json(settings.output_dir / "config_resolved.json", settings.to_dict())
    _model_report(loaded.model, replacement, settings)
    try:
        if args.phase in {"teacher_precompute", "all"}:
            _teacher_precompute(loaded, replacement, data, settings, device); _input_precompute(loaded.processor, data, settings)
            if args.phase == "teacher_precompute": return 0
        stores = _stores(settings, data)
        if args.phase == "all":
            teacher_head = train_teacher_head(stores["train"], stores["test"], data.train, data.test, settings, device)
            generate_teacher_predictions(teacher_head, stores, settings, device)
        inputs = _input_stores(settings, data)
        if args.phase in {"student_train", "all"}:
            head = build_head(settings, settings.text_hidden_size).to(device)
            train_student(loaded.model, replacement, head, data.train, data.test, stores["train"], stores["test"],
                          inputs["train"], inputs["test"], settings, device)
            if args.phase == "student_train": return 0
        if args.phase in {"student_inference", "all"}:
            head = build_head(settings, settings.text_hidden_size).to(device); load_student_parts(settings.output_dir, replacement, head, "best")
            loader = make_evaluation_loader(data.test, inputs["test"], settings.inference_batch_size)
            predictions = settings.output_dir / "metrics" / "student_predictions.csv"
            report = evaluate_student(loaded.model, replacement, head, loader, settings, device, data.test, predictions)
            save_student_inference(report, settings, replacement, predictions)
            if args.phase == "student_inference": return 0
        if args.phase == "all": _compare(settings)
        return 0
    finally: replacement.close()


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


def _replacement(loaded: LoadedBackbone, settings: Settings, device: torch.device):
    vision = VisionDeepStackHomogeneousMoE(settings.vision_hidden_size, settings).to(device)
    language = LanguageDeepStackHomogeneousMoE(settings.text_hidden_size, settings).to(device)
    return DeepStackMultimodalReplacement(loaded.model, vision, language, settings)


def _teacher_precompute(loaded: LoadedBackbone, replacement: Any, data: DatasetBundle,
                        settings: Settings, device: torch.device):
    for split, dataset in (("train", data.train), ("test", data.test)):
        loader = make_indexed_loader(dataset, settings.feature_batch_size, settings.num_workers, False, settings.seed)
        build_teacher_cache(split, loaded.model, loaded.processor, replacement, loader, len(dataset), settings, device)


def _input_precompute(processor: Any, data: DatasetBundle, settings: Settings):
    for split, dataset in (("train", data.train), ("test", data.test)):
        loader = make_indexed_loader(dataset, settings.feature_batch_size, settings.num_workers, False, settings.seed)
        build_processor_cache(split, processor, loader, len(dataset), settings)


def _stores(settings: Settings, data: DatasetBundle):
    stores = {split: TeacherCacheStore(settings.output_dir / "teacher_cache" / f"{split}.pt",
                                       settings.teacher_cache_lru_shards) for split in ("train", "test")}
    for split, dataset in (("train", data.train), ("test", data.test)):
        expected = {"split": split, "sample_count": len(dataset), "task": settings.task_name,
                    "split_digest": settings.split_digest, "classification_prompt": settings.classification_prompt,
                    "processor_min_pixels": settings.processor_min_pixels, "processor_max_pixels": settings.processor_max_pixels,
                    "replacement_mode": "qwen3_vl_native_deepstack_teacher_targets"}
        changed = [key for key, value in expected.items() if stores[split].metadata.get(key) != value]
        if changed: raise RuntimeError(f"Teacher cache metadata mismatch for {split}: {changed}")
    return stores


def _input_stores(settings: Settings, data: DatasetBundle):
    stores = {split: ProcessorCacheStore(settings.output_dir / "processor_cache" / f"{split}.pt",
                                         settings.teacher_cache_lru_shards) for split in ("train", "test")}
    for split, dataset in (("train", data.train), ("test", data.test)):
        validate_processor_cache(stores[split], split, len(dataset), settings)
    return stores


def _architecture_from_cache(settings: Settings, store: TeacherCacheStore):
    for name in ("vision_depth", "vision_hidden_size", "text_depth", "text_hidden_size"):
        setattr(settings, name, int(store.metadata[name]))
    settings.deepstack_visual_indexes = tuple(store.metadata["deepstack_visual_indexes"])


def _model_report(model: torch.nn.Module, replacement: Any, settings: Settings):
    vision = replacement.vision_surrogate.parameter_breakdown(); language = replacement.language_surrogate.parameter_breakdown()
    head = build_head(settings, settings.text_hidden_size); head_params = module_parameters(head)
    optical_and_attention = {
        id(parameter): parameter for parameter in replacement.trainable_parameters()
        if parameter.requires_grad
    }
    trainable = sum(parameter.numel() for parameter in optical_and_attention.values()) + head_params
    write_json(settings.output_dir / "model.json", {"model_id": settings.model_id, "task": settings.task_name,
        "student_language_mode": settings.student_language_mode, "vision_depth": settings.vision_depth,
        "vision_hidden_size": settings.vision_hidden_size, "text_depth": settings.text_depth,
        "text_hidden_size": settings.text_hidden_size, "deepstack_visual_indexes": list(settings.deepstack_visual_indexes or []),
        "vision_tap_stages": list(settings.vision_tap_stages), "native_deepstack_preserved": True,
        "vision": vision, "language": language if settings.student_language_mode == "optical_moe" else {"frozen_electronic": True},
        "transformer_alignment": replacement.alignment_specification(),
        "physical_router": {
            "implementation": settings.router_implementation,
            "decision_domain": "electronic",
            "amplitude_slm_weight_domain": settings.amplitude_slm_weight_domain,
            "amplitude_slm_input_normalization": settings.amplitude_slm_input_normalization,
            "amplitude_to_phase_relay": settings.amplitude_phase_relay,
            "amplitude_and_phase_planes_coplanar": True,
            "phase_prompt_used": False,
            "interlayer_distance_m": settings.expert_interlayer_distance_m,
            "last_expert_to_global_distance_m": settings.last_expert_to_global_distance_m,
            "global_to_ccd_distance_m": settings.global_to_detector_distance_m,
        },
        "final_detector_layernorm_scope": settings.detector_layernorm_scope,
        "head": head.specification(), "student_trainable_parameters": trainable,
        "qwen_total_parameters": module_parameters(model), "qwen_original_trainable_parameters": 0})


def _compare(settings: Settings):
    teacher = json.loads((settings.output_dir / "metrics" / "teacher_inference.json").read_text())
    student = json.loads((settings.output_dir / "metrics" / "student_inference.json").read_text())
    write_json(settings.output_dir / "metrics" / "comparison.json", {"teacher": teacher, "student": student,
               "student_minus_teacher": {name: student[name] - teacher[name] for name in ("mae", "srcc", "plcc")}})


def _dirs(root: Path):
    for name in ("teacher_cache", "processor_cache", "metrics", "checkpoints", "figures"): (root / name).mkdir(parents=True, exist_ok=True)


def _log(message: str): print(f"[{datetime.now(timezone.utc).isoformat(timespec='seconds')}] {message}", flush=True)


if __name__ == "__main__": raise SystemExit(main())
