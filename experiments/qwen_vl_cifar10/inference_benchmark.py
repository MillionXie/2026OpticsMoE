from __future__ import annotations

import argparse
import csv
import gc
import importlib.metadata
import subprocess
import time
from pathlib import Path
from typing import Any, Sequence

if __package__ in {None, ""}:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from experiments.qwen_vl_cifar10.config import EXPERIMENT_DIR, parse_args_with_config
from experiments.qwen_vl_cifar10.data import load_cifar10, make_image_loader
from experiments.qwen_vl_cifar10.evaluate import (
    classification_metrics,
    write_confusion_matrix_csv,
    write_predictions_csv,
)
from experiments.qwen_vl_cifar10.generate import parse_class_name
from experiments.qwen_vl_cifar10.inference_profiling import (
    BatchTiming,
    architecture_statistics,
    audit_model_features,
    parameter_statistics,
    summarize_batch_timings,
    timed_processor_components,
)
from experiments.qwen_vl_cifar10.models import (
    SUPPORTED_MODEL_IDS,
    freeze_backbone,
    load_qwen,
    model_input_device,
)
from experiments.qwen_vl_cifar10.progress import log_event, progress_iter, utc_now_iso
from experiments.qwen_vl_cifar10.utils import (
    cuda_peak_memory_by_device_mb,
    cuda_synchronize,
    reset_cuda_peak_memory,
    resolve_device,
    resolve_dtype,
    runtime_metadata,
    set_seed,
    write_json,
)


CLASSIFICATION_PROMPT_TEMPLATE = (
    "Classify this CIFAR-10 image. Reply with exactly one class name from: "
    "{labels}. Do not add any explanation."
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Zero-shot Qwen3-VL CIFAR-10 inference and latency benchmark."
    )
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument(
        "--model-id",
        choices=SUPPORTED_MODEL_IDS,
        default="Qwen/Qwen3-VL-32B-Instruct",
    )
    parser.add_argument("--data-root", type=Path, default=EXPERIMENT_DIR / "data")
    parser.add_argument(
        "--output-dir", type=Path, default=EXPERIMENT_DIR / "runs" / "inference_32b_h20"
    )
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--resize-to", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--test-limit", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--dtype", choices=("bf16", "bfloat16", "fp16", "float16"), default="bf16"
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--device-map", choices=("none", "auto"), default="none")
    parser.add_argument(
        "--attn-implementation",
        choices=("eager", "sdpa", "flash_attention_2"),
        default="sdpa",
    )
    parser.add_argument("--generation-max-new-tokens", type=int, default=8)
    parser.add_argument("--warmup-batches", type=int, default=3)
    parser.add_argument(
        "--min-gpu-memory-gib",
        type=float,
        default=80.0,
        help="Fail before model loading if a single-GPU run exposes less memory.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--download", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--progress", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--audit-features", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--require-all-cuda", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    started_at = utc_now_iso()
    experiment_start = time.perf_counter()
    args = parse_args_with_config(build_parser(), argv)
    _validate_args(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    dtype, dtype_name = resolve_dtype(args.dtype, device)
    device_map = None if args.device_map == "none" else args.device_map
    set_seed(args.seed)
    _validate_gpu_capacity(device, device_map, args.min_gpu_memory_gib)
    config_record = vars(args).copy()
    config_record["data_root"] = str(args.data_root)
    config_record["output_dir"] = str(args.output_dir)
    config_record["resolved_device"] = str(device)
    config_record["resolved_dtype"] = dtype_name
    config_record["resolved_device_map"] = device_map
    write_json(args.output_dir / "config.json", config_record)
    log_event(
        "inference",
        "benchmark started",
        model=args.model_id,
        batch_size=args.batch_size,
    )

    dataset_start = time.perf_counter()
    data = load_cifar10(
        args.data_root,
        args.image_size,
        args.resize_to,
        train_limit=1,
        test_limit=args.test_limit,
        download=args.download,
    )
    dataset_init_sec = time.perf_counter() - dataset_start
    loader = make_image_loader(
        data.test_dataset,
        args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        seed=args.seed,
    )
    log_event(
        "dataset",
        "CIFAR-10 test split ready",
        samples=len(data.test_dataset),
        init_sec=f"{dataset_init_sec:.3f}",
    )

    model_load_start = time.perf_counter()
    loaded = load_qwen(
        args.model_id,
        dtype,
        device,
        device_map,
        args.trust_remote_code,
        attn_implementation=args.attn_implementation,
    )
    freeze_backbone(loaded.model)
    runtime_device = model_input_device(loaded.model, device)
    cuda_synchronize(runtime_device)
    model_load_sec = time.perf_counter() - model_load_start
    model_parameters = parameter_statistics(loaded.model)
    model_architecture = architecture_statistics(loaded.model)
    model_memory_after_load = _cuda_memory_snapshot()
    _validate_model_placement(model_parameters, args.require_all_cuda)
    log_event(
        "model",
        "weights loaded",
        elapsed_sec=f"{model_load_sec:.3f}",
        parameter_gib=f"{model_parameters['parameter_gib']:.3f}",
        runtime_device=runtime_device,
    )

    first_images, _ = next(iter(loader))
    probe_inputs, _ = _prepare_inputs(
        loaded.processor, first_images, data.class_names, runtime_device
    )
    feature_audit: dict[str, Any] = {}
    shape_probe_sec = 0.0
    if args.audit_features:
        probe_start = time.perf_counter()
        feature_audit = audit_model_features(loaded.model, probe_inputs, runtime_device)
        shape_probe_sec = time.perf_counter() - probe_start
        write_json(args.output_dir / "feature_shapes.json", feature_audit)
        log_event(
            "shape_audit",
            "feature dimensions captured",
            feature_dimension=feature_audit.get("feature_dimension"),
            elapsed_sec=f"{shape_probe_sec:.3f}",
        )
    del probe_inputs

    warmup_start = time.perf_counter()
    for _ in range(args.warmup_batches):
        _infer_images(
            loaded.model,
            loaded.processor,
            first_images,
            data.class_names,
            runtime_device,
            args.generation_max_new_tokens,
            batch_index=-1,
            dataset_fetch_sec=0.0,
            end_to_end_start=time.perf_counter(),
        )
    warmup_sec = time.perf_counter() - warmup_start
    gc.collect()
    if runtime_device.type == "cuda":
        torch.cuda.empty_cache()
    reset_cuda_peak_memory(runtime_device)
    log_event(
        "warmup",
        "completed",
        batches=args.warmup_batches,
        elapsed_sec=f"{warmup_sec:.3f}",
    )

    timings: list[BatchTiming] = []
    labels: list[int] = []
    predictions: list[int] = []
    raw_outputs: list[str] = []
    iterator = iter(loader)
    batch_indices = progress_iter(
        range(len(loader)),
        description="CIFAR-10 zero-shot inference",
        enabled=args.progress,
        total=len(loader),
    )
    inference_loop_start = time.perf_counter()
    for batch_index in batch_indices:
        end_to_end_start = time.perf_counter()
        fetch_start = time.perf_counter()
        images, batch_labels = next(iterator)
        dataset_fetch_sec = time.perf_counter() - fetch_start
        outputs, timing = _infer_images(
            loaded.model,
            loaded.processor,
            images,
            data.class_names,
            runtime_device,
            args.generation_max_new_tokens,
            batch_index=batch_index,
            dataset_fetch_sec=dataset_fetch_sec,
            end_to_end_start=end_to_end_start,
        )
        timings.append(timing)
        batch_targets = batch_labels.tolist()
        labels.extend(batch_targets)
        raw_outputs.extend(outputs)
        predictions.extend(
            parse_class_name(value, data.class_names) for value in outputs
        )
    cuda_synchronize(runtime_device)
    inference_loop_sec = time.perf_counter() - inference_loop_start

    evaluation = classification_metrics(labels, predictions, data.class_names)
    artifact_start = time.perf_counter()
    write_predictions_csv(
        args.output_dir / "predictions.csv",
        evaluation.labels,
        evaluation.predictions,
        data.class_names,
        raw_outputs,
    )
    write_confusion_matrix_csv(
        args.output_dir / "confusion_matrix.csv",
        evaluation.confusion_matrix,
        data.class_names,
    )
    _write_batch_timings(args.output_dir / "batch_timings.csv", timings)
    artifact_write_sec = time.perf_counter() - artifact_start

    peak_by_device = cuda_peak_memory_by_device_mb(runtime_device)
    timing_summary = summarize_batch_timings(timings)
    finished_at = utc_now_iso()
    total_wall_sec = time.perf_counter() - experiment_start
    metrics: dict[str, Any] = {
        "schema_version": 1,
        "task": "zero_shot_cifar10_generation",
        "model_id": args.model_id,
        "dtype": dtype_name,
        "device_map": args.device_map,
        "attention_implementation": args.attn_implementation,
        "test_samples": len(labels),
        "batch_size": args.batch_size,
        "dataset": {
            "name": "CIFAR-10",
            "split": "test",
            "root": str(args.data_root),
            "image_size": args.image_size,
            "resize_to": args.resize_to,
            "class_names": data.class_names,
        },
        "generation": {
            "prompt_template": CLASSIFICATION_PROMPT_TEMPLATE,
            "max_new_tokens": args.generation_max_new_tokens,
            "do_sample": False,
            "use_cache": True,
        },
        "accuracy": evaluation.accuracy,
        "macro_f1": evaluation.macro_f1,
        "per_class_accuracy": evaluation.per_class_accuracy,
        "unparsed_predictions": sum(value < 0 for value in predictions),
        "input_tokens": sum(row.input_tokens for row in timings),
        "generated_tokens": sum(row.generated_tokens for row in timings),
        "feature_dimension": feature_audit.get("feature_dimension"),
        "probe_timings": {
            "vision_encoder_forward_sec": feature_audit.get(
                "vision_encoder_forward_sec", 0.0
            ),
            "multimodal_prefill_forward_sec": feature_audit.get(
                "multimodal_prefill_forward_sec", 0.0
            ),
        },
        "parameter_statistics": model_parameters,
        "model_architecture": model_architecture,
        "model_memory_after_load_mb": model_memory_after_load,
        "runtime": runtime_metadata(runtime_device),
        "software_versions": _software_versions(),
        "git": _git_info(Path(__file__).resolve().parents[2]),
        "cuda_peak_memory_by_device_mb": peak_by_device,
        "cuda_peak_memory_max_mb": max(peak_by_device.values(), default=0.0),
        "cuda_peak_memory_total_mb": sum(peak_by_device.values()),
        "timing_summary": timing_summary,
        "experiment_timing": {
            "started_at_utc": started_at,
            "finished_at_utc": finished_at,
            "total_wall_sec": total_wall_sec,
            "dataset_initialization_sec": dataset_init_sec,
            "model_load_sec": model_load_sec,
            "shape_probe_sec": shape_probe_sec,
            "warmup_sec": warmup_sec,
            "measured_inference_loop_sec": inference_loop_sec,
            "artifact_write_sec": artifact_write_sec,
        },
        "timing_protocol": _timing_protocol(args),
        "seed": args.seed,
    }
    write_json(args.output_dir / "inference_metrics.json", metrics)

    from experiments.qwen_vl_cifar10.visualize_inference import write_inference_figure

    visualization_start = time.perf_counter()
    write_inference_figure(
        metrics, timings, feature_audit, args.output_dir / "inference_summary.png"
    )
    metrics["experiment_timing"]["visualization_sec"] = (
        time.perf_counter() - visualization_start
    )
    write_json(args.output_dir / "inference_metrics.json", metrics)
    log_event(
        "inference",
        "benchmark completed",
        accuracy=f"{evaluation.accuracy:.6f}",
        model_ms_per_image=f"{timing_summary['model_generate_sec']['mean_ms_per_image']:.3f}",
        e2e_ms_per_image=f"{timing_summary['end_to_end_sec']['mean_ms_per_image']:.3f}",
        output=args.output_dir,
    )
    return 0


def _prepare_inputs(
    processor: Any,
    images: Sequence[Any],
    class_names: Sequence[str],
    device: torch.device,
) -> tuple[dict[str, Any], dict[str, float | int]]:
    prompt_start = time.perf_counter()
    conversations = _build_conversations(images, class_names)
    prompt_build_sec = time.perf_counter() - prompt_start

    with timed_processor_components(processor) as (image_timer, tokenizer_timer):
        processor_start = time.perf_counter()
        inputs = processor.apply_chat_template(
            conversations,
            tokenize=True,
            add_generation_prompt=True,
            padding=True,
            return_dict=True,
            return_tensors="pt",
        )
        processor_total_sec = time.perf_counter() - processor_start
    processor_framework_sec = max(
        0.0, processor_total_sec - image_timer.total_sec - tokenizer_timer.total_sec
    )
    attention_mask = inputs.get("attention_mask")
    input_tokens = (
        int(attention_mask.sum().item())
        if torch.is_tensor(attention_mask)
        else int(torch.as_tensor(inputs["input_ids"]).numel())
    )

    cuda_synchronize(device)
    transfer_start = time.perf_counter()
    moved = {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in inputs.items()
    }
    moved.pop("token_type_ids", None)
    cuda_synchronize(device)
    transfer_sec = time.perf_counter() - transfer_start
    return moved, {
        "prompt_build_sec": prompt_build_sec,
        "image_preprocess_sec": image_timer.total_sec,
        "tokenizer_sec": tokenizer_timer.total_sec,
        "processor_framework_sec": processor_framework_sec,
        "processor_total_sec": processor_total_sec,
        "host_to_device_sec": transfer_sec,
        "input_tokens": input_tokens,
    }


def _infer_images(
    model: torch.nn.Module,
    processor: Any,
    images: Sequence[Any],
    class_names: Sequence[str],
    device: torch.device,
    max_new_tokens: int,
    *,
    batch_index: int,
    dataset_fetch_sec: float,
    end_to_end_start: float,
) -> tuple[list[str], BatchTiming]:
    inference_start = time.perf_counter()
    inputs, stages = _prepare_inputs(processor, images, class_names, device)
    input_length = int(inputs["input_ids"].shape[1])
    input_tokens = int(stages["input_tokens"])

    cuda_synchronize(device)
    model_start = time.perf_counter()
    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
        )
    cuda_synchronize(device)
    model_generate_sec = time.perf_counter() - model_start

    decode_start = time.perf_counter()
    new_tokens = generated[:, input_length:]
    outputs = list(
        processor.batch_decode(
            new_tokens,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
    )
    decode_sec = time.perf_counter() - decode_start
    complete_inference_sec = time.perf_counter() - inference_start
    timing = BatchTiming(
        batch_index=batch_index,
        image_count=len(images),
        input_tokens=input_tokens,
        generated_tokens=int(new_tokens.numel()),
        dataset_fetch_sec=dataset_fetch_sec,
        prompt_build_sec=stages["prompt_build_sec"],
        image_preprocess_sec=stages["image_preprocess_sec"],
        tokenizer_sec=stages["tokenizer_sec"],
        processor_framework_sec=stages["processor_framework_sec"],
        processor_total_sec=stages["processor_total_sec"],
        host_to_device_sec=stages["host_to_device_sec"],
        model_generate_sec=model_generate_sec,
        decode_postprocess_sec=decode_sec,
        complete_inference_sec=complete_inference_sec,
        end_to_end_sec=time.perf_counter() - end_to_end_start,
    )
    return outputs, timing


def _build_conversations(
    images: Sequence[Any], class_names: Sequence[str]
) -> list[list[dict[str, Any]]]:
    labels = ", ".join(class_names)
    prompt = CLASSIFICATION_PROMPT_TEMPLATE.format(labels=labels)
    return [
        [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        for image in images
    ]


def _write_batch_timings(path: Path, rows: Sequence[BatchTiming]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = (
        list(rows[0].as_dict()) if rows else list(BatchTiming.__dataclass_fields__)
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(row.as_dict() for row in rows)


def _validate_model_placement(
    statistics: dict[str, Any], require_all_cuda: bool
) -> None:
    if not require_all_cuda:
        return
    non_cuda = {
        device: values
        for device, values in statistics["by_device"].items()
        if not device.startswith("cuda") and values["parameters"] > 0
    }
    if non_cuda:
        raise RuntimeError(
            "The benchmark requires all parameters on CUDA, but found CPU/disk/meta placement: "
            f"{non_cuda}. Use a larger GPU or explicitly disable --require-all-cuda and report "
            "offload as a different experimental condition."
        )


def _validate_gpu_capacity(
    device: torch.device, device_map: str | None, minimum_gib: float
) -> None:
    if device.type != "cuda" or minimum_gib <= 0:
        return
    capacities = [
        torch.cuda.get_device_properties(index).total_memory / (1024**3)
        for index in range(torch.cuda.device_count())
    ]
    if device_map is not None:
        available = sum(capacities)
    else:
        index = (
            device.index if device.index is not None else torch.cuda.current_device()
        )
        available = capacities[index] if index < len(capacities) else 0.0
    if available < minimum_gib:
        scope = (
            "visible GPUs combined"
            if device_map is not None
            else "selected CUDA device"
        )
        raise RuntimeError(
            f"Qwen3-VL-32B BF16 preflight requires at least {minimum_gib:.1f} GiB on the "
            f"{scope}, but only {available:.1f} GiB is visible. Check whether the H20 is a "
            "full 96 GB device or a smaller vGPU/MIG slice."
        )


def _timing_protocol(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "clock": "time.perf_counter (monotonic wall time)",
        "cuda_synchronization": "all visible CUDA devices before and after GPU stages",
        "warmup_batches": args.warmup_batches,
        "warmup_excluded_from_measured_distribution": True,
        "dataset_fetch": "DataLoader next(); num_workers=0 recommended for attributable timing",
        "prompt_build": "Python construction of per-image chat conversations",
        "image_preprocess": "actual processor.image_processor calls",
        "tokenizer": "actual processor.tokenizer calls",
        "processor_framework": "chat template, multimodal placeholder expansion, and framework overhead",
        "processor_total": "processor.apply_chat_template(..., tokenize=True) wall time",
        "host_to_device": "all tensor transfers to the model input device",
        "model_generate": "model.generate only; excludes processor, tokenizer, transfer, and decode",
        "decode_postprocess": "new-token slicing, device-to-host conversion, and batch_decode",
        "complete_inference": "prompt construction through decoded class strings; excludes dataset fetch",
        "end_to_end": "dataset fetch through decoded class strings",
        "shape_probe_excluded_from_measured_distribution": True,
        "vision_encoder_forward_probe": (
            "one synchronized get_image_features call on the first batch"
        ),
        "multimodal_prefill_forward_probe": (
            "one synchronized backbone forward with use_cache=False on the first batch"
        ),
        "artifact_and_visualization_time_excluded_from_end_to_end": True,
    }


def _cuda_memory_snapshot() -> dict[str, dict[str, float]]:
    if not torch.cuda.is_available():
        return {}
    return {
        f"cuda:{index}": {
            "allocated_mb": torch.cuda.memory_allocated(index) / (1024**2),
            "reserved_mb": torch.cuda.memory_reserved(index) / (1024**2),
        }
        for index in range(torch.cuda.device_count())
    }


def _software_versions() -> dict[str, str | None]:
    result: dict[str, str | None] = {}
    for package in ("torch", "torchvision", "transformers", "accelerate", "pillow"):
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result[package] = None
    return result


def _git_info(root: Path) -> dict[str, Any]:
    def run(*args: str) -> str | None:
        try:
            return subprocess.check_output(
                ["git", *args], cwd=root, stderr=subprocess.DEVNULL, text=True
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    status = run("status", "--porcelain")
    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(status),
    }


def _validate_args(args: argparse.Namespace) -> None:
    positive = {
        "image_size": args.image_size,
        "batch_size": args.batch_size,
        "generation_max_new_tokens": args.generation_max_new_tokens,
    }
    invalid = [name for name, value in positive.items() if value <= 0]
    if invalid:
        raise ValueError(f"These arguments must be positive: {', '.join(invalid)}")
    if args.test_limit is not None and args.test_limit <= 0:
        raise ValueError("--test-limit must be positive when provided.")
    if args.resize_to is not None and args.resize_to <= 0:
        raise ValueError("--resize-to must be positive when provided.")
    if args.warmup_batches < 0:
        raise ValueError("--warmup-batches cannot be negative.")


if __name__ == "__main__":
    raise SystemExit(main())
