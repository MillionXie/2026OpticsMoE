from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

try:
    import torch
except (ImportError, OSError) as exc:
    raise RuntimeError(
        "PyTorch could not be imported. Install a torch/torchvision build compatible with your "
        "Python, operating system, and CUDA runtime before running this experiment."
    ) from exc

from experiments.qwen_vl_cifar10.benchmark import BenchmarkResult, benchmark_callable
from experiments.qwen_vl_cifar10.config import EXPERIMENT_DIR, parse_args_with_config
from experiments.qwen_vl_cifar10.data import (
    CIFAR10Data,
    load_cifar10,
    make_image_loader,
)
from experiments.qwen_vl_cifar10.evaluate import (
    EvaluationResult,
    classification_metrics,
    write_confusion_matrix_csv,
    write_predictions_csv,
)
from experiments.qwen_vl_cifar10.features import (
    FEATURE_SOURCES,
    cache_metadata,
    extract_dataset_features,
    extract_feature_batch,
    load_feature_cache,
    save_feature_cache,
)
from experiments.qwen_vl_cifar10.generate import generate_batch, run_generation
from experiments.qwen_vl_cifar10.models import (
    DEFAULT_MODEL_ID,
    SUPPORTED_MODEL_IDS,
    MLPHead,
    apply_lora,
    freeze_backbone,
    load_qwen,
    model_input_device,
)
from experiments.qwen_vl_cifar10.progress import log_event, utc_now_iso
from experiments.qwen_vl_cifar10.train import train_lora_classifier, train_mlp_head
from experiments.qwen_vl_cifar10.utils import (
    cuda_peak_memory_by_device_mb,
    cuda_peak_memory_mb,
    cuda_synchronize,
    reset_cuda_peak_memory,
    resolve_device,
    resolve_dtype,
    runtime_metadata,
    set_seed,
    write_json,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train/evaluate Qwen3-VL CIFAR-10 classifiers without modifying the existing project."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="JSON config file. CLI arguments override config values.",
    )
    parser.add_argument("--mode", choices=("mlp", "generate", "lora"), default="mlp")
    parser.add_argument(
        "--model-id", choices=SUPPORTED_MODEL_IDS, default=DEFAULT_MODEL_ID
    )
    parser.add_argument("--data-root", type=Path, default=EXPERIMENT_DIR / "data")
    parser.add_argument(
        "--output-dir", type=Path, default=EXPERIMENT_DIR / "runs" / "qwen_vl_cifar10"
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=32,
        help="Unmodified dataset image size; this option does not resize CIFAR-10.",
    )
    parser.add_argument(
        "--resize-to",
        type=int,
        default=None,
        help="Optional explicit PIL resize before the Qwen processor (disabled by default).",
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument(
        "--feature-batch-size",
        type=int,
        default=None,
        help="Batch size for Qwen image feature extraction; defaults to --batch-size.",
    )
    parser.add_argument(
        "--head-batch-size",
        type=int,
        default=None,
        help="Batch size for cached MLP-head training; defaults to --batch-size.",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument(
        "--feature-source", choices=FEATURE_SOURCES, default="visual_tokens_mean"
    )
    parser.add_argument(
        "--cache-features", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--dtype",
        choices=("float32", "fp32", "float16", "fp16", "bfloat16", "bf16"),
        default="bf16",
    )
    parser.add_argument("--device", default="cuda", help="cpu, cuda, cuda:N, or auto")
    parser.add_argument(
        "--device-map",
        choices=("none", "auto"),
        default="none",
        help="Use 'auto' with accelerate for large/multi-GPU models.",
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup-batches", type=int, default=10)
    parser.add_argument("--benchmark-batches", type=int, default=50)
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--test-limit", type=int, default=None)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument(
        "--progress",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show tqdm batch progress and timestamped epoch records.",
    )
    parser.add_argument(
        "--download", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--generation-max-new-tokens", type=int, default=8)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora-target-modules",
        default="q_proj,k_proj,v_proj,o_proj",
        help="Comma-separated selected modules. mm_mlp/projector targets are rejected.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    experiment_started_at_utc = utc_now_iso()
    experiment_start = time.perf_counter()
    args = parse_args_with_config(build_parser(), argv)
    _resolve_batch_sizes(args)
    _validate_args(args)
    _apply_smoke_settings(args)
    if args.mode == "lora" and args.feature_source in {
        "visual_tokens_mean",
        "vision_pooler",
    }:
        warnings.warn(
            "LoRA adapters on language attention modules require a multimodal feature path; "
            "using multimodal_image_tokens_mean.",
            stacklevel=2,
        )
        args.feature_source = "multimodal_image_tokens_mean"
    if args.mode == "lora" and args.cache_features:
        raise ValueError(
            "--cache-features is incompatible with LoRA because features change during tuning."
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    requested_device = resolve_device(args.device)
    dtype, dtype_name = resolve_dtype(args.dtype, requested_device)
    device_map = None if args.device_map == "none" else args.device_map
    set_seed(args.seed)
    reset_cuda_peak_memory(requested_device)
    log_event(
        "experiment",
        "started",
        mode=args.mode,
        model=args.model_id,
        device=requested_device,
        feature_batch=args.feature_batch_size,
        head_batch=args.head_batch_size,
    )

    config = _config_dict(args, requested_device, dtype_name, device_map)
    write_json(args.output_dir / "config.json", config)

    # Dataset construction/download is reported separately from model execution.
    log_event("dataset", "loading CIFAR-10", root=args.data_root)
    dataset_start = time.perf_counter()
    data = load_cifar10(
        args.data_root,
        args.image_size,
        args.resize_to,
        args.train_limit,
        args.test_limit,
        download=args.download,
    )
    dataset_load_time = time.perf_counter() - dataset_start
    log_event(
        "dataset",
        "ready",
        train_samples=len(data.train_dataset),
        test_samples=len(data.test_dataset),
        elapsed_sec=f"{dataset_load_time:.3f}",
    )

    if args.mode == "mlp":
        metrics = _run_mlp(args, data, requested_device, dtype, dtype_name, device_map)
    elif args.mode == "generate":
        metrics = _run_generate(
            args, data, requested_device, dtype, dtype_name, device_map
        )
    else:
        metrics = _run_lora(args, data, requested_device, dtype, dtype_name, device_map)

    experiment_finished_at_utc = utc_now_iso()
    total_wall_time = time.perf_counter() - experiment_start
    _attach_research_timing(
        metrics,
        args,
        requested_device,
        experiment_started_at_utc,
        experiment_finished_at_utc,
        dataset_load_time,
        total_wall_time,
    )
    write_json(args.output_dir / "metrics.json", metrics)
    write_json(args.output_dir / "timing.json", metrics["timing"])
    from experiments.qwen_vl_cifar10.visualize_results import write_run_figure

    write_run_figure(metrics, args.output_dir / "run_summary.png")
    log_event(
        "experiment",
        "completed",
        accuracy=f"{metrics['accuracy']:.6f}",
        total_wall_sec=f"{total_wall_time:.3f}",
        metrics=args.output_dir / "metrics.json",
    )
    return 0


def _run_mlp(
    args: argparse.Namespace,
    data: CIFAR10Data,
    requested_device: torch.device,
    dtype: torch.dtype,
    dtype_name: str,
    device_map: str | None,
) -> dict[str, Any]:
    train_metadata = cache_metadata(
        args.model_id,
        args.feature_source,
        args.image_size,
        args.resize_to,
        dtype_name,
        "train",
        len(data.train_dataset),
        data.class_names,
    )
    test_metadata = cache_metadata(
        args.model_id,
        args.feature_source,
        args.image_size,
        args.resize_to,
        dtype_name,
        "test",
        len(data.test_dataset),
        data.class_names,
    )
    train_cache_path = args.output_dir / "train_features.pt"
    test_cache_path = args.output_dir / "test_features.pt"
    train_cached = (
        load_feature_cache(train_cache_path, train_metadata)
        if args.cache_features
        else None
    )
    test_cached = (
        load_feature_cache(test_cache_path, test_metadata)
        if args.cache_features
        else None
    )
    both_cached = train_cached is not None and test_cached is not None
    log_event(
        "feature_cache",
        "lookup complete",
        train_hit=train_cached is not None,
        test_hit=test_cached is not None,
    )

    train_loader = make_image_loader(
        data.train_dataset, args.feature_batch_size, False, args.num_workers, args.seed
    )
    test_loader = make_image_loader(
        data.test_dataset, args.feature_batch_size, False, args.num_workers, args.seed
    )
    model: torch.nn.Module | None = None
    processor: Any = None
    runtime_device = requested_device
    model_load_time = 0.0
    benchmark = BenchmarkResult(0.0, 0.0, 0.0)
    extracted_images = 0
    feature_elapsed = 0.0
    train_feature_elapsed = 0.0
    test_feature_elapsed = 0.0

    if not both_cached:
        log_event("model_load", "loading Qwen weights", model=args.model_id)
        cuda_synchronize(requested_device)
        load_start = time.perf_counter()
        loaded = load_qwen(
            args.model_id, dtype, requested_device, device_map, args.trust_remote_code
        )
        freeze_backbone(loaded.model)
        runtime_device = model_input_device(loaded.model, requested_device)
        cuda_synchronize(runtime_device)
        model_load_time = time.perf_counter() - load_start
        model, processor = loaded.model, loaded.processor
        log_event(
            "model_load",
            "model ready",
            elapsed_sec=f"{model_load_time:.3f}",
            runtime_device=runtime_device,
        )

        benchmark_images, _ = next(iter(test_loader))

        def feature_operation() -> int:
            assert model is not None
            extract_feature_batch(
                model, processor, benchmark_images, args.feature_source, runtime_device
            )
            return len(benchmark_images)

        benchmark = benchmark_callable(
            feature_operation, runtime_device, warmup_batches=0, benchmark_batches=0
        )

    if train_cached is not None:
        train_features, train_labels = train_cached
    else:
        assert model is not None
        train_result = extract_dataset_features(
            model,
            processor,
            train_loader,
            args.feature_source,
            runtime_device,
            description="train feature extraction",
            show_progress=args.progress,
        )
        train_features, train_labels = train_result.features, train_result.labels
        feature_elapsed += train_result.elapsed_sec
        train_feature_elapsed = train_result.elapsed_sec
        extracted_images += train_result.image_count
        if args.cache_features:
            save_feature_cache(train_cache_path, train_result, train_metadata)
            log_event("feature_cache", "saved train features", path=train_cache_path)

    if test_cached is not None:
        test_features, test_labels = test_cached
    else:
        assert model is not None
        test_result = extract_dataset_features(
            model,
            processor,
            test_loader,
            args.feature_source,
            runtime_device,
            description="test feature extraction",
            show_progress=args.progress,
        )
        test_features, test_labels = test_result.features, test_result.labels
        feature_elapsed += test_result.elapsed_sec
        test_feature_elapsed = test_result.elapsed_sec
        extracted_images += test_result.image_count
        if args.cache_features:
            save_feature_cache(test_cache_path, test_result, test_metadata)
            log_event("feature_cache", "saved test features", path=test_cache_path)

    if train_features.ndim != 2 or test_features.ndim != 2:
        raise RuntimeError("Expected 2D [samples, feature_dim] feature tensors.")
    if train_features.shape[1] != test_features.shape[1]:
        raise RuntimeError("Train/test feature dimensions do not match.")
    feature_dim = int(train_features.shape[1])
    head = MLPHead(feature_dim, args.hidden_dim, args.dropout)
    log_event(
        "head_training",
        "training cached-feature MLP",
        feature_dim=feature_dim,
        batch_size=args.head_batch_size,
        epochs=args.epochs,
    )
    training = train_mlp_head(
        head,
        train_features,
        train_labels,
        test_features,
        test_labels,
        data.class_names,
        runtime_device,
        args.head_batch_size,
        args.epochs,
        args.learning_rate,
        args.weight_decay,
        args.seed,
        show_progress=args.progress,
    )
    _save_head(args.output_dir / "best_head.pt", head, args, feature_dim)

    if model is not None:
        benchmark_images, _ = next(iter(test_loader))

        def end_to_end_operation() -> int:
            features = extract_feature_batch(
                model, processor, benchmark_images, args.feature_source, runtime_device
            )
            with torch.inference_mode():
                head(features).argmax(dim=-1)
            return len(benchmark_images)

        end_to_end = benchmark_callable(
            end_to_end_operation,
            runtime_device,
            args.warmup_batches,
            args.benchmark_batches,
        )
        benchmark = BenchmarkResult(
            benchmark.first_batch_latency_sec,
            end_to_end.steady_state_latency_ms_per_image,
            end_to_end.steady_state_images_per_second,
            warmup_time_sec=end_to_end.warmup_time_sec,
            measurement_time_sec=end_to_end.measurement_time_sec,
            measured_images=end_to_end.measured_images,
            total_time_sec=benchmark.total_time_sec + end_to_end.total_time_sec,
        )
        benchmark_scope = "qwen_feature_extraction_plus_head"
    else:
        cached_batch = test_features[: args.head_batch_size].to(runtime_device)
        head.to(runtime_device).eval()

        def cached_head_operation() -> int:
            with torch.inference_mode():
                head(cached_batch).argmax(dim=-1)
            return len(cached_batch)

        benchmark = benchmark_callable(
            cached_head_operation,
            runtime_device,
            args.warmup_batches,
            args.benchmark_batches,
        )
        benchmark_scope = "head_only_cache_hit"

    evaluation = training.evaluation
    _write_evaluation_outputs(args.output_dir, evaluation, data.class_names)
    return _metrics(
        args,
        data,
        dtype_name,
        runtime_device,
        evaluation,
        model_load_time,
        feature_elapsed,
        extracted_images,
        training.elapsed_sec,
        training.images_per_second,
        benchmark,
        training.train_loss,
        extra={
            "feature_dim": feature_dim,
            "feature_cache_hit": both_cached,
            "train_feature_cache_hit": train_cached is not None,
            "test_feature_cache_hit": test_cached is not None,
            "benchmark_scope": benchmark_scope,
            "training_history": training.history,
            "train_feature_extraction_sec": train_feature_elapsed,
            "test_feature_extraction_sec": test_feature_elapsed,
            "evaluation_total_sec": training.evaluation_elapsed_sec,
        },
    )


def _run_generate(
    args: argparse.Namespace,
    data: CIFAR10Data,
    requested_device: torch.device,
    dtype: torch.dtype,
    dtype_name: str,
    device_map: str | None,
) -> dict[str, Any]:
    test_loader = make_image_loader(
        data.test_dataset, args.feature_batch_size, False, args.num_workers, args.seed
    )
    log_event("model_load", "loading Qwen weights", model=args.model_id)
    cuda_synchronize(requested_device)
    load_start = time.perf_counter()
    loaded = load_qwen(
        args.model_id, dtype, requested_device, device_map, args.trust_remote_code
    )
    freeze_backbone(loaded.model)
    runtime_device = model_input_device(loaded.model, requested_device)
    cuda_synchronize(runtime_device)
    model_load_time = time.perf_counter() - load_start
    log_event("model_load", "model ready", elapsed_sec=f"{model_load_time:.3f}")
    benchmark_images, _ = next(iter(test_loader))

    def generation_operation() -> int:
        generate_batch(
            loaded.model,
            loaded.processor,
            benchmark_images,
            data.class_names,
            runtime_device,
            args.generation_max_new_tokens,
        )
        return len(benchmark_images)

    benchmark = benchmark_callable(
        generation_operation,
        runtime_device,
        args.warmup_batches,
        args.benchmark_batches,
    )
    generated = run_generation(
        loaded.model,
        loaded.processor,
        test_loader,
        data.class_names,
        runtime_device,
        args.generation_max_new_tokens,
        show_progress=args.progress,
    )
    evaluation = classification_metrics(
        generated.labels, generated.predictions, data.class_names
    )
    _write_evaluation_outputs(
        args.output_dir, evaluation, data.class_names, raw_outputs=generated.raw_outputs
    )
    torch.save(
        {"mode": "generate", "head_state_dict": None, "model_id": args.model_id},
        args.output_dir / "best_head.pt",
    )
    return _metrics(
        args,
        data,
        dtype_name,
        runtime_device,
        evaluation,
        model_load_time,
        0.0,
        0,
        0.0,
        0.0,
        benchmark,
        None,
        extra={
            "benchmark_scope": "qwen_generation",
            "generation_total_sec": generated.elapsed_sec,
            "generation_images_per_second": (
                generated.image_count / generated.elapsed_sec
                if generated.elapsed_sec > 0
                else 0.0
            ),
            "unparsed_predictions": sum(value < 0 for value in generated.predictions),
        },
    )


def _run_lora(
    args: argparse.Namespace,
    data: CIFAR10Data,
    requested_device: torch.device,
    dtype: torch.dtype,
    dtype_name: str,
    device_map: str | None,
) -> dict[str, Any]:
    train_loader = make_image_loader(
        data.train_dataset, args.feature_batch_size, True, args.num_workers, args.seed
    )
    test_loader = make_image_loader(
        data.test_dataset, args.feature_batch_size, False, args.num_workers, args.seed
    )
    log_event("model_load", "loading Qwen weights for LoRA", model=args.model_id)
    cuda_synchronize(requested_device)
    load_start = time.perf_counter()
    loaded = load_qwen(
        args.model_id, dtype, requested_device, device_map, args.trust_remote_code
    )
    freeze_backbone(loaded.model)
    targets = [
        value.strip() for value in args.lora_target_modules.split(",") if value.strip()
    ]
    model = apply_lora(
        loaded.model, targets, args.lora_r, args.lora_alpha, args.lora_dropout
    )
    runtime_device = model_input_device(model, requested_device)
    cuda_synchronize(runtime_device)
    model_load_time = time.perf_counter() - load_start
    log_event("model_load", "LoRA model ready", elapsed_sec=f"{model_load_time:.3f}")

    benchmark_images, _ = next(iter(test_loader))
    cuda_synchronize(runtime_device)
    first_start = time.perf_counter()
    probe_features = extract_feature_batch(
        model, loaded.processor, benchmark_images, args.feature_source, runtime_device
    )
    cuda_synchronize(runtime_device)
    first_batch_latency = time.perf_counter() - first_start
    feature_dim = int(probe_features.shape[-1])
    head = MLPHead(feature_dim, args.hidden_dim, args.dropout)
    training = train_lora_classifier(
        model,
        loaded.processor,
        head,
        train_loader,
        test_loader,
        args.feature_source,
        data.class_names,
        runtime_device,
        args.epochs,
        args.learning_rate,
        args.weight_decay,
        len(data.train_dataset),
        show_progress=args.progress,
    )
    _save_head(args.output_dir / "best_head.pt", head, args, feature_dim)
    torch.save(
        {
            "model_id": args.model_id,
            "target_modules": targets,
            "state_dict": training.trainable_model_state,
        },
        args.output_dir / "best_lora_adapter.pt",
    )

    def end_to_end_operation() -> int:
        features = extract_feature_batch(
            model,
            loaded.processor,
            benchmark_images,
            args.feature_source,
            runtime_device,
        )
        with torch.inference_mode():
            head(features).argmax(dim=-1)
        return len(benchmark_images)

    steady = benchmark_callable(
        end_to_end_operation,
        runtime_device,
        args.warmup_batches,
        args.benchmark_batches,
    )
    benchmark = BenchmarkResult(
        first_batch_latency,
        steady.steady_state_latency_ms_per_image,
        steady.steady_state_images_per_second,
        warmup_time_sec=steady.warmup_time_sec,
        measurement_time_sec=steady.measurement_time_sec,
        measured_images=steady.measured_images,
        total_time_sec=first_batch_latency + steady.total_time_sec,
    )
    evaluation = training.evaluation
    _write_evaluation_outputs(args.output_dir, evaluation, data.class_names)
    return _metrics(
        args,
        data,
        dtype_name,
        runtime_device,
        evaluation,
        model_load_time,
        0.0,
        0,
        training.elapsed_sec,
        training.images_per_second,
        benchmark,
        training.train_loss,
        extra={
            "feature_dim": feature_dim,
            "benchmark_scope": "qwen_lora_plus_head",
            "lora_target_modules": targets,
            "training_history": training.history,
            "evaluation_total_sec": training.evaluation_elapsed_sec,
        },
    )


def _metrics(
    args: argparse.Namespace,
    data: CIFAR10Data,
    dtype_name: str,
    device: torch.device,
    evaluation: EvaluationResult,
    model_load_time: float,
    feature_elapsed: float,
    extracted_images: int,
    train_elapsed: float,
    train_images_per_second: float,
    benchmark: BenchmarkResult,
    train_loss: float | None,
    extra: dict[str, Any],
) -> dict[str, Any]:
    peak_memory_by_device = cuda_peak_memory_by_device_mb(device)
    return {
        "model_id": args.model_id,
        "mode": args.mode,
        "feature_source": args.feature_source,
        "image_size": args.image_size,
        "resize_to": args.resize_to,
        "train_samples": len(data.train_dataset),
        "test_samples": len(data.test_dataset),
        "accuracy": evaluation.accuracy,
        "macro_f1": evaluation.macro_f1,
        "per_class_accuracy": evaluation.per_class_accuracy,
        "train_loss": train_loss,
        "eval_loss": evaluation.loss,
        "model_load_time_sec": model_load_time,
        "feature_extraction_total_sec": feature_elapsed,
        "feature_extraction_images_per_second": (
            extracted_images / feature_elapsed if feature_elapsed > 0 else 0.0
        ),
        "head_train_total_sec": train_elapsed,
        "head_train_images_per_second": train_images_per_second,
        "end_to_end_latency_ms_per_image": benchmark.steady_state_latency_ms_per_image,
        "end_to_end_images_per_second": benchmark.steady_state_images_per_second,
        "first_batch_latency_sec": benchmark.first_batch_latency_sec,
        "steady_state_latency_ms_per_image": benchmark.steady_state_latency_ms_per_image,
        "benchmark_warmup_time_sec": benchmark.warmup_time_sec,
        "benchmark_measurement_time_sec": benchmark.measurement_time_sec,
        "benchmark_measured_images": benchmark.measured_images,
        "benchmark_total_time_sec": benchmark.total_time_sec,
        "cuda_peak_memory_mb": cuda_peak_memory_mb(device),
        "cuda_peak_memory_total_mb": sum(peak_memory_by_device.values()),
        "cuda_peak_memory_by_device_mb": peak_memory_by_device,
        "dtype": dtype_name,
        "device": str(device),
        "batch_size": args.batch_size,
        "feature_batch_size": args.feature_batch_size,
        "head_batch_size": args.head_batch_size,
        "epochs": args.epochs,
        "seed": args.seed,
        **extra,
    }


def _save_head(
    path: Path, head: MLPHead, args: argparse.Namespace, feature_dim: int
) -> None:
    torch.save(
        {
            "state_dict": {
                key: value.detach().cpu() for key, value in head.state_dict().items()
            },
            "feature_dim": feature_dim,
            "hidden_dim": args.hidden_dim,
            "dropout": args.dropout,
            "num_classes": 10,
            "model_id": args.model_id,
            "feature_source": args.feature_source,
            "mode": args.mode,
        },
        path,
    )


def _write_evaluation_outputs(
    output_dir: Path,
    evaluation: EvaluationResult,
    class_names: list[str],
    raw_outputs: list[str] | None = None,
) -> None:
    write_predictions_csv(
        output_dir / "predictions.csv",
        evaluation.labels,
        evaluation.predictions,
        class_names,
        raw_outputs,
    )
    write_confusion_matrix_csv(
        output_dir / "confusion_matrix.csv", evaluation.confusion_matrix, class_names
    )


def _config_dict(
    args: argparse.Namespace,
    device: torch.device,
    dtype_name: str,
    device_map: str | None,
) -> dict[str, Any]:
    values = vars(args).copy()
    values["data_root"] = str(args.data_root)
    values["output_dir"] = str(args.output_dir)
    values["resolved_device"] = str(device)
    values["resolved_dtype"] = dtype_name
    values["device_map"] = device_map
    values["tune_mm_mlp"] = False
    return values


def _resolve_batch_sizes(args: argparse.Namespace) -> None:
    if args.feature_batch_size is None:
        args.feature_batch_size = args.batch_size
    if args.head_batch_size is None:
        args.head_batch_size = args.batch_size


def _attach_research_timing(
    metrics: dict[str, Any],
    args: argparse.Namespace,
    device: torch.device,
    started_at_utc: str,
    finished_at_utc: str,
    dataset_load_time_sec: float,
    total_wall_time_sec: float,
) -> None:
    stages = {
        "dataset_load_sec": dataset_load_time_sec,
        "model_load_sec": float(metrics.get("model_load_time_sec", 0.0)),
        "train_feature_extraction_sec": float(
            metrics.get("train_feature_extraction_sec", 0.0)
        ),
        "test_feature_extraction_sec": float(
            metrics.get("test_feature_extraction_sec", 0.0)
        ),
        "feature_extraction_total_sec": float(
            metrics.get("feature_extraction_total_sec", 0.0)
        ),
        "head_or_adapter_train_sec": float(metrics.get("head_train_total_sec", 0.0)),
        "evaluation_total_sec": float(metrics.get("evaluation_total_sec", 0.0)),
        "generation_total_sec": float(metrics.get("generation_total_sec", 0.0)),
        "benchmark_total_sec": float(metrics.get("benchmark_total_time_sec", 0.0)),
    }
    timing = {
        "schema_version": 1,
        "experiment_started_at_utc": started_at_utc,
        "experiment_finished_at_utc": finished_at_utc,
        "total_wall_time_sec": total_wall_time_sec,
        "stages": stages,
        "benchmark": {
            "first_batch_latency_sec": metrics.get("first_batch_latency_sec", 0.0),
            "steady_state_latency_ms_per_image": metrics.get(
                "steady_state_latency_ms_per_image", 0.0
            ),
            "steady_state_images_per_second": metrics.get(
                "end_to_end_images_per_second", 0.0
            ),
            "warmup_batches": args.warmup_batches,
            "measured_batches": args.benchmark_batches,
            "warmup_time_sec": metrics.get("benchmark_warmup_time_sec", 0.0),
            "measurement_time_sec": metrics.get("benchmark_measurement_time_sec", 0.0),
            "measured_images": metrics.get("benchmark_measured_images", 0),
            "total_time_sec": metrics.get("benchmark_total_time_sec", 0.0),
            "scope": metrics.get("benchmark_scope"),
        },
        "methodology": {
            "duration_clock": "time.perf_counter (monotonic)",
            "timestamp_clock": "UTC wall clock",
            "cuda_synchronized_for_gpu_stages": True,
            "dataset_time_includes_download_and_dataset_construction": True,
            "model_load_time_includes_processor_and_weight_loading": True,
            "feature_time_excludes_cache_serialization": True,
            "training_time_excludes_per_epoch_evaluation": True,
            "total_wall_time_includes_dataset_model_training_evaluation_and_benchmark": True,
            "visualization_time_excluded": True,
        },
    }
    metrics["experiment_started_at_utc"] = started_at_utc
    metrics["experiment_finished_at_utc"] = finished_at_utc
    metrics["total_wall_time_sec"] = total_wall_time_sec
    metrics["dataset_load_time_sec"] = dataset_load_time_sec
    metrics["runtime"] = runtime_metadata(device)
    metrics["timing"] = timing


def _apply_smoke_settings(args: argparse.Namespace) -> None:
    if not args.smoke_test:
        return
    args.train_limit = min(args.train_limit or 32, 32)
    args.test_limit = min(args.test_limit or 32, 32)
    args.epochs = min(args.epochs, 1)
    args.warmup_batches = min(args.warmup_batches, 1)
    args.benchmark_batches = min(args.benchmark_batches, 2)


def _validate_args(args: argparse.Namespace) -> None:
    choices = {
        "mode": ({"mlp", "generate", "lora"}, args.mode),
        "model_id": (set(SUPPORTED_MODEL_IDS), args.model_id),
        "feature_source": (set(FEATURE_SOURCES), args.feature_source),
        "dtype": (
            {"float32", "fp32", "float16", "fp16", "bfloat16", "bf16"},
            args.dtype,
        ),
        "device_map": ({"none", "auto"}, args.device_map),
    }
    invalid_choices = [
        f"{name}={value!r}"
        for name, (allowed, value) in choices.items()
        if value not in allowed
    ]
    if invalid_choices:
        raise ValueError(f"Invalid config choices: {', '.join(invalid_choices)}")

    positive = {
        "image_size": args.image_size,
        "batch_size": args.batch_size,
        "feature_batch_size": args.feature_batch_size,
        "head_batch_size": args.head_batch_size,
        "epochs": args.epochs,
        "hidden_dim": args.hidden_dim,
        "generation_max_new_tokens": args.generation_max_new_tokens,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
    }
    invalid = [name for name, value in positive.items() if value <= 0]
    if invalid:
        raise ValueError(f"These arguments must be positive: {', '.join(invalid)}")
    if args.resize_to is not None and args.resize_to <= 0:
        raise ValueError("--resize-to must be positive when provided.")
    if args.warmup_batches < 0 or args.benchmark_batches < 0:
        raise ValueError("Benchmark batch counts cannot be negative.")
    if not 0 <= args.dropout < 1 or not 0 <= args.lora_dropout < 1:
        raise ValueError("Dropout values must be in [0, 1).")


if __name__ == "__main__":
    raise SystemExit(main())
