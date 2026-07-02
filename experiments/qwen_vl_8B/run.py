from __future__ import annotations

import argparse
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from .benchmark import benchmark_inference
from .datasets import DatasetBundle, load_dataset, make_loader
from .features import extract_and_cache, load_feature_cache
from .io_utils import resolve_device, resolve_dtype, runtime_metadata, set_seed, write_json
from .modeling import MLPHead, LoadedBackbone, load_backbone, parameter_report
from .settings import Settings, load_settings
from .training import load_head, train_head
from .visualize import generate_figures


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Qwen3-VL-8B frozen-feature MLP classifier")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--phase", choices=("all", "extract", "train", "inference", "visualize"), default="all"
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--cache-dir", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = load_settings(args.config)
    for name in ("output_dir", "data_root", "cache_dir"):
        value = getattr(args, name)
        if value is not None:
            setattr(settings, name, value.expanduser().resolve())
    settings.validate()
    output_dir = settings.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "config_resolved.json", settings.to_dict())
    if args.phase == "visualize":
        _log("visualization started")
        generate_figures(output_dir)
        _log(f"visualization finished: {output_dir / 'figures'}")
        return 0

    set_seed(settings.seed)
    device = resolve_device(settings.device)
    write_json(output_dir / "environment.json", runtime_metadata(device))
    _log(f"dataset loading started: {settings.dataset}")
    dataset_started = time.perf_counter()
    data = _load_data(settings)
    dataset_time = time.perf_counter() - dataset_started
    dataset_record = dict(data.metadata)
    dataset_record["initialization_time_sec"] = dataset_time
    write_json(output_dir / "dataset.json", dataset_record)
    _log(f"dataset ready: train={len(data.train)} test={len(data.test)} classes={len(data.class_names)}")

    loaded: LoadedBackbone | None = None
    if args.phase in {"all", "extract", "inference"}:
        _log(f"model loading started: {settings.model_id}")
        loaded = load_backbone(
            settings.model_id,
            settings.cache_dir,
            settings.local_files_only,
            resolve_dtype(settings.dtype),
            device,
            settings.attn_implementation,
            settings.processor_min_pixels,
            settings.processor_max_pixels,
        )
        model_record = parameter_report(loaded.model)
        model_record["model_id"] = settings.model_id
        model_record["load_time_sec"] = loaded.load_time_sec
        write_json(output_dir / "model.json", model_record)
        _log(f"model ready: load_time={loaded.load_time_sec:.3f}s device={device}")

    train_features: torch.Tensor | None = None
    train_labels: torch.Tensor | None = None
    test_features: torch.Tensor | None = None
    test_labels: torch.Tensor | None = None
    if args.phase in {"all", "extract", "train"}:
        train_features, train_labels = _features_for_split(
            "train", data, settings, loaded, device
        )
        test_features, test_labels = _features_for_split(
            "test", data, settings, loaded, device
        )
        if args.phase == "extract":
            return 0

    head: torch.nn.Module | None = None
    training_report: dict[str, Any] | None = None
    if args.phase in {"all", "train"}:
        assert train_features is not None and train_labels is not None
        assert test_features is not None and test_labels is not None
        head = MLPHead(
            int(train_features.shape[1]), settings.hidden_dim, len(data.class_names), settings.dropout
        )
        _log(f"MLP training started: epochs={settings.epochs} feature_dim={train_features.shape[1]}")
        head, training_report = train_head(
            head,
            train_features,
            train_labels,
            test_features,
            test_labels,
            data.class_names,
            device,
            output_dir,
            settings.head_batch_size,
            settings.epochs,
            settings.validation_fraction,
            settings.learning_rate,
            settings.weight_decay,
            settings.seed,
            settings.progress,
        )
        if loaded is not None:
            report = parameter_report(loaded.model, head)
            report["model_id"] = settings.model_id
            report["load_time_sec"] = loaded.load_time_sec
            write_json(output_dir / "model.json", report)
        if args.phase == "train":
            generate_figures(output_dir)
            _log(f"training finished: {output_dir}")
            return 0

    if args.phase == "inference":
        head, _ = load_head(output_dir / "checkpoints" / "best_mlp.pt", device)
        assert loaded is not None
        report = parameter_report(loaded.model, head)
        report["model_id"] = settings.model_id
        report["load_time_sec"] = loaded.load_time_sec
        write_json(output_dir / "model.json", report)
    if args.phase in {"all", "inference"}:
        assert loaded is not None and head is not None
        test_loader = make_loader(
            data.test,
            settings.inference_batch_size,
            settings.num_workers,
            False,
            settings.seed,
        )
        _log("synchronized end-to-end inference benchmark started")
        inference_report = benchmark_inference(
            loaded.model,
            loaded.processor,
            head,
            test_loader,
            data.class_names,
            device,
            output_dir,
            settings.warmup_batches,
            settings.benchmark_batches,
            settings.progress,
        )
        summary = {
            "dataset": settings.dataset,
            "model_id": settings.model_id,
            "feature_dimension": inference_report["feature_shapes"].get("pooled_features", [None])[-1],
            "metrics": inference_report["metrics"],
            "timing": inference_report["timing"],
            "training": {
                "best_epoch": training_report.get("best_epoch") if training_report else None,
                "best_validation_top1_accuracy": (
                    training_report.get("best_validation_top1_accuracy") if training_report else None
                ),
            },
        }
        write_json(output_dir / "summary.json", summary)
        generate_figures(output_dir)
        _log(
            f"run finished: top1={inference_report['metrics']['top1_accuracy']:.4f} "
            f"output={output_dir}"
        )
    return 0


def _load_data(settings: Settings) -> DatasetBundle:
    return load_dataset(
        settings.dataset,
        settings.data_root,
        settings.download,
        settings.resize_to,
        settings.train_limit,
        settings.test_limit,
        settings.imagefolder_train,
        settings.imagefolder_test,
    )


def _features_for_split(
    split: str,
    data: DatasetBundle,
    settings: Settings,
    loaded: LoadedBackbone | None,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    cache_path = settings.output_dir / "features" / f"{split}.pt"
    dataset = data.train if split == "train" else data.test
    expected = {
        "dataset": settings.dataset,
        "split": split,
        "samples": len(dataset),
        "num_classes": len(data.class_names),
        "model_id": settings.model_id,
        "resize_to": settings.resize_to,
        "processor_min_pixels": settings.processor_min_pixels,
        "processor_max_pixels": settings.processor_max_pixels,
        "pooling": "mean_over_merged_visual_tokens",
    }
    if settings.cache_features and cache_path.is_file():
        features, labels, metadata = load_feature_cache(cache_path)
        if metadata == expected:
            _log(f"reusing {split} feature cache: {cache_path}")
            return features, labels
        _log(f"ignoring incompatible {split} feature cache: {cache_path}")
    if loaded is None:
        raise RuntimeError(f"Missing or incompatible {split} feature cache; run --phase extract first")
    loader = make_loader(
        dataset, settings.feature_batch_size, settings.num_workers, False, settings.seed
    )
    _log(f"extracting {split} visual features: samples={len(dataset)}")
    features, labels, _ = extract_and_cache(
        loaded.model,
        loaded.processor,
        loader,
        device,
        split,
        settings.output_dir,
        expected,
        settings.cache_dtype,
        settings.progress,
    )
    return features, labels


def _log(message: str) -> None:
    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    print(f"[{timestamp}] {message}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
