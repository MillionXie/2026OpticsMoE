from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from .benchmark import benchmark_inference
from .datasets import DatasetBundle, load_dataset, make_loader
from .download import download_checkpoint
from .features import extract_and_cache, load_feature_cache
from .io_utils import resolve_device, resolve_dtype, runtime_metadata, set_seed, write_json
from .modeling import MLPHead, LoadedBackbone, load_backbone, parameter_report
from .settings import Settings, load_settings, resolve_model_id, resolve_path
from .training import load_head, train_head
from .visualize import generate_figures


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Qwen3-VL-8B full multimodal frozen feature + MLP classifier"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--phase",
        choices=("all", "download", "extract", "train", "inference", "visualize"),
        default="all",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--local-files-only", action=argparse.BooleanOptionalAction, default=None
    )
    parser.add_argument("--download-workers", type=int, default=2)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = load_settings(args.config)
    config_path = resolve_path(args.config, Path.cwd(), "config")
    for name in ("output_dir", "data_root", "cache_dir"):
        value = getattr(args, name)
        if value is not None:
            setattr(settings, name, resolve_path(value, Path.cwd(), name))
    if args.model_id is not None:
        settings.model_id = resolve_model_id(args.model_id, config_path.parent)
    if args.device is not None:
        settings.device = args.device
    if args.local_files_only is not None:
        settings.local_files_only = args.local_files_only
    settings.validate()
    output_dir = settings.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    _restore_download_cache(settings, args.phase)
    write_json(output_dir / "config_resolved.json", settings.to_dict())
    if args.phase == "download":
        _log(f"checkpoint download started: {settings.model_id}")
        snapshot = download_checkpoint(
            settings.model_id,
            settings.cache_dir,
            max_workers=args.download_workers,
            disable_xet=True,
        )
        write_json(
            output_dir / "download.json",
            {
                "model_id": settings.model_id,
                "snapshot": str(snapshot),
                "cache_dir": str(settings.cache_dir) if settings.cache_dir else None,
                "xet_disabled": True,
            },
        )
        _log(f"checkpoint download finished: {snapshot}")
        return 0
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
        model_record["feature_path"] = "full_multimodal_answer_position_hidden_state"
        model_record["classification_prompt"] = settings.classification_prompt
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
            report["feature_path"] = "full_multimodal_answer_position_hidden_state"
            report["classification_prompt"] = settings.classification_prompt
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
        report["feature_path"] = "full_multimodal_answer_position_hidden_state"
        report["classification_prompt"] = settings.classification_prompt
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
            settings.classification_prompt,
        )
        summary = {
            "dataset": settings.dataset,
            "model_id": settings.model_id,
            "feature_path": "full_multimodal_answer_position_hidden_state",
            "classification_prompt": settings.classification_prompt,
            "feature_dimension": inference_report["feature_shapes"].get(
                "feature_dimension"
            ),
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
        "classification_prompt": settings.classification_prompt,
        "feature_source": "full_multimodal_answer_position_hidden_state",
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
    _log(f"extracting {split} multimodal features: samples={len(dataset)}")
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
        settings.classification_prompt,
    )
    return features, labels


def _log(message: str) -> None:
    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    print(f"[{timestamp}] {message}", flush=True)


def _restore_download_cache(settings: Settings, phase: str) -> None:
    """Reuse the cache selected by a previous download-only phase."""

    if phase == "download" or not settings.local_files_only or settings.cache_dir is not None:
        return
    record_path = settings.output_dir / "download.json"
    if not record_path.is_file():
        return
    with record_path.open("r", encoding="utf-8") as handle:
        record = json.load(handle)
    cache_dir = record.get("cache_dir")
    if cache_dir:
        candidate = Path(cache_dir)
    else:
        snapshot = record.get("snapshot")
        if not snapshot:
            return
        snapshot_path = Path(snapshot)
        # <cache>/models--org--repo/snapshots/<revision>
        if len(snapshot_path.parents) < 3:
            return
        candidate = snapshot_path.parents[2]
    if candidate.is_dir():
        settings.cache_dir = candidate.resolve()
        _log(f"reusing download cache directory: {settings.cache_dir}")


if __name__ == "__main__":
    raise SystemExit(main())
