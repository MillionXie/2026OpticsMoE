from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Any

import torch

from .datasets import DatasetBundle, load_spaq, make_loader
from .data_prepare import ensure_spaq_dataset
from .features import cache_metadata, extract_and_cache, load_feature_cache
from .io_utils import resolve_device, resolve_dtype, runtime_metadata, set_seed, write_json
from .modeling import MultitaskRegressionHead, load_backbone, model_report
from .settings import Settings, load_settings, normalize_hub_cache_dir, resolve_model_id, resolve_path
from .training import evaluate_test, load_final_head, train_regression_head
from .visualize import save_figures


PHASES = ("download", "prepare_data", "extract", "extract_features", "train", "test", "visualize", "all")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Frozen Qwen3-VL-2B full multimodal SPAQ text-conditioned multitask IQA regression"
        )
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--phase", choices=PHASES, default="all")
    parser.add_argument("--device")
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--model-id")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--local-files-only", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = load_settings(args.config)
    _apply_overrides(settings, args, args.config.resolve().parent)
    settings.validate()
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(settings.seed)
    write_json(settings.output_dir / "resolved_config.json", settings.to_dict())
    requested_device = torch.device(settings.device)
    write_json(settings.output_dir / "environment.json", runtime_metadata(requested_device))
    preparation = ensure_spaq_dataset(settings)
    write_json(settings.output_dir / "data_preparation.json", preparation)
    if args.phase == "download":
        return 0
    data = load_spaq(settings, persist_split=True)
    write_json(settings.output_dir / "dataset.json", data.metadata)
    print(
        f"SPAQ ready: train_images={len(data.train_records)} test_images={len(data.test_records)} "
        f"train_pairs={len(data.train)} test_pairs={len(data.test)} validation=none"
    )
    if args.phase == "prepare_data":
        return 0

    train_path = settings.output_dir / "features" / "train.pt"
    test_path = settings.output_dir / "features" / "test.pt"
    train_metadata = _feature_metadata(settings, data, "train", len(data.train))
    test_metadata = _feature_metadata(settings, data, "test", len(data.test))
    extract_phase = args.phase in {"extract", "extract_features", "all"}
    if extract_phase:
        caches_ready = False
        if settings.cache_features and train_path.is_file() and test_path.is_file():
            load_feature_cache(train_path, train_metadata)
            load_feature_cache(test_path, test_metadata)
            caches_ready = True
            print("reusing validated train and test frozen feature caches; Qwen load skipped")
        if not caches_ready:
            device = resolve_device(settings.device)
            loaded = load_backbone(
                model_id=settings.model_id,
                cache_dir=normalize_hub_cache_dir(settings.cache_dir, settings.model_id),
                local_files_only=settings.local_files_only,
                dtype=resolve_dtype(settings.dtype),
                device=device,
                attn_implementation=settings.attn_implementation,
                min_pixels=settings.processor_min_pixels,
                max_pixels=settings.processor_max_pixels,
            )
            report_head = MultitaskRegressionHead(
                settings.expected_feature_dim,
                settings.head_hidden_dim,
                settings.dropout,
                settings.head_output_activation,
            )
            report = model_report(loaded.model, report_head, settings.expected_feature_dim)
            report.update(
                {
                    "model_id": settings.model_id,
                    "load_time_sec": loaded.load_time_sec,
                    "processor_min_pixels": settings.processor_min_pixels,
                    "processor_max_pixels": settings.processor_max_pixels,
                }
            )
            write_json(settings.output_dir / "model.json", report)
            _extract_or_validate(
                loaded.model, loaded.processor, data.train, train_path, train_metadata, settings, device, "train"
            )
            _extract_or_validate(
                loaded.model, loaded.processor, data.test, test_path, test_metadata, settings, device, "test"
            )
            del loaded
            if device.type == "cuda":
                torch.cuda.empty_cache()
        if args.phase in {"extract", "extract_features"}:
            return 0

    history = None
    predictions = None
    if args.phase in {"train", "all"}:
        device = resolve_device(settings.device)
        train_cache = load_feature_cache(train_path, train_metadata)
        _, history = train_regression_head(train_cache, settings, device)
        if args.phase == "train":
            save_figures(settings.output_dir, history=history, predictions=[])
            return 0

    if args.phase in {"test", "all"}:
        device = resolve_device(settings.device)
        test_cache = load_feature_cache(test_path, test_metadata)
        head = load_final_head(settings, device)
        predictions, metrics = evaluate_test(head, test_cache, settings, device)
        print(
            "SPAQ test macro: "
            f"MAE={metrics['macro_average']['mae']:.4f} "
            f"SRCC={metrics['macro_average']['srcc']:.4f} "
            f"PLCC={metrics['macro_average']['plcc']:.4f}"
        )
        if args.phase == "test":
            save_figures(settings.output_dir, predictions=predictions)
            return 0

    if args.phase in {"visualize", "all"}:
        paths = save_figures(settings.output_dir, history=history, predictions=predictions)
        print(f"saved {len(paths)} figure(s) under {settings.output_dir / 'figures'}")
    return 0


def _extract_or_validate(
    model: torch.nn.Module,
    processor: Any,
    dataset: Any,
    cache_path: Path,
    metadata: dict[str, Any],
    settings: Settings,
    device: torch.device,
    split: str,
) -> None:
    if cache_path.is_file() and settings.cache_features:
        load_feature_cache(cache_path, metadata)
        print(f"reusing validated {split} feature cache: {cache_path}")
        return
    loader = make_loader(
        dataset,
        batch_size=settings.feature_batch_size,
        num_workers=settings.num_workers,
        shuffle=False,
        seed=settings.seed,
    )
    extract_and_cache(
        model=model,
        processor=processor,
        loader=loader,
        device=device,
        split=split,
        cache_path=cache_path,
        expected_metadata=metadata,
        cache_dtype=settings.cache_dtype,
        expected_feature_dim=settings.expected_feature_dim,
        progress=settings.progress,
    )


def _feature_metadata(
    settings: Settings,
    data: DatasetBundle,
    split: str,
    samples: int,
) -> dict[str, Any]:
    records = data.train_records if split == "train" else data.test_records
    record_payload = "\n".join(
        f"{record.image_name}|" + "|".join(f"{key}:{record.scores[key]:.12g}" for key in sorted(record.scores))
        for record in records
    )
    identity = {
        **data.cache_identity,
        "split_image_digest": hashlib.sha256(record_payload.encode("utf-8")).hexdigest(),
    }
    return cache_metadata(
        split=split,
        split_samples=samples,
        model_id=settings.model_id,
        processor_min_pixels=settings.processor_min_pixels,
        processor_max_pixels=settings.processor_max_pixels,
        dtype=settings.dtype,
        attn_implementation=settings.attn_implementation,
        expected_feature_dim=settings.expected_feature_dim,
        dataset_identity=identity,
    )


def _apply_overrides(settings: Settings, args: argparse.Namespace, config_dir: Path) -> None:
    if args.device:
        settings.device = args.device
    if args.cache_dir is not None:
        settings.cache_dir = resolve_path(args.cache_dir, Path.cwd(), "cache_dir")
    if args.output_dir is not None:
        settings.output_dir = resolve_path(args.output_dir, Path.cwd(), "output_dir")
    if args.model_id:
        settings.model_id = resolve_model_id(args.model_id, config_dir)
    if args.epochs is not None:
        settings.epochs = args.epochs
    if args.local_files_only:
        settings.local_files_only = True


if __name__ == "__main__":
    raise SystemExit(main())
