from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import torch

from experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval.io_utils import (
    environment_report,
    seed_everything,
    write_json,
)
from experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval.prepare_grocery_retrieval_subset import (
    prepare_grocery_subset,
)

from .modeling import build_model, parameter_report
from .settings import load_settings
from .training import evaluate_and_save, train


PHASES = {"prepare_data", "train", "evaluate", "all"}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Direct ten-class electronic CNN baselines for Grocery-10"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--phase", choices=sorted(PHASES), default="all")
    parser.add_argument("--checkpoint", default=None)
    args = parser.parse_args(argv)

    settings = load_settings(args.config)
    seed_everything(settings.random_seed)
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(settings.output_dir / "resolved_config.json", settings.to_dict())
    write_json(settings.output_dir / "environment.json", environment_report())
    bundle = prepare_grocery_subset(settings, persist=True)
    write_json(settings.output_dir / "dataset.json", bundle.metadata)
    if args.phase == "prepare_data":
        print(
            f"Prepared Grocery-10 direct classification: "
            f"train={len(bundle.train_samples)} test={len(bundle.test_samples)}",
            flush=True,
        )
        return 0

    device = torch.device(
        settings.device
        if settings.device != "cuda" or torch.cuda.is_available()
        else "cpu"
    )
    model = build_model(settings, device)
    report = parameter_report(model)
    report.update(
        {
            "device": str(device),
            "selected_skus": list(bundle.class_names),
            "manifest_sha256": bundle.manifest_digest,
        }
    )
    write_json(settings.output_dir / "model.json", report)
    print(
        f"{settings.model_name} direct classifier: "
        f"parameters={report['parameters']:,} "
        f"trainable={report['trainable_parameters']:,} "
        f"output=[B,10], similarity_matching=false",
        flush=True,
    )
    if args.phase in {"train", "all"}:
        train(model, bundle, settings, device)
        if args.phase == "train":
            return 0
    if args.phase in {"evaluate", "all"}:
        metrics = evaluate_and_save(
            model,
            bundle,
            settings,
            device,
            Path(args.checkpoint).expanduser().resolve()
            if args.checkpoint
            else None,
        )
        print(
            f"{settings.model_name} direct test: "
            f"Top-1={metrics['top1_accuracy']:.4f} "
            f"Top-3={metrics['top3_accuracy']:.4f} "
            f"macro-F1={metrics['macro_f1']:.4f}",
            flush=True,
        )
    return 0
