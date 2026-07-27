from __future__ import annotations

import argparse
from pathlib import Path

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
from .training import evaluate, train


PHASES = {"prepare_data", "train", "evaluate", "all"}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Electronic CNN baselines for the fixed Grocery-10 retrieval task"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--phase", choices=sorted(PHASES), default="all")
    parser.add_argument("--checkpoint", default=None)
    args = parser.parse_args()

    settings = load_settings(args.config)
    seed_everything(settings.random_seed)
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(settings.output_dir / "resolved_config.json", settings.to_dict())
    write_json(settings.output_dir / "environment.json", environment_report())
    bundle = prepare_grocery_subset(settings, persist=True)
    if args.phase == "prepare_data":
        print(
            f"Prepared Grocery-{len(bundle.class_names)} for {settings.model_name}: "
            f"train={len(bundle.train_samples)} test={len(bundle.test_samples)} "
            f"gallery={len(bundle.gallery_samples)}"
        )
        return 0

    device = torch.device(
        settings.device
        if settings.device != "cuda" or torch.cuda.is_available()
        else "cpu"
    )
    model = build_model(settings, device)
    report = parameter_report(model)
    report["device"] = str(device)
    report["selected_skus"] = list(bundle.class_names)
    report["manifest_sha256"] = bundle.manifest_digest
    write_json(settings.output_dir / "model.json", report)
    print(
        f"{settings.model_name}: parameters={report['parameters']:,} "
        f"trainable={report['trainable_parameters']:,} "
        f"feature_dim={report['feature_dim']} embedding_dim={report['embedding_dim']}"
    )
    if args.phase in {"train", "all"}:
        train(model, bundle, settings, device)
        if args.phase == "train":
            return 0
    if args.phase in {"evaluate", "all"}:
        result = evaluate(
            model,
            bundle,
            settings,
            device,
            (
                Path(args.checkpoint).expanduser().resolve()
                if args.checkpoint
                else None
            ),
        )
        metrics = result.metrics
        print(
            f"{settings.model_name} test: "
            f"Top-1={metrics['top1_retrieval_accuracy']:.4f} "
            f"Top-3={metrics['top3_retrieval_accuracy']:.4f} "
            f"MRR={metrics['mrr']:.4f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
