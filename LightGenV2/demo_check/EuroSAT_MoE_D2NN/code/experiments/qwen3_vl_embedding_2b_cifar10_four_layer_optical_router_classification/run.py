from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval.io_utils import (
    environment_report,
    seed_everything,
    write_json,
)

from .data import prepare_cifar10
from .modeling import (
    architecture_report,
    build_classification_student,
    load_backbone,
    load_backbone_only_warmstart,
)
from .settings import load_settings, save_resolved_config
from .training import evaluate, gradient_check, load_checkpoint, train


PHASES = {"prepare_data", "check", "train", "evaluate"}


def _device(settings: Any) -> torch.device:
    requested = str(settings.device)
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("The configuration requires CUDA, but CUDA is unavailable")
    return torch.device(requested)


def _initialization_report(settings: Any, replacement: Any) -> dict[str, Any]:
    if settings.classification_initialization == "warmstart_body":
        return load_backbone_only_warmstart(settings, replacement)
    return {
        "mode": "random_compact_student_and_fresh_cifar10_head",
        "reason": "explicit classification.initialization=random",
        "router_optimization_seed": int(settings.router_optimization_seed),
        "classification_head_seed": int(settings.classification_head_seed),
        "source_checkpoint_loaded": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="CIFAR-10 classification with the four-stage routed optical Qwen student"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--phase", choices=sorted(PHASES), required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--resume-checkpoint", default=None)
    parser.add_argument("--check-batch-size", type=int, default=2)
    args = parser.parse_args()

    if args.phase == "evaluate" and args.checkpoint is None:
        parser.error("--phase evaluate requires --checkpoint")
    if args.phase != "evaluate" and args.checkpoint is not None:
        parser.error("--checkpoint is only valid for --phase evaluate")
    if args.resume_checkpoint is not None and args.phase != "train":
        parser.error("--resume-checkpoint is only valid for --phase train")

    settings = load_settings(args.config)
    seed_everything(settings.router_optimization_seed)
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    save_resolved_config(settings)
    environment = environment_report()
    environment["task"] = "cifar10_classification"
    write_json(settings.output_dir / "environment.json", environment)
    bundle = prepare_cifar10(settings, persist=True)
    print(
        f"[data] counts={bundle.metadata['counts']} "
        f"split_sha256={bundle.metadata['split_sha256']}",
        flush=True,
    )
    if args.phase == "prepare_data":
        return 0

    loaded = load_backbone(settings, _device(settings))
    print(
        f"[backbone] model={settings.model_id} device={loaded.device} "
        f"load_sec={loaded.load_time_sec:.2f}",
        flush=True,
    )
    replacement, head = build_classification_student(loaded, settings)
    try:
        write_json(
            settings.output_dir / "student_architecture.json",
            architecture_report(replacement, head, settings),
        )
        if args.phase == "evaluate":
            load_checkpoint(
                Path(args.checkpoint).expanduser().resolve(),
                replacement,
                head,
                bundle,
                settings,
            )
            result = evaluate(
                loaded,
                replacement,
                head,
                bundle.test,
                settings,
                split="test",
            )
            write_json(settings.output_dir / "evaluation_metrics.json", result)
            return 0

        resume = (
            Path(args.resume_checkpoint).expanduser().resolve()
            if args.resume_checkpoint
            else None
        )
        if resume is None:
            report = _initialization_report(settings, replacement)
            write_json(
                settings.output_dir / "classification_initialization_report.json",
                report,
            )
            print(f"[initialization] {json.dumps(report, ensure_ascii=False)}", flush=True)

        if args.phase == "check":
            result = gradient_check(
                loaded,
                replacement,
                head,
                bundle,
                settings,
                batch_size=args.check_batch_size,
            )
            write_json(settings.output_dir / "gradient_check.json", result)
            return 0

        train(
            loaded,
            replacement,
            head,
            bundle,
            settings,
            resume_checkpoint=resume,
        )
    finally:
        replacement.close()
    return 0


__all__ = ["main"]
