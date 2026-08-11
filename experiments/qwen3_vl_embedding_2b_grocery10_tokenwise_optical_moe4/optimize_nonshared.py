"""Reproducible Grocery31 -> Grocery10 optimization pipeline for nonshared experts."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch


EXPERIMENT = "experiments.qwen3_vl_embedding_2b_grocery10_tokenwise_optical_moe4"


def _run(command: list[str], *, dry_run: bool) -> None:
    print("+ " + " ".join(command), flush=True)
    if not dry_run:
        subprocess.run(command, check=True)


def _checkpoint_epoch(path: Path) -> int:
    if not path.is_file():
        return 0
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return int(payload.get("epoch", 0))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    experiment_root = Path(__file__).resolve().parent
    pretrain_config = experiment_root / "configs/optimization/nonshared_grocery31_pretrain.yaml"
    finetune_config = experiment_root / "configs/optimization/nonshared_grocery10_finetune.yaml"
    pretrain_output = experiment_root / "runs/optimization/nonshared_grocery31_pretrain"
    finetune_output = experiment_root / "runs/optimization/nonshared_grocery10_from31_finetune"
    pretrain_checkpoint = pretrain_output / "checkpoints/ema_last_checkpoint.pt"
    pretrain_last = pretrain_output / "checkpoints/last_checkpoint.pt"
    finetune_last = finetune_output / "checkpoints/last_checkpoint.pt"
    final_checkpoint = (
        finetune_output
        / "checkpoints/ema_best_observed_test_top1_checkpoint.pt"
    )
    started = time.time()
    stages: list[dict[str, object]] = []

    pretrain_epoch = _checkpoint_epoch(pretrain_last)
    pretrain_skipped = pretrain_epoch >= 40 and pretrain_checkpoint.is_file() and not args.force
    if not pretrain_skipped:
        command = [
                sys.executable,
                "-m",
                EXPERIMENT,
                "--config",
                str(pretrain_config),
                "--phase",
                "all",
            ]
        if pretrain_epoch > 0 and not args.force:
            command.extend(["--resume-checkpoint", str(pretrain_last)])
        _run(command, dry_run=args.dry_run)
    stages.append(
        {
            "stage": "grocery31_pretrain",
            "config": str(pretrain_config),
            "checkpoint": str(pretrain_checkpoint),
            "skipped": pretrain_skipped,
        }
    )

    finetune_epoch = _checkpoint_epoch(finetune_last)
    finetune_skipped = finetune_epoch >= 40 and final_checkpoint.is_file() and not args.force
    if not finetune_skipped:
        if not args.dry_run and not pretrain_checkpoint.is_file():
            raise FileNotFoundError(
                f"Pretraining checkpoint was not created: {pretrain_checkpoint}"
            )
        command = [
                sys.executable,
                "-m",
                EXPERIMENT,
                "--config",
                str(finetune_config),
                "--phase",
                "all",
            ]
        if finetune_epoch > 0 and not args.force:
            command.extend(["--resume-checkpoint", str(finetune_last)])
        else:
            command.extend(["--initialize-checkpoint", str(pretrain_checkpoint)])
        _run(command, dry_run=args.dry_run)
    stages.append(
        {
            "stage": "grocery10_finetune",
            "config": str(finetune_config),
            "checkpoint": str(final_checkpoint),
            "skipped": finetune_skipped,
        }
    )

    summary = {
        "pipeline": "nonshared_tokenwise_grocery31_to_grocery10",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.time() - started,
        "dry_run": args.dry_run,
        "stages": stages,
        "changes_from_baseline": [
            "31-SKU packaged-product pretraining",
            "P=10 K=2 full-negative contrastive batches",
            "8x pointwise teacher KD",
            "pairwise relational embedding KD",
            "fixed teacher-gallery prototype cross-entropy",
            "EMA weights",
            "stronger target augmentation",
            "3% block phase dropout during target fine-tuning",
            "weights-only stage transfer with fresh optimizer",
        ],
    }
    if not args.dry_run:
        finetune_output.mkdir(parents=True, exist_ok=True)
        (finetune_output / "optimization_pipeline.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
