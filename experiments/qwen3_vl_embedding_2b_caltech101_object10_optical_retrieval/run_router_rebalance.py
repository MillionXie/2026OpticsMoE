"""Continue the fixed epoch-50 EMA model with explicit Top-2 load balancing."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


MODULE = "experiments.qwen3_vl_embedding_2b_caltech101_object10_optical_retrieval"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--checkpoint", default=None)
    args = parser.parse_args()
    experiment = Path(__file__).resolve().parent
    repository = experiment.parents[1]
    config = experiment / "configs" / "caltech101_10class_router_rebalance.yaml"
    checkpoint = (
        Path(args.checkpoint).expanduser().resolve()
        if args.checkpoint
        else experiment
        / "runs"
        / "caltech101_10class_finetune_epoch50"
        / "ema_last_checkpoint.pt"
    )
    output = experiment / "runs" / "caltech101_10class_router_rebalance_epoch55"
    commands = [
        [
            sys.executable,
            "-m",
            MODULE,
            "--config",
            str(config),
            "--phase",
            "cache_teacher_embeddings",
        ],
        [
            sys.executable,
            "-m",
            MODULE,
            "--config",
            str(config),
            "--phase",
            "train",
            "--resume-checkpoint",
            str(checkpoint),
        ],
        [
            sys.executable,
            "-m",
            MODULE,
            "--config",
            str(config),
            "--phase",
            "evaluate",
            "--checkpoint",
            str(output / "ema_last_checkpoint.pt"),
        ],
        [
            sys.executable,
            "-m",
            f"{MODULE}.export_paper_analysis",
            "--config",
            str(config),
            "--checkpoint",
            str(output / "ema_last_checkpoint.pt"),
            "--output-dir",
            str(output / "routing_analysis"),
        ],
    ]
    if not args.dry_run and not checkpoint.is_file():
        raise FileNotFoundError(f"Fixed epoch-50 EMA checkpoint is missing: {checkpoint}")
    for command in commands:
        print("+ " + " ".join(command), flush=True)
        if not args.dry_run:
            subprocess.run(command, cwd=repository, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
