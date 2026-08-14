"""Run the stronger router gate-bias calibration branch."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


MODULE = "experiments.qwen3_vl_embedding_2b_caltech101_object10_optical_retrieval"


def main() -> int:
    experiment = Path(__file__).resolve().parent
    repository = experiment.parents[1]
    config = experiment / "configs" / "caltech101_10class_router_bias_rebalance_strong.yaml"
    checkpoint = experiment / "runs" / "caltech101_10class_finetune_epoch50" / "ema_last_checkpoint.pt"
    output = experiment / "runs" / "caltech101_10class_router_bias_rebalance_strong_epoch53"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Fixed epoch-50 EMA checkpoint is missing: {checkpoint}")
    commands = [
        [sys.executable, "-m", MODULE, "--config", str(config), "--phase", "cache_teacher_embeddings"],
        [sys.executable, "-m", MODULE, "--config", str(config), "--phase", "train", "--resume-checkpoint", str(checkpoint)],
        [sys.executable, "-m", MODULE, "--config", str(config), "--phase", "evaluate", "--checkpoint", str(output / "last_checkpoint.pt")],
        [sys.executable, "-m", f"{MODULE}.export_paper_analysis", "--config", str(config), "--checkpoint", str(output / "last_checkpoint.pt"), "--output-dir", str(output / "routing_analysis")],
    ]
    for command in commands:
        print("+ " + " ".join(command), flush=True)
        subprocess.run(command, cwd=repository, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

