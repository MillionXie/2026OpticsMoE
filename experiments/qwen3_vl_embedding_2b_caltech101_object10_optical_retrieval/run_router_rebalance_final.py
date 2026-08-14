"""One-epoch load calibration followed by five-epoch fixed-router recovery."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


MODULE = "experiments.qwen3_vl_embedding_2b_caltech101_object10_optical_retrieval"


def _run(command: list[str], repository: Path) -> None:
    print("+ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=repository, check=True)


def main() -> int:
    experiment = Path(__file__).resolve().parent
    repository = experiment.parents[1]
    original = experiment / "runs" / "caltech101_10class_finetune_epoch50" / "ema_last_checkpoint.pt"
    calibration_config = experiment / "configs" / "caltech101_10class_router_rebalance_final_calibration.yaml"
    recovery_config = experiment / "configs" / "caltech101_10class_router_rebalance_final_recovery.yaml"
    calibration = experiment / "runs" / "caltech101_10class_router_rebalance_final_calibration_epoch51"
    recovery = experiment / "runs" / "caltech101_10class_router_rebalance_final_epoch56"
    if not original.is_file():
        raise FileNotFoundError(f"Fixed epoch-50 EMA checkpoint is missing: {original}")
    for config in (calibration_config, recovery_config):
        _run([sys.executable, "-m", MODULE, "--config", str(config), "--phase", "cache_teacher_embeddings"], repository)
    _run([sys.executable, "-m", MODULE, "--config", str(calibration_config), "--phase", "train", "--resume-checkpoint", str(original)], repository)
    calibrated = calibration / "last_checkpoint.pt"
    _run([sys.executable, "-m", MODULE, "--config", str(recovery_config), "--phase", "train", "--resume-checkpoint", str(calibrated)], repository)
    final = recovery / "last_checkpoint.pt"
    _run([sys.executable, "-m", MODULE, "--config", str(recovery_config), "--phase", "evaluate", "--checkpoint", str(final)], repository)
    _run([sys.executable, "-m", f"{MODULE}.export_paper_analysis", "--config", str(recovery_config), "--checkpoint", str(final), "--output-dir", str(recovery / "routing_analysis")], repository)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

