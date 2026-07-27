from __future__ import annotations

import argparse
import json
import platform
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval.io_utils import (
    write_json,
)
from experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval.prepare_grocery_retrieval_subset import (
    prepare_grocery_subset,
)

from .modeling import TwoPlaneD2NNClassifier
from .settings import load_settings
from .training import evaluate_and_save, train


PHASES = {"prepare_data", "train", "test", "all"}


def _seed(value: int) -> None:
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    torch.cuda.manual_seed_all(value)


def _environment() -> dict[str, object]:
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "cuda_device_name": (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        ),
    }


def _device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Configuration requests CUDA, but CUDA is unavailable")
    return torch.device(requested)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Two-plane all-optical Grocery-10 D2NN classification baseline"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--phase", choices=sorted(PHASES), default="all")
    arguments = parser.parse_args(argv)
    settings = load_settings(arguments.config)
    _seed(settings.random_seed)
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(settings.output_dir / "resolved_config.json", settings.to_dict())
    write_json(settings.output_dir / "environment.json", _environment())

    bundle = prepare_grocery_subset(settings, persist=True)
    write_json(settings.output_dir / "dataset.json", bundle.metadata)
    write_json(
        settings.output_dir / "metrics" / "comparison_scope.json",
        {
            "same_selected_skus_and_official_split_as_optical_experiment": True,
            "manifest_sha256": bundle.manifest_digest,
            "important_caveat": (
                "This baseline performs closed-set ten-region classification, "
                "whereas the experiment model performs gallery retrieval. "
                "Top-1 values answer related but not identical questions."
            ),
        },
    )
    if arguments.phase == "prepare_data":
        print(
            f"Prepared Grocery-10 train={len(bundle.train_samples)} "
            f"test={len(bundle.test_samples)} digest={bundle.manifest_digest}",
            flush=True,
        )
        return 0

    device = _device(settings.device)
    model = TwoPlaneD2NNClassifier(settings).to(device)
    write_json(settings.output_dir / "model.json", model.parameter_report())
    report = model.parameter_report()
    print(
        f"D2NN2 phase parameters={report['optical_phase_parameters']:,} "
        f"(local={report['first_local_phase_parameters']:,}, "
        f"global={report['second_global_phase_parameters']:,}); "
        "electronic_trainable=0",
        flush=True,
    )
    if arguments.phase in {"train", "all"}:
        train(model, bundle, settings, device)
    if arguments.phase in {"test", "all"}:
        evaluate_and_save(model, bundle, settings, device)
    return 0
