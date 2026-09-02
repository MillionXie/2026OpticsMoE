from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

from experiments.qwen3_vl_embedding_2b_lsp_pose_optical_moe16.datasets import (
    prepare_lsp,
)

from .modeling import (
    build_router_student,
    load_vision_backbone,
    materialize_common_initialization,
)
from .protocol import persist_protocol, split_train_development
from .settings import load_settings, save_resolved_config
from .training import evaluate_sealed_test, train


PHASES = ("prepare_data", "materialize_initialization", "train", "evaluate")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _device(requested: str) -> torch.device:
    if requested.startswith("cuda") and not torch.cuda.is_available():
        print("WARNING: CUDA unavailable; falling back to CPU", flush=True)
        return torch.device("cpu")
    return torch.device(requested)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LSP robust Vision2 electronic/optical Router ablation"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--phase", required=True, choices=PHASES)
    parser.add_argument("--checkpoint", type=Path, default=None)
    args = parser.parse_args(argv)
    if args.phase == "evaluate" and args.checkpoint is None:
        parser.error("--phase evaluate requires an explicit --checkpoint")
    if args.phase != "evaluate" and args.checkpoint is not None:
        parser.error("--checkpoint is only valid with --phase evaluate")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    settings = load_settings(args.config)
    seed_everything(settings.initialization_seed)
    save_resolved_config(settings)
    # Do not persist the legacy two-way train/test CSV: this experiment owns a
    # three-way train/development/sealed-test contract and writes only that
    # authoritative manifest below.
    official = prepare_lsp(settings, persist=False)
    bundle = split_train_development(
        official,
        development_count=settings.development_count,
        seed=settings.development_seed,
    )
    persist_protocol(bundle, settings.output_dir)
    print(
        f"LSP Router train={len(bundle.train):,} dev={len(bundle.development):,} "
        f"sealed_test={len(bundle.test):,} backend={settings.router_backend} "
        f"top_k={settings.top_k}",
        flush=True,
    )
    if args.phase == "prepare_data":
        return 0

    device = _device(settings.device)
    print(f"loading frozen {settings.model_id} on {device}", flush=True)
    loaded = load_vision_backbone(settings, device)
    if args.phase == "materialize_initialization":
        # This command is run once with E2. Router state is deliberately
        # excluded, so E1/E2/E4/O2 consume exactly the same body and head.
        model = build_router_student(loaded, settings)
        try:
            report = materialize_common_initialization(
                model, settings, settings.common_initialization_checkpoint
            )
        finally:
            model.restore_native()
        print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)
        return 0
    if args.phase == "train":
        report = train(loaded, bundle, settings)
    else:
        print(
            "[sealed_test] This explicit command evaluates the official last "
            "1000 LSP images once. Do not use the result to choose another epoch.",
            flush=True,
        )
        report = evaluate_sealed_test(
            loaded, bundle, settings, args.checkpoint
        )
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)
    return 0


__all__ = ["main", "parse_args", "seed_everything"]
