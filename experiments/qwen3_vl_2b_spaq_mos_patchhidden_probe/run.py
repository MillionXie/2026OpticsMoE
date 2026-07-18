from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path
from typing import Any

import torch

from experiments.qwen3_vl_2b_spaq_mos_vision_homogeneous_moe9.io_utils import (resolve_device, resolve_dtype,
                                                                               set_seed, write_json)
from experiments.qwen3_vl_2b_spaq_mos_vision_homogeneous_moe9.modeling import load_backbone, module_parameters

from .capture import VisionPatchBypass
from .data import load_probe_data
from .features import extract_patch_features, load_processor_stores
from .settings import Settings, load_settings
from .training import build_head, inference, train_probe


PHASES = ("prepare_data", "extract_features", "train", "test", "all")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="Frozen Qwen3-VL-2B patch-hidden direct MOS regression probe (no optical MoE)"
    )
    result.add_argument("--config", type=Path, required=True)
    result.add_argument("--phase", choices=PHASES, default="all")
    result.add_argument("--device")
    result.add_argument("--epochs", type=int)
    result.add_argument("--output-dir", type=Path)
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    settings = load_settings(args.config)
    if args.device: settings.device = args.device
    if args.epochs is not None: settings.epochs = args.epochs
    if args.output_dir: settings.output_dir = args.output_dir.resolve()
    settings.validate(); set_seed(settings.seed); _dirs(settings.output_dir)
    data, source_settings = load_probe_data(settings)
    write_json(settings.output_dir / "config_resolved.json", settings.to_dict())
    write_json(settings.output_dir / "dataset.json", data.metadata)
    if args.phase == "prepare_data":
        print(f"SPAQ patch probe data ready: train={len(data.train)} test={len(data.test)}", flush=True)
        return 0
    device = resolve_device(settings.device)
    write_json(settings.output_dir / "environment.json", {
        "python": platform.python_version(), "torch": torch.__version__, "device": str(device),
        "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
    })
    if args.phase in {"extract_features", "all"}:
        stores = load_processor_stores(settings, data, source_settings)
        loaded = load_backbone(settings.model_id, settings.cache_dir, settings.local_files_only,
                               resolve_dtype(settings.dtype), device, settings.attn_implementation,
                               settings.processor_min_pixels, settings.processor_max_pixels)
        bypass = VisionPatchBypass(loaded.model)
        try:
            for split, dataset in (("train", data.train), ("test", data.test)):
                extract_patch_features(split, loaded.model, bypass, dataset, stores[split], settings, device)
            feature_dim = int(loaded.model.config.vision_config.hidden_size)
            head = build_head(feature_dim, settings)
            write_json(settings.output_dir / "model.json", {
                "model_id": settings.model_id,
                "student": "frozen Qwen patch embedding -> token mean -> normalized linear MOS head",
                "vision_transformer_used": False, "optical_moe_used": False, "language_model_used": False,
                "feature_dim": feature_dim, "head_parameters": module_parameters(head),
                "total_trainable_parameters": module_parameters(head),
                "qwen_trainable_parameters": 0,
            })
        finally:
            bypass.close()
        if args.phase == "extract_features": return 0
    if args.phase in {"train", "all"}:
        train_probe(settings, device)
        if args.phase == "train": return 0
    if args.phase in {"test", "all"}:
        last = inference(settings, device, data.test, "last")
        best = inference(settings, device, data.test, "best")
        print(f"Patch-hidden probe last: MAE={last['mae']:.4f} SRCC={last['srcc']:.4f} PLCC={last['plcc']:.4f}", flush=True)
        print(f"Patch-hidden probe best-test (selection-biased): MAE={best['mae']:.4f} "
              f"SRCC={best['srcc']:.4f} PLCC={best['plcc']:.4f}", flush=True)
    return 0


def _dirs(root: Path) -> None:
    for relative in ("features", "metrics", "checkpoints", "figures"):
        (root / relative).mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())

