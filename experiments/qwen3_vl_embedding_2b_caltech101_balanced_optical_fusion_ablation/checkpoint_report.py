from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from .modeling import balanced_checkpoint_architecture, _range_gate
from .settings import load_settings


def build_report(config: str | Path, checkpoint: str | Path) -> dict[str, Any]:
    settings = load_settings(config)
    checkpoint_path = Path(checkpoint).expanduser().resolve()
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    metadata = dict(payload.get("metadata", {}))
    expected_architecture = balanced_checkpoint_architecture(
        settings.fusion_mode,
        settings.fusion_alpha_min,
        settings.fusion_alpha_max,
        settings.fusion_rms_epsilon,
    )
    saved_architecture = metadata.get("optical_architecture")
    if str(saved_architecture) != expected_architecture:
        raise RuntimeError(
            "Fusion checkpoint/config architecture mismatch: "
            f"saved={saved_architecture!r}, expected={expected_architecture!r}. "
            "Gate logits can only be decoded with the alpha range that trained them."
        )
    gates: dict[str, dict[str, float]] = {}
    for modality, state_name in (
        ("vision", "vision_optical"),
        ("language", "language_optical"),
    ):
        state = payload[state_name]
        for block in ("block1", "block2"):
            raw = state[f"core.{block}_optical_fusion_logit"].float()
            alpha = _range_gate(
                raw, settings.fusion_alpha_min, settings.fusion_alpha_max
            )
            electronic_only = settings.fusion_mode == "electronic_only"
            gates[f"{modality}_{block}"] = {
                "raw_logit": float(raw),
                "alpha": float(alpha),
                "nominal_electronic_coefficient": float(1.0 - alpha),
                "nominal_optical_coefficient": float(alpha),
                "electronic_coefficient": (
                    1.0 if electronic_only else float(1.0 - alpha)
                ),
                "optical_coefficient": 0.0 if electronic_only else float(alpha),
            }
    return {
        "checkpoint": str(checkpoint_path),
        "epoch": int(payload["epoch"]),
        "train_loss": float(payload["train_loss"]),
        "weight_variant": metadata.get("weight_variant"),
        "checkpoint_architecture": metadata.get("optical_architecture"),
        "selection_criterion": metadata.get("selection_criterion"),
        "test_metrics_used_for_selection": metadata.get(
            "test_metrics_used_for_selection"
        ),
        "fusion_mode": settings.fusion_mode,
        "alpha_range": [settings.fusion_alpha_min, settings.fusion_alpha_max],
        "rms_epsilon": settings.fusion_rms_epsilon,
        "gates": gates,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Report four learned fusion alphas")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = build_report(args.config, args.checkpoint)
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["build_report", "main"]
