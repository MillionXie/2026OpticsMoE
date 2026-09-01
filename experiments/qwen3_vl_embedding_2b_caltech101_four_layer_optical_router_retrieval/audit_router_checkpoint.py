from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval.optics.geometry import (
    MoEGeometry,
)

from .modeling import architecture_label
from .router import OpticalDetectorTopKRouter
from .settings import load_settings


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _wrapped_difference(current: torch.Tensor, initial: torch.Tensor) -> torch.Tensor:
    return torch.remainder(current - initial + torch.pi, 2.0 * torch.pi) - torch.pi


def audit(config: Path, checkpoint: Path, output: Path | None) -> dict[str, object]:
    settings = load_settings(config)
    if settings.router_backend != "optical":
        raise ValueError("Router-phase movement audit requires an optical config")
    checkpoint = checkpoint.expanduser().resolve()
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    actual_architecture = str(payload.get("metadata", {}).get("optical_architecture", ""))
    expected_architecture = architecture_label(settings)
    if actual_architecture != expected_architecture:
        raise RuntimeError("Checkpoint/config architecture mismatch")
    geometry = MoEGeometry(
        settings.canvas_size,
        settings.active_size,
        settings.expert_size,
        settings.expert_pitch,
        settings.num_experts,
        settings.expert_grid_rows,
        settings.expert_grid_cols,
    )
    initial_router = OpticalDetectorTopKRouter(geometry, settings)
    initial = initial_router.phase().detach().float()
    stacks: dict[str, object] = {}
    key = "core.optical_branch.core.router.raw_router_phase"
    for name in ("vision", "language"):
        raw = payload[f"{name}_optical"][key].float()
        current = 2.0 * torch.pi * torch.sigmoid(raw)
        difference = _wrapped_difference(current, initial)
        stacks[name] = {
            "phase_size": list(current.shape),
            "phase_mean_rad": float(current.mean()),
            "phase_std_rad": float(current.std(unbiased=False)),
            "delta_from_deterministic_initial_rms_rad": float(
                difference.square().mean().sqrt()
            ),
            "delta_from_deterministic_initial_max_abs_rad": float(
                difference.abs().max()
            ),
            "moved_fraction_abs_delta_gt_0p01_rad": float(
                (difference.abs() > 0.01).float().mean()
            ),
            "finite": bool(torch.isfinite(current).all()),
        }
    report = {
        "schema_version": 1,
        "config": str(config.expanduser().resolve()),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "checkpoint_architecture": actual_architecture,
        "checkpoint_epoch": int(payload["epoch"]),
        "router_learning_rate": settings.router_learning_rate,
        "initialization": "deterministic_four_spot_phase_hologram",
        "stacks": stacks,
    }
    if output is not None:
        output = output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit optical-router phase movement")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    print(json.dumps(audit(args.config, args.checkpoint, args.output), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
