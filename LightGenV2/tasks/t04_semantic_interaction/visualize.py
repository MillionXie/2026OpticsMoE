"""Render all phase planes in a selected OpenMoji checkpoint."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch


def render(checkpoint: Path, output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    values = [
        (name, 2.0 * math.pi * torch.sigmoid(raw.float()))
        for name, raw in payload["model"].items()
        if "raw_phase" in name or "raw_router_phase" in name
    ]
    if not values:
        raise RuntimeError("No OpenMoji phase tensors found")
    columns = min(4, len(values))
    rows = math.ceil(len(values) / columns)
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(3.4 * columns, 3.0 * rows),
        squeeze=False,
        constrained_layout=True,
    )
    statistics = []
    image = None
    for axis, (name, phase) in zip(axes.ravel(), values):
        mean = torch.angle(torch.exp(1j * phase).mean())
        residual = torch.remainder(phase - mean + math.pi, 2.0 * math.pi) - math.pi
        image = axis.imshow(residual.numpy(), cmap="RdBu_r", vmin=-math.pi, vmax=math.pi)
        axis.set_title(name.replace("optical_branch.core.", ""), fontsize=7)
        axis.set_xlabel("x pixel")
        axis.set_ylabel("y pixel")
        statistics.append(
            {
                "name": name,
                "shape": list(phase.shape),
                "circular_residual_std_rad": float(residual.std(unbiased=False)),
                "physical_phase_min_rad": float(phase.min()),
                "physical_phase_max_rad": float(phase.max()),
            }
        )
    for axis in axes.ravel()[len(values) :]:
        axis.set_visible(False)
    if image is not None:
        figure.colorbar(image, ax=axes.ravel().tolist(), label="relative phase (rad)")
    figure.suptitle(f"OpenMoji selected phase masks (epoch {payload.get('epoch')})")
    figure.savefig(output_dir / "best_phase_overview.png", dpi=180)
    figure.savefig(output_dir / "best_phase_overview.pdf")
    plt.close(figure)
    report = {
        "checkpoint": str(checkpoint.resolve()),
        "epoch": payload.get("epoch"),
        "plane_count": len(values),
        "planes": statistics,
    }
    (output_dir / "phase_statistics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


__all__ = ["render"]
