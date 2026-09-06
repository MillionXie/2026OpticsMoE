"""Render physical phase planes directly from a T01 student checkpoint."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


def _short_label(modality: str, key: str) -> str:
    if ".router." in key:
        part = "router"
    elif ".experts." in key:
        expert = key.split(".experts.", 1)[1].split(".", 1)[0]
        part = f"expert {expert}"
    elif ".global_phase." in key:
        part = "global"
    elif ".phase1." in key:
        part = "dense stage 1"
    elif ".phase2." in key:
        part = "dense stage 2"
    else:
        part = key
    return f"{modality} · {part}"


def render(checkpoint: Path, output_dir: Path) -> dict[str, object]:
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    planes: list[tuple[str, np.ndarray]] = []
    statistics: dict[str, object] = {}
    for group, modality in (
        ("vision_optical", "Vision"),
        ("language_optical", "Language"),
    ):
        state = saved.get(group)
        if not isinstance(state, dict):
            raise ValueError(f"Checkpoint is missing {group}")
        for key, value in state.items():
            if "phase" not in key or not key.endswith("raw_phase"):
                continue
            phase = (2.0 * math.pi * torch.sigmoid(value.float())).numpy()
            label = _short_label(modality, key)
            planes.append((label, phase))
            statistics[label] = {
                "checkpoint_key": f"{group}.{key}",
                "shape_hw": list(phase.shape),
                "mean_rad": float(phase.mean()),
                "std_rad": float(phase.std()),
                "minimum_rad": float(phase.min()),
                "maximum_rad": float(phase.max()),
            }
    if not planes:
        raise ValueError("Checkpoint contains no raw phase plane")
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 7,
            "axes.titlesize": 7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    columns = min(6, len(planes))
    rows = math.ceil(len(planes) / columns)
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(18.0 / 2.54, max(4.0, 4.0 * rows) / 2.54),
        constrained_layout=True,
        squeeze=False,
    )
    image = None
    for axis, (label, phase) in zip(axes.flat, planes):
        image = axis.imshow(
            phase,
            cmap="twilight",
            vmin=0.0,
            vmax=2.0 * math.pi,
            interpolation="nearest",
        )
        axis.set_title(label)
        axis.set_xticks([])
        axis.set_yticks([])
    for axis in axes.flat[len(planes) :]:
        axis.axis("off")
    colorbar = figure.colorbar(image, ax=axes, fraction=0.015, pad=0.01)
    colorbar.set_label("phase (rad)")
    colorbar.set_ticks((0.0, math.pi, 2.0 * math.pi), labels=("0", "π", "2π"))
    png = output_dir / "best_phase_overview.png"
    figure.savefig(png, dpi=300)
    figure.savefig(png.with_suffix(".pdf"))
    plt.close(figure)
    report = {
        "schema_version": 1,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_epoch": saved.get("epoch"),
        "weight_variant": saved.get("metadata", {}).get("weight_variant"),
        "physical_parameterization": "phase_rad = 2*pi*sigmoid(raw_phase)",
        "plane_count": len(planes),
        "planes": statistics,
    }
    (output_dir / "phase_statistics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    report = render(Path(args.checkpoint), Path(args.output_dir))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

