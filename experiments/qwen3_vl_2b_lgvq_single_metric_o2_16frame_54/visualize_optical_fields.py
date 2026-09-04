"""Render the six simulated CCD planes for representative LGVQ videos.

The plots are visualization-only.  Every panel is derived from the raw
simulated detector intensity before the learned electronic CCD readout; the
percentile/log mapping is applied only to the PNG/PDF, never to inference.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch

from .data import LGVQSingleMetricDataset, load_single_metric_cache
from .modeling import build_model
from .settings import load_settings


STAGES = (
    ("parallel_router", "Vision optical router"),
    ("parallel_expert", "Vision Top-2 experts"),
    ("parallel_global", "Vision global mask"),
    ("serial_router", "Sequence optical router"),
    ("serial_expert", "Sequence Top-2 experts"),
    ("serial_global", "Sequence global mask"),
)


def _safe_stem(value: object, *, maximum_length: int = 96) -> str:
    """Return a portable filename component for dataset-owned sample IDs."""

    text = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("._")
    return (text or "sample")[:maximum_length]


def _display_map(value: torch.Tensor) -> np.ndarray:
    array = value.detach().float().cpu().numpy()
    array = np.maximum(array, 0.0)
    positive = array[array > 0]
    scale = float(np.percentile(positive, 99.7)) if positive.size else 1.0
    return np.log1p(np.minimum(array / max(scale, 1.0e-12), 1.0)) / np.log(2.0)


def _sample_indices(dataset: LGVQSingleMetricDataset, count: int) -> list[int]:
    targets = np.asarray(
        [float(dataset[index]["target"]) for index in range(len(dataset))]
    )
    quantiles = np.linspace(0.15, 0.85, count)
    wanted = np.quantile(targets, quantiles)
    selected: list[int] = []
    for value in wanted:
        order = np.argsort(np.abs(targets - value))
        selected.append(next(int(index) for index in order if int(index) not in selected))
    return selected


def render(
    config: Path,
    checkpoint: Path,
    output_dir: Path,
    *,
    device_name: str,
    sample_count: int,
) -> dict[str, Any]:
    settings = load_settings(config)
    payload = load_single_metric_cache(settings)
    device = torch.device(
        ("cuda" if torch.cuda.is_available() else "cpu")
        if device_name == "auto"
        else device_name
    )
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = build_model(settings)
    model.load_state_dict(saved["state_dict"], strict=True)
    model.to(device).eval()

    captured: dict[str, list[torch.Tensor]] = {}

    def hook(name: str):
        def collect(_module, _arguments, output) -> None:
            captured.setdefault(name, []).append(output.detach().abs().square().cpu())

        return collect

    handles = [
        model.parallel_router.propagation.register_forward_hook(hook("parallel_router")),
        model.parallel_optics.propagation.register_forward_hook(hook("parallel_optics")),
        model.serial_router.propagation.register_forward_hook(hook("serial_router")),
        model.serial_optics.propagation.register_forward_hook(hook("serial_optics")),
    ]
    dataset = LGVQSingleMetricDataset(payload, "test")
    selected = _sample_indices(dataset, sample_count)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    figures: list[str] = []
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 7,
            "axes.titlesize": 7,
            "axes.labelsize": 7,
        }
    )
    try:
        for rank, index in enumerate(selected, 1):
            item = dataset[index]
            captured.clear()
            with torch.no_grad():
                result = model(
                    item["vision_tokens"].unsqueeze(0).to(device),
                    item["quality_tokens"].unsqueeze(0).to(device),
                    item["language_tokens"].unsqueeze(0).to(device),
                    item["language_mask"].unsqueeze(0).to(device),
                    optical_enabled=True,
                )
            fields = {
                "parallel_router": captured["parallel_router"][0][0],
                "parallel_expert": captured["parallel_optics"][0][0],
                "parallel_global": captured["parallel_optics"][1][0],
                "serial_router": captured["serial_router"][0][0],
                "serial_expert": captured["serial_optics"][0][0],
                "serial_global": captured["serial_optics"][1][0],
            }
            figure, axes = plt.subplots(2, 3, figsize=(7.0, 2.25), constrained_layout=True)
            margin = settings.geometry.active_margin
            for axis, (name, title) in zip(axes.flat, STAGES):
                field = fields[name][
                    margin : margin + settings.geometry.active_size,
                    margin : margin + settings.geometry.active_size,
                ]
                axis.imshow(_display_map(field), cmap="viridis", vmin=0.0, vmax=1.0)
                axis.set_title(title)
                axis.set_xticks([])
                axis.set_yticks([])
                rows.append(
                    {
                        "sample_id": item["sample_id"],
                        "stage": name,
                        "raw_mean_intensity": float(field.mean()),
                        "raw_p99_intensity": float(torch.quantile(field.float(), 0.99)),
                        "raw_max_intensity": float(field.max()),
                    }
                )
            target = float(item["target"])
            prediction = float(result["prediction"][0])
            figure.suptitle(
                f"{settings.frame_count}-frame Temporal | {item['sample_id']} | "
                f"target {target:.2f}, prediction {prediction:.2f}",
                fontsize=7,
            )
            stem = (
                f"sample_{rank:02d}_{_safe_stem(item['sample_id'])}_six_ccd_planes"
            )
            png = output_dir / f"{stem}.png"
            pdf = output_dir / f"{stem}.pdf"
            figure.savefig(png, dpi=300, facecolor="white")
            figure.savefig(pdf, facecolor="white")
            plt.close(figure)
            figures.extend((str(png.resolve()), str(pdf.resolve())))
    finally:
        for handle in handles:
            handle.remove()

    csv_path = output_dir / "raw_ccd_intensity_statistics.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = {
        "schema_version": 1,
        "config": str(config.resolve()),
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_epoch": int(saved.get("epoch", -1)),
        "frame_count": settings.frame_count,
        "sample_count": sample_count,
        "figures": figures,
        "raw_statistics_csv": str(csv_path.resolve()),
        "display_only_mapping": "clip at each plane's raw p99.7, then log1p; inference unchanged",
    }
    (output_dir / "visualization_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--sample-count", type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.sample_count <= 12:
        parser.error("--sample-count must be between 1 and 12")
    report = render(
        args.config,
        args.checkpoint,
        args.output_dir,
        device_name=args.device,
        sample_count=args.sample_count,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
