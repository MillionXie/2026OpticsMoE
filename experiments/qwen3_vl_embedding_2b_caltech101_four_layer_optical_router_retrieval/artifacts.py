from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import torch

from experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval.optical_artifacts import (
    phase_tensors,
    tensor_stats,
)


def _router_phases(replacement: Any) -> dict[str, torch.Tensor]:
    if replacement.router_backend != "optical":
        return {}
    return {
        name: surrogate.core.optical_branch.core.router.active_phase()
        .detach()
        .cpu()
        .float()
        for name, surrogate in (
            ("vision", replacement.vision_surrogate),
            ("language", replacement.language_surrogate),
        )
    }


def _last_batch_router_diagnostics(replacement: Any) -> dict[str, Any]:
    """Snapshot the final live training batch without pretending it is epoch-wide."""

    output: dict[str, Any] = {
        "scope": "last_training_batch_live_forward_not_epoch_aggregate"
    }
    for name, surrogate in (
        ("vision", replacement.vision_surrogate),
        ("language", replacement.language_surrogate),
    ):
        routing = surrogate.core.optical_branch.core.last_routing
        if not routing:
            continue
        values: dict[str, Any] = {}
        for key in (
            "capture_fraction",
            "detector_energy_fraction",
            "probabilities",
            "load",
            "importance",
        ):
            value = routing.get(key)
            if not isinstance(value, torch.Tensor):
                continue
            value = value.detach().cpu().float()
            values[key] = {
                "shape": list(value.shape),
                "mean": float(value.mean()),
                "min": float(value.min()),
                "max": float(value.max()),
                "per_expert_mean": (
                    value.mean(dim=0).tolist() if value.ndim == 2 else None
                ),
            }
        output[name] = values
    return output


def save_phase_snapshot(
    replacement: Any,
    output_dir: Path,
    *,
    epoch: int,
    train_loss: float,
    weight_variant: str,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    feature = {
        name: phase_tensors(core)
        for name, core in replacement.optical_artifact_cores().items()
    }
    routers = _router_phases(replacement)
    payload = {
        "schema_version": 2,
        "epoch": int(epoch),
        "train_loss": float(train_loss),
        "weight_variant": str(weight_variant),
        "router_backend": replacement.router_backend,
        "vision": feature["vision"],
        "language": feature["language"],
        "router_physical_phase_rad": routers,
        "last_batch_router_diagnostics": _last_batch_router_diagnostics(
            replacement
        ),
    }
    destination = output_dir / "phase_parameters.pt"
    temporary = destination.with_suffix(".pt.tmp")
    torch.save(payload, temporary)
    temporary.replace(destination)
    report: dict[str, Any] = {
        "schema_version": 2,
        "epoch": int(epoch),
        "train_loss": float(train_loss),
        "weight_variant": str(weight_variant),
        "router_backend": replacement.router_backend,
        "phase_parameters": str(destination),
        "feature_stacks": {
            name: {key: tensor_stats(value) for key, value in values.items()}
            for name, values in feature.items()
        },
        "router_phases": {
            name: tensor_stats(value) for name, value in routers.items()
        },
        "last_batch_router_diagnostics": _last_batch_router_diagnostics(
            replacement
        ),
    }
    (output_dir / "phase_parameters.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return report


def _relative_phase(value: torch.Tensor) -> tuple[torch.Tensor, float, float]:
    value = value.detach().cpu().float()
    valid = value[value != 0.0] if torch.any(value != 0.0) else value.reshape(-1)
    mean = torch.angle(torch.exp(1j * valid).mean())
    residual = torch.remainder(value - mean + math.pi, 2.0 * math.pi) - math.pi
    residual = residual.masked_fill(value == 0.0, torch.nan)
    measured = torch.remainder(valid - mean + math.pi, 2.0 * math.pi) - math.pi
    return residual, float(torch.remainder(mean, 2.0 * math.pi)), float(
        measured.std(unbiased=False)
    )


def save_phase_preview(replacement: Any, path: Path, *, title: str) -> None:
    import matplotlib.pyplot as plt

    feature = {
        name.title(): phase_tensors(core)
        for name, core in replacement.optical_artifact_cores().items()
    }
    routers = {name.title(): value for name, value in _router_phases(replacement).items()}
    columns = 3 if routers else 2
    figure, axes = plt.subplots(
        2, columns, figsize=(5.0 * columns, 9.0), constrained_layout=True
    )
    if columns == 2:
        axes = axes.reshape(2, 2)
    panels: list[tuple[int, int, str, torch.Tensor]] = []
    for row, stack in enumerate(("Vision", "Language")):
        panels.extend(
            [
                (
                    row,
                    0,
                    f"{stack} expert mosaic",
                    feature[stack]["physical_expert_mosaic_rad"],
                ),
                (
                    row,
                    1,
                    f"{stack} global phase",
                    feature[stack]["physical_global_phase_rad"],
                ),
            ]
        )
        if routers:
            panels.append((row, 2, f"{stack} router phase", routers[stack]))
    residuals = [_relative_phase(value) for _, _, _, value in panels]
    limit = min(math.pi, max(0.05, 3.0 * max(item[2] for item in residuals)))
    image = None
    for (row, column, label, _), (residual, mean, std) in zip(panels, residuals):
        image = axes[row, column].imshow(
            residual.numpy(), cmap="RdBu_r", vmin=-limit, vmax=limit
        )
        axes[row, column].set_title(
            f"{label}\nmean={mean:.4f} rad, relative std={std:.4f} rad"
        )
        axes[row, column].set_xlabel("x pixel")
        axes[row, column].set_ylabel("y pixel")
    if image is not None:
        figure.colorbar(
            image,
            ax=axes.ravel().tolist(),
            label="phase after circular-mean removal (rad)",
        )
    figure.suptitle(f"{title}\nshared range +/-{limit:.4f} rad")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)


__all__ = ["save_phase_preview", "save_phase_snapshot"]
